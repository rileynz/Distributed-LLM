from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import signal
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__, backend, config, tunnel
from .allocator import fit_label, plan
from .discovery import DiscoveryResponder
from .hardware import detect, lan_ip, refresh_dynamic
from .models import CATALOGUE, get_model
from .security import RateLimiter, new_join_code, new_token


HEARTBEAT_STALE_SECONDS = 18
HF_SPEC = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?$")


def _static_root() -> Path:
    return Path(__file__).resolve().parent.parent / "static"


def _read_json(handler: BaseHTTPRequestHandler, maximum: int = 1_000_000) -> dict:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        raise ValueError("invalid content length")
    if length < 0 or length > maximum:
        raise ValueError("request body is too large")
    if not length:
        return {}
    try:
        payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _probe(url: str, timeout: float = 0.6) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


class CoordinatorState:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.install_lock = threading.Lock()
        self.rate_limiter = RateLimiter()
        self.hardware = detect(run_benchmark=True).to_dict()
        self.local_rpc = backend.ManagedProcess(
            "local-rpc", config.logs_dir() / "local-rpc.log"
        )
        self.inference = backend.ManagedProcess(
            "inference", config.logs_dir() / "inference.log"
        )
        self.data = self._load_state()
        self.tunnel_broker: tunnel.TunnelBroker | None = None
        self.install_status = {"state": "idle", "message": ""}
        self._start_local_rpc_if_possible()

    def start_tunnel_broker(self) -> None:
        if self.tunnel_broker:
            return
        broker = tunnel.TunnelBroker(
            "0.0.0.0",
            int(self.cfg["tunnel_port"]),
            config.data_dir() / "tunnel-cert.pem",
            config.data_dir() / "tunnel-key.pem",
            self._authenticate_tunnel,
            advertised_hosts=(lan_ip(),),
        )
        broker.start()
        self.cfg["tunnel_port"] = broker.port
        config.save(self.cfg)
        self.tunnel_broker = broker

    def _authenticate_tunnel(self, node_id: str, token: str) -> bool:
        with self.lock:
            node = self.data.get("nodes", {}).get(node_id)
            return bool(
                node
                and token
                and hmac.compare_digest(token, str(node.get("token", "")))
            )

    def _tunnel_details(self) -> dict:
        broker = self.tunnel_broker
        if not broker:
            return {}
        return {
            "tunnel_port": broker.port,
            "tunnel_fingerprint": broker.fingerprint,
        }

    def _load_state(self) -> dict:
        try:
            loaded = json.loads(config.state_path().read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
            loaded = {}
        loaded.setdefault("cluster_id", secrets.token_hex(12))
        loaded.setdefault("join_code", self.cfg.get("join_code") or new_join_code())
        loaded.setdefault("nodes", {})
        loaded.setdefault("pending", {})
        loaded.setdefault("active_model", "")
        loaded.setdefault("last_plan", {})
        self.cfg["join_code"] = loaded["join_code"]
        config.save(self.cfg)
        self._persist(loaded)
        return loaded

    def _persist(self, payload: dict | None = None) -> None:
        config.save_json_atomic(config.state_path(), payload or self.data)

    def _start_local_rpc_if_possible(self) -> None:
        tools = backend.discover(
            Path(self.cfg["backend_dir"]) if self.cfg.get("backend_dir") else None
        )
        if not tools.agent_ready or self.local_rpc.running:
            return
        total = int(max(0, (self.hardware.get("ram_free_gb", 0) - 2.0) * 1024))
        if total < 1024:
            return
        command = backend.rpc_command(
            tools, "127.0.0.1", int(self.cfg["rpc_port"]), total
        )
        self.local_rpc.start(command)

    def public_status(self) -> dict:
        with self.lock:
            self.hardware = refresh_dynamic(self.hardware)
            now = time.time()
            nodes = []
            for node in self.data["nodes"].values():
                item = dict(node)
                item.pop("token", None)
                item["online"] = now - float(item.get("last_seen", 0)) <= HEARTBEAT_STALE_SECONDS
                nodes.append(item)
            pending = []
            for item in self.data["pending"].values():
                public = dict(item)
                pending.append(public)
            models = []
            plan_nodes = self.all_compute_nodes()
            for model in CATALOGUE:
                result = plan(
                    plan_nodes,
                    float(model["size_gb"]),
                    context_size=int(self.cfg["context_size"]),
                    auto_exclude_slow=bool(self.cfg["auto_exclude_slow_nodes"]),
                )
                models.append({**model, "fit": fit_label(result), "plan": result})
            model_api = f"http://{lan_ip()}:{self.cfg['inference_port']}"
            inference_healthy = _probe(
                f"http://127.0.0.1:{self.cfg['inference_port']}/health"
            )
            if (
                self.data.get("active_model")
                and not self.inference.running
                and not inference_healthy
            ):
                self.data["active_model"] = ""
                self._persist()
            return {
                "app_version": __version__,
                "cluster_name": self.cfg["cluster_name"],
                "cluster_id": self.data["cluster_id"],
                "join_code": "••••••",
                "trusted_lan_only": True,
                "nodes": sorted(nodes, key=lambda item: item.get("name", "")),
                "pending": pending,
                "local_hardware": self.hardware,
                "backend": {
                    **backend.discover(
                        Path(self.cfg["backend_dir"]) if self.cfg.get("backend_dir") else None
                    ).__dict__,
                    "install": self.install_status,
                    "local_rpc": self.local_rpc.status(),
                },
                "portable_tunnel": {
                    "running": bool(self.tunnel_broker),
                    "port": (
                        self.tunnel_broker.port if self.tunnel_broker else self.cfg["tunnel_port"]
                    ),
                    "connected_workers": (
                        len(self.tunnel_broker.sessions) if self.tunnel_broker else 0
                    ),
                    "last_error": (
                        self.tunnel_broker.last_error if self.tunnel_broker else ""
                    ),
                },
                "inference": {
                    **self.inference.status(),
                    "healthy": inference_healthy,
                    "url": model_api,
                    "active_model": self.data.get("active_model", ""),
                },
                "models": models,
                "last_plan": self.data.get("last_plan", {}),
                "settings": {
                    "context_size": self.cfg["context_size"],
                    "parallel_requests": self.cfg["parallel_requests"],
                    "kv_cache": self.cfg["kv_cache"],
                    "auto_exclude_slow_nodes": self.cfg["auto_exclude_slow_nodes"],
                },
            }

    def all_compute_nodes(self) -> list[dict]:
        now = time.time()
        local = {
            "node_id": "local",
            "name": f"{self.hardware['hostname']} (coordinator)",
            "hardware": self.hardware,
            "local": True,
            "approved": True,
            "rpc_ready": self.local_rpc.running,
            "rtt_ms": 0.1,
            "rpc_host": "127.0.0.1",
            "rpc_port": int(self.cfg["rpc_port"]),
        }
        remote = []
        for node in self.data.get("nodes", {}).values():
            if now - float(node.get("last_seen", 0)) > HEARTBEAT_STALE_SECONDS:
                continue
            item = {
                **node,
                "approved": True,
                "local": False,
                "rpc_ready": bool(node.get("rpc_running")),
            }
            if node.get("worker_connection_mode") == "outbound":
                endpoint = (
                    self.tunnel_broker.endpoint(str(node["node_id"]))
                    if self.tunnel_broker
                    else None
                )
                item["rpc_ready"] = endpoint is not None
                item["tunnel_running"] = endpoint is not None
                if endpoint:
                    item["rpc_host"], item["rpc_port"] = endpoint
            remote.append(item)
        return [local, *remote]

    def check_admin(self, supplied: str) -> bool:
        return hmac.compare_digest(str(supplied), str(self.data["join_code"]))

    def enroll(self, payload: dict, address: str) -> tuple[int, dict]:
        if not self.rate_limiter.allowed(address):
            return 429, {"error": "Too many join attempts; wait one minute"}
        if not hmac.compare_digest(
            str(payload.get("join_code", "")), str(self.data["join_code"])
        ):
            return 403, {"error": "Join code was not accepted"}
        hardware = payload.get("hardware")
        if not isinstance(hardware, dict):
            return 400, {"error": "hardware report is required"}
        node_id = str(payload.get("node_id") or secrets.token_hex(12))
        supplied_token = str(payload.get("node_token") or "")
        with self.lock:
            existing = self.data["nodes"].get(node_id)
            if existing and supplied_token and hmac.compare_digest(
                supplied_token, str(existing.get("token", ""))
            ):
                existing["last_seen"] = time.time()
                existing["hardware"] = hardware
                self._persist()
                return 200, {
                    "status": "approved",
                    "node_id": node_id,
                    "node_token": supplied_token,
                    "cluster_id": self.data["cluster_id"],
                    "cluster_name": self.cfg["cluster_name"],
                    **self._tunnel_details(),
                }
            for old_request_id, old_request in list(self.data["pending"].items()):
                if old_request.get("node_id") == node_id:
                    self.data["pending"].pop(old_request_id, None)
            request_id = secrets.token_urlsafe(24)
            self.data["pending"][request_id] = {
                "request_id": request_id,
                "node_id": node_id,
                "name": str(payload.get("name") or hardware.get("hostname") or node_id),
                "hardware": hardware,
                "rpc_host": str(payload.get("rpc_host") or address),
                "rpc_port": int(payload.get("rpc_port") or 50052),
                "worker_connection_mode": str(
                    payload.get("worker_connection_mode") or "direct"
                ),
                "requested_at": time.time(),
                "address": address,
            }
            self._persist()
        return 202, {
            "status": "pending",
            "request_id": request_id,
            "node_id": node_id,
            "cluster_id": self.data["cluster_id"],
            "cluster_name": self.cfg["cluster_name"],
            **self._tunnel_details(),
        }

    def enrollment_status(self, request_id: str) -> tuple[int, dict]:
        with self.lock:
            pending = self.data["pending"].get(request_id)
            if pending:
                return 202, {"status": "pending"}
            for node in self.data["nodes"].values():
                if node.get("request_id") == request_id:
                    return 200, {
                        "status": "approved",
                        "node_id": node["node_id"],
                        "node_token": node["token"],
                        "cluster_id": self.data["cluster_id"],
                        "cluster_name": self.cfg["cluster_name"],
                        **self._tunnel_details(),
                    }
        return 404, {"error": "Join request was not found"}

    def approve(self, request_id: str) -> tuple[int, dict]:
        with self.lock:
            pending = self.data["pending"].pop(request_id, None)
            if not pending:
                return 404, {"error": "Pending node was not found"}
            token = new_token()
            node = {
                **pending,
                "token": token,
                "request_id": request_id,
                "approved_at": time.time(),
                "last_seen": 0,
                "rpc_running": False,
                "rtt_ms": 0,
            }
            self.data["nodes"][node["node_id"]] = node
            self._persist()
        return 200, {"status": "approved", "name": node["name"]}

    def reject(self, request_id: str) -> tuple[int, dict]:
        with self.lock:
            removed = self.data["pending"].pop(request_id, None)
            self._persist()
        return (200, {"status": "rejected"}) if removed else (404, {"error": "Not found"})

    def remove_node(self, node_id: str) -> tuple[int, dict]:
        with self.lock:
            removed = self.data["nodes"].pop(node_id, None)
            self._persist()
        return (200, {"status": "removed"}) if removed else (404, {"error": "Not found"})

    def heartbeat(self, payload: dict, token: str) -> tuple[int, dict]:
        node_id = str(payload.get("node_id", ""))
        with self.lock:
            node = self.data["nodes"].get(node_id)
            if not node or not token or not hmac.compare_digest(
                token, str(node.get("token", ""))
            ):
                return 401, {"error": "Node authentication failed"}
            sent = float(payload.get("sent_at", time.time()))
            connection_mode = str(
                payload.get(
                    "worker_connection_mode",
                    node.get("worker_connection_mode", "direct"),
                )
            )
            tunnel_endpoint = (
                self.tunnel_broker.endpoint(node_id)
                if connection_mode == "outbound" and self.tunnel_broker
                else None
            )
            node.update({
                "last_seen": time.time(),
                "hardware": payload.get("hardware", node.get("hardware", {})),
                "rpc_running": (
                    bool(payload.get("rpc_running")) and tunnel_endpoint is not None
                    if connection_mode == "outbound"
                    else bool(payload.get("rpc_running"))
                ),
                "rpc_host": str(payload.get("rpc_host") or node.get("rpc_host")),
                "rpc_port": int(payload.get("rpc_port") or node.get("rpc_port")),
                "worker_connection_mode": connection_mode,
                "tunnel_running": tunnel_endpoint is not None,
                "backend_ready": bool(payload.get("backend_ready")),
                "last_error": str(payload.get("last_error", ""))[:500],
                "rtt_ms": round(max(0.0, (time.time() - sent) * 1000), 2),
            })
            self._persist()
        return 200, {
            "status": "ok",
            "desired": {"rpc_running": True},
            "heartbeat_seconds": 5,
            **self._tunnel_details(),
        }

    def start_backend_install(self, variant: str) -> tuple[int, dict]:
        if self.install_lock.locked():
            return 409, {"error": "Backend installation is already running"}

        def worker() -> None:
            with self.install_lock:
                self.install_status = {"state": "running", "message": "Downloading llama.cpp"}
                try:
                    result = backend.install(variant)
                    self.cfg["backend_dir"] = result["directory"]
                    self.cfg["backend_variant"] = variant
                    config.save(self.cfg)
                    self.install_status = {
                        "state": "ready",
                        "message": f"Installed {result['tag']} ({result['asset']})",
                    }
                    self._start_local_rpc_if_possible()
                except Exception as exc:
                    self.install_status = {"state": "error", "message": str(exc)}

        threading.Thread(target=worker, daemon=True).start()
        return 202, {"status": "started"}

    def start_model(self, model_id: str, custom: dict | None = None) -> tuple[int, dict]:
        model = get_model(model_id)
        if not model:
            custom = custom or {}
            source = str(custom.get("source", "")).strip()
            try:
                size_gb = float(custom.get("size_gb", 0))
            except (TypeError, ValueError):
                size_gb = 0
            if source.lower().endswith(".gguf"):
                model_path = Path(source).expanduser()
                if not model_path.is_file():
                    return 400, {"error": "Custom GGUF file was not found on the coordinator"}
                source = str(model_path.resolve())
                size_gb = model_path.stat().st_size / (1024 ** 3)
            elif not HF_SPEC.fullmatch(source):
                return 400, {
                    "error": "Use a coordinator .gguf path or Hugging Face repo:quant"
                }
            if not 0.1 <= size_gb <= 1000:
                return 400, {"error": "Custom model size must be between 0.1 and 1000 GB"}
            model = {
                "id": "custom",
                "name": str(custom.get("name") or source)[:120],
                "hf": source,
                "size_gb": round(size_gb, 3),
                "description": "Custom GGUF model",
            }
        self._start_local_rpc_if_possible()
        nodes = self.all_compute_nodes()
        result = plan(
            nodes,
            float(model["size_gb"]),
            context_size=int(self.cfg["context_size"]),
            auto_exclude_slow=bool(self.cfg["auto_exclude_slow_nodes"]),
        )
        if not result["fits"]:
            return 409, {
                "error": "The model does not fit safely on the ready devices",
                "plan": result,
            }
        by_id = {str(node["node_id"]): node for node in nodes}
        endpoints: list[str] = []
        weights: list[float] = []
        for placement in result["placements"]:
            if not placement["included"]:
                continue
            node = by_id[placement["node_id"]]
            endpoints.append(f"{node['rpc_host']}:{int(node['rpc_port'])}")
            weights.append(float(placement["weight"]))
        tools = backend.discover(
            Path(self.cfg["backend_dir"]) if self.cfg.get("backend_dir") else None
        )
        if not tools.coordinator_ready:
            return 409, {"error": "Install the llama.cpp backend first"}
        command = backend.server_command(
            tools,
            model["hf"],
            endpoints,
            weights,
            port=int(self.cfg["inference_port"]),
            context_size=int(self.cfg["context_size"]),
            parallel=int(self.cfg["parallel_requests"]),
            kv_cache=str(self.cfg["kv_cache"]),
        )
        try:
            self.inference.start(command)
        except OSError as exc:
            return 500, {"error": f"Could not start llama-server: {exc}"}
        with self.lock:
            self.data["active_model"] = model["id"]
            self.data["last_plan"] = result
            self._persist()
        return 202, {
            "status": "starting",
            "model": model,
            "plan": result,
            "chat_url": f"http://{lan_ip()}:{self.cfg['inference_port']}",
        }

    def stop_model(self) -> tuple[int, dict]:
        self.inference.stop()
        with self.lock:
            self.data["active_model"] = ""
            self._persist()
        return 200, {"status": "stopped"}

    def update_settings(self, payload: dict) -> tuple[int, dict]:
        updated = dict(self.cfg)
        for key in (
            "context_size", "parallel_requests", "kv_cache",
            "auto_exclude_slow_nodes",
        ):
            if key in payload:
                updated[key] = payload[key]
        errors = config.validate(updated)
        if errors:
            return 400, {"error": "; ".join(errors)}
        self.cfg = updated
        config.save(self.cfg)
        return 200, {"status": "saved"}

    def close(self) -> None:
        self.inference.stop()
        self.local_rpc.stop()
        if self.tunnel_broker:
            self.tunnel_broker.close()
            self.tunnel_broker = None


def make_handler(state: CoordinatorState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DistributedLLM/0.3"

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, status: int, payload: dict | bytes, content_type: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _admin(self) -> bool:
            supplied = self.headers.get("X-DLLM-Admin", "")
            if state.check_admin(supplied):
                return True
            self._send(403, {"error": "Enter the cluster join code to manage it"})
            return False

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                try:
                    body = (_static_root() / "index.html").read_bytes()
                    self._send(200, body, "text/html; charset=utf-8")
                except OSError:
                    self._send(500, {"error": "Dashboard file is missing"})
            elif parsed.path == "/api/status":
                self._send(200, state.public_status())
            elif parsed.path == "/api/enroll/status":
                request_id = parse_qs(parsed.query).get("request_id", [""])[0]
                code, payload = state.enrollment_status(request_id)
                self._send(code, payload)
            elif parsed.path == "/api/local/join-code":
                if self.client_address[0] in {"127.0.0.1", "::1"}:
                    self._send(200, {"join_code": state.data["join_code"]})
                else:
                    self._send(403, {"error": "Join code is only revealed on the coordinator"})
            elif parsed.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "Not found"})

        def do_POST(self) -> None:
            try:
                payload = _read_json(self)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            path = urlparse(self.path).path
            if path == "/api/enroll":
                code, result = state.enroll(payload, self.client_address[0])
            elif path == "/api/heartbeat":
                token = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                code, result = state.heartbeat(payload, token)
            else:
                if not self._admin():
                    return
                routes = {
                    "/api/pending/approve": lambda: state.approve(str(payload.get("request_id", ""))),
                    "/api/pending/reject": lambda: state.reject(str(payload.get("request_id", ""))),
                    "/api/nodes/remove": lambda: state.remove_node(str(payload.get("node_id", ""))),
                    "/api/backend/install": lambda: state.start_backend_install(str(payload.get("variant", "auto"))),
                    "/api/model/start": lambda: state.start_model(
                        str(payload.get("model_id", "")), payload.get("custom")
                    ),
                    "/api/model/stop": state.stop_model,
                    "/api/settings": lambda: state.update_settings(payload),
                }
                action = routes.get(path)
                if not action:
                    self._send(404, {"error": "Not found"})
                    return
                code, result = action()
            self._send(code, result)

    return Handler


def run(cfg: dict) -> None:
    state = CoordinatorState(cfg)
    state.start_tunnel_broker()
    url = f"http://{lan_ip()}:{cfg['control_port']}"
    responder = DiscoveryResponder(cfg["cluster_name"], url, state.data["cluster_id"])
    responder.start()
    server = ThreadingHTTPServer(("0.0.0.0", int(cfg["control_port"])), make_handler(state))
    server.daemon_threads = True
    stopping = threading.Event()

    def shutdown(_signum=None, _frame=None) -> None:
        if not stopping.is_set():
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
    print(f"Distributed LLM coordinator: {url}")
    print(f"Join code: {state.data['join_code']}")
    print(f"Portable worker tunnel: {lan_ip()}:{state.tunnel_broker.port}")
    print("Portable workers use an encrypted outbound tunnel; direct RPC remains LAN-only.")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        responder.stop()
        state.close()
        server.server_close()
