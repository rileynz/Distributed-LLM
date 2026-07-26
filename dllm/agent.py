from __future__ import annotations

import signal
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from . import backend, config, tunnel
from .discovery import discover
from .hardware import detect, refresh_dynamic
from .http_utils import request_json


class Agent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.stop_event = threading.Event()
        self.hardware = detect(run_benchmark=True).to_dict()
        self.rpc = backend.ManagedProcess("rpc-agent", config.logs_dir() / "rpc-agent.log")
        self.tunnel: tunnel.TunnelClient | None = None
        self.last_error = ""

    @property
    def tools(self) -> backend.Tools:
        directory = Path(self.cfg["backend_dir"]) if self.cfg.get("backend_dir") else None
        return backend.discover(directory)

    def install_backend(self) -> None:
        if self.tools.agent_ready:
            return
        result = backend.install(str(self.cfg.get("backend_variant", "auto")))
        self.cfg["backend_dir"] = result["directory"]
        config.save(self.cfg)

    def enroll(self) -> None:
        url = self.cfg["coordinator_url"].rstrip("/")
        payload = {
            "join_code": self.cfg["join_code"],
            "node_id": self.cfg.get("node_id", ""),
            "node_token": self.cfg.get("node_token", ""),
            "name": self.cfg["name"],
            "hardware": self.hardware,
            "rpc_host": (
                ""
                if self.cfg.get("worker_connection_mode") == "outbound"
                else self.hardware["lan_ip"]
            ),
            "rpc_port": int(self.cfg["rpc_port"]),
            "worker_connection_mode": self.cfg.get("worker_connection_mode", "direct"),
        }
        status, response = request_json(
            f"{url}/api/enroll", method="POST", payload=payload, timeout=10
        )
        if status == 200:
            self._save_identity(response)
            return
        if status != 202:
            raise RuntimeError(response.get("error", f"Join failed with HTTP {status}"))
        request_id = str(response["request_id"])
        self.cfg["node_id"] = str(response["node_id"])
        config.save(self.cfg)
        print("Waiting for approval in the coordinator dashboard...")
        while not self.stop_event.wait(2):
            status, response = request_json(
                f"{url}/api/enroll/status?request_id={request_id}", timeout=8
            )
            if status == 200:
                self._save_identity(response)
                return
            if status not in {202, 404}:
                raise RuntimeError(response.get("error", "Approval check failed"))
            if status == 404:
                raise RuntimeError("The coordinator rejected or forgot this join request")
        raise RuntimeError("Stopped while waiting for approval")

    def _save_identity(self, response: dict) -> None:
        self.cfg["node_id"] = str(response["node_id"])
        self.cfg["node_token"] = str(response["node_token"])
        self.cfg["cluster_id"] = str(
            response.get("cluster_id", self.cfg.get("cluster_id", ""))
        )
        self.cfg["cluster_name"] = str(
            response.get("cluster_name", self.cfg.get("cluster_name", ""))
        )
        if response.get("tunnel_port"):
            self.cfg["tunnel_port"] = int(response["tunnel_port"])
        if response.get("tunnel_fingerprint"):
            self.cfg["tunnel_fingerprint"] = str(response["tunnel_fingerprint"])
        config.save(self.cfg)

    def recover_coordinator(self) -> bool:
        matches = discover(timeout=2.0)
        cluster_id = str(self.cfg.get("cluster_id", ""))
        cluster_name = str(self.cfg.get("cluster_name", ""))
        preferred = [
            item for item in matches
            if (cluster_id and item.cluster_id == cluster_id)
            or (not cluster_id and cluster_name and item.name == cluster_name)
        ]
        if not preferred:
            return False
        new_url = preferred[0].url.rstrip("/")
        if new_url == self.cfg["coordinator_url"].rstrip("/"):
            return False
        self.cfg["coordinator_url"] = new_url
        config.save(self.cfg)
        print(f"Coordinator rediscovered at {new_url}")
        return True

    def start_rpc(self) -> None:
        if self.rpc.running:
            return
        tools = self.tools
        if not tools.agent_ready:
            raise RuntimeError("llama.cpp RPC backend is not installed")
        free_mb = int(max(0, (float(self.hardware["ram_free_gb"]) - 1.0) * 1024))
        if free_mb < 1024:
            raise RuntimeError("Less than 1 GB safe memory remains for model tensors")
        bind_host = (
            "127.0.0.1"
            if self.cfg.get("worker_connection_mode") == "outbound"
            else "0.0.0.0"
        )
        command = backend.rpc_command(tools, bind_host, int(self.cfg["rpc_port"]), free_mb)
        self.rpc.start(command)

    def start_tunnel(self) -> None:
        if self.cfg.get("worker_connection_mode") != "outbound":
            return
        if self.tunnel and self.tunnel.thread and self.tunnel.thread.is_alive():
            return
        parsed = urlparse(str(self.cfg["coordinator_url"]))
        if not parsed.hostname:
            raise RuntimeError("Coordinator URL does not contain a hostname")
        self.tunnel = tunnel.TunnelClient(
            parsed.hostname,
            int(self.cfg["tunnel_port"]),
            str(self.cfg["node_id"]),
            str(self.cfg["node_token"]),
            str(self.cfg.get("tunnel_fingerprint", "")),
            int(self.cfg["rpc_port"]),
        )
        self.tunnel.start()

    def stop_tunnel(self) -> None:
        if self.tunnel:
            self.tunnel.close()
            self.tunnel = None

    def heartbeat(self) -> dict:
        url = self.cfg["coordinator_url"].rstrip("/")
        self.hardware = refresh_dynamic(self.hardware)
        sent = time.time()
        status, response = request_json(
            f"{url}/api/heartbeat",
            method="POST",
            payload={
                "node_id": self.cfg["node_id"],
                "hardware": self.hardware,
                "rpc_running": self.rpc.running,
                "rpc_host": (
                    ""
                    if self.cfg.get("worker_connection_mode") == "outbound"
                    else self.hardware["lan_ip"]
                ),
                "rpc_port": int(self.cfg["rpc_port"]),
                "worker_connection_mode": self.cfg.get(
                    "worker_connection_mode", "direct"
                ),
                "tunnel_running": bool(self.tunnel and self.tunnel.running),
                "backend_ready": self.tools.agent_ready,
                "last_error": self.last_error,
                "sent_at": sent,
            },
            headers={"Authorization": f"Bearer {self.cfg['node_token']}"},
            timeout=10,
        )
        if status == 401:
            self.cfg["node_token"] = ""
            config.save(self.cfg)
            raise PermissionError("Coordinator no longer accepts this node token")
        if status != 200:
            raise RuntimeError(response.get("error", f"Heartbeat failed with HTTP {status}"))
        return response

    def run(self) -> None:
        print(f"Distributed LLM agent: {self.cfg['name']}")
        print(f"Coordinator: {self.cfg['coordinator_url']}")
        if not self.tools.agent_ready:
            print("Installing the compatible llama.cpp backend...")
            self.install_backend()
        if not self.cfg.get("node_token"):
            self.enroll()
        self.start_rpc()
        self.start_tunnel()
        failures = 0
        while not self.stop_event.is_set():
            try:
                response = self.heartbeat()
                failures = 0
                desired = response.get("desired", {})
                if desired.get("rpc_running") and not self.rpc.running:
                    self.start_rpc()
                wait = max(2, min(int(response.get("heartbeat_seconds", 5)), 30))
            except PermissionError:
                self.stop_tunnel()
                self.rpc.stop()
                self.enroll()
                self.start_rpc()
                self.start_tunnel()
                wait = 2
            except Exception as exc:
                failures += 1
                self.last_error = str(exc)
                if failures >= 2:
                    self.recover_coordinator()
                wait = min(30, 2 ** min(failures, 5))
                print(f"Coordinator connection failed; retrying in {wait}s: {exc}")
            self.stop_event.wait(wait)
        self.stop_tunnel()
        self.rpc.stop()

    def stop(self, _signum=None, _frame=None) -> None:
        self.stop_event.set()


def run(cfg: dict) -> None:
    agent = Agent(cfg)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, agent.stop)
        signal.signal(signal.SIGTERM, agent.stop)
    agent.run()
