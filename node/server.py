"""
Node server.
- Serves inference requests over TCP (existing protocol).
- Runs a lightweight HTTP management server on port+10000 for:
    GET  /health          — liveness check
    GET  /benchmark       — run benchmark, return score
    POST /download        — start downloading a model in background
    GET  /download_status — check download progress
- Announces hardware + benchmark score to registry.
- Hot-reloads model when active_model.json or model.pt changes.
- Passes `extra` state (RoPE tensors for Llama) alongside hidden state.

Usage:
    python node/server.py --port 9001 --start 0 --end 6
"""

import argparse, json, os, socket, sys, threading, time, traceback, warnings
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests as http
import torch

from config import MODEL_CONFIG, CHECKPOINT_PATH
from models.active import read_active, ACTIVE_MODEL_PATH
from models.benchmark import benchmark_node_local
from shared.model_def import CharTokenizer, TinyGPT
from shared.protocol import send_msg, recv_msg


# ── Model loading ─────────────────────────────────────────────────────────────

def load_tinygpt():
    tok = CharTokenizer()
    MODEL_CONFIG.vocab_size = tok.vocab_size
    model = TinyGPT(MODEL_CONFIG)
    try:
        sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model

def load_hf(cfg, start, end, total):
    from models.hf_wrapper import HFNodeModel
    arch = cfg.get("arch", "auto")
    return HFNodeModel(cfg["hf_id"], cfg["cache_dir"], start, end, total, arch=arch)

def load_model(start, end, total):
    cfg = read_active()
    if cfg and cfg.get("type") == "hf":
        return load_hf(cfg, start, end, total), "hf"
    return load_tinygpt(), "tinygpt"

def total_layers_for_active():
    cfg = read_active()
    if cfg and cfg.get("type") == "hf" and cfg.get("n_layers"):
        return int(cfg["n_layers"])
    return MODEL_CONFIG.n_layer


# ── TinyGPT compatibility shim ────────────────────────────────────────────────
# TinyGPT doesn't have the `extra` parameter. This wrapper adds it.

class TinyGPTShim:
    def __init__(self, model): self._m = model
    def forward_embed(self, idx):             return self._m.forward_embed(idx), None
    def forward_blocks(self, x, extra, s, e): return self._m.forward_blocks(x, s, e), extra
    def forward_head(self, x, extra):         return self._m.forward_head(x)


def wrap_model(raw_model, model_type: str):
    if model_type == "tinygpt":
        return TinyGPTShim(raw_model)
    return raw_model  # HFNodeModel already has the right interface


# ── Hot reload ────────────────────────────────────────────────────────────────

class HotReloadingModel:
    def __init__(self, label, start, end, total, poll=3.0):
        self.label  = label
        self.start  = start
        self.end    = end
        self.total  = total
        self.lock   = threading.RLock()
        self._stop  = threading.Event()

        raw, mtype = load_model(start, end, total)
        self.model = wrap_model(raw, mtype)
        self._mtime = self._latest_mtime()

    def _latest_mtime(self):
        mt = []
        for p in [ACTIVE_MODEL_PATH, Path(CHECKPOINT_PATH)]:
            try: mt.append(os.path.getmtime(p))
            except OSError: pass
        return max(mt) if mt else 0.0

    def start_watching(self, poll=3.0):
        def loop():
            while not self._stop.is_set():
                time.sleep(poll)
                try:
                    mt = self._latest_mtime()
                    if mt != self._mtime:
                        self._reload(mt)
                except Exception:
                    pass
        threading.Thread(target=loop, daemon=True).start()

    def _reload(self, new_mtime):
        try:
            raw, mtype = load_model(self.start, self.end, self.total)
            new_model  = wrap_model(raw, mtype)
        except Exception:
            print(f"[{self.label}] reload failed (file may be mid-write) — retrying.")
            return
        with self.lock:
            self.model  = new_model
            self._mtime = new_mtime
        print(f"[{self.label}] hot-reloaded model.")

    def stop(self): self._stop.set()


# ── Management HTTP server ────────────────────────────────────────────────────

class _DownloadState:
    def __init__(self):
        self.status    = "idle"   # idle | downloading | done | error
        self.model_id  = ""
        self.progress  = ""
        self.error_msg = ""
        self.lock      = threading.Lock()

_dl_state = _DownloadState()


def _make_mgmt_handler(hot_model, label, dl_state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass  # suppress access log

        def _send(self, code, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "label": label})

            elif self.path == "/benchmark":
                result = benchmark_node_local()
                self._send(200, result)

            elif self.path == "/download_status":
                with dl_state.lock:
                    self._send(200, {
                        "status":   dl_state.status,
                        "model_id": dl_state.model_id,
                        "progress": dl_state.progress,
                        "error":    dl_state.error_msg,
                    })
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/download":
                length = int(self.headers.get("Content-Length", 0))
                body   = json.loads(self.rfile.read(length))
                model_id  = body.get("model_id", "")
                hf_id     = body.get("hf_id", "")
                cache_dir = body.get("cache_dir", "")
                arch      = body.get("arch", "auto")

                with dl_state.lock:
                    if dl_state.status == "downloading":
                        self._send(409, {"error": "already downloading"})
                        return
                    dl_state.status   = "downloading"
                    dl_state.model_id = model_id
                    dl_state.progress = "starting"
                    dl_state.error_msg = ""

                def do_download():
                    try:
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                        with dl_state.lock:
                            dl_state.progress = "downloading tokenizer"
                        AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
                        with dl_state.lock:
                            dl_state.progress = "downloading model weights"
                        AutoModelForCausalLM.from_pretrained(
                            hf_id, cache_dir=cache_dir,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                        )
                        with dl_state.lock:
                            dl_state.status   = "done"
                            dl_state.progress = "complete"
                        print(f"[{label}] downloaded {model_id}")
                    except Exception as e:
                        with dl_state.lock:
                            dl_state.status    = "error"
                            dl_state.error_msg = str(e)
                        print(f"[{label}] download failed: {e}")

                threading.Thread(target=do_download, daemon=True).start()
                self._send(200, {"status": "started"})
            else:
                self._send(404, {"error": "not found"})

    return Handler


def start_mgmt_server(hot_model, label: str, mgmt_port: int):
    handler = _make_mgmt_handler(hot_model, label, _dl_state)
    srv = HTTPServer(("0.0.0.0", mgmt_port), handler)
    t   = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"[{label}] management server on port {mgmt_port}")
    return srv


# ── Registry client ───────────────────────────────────────────────────────────

class RegistryClient:
    def __init__(self, url, host, port, start, end, label, hw):
        self.url   = url.rstrip("/")
        self.label = label
        self._p    = {"host": host, "port": port,
                      "start_layer": start, "end_layer": end,
                      "label": label, "hw_specs": hw}
        self._served = 0
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

    def increment(self):
        with self._lock: self._served += 1

    def announce(self):
        while not self._stop.is_set():
            try:
                r = http.post(f"{self.url}/announce", json=self._p, timeout=5)
                if r.status_code == 200:
                    iv = r.json().get("heartbeat_interval", 5)
                    print(f"[{self.label}] registered with registry at {self.url}")
                    return iv
                print(f"[{self.label}] registry rejected: {r.text} — retrying in 3s")
            except Exception as e:
                print(f"[{self.label}] registry unreachable ({e}) — retrying in 3s")
            time.sleep(3)
        return 5.0

    def start_heartbeat(self, iv):
        def loop():
            while not self._stop.is_set():
                time.sleep(iv)
                try:
                    with self._lock: served = self._served
                    http.post(f"{self.url}/heartbeat", json={
                        "host": self._p["host"], "port": self._p["port"],
                        "requests_served": served,
                    }, timeout=4)
                except Exception: pass
        threading.Thread(target=loop, daemon=True).start()

    def stop(self): self._stop.set()


# ── Inference request handling ────────────────────────────────────────────────

def forward_to_next(host, port, message):
    with socket.create_connection((host, port), timeout=30) as s:
        send_msg(s, message)
        return recv_msg(s)


def handle_connection(conn, addr, hot, start, end, total, label, reg):
    try:
        try:
            req = recv_msg(conn)
        except ConnectionError:
            return  # startup probe

        chain = req["chain"]
        ci    = req["chain_index"]
        timing = req.get("timing", [])
        extra  = req.get("extra", None)   # RoPE state for Llama, None otherwise
        is_first = start == 0
        is_last  = end   == total

        t0 = time.perf_counter()
        with hot.lock:
            model = hot.model
            if is_first:
                x, extra = model.forward_embed(req["data"])
            else:
                x = req["data"]
            x, extra = model.forward_blocks(x, extra, start, end)
            logits = model.forward_head(x, extra) if is_last else None
        compute_ms = (time.perf_counter() - t0) * 1000

        timing.append({"label": label, "compute_ms": round(compute_ms, 2)})
        print(f"[{label}] layers {start}-{end-1} | "
              f"shape {tuple(x.shape)} | {compute_ms:.1f}ms | "
              f"from {addr[0]}:{addr[1]}")

        if is_last:
            response = {"type": "result", "data": logits, "timing": timing}
        else:
            next_n = chain[ci + 1]
            # Give a helpful error if the next node isn't reachable
            try:
                response = forward_to_next(next_n["host"], next_n["port"], {
                    "type": "hidden", "data": x, "extra": extra,
                    "chain": chain, "chain_index": ci + 1, "timing": timing,
                })
            except ConnectionRefusedError:
                raise ConnectionRefusedError(
                    f"Could not reach the next node '{next_n['label']}' at "
                    f"{next_n['host']}:{next_n['port']}. "
                    f"Is it running? Check that node is started and its firewall allows port {next_n['port']}."
                )

        send_msg(conn, response)
        reg.increment()

    except Exception:
        tb = traceback.format_exc()
        print(f"[{label}] ERROR:\n{tb}")
        try:
            send_msg(conn, {"type": "error", "data": tb})
        except Exception: pass
    finally:
        conn.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     type=int, required=True)
    parser.add_argument("--start",    type=int, required=True)
    parser.add_argument("--end",      type=int, required=True)
    parser.add_argument("--registry", default=os.environ.get("REGISTRY_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--label",    default=None)
    parser.add_argument("--no-hot-reload",   action="store_true")
    parser.add_argument("--reload-interval", type=float, default=3.0)
    args = parser.parse_args()

    if not (0 <= args.start < args.end):
        parser.error("--start must be >= 0 and less than --end")

    label = args.label or f"node@{args.port}"
    total = total_layers_for_active()

    if args.end > total:
        parser.error(
            f"--end {args.end} exceeds the active model's total layers ({total}). "
            f"Max valid value is {total}."
        )

    print(f"[{label}] benchmarking hardware...")
    hw = benchmark_node_local()
    print(f"[{label}] RAM {hw['ram_free_gb']:.1f}GB free | "
          f"{hw['tokens_per_sec']:.0f} tok/s | "
          f"GPU {'yes' if hw['has_gpu'] else 'no'}")

    print(f"[{label}] loading model (layers {args.start}-{args.end-1} of {total})...")
    hot = HotReloadingModel(label, args.start, args.end, total)
    if not args.no_hot_reload:
        hot.start_watching(args.reload_interval)
        print(f"[{label}] hot reload active.")

    reg = RegistryClient(args.registry, args.host, args.port,
                         args.start, args.end, label, hw)
    iv  = reg.announce()
    reg.start_heartbeat(iv)

    mgmt_port = args.port + 10000
    start_mgmt_server(hot, label, mgmt_port)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(16)
    print(f"[{label}] ready — listening on {args.host}:{args.port}")

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(
                target=handle_connection,
                args=(conn, addr, hot, args.start, args.end, total, label, reg),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        print(f"\n[{label}] shutting down.")
    finally:
        reg.stop()
        hot.stop()
        srv.close()


if __name__ == "__main__":
    main()
