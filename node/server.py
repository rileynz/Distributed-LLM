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
from shared.netinfo import detect_lan_ip, describe_bind_error
from shared.protocol import send_msg, recv_msg


# ── Registry sync (network-wide active model + this node's layer range) ───────
# These make the registry the one source of truth both concepts, instead of
# a purely local active_model.json a remote node has no way to see: any node,
# on any machine, ends up running whatever model/layer-range the registry
# currently says — not just nodes that happen to share a filesystem with
# whoever triggered the change.

def pull_active_model_from_registry(registry_url: str):
    """Best-effort: asks the registry what model should be active, and if
    it disagrees with (or is newer than) the local active_model.json,
    overwrites the local file with it.
    Returns (updated: bool, reason: str) — reason is always a short label
    ("up to date", "unreachable (...)", "updated to <id>", etc.) so the
    caller can log state *transitions* (e.g. reachable -> unreachable)
    instead of this failing completely silently, which is exactly what
    made a real sync failure indistinguishable from "nothing to do"."""
    try:
        r = http.get(f"{registry_url.rstrip('/')}/active_model", timeout=3)
        if r.status_code != 200:
            return False, f"registry returned HTTP {r.status_code}"
        remote = r.json().get("active")
        if not remote:
            return False, "registry has no active model set yet"
        local = read_active()
        if local is not None and remote.get("updated_at", 0) <= local.get("updated_at", 0):
            return False, "up to date"
        with open(ACTIVE_MODEL_PATH, "w") as f:
            json.dump(remote, f, indent=2)
        return True, f"updated to {remote.get('model_id', '?')}"
    except Exception as e:
        return False, f"unreachable ({e})"


def pull_assignment_from_registry(registry_url: str, label: str):
    """Best-effort: asks the registry for this node's (label-keyed) layer
    assignment. Returns ((start, end, total) or None, reason) — the
    assignment is None if there isn't one yet (e.g. this is a pool node
    awaiting a model pick), same reasoning as pull_active_model_from_registry
    above for why `reason` exists."""
    try:
        r = http.get(f"{registry_url.rstrip('/')}/assignments", timeout=3)
        if r.status_code != 200:
            return None, f"registry returned HTTP {r.status_code}"
        a = (r.json().get("assignments") or {}).get(label)
        if not a:
            return None, "no assignment yet"
        return (int(a["start_layer"]), int(a["end_layer"]), int(a["total_layers"])), "ok"
    except Exception as e:
        return None, f"unreachable ({e})"


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
    arch         = cfg.get("arch", "auto")
    quantization = cfg.get("quantization", "none")
    return HFNodeModel(cfg["hf_id"], cfg["cache_dir"], start, end, total,
                        arch=arch, quantization=quantization)

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
# TinyGPT's forward_blocks takes (start, end) directly rather than having
# them baked in at construction time like the HF wrappers do. This shim
# adapts it to the same forward_embed/forward_blocks/forward_head contract
# HFNodeModel exposes, including past_length / past_kv for KV caching.

class TinyGPTShim:
    def __init__(self, model, start, end):
        self._m     = model
        self._start = start
        self._end   = end

    def forward_embed(self, idx, past_length=0):
        return self._m.forward_embed(idx, past_length=past_length), None

    def forward_blocks(self, x, extra, past_kv=None, use_cache=False):
        past_kvs = past_kv if use_cache else None
        x, present_kvs = self._m.forward_blocks(x, self._start, self._end, past_kvs=past_kvs)
        return x, extra, (present_kvs if use_cache else None)

    def forward_head(self, x, extra):
        return self._m.forward_head(x)


def wrap_model(raw_model, model_type: str, start: int, end: int):
    if model_type == "tinygpt":
        return TinyGPTShim(raw_model, start, end)
    return raw_model  # HFNodeModel already has the right interface


# ── KV cache store ────────────────────────────────────────────────────────────
# Each node keeps its OWN cache of in-flight requests' keys/values — caches
# never cross the wire, only the (small) new-token hidden state and a
# request_id do. Freed as soon as the client signals it's done with a
# request; swept automatically after an idle timeout as a backstop in case
# that signal never arrives (client crash, dropped connection, etc.).

class KVCacheStore:
    IDLE_TIMEOUT_S = 600  # 10 minutes of inactivity before a forgotten request is freed

    def __init__(self):
        self._entries = {}   # request_id -> {"cache": ..., "last_used": float}
        self._lock    = threading.Lock()
        self._stop    = threading.Event()

    @staticmethod
    def _fresh(is_hf: bool):
        if is_hf:
            from transformers import DynamicCache
            return DynamicCache()
        return {}  # TinyGPT: {layer_idx: (k, v)}

    def get(self, request_id: str, is_hf: bool):
        """Returns this request's cache, creating an empty one if new."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                entry = {"cache": self._fresh(is_hf), "last_used": time.time()}
                self._entries[request_id] = entry
            else:
                entry["last_used"] = time.time()
            return entry["cache"]

    def put(self, request_id: str, cache):
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None:
                entry["cache"]     = cache
                entry["last_used"] = time.time()

    def close(self, request_id: str):
        with self._lock:
            self._entries.pop(request_id, None)

    def clear(self):
        """Called whenever the model hot-reloads — old caches reference
        the previous model's weights/shapes and are no longer meaningful."""
        with self._lock:
            self._entries.clear()

    def start_sweeper(self, interval=60):
        def loop():
            while not self._stop.is_set():
                time.sleep(interval)
                now = time.time()
                with self._lock:
                    stale = [rid for rid, e in self._entries.items()
                             if now - e["last_used"] > self.IDLE_TIMEOUT_S]
                    for rid in stale:
                        del self._entries[rid]
        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def __len__(self):
        with self._lock:
            return len(self._entries)


# ── Hot reload ────────────────────────────────────────────────────────────────

class HotReloadingModel:
    def __init__(self, label, start, end, total, registry_url=None):
        self.label    = label
        self.start    = start   # None until this node has an assignment
        self.end      = end     # None until this node has an assignment
        self.total    = total   # None until this node has an assignment
        self.registry_url = registry_url
        self.assigned = start is not None and end is not None
        self.lock   = threading.RLock()
        self._stop  = threading.Event()
        self.kv_store = KVCacheStore()
        self.model      = None
        self.model_type = None
        self._mtime = 0.0

        if self.assigned:
            if self.registry_url:
                pull_active_model_from_registry(self.registry_url)  # best-effort initial sync
            raw, mtype = load_model(start, end, total)
            self.model      = wrap_model(raw, mtype, start, end)
            self.model_type = mtype
            self._mtime = self._latest_mtime()

        # Tracks the last-seen reason from each registry poll so start_watching
        # can log state *transitions* only (e.g. reachable -> unreachable, or
        # unreachable -> recovered) instead of either spamming every 3 seconds
        # or — the previous behavior — never logging anything at all, which is
        # exactly what made a real sync failure indistinguishable from "there
        # was nothing to update."
        self._last_model_reason  = None
        self._last_assign_reason = None

    def _log_transition(self, kind, reason):
        attr = f"_last_{kind}_reason"
        if reason != getattr(self, attr, None):
            print(f"[{self.label}] {kind} poll: {reason}")
        setattr(self, attr, reason)

    def _latest_mtime(self):
        mt = []
        for p in [ACTIVE_MODEL_PATH, Path(CHECKPOINT_PATH)]:
            try: mt.append(os.path.getmtime(p))
            except OSError: pass
        return max(mt) if mt else 0.0

    def start_watching(self, poll=3.0, on_assigned=None):
        """Runs forever in the background, reconciling this node against
        the registry every `poll` seconds:
          - if unassigned, checks for a first-time layer assignment
          - if assigned, checks for a re-split (a new assignment) or a
            model swap (local file changed, e.g. because a pull above
            just updated it, or because it changed locally)
        `on_assigned(start, end, total)` is called whenever a (re-)assignment
        takes effect, so the caller can re-announce to the registry with
        the correct layer range — a plain local reload doesn't need this,
        since the registry already knows this node's range in that case.
        """
        def loop():
            while not self._stop.is_set():
                time.sleep(poll)
                try:
                    if self.registry_url:
                        got, reason = pull_assignment_from_registry(self.registry_url, self.label)
                        self._log_transition("assignment", reason)
                        if got and got != (self.start, self.end, self.total):
                            _, mreason = pull_active_model_from_registry(self.registry_url)
                            self._log_transition("active-model", mreason)
                            # A new layer split is pushed to the registry as
                            # soon as a model switch is triggered, well
                            # before the new model finishes downloading and
                            # gets broadcast as active — so `got`'s total
                            # layer count can briefly belong to a DIFFERENT
                            # model than the one still active locally.
                            # Activating against that mismatch would reload
                            # the OLD model sliced as if it were the NEW
                            # one. Wait for the active model to actually
                            # catch up to this assignment's layer count
                            # before reloading.
                            active_total = total_layers_for_active()
                            if got[2] != active_total:
                                self._log_transition(
                                    "assignment",
                                    f"new range expects a {got[2]}-layer model, but the "
                                    f"active model has {active_total} layers — waiting "
                                    f"for the model switch to catch up before reloading."
                                )
                                continue
                            if self._activate(*got) and on_assigned:
                                on_assigned(*got)
                            continue
                        _, mreason = pull_active_model_from_registry(self.registry_url)
                        self._log_transition("active-model", mreason)
                    if not self.assigned:
                        continue
                    mt = self._latest_mtime()
                    if mt != self._mtime:
                        self._reload(mt)
                except Exception:
                    pass
        threading.Thread(target=loop, daemon=True).start()

    def _activate(self, start, end, total) -> bool:
        """(Re)loads the model for a (possibly new) layer range. Used both
        for a pool node's first assignment and for re-splitting an
        already-serving node. Returns True on success."""
        if not (0 <= start < end <= total):
            print(f"[{self.label}] ⚠ received an invalid assignment (layers {start}-{end-1} "
                  f"of {total}) from the registry — ignoring it, keeping current state.")
            return False
        try:
            raw, mtype = load_model(start, end, total)
            new_model  = wrap_model(raw, mtype, start, end)
        except Exception:
            print(f"[{self.label}] activation failed (model file may be mid-write, "
                  f"or weights still downloading) — retrying.")
            return False
        with self.lock:
            self.start, self.end, self.total = start, end, total
            self.model, self.model_type = new_model, mtype
            self.assigned = True
            self._mtime = self._latest_mtime()
            n_dropped = len(self.kv_store)
            self.kv_store.clear()
        extra = f", dropped {n_dropped} in-flight cache(s)" if n_dropped else ""
        print(f"[{self.label}] now serving layers {start}-{end-1} of {total}{extra}.")
        return True

    def _reload(self, new_mtime):
        # Re-derive the total layer count from whatever model is *now*
        # active rather than reusing the possibly-stale self.total — a
        # plain reload (as opposed to a fresh /assignments push) never
        # touched this before, so switching to a model with a different
        # layer count than the one this node was started/last assigned
        # for would silently misclassify first/last-node status, or
        # crash-loop forever trying to slice layers that don't exist.
        total = total_layers_for_active()
        if self.end > total:
            print(f"[{self.label}] ⚠ active model changed to one with only {total} layers, "
                  f"but this node is configured for layers {self.start}-{self.end - 1}. "
                  f"It can't serve the new model as configured — restart it with a layer "
                  f"range that fits (or add it to the pool and use the MODELS menu's "
                  f"auto-assign instead), or switch back to a model with at least "
                  f"{self.end} layers.")
            return
        try:
            raw, mtype = load_model(self.start, self.end, total)
            new_model  = wrap_model(raw, mtype, self.start, self.end)
        except Exception:
            print(f"[{self.label}] reload failed (file may be mid-write) — retrying.")
            return
        with self.lock:
            self.model      = new_model
            self.model_type = mtype
            self.total      = total
            self._mtime     = new_mtime
            # Old caches reference the previous model's weights/shapes —
            # they're meaningless (and possibly wrong-shaped) now.
            n_dropped = len(self.kv_store)
            self.kv_store.clear()
        extra = f", dropped {n_dropped} in-flight cache(s)" if n_dropped else ""
        print(f"[{self.label}] hot-reloaded model{extra}.")

    def stop(self):
        self._stop.set()
        self.kv_store.stop()


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
                # Optional: this node's own layer assignment. When given,
                # only the shard files this node actually needs get
                # downloaded instead of the whole checkpoint. Absent (e.g.
                # an older caller, or a pool node not yet assigned) just
                # means "download everything", same as before.
                start_layer   = body.get("start_layer")
                end_layer     = body.get("end_layer")
                total_layers  = body.get("total_layers")

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
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                        with dl_state.lock:
                            dl_state.progress = "downloading tokenizer"
                        from transformers import AutoTokenizer
                        AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)

                        have_assignment = None not in (start_layer, end_layer, total_layers)
                        used_selective  = False
                        if have_assignment:
                            try:
                                with dl_state.lock:
                                    dl_state.progress = f"downloading layers {start_layer}-{end_layer-1} only"

                                def _cb(done, total, fname):
                                    with dl_state.lock:
                                        dl_state.progress = f"downloading weights ({done}/{total}: {fname})"

                                from models.lazy_loader import download_needed_shards
                                download_needed_shards(hf_id, cache_dir, arch,
                                                        start_layer, end_layer, total_layers,
                                                        progress_cb=_cb)
                                used_selective = True
                            except Exception as e:
                                print(f"[{label}] selective download unavailable ({e}) "
                                      f"— falling back to downloading the whole checkpoint.")

                        if not used_selective:
                            with dl_state.lock:
                                dl_state.progress = "downloading model weights"
                            from transformers import AutoModelForCausalLM
                            AutoModelForCausalLM.from_pretrained(
                                hf_id, cache_dir=cache_dir,
                                torch_dtype=torch.float32,
                                low_cpu_mem_usage=True,
                            )

                        with dl_state.lock:
                            dl_state.status   = "done"
                            dl_state.progress = "complete"
                        print(f"[{label}] downloaded {model_id}"
                              f"{' (layers only)' if used_selective else ' (full checkpoint)'}")
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
    try:
        srv = HTTPServer(("0.0.0.0", mgmt_port), handler)
    except OSError as e:
        print(f"\n[{label}] {describe_bind_error(e, mgmt_port)}\n")
        sys.exit(1)
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

    def reannounce_now(self, start, end):
        """Single-attempt re-announce with an updated layer range (e.g. a
        pool node just got its first assignment, or was re-split). Unlike
        announce(), this doesn't retry forever — if it fails, the next
        heartbeat/poll cycle will naturally retry via the normal path."""
        with self._lock:
            self._p["start_layer"] = start
            self._p["end_layer"]   = end
        try:
            http.post(f"{self.url}/announce", json=self._p, timeout=5)
        except Exception:
            pass

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

def _cached_length(past_kv, is_hf: bool) -> int:
    """How many positions this node has already cached for its own first
    owned layer. Only the first node in the chain needs this (to compute
    correct absolute positions for embeddings/RoPE) — derived from the
    node's own cache rather than trusted from the client, so it can never
    drift out of sync with what's actually stored."""
    if past_kv is None:
        return 0
    if is_hf:
        try:
            return int(past_kv.get_seq_length())
        except Exception:
            return 0
    # TinyGPT: past_kv is {layer_idx: (k, v)}; the first node's lowest
    # owned layer_idx is always 0.
    kv0 = past_kv.get(0)
    return kv0[0].shape[2] if kv0 is not None else 0


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

        # Lightweight control message: the client is done with a request —
        # free its cache now instead of waiting for the idle sweeper.
        if req.get("type") == "close":
            hot.kv_store.close(req.get("request_id", ""))
            send_msg(conn, {"type": "closed"})
            return

        chain      = req["chain"]
        ci         = req["chain_index"]
        timing     = req.get("timing", [])
        extra      = req.get("extra", None)   # RoPE state for Llama, None otherwise
        request_id = req.get("request_id")
        use_cache  = bool(request_id) and bool(req.get("use_cache", False))

        if hot.model is None:
            send_msg(conn, {"error": f"node '{label}' has no model assigned yet — "
                             "it shouldn't have been in the chain. Try again shortly."})
            return

        is_first   = start == 0
        is_last    = end   == total

        t0 = time.perf_counter()
        with hot.lock:
            model  = hot.model
            is_hf  = (hot.model_type == "hf")
            past_kv = hot.kv_store.get(request_id, is_hf) if use_cache else None

            if is_first:
                past_length = _cached_length(past_kv, is_hf) if use_cache else 0
                x, extra = model.forward_embed(req["data"], past_length=past_length)
            else:
                x = req["data"]
            x, extra, present_kv = model.forward_blocks(x, extra, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                hot.kv_store.put(request_id, present_kv)
            logits = model.forward_head(x, extra) if is_last else None
        compute_ms = (time.perf_counter() - t0) * 1000

        timing.append({"label": label, "compute_ms": round(compute_ms, 2)})
        cache_note = " | cached" if use_cache else ""
        print(f"[{label}] layers {start}-{end-1} | "
              f"shape {tuple(x.shape)} | {compute_ms:.1f}ms{cache_note} | "
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
                    "request_id": request_id, "use_cache": use_cache,
                })
            except ConnectionRefusedError:
                raise ConnectionRefusedError(
                    f"Could not reach the next node '{next_n['label']}' at "
                    f"{next_n['host']}:{next_n['port']}. "
                    f"Is it running? Check that node is started, its firewall allows "
                    f"port {next_n['port']}, and that {next_n['host']} is that node's "
                    f"actual LAN IP (not 127.0.0.1, unless it really is on this same machine)."
                )
            except OSError as e:
                raise ConnectionRefusedError(
                    f"Could not reach the next node '{next_n['label']}' at "
                    f"{next_n['host']}:{next_n['port']} ({e}). "
                    f"If that address looks wrong (e.g. 127.0.0.1 for a node on another "
                    f"machine), restart it with the correct --host, or accept the "
                    f"auto-detected IP the launcher suggests."
                )

        send_msg(conn, response)
        reg.increment()

    except (ConnectionError, TimeoutError) as e:
        # A node somewhere down the chain died or dropped off the network —
        # this is an environment failure, not a bug. Send back a short,
        # clean message and a flag so the client knows retrying the same
        # in-flight request isn't safe (a node earlier in the chain may
        # have already committed a KV-cache update for this step) but that
        # this is a known, recoverable situation rather than a crash.
        print(f"[{label}] node unreachable: {e}")
        try:
            send_msg(conn, {"type": "error", "node_failure": True, "data": str(e)})
        except Exception: pass
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
    parser.add_argument("--host",     default=None,
                         help="IP address this node advertises to the registry so "
                              "other machines can reach it. Auto-detected as this "
                              "machine's LAN IP if not given. The node itself always "
                              "listens on all network interfaces, so this only "
                              "controls what address gets handed out — pass 127.0.0.1 "
                              "explicitly if you really want a loopback-only node.")
    parser.add_argument("--port",     type=int, required=True)
    parser.add_argument("--start",    type=int, default=None,
                         help="Start layer (inclusive). Omit both --start and --end to "
                              "join as an unassigned 'pool' node instead — it announces "
                              "itself and waits for a layer range + model to be pushed "
                              "via the registry (see run.py's ADD NODE / MODELS menu, "
                              "which benchmarks the pool and recommends + assigns for you).")
    parser.add_argument("--end",      type=int, default=None,
                         help="End layer (exclusive). See --start.")
    parser.add_argument("--registry", default=os.environ.get("REGISTRY_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--label",    default=None)
    parser.add_argument("--no-hot-reload",   action="store_true")
    parser.add_argument("--reload-interval", type=float, default=3.0)
    args = parser.parse_args()

    if args.host is None:
        args.host = detect_lan_ip()

    pool_mode = args.start is None and args.end is None
    if not pool_mode:
        if args.start is None or args.end is None:
            parser.error("--start and --end must be given together "
                          "(or both omitted to join as an unassigned pool node).")
        if not (0 <= args.start < args.end):
            parser.error("--start must be >= 0 and less than --end")

    label = args.label or f"node@{args.port}"

    print(f"[{label}] benchmarking hardware...")
    hw = benchmark_node_local()
    print(f"[{label}] RAM {hw['ram_free_gb']:.1f}GB free | "
          f"{hw['tokens_per_sec']:.0f} tok/s | "
          f"GPU {'yes' if hw['has_gpu'] else 'no'}")

    if pool_mode:
        print(f"[{label}] no --start/--end given — joining as an unassigned pool node.")
        print(f"[{label}] waiting for a layer range + model to be assigned via the registry...")
        total = None
    else:
        pull_active_model_from_registry(args.registry)
        total = total_layers_for_active()
        if args.end > total:
            parser.error(
                f"--end {args.end} exceeds the active model's total layers ({total}). "
                f"Max valid value is {total}."
            )
        print(f"[{label}] loading model (layers {args.start}-{args.end-1} of {total})...")

    hot = HotReloadingModel(label, args.start, args.end, total, registry_url=args.registry)
    hot.kv_store.start_sweeper()

    reg = RegistryClient(args.registry, args.host, args.port,
                         args.start, args.end, label, hw)
    iv  = reg.announce()
    reg.start_heartbeat(iv)

    if not args.no_hot_reload:
        hot.start_watching(args.reload_interval,
                            on_assigned=lambda s, e, t: reg.reannounce_now(s, e))
        print(f"[{label}] hot reload / auto-assignment watcher active.")
    elif pool_mode:
        print(f"[{label}] ⚠  --no-hot-reload was given on a pool node — it will never "
              f"pick up an assignment. Drop that flag if you want auto-provisioning.")

    mgmt_port = args.port + 10000
    start_mgmt_server(hot, label, mgmt_port)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Always listen on every interface. `args.host` is only what we tell the
    # registry other machines should use to reach us — binding to that exact
    # address (instead of 0.0.0.0) was the bug that made cross-machine setups
    # silently fail: it either bound to loopback only, or (if a LAN IP was
    # given manually) could fail to bind at all if that IP briefly wasn't up.
    try:
        srv.bind(("0.0.0.0", args.port))
    except OSError as e:
        print(f"\n[{label}] {describe_bind_error(e, args.port)}\n")
        sys.exit(1)
    srv.listen(16)
    print(f"[{label}] ready — listening on all interfaces, port {args.port}")
    print(f"[{label}] advertising to registry as {args.host}:{args.port}")
    if args.host == "127.0.0.1":
        print(f"[{label}] ⚠  advertising 127.0.0.1 — this node will only be reachable "
              f"from THIS machine. If other machines need to connect to it, restart "
              f"with --host <this machine's LAN IP>.")

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(
                target=handle_connection,
                args=(conn, addr, hot, hot.start, hot.end, hot.total, label, reg),
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
