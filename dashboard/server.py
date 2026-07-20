"""
Web dashboard — live view of the node pipeline as tokens flow through it,
plus model switching/downloading/quantization from a browser instead of
the terminal menu.

This never starts, stops, or manages node processes — it talks to the
same registry and per-node management HTTP endpoints download_model.py
and coordinator/client.py already use. Nothing here can do anything a
person couldn't already do from the CLI.

Usage:
    python dashboard/server.py
    python dashboard/server.py --port 7000 --registry http://127.0.0.1:8000
"""

import argparse
import os
import sys
import socket
import threading
import time
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request, send_from_directory
import requests as http

from config import MODEL_CONFIG
from models.active import read_active, write_hf, write_tinygpt
from models.catalogue import CATALOGUE, get_model
from coordinator.client import generate
from shared.provisioning import (
    nodes_identity, split_layers_by_capacity, too_many_nodes_for_model,
    push_assignments, start_node_download, poll_node_download,
)
from shared.netinfo import describe_bind_error

STATIC_DIR = Path(__file__).resolve().parent / "static"

REGISTRY_URL = os.environ.get("REGISTRY_URL", "http://127.0.0.1:8000")

app = Flask(__name__, static_folder=None)


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# ── Registry helpers ───────────────────────────────────────────────────────────

def _fetch_registry_status():
    """Returns (status_dict, error_string). Exactly one is None."""
    try:
        r = http.get(f"{REGISTRY_URL}/status", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _alive_nodes():
    status, err = _fetch_registry_status()
    if err or status is None:
        return []
    return [n for n in status.get("nodes", []) if n.get("alive")]


# ── Topology ───────────────────────────────────────────────────────────────────

@app.get("/api/topology")
def api_topology():
    cfg    = read_active() or {}
    status, err = _fetch_registry_status()

    if err:
        return jsonify({
            "registry_ok": False,
            "error": err,
            "chain_ready": False,
            "chain_error": None,
            "nodes": [],
            "pool_nodes": 0,
            "model_id": cfg.get("model_id", "tinygpt"),
            "quantization": cfg.get("quantization", "none"),
        })

    all_nodes = status.get("nodes", [])
    # The pipeline view shows the actual inference chain — pool (unassigned)
    # nodes aren't part of it yet, and their start_layer is None, which
    # can't be sorted against an int.
    assigned = [n for n in all_nodes if n.get("start_layer") is not None]
    nodes    = sorted(assigned, key=lambda n: n["start_layer"])
    pool_n   = sum(1 for n in all_nodes if n.get("start_layer") is None and n.get("alive"))
    return jsonify({
        "registry_ok": True,
        "error": None,
        "chain_ready": status.get("chain_ready", False),
        "chain_error": status.get("chain_error"),
        "nodes": nodes,
        "pool_nodes": pool_n,
        "model_id": cfg.get("model_id", "tinygpt"),
        "quantization": cfg.get("quantization", "none"),
    })


# ── Node gate (confirm the full network before any distributed download) ─────
# Mirrors the CLI's hard-stop gate in shared.provisioning.cli_confirm_nodes:
# nothing gets split or downloaded until the person has explicitly reviewed
# and confirmed the current node list from the browser. "Confirmed" is
# tracked as the *identity* of that node set (host, port) rather than a
# timestamp, so it stays valid across polls as long as the set hasn't
# changed, and is automatically invalidated the instant it does.

_prov_lock = threading.Lock()
_prov_confirmed_identity = None


@app.get("/api/provision/nodes")
def api_provision_nodes():
    nodes = _alive_nodes()
    with _prov_lock:
        confirmed = (_prov_confirmed_identity is not None and
                     _prov_confirmed_identity == nodes_identity(nodes))
    return jsonify({
        "confirmed": confirmed,
        "nodes": [{
            "label":       n["label"],
            "host":        n["host"],
            "port":        n["port"],
            "hw_specs":    n.get("hw_specs", {}),
            "start_layer": n.get("start_layer"),
            "end_layer":   n.get("end_layer"),
        } for n in sorted(nodes, key=lambda x: x["label"])],
    })


@app.post("/api/provision/confirm")
def api_provision_confirm():
    global _prov_confirmed_identity
    nodes = _alive_nodes()
    with _prov_lock:
        _prov_confirmed_identity = nodes_identity(nodes)
    return jsonify({"confirmed": True, "n_nodes": len(nodes)})


def _nodes_confirmed(nodes: list[dict]) -> bool:
    with _prov_lock:
        return (_prov_confirmed_identity is not None and
                _prov_confirmed_identity == nodes_identity(nodes))


@app.get("/api/catalogue")
def api_catalogue():
    items = [{
        "id": "tinygpt",
        "name": "TinyGPT (built-in)",
        "description": "The small from-scratch model this project ships with — no download needed.",
        "download_gb": 0,
        "quality": 1,
        "n_layers": MODEL_CONFIG.n_layer,
        "min_nodes": 1,
        "tags": ["built-in"],
        "notes": "",
        "supports_quantization": False,
    }]
    for m in CATALOGUE:
        items.append({
            "id": m.id,
            "name": m.name,
            "description": m.description,
            "download_gb": m.download_gb,
            "quality": m.quality,
            "n_layers": m.n_layers,
            "min_nodes": m.min_nodes,
            "tags": list(m.tags),
            "notes": m.notes,
            "supports_quantization": m.supports_quantization,
        })
    return jsonify(items)


# ── Model switch / download, with live progress ───────────────────────────────

_switch_lock  = threading.Lock()
_switch_state = {"status": "idle", "detail": "", "model_id": None}


def _set_switch_state(**kwargs):
    with _switch_lock:
        _switch_state.update(kwargs)


@app.get("/api/switch/status")
def api_switch_status():
    with _switch_lock:
        return jsonify(dict(_switch_state))


@app.post("/api/switch")
def api_switch():
    body         = request.get_json(force=True, silent=True) or {}
    model_id     = (body.get("model_id") or "").strip()
    quantization = body.get("quantization", "none")

    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    if quantization not in ("none", "int8", "int4"):
        return jsonify({"error": "quantization must be 'none', 'int8', or 'int4'"}), 400

    with _switch_lock:
        if _switch_state["status"] == "working":
            return jsonify({"error": "A model switch is already in progress"}), 409

    if model_id == "tinygpt":
        # TinyGPT doesn't need the node list *confirmed* (no download,
        # nothing to review) but it still needs to know which nodes are
        # online right now so it can re-split THEIR layers for TinyGPT's
        # layer count — skipping that (as this used to) left nodes still
        # holding a stale split from whatever model was active before,
        # which never matches TinyGPT and left them waiting forever.
        nodes = _alive_nodes()
        threading.Thread(target=_switch_tinygpt_worker, args=(nodes,), daemon=True).start()
        return jsonify({"status": "started"})

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": f"Unknown model id '{model_id}'"}), 404
    if quantization != "none" and not entry.supports_quantization:
        quantization = "none"  # silently ignore rather than fail — model just doesn't support it

    # Hard stop: a real distributed download only ever splits across a
    # node list the person has actually reviewed and confirmed in the
    # Network panel — re-checked here (not just trusted from the click)
    # so a stale confirmation can never be replayed against a node set
    # that has since changed.
    nodes = _alive_nodes()
    if nodes and not _nodes_confirmed(nodes):
        return jsonify({"error": "Your node list hasn't been confirmed (or it changed since "
                                  "you last confirmed it). Review it in the Network panel "
                                  "above, click Confirm, then try again."}), 409

    threading.Thread(target=_switch_worker, args=(entry, quantization, nodes), daemon=True).start()
    return jsonify({"status": "started"})


def _push_active_model_to_registry():
    """Broadcasts the just-written local active_model.json to the
    registry, so nodes on OTHER machines actually find out about the
    switch — not just ones sharing this disk with the dashboard.
    Best-effort: swallows errors, same as every other registry call here."""
    cfg = read_active()
    if not cfg:
        return
    try:
        http.post(f"{REGISTRY_URL}/active_model", json=cfg, timeout=5)
    except Exception:
        pass


def _switch_tinygpt_worker(nodes):
    _set_switch_state(status="working", detail="Switching to TinyGPT...", model_id="tinygpt")
    try:
        if nodes:
            total = MODEL_CONFIG.n_layer
            if len(nodes) > total:
                _set_switch_state(
                    status="error",
                    detail=f"You have {len(nodes)} node(s) online, but TinyGPT only has "
                           f"{total} layers — at most {total} node(s) can be used for it. "
                           f"Bring fewer nodes online, or switch to a bigger model instead."
                )
                return
            assignments = split_layers_by_capacity(nodes, total)
            if not push_assignments(REGISTRY_URL, assignments):
                _set_switch_state(status="error",
                                   detail="Could not reach the registry to push the layer split.")
                return
        write_tinygpt()
        _push_active_model_to_registry()
        _set_switch_state(status="done",
                           detail="Switched to TinyGPT. Every node — this machine or "
                                  "any other — will pick this up within a few seconds.")
    except Exception as e:
        _set_switch_state(status="error", detail=f"Failed to switch: {e}")


def _switch_worker(entry, quantization, nodes):
    _set_switch_state(status="working", detail="Preparing download...", model_id=entry.id)
    try:
        cache_dir = str(Path("models/cache") / entry.id)

        if not nodes:
            _set_switch_state(detail="No nodes online — downloading locally (this may take a while)...")
            ok = _local_download(entry)
        else:
            ok = _distributed_download(entry, nodes, cache_dir)

        if not ok:
            # _local_download/_distributed_download already set a specific
            # detail describing exactly what failed — keep it rather than
            # replace it with a generic message that could be misleading
            # (e.g. "check node terminals" when no node was even involved).
            with _switch_lock:
                reason = _switch_state["detail"]
            _set_switch_state(status="error", detail=reason or "Download failed.")
            return

        write_hf(entry.id, entry.hf_id, cache_dir, entry.n_layers,
                 arch=entry.arch, quantization=quantization)
        _push_active_model_to_registry()
        _set_switch_state(
            status="done",
            detail=f"Switched to {entry.name} ({quantization}). "
                   f"Every node — this machine or any other — will pick this up "
                   f"within a few seconds.")
    except Exception as e:
        _set_switch_state(status="error", detail=f"Unexpected error: {e}")


def _distributed_download(entry, nodes, cache_dir) -> bool:
    if too_many_nodes_for_model(nodes, entry):
        _set_switch_state(detail=f"You have {len(nodes)} node(s) online, but {entry.name} only "
                                  f"has {entry.n_layers} layers — at most {entry.n_layers} node(s) "
                                  f"can be used for it. Pick a model with more layers, or confirm "
                                  f"a smaller node list.")
        return False

    per_layer_gb = entry.total_ram_gb / entry.n_layers if entry.n_layers else None
    assignments = split_layers_by_capacity(nodes, entry.n_layers, per_layer_gb=per_layer_gb)
    if not push_assignments(REGISTRY_URL, assignments):
        _set_switch_state(detail="Could not reach the registry to push the layer split.")
        return False

    payload = {"model_id": entry.id, "hf_id": entry.hf_id,
               "cache_dir": cache_dir, "arch": entry.arch}

    started = []
    for node in nodes:
        assignment = assignments.get(node["label"])
        if assignment is None:
            continue
        ok, _msg = start_node_download(node, payload, assignment)
        if ok:
            started.append(node)

    if not started:
        _set_switch_state(detail="Could not reach any node's management port — downloading locally...")
        return _local_download(entry)

    _set_switch_state(detail=f"Downloading ~{entry.download_gb}GB across {len(started)} node(s)...")

    _DOWNLOAD_TIMEOUT_S = 45 * 60
    t_start = time.time()
    while True:
        time.sleep(2)
        all_done, any_error, parts = True, False, []
        for node in started:
            s   = poll_node_download(node)
            st  = s.get("status", "unknown")
            lbl = node["label"]
            progress = s.get("progress", "")
            if st in ("error", "unreachable"):
                parts.append(f"{lbl}: {'unreachable' if st == 'unreachable' else 'error'} — "
                              f"{s.get('error', 'unknown error')}")
                any_error = True
            else:
                parts.append(f"{lbl}: {st}" + (f" ({progress})" if progress else ""))
            if st != "done":
                all_done = False
        _set_switch_state(detail="  |  ".join(parts))
        if any_error:
            return False
        if all_done:
            return True
        if time.time() - t_start > _DOWNLOAD_TIMEOUT_S:
            _set_switch_state(detail="Timed out waiting for nodes to finish downloading "
                                      "(45 min) — " + "  |  ".join(parts))
            return False


def _local_download(entry) -> bool:
    cache_dir = Path("models/cache") / entry.id
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        AutoTokenizer.from_pretrained(entry.hf_id, cache_dir=str(cache_dir))
        AutoModelForCausalLM.from_pretrained(
            entry.hf_id, cache_dir=str(cache_dir),
            torch_dtype="auto", low_cpu_mem_usage=True,
        )
        return True
    except Exception as e:
        _set_switch_state(detail=f"Local download failed: {e}")
        return False


# ── Generation, with live per-token pipeline events ───────────────────────────

_gen_lock      = threading.Lock()
_gen_sessions  = {}          # session_id -> state dict
_SESSION_TTL_S = 1800        # drop sessions nobody has polled in 30 minutes


@app.post("/api/generate")
def api_generate():
    body       = request.get_json(force=True, silent=True) or {}
    prompt     = (body.get("prompt") or "").strip()
    max_tokens = body.get("max_tokens", 40)

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        return jsonify({"error": "max_tokens must be an integer"}), 400
    max_tokens = max(1, min(max_tokens, 200))  # keep a browser session bounded

    session_id = uuid.uuid4().hex
    with _gen_lock:
        _gen_sessions[session_id] = {
            "status": "running", "text": prompt, "steps": [],
            "error": None, "last_seen": time.time(),
        }
    threading.Thread(target=_generate_worker, args=(session_id, prompt, max_tokens),
                      daemon=True).start()
    return jsonify({"session_id": session_id})


@app.get("/api/generate/<session_id>")
def api_generate_poll(session_id):
    with _gen_lock:
        s = _gen_sessions.get(session_id)
        if s is None:
            return jsonify({"error": "unknown or expired session"}), 404
        s["last_seen"] = time.time()
        return jsonify({
            "status": s["status"],
            "text":   s["text"],
            "steps":  list(s["steps"]),
            "error":  s["error"],
        })


def _generate_worker(session_id, prompt, max_tokens):
    def on_token(step, new_part, node_timings, cumulative_text):
        with _gen_lock:
            s = _gen_sessions.get(session_id)
            if s is None:
                return
            s["steps"].append({"index": step, "token_text": new_part, "nodes": node_timings})
            s["text"] = cumulative_text

    try:
        result = generate(prompt, REGISTRY_URL, max_new_tokens=max_tokens,
                           stream=False, use_cache=True, on_token=on_token)
        with _gen_lock:
            s = _gen_sessions.get(session_id)
            if s is not None:
                s["status"] = "done"
                s["text"]   = result["text"]
    except Exception as e:
        with _gen_lock:
            s = _gen_sessions.get(session_id)
            if s is not None:
                s["status"] = "error"
                s["error"]  = str(e)


def _session_sweeper():
    while True:
        time.sleep(300)
        now = time.time()
        with _gen_lock:
            stale = [sid for sid, s in _gen_sessions.items()
                     if now - s["last_seen"] > _SESSION_TTL_S]
            for sid in stale:
                del _gen_sessions[sid]


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global REGISTRY_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--registry", default=REGISTRY_URL)
    args = parser.parse_args()
    REGISTRY_URL = args.registry

    threading.Thread(target=_session_sweeper, daemon=True).start()

    print(f"\n  Dashboard running at http://127.0.0.1:{args.port}")
    print(f"  Talking to registry at {REGISTRY_URL}")
    print(f"  Reachable by other devices on your LAN at your machine's IP, same as the registry.\n")

    # Flask's app.run() is built on werkzeug, which catches a bind OSError
    # internally and calls sys.exit(1) itself — a try/except around
    # app.run() never actually sees it. Test the port ourselves first,
    # where we control the message.
    try:
        _probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _probe.bind(("0.0.0.0", args.port))
        _probe.close()
    except OSError as e:
        print(f"\n  {describe_bind_error(e, args.port)}\n")
        sys.exit(1)

    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
