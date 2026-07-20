"""
Registry server — the network phonebook.

Nodes announce themselves here with their layer range and hardware specs.
A node started without --start/--end joins as an unassigned "pool" node
(start_layer/end_layer are None) — it's alive and benchmarked, but not
part of the inference chain until it's given a layer range.

Coordinator queries /chain to get the current live ordered node list
(pool nodes are excluded — they're not assigned to a layer range yet).

download_model.py / run.py query /network_specs (chain nodes) or /pool
(unassigned nodes) to get aggregated hardware info for the model
recommender, then push a decision back with:
  - POST /active_model — "this is the model every node should be running"
  - POST /assignments  — "here is each node's (by label) layer range"
Nodes poll both of these on every hot-reload tick, so a change reaches
every node on every machine, not just ones on the same disk as whoever
made the change.
"""

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request as freq

from config import MODEL_CONFIG
from shared.netinfo import describe_bind_error

app   = Flask(__name__)
_nodes: dict = {}
_assignments: dict = {}     # label -> {start_layer, end_layer, total_layers, updated_at}
_active_model: dict | None = None
_lock = threading.Lock()

HEARTBEAT_TIMEOUT  = 12
HEARTBEAT_INTERVAL = 5


def _alive() -> list[dict]:
    now = time.time()
    with _lock:
        nodes = [n for n in _nodes.values()
                 if now - n["last_heartbeat"] <= HEARTBEAT_TIMEOUT]
    # Pool nodes have start_layer=None, which can't be compared to an int —
    # sort them after every assigned node instead of crashing.
    return sorted(nodes, key=lambda n: (n["start_layer"] is None, n["start_layer"] or 0))


def _active_total_layers() -> int:
    """Total layer count of whatever model is currently active, so
    _validate can check a chain covers ALL of it — not just that
    layers 0..N have no gaps/overlaps. Falls back to the built-in
    TinyGPT's layer count when no HF model is active (or its layer
    count hasn't been recorded)."""
    with _lock:
        am = _active_model
    if am and am.get("type") == "hf" and am.get("n_layers"):
        return int(am["n_layers"])
    return MODEL_CONFIG.n_layer


def _validate(nodes: list[dict]) -> str | None:
    if not nodes:
        return "No nodes are currently online."
    expected = 0
    for n in nodes:
        if n["start_layer"] < expected:
            return (f"Layer overlap: {n['label']} starts at layer {n['start_layer']} "
                    f"but layers 0–{expected-1} are already covered.")
        if n["start_layer"] != expected:
            return (f"Layer gap: expected a node at layer {expected}, "
                    f"but {n['label']} starts at {n['start_layer']}.")
        expected = n["end_layer"]
    total = _active_total_layers()
    if expected != total:
        return (f"Chain only covers layers 0–{expected-1}, but the active model "
                f"needs layers 0–{total-1} ({total} layers total). Add a node "
                f"covering layers {expected}–{total-1}.")
    return None


@app.post("/announce")
def announce():
    data = freq.get_json(silent=True)
    if not data:
        return jsonify({"error": "expected JSON body"}), 400
    for k in ("host", "port", "label"):
        if k not in data:
            return jsonify({"error": f"missing field: {k}"}), 400

    key   = (data["host"], int(data["port"]))
    label = str(data["label"])
    start_layer = data.get("start_layer")
    end_layer   = data.get("end_layer")
    node = {
        "host":            data["host"],
        "port":            int(data["port"]),
        "start_layer":     int(start_layer) if start_layer is not None else None,
        "end_layer":       int(end_layer) if end_layer is not None else None,
        "label":           label,
        "hw_specs":        data.get("hw_specs", {}),
        "last_heartbeat":  time.time(),
        "announced_at":    time.time(),
        "requests_served": 0,
    }
    with _lock:
        is_new = key not in _nodes
        # A node restarted with a different --host/--port (e.g. fixing a
        # bad advertised IP) re-announces under a new key, which would
        # otherwise leave its old entry behind forever, marked "dead" in
        # /status. Since labels are how a node identifies itself across
        # restarts, treat a re-announce under the same label as replacing
        # any previous entry for that label rather than adding a ghost.
        stale_keys = [k for k, n in _nodes.items()
                      if n["label"] == label and k != key]
        for k in stale_keys:
            del _nodes[k]
        _nodes[key] = node

    status = "registered" if is_new else "re-registered"
    layers = (f"layers {node['start_layer']}-{node['end_layer']-1}"
              if node["start_layer"] is not None else "unassigned (pool)")
    print(f"[registry] {status}: {node['label']} @ "
          f"{node['host']}:{node['port']} {layers}")
    return jsonify({"status": status, "heartbeat_interval": HEARTBEAT_INTERVAL})


@app.post("/heartbeat")
def heartbeat():
    data = freq.get_json(silent=True)
    if not data:
        return jsonify({"error": "expected JSON body"}), 400
    key = (data.get("host"), int(data.get("port", 0)))
    with _lock:
        if key not in _nodes:
            return jsonify({"error": "unknown node — send /announce first"}), 404
        _nodes[key]["last_heartbeat"] = time.time()
        if "requests_served" in data:
            _nodes[key]["requests_served"] = int(data["requests_served"])
    return jsonify({"status": "ok"})


@app.get("/chain")
def chain():
    nodes = [n for n in _alive() if n["start_layer"] is not None]
    err   = _validate(nodes)
    if err:
        return jsonify({"error": err, "nodes": []}), 503
    return jsonify({"nodes": [{
        "host":        n["host"],
        "port":        n["port"],
        "start_layer": n["start_layer"],
        "end_layer":   n["end_layer"],
        "label":       n["label"],
    } for n in nodes]})


@app.get("/status")
def status():
    now    = time.time()
    alive  = _alive()
    err    = _validate([n for n in alive if n["start_layer"] is not None])
    with _lock:
        all_nodes = list(_nodes.values())
    return jsonify({
        "chain_ready": err is None,
        "chain_error": err,
        "total_nodes": len(all_nodes),
        "alive_nodes": len([n for n in alive if n["start_layer"] is not None]),
        "pool_nodes":  len([n for n in alive if n["start_layer"] is None]),
        "nodes": [{
            "host":             n["host"],
            "port":             n["port"],
            "start_layer":      n["start_layer"],
            "end_layer":        n["end_layer"],
            "label":            n["label"],
            "assigned":         n["start_layer"] is not None,
            "alive":            now - n["last_heartbeat"] <= HEARTBEAT_TIMEOUT,
            "last_seen_ago_s":  round(now - n["last_heartbeat"], 1),
            "requests_served":  n["requests_served"],
            "hw_specs":         n.get("hw_specs", {}),
        } for n in sorted(all_nodes, key=lambda x: (x["start_layer"] is None, x["start_layer"] or 0))],
        "model": {
            "n_layer": MODEL_CONFIG.n_layer,
            "n_embd":  MODEL_CONFIG.n_embd,
            "n_head":  MODEL_CONFIG.n_head,
        },
    })


@app.get("/pool")
def pool():
    """Unassigned nodes waiting for a layer range + model — what run.py's
    'add nodes, then pick a model' flow benchmarks against."""
    nodes = [n for n in _alive() if n["start_layer"] is None]
    return jsonify({
        "n_nodes": len(nodes),
        "nodes": [{"label": n["label"], "host": n["host"], "port": n["port"],
                    "hw_specs": n.get("hw_specs", {})} for n in nodes],
    })


@app.get("/assignments")
def get_assignments():
    with _lock:
        return jsonify({"assignments": dict(_assignments)})


@app.post("/assignments")
def set_assignments():
    """Bulk-sets layer assignments by label, e.g. after picking a model
    for a pool of nodes: {"assignments": {"node0": {"start_layer": 0,
    "end_layer": 6, "total_layers": 12}, ...}}. Nodes pick these up
    (filtered to their own label) on their next hot-reload poll."""
    data = freq.get_json(silent=True)
    if not data or not isinstance(data.get("assignments"), dict):
        return jsonify({"error": "expected JSON body: "
                         "{\"assignments\": {label: {start_layer, end_layer, total_layers}}}"}), 400
    now = time.time()
    cleaned = {}
    for label, a in data["assignments"].items():
        try:
            cleaned[str(label)] = {
                "start_layer":  int(a["start_layer"]),
                "end_layer":    int(a["end_layer"]),
                "total_layers": int(a["total_layers"]),
                "updated_at":   now,
            }
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": f"invalid assignment for label '{label}' — need "
                             "integer start_layer, end_layer, total_layers"}), 400
    with _lock:
        _assignments.update(cleaned)
    print(f"[registry] layer assignments set for: {', '.join(cleaned)}")
    return jsonify({"status": "ok", "assignments": cleaned})


@app.get("/active_model")
def get_active_model():
    with _lock:
        return jsonify({"active": _active_model})


@app.post("/active_model")
def set_active_model():
    """Sets the network-wide active model. Every node polls this on each
    hot-reload tick and switches to match, regardless of which machine
    made this call — this is what makes model switching actually reach
    nodes on other machines instead of only ones on the same disk."""
    global _active_model
    data = freq.get_json(silent=True)
    if not data:
        return jsonify({"error": "expected JSON body"}), 400
    data = dict(data)
    data.setdefault("updated_at", time.time())
    with _lock:
        _active_model = data
    print(f"[registry] active model set to: {data.get('model_id', '?')}")
    return jsonify({"status": "ok", "active": data})


@app.get("/network_specs")
def network_specs():
    """
    Returns aggregated hardware specs across all alive nodes.
    Used by the model recommender in download_model.py.
    """
    nodes = _alive()
    specs = [n.get("hw_specs", {}) for n in nodes]
    return jsonify({
        "n_nodes":    len(nodes),
        "node_specs": specs,
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def _reaper():
    while True:
        time.sleep(HEARTBEAT_TIMEOUT)
        now = time.time()
        with _lock:
            for n in _nodes.values():
                age = now - n["last_heartbeat"]
                if HEARTBEAT_TIMEOUT < age <= HEARTBEAT_TIMEOUT * 2:
                    print(f"[registry] ⚠  {n['label']} has gone silent "
                          f"({age:.0f}s since last heartbeat)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    threading.Thread(target=_reaper, daemon=True).start()

    print(f"[registry] Starting on {args.host}:{args.port}")
    print(f"[registry] Heartbeat timeout: {HEARTBEAT_TIMEOUT}s")

    from werkzeug.serving import make_server
    # Werkzeug's own server catches a bind OSError internally and calls
    # sys.exit(1) itself before it would ever reach a try/except wrapped
    # around make_server() — so test the port ourselves first, where we
    # control the message.
    try:
        _probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _probe.bind((args.host, args.port))
        _probe.close()
    except OSError as e:
        print(f"\n[registry] {describe_bind_error(e, args.port)}\n")
        sys.exit(1)

    srv = make_server(args.host, args.port, app, threaded=True)
    print(f"[registry] Ready — listening on {args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[registry] Shutting down.")


if __name__ == "__main__":
    main()
