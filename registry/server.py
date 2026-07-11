"""
Registry server — the network phonebook.
Nodes announce themselves here with their layer range and hardware specs.
Coordinator queries /chain to get the current live ordered node list.
download_model.py queries /network_specs to get aggregated hardware info
for the model recommender.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request as freq

from config import MODEL_CONFIG

app   = Flask(__name__)
_nodes: dict = {}
_lock = threading.Lock()

HEARTBEAT_TIMEOUT  = 12
HEARTBEAT_INTERVAL = 5


def _alive() -> list[dict]:
    now = time.time()
    with _lock:
        nodes = [n for n in _nodes.values()
                 if now - n["last_heartbeat"] <= HEARTBEAT_TIMEOUT]
    return sorted(nodes, key=lambda n: n["start_layer"])


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
    return None


@app.post("/announce")
def announce():
    data = freq.get_json(silent=True)
    if not data:
        return jsonify({"error": "expected JSON body"}), 400
    for k in ("host", "port", "start_layer", "end_layer", "label"):
        if k not in data:
            return jsonify({"error": f"missing field: {k}"}), 400

    key = (data["host"], int(data["port"]))
    node = {
        "host":            data["host"],
        "port":            int(data["port"]),
        "start_layer":     int(data["start_layer"]),
        "end_layer":       int(data["end_layer"]),
        "label":           str(data["label"]),
        "hw_specs":        data.get("hw_specs", {}),
        "last_heartbeat":  time.time(),
        "announced_at":    time.time(),
        "requests_served": 0,
    }
    with _lock:
        is_new = key not in _nodes
        _nodes[key] = node

    status = "registered" if is_new else "re-registered"
    print(f"[registry] {status}: {node['label']} @ "
          f"{node['host']}:{node['port']} layers "
          f"{node['start_layer']}-{node['end_layer']-1}")
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
    nodes = _alive()
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
    now   = time.time()
    alive = _alive()
    err   = _validate(alive)
    with _lock:
        all_nodes = list(_nodes.values())
    return jsonify({
        "chain_ready": err is None,
        "chain_error": err,
        "total_nodes": len(all_nodes),
        "alive_nodes": len(alive),
        "nodes": [{
            "host":             n["host"],
            "port":             n["port"],
            "start_layer":      n["start_layer"],
            "end_layer":        n["end_layer"],
            "label":            n["label"],
            "alive":            now - n["last_heartbeat"] <= HEARTBEAT_TIMEOUT,
            "last_seen_ago_s":  round(now - n["last_heartbeat"], 1),
            "requests_served":  n["requests_served"],
            "hw_specs":         n.get("hw_specs", {}),
        } for n in sorted(all_nodes, key=lambda x: x["start_layer"])],
        "model": {
            "n_layer": MODEL_CONFIG.n_layer,
            "n_embd":  MODEL_CONFIG.n_embd,
            "n_head":  MODEL_CONFIG.n_head,
        },
    })


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
    srv = make_server(args.host, args.port, app, threaded=True)
    print(f"[registry] Ready — listening on {args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[registry] Shutting down.")


if __name__ == "__main__":
    main()
