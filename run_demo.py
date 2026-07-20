"""
One-command demo. Starts a registry and every node defined in
config.py as subprocesses (all on this machine), waits for them to
register a complete chain, sends a prompt through the whole pipeline
via coordinator/client.py, prints the result, then shuts everything
down cleanly.

Usage:
    python create_checkpoint.py        # run once
    python run_demo.py "Hello there"   # run the demo
"""

import os
import socket
import subprocess
import sys
import time

import requests as http

from config import NODES, LAYER_SPLITS, CHECKPOINT_PATH, REGISTRY_HOST, REGISTRY_PORT, REGISTRY_URL
from coordinator.client import generate
from models.active import write_tinygpt


def wait_for_port(host, port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def wait_for_chain(registry_url, n_nodes, timeout=30):
    """Waits until the registry reports a complete, valid chain with all
    n_nodes alive — not just that each node's port happens to be open."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = http.get(f"{registry_url}/status", timeout=3)
            if r.status_code == 200:
                st = r.json()
                if st.get("chain_ready") and st.get("alive_nodes", 0) >= n_nodes:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: {CHECKPOINT_PATH} not found. Run `python create_checkpoint.py` first.")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The distributed network says: "

    assert len(NODES) == len(LAYER_SPLITS), "NODES and LAYER_SPLITS must be the same length"

    # This demo's topology (NODES/LAYER_SPLITS in config.py) is fixed to
    # TinyGPT's layer count. If a previous `python run.py` session used
    # the MODELS menu to switch to a real HF model, active_model.json
    # would still point at that model — every node here would then load
    # the wrong model/layer-count combination. Force TinyGPT so this demo
    # always runs against the topology it was actually written for.
    write_tinygpt()

    node_procs = []
    registry_proc = None
    try:
        print(f"Launching registry on {REGISTRY_HOST}:{REGISTRY_PORT}...")
        registry_proc = subprocess.Popen([
            sys.executable, "registry/server.py",
            "--host", REGISTRY_HOST, "--port", str(REGISTRY_PORT),
        ])
        if not wait_for_port(REGISTRY_HOST, REGISTRY_PORT):
            raise RuntimeError("Registry did not start in time.")
        print("Registry online.\n")

        for i, ((host, port), (start, end)) in enumerate(zip(NODES, LAYER_SPLITS)):
            cmd = [
                sys.executable, "node/server.py",
                "--host", host,
                "--port", str(port),
                "--start", str(start),
                "--end", str(end),
                "--label", f"node{i}",
                "--registry", REGISTRY_URL,
            ]
            print(f"Launching node {i}: layers {start}-{end - 1}, listening on {host}:{port}")
            node_procs.append(subprocess.Popen(cmd))

        print("\nWaiting for all nodes to come online and register with the registry...")
        for host, port in NODES:
            if not wait_for_port(host, port):
                raise RuntimeError(f"Node at {host}:{port} did not start in time. "
                                    f"Check the [nodeN] output above for errors.")
        if not wait_for_chain(REGISTRY_URL, len(NODES)):
            raise RuntimeError("Nodes came up but never formed a complete chain. "
                                "Check the node output above, or the registry's "
                                f"{REGISTRY_URL}/status for details.")
        print("All nodes online and chain is ready.\n")

        result = generate(prompt, REGISTRY_URL, stream=False)
        print("\n=== RESULT ===")
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {result['text']!r}")
        print("==============")
        print("\n(If the output is random noise, run `python train.py` first to teach the "
              "model some real text — then re-run this demo. Either way, what matters here "
              "is that the request was tokenized, embedded on node 0, passed through node "
              "0's layers, forwarded over a real TCP socket to node 1, run through node 1's "
              "layers, turned into logits, and relayed all the way back — i.e. the "
              "distributed pipeline worked end to end.)")

    finally:
        print("\nShutting down...")
        all_procs = node_procs + ([registry_proc] if registry_proc else [])
        for proc in all_procs:
            proc.terminate()
        for proc in all_procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("Done.")


if __name__ == "__main__":
    main()
