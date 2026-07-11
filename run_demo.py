"""
One-command demo. Launches every node defined in config.py as a
subprocess, waits for them to be ready, sends a prompt through the
whole chain via coordinator/client.py, prints the result, then shuts
every node down cleanly.

Usage:
    python create_checkpoint.py        # run once
    python run_demo.py "Hello there"   # run the demo
"""

import socket
import subprocess
import sys
import time

from config import NODES, LAYER_SPLITS, MODEL_CONFIG, CHECKPOINT_PATH
from coordinator.client import generate

import os


def wait_for_port(host, port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: {CHECKPOINT_PATH} not found. Run `python create_checkpoint.py` first.")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The distributed network says: "

    assert len(NODES) == len(LAYER_SPLITS), "NODES and LAYER_SPLITS must be the same length"

    processes = []
    try:
        for i, ((host, port), (start, end)) in enumerate(zip(NODES, LAYER_SPLITS)):
            is_last = i == len(NODES) - 1
            cmd = [
                sys.executable, "node/server.py",
                "--host", host,
                "--port", str(port),
                "--start", str(start),
                "--end", str(end),
                "--label", f"node{i}",
            ]
            if not is_last:
                next_host, next_port = NODES[i + 1]
                cmd += ["--next-host", next_host, "--next-port", str(next_port)]

            print(f"Launching node {i}: layers {start}-{end - 1}, listening on {host}:{port}"
                  + ("" if is_last else f", forwards to {NODES[i + 1]}"))
            proc = subprocess.Popen(cmd)
            processes.append(proc)

        print("\nWaiting for all nodes to come online...")
        for host, port in NODES:
            if not wait_for_port(host, port):
                raise RuntimeError(f"Node at {host}:{port} did not start in time. "
                                    f"Check the [nodeN] output above for errors.")
        print("All nodes online.\n")

        output = generate(prompt)
        print("\n=== RESULT ===")
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {output!r}")
        print("==============")
        print("\n(If the output is random noise, run `python train.py` first to teach the "
              "model some real text — then re-run this demo. Either way, what matters here "
              "is that the request was tokenized, embedded on node 0, passed through node "
              "0's layers, forwarded over a real TCP socket to node 1, run through node 1's "
              "layers, turned into logits, and relayed all the way back — i.e. the "
              "distributed pipeline worked end to end.)")

    finally:
        print("\nShutting down nodes...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("Done.")


if __name__ == "__main__":
    main()
