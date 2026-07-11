"""
Model manager — benchmark your network, get recommendations,
download models across all nodes simultaneously, switch active model.

Usage:
    python download_model.py
    python download_model.py --list
    python download_model.py --switch tinygpt
    python download_model.py --benchmark
"""

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests as http

from models.active     import write_hf, write_tinygpt, read_active, active_model_id
from models.catalogue  import CATALOGUE, get_model
from models.benchmark  import aggregate_score, print_score_report
from models.recommender import recommend, print_recommendations

MODELS_CACHE_DIR = Path("models/cache")
DEFAULT_REGISTRY = os.environ.get("REGISTRY_URL", "http://127.0.0.1:8000")


# ── Registry helpers ──────────────────────────────────────────────────────────

def get_nodes_from_registry(registry_url: str) -> list[dict]:
    try:
        r = http.get(f"{registry_url}/status", timeout=5)
        if r.status_code == 200:
            return [n for n in r.json()["nodes"] if n["alive"]]
    except Exception: pass
    return []


def get_network_specs(registry_url: str) -> list[dict]:
    try:
        r = http.get(f"{registry_url}/network_specs", timeout=5)
        if r.status_code == 200 and r.json()["n_nodes"] > 0:
            return r.json()["node_specs"]
    except Exception: pass
    from models.benchmark import benchmark_node_local
    print("  Registry not reachable — benchmarking this machine only.")
    return [benchmark_node_local()]


def network_summary(specs: list[dict]) -> str:
    total_ram = sum(s.get("ram_free_gb", 0) for s in specs)
    has_gpu   = any(s.get("has_gpu", False) for s in specs)
    return (f"Network: {len(specs)} node(s) | "
            f"{total_ram:.1f}GB total free RAM | "
            f"GPU: {'yes' if has_gpu else 'no'}")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(registry_url: str):
    print("\n  Benchmarking network nodes...")
    nodes = get_nodes_from_registry(registry_url)

    if not nodes:
        print("  No nodes online — benchmarking this machine only.")
        from models.benchmark import benchmark_node_local
        results = [benchmark_node_local()]
    else:
        results = []
        for node in nodes:
            mgmt_port = node["port"] + 10000
            try:
                r = http.get(f"http://{node['host']}:{mgmt_port}/benchmark", timeout=30)
                if r.status_code == 200:
                    bm = r.json()
                    bm["label"] = node["label"]
                    results.append(bm)
                    print(f"  {node['label']}: {bm.get('tokens_per_sec', 0):.0f} tok/s | "
                          f"{bm.get('ram_free_gb', 0):.1f}GB RAM free")
                else:
                    print(f"  {node['label']}: benchmark failed ({r.text})")
            except Exception as e:
                print(f"  {node['label']}: could not reach management port {mgmt_port} ({e})")

    agg = aggregate_score(results)
    print_score_report(agg)
    return agg, results


# ── Distributed download ──────────────────────────────────────────────────────

def trigger_distributed_download(entry, registry_url: str) -> bool:
    """
    Tells every online node to download the model simultaneously,
    then polls until all are done or any fails.
    """
    nodes = get_nodes_from_registry(registry_url)
    cache_dir = str(MODELS_CACHE_DIR / entry.id)

    payload = {
        "model_id":  entry.id,
        "hf_id":     entry.hf_id,
        "cache_dir": cache_dir,
        "arch":      entry.arch,
    }

    if not nodes:
        # No remote nodes — download locally
        print("\n  No remote nodes online. Downloading locally...")
        return _local_download(entry)

    print(f"\n  Triggering download on {len(nodes)} node(s) simultaneously...")
    started = []
    for node in nodes:
        mgmt_port = node["port"] + 10000
        try:
            r = http.post(
                f"http://{node['host']}:{mgmt_port}/download",
                json=payload, timeout=10
            )
            if r.status_code == 200:
                print(f"  ✓ {node['label']} — download started")
                started.append(node)
            else:
                print(f"  ✗ {node['label']} — {r.json().get('error', r.text)}")
        except Exception as e:
            print(f"  ✗ {node['label']} — could not reach management port ({e})")

    if not started:
        print("  Could not start download on any node. Downloading locally as fallback...")
        return _local_download(entry)

    # Poll until all started nodes are done — only print when status changes
    print(f"\n  Downloading ~{entry.download_gb}GB across all nodes...")
    print("  (Each node downloads independently — total wall-clock time ≈ single node)\n")

    last_status = {}   # label → last status string printed
    t_start = time.time()

    while True:
        time.sleep(2)
        all_done  = True
        any_error = False

        for node in started:
            mgmt_port = node["port"] + 10000
            lbl = node["label"]
            try:
                r = http.get(f"http://{node['host']}:{mgmt_port}/download_status", timeout=5)
                s        = r.json()
                status   = s.get("status", "unknown")
                progress = s.get("progress", "")
                line     = f"{status} — {progress}" if progress else status

                if status not in ("done",):
                    all_done = False
                if status == "error":
                    any_error = True
                    if last_status.get(lbl) != line:
                        print(f"  ✗ {lbl}: {s.get('error', 'unknown error')}")
                        last_status[lbl] = line
                else:
                    if last_status.get(lbl) != line:
                        elapsed = int(time.time() - t_start)
                        print(f"  {lbl}: {line}  ({elapsed}s)")
                        last_status[lbl] = line
            except Exception:
                all_done = False

        if any_error:
            print("\n  One or more nodes failed. Check those node terminals for details.")
            return False
        if all_done:
            elapsed = int(time.time() - t_start)
            print(f"\n  ✓ All {len(started)} node(s) finished downloading {entry.name} ({elapsed}s total).")
            return True


def _local_download(entry) -> bool:
    cache_dir = MODELS_CACHE_DIR / entry.id
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import warnings
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        print(f"  Downloading tokenizer...")
        AutoTokenizer.from_pretrained(entry.hf_id, cache_dir=str(cache_dir))
        print(f"  Downloading model weights (~{entry.download_gb}GB)...")
        AutoModelForCausalLM.from_pretrained(
            entry.hf_id, cache_dir=str(cache_dir),
            torch_dtype="auto", low_cpu_mem_usage=True
        )
        print(f"\n  Download complete.")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False


def switch_to_hf(entry):
    cache_dir = MODELS_CACHE_DIR / entry.id
    write_hf(entry.id, entry.hf_id, str(cache_dir), entry.n_layers, arch=entry.arch)
    print(f"\n  Switched to {entry.name}.")
    print(f"  Running nodes will hot-reload within a few seconds.")
    print(f"  IMPORTANT: Each node's --start/--end must cover the right layers")
    print(f"  for a {entry.n_layers}-layer model.")
    print(f"  Example with 2 nodes: node0 --start 0 --end {entry.n_layers//2}  "
          f"node1 --start {entry.n_layers//2} --end {entry.n_layers}")


def switch_to_tinygpt():
    write_tinygpt()
    print("\n  Switched to TinyGPT. Nodes will hot-reload within a few seconds.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry",  default=DEFAULT_REGISTRY)
    parser.add_argument("--list",      action="store_true", help="show catalogue and exit")
    parser.add_argument("--benchmark", action="store_true", help="run benchmark and exit")
    parser.add_argument("--switch",    metavar="MODEL_ID",
                        help="switch to an already-downloaded model (or 'tinygpt')")
    args = parser.parse_args()

    print("\n=== Distributed LLM — Model Manager ===")
    print(f"Registry: {args.registry}")
    print(f"Currently active: {active_model_id()}")

    if args.switch:
        if args.switch == "tinygpt":
            switch_to_tinygpt()
        else:
            entry = get_model(args.switch)
            if not entry:
                print(f"Unknown model '{args.switch}'. "
                      f"Available: {', '.join(m.id for m in CATALOGUE)}")
                sys.exit(1)
            switch_to_hf(entry)
        return

    if args.benchmark:
        run_benchmark(args.registry)
        return

    # ── Full interactive flow ─────────────────────────────────────────────
    # 1. Benchmark
    print()
    do_bench = input("  Run network benchmark first? [Y/n]: ").strip().upper()
    if do_bench in ("", "Y"):
        agg, bench_results = run_benchmark(args.registry)
        score = agg["score"]
    else:
        score = 0
        bench_results = get_network_specs(args.registry)

    # 2. Recommend
    specs   = get_network_specs(args.registry)
    summary = network_summary(specs)
    results = recommend(specs)

    print(f"\n{summary}")
    if score:
        print(f"Network score: {score:.0f} — models marked ★ are well-suited")
    print_recommendations(results)

    if args.list:
        return

    if not results:
        print("No compatible models for your current hardware.\n")
        return

    print("Options:")
    print("  Enter a number to download + switch to that model")
    print("  'benchmark' to re-run the benchmark")
    print("  'tinygpt'   to switch back to the trained built-in model")
    print("  Enter to cancel\n")

    choice = input("  Your choice: ").strip().lower()
    if not choice: return

    if choice == "tinygpt":
        switch_to_tinygpt()
        return
    if choice == "benchmark":
        run_benchmark(args.registry)
        return

    try:
        idx = int(choice) - 1
        assert 0 <= idx < len(results)
    except (ValueError, AssertionError):
        print(f"Invalid choice '{choice}'.")
        return

    entry, _ = results[idx]
    cache_dir = MODELS_CACHE_DIR / entry.id
    already   = cache_dir.exists() and any(cache_dir.iterdir())

    if already:
        print(f"\n  {entry.name} is already downloaded.")
        if input("  Switch to it now? [Y/n]: ").strip().upper() in ("", "Y"):
            switch_to_hf(entry)
        return

    print(f"\n  {entry.name} — {entry.description}")
    print(f"  Download: ~{entry.download_gb}GB  |  Layers: {entry.n_layers}  |  "
          f"Arch: {entry.arch}")
    if input("\n  Download across all nodes and switch? [Y/n]: ").strip().upper() not in ("", "Y"):
        return

    if trigger_distributed_download(entry, args.registry):
        switch_to_hf(entry)


if __name__ == "__main__":
    main()
