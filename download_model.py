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

from config import MODEL_CONFIG
from models.active     import write_hf, write_tinygpt, read_active, active_model_id
from models.catalogue  import CATALOGUE, get_model
from models.benchmark  import aggregate_score, print_score_report
from models.recommender import recommend, print_recommendations
from shared.provisioning import (
    cli_confirm_nodes, split_layers_by_capacity, too_many_nodes_for_model,
    push_assignments, start_node_download, poll_node_download,
)

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
    Hard-stop gate first (see shared.provisioning.cli_confirm_nodes),
    then splits this model's layers evenly across EVERY confirmed node
    — not just unassigned pool nodes; any node that already had a
    range from a previous model gets re-split too, since that old range
    almost certainly doesn't match this model's layer count. Only once
    that split is pushed to the registry does each node get told to
    download, with its own range attached so it fetches only its shard.
    """
    nodes = cli_confirm_nodes(registry_url)
    if nodes is None:
        print("\n  Cancelled — nothing was downloaded.")
        return False

    cache_dir = str(MODELS_CACHE_DIR / entry.id)

    if not nodes:
        print("\n  No nodes online. Downloading locally...")
        return _local_download(entry)

    if too_many_nodes_for_model(nodes, entry):
        print(f"\n  You have {len(nodes)} node(s) online, but {entry.name} only has "
              f"{entry.n_layers} layers — at most {entry.n_layers} node(s) can be used "
              f"for it (each node needs at least one whole layer). Pick a model with "
              f"more layers, or bring fewer nodes into this split.")
        return False

    per_layer_gb = entry.total_ram_gb / entry.n_layers if entry.n_layers else None
    assignments = split_layers_by_capacity(nodes, entry.n_layers, per_layer_gb=per_layer_gb)

    ram_by_label = {n["label"]: n.get("hw_specs", {}).get("ram_free_gb", 0.0) for n in nodes}
    print(f"\n  Layer plan for {entry.name} ({entry.n_layers} layers) "
          f"across {len(assignments)} node(s) — split proportional to free RAM:")
    for label in sorted(assignments):
        a = assignments[label]
        print(f"    {label}:  layers {a['start_layer']}-{a['end_layer']-1}  "
              f"({ram_by_label.get(label, 0):.1f}GB free RAM)")

    if not push_assignments(registry_url, assignments):
        print("\n  Could not reach the registry to push the layer split. Aborting.")
        return False

    payload = {"model_id": entry.id, "hf_id": entry.hf_id,
               "cache_dir": cache_dir, "arch": entry.arch}

    print(f"\n  Triggering download on {len(assignments)} node(s) simultaneously...")
    started = []
    for node in nodes:
        assignment = assignments.get(node["label"])
        if assignment is None:
            continue
        ok, msg = start_node_download(node, payload, assignment)
        if ok:
            print(f"  ✓ {node['label']} — download started")
            started.append(node)
        else:
            print(f"  ✗ {node['label']} — {msg}")

    if not started:
        print("  Could not start download on any node. Downloading locally as fallback...")
        return _local_download(entry)

    # Poll until all started nodes are done — only print when status changes
    print(f"\n  Downloading ~{entry.download_gb}GB across all nodes...")
    print("  (Each node downloads independently — total wall-clock time ≈ single node)\n")

    last_status = {}   # label → last status string printed
    t_start = time.time()
    _DOWNLOAD_TIMEOUT_S = 45 * 60

    while True:
        time.sleep(2)
        all_done  = True
        any_error = False

        for node in started:
            lbl = node["label"]
            s        = poll_node_download(node)
            status   = s.get("status", "unknown")
            progress = s.get("progress", "")
            line     = f"{status} — {progress}" if progress else status

            if status not in ("done",):
                all_done = False
            if status in ("error", "unreachable"):
                any_error = True
                if last_status.get(lbl) != line:
                    print(f"  ✗ {lbl}: {'unreachable — ' if status == 'unreachable' else ''}"
                          f"{s.get('error', 'unknown error')}")
                    last_status[lbl] = line
            else:
                if last_status.get(lbl) != line:
                    elapsed = int(time.time() - t_start)
                    print(f"  {lbl}: {line}  ({elapsed}s)")
                    last_status[lbl] = line

        if any_error:
            print("\n  One or more nodes failed (or stopped responding). Check those node "
                  "terminals for details.")
            return False
        if all_done:
            elapsed = int(time.time() - t_start)
            print(f"\n  ✓ All {len(started)} node(s) finished downloading {entry.name} ({elapsed}s total).")
            return True
        if time.time() - t_start > _DOWNLOAD_TIMEOUT_S:
            print(f"\n  Timed out after {_DOWNLOAD_TIMEOUT_S // 60} minutes waiting for nodes "
                  f"to finish downloading. Check those node terminals for details.")
            return False


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


def ask_quantization(entry, recommended: str = "none") -> str:
    """Prompts for a quantization level. On a GPU node, both int8/int4
    use bitsandbytes. On a CPU-only node, int8 uses real PyTorch-native
    dynamic quantization instead (no GPU needed); int4 on CPU falls
    back to that same CPU int8 path, since there's no CPU-native 4-bit
    kernel here without a GPU."""
    if not entry.supports_quantization:
        return "none"

    default_num = {"none": "1", "int8": "2", "int4": "3"}[recommended]
    print("\n  Quantization:")
    print("    1) none — full precision")
    print("    2) int8 — ~2x smaller; real on both GPU (bitsandbytes) and CPU "
          "(PyTorch-native, no GPU needed)")
    print("    3) int4 — ~4x smaller, some quality loss; needs a CUDA GPU "
          "(falls back to CPU int8 above if this node has none)")
    if recommended != "none":
        print(f"  Recommended for your network: {recommended} (option {default_num})")
    choice = input(f"  Choice [{default_num}]: ").strip() or default_num
    return {"1": "none", "2": "int8", "3": "int4"}.get(choice, recommended)


def push_active_model_to_registry(registry_url: str):
    """Broadcasts the just-written local active_model.json to the
    registry, so nodes on OTHER machines actually find out about the
    switch too — not just ones that happen to share this disk. This is
    what makes a switch actually reach a remote node: local nodes see it
    via the file directly; every node also polls the registry, which is
    reachable from anywhere. Best-effort: if the registry is unreachable
    right now, remote nodes just keep running whatever they last picked
    up until it becomes reachable again."""
    cfg = read_active()
    if not cfg:
        return
    try:
        http.post(f"{registry_url}/active_model", json=cfg, timeout=5)
    except Exception as e:
        print(f"  (Could not notify the registry — remote nodes may not see this "
              f"switch until it's reachable again: {e})")


def switch_to_hf(entry, registry_url: str, quantization: str = "none",
                  layers_already_assigned: bool = False):
    cache_dir = MODELS_CACHE_DIR / entry.id
    write_hf(entry.id, entry.hf_id, str(cache_dir), entry.n_layers,
             arch=entry.arch, quantization=quantization)
    push_active_model_to_registry(registry_url)
    print(f"\n  Switched to {entry.name} (quantization: {quantization}).")
    print(f"  Every node — on this machine or any other — will pick this up within a few seconds.")
    if layers_already_assigned:
        print(f"  Each node's layer range for this {entry.n_layers}-layer model was already "
              f"confirmed and pushed above.")
    else:
        print(f"  IMPORTANT: this only flips the active model — it does NOT push a new layer")
        print(f"  split. Each node needs a correct range for a {entry.n_layers}-layer model")
        print(f"  already (either restart it manually with the right --start/--end, or run")
        print(f"  this script without --switch once to get the layers auto-split for you).")


def switch_to_tinygpt(registry_url: str) -> bool:
    """Switching to TinyGPT needs the exact same re-provisioning as
    switching to any HF model: nodes online right now almost certainly
    still hold a layer split for whatever model was active before,
    which won't match TinyGPT's layer count. Skipping that step (as
    this used to) left nodes waiting forever for an assignment that
    was never coming — i.e. the switch never actually took effect on
    already-running nodes."""
    nodes = cli_confirm_nodes(registry_url)
    if nodes is None:
        print("\n  Cancelled — nothing was switched.")
        return False

    if nodes:
        total = MODEL_CONFIG.n_layer
        if len(nodes) > total:
            print(f"\n  You have {len(nodes)} node(s) online, but TinyGPT only has "
                  f"{total} layers — at most {total} node(s) can be used for it. "
                  f"Bring fewer nodes online, or switch to a bigger model instead.")
            return False
        assignments = split_layers_by_capacity(nodes, total)
        print(f"\n  Layer plan for TinyGPT ({total} layers) across {len(assignments)} node(s):")
        for label in sorted(assignments):
            a = assignments[label]
            print(f"    {label}:  layers {a['start_layer']}-{a['end_layer']-1}")
        if not push_assignments(registry_url, assignments):
            print("\n  Could not reach the registry to push the layer split. Aborting.")
            return False

    write_tinygpt()
    push_active_model_to_registry(registry_url)
    print("\n  Switched to TinyGPT. Every node — this machine or any other — "
          "will pick this up within a few seconds.")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry",  default=DEFAULT_REGISTRY)
    parser.add_argument("--list",      action="store_true", help="show catalogue and exit")
    parser.add_argument("--benchmark", action="store_true", help="run benchmark and exit")
    parser.add_argument("--switch",    metavar="MODEL_ID",
                        help="switch to an already-downloaded model (or 'tinygpt')")
    parser.add_argument("--quant", choices=["none", "int8", "int4"], default="none",
                        help="quantization to use with --switch (int8 works on CPU "
                             "or GPU; int4 needs a CUDA GPU, falls back to CPU int8)")
    args = parser.parse_args()

    print("\n=== Distributed LLM — Model Manager ===")
    print(f"Registry: {args.registry}")
    print(f"Currently active: {active_model_id()}")

    if args.switch:
        if args.switch == "tinygpt":
            switch_to_tinygpt(args.registry)
        else:
            entry = get_model(args.switch)
            if not entry:
                print(f"Unknown model '{args.switch}'. "
                      f"Available: {', '.join(m.id for m in CATALOGUE)}")
                sys.exit(1)
            switch_to_hf(entry, args.registry, quantization=args.quant)
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
        switch_to_tinygpt(args.registry)
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

    entry, _, quant_needed = results[idx]

    # Always go through the download-trigger step rather than short-
    # circuiting on "the local cache_dir looks non-empty" — since nodes
    # now download only their own layer's shards, cache_dir can be
    # non-empty while still missing shards a *different* node needs.
    # hf_hub_download skips re-fetching anything already cached, so this
    # is a fast no-op wherever nothing is actually missing.
    print(f"\n  {entry.name} — {entry.description}")
    print(f"  Download: ~{entry.download_gb}GB  |  Layers: {entry.n_layers}  |  "
          f"Arch: {entry.arch}")
    if input("\n  Download (only whatever's missing) across all nodes and switch? [Y/n]: "
             ).strip().upper() not in ("", "Y"):
        return

    quant = ask_quantization(entry, recommended=quant_needed)
    if trigger_distributed_download(entry, args.registry):
        switch_to_hf(entry, args.registry, quantization=quant, layers_already_assigned=True)


if __name__ == "__main__":
    main()
