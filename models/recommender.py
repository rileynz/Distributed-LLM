"""
Benchmarks the network's available compute and recommends models
from the catalogue that will actually fit and run well.
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import torch

from models.catalogue import CATALOGUE, ModelEntry


def benchmark_local() -> dict:
    """
    Measures this machine's hardware. Called by nodes at startup
    and included in their /announce payload so the registry can
    aggregate specs across the whole network.
    """
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(".")

    # CPU speed: time a small matmul — rough but consistent proxy
    t0 = time.perf_counter()
    a = torch.randn(512, 512)
    for _ in range(20):
        a = a @ a.T
    cpu_ms = (time.perf_counter() - t0) * 1000 / 20

    has_gpu = torch.cuda.is_available()
    gpu_ram_gb = 0.0
    if has_gpu:
        props = torch.cuda.get_device_properties(0)
        gpu_ram_gb = props.total_memory / (1024 ** 3)

    return {
        "ram_total_gb":     round(vm.total     / (1024 ** 3), 2),
        "ram_available_gb": round(vm.available  / (1024 ** 3), 2),
        "disk_free_gb":     round(disk.free     / (1024 ** 3), 2),
        "cpu_matmul_ms":    round(cpu_ms, 1),
        "has_gpu":          has_gpu,
        "gpu_ram_gb":       round(gpu_ram_gb, 2),
        "cpu_count":        psutil.cpu_count(logical=False) or 1,
    }


def recommend(node_specs: list[dict]) -> list[tuple[ModelEntry, str]]:
    """
    Given a list of per-node hardware specs (as returned by /status
    and included in each node's announce payload), returns a list of
    (ModelEntry, reason_string) pairs for models that can run on this
    network, ordered best-fit first.

    node_specs: list of dicts, each with keys from benchmark_local().
    """
    if not node_specs:
        return []

    n_nodes          = len(node_specs)
    total_ram_gb     = sum(s.get("ram_free_gb", s.get("ram_available_gb", 0)) for s in node_specs)
    min_ram_per_node = min(s.get("ram_free_gb", s.get("ram_available_gb", 0)) for s in node_specs)
    any_gpu          = any(s.get("has_gpu", False) for s in node_specs)
    total_disk_gb    = min(s.get("disk_free_gb", 0) for s in node_specs)  # bottleneck node

    results = []
    for model in CATALOGUE:
        reasons_ok  = []
        reasons_bad = []

        # Node count
        if n_nodes < model.min_nodes:
            reasons_bad.append(
                f"needs {model.min_nodes} node(s), you have {n_nodes}"
            )
        else:
            reasons_ok.append(f"{n_nodes} node(s) ✓")

        # RAM per node
        if min_ram_per_node < model.min_ram_per_node_gb:
            reasons_bad.append(
                f"needs {model.min_ram_per_node_gb}GB RAM/node, "
                f"weakest node has {min_ram_per_node:.1f}GB free"
            )
        else:
            reasons_ok.append(f"{min_ram_per_node:.1f}GB RAM/node ✓")

        # Total RAM
        if total_ram_gb < model.total_ram_gb:
            reasons_bad.append(
                f"needs {model.total_ram_gb}GB total RAM, "
                f"network has {total_ram_gb:.1f}GB free"
            )
        else:
            reasons_ok.append(f"{total_ram_gb:.1f}GB total RAM ✓")

        # Disk space for download
        if total_disk_gb < model.download_gb * 1.2:  # 20% buffer
            reasons_bad.append(
                f"needs {model.download_gb}GB disk, "
                f"you have {total_disk_gb:.1f}GB free"
            )
        else:
            reasons_ok.append(f"{total_disk_gb:.1f}GB disk ✓")

        if reasons_bad:
            continue  # skip incompatible models

        reason = ", ".join(reasons_ok)
        if any_gpu:
            reason += " (GPU detected — will be faster)"
        results.append((model, reason))

    # Sort: highest quality first, then lightest download as tiebreaker
    results.sort(key=lambda t: (-t[0].quality, t[0].download_gb))
    return results


def print_recommendations(results: list[tuple[ModelEntry, str]],
                           network_summary: str = ""):
    if network_summary:
        print(f"\n{network_summary}")

    if not results:
        print("\n  No models in the catalogue fit your current network.")
        print("  Try adding more nodes or freeing up RAM.")
        return

    print(f"\n  {'#':<3}  {'Model':<20}  {'Size':>6}  {'Quality':^9}  {'Notes'}")
    print("  " + "─" * 72)
    for i, (model, reason) in enumerate(results, 1):
        quality_stars = "★" * model.quality + "☆" * (5 - model.quality)
        print(f"  {i:<3}  {model.name:<20}  "
              f"{model.download_gb:>4.1f}GB  {quality_stars}  {model.description}")
        print(f"       {reason}")
        if model.notes:
            print(f"       ↳ {model.notes}")
        print()
