"""
Benchmarks the network's available compute and recommends models
from the catalogue that will actually fit and run well.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.catalogue import CATALOGUE, ModelEntry

# Actual hardware specs come from models.benchmark.benchmark_node_local(),
# which every node runs at startup and includes in its /announce payload.
# (This module used to have its own separate benchmark_local() with
# differently-named fields — e.g. ram_available_gb instead of
# ram_free_gb — which was never actually called anywhere. Removed to
# avoid two "sources of truth" for the same measurement drifting apart.)


# Rough, conservative RAM divisors for quantization — bitsandbytes on a
# GPU, or CPU-native dynamic quantization for int8 on CPU-only nodes.
# Weights shrink close to these ratios; activation/overhead memory
# doesn't, so we stay conservative rather than promise a fit that then
# OOMs.
_QUANT_RAM_DIVISOR = {"none": 1.0, "int8": 1.8, "int4": 3.2}


def _check_fit(model: ModelEntry, n_nodes: int, min_ram_per_node: float,
               total_ram_gb: float, total_disk_gb: float, quant_level: str):
    """Checks one model against network specs at a given quantization level.
    Returns (reasons_ok, reasons_bad)."""
    divisor       = _QUANT_RAM_DIVISOR[quant_level]
    eff_total_ram = model.total_ram_gb / divisor

    # The weakest node doesn't need model.min_ram_per_node_gb's "average
    # share" — layers are split proportional to each node's free RAM
    # (see shared/provisioning.py's split_layers_by_capacity), so a weak
    # node just ends up with a thin slice instead of needing to carry an
    # equal fraction of the whole model. What actually matters is
    # whether it can hold at least roughly one layer's worth of weights
    # plus a bit of headroom for the interpreter/OS/activations — a much
    # lower bar, and the right one for a mixed fleet of small and large
    # devices.
    per_layer_gb = (eff_total_ram / model.n_layers) if model.n_layers else eff_total_ram
    eff_min_ram  = max(1.5, per_layer_gb + 1.0)

    reasons_ok, reasons_bad = [], []

    if n_nodes < model.min_nodes:
        reasons_bad.append(f"needs {model.min_nodes} node(s), you have {n_nodes}")
    else:
        reasons_ok.append(f"{n_nodes} node(s) ✓")

    if min_ram_per_node < eff_min_ram:
        reasons_bad.append(
            f"needs {eff_min_ram:.1f}GB RAM on your weakest node (to hold at least "
            f"one layer's worth), it has {min_ram_per_node:.1f}GB free"
        )
    else:
        reasons_ok.append(f"{min_ram_per_node:.1f}GB RAM/node ✓")

    if total_ram_gb < eff_total_ram:
        reasons_bad.append(
            f"needs {eff_total_ram:.1f}GB total RAM, network has {total_ram_gb:.1f}GB free"
        )
    else:
        reasons_ok.append(f"{total_ram_gb:.1f}GB total RAM ✓")

    if total_disk_gb < model.download_gb * 1.2:  # 20% buffer
        reasons_bad.append(f"needs {model.download_gb}GB disk, you have {total_disk_gb:.1f}GB free")
    else:
        reasons_ok.append(f"{total_disk_gb:.1f}GB disk ✓")

    return reasons_ok, reasons_bad


def recommend(node_specs: list[dict]) -> list[tuple[ModelEntry, str, str]]:
    """
    Given a list of per-node hardware specs (as returned by /status
    and included in each node's announce payload), returns a list of
    (ModelEntry, reason_string, quant_needed) triples for models that can
    run on this network, ordered best-fit first.

    quant_needed is "none", "int8", or "int4" — the lightest quantization
    level (if any) needed for the model to fit. int8 is offered
    regardless of GPU (CPU-native dynamic quantization, no GPU needed);
    int4 fallback is only offered when a GPU was detected, since that
    path still needs bitsandbytes + CUDA.

    node_specs: list of dicts, each with keys from
    models.benchmark.benchmark_node_local().
    """
    if not node_specs:
        return []

    n_nodes          = len(node_specs)
    total_ram_gb     = sum(s.get("ram_free_gb", 0) for s in node_specs)
    min_ram_per_node = min(s.get("ram_free_gb", 0) for s in node_specs)
    any_gpu          = any(s.get("has_gpu", False) for s in node_specs)
    total_disk_gb    = min(s.get("disk_free_gb", 0) for s in node_specs)  # bottleneck node

    results = []
    for model in CATALOGUE:
        reasons_ok, reasons_bad = _check_fit(
            model, n_nodes, min_ram_per_node, total_ram_gb, total_disk_gb, "none"
        )
        quant_needed = "none"

        if reasons_bad and model.supports_quantization:
            quant_levels = ("int8", "int4") if any_gpu else ("int8",)
            for level in quant_levels:
                ok2, bad2 = _check_fit(
                    model, n_nodes, min_ram_per_node, total_ram_gb, total_disk_gb, level
                )
                if not bad2:
                    reasons_ok, reasons_bad, quant_needed = ok2, bad2, level
                    break

        if reasons_bad:
            continue  # skip incompatible models even with quantization

        reason = ", ".join(reasons_ok)
        if quant_needed != "none":
            reason += f"  [requires {quant_needed} quantization]"
            if quant_needed == "int8" and not any_gpu:
                reason += " (CPU-native, no GPU needed)"
        if any_gpu:
            reason += " (GPU detected — will be faster)"
        results.append((model, reason, quant_needed))

    # Sort: highest quality first, then lightest download as tiebreaker
    results.sort(key=lambda t: (-t[0].quality, t[0].download_gb))
    return results


def print_recommendations(results: list[tuple[ModelEntry, str, str]],
                           network_summary: str = ""):
    if network_summary:
        print(f"\n{network_summary}")

    if not results:
        print("\n  No models in the catalogue fit your current network.")
        print("  Try adding more nodes or freeing up RAM.")
        return

    print(f"\n  {'#':<3}  {'Model':<20}  {'Size':>6}  {'Quality':^9}  {'Notes'}")
    print("  " + "─" * 72)
    for i, (model, reason, quant_needed) in enumerate(results, 1):
        quality_stars = "★" * model.quality + "☆" * (5 - model.quality)
        print(f"  {i:<3}  {model.name:<20}  "
              f"{model.download_gb:>4.1f}GB  {quality_stars}  {model.description}")
        print(f"       {reason}")
        if model.notes:
            print(f"       ↳ {model.notes}")
        print()
