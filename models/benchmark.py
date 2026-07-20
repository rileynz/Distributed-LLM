"""
Network performance benchmark.

Runs a real inference timing test on each node and aggregates
a single network score (tokens/sec). Used by download_model.py
to recommend models your hardware can actually run smoothly.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


SCORE_TIERS = [
    (0,   "Minimal",     "DistilGPT-2 or GPT-2 Small only"),
    (10,  "Light",       "GPT-2 Small/Medium, OPT-125M"),
    (25,  "Standard",    "GPT-2 Medium/Large, OPT-350M, TinyLlama"),
    (50,  "Performance", "GPT-2 XL, OPT-1.3B, GPT-Neo 1.3B"),
    (100, "High",        "OPT-2.7B, GPT-Neo 2.7B and above"),
]


def score_tier(score: float) -> tuple[str, str]:
    tier_name = desc = ""
    for min_s, name, d in SCORE_TIERS:
        if score >= min_s:
            tier_name, desc = name, d
    return tier_name, desc


def benchmark_node_local(n_tokens: int = 30) -> dict:
    """
    Runs a quick local inference loop using TinyGPT (or a tiny dummy
    model if TinyGPT isn't loaded) and returns tokens/sec + system info.
    Called on each node; results reported back via /heartbeat extras.
    """
    import psutil, shutil
    vm   = psutil.virtual_memory()
    disk = shutil.disk_usage(".")

    # Simple matrix multiply throughput proxy (no model needed)
    t0 = time.perf_counter()
    a  = torch.randn(256, 256)
    for _ in range(50):
        a = torch.relu(a @ a.T)
    matmul_ms = (time.perf_counter() - t0) * 1000 / 50

    # Token throughput estimation: 1 forward pass on a tiny model
    try:
        from config import MODEL_CONFIG, CHECKPOINT_PATH
        from shared.model_def import CharTokenizer, TinyGPT
        tok = CharTokenizer()
        MODEL_CONFIG.vocab_size = tok.vocab_size
        model = TinyGPT(MODEL_CONFIG)
        try:
            sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
        except Exception:
            pass
        model.eval()

        dummy = torch.zeros(1, 32, dtype=torch.long)
        # warmup
        with torch.no_grad():
            for _ in range(3):
                x = model.forward_embed(dummy)
                x, _ = model.forward_blocks(x, 0, MODEL_CONFIG.n_layer)
                model.forward_head(x)
        # time it
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_tokens):
                x = model.forward_embed(dummy)
                x, _ = model.forward_blocks(x, 0, MODEL_CONFIG.n_layer)
                model.forward_head(x)
        tok_per_sec = n_tokens / (time.perf_counter() - t0)
    except Exception:
        # fallback: estimate from matmul speed
        tok_per_sec = max(1.0, 500 / matmul_ms)

    return {
        "tokens_per_sec":   round(tok_per_sec, 1),
        "matmul_ms":        round(matmul_ms, 2),
        "ram_total_gb":     round(vm.total    / (1024**3), 2),
        "ram_free_gb":      round(vm.available/ (1024**3), 2),
        "disk_free_gb":     round(disk.free   / (1024**3), 2),
        "has_gpu":          torch.cuda.is_available(),
        "gpu_ram_gb":       round(torch.cuda.get_device_properties(0).total_memory/(1024**3), 2)
                            if torch.cuda.is_available() else 0.0,
    }


def aggregate_score(node_results: list[dict]) -> dict:
    """
    Given a list of per-node benchmark dicts, returns an aggregate
    network score and tier.
    """
    if not node_results:
        return {"score": 0, "tier": "Unknown", "tier_desc": "No nodes benchmarked"}

    total_tps    = sum(r.get("tokens_per_sec", 0) for r in node_results)
    total_ram    = sum(r.get("ram_free_gb", 0) for r in node_results)
    min_ram_node = min(r.get("ram_free_gb", 0) for r in node_results)
    any_gpu      = any(r.get("has_gpu", False) for r in node_results)
    total_disk   = min(r.get("disk_free_gb", 0) for r in node_results)  # bottleneck

    # Score = tokens/sec per node × node count bonus × GPU bonus
    score = total_tps * (1 + 0.2 * (len(node_results) - 1))
    if any_gpu:
        score *= 3.0
    score = round(score, 1)

    tier_name, tier_desc = score_tier(score)

    return {
        "score":          score,
        "tier":           tier_name,
        "tier_desc":      tier_desc,
        "n_nodes":        len(node_results),
        "total_tps":      round(total_tps, 1),
        "total_ram_gb":   round(total_ram, 1),
        "min_ram_node_gb": round(min_ram_node, 1),
        "total_disk_gb":  round(total_disk, 1),
        "has_gpu":        any_gpu,
    }


def print_score_report(agg: dict):
    print()
    print("━" * 50)
    print(f"  Network Performance Score: {agg['score']:.0f}")
    print(f"  Tier: {agg['tier']} — {agg['tier_desc']}")
    print("━" * 50)
    print(f"  Nodes:        {agg['n_nodes']}")
    print(f"  Throughput:   {agg['total_tps']:.1f} tokens/sec total")
    print(f"  RAM:          {agg['total_ram_gb']:.1f}GB total free "
          f"({agg['min_ram_node_gb']:.1f}GB min per node)")
    print(f"  Disk:         {agg['total_disk_gb']:.1f}GB free")
    print(f"  GPU:          {'Yes' if agg['has_gpu'] else 'No'}")
    print("━" * 50)
    print()
