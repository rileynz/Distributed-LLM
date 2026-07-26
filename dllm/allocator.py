from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Placement:
    node_id: str
    name: str
    included: bool
    reason: str
    weight: float
    usable_ram_gb: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _score(node: dict) -> float:
    hw = node.get("hardware", {})
    bandwidth = max(float(hw.get("memory_bandwidth_gbps", 0) or 0), 0.1)
    cpu = max(float(hw.get("cpu_score", 0) or 0), 1.0)
    cores = max(int(hw.get("logical_cores", 1) or 1), 1)
    rtt = max(float(node.get("rtt_ms", 0.5) or 0.5), 0.1)
    gpu_bonus = 1.45 if hw.get("gpus") else 1.0
    return ((bandwidth * 0.72) + ((cpu / 10_000) * 0.18) + (cores * 0.10)) * gpu_bonus / (1 + rtt / 25)


def _usable_ram(node: dict, coordinator: bool = False) -> float:
    free_ram = float(node.get("hardware", {}).get("ram_free_gb", 0) or 0)
    reserve = 2.0 if coordinator else 1.0
    return max(0.0, free_ram - reserve)


def _bounded_weights(selected: list[dict], required: float) -> list[float]:
    if not selected:
        return []
    scores = [max(item["_score"], 0.001) for item in selected]
    score_total = sum(scores)
    caps = [min(1.0, item["_usable"] / max(required, 0.001)) for item in selected]
    weights = [min(score / score_total, cap) for score, cap in zip(scores, caps)]
    remaining = max(0.0, 1.0 - sum(weights))
    for _ in range(len(selected) * 3):
        if remaining <= 1e-9:
            break
        available = [index for index, cap in enumerate(caps) if weights[index] + 1e-9 < cap]
        if not available:
            break
        available_score = sum(scores[index] for index in available)
        distributed = 0.0
        for index in available:
            share = remaining * scores[index] / available_score
            addition = min(share, caps[index] - weights[index])
            weights[index] += addition
            distributed += addition
        if distributed <= 1e-12:
            break
        remaining -= distributed
    total = sum(weights)
    if total >= 0.999999:
        return [weight / total for weight in weights]
    return weights


def plan(
    nodes: list[dict],
    model_size_gb: float,
    *,
    context_size: int = 4096,
    include_local: bool = True,
    auto_exclude_slow: bool = True,
) -> dict:
    """Produce a conservative, capability-based RPC tensor split.

    The model file size is expanded for runtime buffers and KV cache. Devices
    under 2 GB free RAM are kept connected for monitoring but are not used for
    model tensors.
    """
    context_overhead = max(context_size, 512) / 4096 * 0.45
    required = model_size_gb * 1.16 + context_overhead
    candidates: list[dict] = []
    placements: list[Placement] = []

    for node in nodes:
        node_id = str(node.get("node_id") or node.get("id") or "local")
        name = str(node.get("name") or node_id)
        is_local = bool(node.get("local"))
        usable = _usable_ram(node, coordinator=is_local)
        score = _score(node)
        if is_local and not include_local:
            placements.append(Placement(node_id, name, False, "Local compute disabled", 0, usable, score))
        elif usable < 1.0:
            placements.append(Placement(node_id, name, False, "Less than 1 GB safe inference memory", 0, usable, score))
        elif not node.get("approved", True):
            placements.append(Placement(node_id, name, False, "Waiting for approval", 0, usable, score))
        elif not is_local and not node.get("rpc_ready", False):
            placements.append(Placement(node_id, name, False, "RPC backend is not ready", 0, usable, score))
        else:
            entry = dict(node)
            entry["_id"] = node_id
            entry["_name"] = name
            entry["_usable"] = usable
            entry["_score"] = score
            candidates.append(entry)

    candidates.sort(key=lambda item: item["_score"], reverse=True)
    fastest = candidates[0]["_score"] if candidates else 0
    selected: list[dict] = []
    capacity = 0.0
    for candidate in candidates:
        if auto_exclude_slow and capacity >= required:
            placements.append(Placement(
                candidate["_id"], candidate["_name"], False,
                "Not needed for this model; kept available for monitoring",
                0, candidate["_usable"], candidate["_score"],
            ))
            continue
        selected.append(candidate)
        capacity += candidate["_usable"]

    selected_ids = {item["_id"] for item in selected}
    raw_weights = _bounded_weights(selected, required)

    for candidate, raw in zip(selected, raw_weights):
        placements.append(Placement(
            candidate["_id"], candidate["_name"], True,
            "Selected by RAM and measured performance",
            round(raw, 6),
            round(candidate["_usable"], 3),
            round(candidate["_score"], 3),
        ))

    placements.sort(key=lambda item: (not item.included, -item.score, item.name))
    return {
        "fits": capacity >= required,
        "model_size_gb": round(model_size_gb, 3),
        "estimated_required_gb": round(required, 3),
        "safe_capacity_gb": round(capacity, 3),
        "placements": [item.to_dict() for item in placements],
        "selected_ids": sorted(selected_ids),
    }


def fit_label(plan_result: dict) -> str:
    if not plan_result["fits"]:
        return "Will not fit"
    ratio = plan_result["safe_capacity_gb"] / max(plan_result["estimated_required_gb"], 0.001)
    selected = [p for p in plan_result["placements"] if p["included"]]
    if ratio >= 1.8 and len(selected) <= 2:
        return "Fast"
    if ratio >= 1.25:
        return "Should run"
    return "May be slow"
