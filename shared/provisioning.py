"""
Shared "provision the whole network before downloading" logic.

Before this existed, the layer split only ever covered nodes that
already had an assignment (manual --start/--end, or a previous pool
pick), and three different call sites — download_model.py, run.py's
MODELS menu, and the web dashboard — each duplicated their own copy of
the "does this node have a layer range?" check. Any node that hadn't
been through that check yet (e.g. a fresh, unassigned pool node) fell
through to downloading the ENTIRE checkpoint instead of just its
shard, silently defeating the whole point of splitting a model across
weak devices.

This module is the one place that logic lives now:
  1. get_all_alive_nodes()   — every node the registry currently knows
     about, pool and already-assigned alike.
  2. cli_confirm_nodes()     — a hard-stop terminal gate: nothing gets
     split or downloaded until the person explicitly confirms this is
     their full node list.
  3. split_layers_by_capacity() — splits a model's layers across
     *every* confirmed node proportional to each node's free RAM
     (falling back to split_layers_evenly's plain equal split when
     RAM info isn't available), fresh each time, so switching models
     always re-provisions the whole network.
  4. push_assignments() / start_node_download() / poll_node_download()
     — the actual registry + per-node HTTP calls, used identically by
     the CLI and the dashboard so they can never disagree.

Used by: download_model.py, run.py (via download_model.py), and
dashboard/server.py.
"""

import requests as http


# ── Reading the network ───────────────────────────────────────────────────────

def get_all_alive_nodes(registry_url: str) -> list[dict]:
    """Every node the registry currently considers alive — pool nodes
    AND already-assigned ones. Deliberately the full network, not just
    /pool's unassigned subset, since a fresh distributed download
    re-splits every node from scratch."""
    try:
        r = http.get(f"{registry_url.rstrip('/')}/status", timeout=5)
        if r.status_code == 200:
            return [n for n in r.json().get("nodes", []) if n.get("alive")]
    except Exception:
        pass
    return []


def nodes_identity(nodes: list[dict]) -> frozenset:
    """A comparable fingerprint for 'this exact set of nodes', keyed by
    (host, port). Used to detect the node list changing between a
    confirmation step and the moment a download actually fires, so a
    stale confirmation can never be used to greenlight a different set
    of nodes than the one that was actually reviewed."""
    return frozenset((n["host"], n["port"]) for n in nodes)


# ── The CLI gate ──────────────────────────────────────────────────────────────

def cli_confirm_nodes(registry_url: str) -> list[dict] | None:
    """
    Hard-stop terminal gate. Shows every currently alive node (pool and
    already-assigned alike) and blocks until the person explicitly
    types 'go'. Nothing is split, pushed, or downloaded before that.

    Typing 'r' (or just Enter) re-fetches and redisplays the list, so
    nodes started in the meantime show up. Typing 'q' cancels.

    Returns the confirmed node list (possibly empty, if the person
    types 'go' with none online — that's a deliberate way to say
    "just download locally"), or None if cancelled.
    """
    while True:
        nodes = get_all_alive_nodes(registry_url)
        print(f"\n  {len(nodes)} node(s) currently online:")
        if not nodes:
            print("    (none yet — start some with `python node/server.py`, or "
                  "type 'go' to download locally on this machine instead)")
        else:
            for n in sorted(nodes, key=lambda x: x["label"]):
                hw  = n.get("hw_specs", {})
                ram = hw.get("ram_free_gb", 0)
                cur = (f"currently layers {n['start_layer']}-{n['end_layer']-1}"
                       if n.get("start_layer") is not None else "currently unassigned")
                print(f"    {n['label']:<16} {n['host']}:{n['port']:<6}  "
                      f"{ram:>5.1f}GB free RAM   ({cur})")

        choice = input(
            "\n  Add any more nodes now if you want them included in the split. "
            "Type 'go' once every node you want is listed above, 'r' to refresh, "
            "or 'q' to cancel: "
        ).strip().lower()

        if choice in ("go", "g"):
            return nodes
        if choice in ("q", "quit", "cancel"):
            return None
        # Anything else (including blank, 'r', 'refresh', a typo) just
        # loops back around and re-fetches — refreshing is always safe.


# ── Splitting layers across the confirmed set ────────────────────────────────

def split_layers_evenly(nodes: list[dict], total_layers: int) -> dict:
    """
    Splits [0, total_layers) evenly across every node in `nodes`, in
    order, giving the first `total_layers % len(nodes)` nodes one extra
    layer each. Nodes are sorted by label first so the plan is
    deterministic regardless of what order they came back from the
    registry in.

    Assumes len(nodes) <= total_layers — callers should check that
    first (see too_many_nodes_for_model below) and refuse rather than
    call this with more nodes than layers, since handing a node a
    0-width range would leave it with a stale assignment the rest of
    the registry can't safely reconcile.

    Returns {label: {"start_layer", "end_layer", "total_layers"}}.

    Kept around (unused by the main provisioning flow, which now uses
    split_layers_by_capacity below) as a simple, dependency-free
    fallback for anything that genuinely wants a naive equal split.
    """
    if not nodes or total_layers <= 0:
        return {}
    ordered = sorted(nodes, key=lambda n: n["label"])
    n = len(ordered)
    base, extra = total_layers // n, total_layers % n
    assignments, start = {}, 0
    for i, node in enumerate(ordered):
        size = base + (1 if i < extra else 0)
        end  = start + size
        assignments[node["label"]] = {
            "start_layer": start, "end_layer": end, "total_layers": total_layers,
        }
        start = end
    return assignments


# ── RAM-proportional ("smart") splitting ──────────────────────────────────────
#
# An even split is the wrong default for a mixed fleet: a 4GB laptop and
# a 16GB desktop each getting the same number of layers either starves
# the 4GB machine or leaves the 16GB one's extra RAM sitting unused.
# This apportions layers proportional to each node's free RAM instead —
# the same principle several other distributed-inference projects use
# for heterogeneous clusters (e.g. llama.cpp's --tensor-split and exo's
# memory-weighted partitioning size each device's share by its own
# capacity rather than splitting node-count evenly); this implementation
# is our own, tailored to our {label: {start,end,total}} assignment
# format rather than ported from either.

def _apportion(weights: list[float], total: int) -> list[int]:
    """Largest-remainder apportionment (a.k.a. Hamilton's method): splits
    `total` whole units across len(weights) buckets proportional to
    `weights`, so the rounding error from going fractional -> integer
    doesn't systematically favor any one bucket. Every bucket gets at
    least 1 unit, and the result always sums to exactly `total` —
    assuming len(weights) <= total (see too_many_nodes_for_model)."""
    n = len(weights)
    if n == 0 or total <= 0:
        return []
    if n >= total:
        # Not enough units for everyone to get their proportional share
        # AND at least one each: give exactly one to the `total`
        # heaviest-weighted buckets. Callers should avoid this case in
        # practice (too_many_nodes_for_model), but stay correct anyway.
        order = sorted(range(n), key=lambda i: -weights[i])
        counts = [0] * n
        for i in order[:total]:
            counts[i] = 1
        return counts

    total_weight = sum(weights) or 1.0
    raw     = [total * w / total_weight for w in weights]
    counts  = [max(1, int(x)) for x in raw]  # floor, bumped up to at least 1
    remainder = [x - int(x) for x in raw]

    diff = total - sum(counts)
    if diff > 0:
        # Hand the leftover units to whoever was closest to rounding up.
        order = sorted(range(n), key=lambda i: -remainder[i])
        for i in order[:diff]:
            counts[i] += 1
    elif diff < 0:
        # Over-allocated — happens when several low-weight buckets all
        # got bumped to the "at least 1" floor. Trim back from the
        # largest allocations first, never below 1. Bounded loop: each
        # pass can remove at most n units, and |diff| < n by
        # construction, so this always terminates well before the
        # safety bound.
        order = sorted(range(n), key=lambda i: -counts[i])
        idx = 0
        safety_bound = n * total + 10
        while diff < 0 and idx < safety_bound:
            i = order[idx % n]
            if counts[i] > 1:
                counts[i] -= 1
                diff += 1
            idx += 1
    return counts


def _apply_capacity_caps(node_ram_gb: list[float], counts: list[int],
                          per_layer_gb: float) -> list[int]:
    """Caps any node whose allocated layer count would need more than
    ~75% of its own free RAM (leaving headroom for activations, KV
    cache, and everything else that isn't the raw weights). Whatever
    that capping frees up gets handed out one layer at a time to
    whichever node still has the most headroom under its own cap — a
    simple greedy water-fill. If every node is already at its own cap
    (not enough combined 'safe' capacity for this model at all), the
    leftover is spread proportional to each node's original (uncapped)
    share rather than piled onto a single node — this function's job is
    a safety cap on individual nodes, not a feasibility guarantee
    (that's models/recommender.py's job, checked before this ever runs)."""
    n = len(counts)
    if per_layer_gb is None or per_layer_gb <= 0 or n == 0:
        return counts
    SAFETY_MARGIN = 0.75
    caps = [max(1, int((ram * SAFETY_MARGIN) / per_layer_gb)) for ram in node_ram_gb]

    new_counts = [min(counts[i], caps[i]) for i in range(n)]
    remaining  = sum(counts) - sum(new_counts)

    for _ in range(remaining):
        headroom = [caps[i] - new_counts[i] for i in range(n)]
        i = max(range(n), key=lambda j: headroom[j])
        if headroom[i] <= 0:
            break  # nobody has headroom left — stop and fall through below
        new_counts[i] += 1
        remaining -= 1

    if remaining > 0:
        # Every node is simultaneously over its own cap — spreading the
        # shortfall proportional to each node's ORIGINAL share (rather
        # than repeatedly picking whichever node "wins" a tie, which for
        # equally-sized nodes would silently pile everything onto just
        # one of them) keeps a symmetric input symmetric.
        extra = _apportion([max(c, 0.01) for c in counts], remaining)
        for i in range(n):
            new_counts[i] += extra[i]
    return new_counts


def split_layers_by_capacity(nodes: list[dict], total_layers: int,
                              per_layer_gb: float | None = None) -> dict:
    """
    Like split_layers_evenly, but hands out layers proportional to each
    node's free RAM (from its benchmarked hw_specs.ram_free_gb) instead
    of splitting node-count evenly. This is the split used by the real
    provisioning flow (download_model.py, dashboard/server.py) — a
    mixed fleet of small and large devices gets layers sized to what
    each device can actually hold, rather than an equal share that
    either starves the small ones or wastes the large ones' headroom.

    per_layer_gb, if given (typically entry.total_ram_gb / entry.n_layers
    from the model's catalogue entry — a rough per-layer average), adds
    a safety cap: no node is ever handed more layers than ~75% of
    its own free RAM could plausibly hold, with the overflow
    redistributed to nodes that still have room. Omit it (or pass 0/None)
    to skip the cap — reasonable for tiny models (e.g. TinyGPT) where
    even a "wrong" split can't meaningfully overload anything.

    Same invariants and return shape as split_layers_evenly: every node
    gets >= 1 layer, deterministic node ordering (by label), assumes
    len(nodes) <= total_layers (see too_many_nodes_for_model).
    """
    if not nodes or total_layers <= 0:
        return {}
    ordered = sorted(nodes, key=lambda n: n["label"])
    ram     = [n.get("hw_specs", {}).get("ram_free_gb", 0.0) for n in ordered]
    # Floor every weight at a small positive value so a node reporting
    # 0 free RAM (missing hw_specs, benchmark hiccup, etc.) still gets
    # counted rather than silently getting a 0-width share.
    weights = [max(r, 0.05) for r in ram]

    counts = _apportion(weights, total_layers)
    if per_layer_gb:
        counts = _apply_capacity_caps(ram, counts, per_layer_gb)

    assignments, start = {}, 0
    for node, size in zip(ordered, counts):
        end = start + size
        assignments[node["label"]] = {
            "start_layer": start, "end_layer": end, "total_layers": total_layers,
        }
        start = end
    return assignments


def too_many_nodes_for_model(nodes: list[dict], entry) -> bool:
    """True if there are more confirmed nodes than the model has layers
    (each node needs at least one whole layer). Callers should refuse
    to split/download in this case rather than leave some nodes with no
    assignment at all — see split_layers_evenly's docstring."""
    return len(nodes) > entry.n_layers


# ── Pushing the split + triggering downloads ──────────────────────────────────

def push_assignments(registry_url: str, assignments: dict) -> bool:
    if not assignments:
        return True
    try:
        r = http.post(f"{registry_url.rstrip('/')}/assignments",
                      json={"assignments": assignments}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def start_node_download(node: dict, model_payload: dict, assignment: dict | None,
                         timeout: float = 10) -> tuple[bool, str]:
    """POSTs /download to a single node's management port, attaching
    its layer assignment when there is one so it fetches only its own
    shard instead of the whole checkpoint. Returns (ok, message)."""
    mgmt_port = node["port"] + 10000
    payload = dict(model_payload)
    if assignment is not None:
        payload["start_layer"]  = assignment["start_layer"]
        payload["end_layer"]    = assignment["end_layer"]
        payload["total_layers"] = assignment["total_layers"]
    try:
        r = http.post(f"http://{node['host']}:{mgmt_port}/download",
                      json=payload, timeout=timeout)
        if r.status_code == 200:
            return True, "started"
        try:
            msg = r.json().get("error", r.text)
        except Exception:
            msg = r.text
        return False, msg
    except Exception as e:
        return False, str(e)


def poll_node_download(node: dict, timeout: float = 5) -> dict:
    mgmt_port = node["port"] + 10000
    try:
        r = http.get(f"http://{node['host']}:{mgmt_port}/download_status", timeout=timeout)
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}
