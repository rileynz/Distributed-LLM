"""
Coordinator — sends prompts through the distributed node chain.
Streams each token to the terminal as it's generated.
Auto-selects tokenizer based on active_model.json.
"""

import argparse, os, socket, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests as http
import torch

from config import GENERATION_MAX_NEW_TOKENS, MODEL_CONFIG
from models.active import read_active
from shared.protocol import send_msg, recv_msg


def get_tokenizer():
    cfg = read_active()
    if cfg and cfg.get("type") == "hf":
        from transformers import AutoTokenizer
        import warnings as _w
        _w.filterwarnings("ignore")
        tok = AutoTokenizer.from_pretrained(
            cfg["hf_id"], cache_dir=cfg.get("cache_dir")
        )
        eos = tok.eos_token_id
        return (tok.encode,
                lambda ids: tok.decode(ids, skip_special_tokens=True),
                tok.vocab_size, eos, True)
    from shared.model_def import CharTokenizer
    tok = CharTokenizer()
    return tok.encode, tok.decode, tok.vocab_size, None, False


def fetch_chain(registry_url: str) -> list[dict]:
    try:
        r = http.get(f"{registry_url}/chain", timeout=5)
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach registry at {registry_url}.\n"
            f"Is the registry running?  ({e})"
        )
    if r.status_code != 200:
        raise RuntimeError(
            f"Chain not ready: {r.json().get('error', r.text)}\n"
            f"Check that nodes are running and cover all layers."
        )
    nodes = r.json()["nodes"]
    if not nodes:
        raise RuntimeError("No nodes online — start your node terminals first.")

    # Validate chain covers all model layers — gives a clear error instead of
    # a cryptic ConnectionRefusedError buried inside a node log.
    cfg   = read_active()
    total = (int(cfg["n_layers"]) if cfg and cfg.get("type") == "hf"
             and cfg.get("n_layers") else MODEL_CONFIG.n_layer)
    last  = nodes[-1]["end_layer"]
    if last != total:
        raise RuntimeError(
            f"Chain incomplete: covers layers 0–{last-1}, model needs 0–{total-1}.\n"
            f"Start a node: python run.py → 3 (NODE) → start={last}  end={total}"
        )
    return nodes


def run_through_network(idx, chain, extra=None):
    first = chain[0]
    try:
        with socket.create_connection((first["host"], first["port"]), timeout=30) as s:
            send_msg(s, {"type":"tokens","data":idx,"extra":extra,
                         "chain":chain,"chain_index":0,"timing":[]})
            resp = recv_msg(s)
    except ConnectionRefusedError:
        raise RuntimeError(
            f"Cannot reach node '{first['label']}' at "
            f"{first['host']}:{first['port']} — is it still running?"
        )
    if resp.get("type") == "error":
        raise RuntimeError(f"Node error:\n{resp['data']}")
    return resp["data"], resp.get("timing", [])


def generate(prompt: str, registry_url: str,
             max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
             stream: bool = True):
    encode, decode, _, eos_id, is_hf = get_tokenizer()
    block_size = 1024 if is_hf else 128
    chain  = fetch_chain(registry_url)
    ids    = encode(prompt)
    if not ids:
        raise ValueError("Prompt contained no recognisable characters.")

    idx       = torch.tensor([ids], dtype=torch.long)
    generated = list(ids)
    timings   = []

    if stream:
        sys.stdout.write(decode(generated))
        sys.stdout.flush()

    for _ in range(max_new_tokens):
        ctx = idx[:, -block_size:]
        t0  = time.perf_counter()
        logits, timing = run_through_network(ctx, chain)
        rt_ms = (time.perf_counter() - t0) * 1000
        timings.append({"nodes": timing, "round_trip_ms": round(rt_ms, 2)})

        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if eos_id is not None and next_id == eos_id:
            break

        generated.append(next_id)
        idx = torch.tensor([generated], dtype=torch.long)

        if stream:
            # Decode and print only the new token
            new_text = decode(generated)
            old_text = decode(generated[:-1])
            new_part = new_text[len(old_text):]
            sys.stdout.write(new_part)
            sys.stdout.flush()

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return {"text": decode(generated), "timing": timings, "chain": chain}


def print_timing(result: dict):
    timings = result["timing"]
    chain   = result["chain"]
    n = len(timings)
    if not n:
        return

    total_rt    = sum(t["round_trip_ms"] for t in timings)
    avg_rt      = total_rt / n
    node_totals: dict[str, float] = {}
    for tt in timings:
        for nt in tt["nodes"]:
            node_totals[nt["label"]] = node_totals.get(nt["label"], 0) + nt["compute_ms"]
    avg_compute = sum(node_totals.values()) / n
    avg_net     = avg_rt - avg_compute
    last_nodes  = {t["label"]: t["compute_ms"] for t in timings[-1]["nodes"]}

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  {n} tokens  |  avg {avg_rt:.1f} ms/token  |  {total_rt/1000:.1f}s total")
    print(sep)
    for node in chain:
        lbl = node["label"]
        ac  = node_totals.get(lbl, 0) / n
        lc  = last_nodes.get(lbl, 0)
        print(f"  {lbl:<14}  layers {node['start_layer']:>2}–{node['end_layer']-1:<2}"
              f"   avg {ac:6.1f}ms   last {lc:6.1f}ms")
    print(sep)
    print(f"  Compute   {avg_compute:6.1f} ms/token")
    print(f"  Network   {avg_net:6.1f} ms/token")
    print(f"  Total     {avg_rt:6.1f} ms/token")
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt",     nargs="*")
    parser.add_argument("--registry", default=os.environ.get("REGISTRY_URL",
                                                              "http://127.0.0.1:8000"))
    parser.add_argument("--tokens",   type=int, default=GENERATION_MAX_NEW_TOKENS)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    def run_once(prompt: str):
        try:
            chain = fetch_chain(args.registry)
            cfg   = read_active()
            mid   = cfg.get("model_id", "tinygpt") if cfg else "tinygpt"
            chain_s = " → ".join(n["label"] for n in chain)
            print(f"\n  Model: {mid}  │  {chain_s}")
            print(f"  {'─' * 48}")
            result = generate(prompt, args.registry, args.tokens,
                              stream=not args.no_stream)
            if args.no_stream:
                print(result["text"])
            print_timing(result)
        except (RuntimeError, ValueError) as e:
            print(f"\n  ERROR: {e}\n")

    if args.prompt:
        run_once(" ".join(args.prompt))
        return

    print(f"  Registry: {args.registry}  │  Ctrl+C to stop\n")
    try:
        while True:
            p = input("  Prompt: ").strip()
            if p:
                run_once(p)
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
