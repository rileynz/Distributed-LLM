"""
Distributed LLM — interactive launcher and live dashboard.
    python run.py
"""

import os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests as http

from ui import (C, clear, bold, cyan, green, red, yellow, gray, dim,
                row, top_bar, bot_bar, hline, table, Spinner,
                term_width, _strip_ansi, _visible_len)
from config import CHECKPOINT_PATH, MODEL_CONFIG
from models.active import active_model_id, read_active
from shared.netinfo import detect_lan_ip

DEFAULT_REGISTRY = os.environ.get("REGISTRY_URL", "http://127.0.0.1:8000")


# ── Helpers ───────────────────────────────────────────────────────────────────

def W() -> int:
    """Dashboard width — responsive, capped at 80."""
    return min(80, max(60, term_width() - 2))


def fetch_status(registry_url: str) -> dict | None:
    try:
        r = http.get(f"{registry_url}/status", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def ask(prompt: str, default: str = "", valid: list | None = None) -> str:
    hint = f" {gray(f'[{default}]')}" if default else ""
    while True:
        try:
            val = input(f"  {cyan('→')} {prompt}{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        val = val or default
        if valid is None or val in valid:
            return val
        print(f"  {red('✗')} Enter one of: {', '.join(valid)}")


def ok(msg):   print(f"  {green('✓')}  {msg}")
def warn(msg): print(f"  {yellow('⚠')}  {msg}")
def err(msg):  print(f"  {red('✗')}  {msg}")
def info(msg): print(f"  {gray('·')}  {msg}")


def section_header(title: str):
    bar = cyan("─" * (len(title) + 4))
    print(f"\n  {bar}")
    print(f"  {cyan('│')} {bold(title)} {cyan('│')}")
    print(f"  {bar}\n")


# ── Dashboard ─────────────────────────────────────────────────────────────────

def draw_dashboard(registry_url: str, last_refresh: float):
    clear()
    w      = W()
    status = fetch_status(registry_url)
    cfg    = read_active()

    model_id   = cfg.get("model_id", "tinygpt") if cfg else "tinygpt"
    model_type = cfg.get("type",     "tinygpt") if cfg else "tinygpt"
    n_layers   = (cfg.get("n_layers") or MODEL_CONFIG.n_layer) if cfg else MODEL_CONFIG.n_layer

    # ── Header ────────────────────────────────────────────────────────────────
    print(top_bar(w))
    print(row(bold(cyan("⚡  Distributed LLM Network")), w))
    print(row(gray("Split AI inference across multiple machines"), w))
    print(hline(w))

    # Model + registry status
    model_col  = cyan if model_type == "hf" else yellow
    reg_icon   = green("● ONLINE") if status else red("○ OFFLINE")
    ago        = int(time.time() - last_refresh)
    refresh_s  = gray(f"refreshed {ago}s ago") if ago > 0 else gray("just refreshed")

    print(row(f"{bold('Model')}   {model_col(model_id)}  {gray(f'({n_layers} layers)')}", w))
    print(row(f"{bold('Registry')} {gray(registry_url)}  {reg_icon}  {refresh_s}", w))

    # ── Node table ────────────────────────────────────────────────────────────
    print(hline(w))
    print(row(bold(cyan("NODES")), w))
    print(row("", w))

    if status is None:
        print(row(f"  {yellow('⚠')}  Registry not reachable — start it first (option 2).", w))

    elif not status["nodes"]:
        print(row(f"  {yellow('⚠')}  No nodes connected yet — start nodes with option 3.", w))

    else:
        nodes  = status["nodes"]
        t_rows = []
        for n in nodes:
            alive  = n.get("alive", False)
            hw     = n.get("hw_specs", {})
            if n.get("start_layer") is None:
                layers = gray("unassigned")
            else:
                layers = f"{n['start_layer']}–{n['end_layer']-1}"
            status_s = green("● LIVE") if alive else red("○ DEAD")
            ram_s    = f"{hw['ram_free_gb']:.1f}GB" if hw.get("ram_free_gb") else gray("—")
            tps_s    = f"{hw['tokens_per_sec']:.0f} t/s" if hw.get("tokens_per_sec") else gray("—")
            dev_s    = green("GPU") if hw.get("has_gpu") else gray("CPU")
            reqs_s   = str(n.get("requests_served", 0))
            t_rows.append([bold(n["label"]), layers, status_s, ram_s, tps_s, dev_s, reqs_s])

        for line in table(["Label","Layers","Status","RAM","Speed","Device","Reqs"], t_rows):
            print(row(line, w))

        pool_n = status.get("pool_nodes", 0)
        if pool_n:
            print(row("", w))
            print(row(f"  {cyan('·')}  {pool_n} node(s) waiting for a model assignment "
                       f"— pick one in the MODELS menu (option 5).", w))

        # Chain health
        print(row("", w))
        if status.get("chain_ready"):
            alive_n = status.get("alive_nodes", 0)
            chain_s = green(f"✓ Chain ready  ({alive_n} node(s) live)")
        else:
            err_msg = status.get("chain_error") or "incomplete"
            chain_s = red(f"✗ {err_msg}")
        print(row(f"  {chain_s}", w))

    # ── Menu ──────────────────────────────────────────────────────────────────
    print(hline(w))
    for key, name, desc in [
        ("1", "SETUP",    "First time? Step-by-step configuration"),
        ("2", "REGISTRY", "Start the network phonebook (do this first)"),
        ("3", "NODE",     "Host model layers — join the network"),
        ("4", "CLIENT",   "Send prompts, see output + per-node timing"),
        ("5", "MODELS",   "Download real AI models, benchmark hardware"),
        ("6", "DASHBOARD","Web view: live pipeline, model switching in a browser"),
        ("R", "REFRESH",  "Refresh this screen"),
        ("Q", "QUIT",     "Exit"),
    ]:
        key_s  = f"{C.BG_BLUE}{C.LWHITE} {key} {C.RESET}"
        name_s = bold(f"{name:<10}")
        desc_s = gray(desc)
        print(row(f"  {key_s}  {name_s}  {desc_s}", w))

    print(bot_bar(w))
    print()


# ── Option handlers ───────────────────────────────────────────────────────────

def run_setup():
    section_header("Setup Wizard")
    try:
        subprocess.run([sys.executable, "setup_wizard.py"])
    except KeyboardInterrupt:
        pass


def run_registry():
    section_header("Start Registry")
    port = ask("Port", default="8000")
    if not port.isdigit():
        port = "8000"
    print(f"\n  {green('▶')}  Registry starting on port {bold(port)}"
          f"  {gray('— Ctrl+C to stop')}\n")
    try:
        subprocess.run([sys.executable, "registry/server.py", "--port", port])
    except KeyboardInterrupt:
        pass
    print(f"\n  {yellow('■')}  Registry stopped.")


def _auto_assign_layers(registry_url: str, total: int,
                         n_nodes_total: int, node_index: int) -> tuple[int, int]:
    """
    Returns the layer range for this node.
    Primary: the node_index-th slot of an even split.
    Override: if that slot is already taken, pick the first uncovered slot.
    """
    base, extra = total // n_nodes_total, total % n_nodes_total
    splits, start = [], 0
    for i in range(n_nodes_total):
        sz = base + (1 if i < extra else 0)
        splits.append((start, start + sz))
        start += sz

    my_slot = splits[min(node_index, len(splits) - 1)]

    try:
        r = http.get(f"{registry_url}/status", timeout=3)
        if r.status_code == 200:
            alive = [n for n in r.json()["nodes"] if n["alive"] and n["start_layer"] is not None]
            if alive:
                covered = set()
                for n in alive:
                    for layer in range(n["start_layer"], n["end_layer"]):
                        covered.add(layer)
                s, e = my_slot
                if all(l in covered for l in range(s, e)):
                    for slot_s, slot_e in splits:
                        if not any(l in covered for l in range(slot_s, slot_e)):
                            return slot_s, slot_e
    except Exception:
        pass

    return my_slot


def run_node():
    section_header("Start Node")

    registry_url = ask("Registry URL", default=DEFAULT_REGISTRY)

    print()
    info("Easiest: just add this node to the pool — once you've added all your")
    info("nodes, pick MODELS from the main menu to benchmark them together, get")
    info("model recommendations, and have layers auto-assigned. No restart needed.")
    info("(Or set this node's layer range yourself right now, if you prefer.)")
    print()
    mode = ask("Add to pool (recommended), or set layers manually? [pool/manual]",
               default="pool")

    detected_ip = detect_lan_ip()

    if mode.strip().lower() not in ("manual", "m"):
        port  = ask("Port", default="9001")
        label = ask("Label", default=f"node-{port}")
        info(f"This machine's detected LAN IP: {bold(cyan(detected_ip))}")
        info("Other machines will connect to THIS node using the address below.")
        host = ask("Host/IP to advertise for this node", default=detected_ip)

        print()
        print(f"  {cyan('┌─ Starting pool node ───────────────')}")
        print(f"  {cyan('│')}  Label:    {bold(label)}")
        print(f"  {cyan('│')}  Layers:   {gray('unassigned — pick a model in the MODELS menu')}")
        print(f"  {cyan('│')}  Port:     {bold(port)}")
        print(f"  {cyan('│')}  Host:     {bold(host)}  {gray('(what other machines will use)')}")
        print(f"  {cyan('│')}  Registry: {gray(registry_url)}")
        print(f"  {cyan('└────────────────────────────────────')}")
        print()

        cmd = [sys.executable, "node/server.py",
               "--port", port, "--registry", registry_url,
               "--label", label, "--host", host]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        print(f"\n  {yellow('■')}  {label} stopped.")
        return

    # ── Manual layer assignment ───────────────────────────────────────────
    cfg   = read_active()
    active_is_tinygpt = cfg is None or cfg.get("type") == "tinygpt"
    if active_is_tinygpt and not Path(CHECKPOINT_PATH).exists():
        err("No model is active yet, and model.pt (the built-in TinyGPT demo model) "
            "doesn't exist either.")
        info("Either run create_checkpoint.py first, or use the MODELS menu to switch "
             "to a real downloaded model before assigning a node's layers manually.")
        return

    total = (cfg["n_layers"] if cfg and cfg.get("type") == "hf" and cfg.get("n_layers")
             else MODEL_CONFIG.n_layer)
    mid   = active_model_id()

    info(f"Active model: {bold(cyan(mid))}  {gray(f'({total} layers)')}")
    print()

    n_str = ask("How many nodes will be in this network total?", default="2")
    try:
        n_total = max(1, min(int(n_str), total))
    except ValueError:
        n_total = 2

    # Compute and display the full layer plan
    base, extra = total // n_total, total % n_total
    splits, start = [], 0
    for i in range(n_total):
        sz = base + (1 if i < extra else 0)
        splits.append((start, start + sz))
        start += sz

    info(f"Layer plan for {bold(str(n_total))} node(s):")
    for i, (s, e) in enumerate(splits):
        info(f"    Node {i}:  layers {s}–{e-1}   port {9001+i}")
    print()

    idx_str = ask(f"Which node number is THIS terminal?  (0–{n_total-1})", default="0")
    try:
        node_idx = max(0, min(int(idx_str), n_total - 1))
    except ValueError:
        node_idx = 0

    auto_s, auto_e = _auto_assign_layers(registry_url, total, n_total, node_idx)
    info(f"Auto-assigned:  layers {bold(cyan(str(auto_s)))}–{bold(cyan(str(auto_e-1)))}  "
         f"{gray('(press Enter to accept)')}")
    print()

    start_s = ask("Start layer (inclusive)", default=str(auto_s))
    end_s   = ask("End layer   (exclusive)", default=str(auto_e))
    try:
        s, e = int(start_s), int(end_s)
        assert 0 <= s < e <= total
    except Exception:
        err(f"Invalid range — must be 0 ≤ start < end ≤ {total}")
        return

    port  = ask("Port", default=str(9001 + node_idx))
    label = ask("Label", default=f"node{node_idx}")

    detected_ip = detect_lan_ip()
    info(f"This machine's detected LAN IP: {bold(cyan(detected_ip))}")
    info("Other machines (and other nodes) will connect to THIS node using")
    info("the address you give below. Use the detected LAN IP if any node or")
    info("the client will run on a different machine. Only use 127.0.0.1 if")
    info("literally everything runs on this one machine.")
    print()
    host = ask("Host/IP to advertise for this node", default=detected_ip)

    # Warn if layer 0 won't be covered
    if s != 0:
        try:
            r = http.get(f"{registry_url}/status", timeout=2)
            if r.status_code == 200:
                alive = [n for n in r.json()["nodes"] if n["alive"]]
                if not any(n["start_layer"] == 0 for n in alive):
                    warn("No node covers layer 0 yet — chain will be incomplete until one is running.")
        except Exception:
            pass

    # Confirmation summary
    print()
    print(f"  {cyan('┌─ Starting node ────────────────────')}")
    print(f"  {cyan('│')}  Label:   {bold(label)}")
    print(f"  {cyan('│')}  Layers:  {bold(str(s))}–{bold(str(e-1))}  of {total}")
    print(f"  {cyan('│')}  Port:    {bold(port)}")
    print(f"  {cyan('│')}  Host:    {bold(host)}  {gray('(what other machines will use)')}")
    print(f"  {cyan('│')}  Registry: {gray(registry_url)}")
    print(f"  {cyan('└────────────────────────────────────')}")
    print()

    cmd = [sys.executable, "node/server.py",
           "--port", port, "--start", str(s), "--end", str(e),
           "--registry", registry_url, "--label", label, "--host", host]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    print(f"\n  {yellow('■')}  {label} stopped.")


def run_client():
    section_header("Client")

    registry_url = ask("Registry URL", default=DEFAULT_REGISTRY)

    sp = Spinner("Connecting to registry…").start()
    try:
        http.get(f"{registry_url}/health", timeout=4).raise_for_status()
        sp.stop(f"Registry reachable")
    except Exception as e:
        sp.stop()
        err(f"Cannot reach registry at {registry_url}")
        info(f"  {str(e)}")
        return

    # Show model + chain status before entering the prompt loop
    st = fetch_status(registry_url)
    if st:
        cfg = read_active()
        mid = cfg.get("model_id", "tinygpt") if cfg else "tinygpt"
        n_l = cfg.get("n_layers") or MODEL_CONFIG.n_layer if cfg else MODEL_CONFIG.n_layer
        alive = sorted([n for n in st["nodes"] if n["alive"] and n["start_layer"] is not None],
                       key=lambda x: x["start_layer"])
        print()
        info(f"Model:  {bold(cyan(mid))}  {gray(f'({n_l} layers)')}")
        if alive:
            chain_s = "  →  ".join(green(n["label"]) for n in alive)
            info(f"Chain:  {chain_s}")
        else:
            warn("No nodes online — start node terminals first.")
            return

        if not st.get("chain_ready"):
            warn(f"Chain incomplete: {st.get('chain_error', '')}")
            if ask("Continue anyway?", default="n", valid=["y","n"]) == "n":
                return

    print(f"\n  {gray('Ctrl+C to stop')}\n")
    try:
        subprocess.run([sys.executable, "coordinator/client.py",
                        "--registry", registry_url])
    except KeyboardInterrupt:
        pass
    print(f"\n  {yellow('■')}  Client stopped.")


def run_models():
    section_header("Model Manager")
    registry_url = ask("Registry URL", default=DEFAULT_REGISTRY)
    print()

    try:
        subprocess.run([sys.executable, "download_model.py",
                        "--registry", registry_url])
    except KeyboardInterrupt:
        pass


def run_dashboard():
    section_header("Web Dashboard")
    registry_url = ask("Registry URL", default=DEFAULT_REGISTRY)
    port = ask("Dashboard port", default="7000")
    print(f"\n  Once running, open http://127.0.0.1:{port} in your browser")
    print(f"  (or http://<this machine's LAN IP>:{port} from another device on your wifi).")
    print(f"  Press Ctrl+C here to stop it.\n")
    try:
        subprocess.run([sys.executable, "dashboard/server.py",
                        "--registry", registry_url, "--port", port])
    except KeyboardInterrupt:
        pass


# ── Main loop ─────────────────────────────────────────────────────────────────

ACTIONS = {"1": run_setup, "2": run_registry, "3": run_node,
           "4": run_client, "5": run_models, "6": run_dashboard}


def main():
    registry_url  = DEFAULT_REGISTRY
    last_refresh  = time.time()

    while True:
        draw_dashboard(registry_url, last_refresh)
        last_refresh = time.time()

        try:
            choice = input(
                f"  {cyan('→')} {bold('Choose')} {gray('[1-6 / R / Q]')}: "
            ).strip().upper()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {yellow('Goodbye.')}\n")
            sys.exit(0)

        if not choice:
            continue
        if choice == "Q":
            print(f"\n  {yellow('Goodbye.')}\n")
            sys.exit(0)
        if choice == "R":
            continue
        if choice in ACTIONS:
            print()
            ACTIONS[choice]()
            input(f"\n  {gray('Press Enter to return to dashboard…')}")
        else:
            err(f"Unknown option '{choice}'.")
            time.sleep(0.8)


if __name__ == "__main__":
    main()
