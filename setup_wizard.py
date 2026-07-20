"""
Setup wizard — guides you through getting the network running step by step.
Handles Windows firewall rules automatically, tests connectivity between
nodes, and runs a test inference to confirm everything works end-to-end.

Usage:
    python setup_wizard.py
"""

import os, platform, socket, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.netinfo import detect_lan_ip

REQUIRED_PACKAGES = ["torch", "flask", "requests", "transformers", "psutil",
                     "werkzeug", "huggingface_hub"]
IS_WINDOWS = platform.system() == "Windows"


def hdr(title: str):
    print(f"\n{'─' * 56}")
    print(f"  {title}")
    print(f"{'─' * 56}")


def ok(msg): print(f"  ✓  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def err(msg): print(f"  ✗  {msg}")
def info(msg): print(f"     {msg}")
def ask(prompt, default=""): 
    val = input(f"  → {prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return val or default


# ── Step 1: Python version ────────────────────────────────────────────────────

def check_python():
    hdr("Step 1 / 6 — Python version")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro} — good.")
    else:
        err(f"Python {v.major}.{v.minor} detected. Python 3.10+ required.")
        info("Download from https://python.org/downloads/")
        sys.exit(1)


# ── Step 2: Required packages ─────────────────────────────────────────────────

def check_packages():
    hdr("Step 2 / 6 — Required packages")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError:
            warn(f"{pkg} — NOT FOUND")
            missing.append(pkg)

    if missing:
        print()
        choice = ask(f"Install {len(missing)} missing package(s) now? [Y/n]", "Y").upper()
        if choice == "Y":
            cmd = [sys.executable, "-m", "pip", "install"] + missing
            print()
            subprocess.run(cmd, check=True)
            ok("All packages installed.")
        else:
            err("Cannot continue without required packages.")
            sys.exit(1)
    else:
        ok("All packages present.")


# ── Step 3: Model files ───────────────────────────────────────────────────────

def check_model_files():
    hdr("Step 3 / 6 — Model files")
    checkpoint = Path("model.pt")
    active_f   = Path("active_model.json")

    if not checkpoint.exists():
        warn("model.pt not found — creating it now...")
        subprocess.run([sys.executable, "create_checkpoint.py"], check=True)
    ok(f"model.pt ({checkpoint.stat().st_size // 1024}KB)")

    if not active_f.exists():
        from models.active import write_tinygpt
        write_tinygpt()
        ok("active_model.json created (set to TinyGPT).")
    else:
        import json
        cfg = json.loads(active_f.read_text())
        ok(f"active_model.json — active model: {cfg.get('model_id', 'unknown')}")


# ── Step 4: Network topology ──────────────────────────────────────────────────

def plan_topology():
    hdr("Step 4 / 6 — Network topology")
    from config import MODEL_CONFIG
    from models.active import read_active

    cfg          = read_active()
    total_layers = (cfg["n_layers"] if cfg and cfg.get("type") == "hf" and cfg.get("n_layers")
                    else MODEL_CONFIG.n_layer)
    model_id     = cfg.get("model_id", "tinygpt") if cfg else "tinygpt"

    info(f"Active model: {model_id} ({total_layers} layers)")
    print()

    n_nodes_str = ask("How many nodes will you run (including this machine)?", "2")
    try:
        n_nodes = int(n_nodes_str)
        assert 1 <= n_nodes <= total_layers
    except Exception:
        err(f"Must be between 1 and {total_layers}.")
        return None, None, None

    # Compute suggested layer splits
    base  = total_layers // n_nodes
    extra = total_layers  % n_nodes
    splits = []
    start  = 0
    for i in range(n_nodes):
        size = base + (1 if i < extra else 0)
        splits.append((start, start + size))
        start += size

    this_ip = detect_lan_ip()
    nodes = []
    print()
    info(f"This machine's detected LAN IP: {this_ip}")
    info(f"Suggested layer split across {n_nodes} node(s):")
    for i, (s, e) in enumerate(splits):
        default_port = 9001 + i
        info(f"  Node {i}: layers {s}-{e-1}  (--start {s} --end {e}  --port {default_port})")
        port_str = ask(f"  Port for node {i}", str(default_port))
        host_str = ask(f"  IP/host for node {i} (this machine's IP if it runs here: "
                       f"{this_ip}; otherwise that node's own LAN IP)", this_ip)
        nodes.append({"host": host_str, "port": int(port_str),
                      "start": s, "end": e,
                      "label": f"node{i}"})

    registry_port_str = ask("\nRegistry port", "8000")
    registry_port     = int(registry_port_str)
    registry_host     = ask("Registry host (LAN IP of the machine running the registry, "
                            "or 127.0.0.1)", "127.0.0.1")

    return nodes, registry_host, registry_port


# ── Step 5: Firewall ──────────────────────────────────────────────────────────

def configure_firewall(nodes, registry_host, registry_port):
    hdr("Step 5 / 6 — Firewall configuration")

    local_ports = [p for i, n in enumerate(nodes)
                   for p in [n["port"], n["port"] + 10000]
                   if n["host"] in ("127.0.0.1", "localhost",
                                    detect_lan_ip())]
    # Only open the registry's port locally if the registry actually
    # runs on this machine — otherwise this just adds a pointless local
    # firewall rule for a service that isn't here.
    if registry_host in ("127.0.0.1", "localhost", detect_lan_ip()):
        local_ports.append(registry_port)

    if IS_WINDOWS:
        info("On Windows, inbound TCP ports need a firewall rule.")
        info(f"Ports needed: {local_ports}")
        choice = ask("Add Windows Firewall rules automatically? [Y/n]", "Y").upper()
        if choice == "Y":
            _add_windows_rules(local_ports)
        else:
            info("Skipping — add rules manually in Windows Defender Firewall.")
            info("Allow inbound TCP on ports: " + ", ".join(str(p) for p in local_ports))
    else:
        info("On Linux/Mac, check your firewall (ufw, iptables, or macOS Firewall)")
        info(f"Ports to allow inbound TCP: {local_ports}")
        if platform.system() == "Linux":
            choice = ask("Try to add ufw rules? (only if ufw is active) [y/N]", "N").upper()
            if choice == "Y":
                for port in local_ports:
                    try:
                        subprocess.run(["ufw", "allow", f"{port}/tcp"], check=True)
                        ok(f"ufw: allowed port {port}/tcp")
                    except Exception as e:
                        warn(f"ufw failed for port {port}: {e}")
        else:
            info("macOS: System Preferences → Security & Privacy → Firewall → Allow python")

    # Test port availability locally
    print()
    info("Testing that ports are available on this machine...")
    for port in local_ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                ok(f"Port {port} is free and bindable.")
        except OSError as e:
            warn(f"Port {port} — {e}. Something may already be using it.")


def _add_windows_rules(ports):
    for port in ports:
        rule_name = f"DistributedLLM-{port}"
        try:
            # Remove old rule first (ignore errors — it may not exist)
            subprocess.run(
                ["netsh", "advfirewall", "firewall", "delete", "rule",
                 f"name={rule_name}"],
                capture_output=True
            )
            # Add new rule
            result = subprocess.run(
                ["netsh", "advfirewall", "firewall", "add", "rule",
                 f"name={rule_name}",
                 "dir=in", "action=allow", "protocol=TCP",
                 f"localport={port}"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                ok(f"Firewall rule added for port {port}")
            else:
                warn(f"Could not add rule for port {port}: {result.stdout.strip()}")
                info("Try running this wizard as Administrator.")
        except FileNotFoundError:
            warn(f"netsh not found — skipping firewall rule for port {port}")


# ── Step 6: Connectivity test ─────────────────────────────────────────────────

def connectivity_test(nodes, registry_host, registry_port):
    hdr("Step 6 / 6 — Connectivity test")

    # Test registry
    info(f"Testing registry at {registry_host}:{registry_port}...")
    try:
        with socket.create_connection((registry_host, registry_port), timeout=3):
            ok("Registry is reachable.")
        registry_ok = True
    except Exception:
        warn("Registry not reachable yet (it may not be running — start it first).")
        registry_ok = False

    # Test nodes
    for node in nodes:
        if node["host"] in ("127.0.0.1", "localhost"):
            info(f"Skipping connectivity test for local node {node['label']} "
                 f"(it connects to itself).")
            continue
        info(f"Testing node {node['label']} at {node['host']}:{node['port']}...")
        try:
            with socket.create_connection((node["host"], node["port"]), timeout=3):
                ok(f"Node {node['label']} is reachable.")
        except Exception:
            warn(f"Node {node['label']} at {node['host']}:{node['port']} "
                 f"not reachable yet — it may not be running yet.")

    return registry_ok


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(nodes, registry_host, registry_port):
    hdr("Setup complete — start commands")
    registry_url = f"http://{registry_host}:{registry_port}"

    print()
    print(f"  Run each of these in a SEPARATE terminal window,")
    print(f"  in order (registry first):\n")
    print(f"  ① Registry:")
    print(f"      python run.py  → choose 2 (REGISTRY), port {registry_port}\n")
    for i, node in enumerate(nodes):
        print(f"  {'②③④⑤⑥⑦⑧⑨'[i] if i < 8 else str(i+2)} Node {i} (layers {node['start']}-{node['end']-1}):")
        print(f"      python run.py  → choose 3 (NODE)")
        print(f"      registry URL: {registry_url}")
        print(f"      start layer:  {node['start']}")
        print(f"      end layer:    {node['end']}")
        print(f"      port:         {node['port']}")
        print(f"      label:        {node['label']}")
        print(f"      host to advertise: {node['host']}\n")

    n_last = len(nodes)
    print(f"  {'②③④⑤⑥⑦⑧⑨'[n_last] if n_last < 8 else str(n_last+1)} Client:")
    print(f"      python run.py  → choose 4 (CLIENT)")
    print(f"      registry URL: {registry_url}\n")
    print(f"  To download a real AI model after everything is running:")
    print(f"      python run.py  → choose 5 (MODELS)\n")
    print(f"  Your LAN IP (for other machines): {detect_lan_ip()}")
    print()


def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       Distributed LLM — Setup Wizard                ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    info("This wizard will check your setup and help configure")
    info("the network step by step. It won't change anything")
    info("without asking first.\n")

    check_python()
    check_packages()
    check_model_files()
    nodes, registry_host, registry_port = plan_topology()
    if nodes is None:
        return
    configure_firewall(nodes, registry_host, registry_port)
    connectivity_test(nodes, registry_host, registry_port)
    print_summary(nodes, registry_host, registry_port)


if __name__ == "__main__":
    main()
