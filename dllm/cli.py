from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__, agent, backend, config, coordinator
from .discovery import discover
from .hardware import detect, lan_ip
from .http_utils import request_json
from .security import new_join_code


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return value or default


def _yes(prompt: str, default: bool = True) -> bool:
    marker = "Y/n" if default else "y/N"
    answer = _ask(f"{prompt} ({marker})").lower()
    if not answer:
        return default
    return answer in {"y", "yes", "1", "true"}


def _choice(prompt: str, choices: list[tuple[str, str]], default: str) -> str:
    for key, label in choices:
        print(f"  {key}) {label}")
    valid = {key.lower() for key, _ in choices}
    while True:
        result = _ask(prompt, default).lower()
        if result in valid:
            return result
        print("Choose " + ", ".join(key for key, _ in choices))


def _banner() -> None:
    print("\nDistributed LLM Universal")
    print("Combine suitable Windows and Linux machines for GGUF inference.\n")


def _create_cluster() -> dict:
    cfg = dict(config.DEFAULTS)
    cfg.update({
        "configured": True,
        "role": "coordinator",
        "name": platform.node() or "coordinator",
        "cluster_name": _ask("Cluster name", "Home AI"),
        "join_code": new_join_code(),
        "coordinator_url": f"http://{lan_ip()}:7000",
    })
    cfg["open_browser"] = _yes("Open the dashboard automatically", True)
    config.save(cfg)
    print(f"\nCluster created. Join code: {cfg['join_code']}")
    print(f"Dashboard: http://{lan_ip()}:{cfg['control_port']}")
    if not backend.discover().coordinator_ready and _yes(
        "Install the compatible llama.cpp backend now", True
    ):
        try:
            print("Downloading the official backend. This may take a few minutes...")
            result = backend.install("auto")
            cfg["backend_dir"] = result["directory"]
            config.save(cfg)
            print(f"Backend ready: {result['tag']}")
        except Exception as exc:
            print(f"Backend setup did not finish: {exc}")
            print("The dashboard can retry it later.")
    return cfg


def _pick_cluster() -> str:
    print("Searching the local network...")
    clusters = discover(timeout=2.5)
    if clusters:
        choices = [
            (str(index), f"{item.name} — {item.url}")
            for index, item in enumerate(clusters, 1)
        ]
        choices.append(("m", "Enter an address manually"))
        selected = _choice("Coordinator", choices, "1")
        if selected != "m":
            return clusters[int(selected) - 1].url
    else:
        print("No coordinator answered discovery.")
    return _ask("Coordinator URL", "http://192.168.1.50:7000").rstrip("/")


def _join_cluster(
    url: str | None = None,
    code: str | None = None,
    name: str | None = None,
    *,
    portable: bool = False,
) -> dict:
    cfg = dict(config.DEFAULTS)
    cfg.update({
        "configured": True,
        "role": "agent",
        "name": name or _ask("This machine's name", platform.node() or "worker"),
        "coordinator_url": (url or _pick_cluster()).rstrip("/"),
        "join_code": code or _ask("Six-digit join code"),
        "open_browser": False,
        "worker_connection_mode": "outbound" if portable else "direct",
    })
    errors = config.validate(cfg)
    if errors:
        raise ValueError("; ".join(errors))
    config.save(cfg)
    print("Worker saved. It will now request approval from the coordinator.")
    if portable:
        print("Portable mode: RPC stays on localhost and uses an outbound tunnel.")
    return cfg


def wizard() -> dict:
    _banner()
    choice = _choice("What should this machine do?", [
        ("1", "Create a cluster and show the dashboard"),
        ("2", "Join an existing cluster"),
    ], "1")
    return _create_cluster() if choice == "1" else _join_cluster()


def _open_dashboard(cfg: dict) -> None:
    if not cfg.get("open_browser") or os.environ.get("DLLM_NO_BROWSER"):
        return
    url = f"http://127.0.0.1:{cfg['control_port']}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()


def run_saved(cfg: dict) -> int:
    errors = config.validate(cfg)
    if errors:
        print("Configuration error: " + "; ".join(errors), file=sys.stderr)
        return 2
    if cfg["role"] == "coordinator":
        _open_dashboard(cfg)
        coordinator.run(cfg)
    elif cfg["role"] == "agent":
        agent.run(cfg)
    else:
        print("Run setup first.", file=sys.stderr)
        return 2
    return 0


def doctor(cfg: dict) -> int:
    hardware = detect(run_benchmark=True)
    tools = backend.discover(
        Path(cfg["backend_dir"]) if cfg.get("backend_dir") else None
    )
    print(f"Distributed LLM {__version__} doctor\n")
    print(f"OS:             {hardware.os} {hardware.os_version} ({hardware.architecture})")
    print(f"CPU:            {hardware.cpu}")
    print(f"CPU features:   {', '.join(hardware.cpu_flags) or 'basic'}")
    print(f"RAM:            {hardware.ram_free_gb:.2f} / {hardware.ram_total_gb:.2f} GB free")
    print(f"Memory test:    {hardware.memory_bandwidth_gbps:.2f} GB/s")
    print(f"GPU:            {', '.join(hardware.gpus) or 'not detected'}")
    print(f"llama-server:   {tools.server or 'not installed'}")
    print(f"RPC server:     {tools.rpc_server or 'not installed'}")
    print(f"Role:           {cfg.get('role') or 'setup required'}")
    supported = hardware.architecture in {"x64", "arm64"} and hardware.ram_total_gb >= 1
    print(f"Compatibility:  {'supported' if supported else 'unsupported'}")
    return 0 if supported else 1


def status(cfg: dict) -> int:
    if not cfg.get("configured"):
        print("Setup required. Run without arguments.")
        return 1
    if cfg["role"] == "coordinator":
        url = f"http://127.0.0.1:{cfg['control_port']}/api/status"
    else:
        url = f"{cfg['coordinator_url'].rstrip('/')}/api/status"
    try:
        status_code, payload = request_json(url, timeout=4)
    except Exception as exc:
        print(f"Coordinator unavailable: {exc}")
        return 1
    if status_code != 200:
        print(payload.get("error", f"HTTP {status_code}"))
        return 1
    print(json.dumps({
        "cluster": payload["cluster_name"],
        "workers_online": sum(1 for node in payload["nodes"] if node["online"]),
        "backend_ready": bool(payload["backend"]["rpc_server"]),
        "inference": payload["inference"],
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a cross-platform distributed llama.cpp cluster"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    setup = sub.add_parser("setup", help="Run first-time setup again")
    setup.add_argument(
        "--configure-only",
        action="store_true",
        help="Save setup without starting the configured role",
    )
    sub.add_parser("doctor", help="Check hardware and backend compatibility")
    sub.add_parser("status", help="Show cluster status")
    install = sub.add_parser("install-backend", help="Install llama.cpp")
    install.add_argument(
        "--variant",
        choices=["auto", "cpu", "vulkan", "cuda", "rocm"],
        default="auto",
    )
    join = sub.add_parser("join", help="Configure a headless worker")
    join.add_argument("--coordinator", help="Coordinator URL")
    join.add_argument("--code", help="Six-digit join code")
    join.add_argument("--name", help="Worker name")
    join.add_argument(
        "--portable",
        action="store_true",
        help="Use the admin-free outbound worker tunnel",
    )
    join.add_argument(
        "--configure-only",
        action="store_true",
        help="Save the worker setup without starting it",
    )
    portable = sub.add_parser(
        "portable-worker",
        help="Configure or start an admin-free portable worker",
    )
    portable.add_argument("--coordinator", help="Coordinator URL")
    portable.add_argument("--code", help="Six-digit join code")
    portable.add_argument("--name", help="Worker name")
    portable.add_argument(
        "--configure-only",
        action="store_true",
        help="Save the portable worker setup without starting it",
    )
    sub.add_parser("run", help="Start the configured role")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load()
    packaged_portable = bool(os.environ.get("DLLM_PORTABLE_APP"))
    if args.command == "doctor":
        return doctor(cfg)
    if args.command == "status":
        return status(cfg)
    if args.command == "install-backend":
        result = backend.install(args.variant)
        cfg["backend_dir"] = result["directory"]
        cfg["backend_variant"] = args.variant
        config.save(cfg)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "join":
        cfg = _join_cluster(
            args.coordinator,
            args.code,
            args.name,
            portable=bool(args.portable),
        )
        if args.configure_only:
            print("Configuration saved. Start it with `run.py` or the system service.")
            return 0
        return run_saved(cfg)
    if args.command == "portable-worker" or (
        packaged_portable and args.command is None
    ):
        if not cfg.get("configured"):
            cfg = _join_cluster(
                getattr(args, "coordinator", None),
                getattr(args, "code", None),
                getattr(args, "name", None),
                portable=True,
            )
        elif cfg.get("role") != "agent":
            print(
                "This portable package can only run as a worker. "
                "Move or remove its data folder to configure it again.",
                file=sys.stderr,
            )
            return 2
        elif cfg.get("worker_connection_mode") != "outbound":
            cfg["worker_connection_mode"] = "outbound"
            config.save(cfg)
        if getattr(args, "configure_only", False):
            print("Portable worker configuration saved.")
            return 0
        return run_saved(cfg)
    if args.command == "setup":
        cfg = wizard()
        if args.configure_only:
            print("Configuration saved. Start it with `run.py` or the system service.")
            return 0
        return run_saved(cfg)
    if not cfg.get("configured"):
        if not sys.stdin.isatty():
            print(
                "First run needs a terminal. Use `run.py join --coordinator URL --code 123456` "
                "on a headless worker.",
                file=sys.stderr,
            )
            return 2
        cfg = wizard()
    return run_saved(cfg)
