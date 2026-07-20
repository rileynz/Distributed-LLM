"""
Small shared helper for LAN IP auto-detection.

Used by node/server.py (to pick a sane default for the address it
advertises to the registry), run.py (to pre-fill the same value when
prompting interactively), and setup_wizard.py (for its summary output).
Kept in one place so all three can never disagree with each other.
"""

import errno
import socket


def describe_bind_error(e: OSError, port: int) -> str:
    """
    Turns a raw socket bind failure into an actionable message instead of
    a bare traceback. Covers the two causes people actually hit running
    this project locally:

    - Windows outright refusing the bind (WinError 10013) — almost always
      a leftover Hyper-V/WSL2 port reservation, not a real conflict. This
      is a *block*, not a "port taken" error, so it can hit a totally
      unused port and won't show up in netstat.
    - The port already being in use by something else (cross-platform).
    """
    winerror = getattr(e, "winerror", None)
    lines = [f"Could not open port {port}: {e}"]

    if winerror == 10013:
        lines += [
            "",
            "This is Windows refusing the bind, not a bug in this script — it happens",
            "most often when Hyper-V/WSL2 has reserved a block of ports for itself",
            "(common any time WSL has been used since the last reboot, even on an",
            "otherwise unused port).",
            "",
            "Try, in order:",
            "  1. A different port — this one may just fall in a reserved range.",
            "  2. As Administrator:  wsl --shutdown   then   net stop hns && net start hns",
            "  3. If neither helps: restart the machine, then try again.",
        ]
    elif e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", 10048)):
        lines += [
            "",
            f"Something else is already listening on port {port}.",
            "Pick a different port, or find and stop whatever's using this one.",
        ]
    else:
        lines += [
            "",
            "Try a different port, or check for a firewall/antivirus blocking Python",
            "from opening a listening socket.",
        ]
    return "\n".join(lines)


def detect_lan_ip() -> str:
    """Best-effort guess at this machine's LAN-reachable IP address.

    Opens a UDP socket "connected" to a public address purely so the OS
    tells us which local interface/IP it would route through to get
    there — no packets actually need to be sent or received for this
    trick to work, and no real connection to 8.8.8.8 is made.

    Falls back to 127.0.0.1 if the machine has no network route at all
    (e.g. fully offline), which keeps single-machine use working even
    without connectivity.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
