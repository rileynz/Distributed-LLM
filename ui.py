"""
Terminal UI helpers — colours, box drawing, tables, spinner.
Falls back gracefully to plain text on terminals without colour support.
"""

import os, re, shutil, sys, threading, time

# ── Colour support ────────────────────────────────────────────────────────────

def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            k = ctypes.windll.kernel32
            handle = k.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not k.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            k.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_USE_COLOUR = _supports_colour()


class C:
    RESET   = "\033[0m"  if _USE_COLOUR else ""
    BOLD    = "\033[1m"  if _USE_COLOUR else ""
    DIM     = "\033[2m"  if _USE_COLOUR else ""
    RED     = "\033[31m" if _USE_COLOUR else ""
    GREEN   = "\033[32m" if _USE_COLOUR else ""
    YELLOW  = "\033[33m" if _USE_COLOUR else ""
    BLUE    = "\033[34m" if _USE_COLOUR else ""
    MAGENTA = "\033[35m" if _USE_COLOUR else ""
    CYAN    = "\033[36m" if _USE_COLOUR else ""
    WHITE   = "\033[37m" if _USE_COLOUR else ""
    LWHITE  = "\033[97m" if _USE_COLOUR else ""
    GRAY    = "\033[90m" if _USE_COLOUR else ""
    BG_BLUE = "\033[44m" if _USE_COLOUR else ""


def _wrap(on: str):
    def fn(text: str) -> str:
        return f"{on}{text}{C.RESET}" if _USE_COLOUR else str(text)
    return fn

bold    = _wrap(C.BOLD)
dim     = _wrap(C.DIM)
red     = _wrap(C.RED)
green   = _wrap(C.GREEN)
yellow  = _wrap(C.YELLOW)
blue    = _wrap(C.BLUE)
magenta = _wrap(C.MAGENTA)
cyan    = _wrap(C.CYAN)
gray    = _wrap(C.GRAY)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)

def _visible_len(text: str) -> int:
    return len(_strip_ansi(text))


# ── Terminal ──────────────────────────────────────────────────────────────────

def term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns

def clear():
    os.system("cls" if sys.platform == "win32" else "clear")


# ── Box drawing ───────────────────────────────────────────────────────────────

def top_bar(w: int) -> str:
    return f"╔{'═' * (w - 2)}╗"

def bot_bar(w: int) -> str:
    return f"╚{'═' * (w - 2)}╝"

def hline(w: int, l: str = "╠", r: str = "╣") -> str:
    return f"{l}{'═' * (w - 2)}{r}"

def row(content: str, w: int) -> str:
    """Fit content inside a box row of total width w, truncating if needed."""
    max_content = w - 5          # ║  …content…  ║  (2 left + 1 right space + 2 borders)
    visible     = _visible_len(content)
    if visible > max_content:
        # Truncate preserving ANSI — strip, cut, no ANSI in truncated part
        plain = _strip_ansi(content)[:max_content - 1] + "…"
        content = plain
        visible = _visible_len(content)
    pad = max(0, max_content - visible + 1)   # +1 for the right space
    return f"║  {content}{' ' * pad}║"

def section(title: str, w: int = 72) -> str:
    return "\n".join([hline(w), row(bold(cyan(title)), w)])


# ── Table ─────────────────────────────────────────────────────────────────────

def table(headers: list, rows: list,
          col_colours: list | None = None) -> list:
    if not rows:
        return [f"  {gray('(none)')}"]
    n_cols = len(headers)
    col_w  = [_visible_len(str(h)) for h in headers]
    for r in rows:
        for i, cell in enumerate(r[:n_cols]):
            col_w[i] = max(col_w[i], _visible_len(str(cell)))

    def fmt(cells, header=False):
        parts = []
        for i, cell in enumerate(cells[:n_cols]):
            txt = str(cell)
            pad = " " * max(0, col_w[i] - _visible_len(txt))
            parts.append((bold(txt) if header else txt) + pad)
        return "  " + "   ".join(parts)

    lines = [fmt(headers, header=True),
             "  " + "   ".join("─" * w for w in col_w)]
    for r in rows:
        lines.append(fmt(r))
    return lines


# ── Spinner ───────────────────────────────────────────────────────────────────

class Spinner:
    _FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, message: str = ""):
        self._msg    = message
        self._stop   = threading.Event()
        self._thread = None
        self._started = False

    def start(self) -> "Spinner":
        self._started = True
        if not _USE_COLOUR:
            print(f"  {self._msg}")
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stdout.write(f"\r  {cyan(frame)}  {self._msg} ")
            sys.stdout.flush()
            time.sleep(0.09)
            i += 1

    def stop(self, final_msg: str = ""):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if _USE_COLOUR and self._started:
            sys.stdout.write("\r" + " " * (term_width() - 1) + "\r")
            sys.stdout.flush()
        if final_msg:
            print(f"  {green('✓')}  {final_msg}")
