from __future__ import annotations

import os
import sys
from pathlib import Path


def _portable_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


if "--portable-data" in sys.argv:
    sys.argv.remove("--portable-data")
    os.environ["DLLM_HOME"] = str(_portable_root() / "data")

os.environ["DLLM_PORTABLE_APP"] = "1"
os.environ.setdefault("DLLM_NO_BROWSER", "1")

from dllm.cli import main


raise SystemExit(main())
