from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from dllm import backend


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a pinned llama.cpp backend into a portable package"
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--variant",
        choices=["auto", "cpu", "vulkan", "cuda", "rocm"],
        default="cpu",
    )
    args = parser.parse_args()
    result = backend.install(args.variant, args.destination.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
