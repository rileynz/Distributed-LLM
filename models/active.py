"""
Tracks which model is currently active across the network.

Stored in active_model.json next to model.pt. Nodes read this file on
startup (and when hot-reloading) to decide whether to load TinyGPT or
a real HuggingFace model.

Structure of active_model.json:
{
    "type": "tinygpt" | "hf",
    "model_id": "gpt2",           # only for type=hf
    "hf_id": "gpt2",              # HuggingFace model string
    "cache_dir": "models/cache",  # where the weights are stored on disk
    "n_layers": 12,               # total transformer layer count
    "updated_at": 1234567890.0    # unix timestamp; nodes use this to detect changes
}
"""

import json
import time
from pathlib import Path

ACTIVE_MODEL_PATH = Path("active_model.json")


def read_active() -> dict | None:
    """Returns the active model config dict, or None if not set."""
    if not ACTIVE_MODEL_PATH.exists():
        return None
    try:
        with open(ACTIVE_MODEL_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def write_tinygpt(checkpoint_path: str = "model.pt"):
    """Mark TinyGPT (the built-from-scratch model) as active."""
    _write({
        "type":       "tinygpt",
        "model_id":   "tinygpt",
        "checkpoint": str(checkpoint_path),
        "n_layers":   None,  # read from config.py at runtime
        "updated_at": time.time(),
    })


def write_hf(model_id: str, hf_id: str, cache_dir: str, n_layers: int, arch: str = "auto"):
    """Mark a HuggingFace model as active."""
    _write({
        "type":       "hf",
        "model_id":   model_id,
        "hf_id":      hf_id,
        "cache_dir":  str(cache_dir),
        "n_layers":   n_layers,
        "arch":       arch,
        "updated_at": time.time(),
    })


def _write(data: dict):
    with open(ACTIVE_MODEL_PATH, "w") as f:
        json.dump(data, f, indent=2)


def is_hf_active() -> bool:
    cfg = read_active()
    return cfg is not None and cfg.get("type") == "hf"


def active_model_id() -> str:
    cfg = read_active()
    if cfg is None:
        return "tinygpt"
    return cfg.get("model_id", "tinygpt")
