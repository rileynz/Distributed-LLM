"""
Single source of truth for model shape and sensible defaults.

NODES and LAYER_SPLITS are now *suggestions* shown in run.py's node
launcher — the actual chain is built dynamically by the registry from
whatever nodes have announced themselves. You no longer need to edit
this file to add a new machine to the network; just run node/server.py
on it, point it at the registry, and it joins automatically.
"""

from shared.model_def import ModelConfig

# ── Model shape ──────────────────────────────────────────────────────────────
# Must match what create_checkpoint.py built. If you change n_layer here,
# re-run create_checkpoint.py before starting the network.
MODEL_CONFIG = ModelConfig(
    vocab_size=None,   # filled in at runtime from the tokenizer
    n_layer=4,
    n_embd=64,
    n_head=4,
    block_size=128,
)

CHECKPOINT_PATH = "model.pt"

# ── Registry ─────────────────────────────────────────────────────────────────
REGISTRY_HOST = "127.0.0.1"
REGISTRY_PORT = 8000
REGISTRY_URL  = f"http://{REGISTRY_HOST}:{REGISTRY_PORT}"

# ── Suggested node layout (used as hints in run.py) ──────────────────────────
# These are not enforced — nodes self-register with whatever layer range
# you give them at startup. Adjust to match how many machines you have.
NODES = [
    ("127.0.0.1", 9001),
    ("127.0.0.1", 9002),
]
LAYER_SPLITS = [
    (0, 2),   # suggested for node 0: layers 0-1
    (2, 4),   # suggested for node 1: layers 2-3
]

GENERATION_MAX_NEW_TOKENS = 60
