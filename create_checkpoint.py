"""
Run this ONCE before starting any nodes.

All nodes must run the exact same model weights, or the layers won't
fit together correctly (it would be like swapping pages from two
different books mid-sentence). Since this demo model isn't trained,
we just initialize it randomly here and save it to model.pt so every
node loads the identical file.

Usage:
    python create_checkpoint.py
"""

import torch

from config import MODEL_CONFIG, CHECKPOINT_PATH
from shared.model_def import CharTokenizer, TinyGPT


def main():
    tokenizer = CharTokenizer()
    MODEL_CONFIG.vocab_size = tokenizer.vocab_size

    torch.manual_seed(42)  # reproducible weights
    model = TinyGPT(MODEL_CONFIG)

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
    print(f"  vocab_size={MODEL_CONFIG.vocab_size}  n_layer={MODEL_CONFIG.n_layer}  "
          f"n_embd={MODEL_CONFIG.n_embd}  n_head={MODEL_CONFIG.n_head}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  total parameters: {total_params:,}")


if __name__ == "__main__":
    main()
