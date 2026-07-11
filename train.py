"""
Trains the tiny model on real text so the network actually produces
sensible-ish output instead of random gibberish.

This trains the SAME architecture the nodes already split between
themselves — training doesn't know or care that the model will later
be split across machines, it just produces one normal model.pt file,
exactly like create_checkpoint.py does, except with learned weights
instead of random ones.

Usage:
    python train.py                      # trains on data/sample.txt
    python train.py --data path/to.txt   # trains on your own text file
    python train.py --steps 3000         # train for longer (better results, more time)

After training finishes, model.pt is overwritten with the trained
weights. Just run the network as usual afterwards (run.py / run_demo.py)
— no other changes needed, since every node loads model.pt fresh.
"""

import argparse
import time

import torch
import torch.nn.functional as F

from config import MODEL_CONFIG, CHECKPOINT_PATH
from shared.model_def import CharTokenizer, TinyGPT


def get_batch(data, block_size, batch_size, device):
    """Picks `batch_size` random windows of length block_size+1 from data,
    splits each into an input chunk and a target chunk shifted by one
    character (standard "predict the next character" setup)."""
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in starts])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train the tiny model on real text.")
    parser.add_argument("--data", default="data/sample.txt", help="path to a plain text file to train on")
    parser.add_argument("--steps", type=int, default=1500, help="number of training steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--eval-every", type=int, default=100, help="print loss every N steps")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = CharTokenizer()
    MODEL_CONFIG.vocab_size = tokenizer.vocab_size

    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    if len(ids) < MODEL_CONFIG.block_size + 1:
        raise ValueError(
            f"Training file only has {len(ids)} usable characters, need at least "
            f"{MODEL_CONFIG.block_size + 1}. Use a longer text file."
        )
    data = torch.tensor(ids, dtype=torch.long)
    print(f"Loaded {len(ids):,} characters from {args.data}")

    # 90/10 train/validation split so we can sanity-check it's actually learning
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    if len(val_data) < MODEL_CONFIG.block_size + 1:
        # tiny files: just reuse train data for validation rather than crashing
        val_data = train_data

    torch.manual_seed(42)
    model = TinyGPT(MODEL_CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_time = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train_data, MODEL_CONFIG.block_size, args.batch_size, device)
        _, loss = model.forward_train(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, MODEL_CONFIG.block_size, args.batch_size, device)
                _, val_loss = model.forward_train(vx, vy)
            elapsed = time.time() - start_time
            print(f"step {step:5d}/{args.steps}  train_loss {loss.item():.3f}  "
                  f"val_loss {val_loss.item():.3f}  ({elapsed:.0f}s elapsed)")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nSaved trained weights to {CHECKPOINT_PATH}")
    print("Run `python run.py` (or `run_demo.py`) as usual — the nodes will load these trained weights.")

    # Quick sanity-check sample generation, single-process, no networking involved.
    print("\n--- Quick local sample (not going through the node network) ---")
    model.eval()
    prompt = "The "
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(150):
            context = idx[:, -MODEL_CONFIG.block_size:]
            x = model.forward_embed(context)
            x = model.forward_blocks(x, 0, MODEL_CONFIG.n_layer)
            logits = model.forward_head(x)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    print(tokenizer.decode(idx[0].tolist()))
    print("--- end sample ---")


if __name__ == "__main__":
    main()
