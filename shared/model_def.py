"""
Defines the tiny GPT-style model used by the demo, plus a simple
character-level tokenizer. Every node imports this file so they all
agree on the exact same architecture.

This is intentionally small (a few hundred thousand parameters) so it
can run instantly on a laptop CPU with no GPU and no downloads. It has
NOT been trained, so the text it generates will be gibberish — the
point of this v1 is to prove the distributed *mechanism* works
(splitting a model's layers across separate processes/machines), not
to produce good text. Swap in a real pretrained model later (see
README.md) once the plumbing is solid.
"""

import math
import string

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Maps characters to integer ids and back. Vocab = printable ASCII."""

    def __init__(self):
        chars = sorted(set(string.printable))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        # Unknown characters are skipped rather than crashing the demo.
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i in self.itos)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

class ModelConfig:
    def __init__(self, vocab_size, n_layer=4, n_embd=64, n_head=4, block_size=128):
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        assert n_layer >= 1
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size


# ---------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, past_kv=None):
        """
        past_kv: optional (k, v) tensors cached from previous calls for this
        same request, each shaped (B, n_head, past_T, head_dim). When given,
        the new keys/values are appended to them before attending, so `x`
        only needs to contain the *new* token(s) — not the whole sequence.
        Returns (output, present_kv) where present_kv is the (possibly
        concatenated) (k, v) pair to pass back in on the next call.
        """
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v)

        Tq, Tk   = q.size(2), k.size(2)
        past_len = Tk - Tq
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        # Query position i (0-indexed within this chunk) sits at absolute
        # position (past_len + i) and may attend to any key position <= it.
        row = torch.arange(Tq, device=x.device).unsqueeze(1)
        col = torch.arange(Tk, device=x.device).unsqueeze(0)
        causal_mask = col <= (row + past_len)
        att = att.masked_fill(~causal_mask.view(1, 1, Tq, Tk), float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.proj(out), present_kv


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """One transformer layer. This is the unit we split across nodes."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv


class TinyGPT(nn.Module):
    """
    A minimal GPT. Exposes three separate methods instead of one forward()
    so that a node can run just the slice it's responsible for:

      forward_embed(idx, past_length)        -> token ids to vectors  (first node)
      forward_blocks(x, lo, hi, past_kvs)     -> runs layers lo..hi    (any node)
      forward_head(x)                        -> vectors to logits    (last node)

    past_length / past_kvs enable KV caching: once a request's prompt has
    been processed once, later steps only need to send the single newest
    token — each node keeps that request's own past keys/values locally
    (see node/server.py's KVCacheStore) instead of recomputing attention
    over the whole growing sequence every step.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList(
            [Block(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @torch.no_grad()
    def forward_embed(self, idx, past_length=0):
        B, T = idx.shape
        assert past_length + T <= self.config.block_size, (
            f"sequence length {past_length + T} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(past_length, past_length + T,
                            dtype=torch.long, device=idx.device).unsqueeze(0)
        return self.wte(idx) + self.wpe(pos)

    @torch.no_grad()
    def forward_blocks(self, x, start_layer, end_layer, past_kvs=None):
        """Runs layers [start_layer, end_layer) — end_layer is exclusive.

        past_kvs: optional {layer_idx: (k, v)} dict of this node's own
        cached keys/values from previous calls for the same request.
        Returns (x, present_kvs) — present_kvs has the same shape and
        should be stored by the caller and passed back in next time.
        """
        present_kvs = {}
        for i in range(start_layer, end_layer):
            past_kv = past_kvs.get(i) if past_kvs else None
            x, present_kv = self.h[i](x, past_kv=past_kv)
            present_kvs[i] = present_kv
        return x, present_kvs

    @torch.no_grad()
    def forward_head(self, x):
        x = self.ln_f(x)
        return self.head(x)

    def forward_train(self, idx, targets):
        """
        Full forward pass WITH gradients, for training. Not used by the
        nodes (they use the no_grad methods above for inference) — this
        is only used by train.py.

        idx:     (B, T) input token ids
        targets: (B, T) the token that should come after each position
        Returns: (logits, loss)
        """
        x = self.forward_embed(idx)  # note: forward_embed is itself @torch.no_grad()
        # so we recompute embeddings here without the no_grad decorator interfering
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)

        for block in self.h:
            x, _ = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
