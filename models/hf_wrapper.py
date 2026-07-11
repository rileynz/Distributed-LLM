"""
Multi-architecture HuggingFace model wrapper.
Supports GPT-2 family, OPT, and Llama-family (TinyLlama etc.)
All expose the same interface as TinyGPT so nodes need zero changes.

For Llama-family models, RoPE position embeddings (cos/sin) are computed
on the first node and passed through the message chain alongside the
hidden state tensor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")


def load_hf_model(hf_id: str, cache_dir: str):
    """Loads the full model, auto-detecting the right class."""
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(hf_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, cache_dir=cache_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, config


def detect_arch(model) -> str:
    cls = type(model).__name__
    if "Llama" in cls or "TinyLlama" in cls or "Mistral" in cls or "Phi" in cls:
        return "llama"
    if "OPT" in cls:
        return "opt"
    return "gpt2"  # GPT2, GPTNeo, DistilGPT2, etc.


# ── GPT-2 / GPT-Neo wrapper ───────────────────────────────────────────────────

class GPT2NodeModel(nn.Module):
    def __init__(self, model, start_layer: int, end_layer: int, total_layers: int):
        super().__init__()
        self._is_first = start_layer == 0
        self._is_last  = end_layer == total_layers
        self.blocks    = nn.ModuleList(
            [model.transformer.h[i] for i in range(start_layer, end_layer)]
        )
        if self._is_first:
            self.wte = model.transformer.wte
            self.wpe = model.transformer.wpe
        if self._is_last:
            self.ln_f    = model.transformer.ln_f
            self.lm_head = model.lm_head

    @torch.no_grad()
    def forward_embed(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        return self.wte(idx) + self.wpe(pos), None  # None = no extra state

    @torch.no_grad()
    def forward_blocks(self, x, extra, start, end):
        for block in self.blocks:
            out = block(x, use_cache=False)
            x = out[0] if isinstance(out, tuple) else out
        return x, extra

    @torch.no_grad()
    def forward_head(self, x, extra):
        return self.lm_head(self.ln_f(x))


# ── OPT wrapper ───────────────────────────────────────────────────────────────

class OPTNodeModel(nn.Module):
    def __init__(self, model, start_layer: int, end_layer: int, total_layers: int):
        super().__init__()
        dec = model.model.decoder
        self._is_first = start_layer == 0
        self._is_last  = end_layer == total_layers
        self.blocks    = nn.ModuleList(
            [dec.layers[i] for i in range(start_layer, end_layer)]
        )
        if self._is_first:
            self.embed_tokens     = dec.embed_tokens
            self.embed_positions  = dec.embed_positions
        if self._is_last:
            self.final_layer_norm = dec.final_layer_norm
            self.lm_head          = model.lm_head

    @torch.no_grad()
    def forward_embed(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x   = self.embed_tokens(idx) + self.embed_positions(pos)
        return x, None

    @torch.no_grad()
    def forward_blocks(self, x, extra, start, end):
        for block in self.blocks:
            out = block(x, use_cache=False)
            x   = out[0] if isinstance(out, tuple) else out
        return x, extra

    @torch.no_grad()
    def forward_head(self, x, extra):
        return self.lm_head(self.final_layer_norm(x))


# ── Llama / TinyLlama / Mistral wrapper ──────────────────────────────────────

class LlamaNodeModel(nn.Module):
    """
    Llama uses Rotary Position Embeddings (RoPE) computed once per sequence
    and passed into every block. We compute cos/sin on the first node and
    carry them in `extra` through the message chain.
    """

    def __init__(self, model, start_layer: int, end_layer: int, total_layers: int):
        super().__init__()
        self._is_first = start_layer == 0
        self._is_last  = end_layer == total_layers
        self.blocks    = nn.ModuleList(
            [model.model.layers[i] for i in range(start_layer, end_layer)]
        )
        if self._is_first:
            self.embed_tokens = model.model.embed_tokens
            self.rotary_emb   = model.model.rotary_emb
        if self._is_last:
            self.norm    = model.model.norm
            self.lm_head = model.lm_head

    @torch.no_grad()
    def forward_embed(self, idx):
        x   = self.embed_tokens(idx)
        pos = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, pos)
        return x, (cos, sin)  # carry RoPE through the chain

    @torch.no_grad()
    def forward_blocks(self, x, extra, start, end):
        cos, sin = extra
        for block in self.blocks:
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            out = block(x, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)
            x   = out[0] if isinstance(out, tuple) else out
        return x, extra  # pass RoPE on to the next node

    @torch.no_grad()
    def forward_head(self, x, extra):
        return self.lm_head(self.norm(x))


# ── Unified factory ───────────────────────────────────────────────────────────

_ARCH_MAP = {"gpt2": GPT2NodeModel, "opt": OPTNodeModel, "llama": LlamaNodeModel}


class HFNodeModel(nn.Module):
    """
    Public interface — matches TinyGPT exactly so nodes need no changes.
    forward_embed / forward_blocks / forward_head all take and return the
    same tensor shapes as TinyGPT; `extra` state (e.g. RoPE cos/sin for
    Llama) is carried transparently through the call chain.
    """

    def __init__(self, hf_id: str, cache_dir: str,
                 start_layer: int, end_layer: int, total_layers: int,
                 arch: str = "auto"):
        super().__init__()
        print(f"  Loading {hf_id} (layers {start_layer}-{end_layer-1}) from {cache_dir} ...")
        full_model, _ = load_hf_model(hf_id, cache_dir)

        if arch == "auto":
            arch = detect_arch(full_model)
        print(f"  Architecture detected: {arch}")

        cls = _ARCH_MAP.get(arch)
        if cls is None:
            raise ValueError(f"Unsupported architecture '{arch}'. "
                             f"Supported: {list(_ARCH_MAP)}")
        self._inner = cls(full_model, start_layer, end_layer, total_layers)
        self._arch  = arch

        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── TinyGPT-compatible interface ─────────────────────────────────────────
    # `extra` carries architecture-specific state (None for GPT-2/OPT,
    # (cos, sin) tensors for Llama). It travels in the protocol message
    # alongside `data` under the key "extra".

    @torch.no_grad()
    def forward_embed(self, idx):
        return self._inner.forward_embed(idx)

    @torch.no_grad()
    def forward_blocks(self, x, extra, start, end):
        return self._inner.forward_blocks(x, extra, start, end)

    @torch.no_grad()
    def forward_head(self, x, extra):
        return self._inner.forward_head(x, extra)
