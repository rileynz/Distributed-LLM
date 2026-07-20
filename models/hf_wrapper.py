"""
Multi-architecture HuggingFace model wrapper.
Supports GPT-2 family, OPT, and Llama-family (TinyLlama, Mistral, Phi-3, etc.)
All expose the same interface as TinyGPT so nodes need zero changes.

For Llama-family models, RoPE position embeddings (cos/sin) are computed
on the first node and passed through the message chain alongside the
hidden state tensor.

KV caching: forward_blocks accepts a `past_kv` (a transformers Cache
instance, e.g. DynamicCache) holding this node's own layers' cached
keys/values for one in-flight request, and hands back the same
(internally mutated) object as `present_kv`. The cache itself is created,
stored, and evicted by node/server.py's KVCacheStore — this file only
needs to pass it through to each HF block. Caches never cross the wire;
each node keeps its own locally, keyed by request_id.

Lazy loading: load_hf_model tries models/lazy_loader.py first (only
this node's own layers are ever read off disk) whenever quantization
is "none" and the architecture supports it, falling back to loading
the full checkpoint and keeping just this node's slice otherwise —
same result either way, just a memory/disk-I/O difference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*quantize_per_tensor.*")


def _resolve_quantization(quantization: str):
    """
    Validates the requested quantization against actual hardware.
    Returns (quantization_config_or_None, effective_level). bitsandbytes
    quantization needs a CUDA GPU; on CPU-only nodes this returns
    (None, "none") and the caller instead applies CPU-native dynamic
    int8 quantization afterward (see _apply_cpu_int8 / the
    cpu_quantize_to param on load_hf_model) — a real, if less
    aggressive, alternative that needs no GPU and no extra packages.
    """
    if quantization not in ("int8", "int4"):
        return None, "none"

    if not torch.cuda.is_available():
        return None, "none"

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("  Quantization requires the 'bitsandbytes' and 'accelerate' packages "
              "(pip install bitsandbytes accelerate) — loading full precision instead.")
        return None, "none"

    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True), "int8"
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ), "int4"


def _apply_cpu_int8(module: nn.Module) -> nn.Module:
    """
    CPU-native dynamic int8 quantization — PyTorch's own built-in
    mechanism (torch.ao.quantization.quantize_dynamic), no GPU and no
    extra dependencies needed. Conceptually the same idea CPU-focused
    projects like llama.cpp/ggml use (weights stored as int8 with a
    per-tensor scale, dequantized on the fly per matmul); this is our
    own use of PyTorch's stdlib implementation of that idea, not code
    from either project.

    Only nn.Linear layers are touched — embeddings and norm layers stay
    full precision, since they're a small fraction of total params for
    a transformer and quantizing them risks more accuracy loss than the
    memory savings are worth.

    Applied to the already-sliced per-node module (this node's own
    layers only, after lazy loading has done its job) rather than the
    full checkpoint before slicing — so the temporary full-precision
    memory this briefly needs is only this node's own share, not the
    whole model.

    Real int8, roughly halves this node's weight memory vs full
    precision. Not as small as true 4-bit, but that needs either a GPU
    (bitsandbytes) or GGUF-style kernels this project doesn't
    implement — CPU int8 is the honest, real option available here.
    """
    import torch.ao.quantization as tq
    # Dynamic quantization needs an actual quantized-kernel backend
    # available on this machine (x86/fbgemm on most PCs, qnnpack on
    # ARM) — pick whichever this machine actually supports rather than
    # trusting whatever PyTorch's global default happens to be.
    supported = torch.backends.quantized.supported_engines
    for engine in ("fbgemm", "x86", "qnnpack", "onednn"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            break
    else:
        raise RuntimeError(
            "no quantized CPU backend (fbgemm/x86/qnnpack/onednn) available "
            f"on this machine (supported: {supported})"
        )
    return tq.quantize_dynamic(module, {nn.Linear}, dtype=torch.qint8)


def detect_arch_from_config(config) -> str:
    """Same architecture-family detection as detect_arch, but works from
    just the downloaded config — available before any weights are
    loaded, which is what lets load_hf_model decide whether lazy
    loading applies before it's built anything."""
    names = getattr(config, "architectures", None) or []
    name = names[0] if names else ""
    if ("Llama" in name or "TinyLlama" in name or "Mistral" in name or "Phi" in name
            or "Qwen" in name or "Yi" in name or "InternLM" in name or "DeepSeek" in name):
        # All of these mirror Llama's module layout (model.embed_tokens,
        # model.layers, model.rotary_emb, model.norm, lm_head) even
        # though their architecture name doesn't literally say "Llama" —
        # Qwen2ForCausalLM in particular is what Qwen2.5 (including the
        # 14B model in this catalogue) reports, and misdetecting it as
        # "gpt2" would crash immediately (GPT2NodeModel looks for
        # model.transformer.h, which doesn't exist on these).
        return "llama"
    if "OPT" in name:
        return "opt"
    return "gpt2"  # GPT2, GPTNeo, DistilGPT2, etc.


def load_hf_model(hf_id: str, cache_dir: str,
                   start_layer: int, end_layer: int, total_layers: int,
                   arch: str = "auto", quantization: str = "none"):
    """Loads just what this node needs.

    Tries the memory-efficient lazy per-layer path first (models/
    lazy_loader.py) — only this node's own layers are ever read off
    disk — when quantization is "none" and the architecture supports
    it. Falls back automatically to loading the full checkpoint and
    keeping the relevant slice for anything that path doesn't handle:
    quantized loads, unsupported architectures, or older checkpoints
    without safetensors. Either way the result is equivalent — this is
    purely a memory/disk-I/O optimization, never a behavior change.

    quantization: "none" | "int8" | "int4" — on a CUDA GPU, applied as
    bitsandbytes quantization at load time. On a CPU-only node, "int8"
    is instead applied afterward as real PyTorch dynamic int8
    quantization on this node's own sliced-out layers (see
    _apply_cpu_int8 in HFNodeModel.__init__) — "int4" on CPU falls back
    to that same CPU int8 path with a printed note, since there's no
    CPU-native 4-bit kernel here without a GPU.

    Returns (model, config, arch).
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(hf_id, cache_dir=cache_dir)
    if arch == "auto":
        arch = detect_arch_from_config(config)

    quant_config, effective = _resolve_quantization(quantization)

    if quant_config is None:
        from models.lazy_loader import load_layer_slice, LazyLoadUnsupported, supports_lazy_loading
        if supports_lazy_loading(arch):
            try:
                model, config = load_layer_slice(
                    hf_id, cache_dir, arch, start_layer, end_layer, total_layers)
                print(f"  Loaded only this node's layers ({start_layer}-{end_layer-1} of "
                      f"{total_layers}) — lazy loading active, rest never read from disk.")
                return model, config, arch
            except LazyLoadUnsupported as e:
                print(f"  Lazy loading unavailable ({e}) — loading the full checkpoint instead.")

    # Fallback: load the whole checkpoint, then the caller keeps only
    # this node's slice (still correct, just needs more RAM/disk I/O).
    from transformers import AutoModelForCausalLM
    load_kwargs = dict(cache_dir=cache_dir, low_cpu_mem_usage=True)
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"
        print(f"  Loading with {effective} quantization (bitsandbytes)...")
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    model.eval()
    return model, config, arch


def detect_arch(model) -> str:
    cls = type(model).__name__
    if ("Llama" in cls or "TinyLlama" in cls or "Mistral" in cls or "Phi" in cls
            or "Qwen" in cls or "Yi" in cls or "InternLM" in cls or "DeepSeek" in cls):
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
    def forward_embed(self, idx, past_length=0):
        B, T = idx.shape
        pos = torch.arange(past_length, past_length + T, device=idx.device).unsqueeze(0)
        return self.wte(idx) + self.wpe(pos), None  # None = no extra state

    @torch.no_grad()
    def forward_blocks(self, x, extra, past_kv=None, use_cache=False):
        """
        past_kv: a transformers Cache instance (e.g. DynamicCache) holding
        this node's own layers' cached keys/values from earlier calls for
        the same request, or None to run without a cache (legacy path).
        The block mutates past_kv in place, so we just hand the same
        object back as `present_kv` — the caller (node/server.py) stores
        it under the request's id for next time.
        """
        for block in self.blocks:
            out = block(x, past_key_values=past_kv, use_cache=use_cache)
            x = out[0] if isinstance(out, tuple) else out
        return x, extra, past_kv

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
    def forward_embed(self, idx, past_length=0):
        B, T  = idx.shape
        pos   = torch.arange(past_length, past_length + T, device=idx.device).unsqueeze(0)
        # OPTLearnedPositionalEmbedding computes its own position_ids from
        # attention_mask when position_ids isn't given — we pass explicit
        # absolute position_ids instead so it's correct whether this is a
        # fresh prompt (past_length=0) or a cached decode step (past_length>0).
        # attention_mask is required by the signature but unused in that case.
        dummy_mask = torch.ones(B, past_length + T, dtype=torch.long, device=idx.device)
        pos_emb = self.embed_positions(dummy_mask, position_ids=pos)
        x = self.embed_tokens(idx) + pos_emb
        return x, None

    @torch.no_grad()
    def forward_blocks(self, x, extra, past_kv=None, use_cache=False):
        for block in self.blocks:
            out = block(x, past_key_values=past_kv, use_cache=use_cache)
            x   = out[0] if isinstance(out, tuple) else out
        return x, extra, past_kv

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
    def forward_embed(self, idx, past_length=0):
        x   = self.embed_tokens(idx)
        pos = torch.arange(past_length, past_length + idx.shape[1],
                            device=idx.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, pos)
        return x, (cos, sin, pos)  # carry RoPE + absolute positions through the chain

    @torch.no_grad()
    def forward_blocks(self, x, extra, past_kv=None, use_cache=False):
        cos, sin, pos = extra
        for block in self.blocks:
            out = block(x, position_ids=pos, position_embeddings=(cos, sin),
                        past_key_values=past_kv, use_cache=use_cache)
            x   = out[0] if isinstance(out, tuple) else out
        return x, extra, past_kv  # pass RoPE + cache on to the next node

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
                 arch: str = "auto", quantization: str = "none"):
        super().__init__()
        print(f"  Loading {hf_id} (layers {start_layer}-{end_layer-1}) from {cache_dir} ...")
        full_model, _, arch = load_hf_model(
            hf_id, cache_dir, start_layer, end_layer, total_layers,
            arch=arch, quantization=quantization)
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

        # bitsandbytes (GPU) quantization, if any, already happened
        # inside load_hf_model above via quantization_config. On a
        # CPU-only node, apply real CPU-native dynamic int8 quantization
        # here instead, to just this node's own already-sliced layers —
        # applying it post-slice (rather than to the full checkpoint
        # beforehand) means the temporary full-precision memory this
        # needs is only this node's share, not the whole model.
        if quantization in ("int8", "int4") and not torch.cuda.is_available():
            try:
                if quantization == "int4":
                    print("  int4 quantization needs a CUDA GPU — using CPU int8 dynamic "
                          "quantization instead (real, roughly half the memory of full "
                          "precision — not as small as true int4, but no GPU needed).")
                else:
                    print("  Applying CPU int8 dynamic quantization to this node's layers...")
                self._inner = _apply_cpu_int8(self._inner)
            except Exception as e:
                print(f"  CPU int8 quantization unavailable on this machine ({e}) — "
                      f"continuing with full precision instead.")

    # ── TinyGPT-compatible interface ─────────────────────────────────────────
    # `extra` carries architecture-specific state that must travel alongside
    # the hidden state (None for GPT-2/OPT, (cos, sin, position_ids) for
    # Llama/Mistral/Phi-3's RoPE). `past_kv` / `present_kv` carry this NODE's
    # own KV cache for this request (a transformers Cache instance) — it
    # never leaves the node, only x/extra travel across the wire.

    @torch.no_grad()
    def forward_embed(self, idx, past_length=0):
        return self._inner.forward_embed(idx, past_length=past_length)

    @torch.no_grad()
    def forward_blocks(self, x, extra, past_kv=None, use_cache=False):
        return self._inner.forward_blocks(x, extra, past_kv=past_kv, use_cache=use_cache)

    @torch.no_grad()
    def forward_head(self, x, extra):
        return self._inner.forward_head(x, extra)
