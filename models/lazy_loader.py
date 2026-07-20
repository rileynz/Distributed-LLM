"""
Lazy, layer-sliced loading for HuggingFace checkpoints.

The straightforward path (hf_wrapper.load_hf_model) loads the ENTIRE
model into memory, and only afterward does each node keep just its own
layer range. That's fine for small models but wasteful for anything
large: every single node briefly needs enough RAM to hold the *whole*
checkpoint, even if it only keeps a fraction of it afterward — which
defeats a lot of the point of sharding across weak devices.

This module instead builds an empty model skeleton (near-zero memory,
via accelerate's init_empty_weights) and reads ONLY the tensors this
node's own layer range actually needs, straight off disk via
safetensors' lazy tensor access — nothing else is ever read into RAM.

Scope, deliberately: this only handles the plain full-precision load.
Quantization continues to use the existing full-load-then-slice path
in hf_wrapper.py — combining bitsandbytes quantization with hand-picked
tensor loading is a real can of worms (bnb needs to control how a
Linear layer's weights are packed, not just where they're copied from)
and isn't worth the added risk here. See the README for more.

Falls back cleanly by raising LazyLoadUnsupported whenever the
checkpoint isn't in a format this supports (e.g. an older repo that
only ships pytorch_model.bin) — callers should catch this and fall
back to the full-load path rather than fail node startup.
"""

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open


class LazyLoadUnsupported(Exception):
    """This checkpoint/arch can't be lazily loaded — caller should fall
    back to loading the full model normally."""


# Where things live in each architecture's checkpoint tensor names.
# {} is replaced with the layer index. Prefixes that don't match
# anything in a given checkpoint (e.g. OPT's optional project_in/out,
# only present when word_embed_proj_dim != hidden_size) are simply a
# no-op — nothing is required to exist for every prefix.
_ARCH_SPECS = {
    "gpt2": {
        "layer_prefix":   "transformer.h.{}.",
        "first_prefixes": ["transformer.wte.", "transformer.wpe."],
        "last_prefixes":  ["transformer.ln_f.", "lm_head."],
        "embed_key":      "transformer.wte.weight",
        "lm_head_key":    "lm_head.weight",
    },
    "opt": {
        "layer_prefix":   "model.decoder.layers.{}.",
        "first_prefixes": ["model.decoder.embed_tokens.", "model.decoder.embed_positions.",
                            "model.decoder.project_in."],
        "last_prefixes":  ["model.decoder.final_layer_norm.", "model.decoder.project_out.",
                            "lm_head."],
        "embed_key":      "model.decoder.embed_tokens.weight",
        "lm_head_key":    "lm_head.weight",
    },
    "llama": {
        "layer_prefix":   "model.layers.{}.",
        "first_prefixes": ["model.embed_tokens."],
        "last_prefixes":  ["model.norm.", "lm_head."],
        "embed_key":      "model.embed_tokens.weight",
        "lm_head_key":    "lm_head.weight",
    },
}


def supports_lazy_loading(arch: str) -> bool:
    return arch in _ARCH_SPECS


def _find_checkpoint_dir(cache_dir: str) -> Path:
    """Each model in this project gets its own dedicated cache_dir (see
    models/catalogue.py) — exactly one checkpoint ever lives under it.
    HuggingFace's own caching nests the actual files a few directories
    down (models--org--name/snapshots/<hash>/...), so search for
    wherever config.json actually landed rather than assume a fixed
    depth."""
    root = Path(cache_dir)
    if not root.exists():
        raise LazyLoadUnsupported(f"Cache directory not found: {cache_dir}")
    matches = sorted(root.rglob("config.json"))
    if not matches:
        raise LazyLoadUnsupported(
            f"No config.json found under {cache_dir} — has this model finished downloading?"
        )
    return matches[0].parent


def _shard_map(checkpoint_dir: Path) -> dict:
    """Returns {tensor_name: shard_filename} for every tensor in the
    checkpoint, without reading any tensor data (index.json is tiny;
    safetensors headers are read separately from the tensor bytes).
    Raises LazyLoadUnsupported if there's no safetensors checkpoint here
    (e.g. an older repo that only ships pytorch_model.bin)."""
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"]

    single_path = checkpoint_dir / "model.safetensors"
    if single_path.exists():
        with safe_open(single_path, framework="pt") as f:
            return {name: "model.safetensors" for name in f.keys()}

    raise LazyLoadUnsupported(
        f"No .safetensors checkpoint found in {checkpoint_dir} "
        f"(older pytorch_model.bin-only repo?) — falling back to a full load."
    )


def _materialize(model, checkpoint_dir: Path, weight_map: dict,
                  needed_names: set, device: str) -> set:
    """Reads exactly `needed_names` off disk — grouped by shard file so
    each shard is opened and mmap'd at most once — and copies them into
    the empty `model` skeleton. Returns the set of names actually found
    and loaded (a name absent from weight_map, e.g. an omitted tied
    lm_head, is silently skipped here; the caller handles that case)."""
    by_file = {}
    for name in needed_names:
        filename = weight_map.get(name)
        if filename is not None:
            by_file.setdefault(filename, []).append(name)

    loaded = set()
    for filename, names in by_file.items():
        with safe_open(checkpoint_dir / filename, framework="pt") as f:
            for name in names:
                tensor = f.get_tensor(name)
                set_module_tensor_to_device(model, name, device=device,
                                             value=tensor, dtype=torch.float32)
                loaded.add(name)
    return loaded


def load_layer_slice(hf_id: str, cache_dir: str, arch: str,
                      start_layer: int, end_layer: int, total_layers: int):
    """
    Builds a model of `arch`'s type where ONLY the tensors this node's
    layer range needs are real (plus embeddings if start_layer==0, plus
    final norm/lm_head if end_layer==total_layers) — everything else
    stays on the `meta` device: near-zero memory, never read from disk.

    Returns (model, config) — the same shape as hf_wrapper.load_hf_model,
    so the existing GPT2NodeModel/OPTNodeModel/LlamaNodeModel classes
    work completely unchanged: they just take references to whichever
    submodules are real for this node and never touch the rest.

    Raises LazyLoadUnsupported if this checkpoint/arch can't be lazily
    loaded — callers should catch this and fall back to a full load.
    """
    try:
        return _load_layer_slice_inner(hf_id, cache_dir, arch, start_layer, end_layer, total_layers)
    except LazyLoadUnsupported:
        raise
    except Exception as e:
        # Guarantee the contract: this function either succeeds or raises
        # LazyLoadUnsupported — never some other exception type a caller's
        # fallback logic might not be catching.
        raise LazyLoadUnsupported(f"Unexpected error during lazy loading of {hf_id}: {e}") from e


def required_tensor_names(weight_map: dict, config, arch: str,
                           start_layer: int, end_layer: int, total_layers: int) -> set:
    """
    Given a checkpoint's full weight_map (tensor name -> shard filename)
    and a node's layer assignment, returns the set of tensor NAMES that
    node actually needs — its own decoder layers, plus embeddings if
    it's the first node, plus final norm/lm_head if it's the last,
    correctly following tied-embedding checkpoints back to whichever
    tensor actually holds the data.

    Pure and local: no disk or network access, just set/string logic.
    Used both to decide which shard FILES to download (only this node's
    needed tensors, not the whole checkpoint) and which tensors to
    actually read off already-downloaded shards (load_layer_slice).
    """
    spec = _ARCH_SPECS.get(arch)
    if spec is None:
        raise LazyLoadUnsupported(f"Lazy loading not implemented for arch '{arch}'.")

    is_first = start_layer == 0
    is_last  = end_layer == total_layers

    needed_prefixes = [spec["layer_prefix"].format(i) for i in range(start_layer, end_layer)]
    if is_first:
        needed_prefixes += spec["first_prefixes"]
    if is_last:
        needed_prefixes += spec["last_prefixes"]

    needed = {name for name in weight_map if any(name.startswith(p) for p in needed_prefixes)}

    lm_head_key = spec["lm_head_key"]
    if is_last and lm_head_key not in weight_map and getattr(config, "tie_word_embeddings", False):
        # Tied embeddings (GPT-2, OPT by default): lm_head.weight isn't a
        # separate tensor in the checkpoint at all — the node that needs
        # it actually needs the embedding tensor instead.
        embed_key = spec["embed_key"]
        if embed_key in weight_map:
            needed.add(embed_key)

    return needed


def download_needed_shards(hf_id: str, cache_dir: str, arch: str,
                            start_layer: int, end_layer: int, total_layers: int,
                            progress_cb=None):
    """
    Downloads only what a node with this layer assignment actually
    needs: every non-weight file (tokenizer, config — small, every node
    needs these regardless) plus, for sharded safetensors checkpoints,
    only the specific shard files containing this node's own layers —
    not the whole multi-GB checkpoint.

    Falls back to downloading every weight file if the checkpoint isn't
    sharded (a single model.safetensors — nothing to select from), the
    architecture isn't one lazy loading understands, or anything about
    figuring out the subset goes wrong. Same end result either way —
    this only ever changes how much gets downloaded, never correctness.

    progress_cb(done, total, filename), if given, is called after each
    individual file finishes.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    import json as _json

    all_files  = list_repo_files(hf_id)
    index_name = "model.safetensors.index.json"

    # Prefer safetensors; only fall back to legacy formats if a repo has
    # no safetensors at all. Some older repos ship both a .safetensors
    # and a legacy pytorch_model.bin version of the same weights for
    # backwards compatibility — without this, we'd download both.
    safetensor_files = {f for f in all_files if f.endswith(".safetensors") and "index" not in f}
    legacy_files      = {f for f in all_files if f.endswith((".bin", ".h5", ".msgpack"))}
    if safetensor_files:
        weight_files, redundant = safetensor_files, legacy_files
    else:
        weight_files, redundant = legacy_files, set()

    # Only fetch the small handful of files loading actually needs
    # (tokenizer + config). Many HF repos also ship alternate export
    # formats alongside the real weights — TFLite, CoreML, ONNX, Rust
    # (rust_model.ot), plus READMEs/.gitattributes/multiple
    # generation_config variants for those other runtimes. None of that
    # is ever read by this project (only transformers + safetensors
    # are used), but naively treating "anything that isn't a weight
    # file" as an aux file downloads all of it too — for some repos
    # that's an order of magnitude more data than the actual checkpoint.
    # This allow-list keeps the "download only what's needed" promise
    # for real: matched on basename since these can be nested (e.g. a
    # repo occasionally ships tokenizer files inside a subfolder).
    _NEEDED_AUX_BASENAMES = {
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "vocab.txt",
        "special_tokens_map.json", "added_tokens.json",
        "spm.model", "tokenizer.model", "sentencepiece.bpe.model",
    }
    from pathlib import PurePosixPath
    aux_files = [f for f in all_files
                 if f not in weight_files and f not in redundant and f != index_name
                 and PurePosixPath(f).name in _NEEDED_AUX_BASENAMES]

    to_fetch = sorted(weight_files)  # safe default: everything

    if index_name in all_files:
        # Needed locally regardless of whether this arch supports lazy
        # loading: even the full-load fallback needs the index to map
        # tensors to shard files correctly for a multi-shard checkpoint.
        index_path = hf_hub_download(hf_id, index_name, cache_dir=cache_dir)
        if supports_lazy_loading(arch):
            try:
                from transformers import AutoConfig
                with open(index_path) as f:
                    weight_map = _json.load(f)["weight_map"]
                config = AutoConfig.from_pretrained(hf_id, cache_dir=cache_dir)
                needed_names = required_tensor_names(weight_map, config, arch,
                                                      start_layer, end_layer, total_layers)
                to_fetch = sorted({weight_map[n] for n in needed_names if n in weight_map})
            except Exception:
                # Couldn't figure out the subset for some reason — fetch
                # everything rather than risk a model missing tensors.
                to_fetch = sorted(weight_files)

    all_to_fetch = aux_files + to_fetch
    total = len(all_to_fetch) + (1 if index_name in all_files else 0)
    done  = 1 if index_name in all_files else 0
    if progress_cb and done:
        progress_cb(done, total, index_name)
    for fname in all_to_fetch:
        hf_hub_download(hf_id, fname, cache_dir=cache_dir)
        done += 1
        if progress_cb:
            progress_cb(done, total, fname)


def _load_layer_slice_inner(hf_id: str, cache_dir: str, arch: str,
                             start_layer: int, end_layer: int, total_layers: int):
    spec = _ARCH_SPECS.get(arch)
    if spec is None:
        raise LazyLoadUnsupported(f"Lazy loading not implemented for arch '{arch}'.")

    from transformers import AutoConfig, AutoModelForCausalLM

    # Resolve everything from the already-downloaded local checkpoint dir —
    # never touch the network here. hf_id is only used for clearer error
    # messages; the actual config comes from disk.
    checkpoint_dir = _find_checkpoint_dir(cache_dir)
    try:
        config = AutoConfig.from_pretrained(checkpoint_dir)
    except Exception as e:
        raise LazyLoadUnsupported(f"Could not read config for {hf_id} at {checkpoint_dir}: {e}") from e

    weight_map = _shard_map(checkpoint_dir)

    with init_empty_weights(include_buffers=False):
        # include_buffers=False is explicit, not just the default: it
        # guarantees small non-learned buffers (e.g. rotary embedding's
        # inv_freq, which is computed from config rather than read from
        # the checkpoint) are always real immediately, regardless of
        # which accelerate version/default is installed. These are a
        # handful of floats each — keeping them off meta costs nothing
        # meaningful and avoids "Cannot copy out of meta tensor" crashes
        # on machines where accelerate's default differs from this one.
        model = AutoModelForCausalLM.from_config(config)

    is_last = end_layer == total_layers
    needed_names = required_tensor_names(weight_map, config, arch, start_layer, end_layer, total_layers)
    _materialize(model, checkpoint_dir, weight_map, needed_names, device="cpu")

    # Tied embeddings: the tensor we pulled above under the embedding's
    # own name needs to end up living at lm_head.weight too.
    lm_head_key = spec["lm_head_key"]
    if is_last and getattr(config, "tie_word_embeddings", False) and lm_head_key not in weight_map:
        embed_key = spec["embed_key"]
        embed_filename = weight_map.get(embed_key)
        if embed_filename is None:
            raise LazyLoadUnsupported(
                f"Model has tied embeddings but expected tensor '{embed_key}' "
                f"was not found in the checkpoint."
            )
        with safe_open(checkpoint_dir / embed_filename, framework="pt") as f:
            tensor = f.get_tensor(embed_key)
        set_module_tensor_to_device(model, lm_head_key, device="cpu",
                                     value=tensor, dtype=torch.float32)

    # Safety net: `_materialize` only copies tensors whose names it found
    # in the checkpoint's weight_map — if a name that should exist for one
    # of this node's own layers is silently missing (partial/incomplete
    # index, stale cache, a mismatch between what was downloaded and what
    # was assigned), nothing above raises an exception; the model would
    # otherwise come back looking "loaded" while still carrying real
    # meta-device placeholders that only blow up later, mid-request, deep
    # inside a forward pass. Check for that explicitly here so it's
    # caught now and converted into the normal LazyLoadUnsupported ->
    # full-checkpoint-fallback path instead.
    still_meta = [name for name, p in model.named_parameters() if p.is_meta]
    if still_meta:
        raise LazyLoadUnsupported(
            f"{len(still_meta)} tensor(s) needed for layers {start_layer}-{end_layer - 1} "
            f"were never found while lazy-loading (e.g. {still_meta[0]}) — the local "
            f"checkpoint may be incomplete or stale."
        )

    model.eval()
    return model, config
