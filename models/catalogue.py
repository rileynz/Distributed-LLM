"""
Catalogue of downloadable pretrained models.

Architecture families supported by hf_wrapper.py:
  "gpt2"  — GPT-2, DistilGPT-2, GPT-Neo (share the transformer.h layout)
  "opt"   — Meta OPT family (model.decoder.layers layout)
  "llama" — Llama / TinyLlama / Phi (RoPE positional encoding, model.layers)

Adding a new model: add an entry below. If the architecture family is
already listed above, it will work with no other changes.
"""

from dataclasses import dataclass, field


@dataclass
class ModelEntry:
    id:                   str
    name:                 str
    description:          str
    hf_id:                str
    arch:                 str        # "gpt2" | "opt" | "llama"
    download_gb:          float
    min_ram_per_node_gb:  float
    total_ram_gb:         float
    n_layers:             int
    min_nodes:            int
    quality:              int        # 1-5
    min_score:            int        # minimum network benchmark score to run well
    notes:                str = ""
    tags:                 list[str] = field(default_factory=list)
    requires_auth:        bool = False  # True if HF login needed (Llama 2 etc.)


CATALOGUE: list[ModelEntry] = [

    # ── GPT-2 family ────────────────────────────────────────────────────────
    ModelEntry(
        id="distilgpt2", name="DistilGPT-2",
        description="Smallest real LM. Runs on any laptop, any single node.",
        hf_id="distilgpt2", arch="gpt2",
        download_gb=0.35, min_ram_per_node_gb=0.8, total_ram_gb=1.0,
        n_layers=6, min_nodes=1, quality=1, min_score=0,
        notes="Good for confirming the pipeline works on limited hardware.",
        tags=["lightweight", "fast"],
    ),
    ModelEntry(
        id="gpt2", name="GPT-2 Small (124M)",
        description="The classic open LM. Real sentences, runs on most laptops.",
        hf_id="gpt2", arch="gpt2",
        download_gb=0.5, min_ram_per_node_gb=1.0, total_ram_gb=1.5,
        n_layers=12, min_nodes=1, quality=2, min_score=5,
        notes="Great starting point. Grammatically correct output.",
        tags=["balanced", "popular"],
    ),
    ModelEntry(
        id="gpt2-medium", name="GPT-2 Medium (345M)",
        description="2× GPT-2 Small. Noticeably better paragraph coherence.",
        hf_id="gpt2-medium", arch="gpt2",
        download_gb=1.4, min_ram_per_node_gb=2.0, total_ram_gb=3.0,
        n_layers=24, min_nodes=2, quality=3, min_score=10,
        notes="First model where output reads like real writing.",
        tags=["balanced", "quality"],
    ),
    ModelEntry(
        id="gpt2-large", name="GPT-2 Large (774M)",
        description="4× GPT-2 Small. Strong creative writing.",
        hf_id="gpt2-large", arch="gpt2",
        download_gb=3.0, min_ram_per_node_gb=2.5, total_ram_gb=6.0,
        n_layers=36, min_nodes=2, quality=3, min_score=15,
        tags=["quality", "creative"],
    ),
    ModelEntry(
        id="gpt2-xl", name="GPT-2 XL (1.5B)",
        description="Best of the GPT-2 family.",
        hf_id="gpt2-xl", arch="gpt2",
        download_gb=6.0, min_ram_per_node_gb=4.0, total_ram_gb=10.0,
        n_layers=48, min_nodes=2, quality=4, min_score=20,
        tags=["quality", "powerful"],
    ),

    # ── GPT-Neo / EleutherAI ────────────────────────────────────────────────
    ModelEntry(
        id="gpt-neo-125m", name="GPT-Neo 125M",
        description="EleutherAI's small open model. Similar quality to GPT-2 Small.",
        hf_id="EleutherAI/gpt-neo-125m", arch="gpt2",
        download_gb=0.5, min_ram_per_node_gb=1.0, total_ram_gb=1.5,
        n_layers=12, min_nodes=1, quality=2, min_score=5,
        tags=["lightweight", "open"],
    ),
    ModelEntry(
        id="gpt-neo-1.3b", name="GPT-Neo 1.3B",
        description="EleutherAI 1.3B — significantly better than GPT-2 XL.",
        hf_id="EleutherAI/gpt-neo-1.3B", arch="gpt2",
        download_gb=5.0, min_ram_per_node_gb=4.0, total_ram_gb=8.0,
        n_layers=24, min_nodes=2, quality=4, min_score=25,
        notes="Trained on The Pile — much more diverse knowledge than GPT-2.",
        tags=["quality", "diverse"],
    ),
    ModelEntry(
        id="gpt-neo-2.7b", name="GPT-Neo 2.7B",
        description="EleutherAI 2.7B — strong open-source output quality.",
        hf_id="EleutherAI/gpt-neo-2.7B", arch="gpt2",
        download_gb=11.0, min_ram_per_node_gb=6.0, total_ram_gb=14.0,
        n_layers=32, min_nodes=3, quality=4, min_score=40,
        tags=["quality", "powerful"],
    ),

    # ── Meta OPT ────────────────────────────────────────────────────────────
    ModelEntry(
        id="opt-125m", name="OPT 125M",
        description="Meta's lightest OPT model. Good alternative to GPT-2 Small.",
        hf_id="facebook/opt-125m", arch="opt",
        download_gb=0.5, min_ram_per_node_gb=1.0, total_ram_gb=1.5,
        n_layers=12, min_nodes=1, quality=2, min_score=5,
        tags=["lightweight", "meta"],
    ),
    ModelEntry(
        id="opt-350m", name="OPT 350M",
        description="Meta OPT 350M — solid mid-range open model.",
        hf_id="facebook/opt-350m", arch="opt",
        download_gb=1.4, min_ram_per_node_gb=2.0, total_ram_gb=3.0,
        n_layers=24, min_nodes=1, quality=2, min_score=10,
        tags=["balanced", "meta"],
    ),
    ModelEntry(
        id="opt-1.3b", name="OPT 1.3B",
        description="Meta OPT 1.3B — competitive with GPT-Neo 1.3B.",
        hf_id="facebook/opt-1.3b", arch="opt",
        download_gb=5.0, min_ram_per_node_gb=4.0, total_ram_gb=8.0,
        n_layers=24, min_nodes=2, quality=4, min_score=25,
        tags=["quality", "meta"],
    ),
    ModelEntry(
        id="opt-2.7b", name="OPT 2.7B",
        description="Meta OPT 2.7B — strong reasoning and creative writing.",
        hf_id="facebook/opt-2.7b", arch="opt",
        download_gb=10.0, min_ram_per_node_gb=6.0, total_ram_gb=14.0,
        n_layers=32, min_nodes=3, quality=4, min_score=40,
        tags=["quality", "meta"],
    ),

    # ── TinyLlama / Llama-family ─────────────────────────────────────────────
    ModelEntry(
        id="tinyllama-1.1b", name="TinyLlama 1.1B",
        description="Best small model for the size. Modern architecture, trained on 3T tokens.",
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", arch="llama",
        download_gb=2.2, min_ram_per_node_gb=2.5, total_ram_gb=4.0,
        n_layers=22, min_nodes=1, quality=4, min_score=15,
        notes="Far better quality/size ratio than GPT-2 or OPT. Best choice for 1-2 nodes.",
        tags=["quality", "efficient", "recommended"],
    ),
]

CATALOGUE_BY_ID: dict[str, ModelEntry] = {m.id: m for m in CATALOGUE}


def get_model(model_id: str) -> ModelEntry | None:
    return CATALOGUE_BY_ID.get(model_id)


def list_models() -> list[ModelEntry]:
    return list(CATALOGUE)
