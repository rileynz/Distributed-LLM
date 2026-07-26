from __future__ import annotations


CATALOGUE = [
    {
        "id": "qwen3-0.6b",
        "name": "Qwen3 0.6B Q8",
        "hf": "Qwen/Qwen3-0.6B-GGUF:Q8_0",
        "size_gb": 0.64,
        "description": "Very fast testing model",
    },
    {
        "id": "qwen3-1.7b",
        "name": "Qwen3 1.7B Q8",
        "hf": "Qwen/Qwen3-1.7B-GGUF:Q8_0",
        "size_gb": 1.83,
        "description": "Small and responsive",
    },
    {
        "id": "qwen3-4b",
        "name": "Qwen3 4B Q4_K_M",
        "hf": "Qwen/Qwen3-4B-GGUF:Q4_K_M",
        "size_gb": 2.50,
        "description": "Good low-power default",
    },
    {
        "id": "qwen3-8b",
        "name": "Qwen3 8B Q4_K_M",
        "hf": "Qwen/Qwen3-8B-GGUF:Q4_K_M",
        "size_gb": 5.03,
        "description": "Best balance for a medium cluster",
    },
    {
        "id": "qwen3-14b",
        "name": "Qwen3 14B Q4_K_M",
        "hf": "Qwen/Qwen3-14B-GGUF:Q4_K_M",
        "size_gb": 9.00,
        "description": "Higher quality; needs more memory bandwidth",
    },
]


def get_model(model_id: str) -> dict | None:
    return next((dict(item) for item in CATALOGUE if item["id"] == model_id), None)

