"""
model.py — OpenAI-compatible client pointed at HuggingFace Router.
"""

from openai import OpenAI

# Supported multimodal models via HuggingFace Router
AVAILABLE_MODELS = {
    "Qwen3-235B-A22B (DeepInfra)": "Qwen/Qwen3-235B-A22B:deepinfra",
    "Qwen2.5-VL-72B (Nebius)": "Qwen/Qwen2.5-VL-72B-Instruct:nebius",
    "Llama-3.2-11B Vision (Fireworks)": "meta-llama/Llama-3.2-11B-Vision-Instruct:fireworks-ai",
}

HF_BASE_URL = "https://router.huggingface.co/v1"


def get_client(api_key: str) -> OpenAI:
    """Return an OpenAI-compatible client using the HuggingFace router."""
    if not api_key or not api_key.strip():
        raise ValueError("A valid HuggingFace API key is required.")
    return OpenAI(base_url=HF_BASE_URL, api_key=api_key.strip())
