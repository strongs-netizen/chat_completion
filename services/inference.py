"""
inference.py — Run multimodal chat completion via the HuggingFace router.
"""

from openai import OpenAI


def build_image_message(prompt: str, image_source: str, source_type: str = "url") -> dict:
    """
    Build the user message payload.

    Args:
        prompt:       The text instruction / question.
        image_source: Either a public URL or a base64 data-URI.
        source_type:  "url" | "base64"

    Returns:
        A messages-list entry formatted for the chat completions API.
    """
    image_content: dict
    if source_type == "base64":
        image_content = {"type": "image_url", "image_url": {"url": image_source}}
    else:
        image_content = {"type": "image_url", "image_url": {"url": image_source}}

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            image_content,
        ],
    }


def run_inference(
    client: OpenAI,
    model_id: str,
    prompt: str,
    image_source: str,
    source_type: str = "url",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> str:
    """
    Call the chat completions endpoint and return the assistant's reply.

    Args:
        client:       Initialised OpenAI client (see model.py).
        model_id:     HuggingFace model string, e.g. "Qwen/Qwen2.5-VL-72B-Instruct:nebius".
        prompt:       User's text instruction.
        image_source: Public URL or base64 data-URI of the image.
        source_type:  "url" | "base64".
        max_tokens:   Maximum tokens for the response.
        temperature:  Sampling temperature (0 = deterministic).
        system_prompt: Optional system-level instruction.

    Returns:
        The model's text response as a string.

    Raises:
        RuntimeError: wraps any API-level errors with a user-friendly message.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(build_image_message(prompt, image_source, source_type))

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc


def run_text_only_inference(
    client: OpenAI,
    model_id: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> str:
    """Text-only fallback (no image attached)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc
