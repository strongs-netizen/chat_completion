"""
preprocessing.py — Validate and prepare image inputs for the model.
"""

import io
import base64
import requests
from PIL import Image


MAX_IMAGE_SIZE_MB = 5
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def validate_image_url(url: str) -> dict:
    """
    Validate that a URL points to an accessible image.

    Returns:
        dict with keys: valid (bool), error (str | None), content_type (str | None)
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return {"valid": False, "error": "URL must start with http:// or https://", "content_type": None}

    try:
        resp = requests.head(url, timeout=8, allow_redirects=True)
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()

        if resp.status_code != 200:
            return {"valid": False, "error": f"URL returned status {resp.status_code}.", "content_type": None}

        content_length = int(resp.headers.get("Content-Length", 0))
        if content_length > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            return {
                "valid": False,
                "error": f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit.",
                "content_type": content_type,
            }

        return {"valid": True, "error": None, "content_type": content_type}

    except requests.exceptions.Timeout:
        return {"valid": False, "error": "Request timed out. Check the URL and try again.", "content_type": None}
    except requests.exceptions.RequestException as exc:
        return {"valid": False, "error": f"Could not reach URL: {exc}", "content_type": None}


def fetch_image_for_preview(url: str) -> Image.Image | None:
    """
    Download and return a PIL Image for Streamlit preview.
    Returns None on failure.
    """
    try:
        resp = requests.get(url.strip(), timeout=10, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img.verify()                          # detect corrupt files early
        img = Image.open(io.BytesIO(resp.content))  # re-open after verify
        return img
    except Exception:
        return None


def uploaded_file_to_base64(uploaded_file) -> tuple[str, str]:
    """
    Convert a Streamlit UploadedFile to a base64 data-URI.

    Returns:
        (data_uri, mime_type)  e.g. ("data:image/jpeg;base64,…", "image/jpeg")
    """
    raw = uploaded_file.read()
    mime = uploaded_file.type or "image/jpeg"
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return data_uri, mime


def resize_if_needed(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Downscale image so neither dimension exceeds max_dim."""
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img
