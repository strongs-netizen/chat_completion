import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from model import get_client, AVAILABLE_MODELS
from preprocessing import (
    validate_image_url,
    fetch_image_for_preview,
    uploaded_file_to_base64,
    resize_if_needed,
)
from inference import run_inference

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Chat · HuggingFace Router",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: -0.03em;
    }
    .stButton > button {
        background: #0f0f0f;
        color: #f5f5f5;
        border: 1.5px solid #0f0f0f;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        padding: 0.5rem 1.4rem;
        transition: background 0.15s, color 0.15s;
    }
    .stButton > button:hover {
        background: #f5f5f5;
        color: #0f0f0f;
    }
    .response-box {
        background: #f7f7f5;
        border-left: 3px solid #0f0f0f;
        padding: 1.2rem 1.4rem;
        border-radius: 0 4px 4px 0;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.95rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    .badge {
        display: inline-block;
        background: #0f0f0f;
        color: #f5f5f5;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 2px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "HuggingFace API Key",
        type="password",
        placeholder="hf_…",
        help="Get yours at https://huggingface.co/settings/tokens",
    )

    model_label = st.selectbox("Model", list(AVAILABLE_MODELS.keys()))
    model_id = AVAILABLE_MODELS[model_label]

    st.markdown("---")
    st.markdown("### Generation Settings")
    max_tokens = st.slider("Max tokens", 64, 1024, 512, step=64)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)

    system_prompt = st.text_area(
        "System prompt (optional)",
        placeholder="e.g. You are a concise assistant that answers in bullet points.",
        height=100,
    )

    st.markdown("---")
    st.caption("Powered by [HuggingFace Router](https://huggingface.co/docs/api-inference/)")

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("# 🔭 Vision Chat")
st.markdown(
    "<span style='font-family:IBM Plex Mono;font-size:0.8rem;color:#888;'>"
    "Multimodal inference via HuggingFace Router</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

# ── Left column: image input ───────────────────────────────────────────────────
with col_left:
    st.markdown("### Image Input")

    input_mode = st.radio(
        "Provide image via",
        ["URL", "Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_source = None
    source_type = "url"

    if input_mode == "URL":
        image_url = st.text_input(
            "Image URL",
            placeholder="https://example.com/photo.jpg",
        )
        if image_url:
            with st.spinner("Checking URL…"):
                check = validate_image_url(image_url)

            if check["valid"]:
                st.success("✓ URL reachable", icon="✅")
                preview = fetch_image_for_preview(image_url)
                if preview:
                    preview = resize_if_needed(preview, max_dim=600)
                    st.image(preview, caption="Preview", use_container_width=True)
                image_source = image_url
                source_type = "url"
            else:
                st.error(f"⚠️ {check['error']}")

    else:  # Upload
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            from PIL import Image as PILImage
            import io

            raw_bytes = uploaded.read()
            uploaded.seek(0)  # reset for base64 conversion

            img = PILImage.open(io.BytesIO(raw_bytes))
            img = resize_if_needed(img, max_dim=600)
            st.image(img, caption=uploaded.name, use_container_width=True)

            data_uri, _ = uploaded_file_to_base64(uploaded)
            image_source = data_uri
            source_type = "base64"

# ── Right column: prompt + output ─────────────────────────────────────────────
with col_right:
    st.markdown("### Prompt")
    prompt = st.text_area(
        "Your instruction",
        value="Describe this image in one sentence.",
        height=120,
        label_visibility="collapsed",
    )

    run_btn = st.button("▶ Run Inference", use_container_width=True)

    if run_btn:
        # ── Guards ──────────────────────────────────────────────────────────
        if not api_key:
            st.error("Please enter your HuggingFace API key in the sidebar.")
            st.stop()
        if not image_source:
            st.error("Please provide a valid image (URL or upload).")
            st.stop()
        if not prompt.strip():
            st.error("Prompt cannot be empty.")
            st.stop()

        # ── Build client & run ──────────────────────────────────────────────
        try:
            client = get_client(api_key)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        with st.spinner(f"Asking {model_label}…"):
            try:
                response = run_inference(
                    client=client,
                    model_id=model_id,
                    prompt=prompt,
                    image_source=image_source,
                    source_type=source_type,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt or None,
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        st.markdown("### Response")
        st.markdown(f'<div class="badge">{model_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)

        st.download_button(
            "⬇ Download response",
            data=response,
            file_name="response.txt",
            mime="text/plain",
        )
