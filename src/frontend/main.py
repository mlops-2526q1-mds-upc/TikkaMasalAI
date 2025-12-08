"""Streamlit frontend for TikkaMasalAI.

This module provides the interactive UI to upload an image, get a prediction
from the backend API, request an explanation heatmap, and ask an LLM for
additional information about the predicted dish.
"""

import hashlib
import io
import mimetypes
from typing import Any, Dict, Optional

from config import get_api_url, get_explain_url, get_llm_url
from explain import extract_top_scores
from heatmap import (
    _try_parse_base64_image,
    find_heatmap_in_payload,
    overlay_heatmap_on_image,
)
from llm import extract_llm_text
import numpy as np
from PIL import Image
from predict import extract_primary_label
import requests
import streamlit as st
from image_utils import prepare_image_for_upload


def main() -> None:
    """Run the Streamlit application.

    The UI exposes three tabs:
    - Get Recipe: Queries an LLM with a suggested or user-entered question.
    - Explain Prediction: Requests an explanation heatmap from the backend.
    - Raw JSON: Displays raw responses for debugging and transparency.
    """
    st.set_page_config(page_title="TikkaMasalAI", page_icon=":curry:", layout="wide")

    # URLs (configurable via secrets/query)
    api_url = get_api_url()
    explain_url = get_explain_url()
    llm_url = get_llm_url()

    # Sidebar: App title and uploader
    st.sidebar.title("Tikka MasalAI 🍲")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a foto of your dish here!",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, PNG, WEBP.",
    )

    # Gate main content on upload
    if not uploaded_file:
        st.title("TikkaMasalAI Dish Classifier")
        st.info("Please upload a food photo in the sidebar to get started.")
        return

    # Load image + compute a content hash to cache prediction across reruns
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    # Sidebar preview of the uploaded image
    st.sidebar.image(image, caption="Preview", use_container_width=True)

    upload_bytes, upload_mime, upload_name, upload_dims, _ = prepare_image_for_upload(
        image, uploaded_file.name
    )
    # Sidebar hint about what will be uploaded
    try:
        st.sidebar.caption(
            f"Uploading scaled image: {upload_dims[0]}×{upload_dims[1]} • {len(upload_bytes)/1024:.0f} KB"
        )
    except Exception:
        pass
    img_hash = hashlib.md5(image_bytes).hexdigest()

    # Cache prediction for same image across reruns
    cache_hit = (
        st.session_state.get("tikka_img_hash") == img_hash
        and st.session_state.get("tikka_prediction") is not None
    )
    if not cache_hit:
        with st.spinner("Identifying the dish..."):
            try:
                response = requests.post(
                    api_url,
                    files={"image": (upload_name, upload_bytes, upload_mime)},
                    timeout=30,
                )
                response.raise_for_status()
                pred_json = response.json()
            except requests.exceptions.RequestException as request_error:
                st.error(f"Request failed: {request_error}")
                return
            except ValueError:
                st.error("The service returned a non-JSON response.")
                return

        st.session_state["tikka_img_hash"] = img_hash
        st.session_state["tikka_prediction"] = pred_json
        primary = extract_primary_label(pred_json)
        st.session_state["tikka_primary_label"] = primary
        # Visual mark: toast to indicate prediction completed (once per new image)
        try:
            if primary:
                st.toast(f"Predicted: {primary}", icon="✅")
            else:
                st.toast("Prediction ready", icon="✅")
        except Exception:
            # Fallback for Streamlit versions without st.toast
            if primary:
                st.success(f"Predicted: {primary.capitalize()}")
        # Reset tab-specific session artifacts on new image
        st.session_state.pop("tikka_llm_response", None)
        st.session_state.pop("tikka_explain_raw", None)
    else:
        pred_json = st.session_state["tikka_prediction"]
        # Primary label may have been computed already
        if "tikka_primary_label" not in st.session_state:
            st.session_state["tikka_primary_label"] = extract_primary_label(pred_json)

    primary_label: Optional[str] = st.session_state.get("tikka_primary_label")
    # Normalize label for display if present
    if primary_label:
        if "_" in primary_label:
            words = [word.capitalize() for word in primary_label.split("_")]
            primary_label = " ".join(words)
        else:
            primary_label = primary_label.capitalize()

    # Main area: Prediction followed by three futher possibile actions
    if primary_label:
        st.markdown(
            f"## ✅ Predicted Dish: <span style='color:#16a34a'>{primary_label}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.header("Predicted Dish")
        st.info("Prediction received, but no obvious label was found in the response.")

    # Segmented control for next steps with formatted labels
    def _format_action_label(opt: str) -> str:
        return "ℹ️ **Dish Information**" if opt == "info" else "👨‍🏫 **Explain Prediction**"

    section = st.segmented_control(
        label="",
        options=["info", "explain"],
        key="seg_actions",
        default="info",
        format_func=_format_action_label,
        width="stretch",
    )
    st.divider()

    # Section: Get Recipe (LLM)
    if section == "info":
        st.subheader("Information About the Dish")
        recipe_question = f"How do I prepare {primary_label}? Provide ingredients and clear step-by-step cooking instructions. Be concise."
        nutrient_value_question = f"What are the nutritional values of {primary_label}? Provide information for an amount in which it is typically consumed."
        inventor_question = f"In which nation was {primary_label} invented? If you can, provide further information like who invented it or when it was invented."

        options = {
            "Recipe": recipe_question,
            "Nutritional Value": nutrient_value_question,
            "Inventor": inventor_question,
        }

        choice = st.selectbox(
            label="What do you want to know about?",
            options=list(options.keys()),
        )

        question = options[choice]

        user_question = st.text_area(
            "Your question:",
            value=question,
            height=120,
            help="You can edit the question before asking the LLM.",
        )
        ask = st.button("Get information from LLM", type="primary")

        if ask and user_question.strip():
            with st.spinner("Asking the LLM for a recipe..."):
                llm_error: Optional[str] = None
                payload = {
                    "prompt": user_question,
                    "temperature": 0.8,
                    # "max_tokens": 64,
                }
                try:
                    r = requests.post(llm_url, json=payload, timeout=60)
                    r.raise_for_status()
                    llm_response_json = r.json()
                except requests.exceptions.RequestException as e:
                    llm_error = str(e)
                except ValueError:
                    llm_error = "The LLM service returned a non-JSON response."

                if llm_response_json is None:
                    st.error(
                        "Failed to get a response from the LLM service. "
                        f"Last error: {llm_error or 'Unknown error'}"
                    )
                else:
                    st.session_state["tikka_llm_response"] = llm_response_json
                    llm_text = extract_llm_text(llm_response_json)
                    if llm_text:
                        st.subheader(choice)
                        st.markdown(llm_text)
                    else:
                        st.info("Received an LLM response, but couldn't find any text to display.")

        # If a previous response exists (e.g., after tab switch), show it
        if st.session_state.get("tikka_llm_response") and not ask:
            llm_text = extract_llm_text(st.session_state["tikka_llm_response"])
            if llm_text:
                st.subheader("Suggested recipe")
                st.markdown(llm_text)

    # Section: Explain Prediction (Heatmap)
    if section == "explain":
        st.subheader("Explain prediction")
        # Show top-5 class probabilities from the earlier prediction
        top5 = extract_top_scores(pred_json, top_n=5)
        if top5:
            st.markdown("**Model certainty for top 5 predictions**")
            st.caption("Bars represent the model's certainty for each class.")
            for label, score in top5:
                # Convert to percentage if needed and clamp to [0, 100]
                pct = float(score)
                pct = pct * 100.0 if pct <= 1.0 else pct
                pct = max(0.0, min(100.0, pct))
                lcol, pcol, vcol = st.columns([2, 6, 2])
                with lcol:
                    st.markdown(f"**{label}**")
                with pcol:
                    st.progress(int(round(pct)))
                with vcol:
                    st.markdown(f"{pct:.1f}%")

        st.markdown("**Visualize attention from the model**")
        do_explain = st.button("Generate Heatmap")

        if do_explain:
            with st.spinner("Requesting explanation heatmap..."):
                try:
                    form_data = {"label": (None, primary_label)} if primary_label else None
                    response = requests.post(
                        explain_url,
                        files={"image": (upload_name, upload_bytes, upload_mime)},
                        data=form_data,
                        timeout=60,
                    )
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "")
                    overlay_img: Optional[Image.Image] = None
                    heatmap_arr: Optional[np.ndarray] = None
                    raw_payload: Optional[Dict[str, Any]] = None

                    if content_type.startswith("image/"):
                        overlay_img = Image.open(io.BytesIO(response.content))
                    else:
                        try:
                            parsed = response.json()
                            raw_payload = parsed
                        except ValueError:
                            parsed = response.text.strip()

                        if isinstance(parsed, str):
                            maybe_img = _try_parse_base64_image(parsed)
                            if maybe_img is not None:
                                overlay_img = maybe_img
                            else:
                                st.info(
                                    "Explain service returned a string, but it wasn't a valid base64 image."
                                )
                        elif isinstance(parsed, dict):
                            found = find_heatmap_in_payload(parsed)
                            raw_payload = parsed
                            if "overlay_image" in found:
                                overlay_img = found["overlay_image"]
                            elif "heatmap" in found:
                                heatmap_arr = found["heatmap"]
                            else:
                                st.info(found.get("error", "No heatmap found in response."))
                        else:
                            st.info("Explain service returned an unsupported payload format.")

                    # Persist raw payload for the Raw JSON tab
                    if raw_payload is not None:
                        st.session_state["tikka_explain_raw"] = raw_payload

                    if overlay_img is not None:
                        st.image(
                            overlay_img,
                            caption="Explanation overlay (from server)",
                            width=480,
                        )
                    elif heatmap_arr is not None:
                        try:
                            blended = overlay_heatmap_on_image(image, heatmap_arr, opacity=0.5)
                            st.image(
                                blended,
                                caption="Heatmap overlay",
                                width=480,
                            )
                        except Exception as e:
                            st.error(f"Failed to overlay heatmap: {e}")
                except requests.exceptions.RequestException as request_error:
                    st.error(f"Explain request failed: {request_error}")


if __name__ == "__main__":
    main()
