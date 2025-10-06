"""
Azure ML real-time scoring script for the PrithivML Food-101 model.

Usage on Azure ML managed online endpoints (v2):
- This file is the entry_script (score.py). AML calls init() once per worker, then run(data).
- Expects JSON input with one of the following shapes:
  {"image": "<base64>"}
  {"images": ["<base64>", ...]}
  {"url": "https://.../image.jpg"}
  {"urls": ["https://...", ...]}
  Optional: {"top_k": 5}

Response shape:
  {"predictions": [{"label": "pizza", "index": 77, "score": 0.92}, ...]}
  For batch inputs, predictions is a list of lists matching input order.

Notes:
- By default, loads the model from the Hugging Face Hub using the name in env MODEL_NAME
  (default: "prithivMLmods/Food-101-93M"). Ensure internet is enabled for the deployment,
  or alternatively pre-package the model and override MODEL_DIR to point to a local path.
- Uses the Food-101 class labels from src.labels.LABELS.
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

try:
    # Prefer direct imports to avoid tight coupling with local wrappers in production scoring
    from transformers import AutoImageProcessor, SiglipForImageClassification
except Exception as e:  # pragma: no cover - surfaced by AML logs
    raise RuntimeError(
        "transformers is required for this scoring script. Ensure it's in the environment."
    ) from e

try:
    # Food-101 labels from the repo
    from src.labels import LABELS
except Exception:
    LABELS = []  # Fallback if not available; indices will be returned as strings


# Globals populated in init()
MODEL = None
PROCESSOR = None
DEVICE = "cpu"
CLASS_NAMES: List[str] = LABELS if LABELS else []


def _log(msg: str) -> None:
    # AML surfaces stdout in logs
    print(f"[score_prithiv_food101] {msg}")


def _setup_hf_cache() -> str:
    # Respect HF_HOME if provided, else create a temp cache path (harmless locally)
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        cache_dir = tempfile.mkdtemp(prefix="transformers_cache_")
        os.environ["HF_HOME"] = cache_dir
    return cache_dir


def _load_model_from_hub(model_name: str, cache_dir: str):
    _log(f"Loading HuggingFace model '{model_name}' (cache: {cache_dir})")
    processor = AutoImageProcessor.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=True
    )
    model = SiglipForImageClassification.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    return processor, model


def _load_model_from_dir(model_dir: str, cache_dir: str):
    # Allow local packaging of HF model (config.json, preprocessor_config.json, etc.)
    _log(f"Loading HuggingFace model from local directory '{model_dir}'")
    processor = AutoImageProcessor.from_pretrained(
        model_dir, cache_dir=cache_dir, use_fast=True
    )
    model = SiglipForImageClassification.from_pretrained(model_dir, cache_dir=cache_dir)
    return processor, model


def init():  # Azure ML calls this once per worker
    global MODEL, PROCESSOR, DEVICE, CLASS_NAMES

    # Device selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Using device: {DEVICE}")

    # Where to load the model from
    model_name = os.environ.get("MODEL_NAME", "prithivMLmods/Food-101-93M")
    model_dir = os.environ.get("MODEL_DIR")  # optionally baked-in artifacts

    cache_dir = _setup_hf_cache()

    # Load model + processor
    if model_dir and os.path.isdir(model_dir):
        PROCESSOR, MODEL = _load_model_from_dir(model_dir, cache_dir)
    else:
        PROCESSOR, MODEL = _load_model_from_hub(model_name, cache_dir)

    MODEL.to(DEVICE)
    MODEL.eval()

    # Load labels if provided via env or keep default Food-101 labels
    labels_path = os.environ.get("LABELS_PATH")
    if labels_path and os.path.exists(labels_path):
        try:
            with open(labels_path, "r") as f:
                CLASS_NAMES = json.load(f)
            _log(f"Loaded {len(CLASS_NAMES)} labels from {labels_path}")
        except Exception as e:  # pragma: no cover
            _log(f"Failed to load labels from {labels_path}: {e}")
    else:
        if CLASS_NAMES:
            _log(f"Using built-in Food-101 labels: {len(CLASS_NAMES)} classes")
        else:
            _log("No labels available; will return indices as strings")


def _b64_to_pil(b64: str) -> Image.Image:
    binary = base64.b64decode(b64)
    return Image.open(io.BytesIO(binary)).convert("RGB")


def _url_to_pil(url: str) -> Image.Image:
    import requests  # lazy import to keep init() light

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _predict_one(
    img: Image.Image, top_k: int = 5
) -> List[Dict[str, Union[str, int, float]]]:
    global MODEL, PROCESSOR, DEVICE, CLASS_NAMES

    inputs = PROCESSOR(images=img, return_tensors="pt")
    # Move tensors to the right device
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)

    k = min(top_k, probs.shape[-1])
    values, indices = torch.topk(probs, k=k)

    results: List[Dict[str, Union[str, int, float]]] = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        if CLASS_NAMES and 0 <= idx < len(CLASS_NAMES):
            label = CLASS_NAMES[idx]
        else:
            label = str(idx)
        results.append({"label": label, "index": int(idx), "score": float(score)})
    return results


def _handle_single(payload: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    if "image" in payload:
        img = _b64_to_pil(payload["image"])
        preds = _predict_one(img, top_k=top_k)
        return {"predictions": preds}
    if "url" in payload:
        img = _url_to_pil(payload["url"])
        preds = _predict_one(img, top_k=top_k)
        return {"predictions": preds}
    return {"error": "Expected 'image' (base64) or 'url' field"}


def _handle_batch(payload: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    if "images" in payload:
        preds = []
        for b64 in payload["images"]:
            try:
                img = _b64_to_pil(b64)
                preds.append(_predict_one(img, top_k=top_k))
            except Exception as e:
                preds.append({"error": str(e)})
        return {"predictions": preds}
    if "urls" in payload:
        preds = []
        for url in payload["urls"]:
            try:
                img = _url_to_pil(url)
                preds.append(_predict_one(img, top_k=top_k))
            except Exception as e:
                preds.append({"error": str(e)})
        return {"predictions": preds}
    return {"error": "Expected 'images' (list of base64) or 'urls' (list)"}


def run(raw_data: str) -> str:  # Azure ML calls this per request
    try:
        # Health probe convenience
        if raw_data and raw_data.strip().lower() in {"ping", "health"}:
            return json.dumps({"status": "ok"})

        payload = json.loads(raw_data) if raw_data else {}
        top_k = int(payload.get("top_k", 5))

        if any(k in payload for k in ("image", "url")):
            response = _handle_single(payload, top_k=top_k)
        elif any(k in payload for k in ("images", "urls")):
            response = _handle_batch(payload, top_k=top_k)
        else:
            response = {
                "error": "Invalid request body. Provide 'image'/'images' or 'url'/'urls'."
            }

        return json.dumps(response)
    except Exception as e:
        # Surface exception in logs and response
        _log(f"Error in run(): {e}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":  # Optional local smoke test (no AML required)
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(
        description="Local test for AML scoring entry script"
    )
    parser.add_argument(
        "--image", type=str, help="Path to an image file to classify", nargs="?"
    )
    parser.add_argument(
        "--url", type=str, help="URL of an image to classify", nargs="?"
    )
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    init()

    if args.image:
        p = pathlib.Path(args.image)
        if not p.exists():
            raise SystemExit(f"Image not found: {p}")
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        request = json.dumps({"image": b64, "top_k": args.top_k})
    elif args.url:
        request = json.dumps({"url": args.url, "top_k": args.top_k})
    else:
        raise SystemExit("Provide --image or --url")

    print(run(request))
