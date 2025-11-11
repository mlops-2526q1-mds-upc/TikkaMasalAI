"""Prediction router for food image classification."""

import base64
from functools import lru_cache
import io
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
from fastapi import APIRouter, File, HTTPException, UploadFile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

from src.labels import LABELS

router = APIRouter(prefix="/predict", tags=["prediction"])

ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT_DIR / "models" / "prithiv"

logger = logging.getLogger(__name__)

# Select device: Apple MPS > CUDA > CPU
DEVICE = (
    torch.device("mps")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
)


@lru_cache
def _load_model_and_preprocessor():
    """Load the model and image processor from disk and prepare them.

    This helper loads the model and its associated image processor from
    the `models/prithiv` directory (controlled by `MODEL_PATH`). The model
    is moved to the selected device and set to evaluation mode.

    Returns:
        Tuple[torch.nn.Module, transformers.AutoImageProcessor]: The loaded
            model and processor. The model is already put in eval() mode and
            moved to `DEVICE`.

    Raises:
        FileNotFoundError: If the model directory does not exist at
            `MODEL_PATH`.
    """
    if not MODEL_PATH.exists():
        logger.error("Model directory not found at %s", MODEL_PATH)
        raise FileNotFoundError(f"Model directory not found at: {MODEL_PATH}.")

    m = SiglipForImageClassification.from_pretrained(str(MODEL_PATH), local_files_only=True)
    p = AutoImageProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    # Move to device and eval
    m.eval().to(DEVICE)
    return m, p


def _id_to_label(i: int) -> str:
    """Safe label lookup with fallback.

    Args:
        i: Numeric class id.

    Returns:
        Human readable label if available from `LABELS`, otherwise a
        fallback string in the format ``class_<i>``.
    """
    try:
        return LABELS[i]
    except Exception:
        return f"class_{i}"


def classify_food(image: Image.Image) -> dict[str, float]:
    """Classify food type from an image and return top predictions.

    Args:
        image: PIL Image object. The function will convert it to RGB if
            necessary.

    Returns:
        A dictionary mapping the top predicted labels to their probabilities
        (floats rounded to 3 decimal places). Up to 5 top predictions are
        returned, sorted by descending probability.

    Raises:
        Any exception raised during model loading or inference is propagated
        to the caller (the endpoint wraps this in an HTTPException).
    """
    model, processor = _load_model_and_preprocessor()
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    # Move tensors to selected device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.float().cpu()
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)

    # Top-K (up to 5 or number of classes)
    k = int(min(5, probs.numel()))
    top_p, top_i = torch.topk(probs, k)
    predictions = {_id_to_label(int(i)): round(float(p), 3) for p, i in zip(top_p, top_i)}
    return predictions


@router.post(
    "",
    summary="Predict Food Class",
    description="Run the image classification model on an uploaded image and return the top predicted labels with their probabilities.\n\nThis endpoint accepts a multipart/form-data POST with a file field named 'image'.",
)
async def predict_endpoint(image: UploadFile = File(...)) -> dict[str, Any]:
    """Predict the food type for an uploaded image.

    This endpoint accepts an image upload (JPEG, PNG, etc.), runs the
    classification model, and returns the top predicted labels with their
    probabilities.

    Args:
        image: Uploaded image file. Must be an image (content-type starting
            with `image/`).

    Returns:
        A dictionary containing:
            - predictions (dict): Top predictions mapping label -> probability
              (float between 0 and 1). Up to 5 top predictions are returned.
            - filename (str): Original filename of the uploaded image.

    Raises:
        HTTPException:
            - 400: If the uploaded file is not an image.
            - 500: If the prediction process fails unexpectedly.

    Example Response:
        {
            "predictions": {"guacamole": 0.996, "salsa": 0.002, ...},
            "filename": "food.jpg"
        }

    Usage:
        The client should POST a multipart/form-data request with a file
        field named `image`. Example HTML form usage:

        <form action="/predict" method="post" enctype="multipart/form-data">
          <input type="file" name="image" />
          <button type="submit">Upload</button>
        </form>
    """
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        result = classify_food(img)

        return {"predictions": result, "filename": image.filename}

    except HTTPException as he:
        # Log client errors at warning level (no stack trace needed)
        try:
            logger.warning("/predict 400: %s", getattr(he, "detail", he))
        except Exception:
            pass
        raise he
    except Exception as e:
        # Log full traceback for easier debugging
        logger.exception("/predict 500: Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def generate_attention_heatmap(image: Image.Image) -> dict[str, Any]:
    """Generate an attention heatmap overlay for an input image.

    This function runs the model with `output_attentions=True` (if supported)
    and builds a heatmap from the last attention layer averaged across heads.
    The resulting heatmap is resized to match the input image and returned as
    a base64-encoded PNG along with prediction metadata.

    Args:
        image: PIL Image object. The function will convert it to RGB if
            necessary.

    Returns:
        A dictionary with the following keys:
            - predicted_class (str): Predicted food label.
            - confidence (float): Model confidence for the predicted class.
            - attention_map (str): Base64-encoded PNG image of the overlay.
            - num_layers (int): Number of attention layers returned by model.
            - num_heads (int): Number of attention heads in the first layer.
            - grid_size (str): Patch grid dimensions used to reshape attention
              (e.g. "13x15").

    Raises:
        ValueError: If the model did not return attention weights.
        Any other exception from model inference is propagated to the caller.
    """
    model, processor = _load_model_and_preprocessor()
    image = image.convert("RGB")

    # Set attention implementation to eager to enable attention outputs if available
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass

    # Process image and get attention
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits.float().cpu()
        attentions = outputs.attentions

        # Get prediction
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_idx = int(logits.argmax(-1).item())
        confidence = float(probs[0, predicted_idx].item())
        predicted_class = _id_to_label(predicted_idx)

    if attentions is None or len(attentions) == 0:
        raise ValueError("Model did not return attention weights")

    # Ensure attention tensors are on CPU before converting to numpy
    attentions = [att.detach().to("cpu") for att in attentions]

    # Average attention across all heads in the last layer
    last_layer_attention = attentions[-1][0].mean(dim=0).numpy()

    # Skip CLS token (first token) and get patch attention
    patch_attention = last_layer_attention[1:, 1:]
    attention_map = patch_attention.mean(axis=0)

    # Calculate grid size (assuming 195 patches = 13x15)
    n_patches = len(attention_map)
    possible_sizes = [(13, 15), (15, 13), (14, 14)]
    grid_h, grid_w = 14, 14  # default

    for h, w in possible_sizes:
        if h * w == n_patches:
            grid_h, grid_w = h, w
            break
    else:
        # Pad if necessary
        grid_h = grid_w = int(math.sqrt(n_patches))
        pad_size = grid_h * grid_w - n_patches
        if pad_size > 0:
            attention_map = np.pad(attention_map, (0, pad_size), constant_values=0)

    attention_grid = attention_map.reshape(grid_h, grid_w)

    # Resize attention map to match image size
    img_array = np.array(image)
    attention_resized = zoom(
        attention_grid,
        (img_array.shape[0] / grid_h, img_array.shape[1] / grid_w),
        order=1,
    )

    # Normalize attention map
    attention_normalized = (attention_resized - attention_resized.min()) / (
        attention_resized.max() - attention_resized.min() + 1e-8
    )

    # Create heatmap overlay using matplotlib (figure backend already set at module import)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(attention_normalized, cmap="jet", alpha=0.5)
    ax.axis("off")
    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    # Encode to base64
    attention_img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "attention_map": attention_img_b64,
        "num_layers": len(attentions),
        "num_heads": attentions[0].shape[1],
        "grid_size": f"{grid_h}x{grid_w}",
    }


@router.post(
    "/explain",
    summary="Generate Attention Heatmap",
    description="Creates an attention-based visualization showing which parts of the image the model focuses on for prediction.",
)
async def explain_prediction(image: UploadFile = File(...)) -> dict[str, Any]:
    """Generate attention-based explainability visualization for a food image.

    This endpoint uses the model's attention weights to create a heatmap overlay
    showing which regions of the image the model focuses on when making its prediction.

    The attention visualization helps understand:
    - Which parts of the image are most important for the classification
    - Whether the model is looking at relevant features (e.g., the food itself vs background)
    - Model interpretability and debugging

    Args:
        image: Uploaded image file. Must be a valid image format (JPEG, PNG, etc.)

    Returns:
        Dictionary containing:
            - predicted_class: Predicted food type (string)
            - confidence: Model's confidence in the prediction (float 0-1)
            - attention_map: Base64-encoded PNG image showing attention heatmap overlay on original image
            - num_layers: Number of attention layers in the model (int)
            - num_heads: Number of attention heads per layer (int)
            - grid_size: Grid dimensions of attention patches (string, e.g., "13x15")
            - filename: Original filename of the uploaded image (string)

    Raises:
        HTTPException:
            - 400: If file is not an image
            - 500: If generation fails

    Example Response:
        ```json
        {
            "predicted_class": "guacamole",
            "confidence": 0.9968,
            "attention_map": "iVBORw0KGgoAAAANS...",
            "num_layers": 12,
            "num_heads": 12,
            "grid_size": "13x15",
            "filename": "food.jpg"
        }
        ```

    Usage:
        The attention_map can be displayed in HTML:
        ```html
        <img src="data:image/png;base64,{attention_map}" />
        ```
    """
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        result = generate_attention_heatmap(img)
        result["filename"] = image.filename

        return result

    except HTTPException as he:
        try:
            logger.warning("/predict/explain 400: %s", getattr(he, "detail", he))
        except Exception:
            pass
        raise he
    except Exception as e:
        logger.exception("/predict/explain 500: Explainability generation failed")
        raise HTTPException(status_code=500, detail=f"Explainability generation failed: {str(e)}")
