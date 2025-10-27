# src/api/deps.py
import os
from functools import lru_cache
from importlib import import_module
from typing import Protocol, Tuple

from fastapi import HTTPException
from src.models.food_classification_model import FoodClassificationModel


class InferenceService(Protocol):
    model_name: str
    def predict(self, image_bytes: bytes) -> int: ...


class _DefaultInferenceService:
    def __init__(self, model: FoodClassificationModel, model_name: str):
        self._model = model
        self.model_name = model_name

    def predict(self, image_bytes: bytes) -> int:
        return self._model.classify(image_bytes)


# Register concrete model classes here (module_path, class_name)
MODEL_REGISTRY = {
    # default v1 model
    "PrithivMlFood101": ("src.models.prithiv_ml_food101", "PrithivMlFood101"),
}


def _construct_model() -> Tuple[FoodClassificationModel, str]:
    """
    Build the concrete model indicated by FOOD_MODEL_CLASS (default: PrithivMlFood101).
    Optionally override its HF id with FOOD_HF_MODEL_ID.
    """
    cls_key = os.getenv("FOOD_MODEL_CLASS", "PrithivMlFood101")
    if cls_key not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=500,
            detail=f"Unknown model class '{cls_key}'. Available: {list(MODEL_REGISTRY)}",
        )

    module_path, class_name = MODEL_REGISTRY[cls_key]
    try:
        module = import_module(module_path)
        cls = getattr(module, class_name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import {class_name} from {module_path}: {e}",
        )

    hf_id = os.getenv("FOOD_HF_MODEL_ID")  # e.g. "prithivMLmods/Food-101-93M"
    try:
        if hf_id:
            try:
                model = cls(hf_id)  # positional
            except TypeError:
                model = cls(model_name=hf_id)  # keyword
        else:
            model = cls()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize the model: {e}")

    model_name = getattr(model, "model_name", class_name)
    return model, model_name


@lru_cache(maxsize=1)
def get_inference_service() -> _DefaultInferenceService:
    model, name = _construct_model()
    return _DefaultInferenceService(model, name)
