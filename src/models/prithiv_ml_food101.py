import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import io
import os
import tempfile
from pathlib import Path

from src.models.food_classification_model import FoodClassificationModel


class PrithivMlFood101(FoodClassificationModel):
    """
    Interface for accessing the PrithivML Food-101 model architecture.
    This model was already trained on the Food-101 dataset and performs well on it.
    It is supposed to serve as a benchmark for our own model finetuning and potentially
    as an alternative to be deployed.
    See it on Huggingface: https://huggingface.co/prithivMLmods/Food-101-93M.
    """

    def __init__(self, model_name: str = "prithivMLmods/Food-101-93M"):
        # Set up proper cache directory for HF Spaces
        if not os.environ.get("HF_HOME"):
            cache_dir = Path(tempfile.gettempdir()) / "transformers_cache"
            cache_dir.mkdir(exist_ok=True)
            os.environ["HF_HOME"] = str(cache_dir)

        # Add retry logic and better error handling
        try:
            self.model = SiglipForImageClassification.from_pretrained(
                model_name,
                cache_dir=os.environ.get("HF_HOME"),
                local_files_only=False,
                force_download=False,
            )
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                cache_dir=os.environ.get("HF_HOME"),
                local_files_only=False,
                force_download=False,
                use_fast=True,  # Use fast processor to avoid warning
            )
            self.model_name = model_name
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def classify(self, image: bytes) -> int:
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

        predicted_idx = torch.argmax(probs).item()
        return predicted_idx
