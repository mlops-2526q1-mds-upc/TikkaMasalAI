import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import io
import os
import tempfile

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
        """
        Always load from the Hugging Face Hub. No local model storage.
        """

        # Set up proper cache directory for HF Spaces (safe no-op locally)
        if not os.environ.get("HF_HOME"):
            cache_dir = tempfile.mkdtemp(prefix="transformers_cache_")
            os.environ["HF_HOME"] = str(cache_dir)

        cache_dir = os.environ.get("HF_HOME")

        # Load from the Hub
        self.model = SiglipForImageClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,
        )
        self.model_name = model_name
        self.model_path = model_name

    def classify(self, image: bytes) -> int:
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

        predicted_idx = torch.argmax(probs).item()
        return predicted_idx
