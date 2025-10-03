import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import os

from src.models.food_classification_model import FoodClassificationModel


class Resnet18(FoodClassificationModel):
    """
    Interface for accessing the Resnet-18 model architecture.
    See the base model here: https://huggingface.co/microsoft/resnet-18.
    """

    def __init__(
        self,
        preprocessor_path: str = "microsoft/resnet-18",
        model_path: str = "microsoft/resnet-18",
    ):
        """
        Always load from the Hugging Face Hub. No local model storage.
        """

        cache_dir = os.environ.get("HF_HOME")

        # Load from the Hub (will cache under HF_HOME if set)
        self.image_processor = AutoImageProcessor.from_pretrained(
            preprocessor_path, cache_dir=cache_dir
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            model_path, cache_dir=cache_dir
        )

        # For metadata/logging
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def classify(self, image: bytes) -> int:
        pil_image = Image.open(io.BytesIO(image))
        inputs = self.image_processor(pil_image, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # model predicts one of the 101 Food-101 classes (if fine-tuned for Food-101).
        # If using the default microsoft/resnet-18 weights, this will predict one
        # of the 1000 ImageNet classes, not Food-101.
        predicted_label = logits.argmax(-1).item()
        return predicted_label
