import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import io

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
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model_name = model_name

    def classify(self, image: bytes) -> int:
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

        predicted_idx = torch.argmax(probs).item()
        return predicted_idx
