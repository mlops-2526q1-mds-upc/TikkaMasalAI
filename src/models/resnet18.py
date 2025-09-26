import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io

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
        self.image_processor = AutoImageProcessor.from_pretrained(preprocessor_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)

    def classify(self, image: bytes) -> int:
        pil_image = Image.open(io.BytesIO(image))
        inputs = self.image_processor(pil_image, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        return predicted_label
