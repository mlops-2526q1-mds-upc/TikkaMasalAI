import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
from src.models.food_classification_model import FoodClassificationModel


class VGG16(FoodClassificationModel):
    """Interface for accessing the VGG-16 model architecture."""

    def __init__(self, weights: str = "IMAGENET1K_V1", num_classes: int = 101):
        self.model = models.vgg16(weights=weights)

        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)

        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def classify(self, image: bytes) -> int:
        pil_image = Image.open(io.BytesIO(image))

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        input_tensor = self.transform(pil_image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(input_batch)
            predicted_idx = torch.argmax(outputs).item()

        return predicted_idx
