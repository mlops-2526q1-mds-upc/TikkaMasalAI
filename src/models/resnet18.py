import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io

from src.models.food_classification_model import FoodClassificationModel


class Resnet18(FoodClassificationModel):
    """
    Interface for accessing the Resnet-18 model architecture.
    
    Can load either:
    - Base pretrained model: https://huggingface.co/microsoft/resnet-18
    - Fine-tuned model: Trained on Food-101 dataset
    """

    def __init__(
        self,
        model_path: str = "microsoft/resnet-18",
        preprocessor_path: str = None,
    ):
        """
        Initialize the Resnet18 model.
        
        Args:
            model_path: Path to model weights. Can be:
                - HuggingFace model ID (e.g., "microsoft/resnet-18")
                - Local path to fine-tuned model (e.g., "models/resnet18-food101")
            preprocessor_path: Optional separate path for preprocessor.
                If None, uses model_path for both model and preprocessor.
        """
        self.model_path = model_path  # Store for MLflow logging
        
        if preprocessor_path is None:
            preprocessor_path = model_path
        
        self.preprocessor_path = preprocessor_path  # Store for MLflow logging
            
        self.image_processor = AutoImageProcessor.from_pretrained(preprocessor_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode

    def classify(self, image: bytes) -> int:
        """
        Classify an image into a food category.
        
        Args:
            image: The image bytes to classify.
            
        Returns:
            int: The index of the predicted class.
        """
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        inputs = self.image_processor(pil_image, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        return predicted_label
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        Convenience method to load a model.
        
        Args:
            model_path: Path to model (local or HuggingFace)
            
        Returns:
            Resnet18: Initialized model instance
        """
        return cls(model_path=model_path)
    
    @classmethod
    def load_finetuned(cls, checkpoint_dir: str):
        """
        Load a fine-tuned Food-101 model.
        
        Args:
            checkpoint_dir: Directory containing fine-tuned model
            
        Returns:
            Resnet18: Model loaded with fine-tuned weights
        """
        return cls(model_path=checkpoint_dir)