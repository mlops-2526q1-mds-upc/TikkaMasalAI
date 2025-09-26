from abc import ABC, abstractmethod


class FoodClassificationModel(ABC):
    """Abstract Base Class that serves as a common interface for all models."""

    @abstractmethod
    def classify(self, image: bytes) -> int:
        """
        Abstract method to classify an image into a food category.

        Args:
            image: The image bytes to classify.

        Returns:
            int: The index of the predicted class. This returns the class index, not the class name.
        """
