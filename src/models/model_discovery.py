"""
Model discovery utility for dynamically finding all models that inherit from FoodClassificationModel.
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Any

from .food_classification_model import FoodClassificationModel


def discover_models(models_dir: Path = None) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically discover all models that inherit from FoodClassificationModel.

    Args:
        models_dir: Path to the models directory. If None, uses the current module's directory.

    Returns:
        Dict mapping display names to model information containing:
        - 'class': The model class
        - 'module': The module name
        - 'class_name': The class name
    """
    if models_dir is None:
        models_dir = Path(__file__).parent

    available_models = {}

    # Iterate through all Python files in the models directory
    for py_file in models_dir.glob("*.py"):
        if (
            py_file.name.startswith("__")
            or py_file.name == "food_classification_model.py"
            or py_file.name == "model_discovery.py"
        ):
            continue

        try:
            # Import the module dynamically
            module_name = f"src.models.{py_file.stem}"
            module = importlib.import_module(module_name)

            # Find all classes in the module that inherit from FoodClassificationModel
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, FoodClassificationModel)
                    and obj != FoodClassificationModel
                    and obj.__module__ == module_name
                ):

                    # Create a user-friendly name
                    display_name = _create_display_name(name)

                    available_models[display_name] = {
                        "class": obj,
                        "module": module_name,
                        "class_name": name,
                    }

        except Exception as e:
            # In a non-Streamlit context, we might want to log or handle this differently
            print(f"Warning: Could not load model from {py_file.name}: {str(e)}")
            continue

    return available_models


def _create_display_name(class_name: str) -> str:
    """
    Create a user-friendly display name from a class name.

    Args:
        class_name: The original class name

    Returns:
        A user-friendly display name
    """
    # Create a user-friendly name
    display_name = class_name

    if "prithiv" in class_name.lower():
        display_name = "PrithivML Food-101 (Benchmark)"
    elif "resnet" in class_name.lower():
        display_name = "ResNet-18"
    elif "vgg" in class_name.lower():
        display_name = "VGG-16"
    elif "efficientnet" in class_name.lower():
        display_name = "EfficientNet"
    elif "mobilenet" in class_name.lower():
        display_name = "MobileNet"
    elif "densenet" in class_name.lower():
        display_name = "DenseNet"

    return display_name


def get_model_names() -> list:
    """
    Get a list of all available model display names.

    Returns:
        List of model display names
    """
    models = discover_models()
    return list(models.keys())


def get_model_info(display_name: str) -> Dict[str, Any]:
    """
    Get model information for a specific model by display name.

    Args:
        display_name: The display name of the model

    Returns:
        Model information dictionary or None if not found
    """
    models = discover_models()
    return models.get(display_name)
