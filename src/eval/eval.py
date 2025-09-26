#!/usr/bin/env python3
"""
Example script demonstrating how to evaluate multiple models on different datasets.

This script shows how to use the enhanced evaluation framework
with different model implementations including VGG16, ResNet18, and PrithivMlFood101.
"""
from src.models.vgg16 import VGG16
from src.models.resnet18 import Resnet18
from src.models.prithiv_ml_food101 import PrithivMlFood101
from src.eval.evaluate_food101 import Food101Evaluator
from src.models.food_classification_model import FoodClassificationModel


def evaluate_food101(
    model: FoodClassificationModel,
    experiment_name: str = "food101_evaluation",
    sample_limit: int = 50,
    random_seed: int = 42,
):
    """Main evaluation function."""
    evaluator = Food101Evaluator(model, experiment_name, sample_limit, random_seed)
    evaluator.run_evaluation()


def main():
    """Demonstrate evaluation with multiple model architectures."""

    print("=" * 90)
    print("Multi-Model Evaluation: VGG16 vs ResNet-18 vs PrithivMlFood101")

    # Food101 Evaluations
    print("\n=== Food101 Evaluations ===")

    print("\n1. Evaluating PrithivMlFood101 on Food101...")
    prithiv_model = PrithivMlFood101()
    evaluate_food101(
        experiment_name="prithivml_food101_evaluation",
        sample_limit=50,  # Small sample for demonstration
        model=prithiv_model,
    )

    print("\n2. Evaluating ResNet-18 on Food101 ...")
    resnet18_food_model = Resnet18()
    evaluate_food101(
        experiment_name="resnet18_food101_evaluation",
        sample_limit=50,  # Small sample for demonstration
        model=resnet18_food_model,
    )

    print("\n3. Evaluating VGG16 on Food101 ...")
    vgg16_food_model = VGG16()
    evaluate_food101(
        experiment_name="vgg16_food101_evaluation",
        sample_limit=50,  # Small sample for demonstration
        model=vgg16_food_model,
    )


if __name__ == "__main__":
    main()
