#!/usr/bin/env python3
"""
Food101 evaluation script for model evaluation with MLflow tracking.

This script evaluates models on the Food101 dataset with MLflow experiment tracking.
"""

from typing import List, Dict, Tuple, Any, Union
from pathlib import Path
import mlflow
from datetime import datetime
import pandas as pd
import glob
import random

from src.models.food_classification_model import FoodClassificationModel
from src.labels import LABELS, index_to_label


class Food101Evaluator:
    """Model evaluator for Food101 dataset with MLflow tracking."""

    def __init__(
        self,
        model: FoodClassificationModel,
        experiment_name: str = "food101_evaluation",
        sample_limit: int = 50,
        random_seed: int = 42,
    ):
        """
        Initialize the Food101 evaluator.

        Args:
            model: FoodClassificationModel instance to use for evaluation (required)
            experiment_name: Name of the MLflow experiment
            sample_limit: Maximum number of samples to evaluate
            random_seed: Random seed for reproducible sampling
        """
        self.DATASET_NAME = "Food101"
        self.experiment_name = experiment_name
        self.sample_limit = sample_limit
        self.model = model
        self.random_seed = random_seed
        self.model_name = self.model.__class__.__name__
        self.data_dir = (
            Path(__file__).parent.parent.parent / "data" / "raw" / "food101" / "data"
        )

    def load_validation_data(self) -> List[Tuple[bytes, int]]:
        """
        Load validation data from parquet files with random sampling.

        Returns:
            List of tuples: (image_bytes, true_index)
        """
        random.seed(self.random_seed)

        validation_files = glob.glob(f"{self.data_dir}/validation-*.parquet")
        print(f"Found {len(validation_files)} validation files")

        # Load all samples first
        all_samples = []

        for file_path in validation_files:
            print(f"Loading from {Path(file_path).name}...")
            df = pd.read_parquet(file_path)

            for _, row in df.iterrows():
                image_data = row["image"]["bytes"]
                true_index = row["label"]

                all_samples.append((image_data, true_index))

        print(f"Total available samples: {len(all_samples)}")

        # Randomly sample the requested number of samples
        if len(all_samples) <= self.sample_limit:
            selected_samples = all_samples
            print(f"Using all {len(selected_samples)} available samples")
        else:
            selected_samples = random.sample(all_samples, self.sample_limit)
            print(
                f"Randomly selected {len(selected_samples)} samples from {len(all_samples)} available"
            )

        print(f"Random seed used: {self.random_seed}")
        return selected_samples

    def calculate_accuracy(
        self, predictions: List[Union[int, str]], ground_truths: List[int]
    ) -> float:
        """
        Calculate exact accuracy for Food101 dataset.

        Args:
            predictions: List of predicted indices or label names
            ground_truths: List of true labels

        Returns:
            Accuracy score as float
        """
        if not predictions or not ground_truths:
            return 0.0

        # Check for exact matches
        exact_matches = 0
        for pred, true in zip(predictions, ground_truths):
            if pred == true:
                exact_matches += 1

        return exact_matches / len(predictions)

    def evaluate_model(
        self, samples: List[Tuple[bytes, int, str]], verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the provided samples.

        Args:
            samples: List of (image_bytes, true_index) tuples
            verbose: Whether to print detailed results

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating model on {len(samples)} samples...")

        predictions = []
        ground_truths = []
        prediction_examples = []
        correct_predictions = 0

        for i, (image_bytes, true_index) in enumerate(samples):
            try:
                predicted_index = self.model.classify(image_bytes)
                predictions.append(predicted_index)
                ground_truths.append(true_index)

                # Check if prediction is correct using dataset-specific logic
                is_correct = predicted_index == true_index
                if is_correct:
                    correct_predictions += 1

                # Convert index to label name for display and logging
                predicted_label_name = index_to_label(predicted_index)

                # Store first 10 examples for MLflow
                if i < 10:
                    prediction_examples.append(
                        {
                            "sample_id": i + 1,
                            "true_label": LABELS[true_index],
                            "predicted_label": predicted_label_name,
                            "predicted_index": predicted_index,
                            "true_index": true_index,
                            "is_correct": is_correct,
                        }
                    )

                if verbose and i < 10:  # Print first 10 predictions
                    status = "✓" if is_correct else "✗"
                    print(
                        f"Sample {i+1:2d}: {status} True='{LABELS[true_index]:25s}' (idx: {true_index}) | Predicted='{predicted_label_name}' (idx: {predicted_index})"
                    )

            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                predictions.append("ERROR")
                ground_truths.append(true_index)

        # Calculate metrics
        total_samples = len(samples)
        successful_predictions = len([p for p in predictions if p != "ERROR"])

        # Calculate accuracy using dataset-specific method
        accuracy = self.calculate_accuracy(predictions, ground_truths)
        success_rate = (
            successful_predictions / total_samples if total_samples > 0 else 0
        )

        results = {
            "total_samples": total_samples,
            "successful_predictions": successful_predictions,
            "correct_predictions": correct_predictions,
            "success_rate": success_rate,
            "accuracy": accuracy,
            "prediction_examples": prediction_examples,
        }

        return results

    def log_mlflow_metrics(self, results: Dict[str, Any]) -> None:
        """
        Log evaluation metrics to MLflow.

        Args:
            results: The results from the evaluation.
        """
        mlflow.log_metric("total_samples", results["total_samples"])
        mlflow.log_metric("successful_predictions", results["successful_predictions"])
        mlflow.log_metric("success_rate", results["success_rate"])
        mlflow.log_metric("correct_predictions", results["correct_predictions"])
        mlflow.log_metric("accuracy", results["accuracy"])

    def log_mlflow_artifacts(self, results: Dict[str, Any]) -> None:
        """
        Log evaluation artifacts to MLflow.

        Args:

        """
        examples_data = []
        for example in results["prediction_examples"]:
            status = "✓" if example.get("is_correct", False) else "✗"
            examples_data.append(
                f"Sample {example['sample_id']}: {status} {example['true_label']} -> {example['predicted_label']}"
            )

        examples_text = "\n".join(examples_data)
        examples_file = f"{self.DATASET_NAME.lower()}_evaluation_examples.txt"
        with open(examples_file, "w") as f:
            f.write(examples_text)
        mlflow.log_artifact(examples_file)

        model_source = (
            getattr(self.model, "model_path", "N/A")
            if hasattr(self.model, "model_path")
            else "N/A"
        )
        summary = f"""{self.model_name} {self.DATASET_NAME} Evaluation Summary
            ========================================={'=' * len(self.DATASET_NAME)}
            Model: {self.model_name} ({model_source})
            Dataset: {self.DATASET_NAME} validation set
            Samples: {results['total_samples']}
            Success Rate: {results['success_rate']:.2%}
            Accuracy: {results['accuracy']:.2%}
            Correct Predictions: {results['correct_predictions']}
        """

        summary_file = f"{self.DATASET_NAME.lower()}_evaluation_summary.txt"
        with open(summary_file, "w") as f:
            f.write(summary)
        mlflow.log_artifact(summary_file)

        # Clean up temporary files
        Path(examples_file).unlink(missing_ok=True)
        Path(summary_file).unlink(missing_ok=True)

    def run_evaluation(self) -> None:
        """Run the complete evaluation pipeline with MLflow tracking."""
        print("=" * 60)
        print(f"{self.model_name} {self.DATASET_NAME} Evaluation with MLflow")
        print("=" * 60)

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("model_class", self.model.__class__.__name__)
            mlflow.log_param("dataset", self.DATASET_NAME)
            mlflow.log_param("sample_limit", self.sample_limit)
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("evaluation_date", datetime.now().isoformat())

            # Log model-specific parameters if available
            if hasattr(self.model, "model_path"):
                mlflow.log_param(
                    "model_source", getattr(self.model, "model_path", "Unknown")
                )
            if hasattr(self.model, "preprocessor_path"):
                mlflow.log_param(
                    "preprocessor_path",
                    getattr(self.model, "preprocessor_path", "Unknown"),
                )

            samples = self.load_validation_data()

            if not samples:
                print(
                    f"No validation samples loaded. Check the {self.DATASET_NAME} dataset connection."
                )
                mlflow.log_param("status", "failed - no data")
                return

            mlflow.log_param("samples_loaded", len(samples))

            results = self.evaluate_model(samples, verbose=True)

            self.log_mlflow_metrics(results)
            self.log_mlflow_artifacts(results)

            self._print_results(results)

    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print evaluation results to console."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total samples processed:       {results['total_samples']}")
        print(f"Successful predictions:        {results['successful_predictions']}")
        print(f"Success rate:                  {results['success_rate']:.2%}")
        print(f"Correct predictions:           {results['correct_predictions']}")
        print(f"Accuracy:                      {results['accuracy']:.2%}")

        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
