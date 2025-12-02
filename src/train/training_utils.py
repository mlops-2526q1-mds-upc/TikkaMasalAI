"""
Training utilities and metrics.

This module contains functions for training configuration,
metrics computation, and training execution.
"""

import os
from typing import Dict, Tuple

from codecarbon import EmissionsTracker
import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments

from src.train.config import TrainingParams


def compute_metrics_fn(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute accuracy metrics for evaluation.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary containing computed metrics
    """
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}


def create_training_args(params: TrainingParams) -> TrainingArguments:
    """
    Create TrainingArguments from configuration parameters.

    Args:
        params: Training parameters configuration

    Returns:
        Configured TrainingArguments object
    """
    return TrainingArguments(
        output_dir=params.get_output_dir(),
        num_train_epochs=params.training.epochs,
        per_device_train_batch_size=params.training.batch_size,
        per_device_eval_batch_size=params.training.batch_size * 2,
        learning_rate=params.training.learning_rate,
        weight_decay=params.training.weight_decay,
        warmup_ratio=params.training.warmup_ratio,
        fp16=params.training.fp16,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=params.output.logging_steps,
        save_total_limit=params.output.save_total_limit,
        report_to=params.output.report_to,
    )


def train_with_emissions_tracking(
    trainer: Trainer, params: TrainingParams, emissions_output_dir: str = "reports"
) -> None:
    """
    Train model with emissions tracking.

    Args:
        trainer: Configured trainer object
        params: Training parameters
        emissions_output_dir: Directory to save emissions data
    """
    print("Starting training...")
    tracker = EmissionsTracker(output_dir=emissions_output_dir)
    tracker.start()

    try:
        trainer.train()
    finally:
        tracker.stop()

    print("Training complete!")


def save_trained_model(trainer: Trainer, params: TrainingParams) -> None:
    """
    Save the trained model and processor.

    Args:
        trainer: Trained trainer object
        params: Training parameters
    """
    output_dir = params.get_output_dir()
    print(f"Saving model to {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model and processor
    trainer.save_model(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)

    print("\nTo use this model:")
    print("  from src.models.resnet18 import Resnet18")
    print(f"  model = Resnet18.load_finetuned('{output_dir}')")
