"""
Model utilities for training.

This module contains functions for model creation, configuration,
and training utilities.
"""

import random
from typing import Dict

import numpy as np
import torch
from transformers import AutoModelForImageClassification


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]
) -> AutoModelForImageClassification:
    """
    Load and configure model for fine-tuning.

    Args:
        model_name: HuggingFace model name or path
        num_labels: Number of output classes
        id2label: Mapping from class index to label name
        label2id: Mapping from label name to class index

    Returns:
        Configured model ready for fine-tuning
    """
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model
