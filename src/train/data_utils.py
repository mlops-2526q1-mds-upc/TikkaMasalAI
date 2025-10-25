"""
Data loading and preprocessing utilities for training.

This module contains functions for loading and preparing datasets
for fine-tuning tasks.
"""

import glob
import io
import os
from typing import Optional, Tuple

from datasets import ClassLabel, Dataset, Features, Image
import pandas as pd
from PIL import Image as PILImage
from transformers import AutoImageProcessor

from src.labels import LABELS


def load_data(
    data_dir: str, train_samples: Optional[int] = None, eval_samples: Optional[int] = None
) -> Tuple[Dataset, Dataset]:
    """
    Load parquet-based Food-101 dataset with optional sampling.

    Args:
        data_dir: Path to directory containing train-*.parquet and validation-*.parquet files
        train_samples: Number of training samples to use (None = all)
        eval_samples: Number of eval samples to use (None = all)
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
        
    Raises:
        ValueError: If no parquet files found in data_dir
    """
    print(f"Loading parquet files from: {data_dir}")

    train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))

    if not train_files or not val_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    print(f"Found {len(train_files)} train files and {len(val_files)} validation files")

    def read_parquet_files(
        files: list[str], max_samples: Optional[int] = None
    ) -> Tuple[list[PILImage.Image], list[int]]:
        """Read images and labels from parquet files."""
        images = []
        labels = []

        for fp in files:
            if max_samples and len(images) >= max_samples:
                break

            df = pd.read_parquet(fp)
            for _, row in df.iterrows():
                if max_samples and len(images) >= max_samples:
                    break

                img_bytes = row["image"]["bytes"]
                label = int(row["label"])
                pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(pil_img)
                labels.append(label)

        return images, labels

    train_images, train_labels = read_parquet_files(train_files, train_samples)
    val_images, val_labels = read_parquet_files(val_files, eval_samples)

    print(f"Loaded {len(train_images)} training samples")
    print(f"Loaded {len(val_images)} validation samples")

    # Create datasets with proper features
    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(names=LABELS),
        }
    )

    train_ds = Dataset.from_dict({"image": train_images, "label": train_labels}, features=features)
    val_ds = Dataset.from_dict({"image": val_images, "label": val_labels}, features=features)
    print("Loaded train and val dataset")

    return train_ds, val_ds


def prepare_datasets(
    raw_train: Dataset, raw_eval: Dataset, image_processor: AutoImageProcessor
) -> Tuple[Dataset, Dataset]:
    """
    Apply transforms to datasets for training.

    Args:
        raw_train: Raw training dataset
        raw_eval: Raw evaluation dataset
        image_processor: Image processor for preprocessing

    Returns:
        Tuple of (processed_train_dataset, processed_eval_dataset)
    """
    def train_transform(examples: dict[str, any]) -> dict[str, any]:
        """Transform function for training data."""
        images = [img.convert("RGB") for img in examples["image"]]
        processed = image_processor(images, return_tensors="pt")
        processed["labels"] = examples["label"]
        return processed

    def eval_transform(examples: dict[str, any]) -> dict[str, any]:
        """Transform function for evaluation data."""
        images = [img.convert("RGB") for img in examples["image"]]
        processed = image_processor(images, return_tensors="pt")
        processed["labels"] = examples["label"]
        return processed

    train_ds = raw_train.with_transform(train_transform)
    eval_ds = raw_eval.with_transform(eval_transform)

    return train_ds, eval_ds
