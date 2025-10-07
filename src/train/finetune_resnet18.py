"""
Fine-tuning script for Food-101 classification.

This script trains a ResNet-18 model on the Food-101 dataset.
After training, the model can be loaded using:
    
    from src.models.resnet18 import Resnet18
    model = Resnet18.load_finetuned("models/resnet18-food101")
"""

import argparse
import os
import glob
import io

import numpy as np
import torch
import pandas as pd
from PIL import Image as PILImage
from datasets import Dataset, Features, Image, ClassLabel
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from src.labels import LABELS


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(
    data_dir: str,
    train_samples: int | None = None,
    eval_samples: int | None = None
) -> tuple[Dataset, Dataset]:
    """
    Load parquet-based Food-101 dataset with optional sampling.
    
    Args:
        data_dir: Path to directory containing train-*.parquet and validation-*.parquet files
        train_samples: Number of training samples to use (None = all)
        eval_samples: Number of eval samples to use (None = all)
    """
    print(f"Loading parquet files from: {data_dir}")
    
    train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))
    
    if not train_files or not val_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(train_files)} train files and {len(val_files)} validation files")
    
    def read_parquet_files(
        files: list[str],
        max_samples: int | None = None
    ) -> tuple[list[PILImage.Image], list[int]]:
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
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=LABELS),
    })
    
    train_ds = Dataset.from_dict(
        {"image": train_images, "label": train_labels},
        features=features
    )
    val_ds = Dataset.from_dict(
        {"image": val_images, "label": val_labels},
        features=features
    )
    print("Loaded train and val dataset")
    
    return train_ds, val_ds


def prepare_datasets(
    raw_train: Dataset,
    raw_eval: Dataset,
    image_processor: AutoImageProcessor
) -> tuple[Dataset, Dataset]:
    """Apply transforms to datasets."""
    
    def train_transform(examples: dict[str, any]) -> dict[str, any]:
        images = [img.convert("RGB") for img in examples["image"]]
        processed = image_processor(images, return_tensors="pt")
        processed["labels"] = examples["label"]
        return processed
    
    def eval_transform(examples: dict[str, any]) -> dict[str, any]:
        images = [img.convert("RGB") for img in examples["image"]]
        processed = image_processor(images, return_tensors="pt")
        processed["labels"] = examples["label"]
        return processed
    
    train_ds = raw_train.with_transform(train_transform)
    eval_ds = raw_eval.with_transform(eval_transform)
    
    return train_ds, eval_ds


def build_model(
    model_name: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int]
) -> AutoModelForImageClassification:
    """Load and configure model for fine-tuning."""
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def compute_metrics_fn(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute accuracy metrics."""
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}


def main() -> None:
    """Fine-tune ResNet-18 on Food-101."""
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on Food-101")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str,
                       default="data/raw/food101/data",
                       help="Path to directory with parquet files")
    parser.add_argument("--train_samples", type=int, default=None,
                       help="Number of training samples (None = all)")
    parser.add_argument("--eval_samples", type=int, default=None,
                       help="Number of eval samples (None = all)")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="microsoft/resnet-18")
    parser.add_argument("--output_dir", type=str, default="models/resnet18-food101")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load data from parquet files
    raw_train, raw_eval = load_data(
        args.data_dir,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples
    )
    print("Load data done")
    
    # Get label mappings
    label_names = raw_train.features["label"].names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    num_labels = len(label_names)
    
    print(f"Training on {len(raw_train)} samples")
    print(f"Evaluating on {len(raw_eval)} samples")
    print(f"Number of classes: {num_labels}")
    
    # Load processor and model
    image_processor = AutoImageProcessor.from_pretrained(args.model)
    model = build_model(args.model, num_labels, id2label, label2id)
    
    # Prepare datasets with transforms
    train_ds, eval_ds = prepare_datasets(raw_train, raw_eval, image_processor)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics_fn,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    
    print("Training complete!")
    print("\nTo use this model:")
    print("  from src.models.resnet18 import Resnet18")
    print(f"  model = Resnet18.load_finetuned('{args.output_dir}')")


if __name__ == "__main__":
    main()