import argparse
import os
import random
from typing import Any, Dict, Tuple
import glob
import io

import numpy as np  # moved heavy imports later where possible
print(f"Numpy: {np}")
import torch  # some tools may flag missing deps; ensure env has torch
print(f"Torch: {torch}")
from datasets import load_dataset, Dataset, Features, Image, ClassLabel
import pandas as pd
from PIL import Image as PILImage
from src.labels import LABELS
print(f"Datasets: {load_dataset}")
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
print(f"Transformers: {AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_splits(data_dir: str | None) -> Tuple[str, str]:
    """
    Decide which dataset loading mode to use and return split names.

    If data_dir provided, assume imagefolder layout with train/ and validation/ or val/.
    Otherwise, use the built-in food101 dataset splits (train/test with provided validation split ratio).
    """
    if data_dir:
        train_split = "train"
        # common conventions: validation or val
        val_dir = "validation" if os.path.isdir(os.path.join(data_dir, "validation")) else "val"
        eval_split = val_dir
        return train_split, eval_split
    else:
        # food101 provides train/test; we'll treat test as eval unless user overrides with --validation_ratio
        return "train", "test"


def build_label_mappings(dataset) -> Tuple[Dict[int, str], Dict[str, int]]:
    # datasets imagefolder exposes features["label"].names when class names are discovered
    label_feature = dataset.features.get("label")
    if label_feature is None or not hasattr(label_feature, "names"):
        # fallback to alphabetical from folder names via cast
        unique_labels = sorted(list({int(x) for x in dataset.unique("label")}))
        id2label = {i: str(i) for i in unique_labels}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id
    id2label = {i: name for i, name in enumerate(label_feature.names)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def load_parquet_food101_as_datasets(root_dir: str) -> Tuple[Dataset, Dataset]:
    data_dir = os.path.join(root_dir, "data")
    train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))

    def read_rows(files):
        images: list[PILImage.Image] = []
        labels: list[int] = []
        for fp in files:
            df = pd.read_parquet(fp)
            for _, row in df.iterrows():
                img_bytes = row["image"]["bytes"]
                label = int(row["label"])  # already index in dataset
                pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(pil_img)
                labels.append(label)
        return images, labels

    train_images, train_labels = read_rows(train_files)
    val_images, val_labels = read_rows(val_files)

    features = Features({
        "image": Image(),
        "label": ClassLabel(names=LABELS),
    })

    train_ds = Dataset.from_dict({"image": train_images, "label": train_labels}, features=features)
    val_ds = Dataset.from_dict({"image": val_images, "label": val_labels}, features=features)
    return train_ds, val_ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on Food-101 or imagefolder data")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to imagefolder dataset root (with train/ and val/)")
    parser.add_argument("--output_dir", type=str, default="models/resnet18-food101", help="Output directory for checkpoints")
    parser.add_argument("--pretrained_model", type=str, default="microsoft/resnet-18", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Arguments: {args}")

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16")

    if args.data_dir is None:
        dataset_name = "food101"
        train_split, eval_split = "train", "validation"
        # food101 in datasets uses split names train/test; create validation from train if needed
        # We'll default to test as eval to keep it simple here
        train_split, eval_split = "train", "test"
        raw_train = load_dataset(dataset_name, split=train_split)
        raw_eval = load_dataset(dataset_name, split=eval_split)
    else:
        parquet_dir = os.path.join(args.data_dir, "data")
        if os.path.isdir(parquet_dir) and (glob.glob(os.path.join(parquet_dir, "train-*.parquet"))):
            # load parquet-based dataset (bytes -> PIL)
            raw_train, raw_eval = load_parquet_food101_as_datasets(args.data_dir)
        else:
            # load local imagefolder with train/ and val/ subdirs
            train_split, eval_split = resolve_splits(args.data_dir)
            raw_train = load_dataset("imagefolder", data_dir=os.path.join(args.data_dir, train_split), split="train")
            raw_eval = load_dataset("imagefolder", data_dir=os.path.join(args.data_dir, eval_split), split="train")

    image_processor = AutoImageProcessor.from_pretrained(args.pretrained_model)

    id2label, label2id = build_label_mappings(raw_train)
    num_labels = len(id2label)
    print(f"Number of labels: {num_labels}")

    model = AutoModelForImageClassification.from_pretrained(
        args.pretrained_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    print(f"Model: {model}")

    def train_transform(examples: Dict[str, Any]) -> Dict[str, Any]:
        images = [image.convert("RGB") for image in examples["image"]]
        processed = image_processor(images=images, do_center_crop=False, return_tensors="pt")
        processed["labels"] = examples["label"]
        return processed

    def eval_transform(examples: Dict[str, Any]) -> Dict[str, Any]:
        images = [image.convert("RGB") for image in examples["image"]]
        processed = image_processor(images=images, do_center_crop=True, return_tensors="pt")
        processed["labels"] = examples["label"]
        return processed

    train_ds = raw_train.with_transform(train_transform)
    eval_ds = raw_eval.with_transform(eval_transform)

    import evaluate
    accuracy = evaluate.load("accuracy")
    print(f"Accuracy: {accuracy}")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        evaluation_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=[args.report_to] if args.report_to else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        label_smoother=None if args.label_smoothing <= 0 else None,  # explicit label smoothing can be added if needed
    )

    trainer.train()

    # Save the best model and processor
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    print("Starting main")
    main()


