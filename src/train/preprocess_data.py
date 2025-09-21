"""
Preprocess the data for training.
Includes a custom dataset class for loading the data from parquet files.
Includes a function for getting the data transforms for training and validation.
Includes a function for creating the data loaders for training and validation.
"""

import os
from torch.utils.data import DataLoader, Dataset
from typing import Callable
import glob
import pandas as pd
from PIL import Image
import io
import torchvision.transforms as transforms
import torch


class Food101ParquetDataset(Dataset):
    """Custom Pytorch Dataset for loading Food101 data from parquet files"""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Callable | None = None,
        dev: bool = False,
    ) -> None:
        self.data_path = data_path
        self.transform = transform
        self.split = split
        self.dev = dev
        # Load all parquet files for the specified split
        if split == "train":
            parquet_pattern = os.path.join(data_path, "train-*.parquet")
        else:
            parquet_pattern = os.path.join(data_path, "validation-*.parquet")

        parquet_files = glob.glob(parquet_pattern)

        if not parquet_files:
            raise ValueError(f"No parquet files found for split '{split}' in {data_path}")

        print(f"Found {len(parquet_files)} parquet files for {split} split")

        # Load and concatenate all parquet files
        dfs = []
        for file in sorted(parquet_files):
            df = pd.read_parquet(file)
            dfs.append(df)

        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.data)} samples for {split} split")

        # Get unique labels and create label mapping
        self.labels = sorted(self.data["label"].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.num_classes = len(self.labels)

        print(f"Number of classes: {self.num_classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Convert image bytes to PIL Image
        image_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get label
        label = self.label_to_idx[row["label"]]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Get data transforms for training and validation"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def create_data_loaders(
    data_root: str = "data/raw/food101/data",
    batch_size: int = 32,
    num_workers: int = 4,
    dev: bool = False,
) -> tuple[DataLoader, DataLoader, int]:
    """Create train and validation data loaders from parquet files"""
    train_transform, val_transform = get_data_transforms()

    # Create datasets from parquet files
    train_dataset = Food101ParquetDataset(data_root, split="train", transform=train_transform)
    val_dataset = Food101ParquetDataset(data_root, split="validation", transform=val_transform)

    # Store num_classes before potentially wrapping in Subset
    num_classes = train_dataset.num_classes

    # if dev, use a subset of the data
    if dev:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(1000, len(train_dataset)))
        )
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(200, len(val_dataset))))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, num_classes
