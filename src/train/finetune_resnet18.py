"""
Fine-tuning script for Food-101 classification.

This script trains a ResNet-18 model on the Food-101 dataset using a modular,
configuration-based approach. It supports both command-line arguments and
YAML configuration files.

After training, the model can be loaded using:

    from src.models.resnet18 import Resnet18
    model = Resnet18.load_finetuned("models/resnet18-food101")

Usage:
    # Using command line arguments (original way)
    uv run -m src.train.finetune_resnet18 --epochs 3 --train_samples 10000 --eval_samples 2000
    
    # Using configuration file
    uv run -m src.train.finetune_resnet18 --config configs/training_quick.yaml
    
    # Override config parameters
    uv run -m src.train.finetune_resnet18 --config configs/training_quick.yaml --epochs 5
"""

import argparse
import os
import sys

from transformers import AutoImageProcessor, Trainer

from src.train.config import TrainingParams, load_config
from src.train.data_utils import load_data, prepare_datasets
from src.train.model_utils import build_model, set_seed
from src.train.training_utils import (
    compute_metrics_fn,
    create_training_args,
    save_trained_model,
    train_with_emissions_tracking,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet-18 on Food-101 with configuration support"
    )
    
    # Configuration file argument
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (overrides other arguments)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/food101/data",
        help="Path to directory with parquet files"
    )
    parser.add_argument(
        "--train_samples", 
        type=int, 
        default=None, 
        help="Number of training samples (None = all)"
    )
    parser.add_argument(
        "--eval_samples", 
        type=int, 
        default=None, 
        help="Number of eval samples (None = all)"
    )

    # Model arguments
    parser.add_argument("--model", type=str, default="microsoft/resnet-18")
    parser.add_argument("--output_dir", type=str, default=None)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def create_params_from_args(args: argparse.Namespace) -> TrainingParams:
    """Create TrainingParams from command line arguments."""
    from src.train.config import DataConfig, ModelConfig, OutputConfig, TrainingConfig
    
    return TrainingParams(
        data=DataConfig(
            data_dir=args.data_dir,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
        ),
        model=ModelConfig(
            model_name=args.model,
            model_type="resnet18",  # Default for this script
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            fp16=args.fp16,
            seed=args.seed,
        ),
        output=OutputConfig(
            output_dir=args.output_dir,
        ),
    )


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}")
            sys.exit(1)
        
        print(f"Loading configuration from: {args.config}")
        params = load_config(args.config)
        
        # Override with command line arguments if provided
        if args.epochs != 5:  # Check if non-default value provided
            params.training.epochs = args.epochs
        if args.train_samples is not None:
            params.data.train_samples = args.train_samples
        if args.eval_samples is not None:
            params.data.eval_samples = args.eval_samples
        if args.output_dir is not None:
            params.output.output_dir = args.output_dir
    else:
        # Use command line arguments
        params = create_params_from_args(args)
    
    # Print configuration
    print("Training Configuration:")
    print(f"  Model: {params.model.model_name}")
    print(f"  Data dir: {params.data.data_dir}")
    print(f"  Train samples: {params.data.train_samples or 'all'}")
    print(f"  Eval samples: {params.data.eval_samples or 'all'}")
    print(f"  Epochs: {params.training.epochs}")
    print(f"  Batch size: {params.training.batch_size}")
    print(f"  Learning rate: {params.training.learning_rate}")
    print(f"  Output dir: {params.get_output_dir()}")
    print()
    
    # Set random seed
    set_seed(params.training.seed)

    # Load data
    raw_train, raw_eval = load_data(
        params.data.data_dir, 
        train_samples=params.data.train_samples, 
        eval_samples=params.data.eval_samples
    )
    print("Data loading complete")

    # Get label mappings
    label_names = raw_train.features["label"].names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    num_labels = len(label_names)

    print(f"Training on {len(raw_train)} samples")
    print(f"Evaluating on {len(raw_eval)} samples")
    print(f"Number of classes: {num_labels}")

    # Load processor and model
    image_processor = AutoImageProcessor.from_pretrained(params.model.model_name)
    model = build_model(params.model.model_name, num_labels, id2label, label2id)

    # Prepare datasets with transforms
    train_ds, eval_ds = prepare_datasets(raw_train, raw_eval, image_processor)

    # Create training arguments
    training_args = create_training_args(params)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics_fn,
    )

    # Train with emissions tracking
    train_with_emissions_tracking(trainer, params)

    # Save final model
    save_trained_model(trainer, params)


if __name__ == "__main__":
    main()
