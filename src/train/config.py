"""
Configuration management for training scripts.

This module provides utilities for loading and managing training parameters
from YAML configuration files.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    data_dir: str = "data/raw/food101/data"
    train_samples: Optional[int] = None
    eval_samples: Optional[int] = None


@dataclass
class ModelConfig:
    """Configuration for model architecture and loading."""
    
    model_name: str = "microsoft/resnet-18"
    model_type: str = "resnet18"  # Used for naming and model selection
    num_labels: Optional[int] = None  # Will be inferred from dataset


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = False
    seed: int = 42


@dataclass
class OutputConfig:
    """Configuration for output and logging."""
    
    output_dir: Optional[str] = None  # Will be auto-generated if None
    base_output_dir: str = "models"
    save_total_limit: int = 2
    logging_steps: int = 50
    report_to: str = "none"


@dataclass
class TrainingParams:
    """Complete training parameters container."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def generate_model_name(self) -> str:
        """
        Generate a descriptive model name based on parameters.
        
        Returns:
            str: Generated model name (e.g., "resnet18-food101-3e-10k-2k")
        """
        # Use model_type for consistent naming
        model_type = self.model.model_type
        
        # Build parameter string
        parts = [model_type, "food101"]
        
        # Add epochs
        if self.training.epochs != 5:  # Only add if not default
            parts.append(f"{self.training.epochs}e")
        
        # Add sample sizes
        if self.data.train_samples is not None:
            parts.append(f"{self.data.train_samples // 1000}k")
        if self.data.eval_samples is not None:
            parts.append(f"{self.data.eval_samples // 1000}k")
        
        # Add learning rate if not default
        if self.training.learning_rate != 5e-4:
            lr_str = f"lr{self.training.learning_rate:.0e}".replace("e-0", "e-")
            parts.append(lr_str)
        
        return "-".join(parts)
    
    def get_output_dir(self) -> str:
        """
        Get the full output directory path.
        
        Returns:
            str: Complete output directory path
        """
        if self.output.output_dir is not None:
            return self.output.output_dir
        
        model_name = self.generate_model_name()
        return os.path.join(self.output.base_output_dir, model_name)


def load_config(config_path: str) -> TrainingParams:
    """
    Load training parameters from a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        TrainingParams: Loaded configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return _dict_to_params(config_dict)


def _dict_to_params(config_dict: Dict[str, Any]) -> TrainingParams:
    """
    Convert dictionary to TrainingParams object.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        TrainingParams: Configuration object
    """
    # Create base params with defaults
    params = TrainingParams()
    
    # Update data config
    if 'data' in config_dict:
        data_config = config_dict['data']
        params.data = DataConfig(
            data_dir=data_config.get('data_dir', params.data.data_dir),
            train_samples=data_config.get('train_samples', params.data.train_samples),
            eval_samples=data_config.get('eval_samples', params.data.eval_samples),
        )
    
    # Update model config
    if 'model' in config_dict:
        model_config = config_dict['model']
        params.model = ModelConfig(
            model_name=model_config.get('model_name', params.model.model_name),
            num_labels=model_config.get('num_labels', params.model.num_labels),
        )
    
    # Update training config
    if 'training' in config_dict:
        training_config = config_dict['training']
        params.training = TrainingConfig(
            epochs=training_config.get('epochs', params.training.epochs),
            batch_size=training_config.get('batch_size', params.training.batch_size),
            learning_rate=training_config.get('learning_rate', params.training.learning_rate),
            weight_decay=training_config.get('weight_decay', params.training.weight_decay),
            warmup_ratio=training_config.get('warmup_ratio', params.training.warmup_ratio),
            fp16=training_config.get('fp16', params.training.fp16),
            seed=training_config.get('seed', params.training.seed),
        )
    
    # Update output config
    if 'output' in config_dict:
        output_config = config_dict['output']
        params.output = OutputConfig(
            output_dir=output_config.get('output_dir', params.output.output_dir),
            base_output_dir=output_config.get('base_output_dir', params.output.base_output_dir),
            save_total_limit=output_config.get('save_total_limit', params.output.save_total_limit),
            logging_steps=output_config.get('logging_steps', params.output.logging_steps),
            report_to=output_config.get('report_to', params.output.report_to),
        )
    
    return params


def save_config(params: TrainingParams, config_path: str) -> None:
    """
    Save training parameters to a YAML configuration file.
    
    Args:
        params: Training parameters to save
        config_path: Path where to save the configuration
    """
    config_dict = {
        'data': {
            'data_dir': params.data.data_dir,
            'train_samples': params.data.train_samples,
            'eval_samples': params.data.eval_samples,
        },
        'model': {
            'model_name': params.model.model_name,
            'num_labels': params.model.num_labels,
        },
        'training': {
            'epochs': params.training.epochs,
            'batch_size': params.training.batch_size,
            'learning_rate': params.training.learning_rate,
            'weight_decay': params.training.weight_decay,
            'warmup_ratio': params.training.warmup_ratio,
            'fp16': params.training.fp16,
            'seed': params.training.seed,
        },
        'output': {
            'output_dir': params.output.output_dir,
            'base_output_dir': params.output.base_output_dir,
            'save_total_limit': params.output.save_total_limit,
            'logging_steps': params.output.logging_steps,
            'report_to': params.output.report_to,
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
