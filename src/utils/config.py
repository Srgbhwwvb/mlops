import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def create_test_config() -> Dict[str, Any]:
    """Create a minimal test configuration for unit testing."""
    return {
        "data": {
            "train_path": "test_data/*/*.png",
            "test_path": "test_data/*.png",
            "val_size": 0.2,
            "random_seed": 42,
        },
        "model": {"name": "resnet50", "num_classes": 12, "pretrained": True},
        "training": {
            "batch_size": 2,  # Small for testing
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "scheduler_config": {"mode": "min", "factor": 0.5, "patience": 3},
        },
        "transforms": {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "train": {
                "RandomHorizontalFlip": 0.5,
                "RandomVerticalFlip": 0.3,
                "RandomRotation": 30,
            },
            "val": {},
        },
        "output": {
            "log_dir": "./test_logs",
            "model_dir": "./test_models",
            "save_frequency": 1,
        },
    }
