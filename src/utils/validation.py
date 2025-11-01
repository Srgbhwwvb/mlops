import os
import yaml
from typing import Dict, Any, List
import numpy as np
from PIL import Image


def validate_config_structure(config: Dict[str, Any]) -> List[str]:
    """Validate configuration structure and return list of errors."""
    errors = []

    required_sections = ["data", "model", "training", "transforms", "output"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Check data section
    if "data" in config:
        data = config["data"]
        if "train_path" not in data:
            errors.append("Missing data.train_path")
        if "val_size" not in data:
            errors.append("Missing data.val_size")
        elif not (0 < data["val_size"] < 1):
            errors.append("data.val_size must be between 0 and 1")

    # Check training section
    if "training" in config:
        training = config["training"]
        required_training = ["batch_size", "epochs", "learning_rate"]
        for param in required_training:
            if param not in training:
                errors.append(f"Missing training.{param}")

    return errors


def validate_image_file(file_path: str) -> bool:
    """Validate that file is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False


def validate_dataset_structure(
    dataset_path: str, expected_classes: List[str]
) -> List[str]:
    """Validate dataset structure and return list of errors."""
    errors = []

    if not os.path.exists(dataset_path):
        errors.append(f"Dataset path does not exist: {dataset_path}")
        return errors

    # Check if it's a directory with class subdirectories
    for class_name in expected_classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            errors.append(f"Missing class directory: {class_name}")
            continue

        # Check for images in class directory
        image_files = [
            f
            for f in os.listdir(class_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            errors.append(f"No images found in class directory: {class_name}")
        else:
            # Validate first few images
            for img_file in image_files[:3]:
                img_path = os.path.join(class_path, img_file)
                if not validate_image_file(img_path):
                    errors.append(f"Invalid image file: {img_path}")

    return errors
