from typing import Dict, Any
import torchvision.transforms as transforms


def _convert_config_value(value):
    """Convert string values to appropriate types for transforms."""
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    return value


def create_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """Create training transforms with data augmentation."""
    # Convert string values if necessary
    image_size = _convert_config_value(config["transforms"]["image_size"])
    flip_prob = _convert_config_value(
        config["transforms"]["train"]["RandomHorizontalFlip"]
    )
    rotation = _convert_config_value(config["transforms"]["train"]["RandomRotation"])

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(flip_prob),
            transforms.RandomRotation(rotation),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["transforms"]["mean"], std=config["transforms"]["std"]
            ),
        ]
    )


def create_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """Create validation transforms without augmentation."""
    image_size = _convert_config_value(config["transforms"]["image_size"])

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["transforms"]["mean"], std=config["transforms"]["std"]
            ),
        ]
    )


def create_test_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """Create test transforms (same as validation)."""
    return create_val_transforms(config)
