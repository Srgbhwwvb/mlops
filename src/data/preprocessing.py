from torchvision import transforms

from config import TransformConfig


def create_train_transforms(config: TransformConfig) -> transforms.Compose:
    """Create training transforms with data augmentation."""
    # Convert string values if necessary
    image_size = config.image_size
    random_horizontal_flip = config.random_horizontal_flip
    rotation = config.random_rotation

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(random_horizontal_flip),
            transforms.RandomRotation(rotation),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.mean,
                std=config.std,
            ),
        ],
    )


def create_val_transforms(config: TransformConfig) -> transforms.Compose:
    """Create validation transforms without augmentation."""
    image_size = config.image_size

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.mean,
                std=config.std,
            ),
        ],
    )


def create_test_transforms(config: TransformConfig) -> transforms.Compose:
    """Create test transforms (same as validation)."""
    return create_val_transforms(config)
