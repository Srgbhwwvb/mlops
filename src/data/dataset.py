import glob
import logging
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from config import DataConfig, TrainingConfig, TransformConfig

from .preprocessing import create_train_transforms, create_val_transforms


class PlantDataset(Dataset):
    """Dataset for plant seedlings classification."""

    LABELS: list[str] = [
        "Black-grass",
        "Charlock",
        "Cleavers",
        "Common Chickweed",
        "Common wheat",
        "Fat Hen",
        "Loose Silky-bent",
        "Maize",
        "Scentless Mayweed",
        "Shepherds Purse",
        "Small-flowered Cranesbill",
        "Sugar beet",
    ]

    def __init__(
        self,
        logger: logging.Logger,
        img_dir: Path,
        transform: Callable | None = None,
        *,
        is_train: bool = True,
    ) -> None:
        self.logger = logger

        self.img_paths = [Path(x) for x in glob.glob(str(img_dir))]

        if not self.img_paths:
            msg = f"No images found in {img_dir}"
            raise ValueError(msg)

        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        img_path: Path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label_name = img_path.parts[-2]
            label = self.LABELS.index(label_name)
            return image, label
        image_name = img_path.name
        return image_name, image

    def print_basic_stats(self) -> None:
        """Print basic dataset statistics."""
        if not self.img_paths:
            self.logger.warning("No images found for statistics")
            return

        # Picking up statistics of classes
        class_counts = Counter()
        total_images = len(self.img_paths)

        for img_path in self.img_paths:
            if self.is_train:
                class_name = img_path.parts[-2]
                class_counts[class_name] += 1

        # Statistics logging
        self.logger.info("DATASET STATISTICS:")
        self.logger.info(f"  Total images: {total_images}")
        self.logger.info(f"  Number of classes: {len(class_counts)}")

        if self.is_train:
            self.logger.info("  Class distribution:")
            for class_name, count in class_counts.most_common():
                percentage = (count / total_images) * 100
                self.logger.info(
                    f"    {class_name}: {count} images ({percentage:.1f}%)"
                )

            # Check disbalance
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                imbalance_ratio = (
                    max_count / min_count if min_count > 0 else float("inf")
                )
                self.logger.info(f"  Class imbalance ratio: {imbalance_ratio:.2f}")

    @classmethod
    def get_class_names(cls) -> list[str]:
        """Get list of class names for testing."""
        return cls.LABELS

    def get_sample_item(self, idx: int = 0) -> tuple:
        """Get sample item for testing without file I/O."""
        if idx >= len(self):
            msg = "Index out of range"
            raise IndexError(msg)

        # Return mock data for testing
        if self.is_train:
            return (torch.rand(3, 224, 224), 0)  # Mock image and label
        return ("test_image.png", torch.rand(3, 224, 224))


def create_data_loaders(
    data_config: DataConfig,
    training_config: TrainingConfig,
    transform_config: TransformConfig,
    logger: logging.Logger,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create train and validation data loaders."""

    # Define transforms using the new preprocessing module
    train_transform = create_train_transforms(transform_config)
    val_transform = create_val_transforms(transform_config)

    # Load dataset
    full_dataset = PlantDataset(
        logger,
        data_config.train_path,
        transform=train_transform,
        is_train=True,
    )

    full_dataset.print_basic_stats()

    # Split dataset
    train_size = int((1 - data_config.val_size) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    dataset_parts = random_split(full_dataset, [train_size, val_size])
    train_dataset = dataset_parts[0]
    val_dataset = dataset_parts[1]

    # Apply val transform to validation set
    val_dataset.dataset.transform = val_transform  # ty:ignore[unresolved-attribute]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 4),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 4),
    )

    return train_loader, val_loader, PlantDataset.LABELS
