import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from typing import Optional, Callable, List, Tuple, Dict, Any
import torch
import logging
from collections import Counter


class PlantDataset(Dataset):
    """Dataset for plant seedlings classification."""

    LABELS = [
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
        self, img_dir: str, transform: Optional[Callable] = None, is_train: bool = True
    ):
        self.img_paths = glob.glob(img_dir)
        self.transform = transform
        self.is_train = is_train

        if not self.img_paths:
            raise ValueError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label_name = img_path.split(os.sep)[-2]
            label = self.LABELS.index(label_name)
            return image, label
        else:
            image_name = os.path.basename(img_path)
            return image_name, image

    def get_sample_item(self, idx: int = 0) -> Tuple:
        """Get sample item for testing without file I/O."""
        if idx >= len(self):
            raise IndexError("Index out of range")

        # Return mock data for testing - create a proper tensor
        if self.is_train:
            # Create a proper image tensor with correct shape and type
            mock_image = torch.rand(3, 224, 224, dtype=torch.float32)
            mock_label = 0
            return mock_image, mock_label
        else:
            # For test mode
            mock_image = torch.rand(3, 224, 224, dtype=torch.float32)
            mock_name = "test_image.png"
            return mock_name, mock_image

    def print_basic_stats(self):
        """Print basic dataset statistics."""
        if not self.img_paths:
            logging.warning("No images found for statistics")
            return

        # Сбор статистики по классам
        class_counts = Counter()
        total_images = len(self.img_paths)

        for img_path in self.img_paths:
            if self.is_train:
                class_name = img_path.split(os.sep)[-2]
                class_counts[class_name] += 1

        # Логируем статистику
        logging.info("DATASET STATISTICS:")
        logging.info(f"  Total images: {total_images}")
        logging.info(f"  Number of classes: {len(class_counts)}")

        if self.is_train:
            logging.info("  Class distribution:")
            for class_name, count in class_counts.most_common():
                percentage = (count / total_images) * 100
                logging.info(f"    {class_name}: {count} images ({percentage:.1f}%)")

            # Проверяем дисбаланс
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                imbalance_ratio = (
                    max_count / min_count if min_count > 0 else float("inf")
                )
                logging.info(f"  Class imbalance ratio: {imbalance_ratio:.2f}")

    @classmethod
    def get_class_names(cls) -> List[str]:
        """Get list of class names for testing."""
        return cls.LABELS

    def get_sample_item(self, idx: int = 0) -> Tuple:
        """Get sample item for testing without file I/O."""
        if idx >= len(self):
            raise IndexError("Index out of range")

        # Return mock data for testing
        if self.is_train:
            return (torch.rand(3, 224, 224), 0)  # Mock image and label
        else:
            return ("test_image.png", torch.rand(3, 224, 224))


def create_data_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train and validation data loaders."""
    from .preprocessing import create_train_transforms, create_val_transforms

    # Define transforms using the new preprocessing module
    train_transform = create_train_transforms(config)
    val_transform = create_val_transforms(config)

    # Load dataset
    full_dataset = PlantDataset(
        config["data"]["train_path"], transform=train_transform, is_train=True
    )

    full_dataset.print_basic_stats()

    # Split dataset
    train_size = int((1 - config["data"]["val_size"]) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply val transform to validation set
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 4),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 4),
    )

    return train_loader, val_loader, full_dataset.LABELS
