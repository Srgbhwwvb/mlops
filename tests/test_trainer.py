import pytest
import torch
from unittest.mock import Mock, MagicMock
from src.training.trainer import PlantTrainer


class MockModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        return torch.randn(x.shape[0], self.num_classes)

    def train(self):
        pass

    def eval(self):
        pass

    def parameters(self):
        return [torch.nn.Parameter(torch.randn(10, 10))]

    def to(self, device):
        return self


def test_trainer_initialization():
    """Test trainer initialization with mock data."""
    # Mock данные
    model = MockModel(12)
    train_loader = Mock()
    val_loader = Mock()
    device = torch.device("cpu")

    config = {
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "batch_size": 16,
            "epochs": 2,
            "patience": 3,
            "scheduler_config": {"mode": "min", "factor": 0.5, "patience": 2},
        },
        "data": {"val_size": 0.2, "random_seed": 42},
        "transforms": {
            "image_size": 224,
            "train": {
                "RandomHorizontalFlip": 0.5,
                "RandomVerticalFlip": 0.3,
                "RandomRotation": 30,
            },
        },
        "output": {"save_frequency": 1},
    }

    class_names = ["class1", "class2", "class3"]

    trainer = PlantTrainer(model, train_loader, val_loader, device, config, class_names)
    assert trainer is not None


def test_trainer_config_conversion():
    """Test config conversion in trainer."""
    model = MockModel(12)
    train_loader = Mock()
    val_loader = Mock()
    device = torch.device("cpu")

    config = {
        "training": {
            "learning_rate": "0.001",  # строка должна преобразоваться в float
            "weight_decay": "0.001",
            "batch_size": "16",
            "epochs": "2",
            "patience": "3",
            "scheduler_config": {"mode": "min", "factor": "0.5", "patience": "2"},
        },
        "data": {"val_size": "0.2", "random_seed": "42"},
        "transforms": {
            "image_size": "224",
            "train": {
                "RandomHorizontalFlip": "0.5",
                "RandomVerticalFlip": "0.3",
                "RandomRotation": "30",
            },
        },
        "output": {"save_frequency": "1"},
    }

    class_names = ["class1", "class2", "class3"]

    trainer = PlantTrainer(model, train_loader, val_loader, device, config, class_names)

    # Проверяем что конфиг преобразован правильно
    assert isinstance(trainer.config["training"]["learning_rate"], float)
    assert trainer.config["training"]["learning_rate"] == 0.001
