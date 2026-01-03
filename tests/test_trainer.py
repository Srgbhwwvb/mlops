from unittest.mock import Mock

import torch

from config import Config, create_test_config, create_test_config_dict
from training import PlantTrainer


class MockModel(torch.nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        return torch.randn(x.shape[0], self.num_classes)

    def train(self):  # ty:ignore[invalid-method-override]
        pass

    def eval(self):
        pass

    def parameters(self):  # ty:ignore[invalid-method-override]
        return [torch.nn.Parameter(torch.randn(10, 10))]

    def to(self, device):  # ty:ignore[invalid-method-override]
        return self


def test_trainer_initialization():
    """Test trainer initialization with mock data."""
    # Mock данные
    model = MockModel(12)
    train_loader = Mock()
    val_loader = Mock()
    device = torch.device("cpu")

    config = create_test_config()

    class_names = ["class1", "class2", "class3"]

    trainer = PlantTrainer(model, train_loader, val_loader, device, config, class_names)
    assert trainer is not None


def test_trainer_config_conversion():
    """Test config conversion in trainer."""
    model = MockModel(12)
    train_loader = Mock()
    val_loader = Mock()
    device = torch.device("cpu")

    config = create_test_config_dict()
    config["training"]["learning_rate"] = "0.001"  # must be parsed as float
    config = Config(config)

    class_names = ["class1", "class2", "class3"]

    trainer = PlantTrainer(model, train_loader, val_loader, device, config, class_names)

    # Проверяем что конфиг преобразован правильно
    assert isinstance(trainer.config.training_config.learning_rate, float)
    assert trainer.config.training_config.learning_rate == 0.001
