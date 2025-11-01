import pytest
import torch
from src.models.resnet import ResNet50, ResNetConfig


def test_resnet_config():
    """Test ResNet configuration."""
    config = ResNetConfig(num_classes=10)
    assert config.num_classes == 10
    assert config.model_type == "resnet"


def test_resnet_creation():
    """Test ResNet model creation."""
    config = ResNetConfig(num_classes=12)
    model = ResNet50(config)

    assert model is not None
    # Проверяем что последний слой имеет правильное количество выходов
    assert model.model.fc.out_features == 12


def test_resnet_forward_pass():
    """Test ResNet forward pass with mock data."""
    config = ResNetConfig(num_classes=12)
    model = ResNet50(config)

    # Mock входные данные
    batch_size = 2
    mock_input = torch.randn(batch_size, 3, 224, 224)

    output = model(mock_input)

    # Проверяем размерность выхода
    assert output.shape == (batch_size, 12)


def test_resnet_from_config():
    """Test creating ResNet from config dictionary."""
    config_dict = {"model": {"num_classes": 8}}
    model = ResNet50.from_config(config_dict)
    assert model.config.num_classes == 8
