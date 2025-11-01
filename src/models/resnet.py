import torch.nn as nn
import torchvision.models as models
from transformers import PreTrainedModel, PretrainedConfig
import torch
from typing import Dict, Any


class ResNetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(self, num_classes=12, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes


class ResNet50(PreTrainedModel):
    config_class = ResNetConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, config.num_classes)

    def forward(self, x):
        return self.model(x)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create model from configuration dictionary."""
        model_config = ResNetConfig(num_classes=config["model"]["num_classes"])
        return cls(model_config)

    def get_expected_input_shape(self) -> tuple:
        """Get expected input shape for testing."""
        return (3, 224, 224)

    def mock_forward(self, batch_size: int = 2) -> torch.Tensor:
        """Mock forward pass for testing without real data."""
        mock_input = torch.randn(batch_size, 3, 224, 224)
        return self.forward(mock_input)
