# from torchvision import models
import timm
import torch
from transformers import PretrainedConfig, PreTrainedModel

from config import ModelConfig


class ResNetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(self, num_classes: int = 12, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.num_classes = num_classes


class ResNet50(PreTrainedModel):
    config_class = ResNetConfig

    def __init__(self, model_config: ResNetConfig) -> None:
        super().__init__(model_config)
        self.model = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0,
        )
        self.model.fc = torch.nn.Linear(2048, model_config.num_classes)  # ty:ignore[unresolved-attribute]

    def forward(self, x: torch.Tensor):
        return self.model(x)

    @classmethod
    def from_config(cls, model_config: ModelConfig) -> "ResNet50":
        """Create model from configuration dictionary."""
        res_net_config = ResNetConfig(num_classes=model_config.num_classes)
        return cls(res_net_config)

    def get_expected_input_shape(self) -> tuple:
        """Get expected input shape for testing."""
        return (3, 224, 224)

    def mock_forward(self, batch_size: int = 2) -> torch.Tensor:
        """Mock forward pass for testing without real data."""
        mock_input = torch.randn(batch_size, 3, 224, 224)
        return self.forward(mock_input)
