import torch.nn as nn
import torchvision.models as models
from transformers import PreTrainedModel, PretrainedConfig
import torch

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