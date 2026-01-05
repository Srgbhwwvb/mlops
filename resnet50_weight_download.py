"""
Script to download used ResNet50 model's weights.
It is used during building a docker container.
"""

import timm

model = timm.create_model(
    'resnet50',
    pretrained=True,
)

