"""ResNet wrapper that exposes a final ``classifier`` linear layer.

ShapFed CSSV reads from the layer named ``CLASSIFIER_LAYER_NAME``, so we wrap
torchvision's ResNet (whose head is called ``fc``) and replace it with our
own ``classifier`` Linear. The convolutional backbone stays untouched.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


_BUILDERS = {
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    "resnet34": (models.resnet34, "ResNet34_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
}


def _build_backbone(arch: str, pretrained: bool) -> nn.Module:
    if arch not in _BUILDERS:
        raise ValueError(f"Unsupported resnet arch: {arch}")
    builder, weights_name = _BUILDERS[arch]
    if pretrained:
        weights = getattr(models, weights_name).DEFAULT
        return builder(weights=weights)
    return builder(weights=None)


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()
        backbone = _build_backbone(arch, pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))
