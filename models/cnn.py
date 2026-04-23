"""Small CNN models. The final classifier layer name is fixed so the Shapley
module can extract class-wise gradients without having to guess."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASSIFIER_LAYER_NAME = "classifier"  # the nn.Linear layer mapping features -> classes


class CIFARCNN(nn.Module):
    def __init__(self, num_classes: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return self.classifier(x)


class MNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return self.classifier(x)


def build_model(cfg) -> nn.Module:
    if cfg.dataset == "cifar10":
        return CIFARCNN(cfg.num_classes, cfg.hidden_dim)
    return MNISTCNN(cfg.num_classes, cfg.hidden_dim)
