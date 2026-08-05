"""Tabular MLP whose head is named ``classifier``.

ShapFed's CSSV reads the layer named ``CLASSIFIER_LAYER_NAME`` and treats its
weight rows as per-class gradient signals, so the head must be a single
``nn.Linear`` with that exact name — same contract the ResNet wrapper honours.
``extract_features`` exposes the penultimate representation for MOON.
"""
from __future__ import annotations

import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        d_in = num_features
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d_in = hidden_dim
        self.fc = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, x):
        return self.fc(x)

    def forward(self, x):
        return self.classifier(self.extract_features(x))
