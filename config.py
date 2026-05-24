"""Configuration for ShapFed + Power-of-Choice federated learning."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Experiment
    exp_name: str = "shapfed_poc"
    seed: int = 42
    device: str = "cuda"  # falls back to cpu if unavailable
    log_dir: str = "logs"

    # Dataset
    dataset: str = "cifar10"         # cifar10 | mnist | fmnist | aptos
    data_root: str = "./data_cache"
    num_classes: int = 10
    batch_size: int = 64
    image_size: int = 224            # only used by image-folder datasets (aptos, ...)
    num_workers: int = 0             # DataLoader workers; bump to 2-4 for aptos

    # Federation
    num_clients: int = 20
    alpha: float = 0.3                # Dirichlet concentration (lower => more non-IID)
    num_rounds: int = 100
    local_epochs: int = 2
    local_lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # Power-of-Choice selection
    #   d = candidate-pool size, m = clients trained per round (m <= d <= N)
    candidate_size_d: int = 10
    active_size_m: int = 5
    selection_strategy: str = "pow_d"  # pow_d | random | full
    # If True, candidate probability is proportional to |D_k|; else uniform.
    size_weighted_candidates: bool = True

    # ShapFed CSSV aggregation
    aggregation: str = "shapfed_wa"    # shapfed_wa | fedavg
    # Temperature to sharpen Shapley weights; 1.0 disables.
    cssv_temperature: float = 1.0
    # Clamp negative contributions to 0 before normalization.
    cssv_clamp_negative: bool = True
    # Exponential moving average for CSSV history; 0.0 disables smoothing.
    cssv_ema: float = 0.5

    # Model
    model: str = "cnn"               # cnn | resnet18 | resnet34 | resnet50
    hidden_dim: int = 64
    pretrained: bool = True          # ImageNet weights for resnet*

    # Evaluation
    eval_every: int = 1
    save_every: int = 25


def get_config(**overrides) -> Config:
    cfg = Config(**overrides)
    return cfg
