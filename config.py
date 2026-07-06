"""Configuration for federated learning: the proposed method + baselines.

A ``method`` preset (see ``METHOD_PRESETS``) expands into the lower-level knobs
(selection_strategy, aggregation, local_solver, regularizer params). The CLI /
explicit kwargs always win over a preset, so you can still ablate individual
pieces, e.g. ``--method proposed --server-momentum 0``.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Experiment
    exp_name: str = "shapfed_poc"
    seed: int = 42
    device: str = "cuda"  # falls back to cpu if unavailable
    log_dir: str = "logs"

    # Method preset: fedavg | fedprox | scaffold | feddyn | fedbn | moon |
    # poc_fedavg | fedce | proposed. Expands to the knobs below.
    method: str = "proposed"

    # Dataset
    # cifar10 | mnist | fmnist | aptos | octmnist | isic | chestxray14 | brats
    dataset: str = "cifar10"
    data_root: str = "./data_cache"
    num_classes: int = 10
    # single_label (CrossEntropy, accuracy/QWK) | multi_label (BCE, AUC/mAP).
    # Auto-set from the dataset registry in main.py for medmnist/brats datasets.
    task: str = "single_label"
    # Cap the training pool to this many samples (stratified), 0 = use all.
    # MedMNIST sets are 100k+ images; capping keeps per-round time and the
    # non-IID setup comparable to the smaller APTOS run.
    max_train_samples: int = 0
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
    # Per-client local test fraction held out for fairness (Jain) evaluation.
    # 0.0 disables per-client eval. Identical (seeded) split across all methods.
    local_test_frac: float = 0.2

    # Client data-quality heterogeneity (single-label only). A fraction
    # `noisy_client_frac` of clients are "low quality": `label_noise_rate` of
    # their TRAINING labels are randomly flipped (test labels stay clean). This
    # is the regime contribution-aware aggregation is designed for — it should
    # down-weight noisy clients where FedAvg/FedDyn trust them and degrade.
    noisy_client_frac: float = 0.0
    label_noise_rate: float = 0.0

    # Local solver (client-side optimization variant):
    #   sgd | fedprox | feddyn | moon | scaffold
    local_solver: str = "sgd"
    fedprox_mu: float = 0.01          # FedProx proximal coefficient
    feddyn_alpha: float = 0.01        # FedDyn dynamic-regularization coefficient
    moon_mu: float = 1.0             # MOON contrastive loss weight
    moon_temperature: float = 0.5    # MOON contrastive temperature

    # Power-of-Choice selection
    #   d = candidate-pool size, m = clients trained per round (m <= d <= N)
    candidate_size_d: int = 10
    active_size_m: int = 5
    selection_strategy: str = "pow_d"  # pow_d | random | full
    # If True, candidate probability is proportional to |D_k|; else uniform.
    size_weighted_candidates: bool = True
    # Anneal Power-of-Choice toward random selection over training (proposed):
    # exploit high-loss clients early, explore broadly late. 0.0 disables.
    poc_anneal: float = 0.0
    # Down-rank candidates with persistently low Shapley reputation during
    # selection (proposed); 0.0 disables (pure highest-loss pow-d).
    reputation_weight: float = 0.0

    # Aggregation
    #   fedavg | shapfed_wa | fedbn | scaffold | feddyn | fedce
    aggregation: str = "shapfed_wa"
    # --- shapfed_wa (proposed) knobs ---
    # Temperature to sharpen Shapley weights; 1.0 disables.
    cssv_temperature: float = 1.0
    # Clamp negative contributions to 0 before normalization (legacy path).
    cssv_clamp_negative: bool = True
    # Exponential moving average for CSSV history; 0.0 disables smoothing.
    cssv_ema: float = 0.5
    # Convex blend of Shapley weights with the FedAvg size prior:
    #   w = (1-lambda) * size_w + lambda * shap_w
    # Graceful degradation: lambda=0 is exactly FedAvg.
    agg_blend_lambda: float = 0.5
    # Anneal blend lambda from agg_blend_lambda toward agg_blend_lambda_final
    # over rounds (lets the Shapley signal take over as it stabilizes).
    agg_blend_lambda_final: float = 0.5
    # Cap any single client's aggregation weight to bound effective-sample-size
    # loss / weight collapse. 1.0 disables.
    cssv_max_weight: float = 1.0
    # Map cosine similarity to [0,1] via (cos+1)/2 before summing (kills the
    # negative-clamp collapse). When True, cssv_clamp_negative is ignored.
    cssv_unit_interval: bool = True
    # Reference used as the CSSV consensus direction each client is scored
    # against: "mean" (original ShapFed) | "median" | "trimmed". Under label
    # noise the mean is polluted by corrupted clients, so a noisy client can
    # still look "aligned" and keep weight; a robust reference (coordinate-wise
    # median / trimmed mean) is far from the corrupted updates, so noisy clients
    # score low and get down-weighted. This is the key lever for the noisy regime.
    cssv_reference: str = "mean"
    # Fraction trimmed from each tail when cssv_reference="trimmed".
    cssv_trim_frac: float = 0.2

    # Principled FedDyn+ShapFed: make the FedDyn drift-correction (h) track the
    # SAME Shapley-weighted average that forms the model, instead of the uniform
    # client sum. When weights are uniform (plain feddyn) this is identical to
    # the original; it only changes shapfed_dyn, where it stops FedDyn's
    # correction from fighting the contribution reweighting. False = legacy.
    feddyn_weight_consistent: bool = True

    # Round-wise local LR schedule to curb late-round divergence under non-IID /
    # noisy clients (constant LR collapses late; see RESULTS_ANALYSIS). "none"
    # keeps the constant LR; "cosine" decays local_lr -> local_lr*lr_min_frac.
    lr_schedule: str = "none"          # none | cosine
    lr_min_frac: float = 0.1

    # Server-side momentum (FedAvgM-style); 0.0 disables.
    server_momentum: float = 0.0

    # Model
    model: str = "cnn"               # cnn | resnet18 | resnet34 | resnet50
    hidden_dim: int = 64
    pretrained: bool = True          # ImageNet weights for resnet*

    # Evaluation
    eval_every: int = 1
    save_every: int = 25


# Method presets: each maps to the lower-level knobs above. Only the fields a
# method actually needs are set; everything else keeps the Config default.
METHOD_PRESETS = {
    "fedavg": dict(
        selection_strategy="random", aggregation="fedavg", local_solver="sgd",
        server_momentum=0.0, poc_anneal=0.0, reputation_weight=0.0,
    ),
    "fedprox": dict(
        selection_strategy="random", aggregation="fedavg", local_solver="fedprox",
        server_momentum=0.0,
    ),
    "scaffold": dict(
        selection_strategy="random", aggregation="scaffold", local_solver="scaffold",
        server_momentum=0.0,
    ),
    "feddyn": dict(
        selection_strategy="random", aggregation="feddyn", local_solver="feddyn",
        server_momentum=0.0,
    ),
    "fedbn": dict(
        selection_strategy="random", aggregation="fedbn", local_solver="sgd",
        server_momentum=0.0,
    ),
    "moon": dict(
        selection_strategy="random", aggregation="fedavg", local_solver="moon",
        server_momentum=0.0,
    ),
    "poc_fedavg": dict(
        selection_strategy="pow_d", aggregation="fedavg", local_solver="sgd",
        server_momentum=0.0,
    ),
    "fedce": dict(
        selection_strategy="random", aggregation="fedce", local_solver="sgd",
        server_momentum=0.0,
    ),
    # Proposed: robust contribution-aware dynamic FL, built for the unreliable-
    # client regime. FedDyn dynamic regularization (strongest baseline optimizer)
    # + Shapley/CSSV aggregation weights that are (a) scored against a robust
    # MEDIAN consensus so label-noise clients score low and get down-weighted,
    # and (b) fed into a weight-CONSISTENT FedDyn server update so the drift
    # correction no longer fights the reweighting. Power-of-Choice is dropped
    # (it selects the highest-loss = corrupted clients under noise); plain random
    # selection with a strong Shapley blend (lambda=0.85). Recover the old
    # variant for ablation via CLI: --selection-strategy pow_d --poc-anneal 0.5
    # --reputation-weight 0.5 --agg-blend-lambda 0.5 --cssv-reference mean
    # --no-feddyn-weight-consistent.
    "proposed": dict(
        selection_strategy="random", aggregation="shapfed_dyn", local_solver="feddyn",
        server_momentum=0.0, poc_anneal=0.0, reputation_weight=0.0,
        agg_blend_lambda=0.85, agg_blend_lambda_final=0.85, cssv_max_weight=0.5,
        cssv_unit_interval=True, cssv_ema=0.5, feddyn_alpha=0.01,
        cssv_reference="median", cssv_trim_frac=0.2, feddyn_weight_consistent=True,
    ),
}


def apply_method_preset(cfg: "Config", method: str) -> "Config":
    """Set the lower-level knobs for a named method preset (in place)."""
    if method not in METHOD_PRESETS:
        raise ValueError(
            f"Unknown method '{method}'. Choices: {sorted(METHOD_PRESETS)}"
        )
    cfg.method = method
    for k, v in METHOD_PRESETS[method].items():
        setattr(cfg, k, v)
    return cfg


def get_config(**overrides) -> Config:
    """Build a Config. If ``method`` is given, apply its preset first, then any
    explicit overrides (overrides win, so per-knob ablations still work)."""
    method = overrides.pop("method", None)
    cfg = Config()
    if method is not None:
        apply_method_preset(cfg, method)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
