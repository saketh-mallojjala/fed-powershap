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
    # | student_depression | depression_student  (tabular mental-health)
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

    # How clients are formed:
    #   "dirichlet" - synthetic label-skew partition (the standard benchmark)
    #   "natural"   - one client per real group in the data (hospital, medical
    #                 specialty). num_clients is then DERIVED from the data.
    # Real silos differ by case mix and referral pathway, not by a symmetric
    # label skew, so "natural" tests something Dirichlet cannot.
    partition_mode: str = "dirichlet"

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

    # ---- DCFI-UPA: Dual-Channel Fair Incentive & Utility-Preserving Agg. ----
    # Training channel = FedDyn with the reliability detector as a SAFETY GATE
    # only (clip/quarantine, never reweight). Reward channel runs off the
    # training path entirely. See federated/dcfi.py.
    dcfi: bool = False
    # Risk-score component weights (each is a robust z-score).
    dcfi_w_loss: float = 0.5          # probe loss -> data quality (weak on its own:
                                      # with clean labels it only tracks non-IID spread)
    dcfi_w_geometry: float = 1.0      # effort density -> did they work?
    dcfi_w_temporal: float = 1.0      # self-similarity -> replay / staleness
    # Risk tier thresholds on the combined score. Set from the honest spread,
    # not tuned against attacks: the relative z-scores are computed over only
    # ~5 active clients, so a tight threshold produces heavy false flagging
    # (measured: 62% of honest observations at medium=1.0).
    dcfi_medium_risk: float = 2.0
    dcfi_high_risk: float = 4.0
    # Absolute overrides, applied regardless of the peer comparison.
    dcfi_gross_ratio: float = 20.0     # effort density this far from peer median
    dcfi_replay_cos: float = 0.999    # replay = near-exact resend of own prev delta
    dcfi_gross_ratio_hi: float = 8.0  # (kept for reference; see dcfi_gross_ratio)
    # Medium-risk response: clip the CLASSIFIER delta toward this quantile of
    # the low-risk peers' norm. The feature extractor is never clipped.
    dcfi_clip_quantile: float = 0.5
    # Where the reliability signal is allowed to act:
    #   "full"        - gate BOTH aggregation and reward (clip/exclude + no pay)
    #   "reward_only" - aggregation is untouched plain FedDyn over all active
    #                   clients; risk decides PAYMENT ELIGIBILITY only. This is
    #                   the recommended configuration: measured, the aggregation
    #                   gate costs accuracy on 8 of 9 conditions while the same
    #                   scores are useful for deciding who gets paid.
    #   "off"         - no gating anywhere (reward still utility/fairness-driven)
    dcfi_gate_mode: str = "full"
    # Quorum floor: the gate may never drop the active set below this
    # fraction. Excess high-risk clients are demoted to clipped instead.
    dcfi_min_keep_frac: float = 0.6
    # High-risk response: quarantine after this many consecutive high rounds.
    dcfi_quarantine_after: int = 2
    dcfi_quarantine_rounds: int = 5
    # Selection: utility + reliability + under-participation bonus.
    dcfi_under_bonus: float = 0.5
    dcfi_selection_utility_weight: float = 0.5
    # Reward channel: budget split, cap and holdback.
    dcfi_budget: float = 1.0
    dcfi_w_base: float = 0.2          # participation floor (individual rationality)
    dcfi_w_utility: float = 0.5
    dcfi_w_fairness: float = 0.3
    dcfi_reward_cap: float = 0.4      # max share of one round's budget
    dcfi_holdback: float = 0.2        # retained until reliability is re-established
    dcfi_release_reliability: float = 0.7
    # Per-client cost, used for the reward-to-cost and individual-rationality
    # metrics. Expressed as a share of the per-round budget.
    dcfi_client_cost: float = 0.05
    # Fraction of the test pool reserved as the server's validation set for
    # utility estimation. Kept disjoint from the reported test split.
    dcfi_val_frac: float = 0.3

    # Free-rider adversaries (proposal §9.3 attack ladder). A fraction
    # `freerider_frac` of clients harvest the global model while contributing
    # little or nothing. Disjoint from the label-noise clients so the two
    # adversary types can be analysed separately.
    #   f1 - random Gaussian update, scaled to look like a plausible update
    #   f2 - zero effort: returns the global model unchanged
    #   f3 - trains on `freerider_subset_frac` of its data only
    #   f4 - trains honestly, then inflates the update by `freerider_scale`
    #        (targets the cosine/contribution signal, not the loss signal)
    freerider_frac: float = 0.0
    freerider_mode: str = "none"        # none | f1 | f2 | f3 | f4
    freerider_scale: float = 3.0        # f4 update-inflation factor
    freerider_subset_frac: float = 0.1  # f3 fraction of local data actually used
    freerider_noise_scale: float = 1.0  # f1 sigma, relative to a typical update

    # Leave-one-out: drop this client id from the federation entirely (-1 = keep
    # all). The Dirichlet partition and the noisy-client draw both run BEFORE the
    # exclusion and are seeded by `seed`, so client k is the same client whether
    # or not it is excluded -- which is what makes V(N) - V(N\{k}) a valid
    # marginal value. Used by scripts/loo_reward_fidelity.py to build the
    # ground-truth contribution the reward is scored against (proposal C1).
    exclude_client: int = -1

    # Client data-quality heterogeneity (single-label only). A fraction
    # `noisy_client_frac` of clients are "low quality": `label_noise_rate` of
    # their TRAINING labels are randomly flipped (test labels stay clean). This
    # is the regime contribution-aware aggregation is designed for — it should
    # down-weight noisy clients where FedAvg/FedDyn trust them and degrade.
    # Label-noise model. "symmetric" spreads a flipped label uniformly over the
    # other classes; "asymmetric" is the standard pair-flip (class c -> c+1),
    # which preserves class structure and is the harder, more realistic case.
    # Reviewer B asked for results under more than one noise type.
    label_noise_type: str = "symmetric"
    # Covariate shift: give `shift_client_frac` of clients a fixed per-client
    # photometric/blur/noise shift of this severity, labels untouched. These are
    # the "high loss but correctly labelled" clients of reviewer A's question.
    shift_client_frac: float = 0.0
    shift_severity: float = 0.0
    shift_mode: str = "all"
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
    # fednoro | fedcorr | feda3i are the noise-aware baselines (reviewer B.1);
    # all three sit on FedDyn so only the aggregation rule differs.
    aggregation: str = "shapfed_wa"
    noise_baseline_warmup: int = 5
    feda3i_gamma: float = 2.0
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
    # Contribution/quality signal driving aggregation weights:
    #   "cssv" - ShapFed cosine of the classifier update vs. consensus (original).
    #   "loss" - down-weight by the global model's loss on the client's own data.
    #            Corrupted clients disagree with the clean consensus -> high loss
    #            -> low weight. This RELIABLY separates label-noise clients, which
    #            the cosine signal does not under partial participation.
    contrib_signal: str = "cssv"   # cssv | loss
    # Sharpness of the loss->weight map: w_shap ∝ exp(-beta * (loss - min_loss)).
    loss_weight_beta: float = 4.0

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

    # --- tabular datasets only ---
    # Which feature set to use on the depression-screening data:
    #   "screening" - drop the "Have you ever had suicidal thoughts?" item. That
    #                 item is a core diagnostic symptom and the top SHAP feature,
    #                 so keeping it nearly saturates the task -- and a saturated
    #                 task cannot discriminate between aggregators. This is the
    #                 headline regime (see docs/INCENTIVE_PROPOSAL.md §9.1).
    #   "full"      - keep every feature (reported as the saturation reference).
    feature_regime: str = "screening"
    # Set automatically from the loader; MLP input width.
    num_features: int = 0
    mlp_depth: int = 2
    mlp_dropout: float = 0.1

    # Model
    model: str = "cnn"               # cnn | resnet18 | resnet34 | resnet50 | mlp
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
        # Moderate blend + tight cap so no single client can dominate: aggressive
        # loss-weighting collapses onto 1-2 clients and destabilizes the global
        # model. beta=1.5 down-weights noisy clients without zeroing the rest.
        agg_blend_lambda=0.5, agg_blend_lambda_final=0.5, cssv_max_weight=0.30,
        cssv_unit_interval=True, cssv_ema=0.5, feddyn_alpha=0.01,
        feddyn_weight_consistent=True,
        # Reliable quality signal: down-weight clients the clean consensus fits
        # poorly (label-noise clients). The cosine/CSSV signal is anti-
        # discriminative under partial participation; loss separates cleanly.
        contrib_signal="loss", loss_weight_beta=1.5,
    ),
    # DCFI-UPA. Training channel is plain FedDyn with UNIFORM weights over the
    # clients that survive the safety gate -- deliberately NOT reward-weighted,
    # since contribution-weighted aggregation was measured to cost accuracy.
    # The gate may clip a suspicious classifier component or quarantine a
    # repeat offender, but it never scales an accepted client's influence.
    # Noise-aware baselines (reviewer B.1). All sit on FedDyn with the same
    # local solver as `proposed`, so the aggregation rule is the only thing
    # that differs between them and the proposed method.
    "fednoro": dict(
        selection_strategy="random", aggregation="fednoro", local_solver="feddyn",
        server_momentum=0.0, feddyn_alpha=0.01, feddyn_weight_consistent=True,
        noise_baseline_warmup=10),
    "fedcorr": dict(
        selection_strategy="random", aggregation="fedcorr", local_solver="feddyn",
        server_momentum=0.0, feddyn_alpha=0.01, feddyn_weight_consistent=True,
        noise_baseline_warmup=5),
    "feda3i": dict(
        selection_strategy="random", aggregation="feda3i", local_solver="feddyn",
        server_momentum=0.0, feddyn_alpha=0.01, feddyn_weight_consistent=True,
        cssv_max_weight=0.3, feda3i_gamma=2.0, noise_baseline_warmup=5),
    "dcfi": dict(
        selection_strategy="random",     # overridden by the DCFI selector
        aggregation="feddyn", local_solver="feddyn",
        server_momentum=0.0, feddyn_alpha=0.01, feddyn_weight_consistent=True,
        dcfi=True,
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
