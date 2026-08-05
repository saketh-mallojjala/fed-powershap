from .cnn import CIFARCNN, MNISTCNN, CLASSIFIER_LAYER_NAME  # noqa: F401


def build_model(cfg):
    arch = getattr(cfg, "model", "cnn")
    if arch == "mlp":
        from .mlp import TabularMLP
        n_feat = getattr(cfg, "num_features", 0)
        if not n_feat:
            raise ValueError(
                "model='mlp' needs cfg.num_features; it is set automatically by "
                "main.apply_dataset_defaults for tabular datasets."
            )
        return TabularMLP(
            num_features=n_feat,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.hidden_dim,
            depth=getattr(cfg, "mlp_depth", 2),
            dropout=getattr(cfg, "mlp_dropout", 0.1),
        )
    if arch in ("resnet18", "resnet34", "resnet50"):
        from .resnet import ResNetClassifier
        return ResNetClassifier(
            num_classes=cfg.num_classes,
            arch=arch,
            pretrained=getattr(cfg, "pretrained", True),
        )
    if cfg.dataset == "cifar10":
        return CIFARCNN(cfg.num_classes, cfg.hidden_dim)
    return MNISTCNN(cfg.num_classes, cfg.hidden_dim)
