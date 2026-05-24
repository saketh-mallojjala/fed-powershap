from .cnn import CIFARCNN, MNISTCNN, CLASSIFIER_LAYER_NAME  # noqa: F401


def build_model(cfg):
    arch = getattr(cfg, "model", "cnn")
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
