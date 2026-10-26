import torch
import torch.nn as nn
from torchvision import models

BACKBONES = ("efficientnet_b0", "resnet18", "resnet50")


def build_model(num_classes=4, backbone="efficientnet_b0", pretrained=True, dropout=0.3):
    if backbone == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif backbone == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    elif backbone == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {BACKBONES}")
    return m


def get_head(model):
    return model.classifier if hasattr(model, "classifier") else model.fc


def set_backbone_trainable(model, trainable: bool):
    head_params = {id(p) for p in get_head(model).parameters()}
    for p in model.parameters():
        p.requires_grad = True if id(p) in head_params else trainable


def save_checkpoint(path, model, backbone, class_names, img_size, extra=None):
    torch.save({
        "state_dict": model.state_dict(),
        "backbone": backbone,
        "class_names": list(class_names),
        "img_size": img_size,
        **(extra or {}),
    }, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(len(ckpt["class_names"]), ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt