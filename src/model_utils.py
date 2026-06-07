"""Model construction and device selection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from config import NUM_CLASSES


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_fusion_model(pretrained: bool = True) -> nn.Module:
    """ResNet18 with 6-channel input (raw RGB + processed RGB)."""
    model = build_model(pretrained=pretrained)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        6,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight.clone()
    model.conv1 = new_conv
    return model
