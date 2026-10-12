"""
Visual Encoder Module for Multimodal Two-Tower News Recommender.

Architecture:
- Backbone: Pretrained ResNet-50 from torchvision.models.
- Lower convolutional blocks (conv1, bn1, layer1, layer2, layer3) frozen.
- AdaptiveAvgPool2d((1, 1)) -> Linear(2048 -> d_model).
- LayerNorm(d_model) + Dropout(0.2) -> v_i in R^{d_model}.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class VisualEncoder(nn.Module):
    """
    Visual feature extractor using ResNet-50 backbone with linear projection,
    layer normalization, and dropout.
    """

    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_lower_blocks: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout_rate = dropout
        self.freeze_lower_blocks = freeze_lower_blocks

        # Initialize ResNet-50 backbone
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
        except Exception as e:
            logger.warning(
                f"Failed to load pretrained ResNet-50 weights ({e}). Initializing randomly."
            )
            resnet = models.resnet50(weights=None)

        # Freeze lower convolutional blocks if requested
        if freeze_lower_blocks:
            frozen_modules = [
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            ]
            for module in frozen_modules:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        # Extract convolutional layers up to layer4
        self.conv_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Adaptive average pooling to produce [B, 2048, 1, 1]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection head: Linear(2048 -> d_model) + LayerNorm + Dropout
        self.in_features = 2048
        self.projection = nn.Linear(self.in_features, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet-50 convolutional blocks.
        Input: [B, 3, 224, 224]
        Output: [B, 2048]
        """
        if self.freeze_lower_blocks:
            with torch.no_grad():
                x = self.conv_stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.adaptive_pool(x)
                x = torch.flatten(x, 1)
        else:
            x = self.conv_stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
        return x

    def project_features(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Projects 2048-d pooled features into d_model dimensional space:
        v_i = Dropout(LayerNorm(Linear(feat)))
        Input: [B, 2048]
        Output: [B, d_model]
        """
        proj = self.projection(feat)
        normed = self.layer_norm(proj)
        out = self.dropout(normed)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw image tensor of shape [B, 3, 224, 224], or precomputed
               feature tensor of shape [B, 2048].

        Returns:
            v_i: Visual embedding tensor of shape [B, d_model].
        """
        if x.dim() == 2 and x.size(1) == self.in_features:
            # Precomputed backbone features passed directly
            return self.project_features(x)
        elif x.dim() == 4:
            # Standard image forward pass [B, 3, H, W]
            backbone_feat = self.extract_backbone_features(x)
            return self.project_features(backbone_feat)
        else:
            raise ValueError(
                f"Expected input shape [B, 3, H, W] or [B, {self.in_features}], got {x.shape}"
            )
