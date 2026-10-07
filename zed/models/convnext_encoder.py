"""
ConvNeXt-SReC Encoder with Efficient Channel Attention (ECA).
A modern, purely convolutional backbone for Successive Refinement Conditional Density Estimation.
Preserves 100% of spatial inductive bias and local translation equivariance,
while incorporating 7x7 depthwise receptive fields, inverted bottlenecks, and channel attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LayerNorm2d(nn.Module):
    """
    LayerNorm applied over channels for (B, C, H, W) tensors (Channels-First).
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean & variance across channel dimension per spatial location
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class EfficientChannelAttention(nn.Module):
    """
    ECA (Efficient Channel Attention) module (Wang et al., CVPR 2020).
    Captures cross-channel interactions without spatial dimensional reduction
    using an adaptive 1D depthwise convolution over global average pooled channel features.
    """
    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        # Calculate adaptive kernel size k
        t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = t if t % 2 != 0 else t + 1
        self.k = max(3, k)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k, padding=self.k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x) # (B, C, 1, 1)
        # Reshape to (B, 1, C) for 1D conv
        y = y.squeeze(-1).transpose(-1, -2) # (B, 1, C)
        y = self.conv(y) # (B, 1, C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        return x * y.expand_as(x)


class ConvNeXtBlock(nn.Module):
    """
    Modernized Inverted Bottleneck Block (ConvNeXt - Liu et al., CVPR 2022) with ECA.
    Structure:
      1. 7x7 Depthwise Conv (Local receptive field expansion)
      2. LayerNorm (Channels-First)
      3. 1x1 Conv (Expand 4x channels)
      4. GELU Activation
      5. Efficient Channel Attention (ECA)
      6. 1x1 Conv (Project back)
      7. Layer Scale with residual addition
    """
    def __init__(self, channels: int, expansion: int = 4, layer_scale_init_value: float = 1.0):
        super().__init__()
        # 7x7 Depthwise Conv
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        
        hidden_dim = channels * expansion
        # 1x1 Pointwise expansion
        self.pwconv1 = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        
        # Channel attention on expanded representation
        self.eca = EfficientChannelAttention(hidden_dim)
        
        # 1x1 Pointwise projection
        self.pwconv2 = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        
        # Layer scale parameter initialized to 1.0 for immediate full gradient flow in shallow 4-block network
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(channels, 1, 1), requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.eca(out)
        out = self.pwconv2(out)
        
        if self.gamma is not None:
            out = self.gamma * out
            
        return residual + out


class ConvNeXtSReCCNN(nn.Module):
    """
    Super-Resolution Conditional Density Estimator using ConvNeXt Blocks.
    Predicts mixture parameters of K discrete logistic distributions for Level l,
    conditioned on the lower-resolution spatial context y^(l+1).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_mixtures: int = 10,
        hidden_channels: int = 64,
        num_blocks: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_mixtures = num_mixtures
        
        # Total output channels: K (weights) + K * C (means) + K * C (log_scales)
        # For C=3, K=10 -> 10 + 30 + 30 = 70 channels
        self.out_channels = num_mixtures * (1 + 2 * in_channels)

        # Context stem processor: 7x7 conv into hidden channels + LayerNorm + GELU
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3),
            LayerNorm2d(hidden_channels),
            nn.GELU()
        )

        # ConvNeXt backbone with full gradient participation
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(channels=hidden_channels, expansion=4, layer_scale_init_value=1.0)
            for _ in range(num_blocks)
        ])

        # Head predicting mixture parameters
        self.head = nn.Sequential(
            LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.out_channels, kernel_size=3, padding=1)
        )

        # Initialize output conv weights close to zero for stable initial distribution
        nn.init.zeros_(self.head[-1].bias)
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)

    def forward(self, low_res_context: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        Args:
            low_res_context: Image context y^(l+1) from lower resolution scale
            target_shape: (B, C, H, W) target spatial dimensions at current level l
            
        Returns:
            params: Mixture parameter logits tensor of shape (B, out_channels, H, W)
        """
        B, C, H, W = target_shape
        
        # Bilinear upsample context to target resolution (H, W)
        if low_res_context.shape[-2:] != (H, W):
            upsampled_ctx = F.interpolate(
                low_res_context, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            upsampled_ctx = low_res_context

        feat = self.stem(upsampled_ctx)
        for block in self.blocks:
            feat = block(feat)

        params = self.head(feat)
        return params
