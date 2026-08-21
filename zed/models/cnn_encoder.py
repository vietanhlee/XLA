"""
Super-Resolution CNN Encoder (SReCCNN) for ZED.
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al.)
Reference: SReC (Cao et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Residual Block with Group Normalization and Swish (SiLU) activation.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)

class SReCCNN(nn.Module):
    """
    CNN Density Estimator per Resolution Level.
    
    Given spatial context y^(l+1) from lower resolution (upsampled 2x to match level l resolution),
    predicts parameters of K discrete logistic distributions for all pixels at level l.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_mixtures: int = 10,
        hidden_channels: int = 64,
        num_res_blocks: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_mixtures = num_mixtures
        
        # Total output channels = K (weights) + K * C (means) + K * C (log_scales)
        # For C=3, K=10 -> 10 + 30 + 30 = 70 channels
        self.out_channels = num_mixtures * (1 + 2 * in_channels)

        # Context Processor (processes upsampled lower-res context y^(l+1))
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Residual backbone
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_channels) for _ in range(num_res_blocks)
        ])

        # Head predicting mixture parameters
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.out_channels, kernel_size=3, padding=1)
        )
        
        # Initialize output conv weights to near zero for stable initial distribution
        nn.init.zeros_(self.out_conv[-1].bias)
        nn.init.normal_(self.out_conv[-1].weight, mean=0.0, std=1e-3)

    def forward(self, low_res_context: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            low_res_context: Image context y^(l+1) from lower resolution scale
            target_shape: (B, C, H, W) target spatial dimensions at current level l
            
        Returns:
            params: Mixture parameter logits tensor of shape (B, out_channels, H, W)
        """
        B, C, H, W = target_shape
        
        # Bilinear upsampling of lower-res context to current target resolution (H, W)
        if low_res_context.shape[-2:] != (H, W):
            upsampled_ctx = F.interpolate(
                low_res_context, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            upsampled_ctx = low_res_context

        feat = self.in_conv(upsampled_ctx)
        for block in self.res_blocks:
            feat = block(feat)

        params = self.out_conv(feat)
        return params
