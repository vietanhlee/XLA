"""
Advanced SReC Encoder integrating Spatial Context, 2D Haar Wavelet Frequency Features,
Residual Blocks, and Spatial Self-Attention.
Paper Extension: "Advanced Zero-Shot Detection of AI-Generated Images"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import ResBlock
from .wavelet import HaarWavelet2D
from .attention import SpatialSelfAttention

class AdvancedSReCCNN(nn.Module):
    """
    Advanced Density Estimator per Resolution Scale.
    Combines:
      - Spatial context y^(l+1)
      - High-frequency 2D Haar Wavelet features (LH, HL, HH)
      - Residual Backbone
      - Spatial Self-Attention Block
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_mixtures: int = 10,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        num_heads: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_mixtures = num_mixtures
        self.out_channels = num_mixtures * (1 + 2 * in_channels)

        # 1. 2D Haar Wavelet Extractor (extracts high-frequency subbands LH, HL, HH)
        self.wavelet = HaarWavelet2D(in_channels=in_channels)
        
        # Spatial channels (C=3) + High-frequency DWT channels (3*C=9) = 12 channels
        fusion_in_channels = in_channels + (3 * in_channels)
        
        # 2. Context & Frequency Feature Fusion Conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.SiLU()
        )

        # 3. Residual backbone
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_channels) for _ in range(num_res_blocks)
        ])

        # 4. Spatial Self-Attention Block
        self.attention = SpatialSelfAttention(hidden_channels, num_heads=num_heads)

        # 5. Output Head predicting logistic mixture parameters
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.out_channels, kernel_size=3, padding=1)
        )

        # Zero-initialize output conv weights for initial distribution stability
        nn.init.zeros_(self.out_conv[-1].bias)
        nn.init.normal_(self.out_conv[-1].weight, mean=0.0, std=1e-3)

    def forward(self, low_res_context: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        Args:
            low_res_context: Spatial context tensor y^(l+1)
            target_shape: Target tensor shape (B, C, H, W)
            
        Returns:
            params: Mixture parameters of shape (B, out_channels, H, W)
        """
        B, C, H, W = target_shape
        
        # Extract 2D Haar Wavelet high-frequency subbands from low-res context
        _, high_freq_subbands = self.wavelet(low_res_context) # (B, 3*C, H_low/2, W_low/2)

        # Upsample spatial context and high-freq features to target (H, W)
        if low_res_context.shape[-2:] != (H, W):
            upsampled_ctx = F.interpolate(
                low_res_context, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            upsampled_ctx = low_res_context

        upsampled_high_freq = F.interpolate(
            high_freq_subbands, size=(H, W), mode="bilinear", align_corners=False
        )

        # Concatenate spatial and high-frequency DWT features: shape (B, C + 3*C, H, W)
        fused_input = torch.cat([upsampled_ctx, upsampled_high_freq], dim=1)

        # Feature processing
        feat = self.fusion_conv(fused_input)
        for block in self.res_blocks:
            feat = block(feat)

        # Global spatial self-attention
        feat = self.attention(feat)

        # Output prediction
        params = self.out_conv(feat)
        return params
