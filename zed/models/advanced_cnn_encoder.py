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
        
        # === FIX: Upsample context lên target resolution TRƯỚC, rồi mới áp wavelet ===
        # Trước đây wavelet(low_res_context) ở H/2 → output H/4 → upsample 4x = PHÁ HỦY tần số.
        # Bây giờ: upsample context → H, wavelet(H) → output H/2, upsample 2x = BẢO TOÀN tần số.
        if low_res_context.shape[-2:] != (H, W):
            upsampled_ctx = F.interpolate(
                low_res_context, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            upsampled_ctx = low_res_context

        # Trích xuất dải tần số cao (LH, HL, HH) từ context ĐÃ upsample lên target resolution
        # Output: high_freq shape (B, 3*C, H/2, W/2) — chỉ giảm 2x thay vì 4x như trước
        _, high_freq_subbands = self.wavelet(upsampled_ctx)

        # Upsample wavelet features chỉ 2x bằng nearest (bảo toàn biên tần, không làm mờ)
        # Bilinear là bộ lọc low-pass — phá hủy chính thông tin tần số cao mà wavelet vừa trích xuất
        upsampled_high_freq = F.interpolate(
            high_freq_subbands, size=(H, W), mode="nearest"
        )

        # Concatenate spatial và high-frequency DWT features: shape (B, C + 3*C, H, W)
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
