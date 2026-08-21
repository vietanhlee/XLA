"""
2D Haar Wavelet Decomposition Module for Frequency-Domain Artifact Extraction.
Decomposes spatial image into Low-Frequency (LL) and High-Frequency (LH, HL, HH) subbands.
High-frequency subbands highlight microscopic spectral artifacts left by AI generators (Diffusion/GANs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HaarWavelet2D(nn.Module):
    """
    2D Haar Wavelet Transform implemented using PyTorch 2D Convolutions with fixed filters.
    Fully differentiable, PyTorch native, and fast.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

        # Define 2D Haar decomposition filters (2x2)
        # LL: Low-frequency average
        ll_kernel = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        # LH: Horizontal high-frequency
        lh_kernel = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        # HL: Vertical high-frequency
        hl_kernel = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        # HH: Diagonal high-frequency
        hh_kernel = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        # Stack filters into shape (4 * C, 1, 2, 2) for depthwise convolution
        filters = torch.stack([ll_kernel, lh_kernel, hl_kernel, hh_kernel], dim=0) # (4, 2, 2)
        filters = filters.repeat(in_channels, 1, 1).unsqueeze(1) # (4 * C, 1, 2, 2)
        
        self.register_buffer("weight", filters)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            ll: Low-frequency subband of shape (B, C, H/2, W/2)
            high_freq: High-frequency subbands concatenated (LH, HL, HH) of shape (B, 3*C, H/2, W/2)
        """
        B, C, H, W = x.shape
        
        # Pad image if H or W is odd
        pad_h = 1 if H % 2 != 0 else 0
        pad_w = 1 if W % 2 != 0 else 0
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Grouped 2D convolution with stride 2
        # Result shape: (B, 4 * C, H/2, W/2)
        out = F.conv2d(x, self.weight, stride=2, groups=C)
        
        # Reshape to (B, C, 4, H/2, W/2)
        H_sub, W_sub = out.shape[-2:]
        out = out.view(B, C, 4, H_sub, W_sub)

        ll = out[:, :, 0, :, :]                             # (B, C, H/2, W/2)
        lh = out[:, :, 1, :, :]                             # (B, C, H/2, W/2)
        hl = out[:, :, 2, :, :]                             # (B, C, H/2, W/2)
        hh = out[:, :, 3, :, :]                             # (B, C, H/2, W/2)

        high_freq = torch.cat([lh, hl, hh], dim=1)         # (B, 3*C, H/2, W/2)
        return ll, high_freq
