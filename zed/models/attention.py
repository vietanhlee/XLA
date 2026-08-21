"""
Spatial Self-Attention Module for Global Context Dependency Modeling.
Allows the density estimator to capture long-range spatial correlations across distant image patches.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSelfAttention(nn.Module):
    """
    Efficient Spatial Self-Attention Block for 2D Feature Maps.
    Uses Adaptive Key/Value Spatial Pooling to provide global context without memory explosion.
    Memory footprint reduced from O((H*W)^2) to O(H*W * K_grid^2).
    """

    def __init__(self, in_channels: int, num_heads: int = 4, kv_grid_size: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.kv_grid_size = kv_grid_size

        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.kv_pool = nn.AdaptiveAvgPool2d((kv_grid_size, kv_grid_size))
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map of shape (B, C, H, W)
            
        Returns:
            out: Attention-enhanced feature map of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x

        norm_x = self.norm(x)
        
        # Query at full resolution (B, num_heads, H*W, head_dim)
        q = self.q_conv(norm_x).view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)

        # Key & Value pooled down to fixed grid (kv_grid_size x kv_grid_size) for O(1) memory bound
        kv_x = self.kv_pool(norm_x)
        N_kv = self.kv_grid_size * self.kv_grid_size

        k = self.k_conv(kv_x).view(B, self.num_heads, self.head_dim, N_kv).transpose(-2, -1)
        v = self.v_conv(kv_x).view(B, self.num_heads, self.head_dim, N_kv).transpose(-2, -1)

        # Efficient Dot-Product Attention: shape (B, num_heads, H*W, N_kv)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)

        out_attn = torch.matmul(attn_weights, v) # (B, num_heads, H*W, head_dim)
        out_attn = out_attn.transpose(-2, -1).contiguous().view(B, C, H, W)

        out = self.proj_out(out_attn) + residual
        return out

