"""
Full Multi-Resolution Advanced ZED Model (Advanced Zero-Shot AI Image Detector).
Integrates:
  - Discretized Logistic Mixture Distribution (K=10)
  - Multi-Resolution Image Pyramid (Levels 0, 1, 2, 3)
  - 2D Haar Wavelet High-Frequency Decomposition
  - Spatial Self-Attention Context Modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

from .logistic_mixture import DiscretizedLogisticMixture
from .advanced_cnn_encoder import AdvancedSReCCNN

class AdvancedZEDModel(nn.Module):
    """
    Advanced Multi-Resolution Zero-Shot AI Image Detector.
    Integrates Frequency-Domain Wavelet Analysis and Spatial Self-Attention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_mixtures: int = 10,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        num_heads: int = 4,
        min_log_scale: float = -7.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_mixtures = num_mixtures

        self.mixture_evaluator = DiscretizedLogisticMixture(
            num_mixtures=num_mixtures, min_log_scale=min_log_scale
        )

        # 3 Advanced SReCCNN modules for scales l = 0, 1, 2
        self.cnn_modules = nn.ModuleList([
            AdvancedSReCCNN(
                in_channels=in_channels,
                num_mixtures=num_mixtures,
                hidden_channels=hidden_channels,
                num_res_blocks=num_res_blocks,
                num_heads=num_heads
            )
            for _ in range(3)
        ])

    def build_pyramid(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Builds multi-scale pyramid levels x^(0), x^(1), x^(2), x^(3)."""
        x_levels = [x]
        y_levels = [x]

        curr_x = x
        for _ in range(3):
            y_next = F.avg_pool2d(curr_x, kernel_size=2, stride=2)
            x_next = torch.round(y_next)
            
            y_levels.append(y_next)
            x_levels.append(x_next)
            curr_x = x_next

        return x_levels, y_levels

    def forward(
        self, x: torch.Tensor, compute_entropy: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing NLL, Entropy, and Decision Statistics.
        
        Args:
            x: Input RGB images tensor (B, C, H, W) in [0, 255]
            compute_entropy: If True, computes exact expected entropy H.
        """
        if x.max() <= 1.0 + 1e-5:
            x = x * 255.0

        x_levels, y_levels = self.build_pyramid(x)

        nll_levels = []
        h_levels = []
        d_levels = []

        total_nll_loss = 0.0

        for l in range(3):
            target_x = x_levels[l]             # x^(l)
            lower_res_ctx = y_levels[l + 1]     # y^(l+1)

            # Predict mixture parameters with Wavelet + Attention Advanced Encoder
            params_l = self.cnn_modules[l](
                low_res_context=lower_res_ctx, target_shape=target_x.shape
            )

            if self.training or not compute_entropy:
                B, C, H, W = target_x.shape
                logit_w, means, log_scales = self.mixture_evaluator.parse_params(params_l, in_channels=C)
                log_p_k = self.mixture_evaluator.log_prob_per_component(target_x, means, log_scales)
                log_p_k_rgb = log_p_k.sum(dim=2)
                log_w = F.log_softmax(logit_w, dim=1)
                log_px = torch.logsumexp(log_w + log_p_k_rgb, dim=1)
                
                nll_map = -log_px / 0.6931471805599453 # bits per pixel
                entropy_map = torch.zeros_like(nll_map)
            else:
                nll_map, entropy_map = self.mixture_evaluator.compute_nll_and_entropy(target_x, params_l)

            avg_nll = nll_map.mean(dim=[-2, -1])
            avg_h = entropy_map.mean(dim=[-2, -1])
            avg_d = avg_nll - avg_h

            nll_levels.append(avg_nll)
            h_levels.append(avg_h)
            d_levels.append(avg_d)

            total_nll_loss = total_nll_loss + avg_nll.mean()

        d0 = d_levels[0]
        d1 = d_levels[1]
        delta01 = d0 - d1

        return {
            "total_loss": total_nll_loss,
            "nll_levels": nll_levels,
            "h_levels": h_levels,
            "d_levels": d_levels,
            "d0": d0,
            "abs_d0": torch.abs(d0),
            "delta01": delta01,
            "abs_delta01": torch.abs(delta01)
        }
