"""
Modern Multi-Resolution ZED Model (Zero-Shot Detection of AI-Generated Images).
Upgrades standard ZED with:
  - ConvNeXt-SReC Encoder (7x7 Depthwise Conv + Inverted Bottleneck + ECA)
  - Preserved Pure Inductive Bias (Zero spatial blurring / no ViT cross-entropy collapse)
  - Top-K% Patch Anomaly Metric and Spatial Variance Anomaly Metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

from .logistic_mixture import DiscretizedLogisticMixture
from .convnext_encoder import ConvNeXtSReCCNN


class ModernZEDModel(nn.Module):
    """
    Modern Multi-Resolution Zero-Shot AI Image Detector.
    Architecture:
      - 3 Resolution scales (Levels 0, 1, 2) driven by ConvNeXt-SReC CNNs.
      - Discretized Logistic Mixture Density Estimator (K=10).
      - Multi-Resolution Image Pyramid (Levels 0, 1, 2, 3).
      - Dual Decision Metrics: Mean Coding Cost D^(0) + Top-10% Patch Anomaly D^(0)_top10.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_mixtures: int = 10,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        min_log_scale: float = -7.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_mixtures = num_mixtures

        # Logistic Mixture Evaluator
        self.mixture_evaluator = DiscretizedLogisticMixture(
            num_mixtures=num_mixtures, min_log_scale=min_log_scale
        )

        # 3 ConvNeXt-SReC CNN modules for resolution scales l = 0, 1, 2
        self.cnn_modules = nn.ModuleList([
            ConvNeXtSReCCNN(
                in_channels=in_channels,
                num_mixtures=num_mixtures,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks
            )
            for _ in range(3)
        ])

    def build_pyramid(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Builds image pyramid levels x^(0), x^(1), x^(2), x^(3) and smooth version y^(l).
        """
        x_levels = [x]
        y_levels = [x]

        curr_x = x
        for _ in range(3):
            # 2x2 Average Pooling
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
        Forward pass computing multi-resolution NLLs, Entropies, and Advanced Decision Statistics.

        Args:
            x: Batch of RGB images (B, C, H, W) with pixel values in range [0, 255]
            compute_entropy: If True, computes exact expected entropy H at test time.

        Returns:
            Dict containing:
                - 'total_loss': Sum of NLLs over all scales (for training)
                - 'nll_levels': List of average NLL values [NLL^(0), NLL^(1), NLL^(2)]
                - 'h_levels': List of average Entropy values [H^(0), H^(1), H^(2)]
                - 'd_levels': List of coding cost gaps D^(l) = NLL^(l) - H^(l)
                - 'd0': Decision statistic D^(0) (Mean)
                - 'abs_d0': |D^(0)|
                - 'd0_top10': Top-10% Patch Anomaly Statistic
                - 'd0_std': Spatial Variance of coding cost gap
                - 'delta01': Decision statistic Delta^01 = D^(0) - D^(1)
                - 'abs_delta01': |Delta^01|
        """
        # Ensure x is in range [0, 255]
        if x.max() <= 1.0 + 1e-5:
            x = x * 255.0

        x_levels, y_levels = self.build_pyramid(x)

        nll_levels = []
        h_levels = []
        d_levels = []
        d_maps = []

        total_nll_loss = 0.0

        # Loop through levels l = 0, 1, 2
        for l in range(3):
            target_x = x_levels[l]             # x^(l)
            lower_res_ctx = y_levels[l + 1]     # y^(l+1)

            # Predict mixture distribution parameters using ConvNeXt-SReC CNN l
            params_l = self.cnn_modules[l](
                low_res_context=lower_res_ctx, target_shape=target_x.shape
            )

            if self.training or not compute_entropy or l == 2:
                # Fast path: compute only NLL map (Level 2 does not participate in D^(0) or Delta^01)
                B, C, H, W = target_x.shape
                logit_w, means, log_scales = self.mixture_evaluator.parse_params(params_l, in_channels=C)
                log_p_k = self.mixture_evaluator.log_prob_per_component(target_x, means, log_scales)
                log_p_k_rgb = log_p_k.sum(dim=2)
                log_w = F.log_softmax(logit_w, dim=1)
                log_px = torch.logsumexp(log_w + log_p_k_rgb, dim=1)

                nll_map = -log_px / 0.6931471805599453 # ln(2) -> bits per pixel
                entropy_map = torch.zeros_like(nll_map)
            else:
                # Test/Inference for Level 0 & Level 1: compute exact NLL and Entropy
                nll_map, entropy_map = self.mixture_evaluator.compute_nll_and_entropy(target_x, params_l)

            # Numerical guard against potential inf/nan
            nll_map = torch.nan_to_num(nll_map, nan=100.0, posinf=100.0, neginf=0.0)
            entropy_map = torch.nan_to_num(entropy_map, nan=0.0, posinf=100.0, neginf=0.0)

            d_map = nll_map - entropy_map # (B, H, W)
            d_maps.append(d_map)

            # Spatial averages across pixels (B, H, W) -> scalar per sample
            avg_nll = nll_map.mean(dim=[-2, -1])          # (B,)
            avg_h = entropy_map.mean(dim=[-2, -1])        # (B,)
            avg_d = avg_nll - avg_h                       # (B,)

            nll_levels.append(avg_nll)
            h_levels.append(avg_h)
            d_levels.append(avg_d)

            total_nll_loss = total_nll_loss + avg_nll.mean()

        d0 = d_levels[0]
        d1 = d_levels[1]
        delta01 = d0 - d1

        # Calculate Top-K% Patch Anomaly for Level 0 (captures localized AI generation artifacts)
        d0_map = d_maps[0] # (B, H, W)
        B, H, W = d0_map.shape
        d0_flat = d0_map.view(B, H * W)
        k_top = max(1, int(0.10 * H * W)) # Top 10%
        topk_vals, _ = torch.topk(d0_flat, k=k_top, dim=1)
        d0_top10 = topk_vals.mean(dim=1) # (B,)
        d0_std = d0_flat.std(dim=1)      # (B,)

        # Ensure total_loss has at least 1 dimension (shape (1,)) so DataParallel can gather across GPUs without warning
        if isinstance(total_nll_loss, torch.Tensor):
            total_loss_tensor = total_nll_loss.unsqueeze(0) if total_nll_loss.dim() == 0 else total_nll_loss
        else:
            total_loss_tensor = torch.tensor([total_nll_loss], device=x.device)

        return {
            "total_loss": total_loss_tensor,
            "nll_levels": nll_levels,
            "h_levels": h_levels,
            "d_levels": d_levels,
            "d0": d0,
            "abs_d0": torch.abs(d0),
            "d0_top10": d0_top10,
            "d0_std": d0_std,
            "delta01": delta01,
            "abs_delta01": torch.abs(delta01)
        }
