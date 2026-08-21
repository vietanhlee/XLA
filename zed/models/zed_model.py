"""
Full Multi-Resolution ZED Model (Zero-Shot Detection of AI-Generated Images).
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

from .logistic_mixture import DiscretizedLogisticMixture
from .cnn_encoder import SReCCNN

class ZEDModel(nn.Module):
    """
    Multi-Resolution Zero-Shot AI Image Detector (ZED).
    
    Contains 3 SReCCNN models (CNN 0, CNN 1, CNN 2) corresponding to resolution scales:
    - Level 0: Full Resolution
    - Level 1: 1/2 Resolution (Subsampled 2x)
    - Level 2: 1/4 Resolution (Subsampled 4x)
    - Level 3: 1/8 Resolution (Subsampled 8x, acts as initial prompt/context for Level 2)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_mixtures: int = 10,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        min_log_scale: float = -7.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_mixtures = num_mixtures
        
        # Logistic Mixture Evaluator
        self.mixture_evaluator = DiscretizedLogisticMixture(
            num_mixtures=num_mixtures, min_log_scale=min_log_scale
        )
        
        # 3 SReCCNN modules for scales l = 0, 1, 2
        self.cnn_modules = nn.ModuleList([
            SReCCNN(
                in_channels=in_channels,
                num_mixtures=num_mixtures,
                hidden_channels=hidden_channels,
                num_res_blocks=num_res_blocks
            )
            for _ in range(3)
        ])

    def build_pyramid(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Builds image pyramid levels x^(0), x^(1), x^(2), x^(3) and smooth version y^(l).
        
        Args:
            x: Input image tensor in [0, 255] range of shape (B, C, H, W)
            
        Returns:
            x_levels: List of discrete rounded image tensors [x^(0), x^(1), x^(2), x^(3)]
            y_levels: List of unrounded float image tensors [y^(0), y^(1), y^(2), y^(3)]
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
        Forward pass computing multi-resolution NLLs, Entropies, and Decision Statistics.
        
        Args:
            x: Batch of RGB images (B, C, H, W) with pixel values in range [0, 255]
            compute_entropy: If True, computes exact expected entropy H at test time.
            
        Returns:
            Dict containing:
                - 'total_loss': Sum of NLLs over all scales (for training)
                - 'nll_levels': List of average NLL values [NLL^(0), NLL^(1), NLL^(2)]
                - 'h_levels': List of average Entropy values [H^(0), H^(1), H^(2)]
                - 'd_levels': List of coding cost gaps D^(l) = NLL^(l) - H^(l)
                - 'd0': Decision statistic D^(0)
                - 'abs_d0': |D^(0)|
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

        total_nll_loss = 0.0

        # Loop through levels l = 0, 1, 2
        for l in range(3):
            target_x = x_levels[l]             # x^(l)
            lower_res_ctx = y_levels[l + 1]     # y^(l+1)
            
            # Predict mixture distribution parameters using CNN l
            params_l = self.cnn_modules[l](
                low_res_context=lower_res_ctx, target_shape=target_x.shape
            )

            if self.training or not compute_entropy:
                # During training, fast path: compute only NLL map
                B, C, H, W = target_x.shape
                logit_w, means, log_scales = self.mixture_evaluator.parse_params(params_l, in_channels=C)
                log_p_k = self.mixture_evaluator.log_prob_per_component(target_x, means, log_scales)
                log_p_k_rgb = log_p_k.sum(dim=2)
                log_w = F.log_softmax(logit_w, dim=1)
                log_px = torch.logsumexp(log_w + log_p_k_rgb, dim=1)
                
                # NLL in bits per pixel
                nll_map = -log_px / 0.6931471805599453 # ln(2)
                entropy_map = torch.zeros_like(nll_map)
            else:
                # Test/Inference: compute exact NLL and Entropy
                nll_map, entropy_map = self.mixture_evaluator.compute_nll_and_entropy(target_x, params_l)

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
