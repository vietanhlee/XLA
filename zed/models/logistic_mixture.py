"""
Discretized Logistic Mixture Distribution for Lossless Image Density Estimation.
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al.)
Reference: SReC (Cao et al., 2020) & PixelCNN++ (Salimans et al., 2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class DiscretizedLogisticMixture(nn.Module):
    """
    Computes Negative Log-Likelihood (NLL) and Expected Entropy (H) 
    using a mixture of K discretized logistic distributions over 8-bit image pixels (0..255).
    """

    def __init__(self, num_mixtures: int = 10, min_log_scale: float = -7.0):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.min_log_scale = min_log_scale

    def parse_params(
        self, params: torch.Tensor, in_channels: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parses raw CNN output parameters into mixture logits, means, and log-scales.
        
        Args:
            params: Tensor of shape (B, K * (1 + 2 * C), H, W)
            in_channels: C = 3 for RGB
        
        Returns:
            logit_weights: (B, K, H, W)
            means: (B, K, C, H, W)
            log_scales: (B, K, C, H, W)
        """
        B, _, H, W = params.shape
        K = self.num_mixtures
        C = in_channels

        # 1. Mixture component logits: shape (B, K, H, W)
        logit_weights = params[:, :K, :, :]
        
        # 2. Means: shape (B, K * C, H, W) -> (B, K, C, H, W)
        means = params[:, K : K + K * C, :, :].view(B, K, C, H, W)
        
        # 3. Log-scales: shape (B, K * C, H, W) -> (B, K, C, H, W)
        log_scales = params[:, K + K * C :, :, :].view(B, K, C, H, W)
        log_scales = torch.clamp(log_scales, min=self.min_log_scale)

        return logit_weights, means, log_scales

    def log_prob_per_component(
        self, x: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes log P_k(x) for each component k across pixel values x in range [0, 255].
        
        Args:
            x: Image tensor of shape (B, C, H, W) with pixel values in [0, 255]
            means: (B, K, C, H, W)
            log_scales: (B, K, C, H, W)
            
        Returns:
            log_probs: (B, K, C, H, W)
        """
        orig_dtype = means.dtype
        # Compute in float32 for rock-solid numerical precision under mixed precision (AMP)
        x_exp = x.unsqueeze(1).float()
        means = means.float()
        log_scales = log_scales.float()
        inv_scales = torch.exp(-log_scales)

        # Scale and center
        centered_x = x_exp - means
        plus_in = inv_scales * (centered_x + 0.5)
        minus_in = inv_scales * (centered_x - 0.5)

        # Cumulative distribution function (CDF) for logistic: sigmoid(z)
        cdf_plus = torch.sigmoid(plus_in)
        cdf_minus = torch.sigmoid(minus_in)

        # Log probability computation with boundary checks for 8-bit discrete values
        # Softplus/log_sigmoid for edge stability
        log_cdf_plus = F.logsigmoid(plus_in)
        log_one_minus_cdf_minus = F.logsigmoid(-minus_in)
        
        # Mid-range difference CDF
        cdf_delta = cdf_plus - cdf_minus
        # Clamp delta to prevent log(0)
        log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-7))

        # Select based on x values
        log_probs = torch.where(
            x_exp <= 0.001,
            log_cdf_plus,
            torch.where(
                x_exp >= 254.999,
                log_one_minus_cdf_minus,
                log_cdf_delta
            )
        )
        return log_probs.to(orig_dtype)

    def compute_nll_and_entropy(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes per-pixel NLL (Negative Log-Likelihood) and Expected Entropy H.
        
        Args:
            x: Target image tensor (B, C, H, W) with values in range [0, 255]
            params: Raw CNN output tensor (B, K * (1 + 2 * C), H, W)
            
        Returns:
            nll_map: Per-pixel NLL map of shape (B, H, W) in bits per pixel
            entropy_map: Per-pixel Expected Entropy map of shape (B, H, W) in bits per pixel
        """
        B, C, H, W = x.shape
        K = self.num_mixtures
        
        logit_weights, means, log_scales = self.parse_params(params, in_channels=C)
        
        # 1. Compute Log Probabilities for target x
        # log_p_k: (B, K, C, H, W)
        log_p_k = self.log_prob_per_component(x, means, log_scales)
        
        # Sum log_p_k across color channels C assuming conditional independence given context
        # log_p_k_rgb: (B, K, H, W)
        log_p_k_rgb = log_p_k.sum(dim=2)
        
        # Log mixture weights: (B, K, H, W)
        log_w = F.log_softmax(logit_weights, dim=1)
        
        # Log-Sum-Exp over mixture components: log P(x) = logsumexp(log_w + log_p_k_rgb)
        # Result shape: (B, H, W)
        log_px = torch.logsumexp(log_w + log_p_k_rgb, dim=1)
        
        # Convert NLL to bits (nats / ln(2) = bits)
        # NLL per pixel (sum over channels C, expressed in bits)
        nll_map = (-log_px / math.log(2.0)).float()

        # 2. Compute Expected Entropy H(x) over all 256 discrete levels per color channel
        entropy_map = self._compute_entropy_exact(logit_weights, means, log_scales, B, C, H, W)

        return nll_map, entropy_map

    def _compute_entropy_exact(
        self,
        logit_weights: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        B: int,
        C: int,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Evaluates exact marginal expected entropy H = sum_c H(P_c) over all discrete 8-bit
        intensities v in [0..255] using vectorized 255-boundary CDF evaluation and pixel chunking.

        Exploits linearity of expectation over mixture weights:
            P_c(v) = sum_k w_k * [CDF_{k,c}(v + 0.5) - CDF_{k,c}(v - 0.5)]
                   = CDF_mix(v + 0.5) - CDF_mix(v - 0.5)
        This eliminates 64 sequential intensity loops, repeat_interleave copies, and redundant
        log/exp/logsumexp conversions while fixing channel-diagonal evaluation and preserving
        100% compatibility with existing trained checkpoints (no extra parameters or buffers).

        Returns:
            entropy_map: Tensor of shape (B, H, W) in float32 bits per pixel.
        """
        device = logit_weights.device
        K = self.num_mixtures

        # Evaluate in float32 for numerical stability even when CNN runs under AMP autocast
        w = F.softmax(logit_weights.float(), dim=1)       # (B, K, H, W)
        means_f = means.float()                           # (B, K, C, H, W)
        inv_scales = torch.exp(-log_scales.float())       # (B, K, C, H, W)

        num_pixels = B * H * W
        w_flat = w.permute(0, 2, 3, 1).reshape(num_pixels, K)
        means_flat = means_f.permute(0, 3, 4, 1, 2).reshape(num_pixels, K, C)
        inv_flat = inv_scales.permute(0, 3, 4, 1, 2).reshape(num_pixels, K, C)

        # 255 bin boundary edges: 0.5, 1.5, ..., 254.5 (created on-the-fly to keep state_dict unchanged)
        edges = torch.arange(0.5, 255.0, 1.0, device=device, dtype=torch.float32)

        # Adaptive spatial pixel chunk size keeps peak VRAM bounded (~250-750MB regardless of batch size)
        chunk_pixels = max(1024, (8192 * 30) // max(1, K * C))
        total_entropy = torch.empty(num_pixels, device=device, dtype=torch.float32)
        inv_ln2 = 1.0 / math.log(2.0)

        for start_idx in range(0, num_pixels, chunk_pixels):
            end_idx = min(start_idx + chunk_pixels, num_pixels)
            n_c = end_idx - start_idx

            m_c = means_flat[start_idx:end_idx].unsqueeze(-1)  # (n_c, K, C, 1)
            s_c = inv_flat[start_idx:end_idx].unsqueeze(-1)    # (n_c, K, C, 1)
            w_c = w_flat[start_idx:end_idx]                    # (n_c, K)

            # Evaluate logistic CDF at all 255 bin edges simultaneously
            cdf = torch.sigmoid(s_c * (edges - m_c))           # (n_c, K, C, 255)

            # Mix component CDFs via batched matrix multiply: (n_c, 1, K) @ (n_c, K, C*255)
            cdf_mix = torch.bmm(
                w_c.unsqueeze(1), cdf.reshape(n_c, K, C * 255)
            ).reshape(n_c, C, 255)

            # Recover exact discrete probabilities over [0..255] per channel
            p_cv = torch.empty((n_c, C, 256), device=device, dtype=torch.float32)
            p_cv[..., 0] = cdf_mix[..., 0]
            p_cv[..., 1:255] = cdf_mix[..., 1:] - cdf_mix[..., :-1]
            p_cv[..., 255] = 1.0 - cdf_mix[..., -1]

            # Clamp tiny FP32 rounding negatives to 0 so torch.special.entr never yields NaN
            p_cv.clamp_(min=0.0, max=1.0)

            # entr(p) = -p * ln(p) with entr(0) = 0; divide by ln(2) for bits per pixel
            total_entropy[start_idx:end_idx] = torch.special.entr(p_cv).sum(dim=(-1, -2)) * inv_ln2

        return total_entropy.reshape(B, H, W)
