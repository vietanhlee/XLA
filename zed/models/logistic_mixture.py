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
        # Expand x to match mixture component dimension K: (B, 1, C, H, W)
        x_exp = x.unsqueeze(1)
        inv_scales = torch.exp(-log_scales)

        # Scale and center
        centered_x = x_exp - means
        plus_in = inv_scales * (centered_x + 0.5)
        minus_in = inv_scales * (centered_x - 0.5)

        # Cumulative distribution function (CDF) for logistic: sigmoid(z)
        cdf_plus = torch.sigmoid(plus_in)
        cdf_minus = torch.sigmoid(minus_in)

        # Log probability computation with boundary checks for 8-bit discrete values
        # Case x == 0: log(cdf_plus)
        # Case x == 255: log(1 - cdf_minus)
        # Case 0 < x < 255: log(cdf_plus - cdf_minus)
        
        # Softplus/log_sigmoid for edge stability
        log_cdf_plus = F.logsigmoid(plus_in)
        log_one_minus_cdf_minus = F.logsigmoid(-minus_in)
        
        # Mid-range difference CDF
        cdf_delta = cdf_plus - cdf_minus
        # Clamp delta to prevent log(0)
        log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-12))

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
        return log_probs

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
        
        # Convert NLL to nats or bits (nats / ln(2) = bits)
        # NLL per pixel (sum over channels C, expressed in bits)
        nll_map = -log_px / math.log(2.0)

        # 2. Compute Expected Entropy H(x) = - sum_{v=0}^{255} P(v) * log2 P(v)
        # To compute H efficiently, we evaluate P(v) for all 256 pixel levels
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
        Evaluates exact expected entropy H over all discrete pixel intensities v in [0..255].
        Returns entropy_map of shape (B, H, W).
        """
        device = logit_weights.device
        K = self.num_mixtures
        log_w = F.log_softmax(logit_weights, dim=1) # (B, K, H, W)

        # Create all 256 intensity levels tensor v: shape (256,)
        v = torch.arange(0, 256, device=device, dtype=logit_weights.dtype)
        
        # We process intensities in chunks to control memory overhead
        chunk_size = 32
        total_entropy = torch.zeros((B, H, W), device=device, dtype=logit_weights.dtype)

        for start_idx in range(0, 256, chunk_size):
            end_idx = min(start_idx + chunk_size, 256)
            v_chunk = v[start_idx:end_idx] # (V_chunk,)

            # Reshape v_chunk to evaluate log_prob_per_component
            # v_chunk_img shape: (V_chunk, C, H, W)
            # Expand to compute batch
            V_c = v_chunk.shape[0]
            
            # Broadcast over (B, V_chunk, C, H, W)
            v_img = v_chunk.view(1, V_c, 1, 1, 1).expand(B, V_c, C, H, W)
            
            # Reshape to (B * V_chunk, C, H, W) for batch evaluation
            v_flat = v_img.reshape(B * V_c, C, H, W)
            
            # Expand means & log_scales: (B * V_chunk, K, C, H, W)
            means_exp = means.repeat_interleave(V_c, dim=0)
            log_scales_exp = log_scales.repeat_interleave(V_c, dim=0)
            
            log_p_k_chunk = self.log_prob_per_component(v_flat, means_exp, log_scales_exp)
            # Reshape back to (B, V_chunk, K, C, H, W)
            log_p_k_chunk = log_p_k_chunk.view(B, V_c, K, C, H, W)
            
            # Sum over channels C: (B, V_chunk, K, H, W)
            log_p_k_rgb_chunk = log_p_k_chunk.sum(dim=3)
            
            # Expand log_w: (B, 1, K, H, W)
            log_w_exp = log_w.unsqueeze(1)
            
            # log P(v) for each v in chunk: (B, V_chunk, H, W)
            log_pv_chunk = torch.logsumexp(log_w_exp + log_p_k_rgb_chunk, dim=2)
            
            # Convert to bits
            log2_pv_chunk = log_pv_chunk / math.log(2.0)
            pv_chunk = torch.exp(log_pv_chunk)
            
            # Contribution: - P(v) * log2 P(v)
            chunk_entropy = - (pv_chunk * log2_pv_chunk).sum(dim=1) # (B, H, W)
            total_entropy += chunk_entropy

        return total_entropy
