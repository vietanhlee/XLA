"""
Configuration module for ZED (Zero-Shot Detection of AI-Generated Images)
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al.)
"""

from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class ModelConfig:
    """Hyperparameters for the ZED multi-resolution SReC architecture."""
    in_channels: int = 3                # RGB images
    num_mixtures: int = 10              # K = 10 discrete logistic mixture components
    hidden_channels: int = 64           # CNN hidden channels
    num_res_blocks: int = 4             # Number of Residual Blocks per scale CNN
    num_scales: int = 3                 # Level 0, Level 1, Level 2 (Level 3 is base prompt)
    min_log_scale: float = -7.0         # Minimum log-scale for numerical stability in logistic mixture

import os

@dataclass
class TrainConfig:
    """Hyperparameters for training the Density Estimator on REAL images."""
    data_dir: str = "data/real_images"  # Directory containing REAL images ONLY
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    image_size: Tuple[int, int] = (256, 256)
    num_workers: int = min(4, max(1, os.cpu_count() or 2))
    device: str = "cuda"                # "cuda" or "cpu"
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True

@dataclass
class DetectConfig:
    """Settings for Zero-Shot Detection on test images (Real vs AI-generated)."""
    checkpoint_path: str = "checkpoints/zed_best.pth"
    real_data_dir: str = "data/test/real"
    fake_data_dir: str = "data/test/fake"
    batch_size: int = 8
    image_size: Tuple[int, int] = (256, 256)
    device: str = "cuda"
    output_dir: str = "results"
