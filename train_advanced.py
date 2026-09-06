"""
Training Entry Point for Advanced ZED Model (Wavelet + Cross-Scale Attention + Robust Augmentations).
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al. - Advanced Upgrade)
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Force unbuffered line-by-line output for tqdm on Google Colab / Notebook
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Add root dir to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from config import ModelConfig, TrainConfig
from zed.models import AdvancedZEDModel
from zed.augmentations import RobustRealImageTransform
from zed.dataset import create_train_val_datasets
from zed.trainer import run_training
from zed.utils import get_safe_device, wrap_model_multigpu


def main():
    parser = argparse.ArgumentParser(description="Train Advanced ZED Model on REAL images.")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\levie\Downloads\img\images", help="Path to folder containing real images (scans recursively).")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set split ratio (0.0 to disable).")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs).")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Early stopping min delta improvement.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    parser.add_argument("--num_workers", type=int, default=min(4, max(1, os.cpu_count() or 2)), help="DataLoader worker threads.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file (.pth) to resume training from.")
    parser.add_argument("--multigpu", action="store_true", default=True, help="Enable multi-GPU DataParallel training if multiple GPUs exist (default: True).")
    parser.add_argument("--no_multigpu", dest="multigpu", action="store_false", help="Force single-GPU mode even if multiple GPUs exist.")
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )

    device = get_safe_device(train_cfg.device)
    print("=== Starting Advanced ZED Training Pipeline ===")
    print("Features: 2D Haar Wavelet (High-Frequency) + Spatial Self-Attention + Robust Augmentations")
    print(f"Device: {device} | Data Directory (Recursive): {train_cfg.data_dir}")

    if not os.path.exists(train_cfg.data_dir):
        print(f"Warning: Directory '{train_cfg.data_dir}' does not exist. Creating directory.")
        os.makedirs(train_cfg.data_dir, exist_ok=True)

    # Robust transform for Advanced ZED training
    train_transform = RobustRealImageTransform(image_size=train_cfg.image_size)

    # 1. Create Datasets with Recursive Scanning and Train/Val Split
    train_dataset, val_dataset = create_train_val_datasets(
        data_dir=train_cfg.data_dir,
        val_split=args.val_split,
        image_size=train_cfg.image_size,
        train_transform=train_transform,
        seed=42
    )

    # Tính toán số workers an toàn để không bị UserWarning trên Colab (tối đa bằng số CPU core)
    workers = min(train_cfg.num_workers, max(1, os.cpu_count() or 2)) if train_cfg.device == "cuda" else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(train_cfg.device == "cuda")
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=(train_cfg.device == "cuda")
        )

    # 2. Build Advanced ZED Model & Optimizer
    model = AdvancedZEDModel(
        in_channels=model_cfg.in_channels,
        num_mixtures=model_cfg.num_mixtures,
        hidden_channels=model_cfg.hidden_channels,
        num_res_blocks=model_cfg.num_res_blocks,
        num_heads=4,
        min_log_scale=model_cfg.min_log_scale
    ).to(device)

    # Multi-GPU Auto Detection & DataParallel Wrap
    model, gpu_count = wrap_model_multigpu(model, device, use_multigpu=args.multigpu)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.num_epochs, eta_min=1e-6
    )

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=train_cfg.mixed_precision and device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.mixed_precision and device.type == "cuda")

    # 3. Launch Training Engine
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_epochs=train_cfg.num_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip_norm=train_cfg.grad_clip_norm,
        mixed_precision=train_cfg.mixed_precision,
        checkpoint_dir=train_cfg.checkpoint_dir,
        save_interval=train_cfg.save_interval,
        resume_path=args.resume,
        model_name="Advanced ZED",
        best_checkpoint_filename="zed_advanced_best.pth",
        last_checkpoint_filename="zed_advanced_last.pth"
    )


if __name__ == "__main__":
    main()
