"""
Training Script for ZED (Zero-Shot Detection of AI-Generated Images).
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al.)

IMPORTANT: Training is performed STRICTLY on REAL IMAGES ONLY! No fake images needed.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ModelConfig, TrainConfig
from zed.models.zed_model import ZEDModel
from zed.dataset import RealImageDataset
from zed.utils import save_checkpoint

from typing import Any

def train_epoch(
    model: ZEDModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    grad_clip_norm: float,
    mixed_precision: bool,
    epoch: int,
    total_epochs: int,
    current_lr: float
) -> float:
    """Trains the model for one full epoch on real images with detailed tqdm progress bar."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:02d}/{total_epochs:02d}]",
        leave=True,
        dynamic_ncols=True
    )
    
    for batch_idx, batch_images in enumerate(pbar, 1):
        # Transfer real images to device (shape B, C, H, W, range [0, 255])
        batch_images = batch_images.to(device)

        optimizer.zero_grad()

        if mixed_precision and device.type == "cuda":
            with torch.amp.autocast(device_type=device.type):
                output_dict = model(batch_images, compute_entropy=False)
                loss = output_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            output_dict = model(batch_images, compute_entropy=False)
            loss = output_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        running_avg_loss = total_loss / batch_idx

        # Construct status dict for tqdm
        postfix = {
            "Batch Loss": f"{loss_val:.4f}",
            "Avg Loss": f"{running_avg_loss:.4f}",
            "LR": f"{current_lr:.6f}"
        }
        if device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            postfix["GPU Mem"] = f"{mem_mb:.0f}MB"

        pbar.set_postfix(postfix)

    return total_loss / max(1, num_batches)

def main():
    parser = argparse.ArgumentParser(description="Train ZED Density Estimator on REAL images.")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\levie\Downloads\img\images", help="Path to folder containing real images.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    args = parser.parse_args()

    # Load Configs
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )

    device = torch.device(train_cfg.device)
    print(f"=== Starting ZED Training Pipeline ===")
    print(f"Device: {device}")
    print(f"Real Images Directory: {train_cfg.data_dir}")

    # Create Dataset & DataLoader
    if not os.path.exists(train_cfg.data_dir):
        print(f"Warning: Directory '{train_cfg.data_dir}' does not exist yet. Creating placeholder dir.")
        os.makedirs(train_cfg.data_dir, exist_ok=True)

    dataset = RealImageDataset(
        data_dir=train_cfg.data_dir,
        image_size=train_cfg.image_size
    )

    if len(dataset) == 0:
        print(f"ERROR: No image files found in '{train_cfg.data_dir}'. Please place real image files (.jpg, .png) there!")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers if train_cfg.device == "cuda" else 0,
        pin_memory=(train_cfg.device == "cuda")
    )

    # Initialize Model, Optimizer, Scheduler & Scaler
    model = ZEDModel(
        in_channels=model_cfg.in_channels,
        num_mixtures=model_cfg.num_mixtures,
        hidden_channels=model_cfg.hidden_channels,
        num_res_blocks=model_cfg.num_res_blocks,
        min_log_scale=model_cfg.min_log_scale
    ).to(device)

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

    best_loss = float("inf")

    print(f"Total Dataset Size: {len(dataset)} images ({len(dataloader)} batches per epoch)")
    print(f"Beginning training for {train_cfg.num_epochs} epochs...\n")

    # Outer Epoch Loop
    for epoch in range(1, train_cfg.num_epochs + 1):
        start_time = time.time()
        current_lr = scheduler.get_last_lr()[0]
        
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip_norm=train_cfg.grad_clip_norm,
            mixed_precision=train_cfg.mixed_precision,
            epoch=epoch,
            total_epochs=train_cfg.num_epochs,
            current_lr=current_lr
        )

        scheduler.step()
        elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start_time))

        tqdm.write(f"✓ [Epoch {epoch:02d}/{train_cfg.num_epochs:02d}] Completed in {elapsed} | Final NLL Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # Save Checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(train_cfg.checkpoint_dir, "zed_best.pth")
            save_checkpoint(model, optimizer, epoch, avg_loss, best_path)

        if epoch % train_cfg.save_interval == 0:
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"zed_epoch_{epoch:02d}.pth")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)

    print("\n🎉 === Training Complete ===")

if __name__ == "__main__":
    main()
