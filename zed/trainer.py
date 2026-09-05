"""
Core Reusable Training Engine for ZED & Advanced ZED Models.
Handles epoch iteration, mixed precision, gradient clipping, validation, early stopping, and checkpointing.
"""

import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from zed.utils import save_checkpoint, EarlyStopping


import sys

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    grad_clip_norm: float,
    mixed_precision: bool,
    epoch: int,
    total_epochs: int,
    current_lr: float,
    model_name: str = "ZED"
) -> float:
    """Trains the density estimator model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] ({model_name} Train)",
        leave=True,
        file=sys.stdout,
        dynamic_ncols=True
    )

    for batch_idx, batch_images in enumerate(pbar, 1):
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

        postfix = {
            "Loss": f"{loss_val:.4f}",
            "Avg Loss": f"{running_avg_loss:.4f}",
            "LR": f"{current_lr:.6f}"
        }
        if device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            postfix["GPU Mem"] = f"{mem_mb:.0f}MB"

        pbar.set_postfix(postfix)

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    epoch: int,
    total_epochs: int,
    model_name: str = "ZED"
) -> float:
    """Evaluates validation NLL loss across validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] ({model_name} Val)  ",
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True
    )

    for batch_images in pbar:
        batch_images = batch_images.to(device)

        if mixed_precision and device.type == "cuda":
            with torch.amp.autocast(device_type=device.type):
                output_dict = model(batch_images, compute_entropy=False)
                loss = output_dict["total_loss"]
        else:
            output_dict = model(batch_images, compute_entropy=False)
            loss = output_dict["total_loss"]

        total_loss += loss.item()

    return total_loss / max(1, num_batches)


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    num_epochs: int,
    patience: int,
    min_delta: float,
    grad_clip_norm: float,
    mixed_precision: bool,
    checkpoint_dir: str,
    save_interval: int,
    resume_path: Optional[str] = None,
    model_name: str = "ZED",
    best_checkpoint_filename: str = "zed_best.pth",
    last_checkpoint_filename: str = "zed_last.pth"
):
    """
    Main training execution loop supporting train/val split, early stopping, checkpointing, and resuming.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
    
    start_epoch = 1
    best_val_loss = float("inf")

    # Resume training if checkpoint path is provided
    if resume_path and os.path.exists(resume_path):
        print(f"\n🔄 Resuming training from checkpoint: {resume_path}")
        checkpoint = load_checkpoint(resume_path, model, optimizer=optimizer, scheduler=scheduler, device=device.type)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("loss", float("inf"))
        early_stopping.best_loss = best_val_loss
        print(f"   -> Resuming from Epoch {start_epoch} (Previous best loss: {best_val_loss:.4f})")
    elif resume_path:
        print(f"⚠️ Resume checkpoint '{resume_path}' not found! Starting training from scratch (Epoch 1).")

    print(f"Beginning {model_name} training for epochs [{start_epoch}/{num_epochs}] (Early stopping patience = {patience})...\n")

    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()
        current_lr = scheduler.get_last_lr()[0]

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip_norm=grad_clip_norm,
            mixed_precision=mixed_precision,
            epoch=epoch,
            total_epochs=num_epochs,
            current_lr=current_lr,
            model_name=model_name
        )

        scheduler.step()
        elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start_time))

        if val_loader is not None:
            val_loss = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                mixed_precision=mixed_precision,
                epoch=epoch,
                total_epochs=num_epochs,
                model_name=model_name
            )
            tqdm.write(f"✓ [{model_name} Epoch {epoch:02d}/{num_epochs:02d}] ({elapsed}) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

            # Check Early Stopping based on validation loss
            is_improved = early_stopping(val_loss)
            if is_improved:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, best_checkpoint_filename)
                save_checkpoint(model, optimizer, epoch, val_loss, best_path, scheduler=scheduler)

            if early_stopping.early_stop:
                tqdm.write(f"\n✋ Early Stopping triggered at Epoch {epoch:02d}! Best Validation Loss: {early_stopping.best_loss:.4f}")
                break
        else:
            tqdm.write(f"✓ [{model_name} Epoch {epoch:02d}/{num_epochs:02d}] ({elapsed}) | Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_path = os.path.join(checkpoint_dir, best_checkpoint_filename)
                save_checkpoint(model, optimizer, epoch, train_loss, best_path, scheduler=scheduler)

        if epoch % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name.lower().replace(' ', '_')}_epoch_{epoch:02d}.pth")
            save_checkpoint(model, optimizer, epoch, train_loss, ckpt_path, scheduler=scheduler)

    # Save last checkpoint
    last_path = os.path.join(checkpoint_dir, last_checkpoint_filename)
    save_checkpoint(model, optimizer, epoch, train_loss, last_path, scheduler=scheduler)
    print(f"\n🎉 === {model_name} Training Complete ===")
