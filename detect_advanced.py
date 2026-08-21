"""
Zero-Shot AI Image Detection & Evaluation for Advanced ZED Model.
Calculates D^(0) and |Delta^01| decision statistics using AdvancedZEDModel (Wavelet + Attention).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from config import ModelConfig
from zed.models import AdvancedZEDModel
from zed.dataset import EvaluationImageDataset
from zed.utils import load_checkpoint, compute_metrics

def evaluate_advanced_zero_shot(
    model: AdvancedZEDModel,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_labels = []
    all_d0 = []
    all_abs_d0 = []
    all_delta01 = []

    print("Evaluating test images with Advanced ZED (computing NLL & exact Entropy)...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Detecting Advanced"):
            images = images.to(device)
            output_dict = model(images, compute_entropy=True)

            d0 = output_dict["d0"].cpu().numpy()
            abs_d0 = output_dict["abs_d0"].cpu().numpy()
            abs_delta01 = output_dict["abs_delta01"].cpu().numpy()

            all_labels.extend(labels.numpy())
            all_d0.extend(d0)
            all_abs_d0.extend(abs_d0)
            all_delta01.extend(abs_delta01)

    return (
        np.array(all_labels),
        np.array(all_d0),
        np.array(all_abs_d0),
        np.array(all_delta01)
    )

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Detection with Advanced ZED Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/zed_advanced_best.pth", help="Path to Advanced ZED checkpoint.")
    parser.add_argument("--real_dir", type=str, default="data/test/real", help="Directory containing test REAL images.")
    parser.add_argument("--fake_dir", type=str, default="data/test/fake", help="Directory containing test FAKE / AI images.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=== Advanced ZED Zero-Shot Detection Evaluation ===")
    print("Features: 2D Haar Wavelet + Spatial Self-Attention")
    print(f"Device: {device}")

    model_cfg = ModelConfig()
    model = AdvancedZEDModel(
        in_channels=model_cfg.in_channels,
        num_mixtures=model_cfg.num_mixtures,
        hidden_channels=model_cfg.hidden_channels,
        num_res_blocks=model_cfg.num_res_blocks,
        num_heads=4,
        min_log_scale=model_cfg.min_log_scale
    ).to(device)

    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device)
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found! Running with randomly initialized Advanced model.")

    dataset = EvaluationImageDataset(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir
    )

    if len(dataset) == 0:
        print(f"ERROR: No evaluation images found in '{args.real_dir}' or '{args.fake_dir}'.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0
    )

    labels, d0_scores, abs_d0_scores, delta01_scores = evaluate_advanced_zero_shot(model, dataloader, device)

    print("\n" + "="*55)
    print(" === ADVANCED ZED ZERO-SHOT DETECTION RESULTS ===")
    print("="*55)

    metrics_d0 = compute_metrics(d0_scores, labels)
    print(f"[Stat D^(0)]   -> ROC-AUC: {metrics_d0['auc']:.2f}% | Best Acc: {metrics_d0['best_acc']:.2f}% | Threshold: {metrics_d0['best_threshold']:.4f}")

    metrics_abs_d0 = compute_metrics(abs_d0_scores, labels)
    print(f"[Stat |D^(0)|] -> ROC-AUC: {metrics_abs_d0['auc']:.2f}% | Best Acc: {metrics_abs_d0['best_acc']:.2f}% | Threshold: {metrics_abs_d0['best_threshold']:.4f}")

    metrics_delta01 = compute_metrics(delta01_scores, labels)
    print(f"[Stat |Delta^01|] -> ROC-AUC: {metrics_delta01['auc']:.2f}% | Best Acc: {metrics_delta01['best_acc']:.2f}% | Threshold: {metrics_delta01['best_threshold']:.4f}")

    print("="*55)

if __name__ == "__main__":
    main()
