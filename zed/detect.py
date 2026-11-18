"""
Zero-Shot AI Image Detection & Evaluation Script.
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al.)

Evaluates test images (Real vs Fake/AI-generated) using the learned real-image density model.
Calculates decision statistics D^(0) and |Delta^01|, and outputs ROC-AUC & Detection Accuracy.
"""

import os
import sys
import argparse
from pathlib import Path

# Safe OpenMP workaround for Windows / multi-library environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Safe UTF-8 console output for Windows / Notebook environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from typing import Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ModelConfig, DetectConfig
from zed.models.zed_model import ZEDModel
from zed.dataset import EvaluationImageDataset, create_split_evaluation_datasets
import json
from zed.utils import (
    load_checkpoint,
    compute_metrics,
    plot_detection_dashboard,
    get_safe_device,
    wrap_model_multigpu,
    export_all_detection_plots
)

def evaluate_zero_shot(
    model: ZEDModel,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates test set images and extracts decision statistics.
    
    Returns:
        labels: True binary labels (0 = Real, 1 = Fake)
        d0_scores: D^(0) coding cost gap
        abs_d0_scores: |D^(0)|
        delta01_scores: |Delta^01| = |D^(0) - D^(1)|
    """
    model.eval()
    
    all_labels = []
    all_d0 = []
    all_abs_d0 = []
    all_delta01 = []

    print("Evaluating test images (computing NLL & exact Entropy)...")
    use_amp = (device.type == "cuda")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Detecting"):
            images = images.to(device, non_blocking=True)

            # Use mixed precision for CNN backbone forward; logistic mixture internally casts to float32
            if use_amp and hasattr(torch.amp, "autocast"):
                with torch.amp.autocast(device_type=device.type):
                    output_dict = model(images, compute_entropy=True)
            elif use_amp:
                with torch.cuda.amp.autocast():
                    output_dict = model(images, compute_entropy=True)
            else:
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
    parser = argparse.ArgumentParser(description="Zero-Shot Detection of AI-Generated Images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/zed_best.pth", help="Path to pre-trained ZED checkpoint.")
    parser.add_argument("--real_dir", type=str, default="data/test/real", help="Directory containing test REAL images.")
    parser.add_argument("--fake_dir", type=str, default="data/test/fake", help="Directory containing test FAKE / AI images.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference (default: 16).")
    parser.add_argument("--num_workers", type=int, default=min(4, max(1, os.cpu_count() or 2)), help="DataLoader worker threads.")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng ảnh AI / Fake được nạp từ folder AI detect (ví dụ: --limit 500 hoặc 1000).")
    parser.add_argument("--limit_fake", type=int, default=None, help="Alias cho --limit: giới hạn số lượng ảnh AI.")
    parser.add_argument("--limit_real", type=int, default=None, help="Giới hạn số lượng ảnh Real được nạp từ real_dir (tùy chọn).")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test images per class (real/fake). E.g. 1000 for fast testing in 2-3 mins.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    parser.add_argument("--multigpu", action="store_true", default=True, help="Enable multi-GPU DataParallel inference if multiple GPUs exist (default: True).")
    parser.add_argument("--test_split", type=float, default=0.30, help="Tỷ lệ tập Test (mặc định: 0.30 = 30%). 70% còn lại dùng làm tập Calibration để tìm threshold.")
    parser.add_argument("--output_dir", type=str, default="results_split", help="Thư mục xuất kết quả metrics và biểu đồ (mặc định: results_split).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cho việc phân chia dữ liệu tái lập (stratified split).")
    parser.add_argument("--scores_npz", type=str, default=None, help="Đường dẫn đến file .npz điểm số đã tính sẵn để chia và đánh giá nhanh không cần inference lại.")

    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # CASE 1: Nạp trực tiếp từ file điểm số đã lưu (.npz) để chia và đánh giá nhanh
    # -------------------------------------------------------------------------
    if args.scores_npz is not None and os.path.exists(args.scores_npz):
        print(f"📦 Loading pre-computed anomaly scores from: {args.scores_npz}")
        cached_data = np.load(args.scores_npz)
        all_labels = cached_data["labels"]
        all_d0 = cached_data["d0"]
        all_abs_d0 = cached_data["abs_d0"]
        all_delta01 = cached_data["delta01"]
        print(f"   -> Loaded {len(all_labels)} total samples ({np.sum(all_labels == 0)} Real, {np.sum(all_labels == 1)} AI/Fake).")

        if args.test_split > 0.0:
            rng = np.random.default_rng(args.seed)
            real_indices = np.where(all_labels == 0)[0]
            fake_indices = np.where(all_labels == 1)[0]
            rng.shuffle(real_indices)
            rng.shuffle(fake_indices)

            n_real_test = max(1, int(len(real_indices) * args.test_split))
            n_fake_test = max(1, int(len(fake_indices) * args.test_split))

            test_idx = np.concatenate([real_indices[:n_real_test], fake_indices[:n_fake_test]])
            calib_idx = np.concatenate([real_indices[n_real_test:], fake_indices[n_fake_test:]])

            calib_labels = all_labels[calib_idx]
            calib_d0 = all_d0[calib_idx]
            calib_abs_d0 = all_abs_d0[calib_idx]
            calib_delta01 = all_delta01[calib_idx]

            test_labels = all_labels[test_idx]
            test_d0 = all_d0[test_idx]
            test_abs_d0 = all_abs_d0[test_idx]
            test_delta01 = all_delta01[test_idx]

            print(f"\n📂 Stratified Split Completed (Seed={args.seed}, Test Split={args.test_split*100:.1f}%):")
            print(f"   -> Calibration Set ({(1-args.test_split)*100:.1f}%): {len(calib_labels)} samples ({np.sum(calib_labels == 0)} Real, {np.sum(calib_labels == 1)} AI)")
            print(f"   -> Test Set ({args.test_split*100:.1f}%): {len(test_labels)} samples ({np.sum(test_labels == 0)} Real, {np.sum(test_labels == 1)} AI)")
        else:
            calib_labels, calib_d0, calib_abs_d0, calib_delta01 = all_labels, all_d0, all_abs_d0, all_delta01
            test_labels, test_d0, test_abs_d0, test_delta01 = all_labels, all_d0, all_abs_d0, all_delta01

    # -------------------------------------------------------------------------
    # CASE 2: Inference trực tiếp từ các thư mục ảnh
    # -------------------------------------------------------------------------
    else:
        fake_limit = args.limit if args.limit is not None else (args.limit_fake if args.limit_fake is not None else args.max_samples)
        real_limit = args.limit_real if args.limit_real is not None else args.max_samples

        device = get_safe_device(args.device)
        print(f"=== ZED Zero-Shot Detection Evaluation ===")
        print(f"Device: {device} | Batch Size: {args.batch_size}")
        print(f"Real Dir: {args.real_dir}")
        print(f"Fake Dir: {args.fake_dir}")
        print(f"Output Dir: {results_dir}")
        print(f"Split Setting -> Test: {args.test_split*100:.1f}% | Calibration: {(1.0-args.test_split)*100:.1f}% | Seed: {args.seed}")
        print(f"Limit Settings -> AI/Fake limit: {fake_limit if fake_limit is not None else 'Unlimited'} | Real limit: {real_limit if real_limit is not None else 'Unlimited'}")

        # Load Model
        model_cfg = ModelConfig()
        model = ZEDModel(
            in_channels=model_cfg.in_channels,
            num_mixtures=model_cfg.num_mixtures,
            hidden_channels=model_cfg.hidden_channels,
            num_res_blocks=model_cfg.num_res_blocks,
            min_log_scale=model_cfg.min_log_scale
        ).to(device)

        if os.path.exists(args.checkpoint):
            load_checkpoint(args.checkpoint, model, device=device.type)
        else:
            print(f"Warning: Checkpoint '{args.checkpoint}' not found! Running evaluation with randomly initialized model.")

        # Multi-GPU Auto Detection & DataParallel Wrap
        model, gpu_count = wrap_model_multigpu(model, device, use_multigpu=args.multigpu)

        workers = min(args.num_workers, max(1, os.cpu_count() or 2)) if device.type == "cuda" else 0

        if args.test_split > 0.0:
            calib_ds, test_ds = create_split_evaluation_datasets(
                real_dir=args.real_dir,
                fake_dir=args.fake_dir,
                test_split=args.test_split,
                seed=args.seed,
                max_samples_per_class=args.max_samples,
                max_fake_samples=fake_limit,
                max_real_samples=real_limit
            )
            calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"), persistent_workers=(workers > 0))
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"), persistent_workers=(workers > 0))

            print("\n--- Phase 1/2: Inference on Calibration Set (70%) ---")
            calib_labels, calib_d0, calib_abs_d0, calib_delta01 = evaluate_zero_shot(model, calib_loader, device)

            print("\n--- Phase 2/2: Inference on Test Set (30%) ---")
            test_labels, test_d0, test_abs_d0, test_delta01 = evaluate_zero_shot(model, test_loader, device)
        else:
            full_dataset = EvaluationImageDataset(
                real_dir=args.real_dir,
                fake_dir=args.fake_dir,
                max_samples_per_class=args.max_samples,
                max_fake_samples=fake_limit,
                max_real_samples=real_limit
            )
            if len(full_dataset) == 0:
                print(f"ERROR: No evaluation images found in '{args.real_dir}' or '{args.fake_dir}'.")
                return

            full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"), persistent_workers=(workers > 0))
            all_labels, all_d0, all_abs_d0, all_delta01 = evaluate_zero_shot(model, full_loader, device)
            calib_labels, calib_d0, calib_abs_d0, calib_delta01 = all_labels, all_d0, all_abs_d0, all_delta01
            test_labels, test_d0, test_abs_d0, test_delta01 = all_labels, all_d0, all_abs_d0, all_delta01

    # -------------------------------------------------------------------------
    # STEP 1: CALIBRATION PHASE (Tập 70% - Tối ưu hóa ngưỡng quyết định tau*)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" === STEP 1: CALIBRATION ON 70% SPLIT (Threshold Optimization) ===")
    print("="*70)
    metrics_calib_d0 = compute_metrics(calib_d0, calib_labels)
    metrics_calib_abs_d0 = compute_metrics(calib_abs_d0, calib_labels)
    metrics_calib_delta01 = compute_metrics(calib_delta01, calib_labels)

    tau_d0 = metrics_calib_d0["best_threshold"]
    tau_abs_d0 = metrics_calib_abs_d0["best_threshold"]
    tau_delta01 = metrics_calib_delta01["best_threshold"]

    print(f"[Calib | D^(0)]     -> ROC-AUC: {metrics_calib_d0['auc']:.2f}% | Balanced Acc: {metrics_calib_d0['best_acc']:.2f}% | Optimal tau*: {tau_d0:.4f}")
    print(f"[Calib | |D^(0)|]   -> ROC-AUC: {metrics_calib_abs_d0['auc']:.2f}% | Balanced Acc: {metrics_calib_abs_d0['best_acc']:.2f}% | Optimal tau*: {tau_abs_d0:.4f}")
    print(f"[Calib | |Delta^01|] -> ROC-AUC: {metrics_calib_delta01['auc']:.2f}% | Balanced Acc: {metrics_calib_delta01['best_acc']:.2f}% | Optimal tau*: {tau_delta01:.4f}")

    # -------------------------------------------------------------------------
    # STEP 2: TEST OUT-OF-SAMPLE EVALUATION (Tập 30% - Áp dụng ngưỡng cố định tau*)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" === STEP 2: OUT-OF-SAMPLE TEST ON 30% SPLIT (Fixed Threshold tau*) ===")
    print("="*70)
    metrics_test_d0 = compute_metrics(test_d0, test_labels, fixed_threshold=tau_d0)
    metrics_test_abs_d0 = compute_metrics(test_abs_d0, test_labels, fixed_threshold=tau_abs_d0)
    metrics_test_delta01 = compute_metrics(test_delta01, test_labels, fixed_threshold=tau_delta01)

    print(f"\n[Test Result: D^(0)]")
    print(f"  Fixed tau*: {tau_d0:.4f} (from 70% Calib) | Test ROC-AUC: {metrics_test_d0['auc']:.2f}%")
    print(f"  Balanced Acc: {metrics_test_d0['balanced_acc']:.2f}% (Internal Upperbound: {metrics_test_d0['best_acc_internal']:.2f}%)")
    print(f"  Accuracy: {metrics_test_d0['accuracy']:.2f}% | Precision: {metrics_test_d0['precision']:.2f}% | Recall (TPR): {metrics_test_d0['recall']:.2f}% | FPR: {metrics_test_d0['fpr']:.2f}% | F1: {metrics_test_d0['f1']:.2f}%")
    print(f"  Confusion Matrix (TN, FP / FN, TP): {metrics_test_d0['confusion_matrix']}")

    print(f"\n[Test Result: |D^(0)|]")
    print(f"  Fixed tau*: {tau_abs_d0:.4f} (from 70% Calib) | Test ROC-AUC: {metrics_test_abs_d0['auc']:.2f}%")
    print(f"  Balanced Acc: {metrics_test_abs_d0['balanced_acc']:.2f}% (Internal Upperbound: {metrics_test_abs_d0['best_acc_internal']:.2f}%)")
    print(f"  Accuracy: {metrics_test_abs_d0['accuracy']:.2f}% | Precision: {metrics_test_abs_d0['precision']:.2f}% | Recall (TPR): {metrics_test_abs_d0['recall']:.2f}% | FPR: {metrics_test_abs_d0['fpr']:.2f}% | F1: {metrics_test_abs_d0['f1']:.2f}%")
    print(f"  Confusion Matrix (TN, FP / FN, TP): {metrics_test_abs_d0['confusion_matrix']}")

    print(f"\n[Test Result: |Delta^01|]")
    print(f"  Fixed tau*: {tau_delta01:.4f} (from 70% Calib) | Test ROC-AUC: {metrics_test_delta01['auc']:.2f}%")
    print(f"  Balanced Acc: {metrics_test_delta01['balanced_acc']:.2f}% (Internal Upperbound: {metrics_test_delta01['best_acc_internal']:.2f}%)")
    print(f"  Accuracy: {metrics_test_delta01['accuracy']:.2f}% | Precision: {metrics_test_delta01['precision']:.2f}% | Recall (TPR): {metrics_test_delta01['recall']:.2f}% | FPR: {metrics_test_delta01['fpr']:.2f}% | F1: {metrics_test_delta01['f1']:.2f}%")
    print(f"  Confusion Matrix (TN, FP / FN, TP): {metrics_test_delta01['confusion_matrix']}")
    print("="*70)

    # -------------------------------------------------------------------------
    # STEP 3: LƯU TRỮ KẾT QUẢ VÀ XUẤT HÌNH ẢNH VECTOR (PDF + PNG)
    # -------------------------------------------------------------------------
    # 1. Save split scores array (.npz)
    split_scores_path = results_dir / "detection_scores_split.npz"
    np.savez_compressed(
        split_scores_path,
        calib_labels=calib_labels,
        calib_d0=calib_d0,
        calib_abs_d0=calib_abs_d0,
        calib_delta01=calib_delta01,
        test_labels=test_labels,
        test_d0=test_d0,
        test_abs_d0=test_abs_d0,
        test_delta01=test_delta01,
        tau_d0=tau_d0,
        tau_abs_d0=tau_abs_d0,
        tau_delta01=tau_delta01
    )
    print(f"\n💾 Saved split detection scores: {split_scores_path}")

    # 2. Save Calibration Metrics JSON
    summary_calib = {
        "dataset_split": {"calib_samples": len(calib_labels), "test_samples": len(test_labels), "seed": args.seed},
        "d0": metrics_calib_d0,
        "abs_d0": metrics_calib_abs_d0,
        "delta01": metrics_calib_delta01
    }
    calib_json_path = results_dir / "detection_metrics_calib.json"
    with open(calib_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_calib, f, indent=2)
    print(f"📄 Saved calibration metrics JSON: {calib_json_path}")

    # 3. Save Test Metrics JSON
    summary_test = {
        "dataset_split": {"calib_samples": len(calib_labels), "test_samples": len(test_labels), "seed": args.seed},
        "calibrated_thresholds": {"d0": tau_d0, "abs_d0": tau_abs_d0, "delta01": tau_delta01},
        "d0": metrics_test_d0,
        "abs_d0": metrics_test_abs_d0,
        "delta01": metrics_test_delta01
    }
    test_json_path = results_dir / "detection_metrics_test.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_test, f, indent=2)
    print(f"📄 Saved out-of-sample test metrics JSON: {test_json_path}")

    # 4. Export standalone individual plots (PDF + PNG) evaluated on Test Set (with tau* from Calib)
    plots_dir = results_dir / "plots" / "standard_zed"
    test_scores_dict = {
        "d0": test_d0,
        "abs_d0": test_abs_d0,
        "delta01": test_delta01
    }
    export_all_detection_plots(
        scores_dict=test_scores_dict,
        labels=test_labels,
        metrics_dict=summary_test,
        output_dir=str(plots_dir),
        model_name="Standard ZED (Test 30%)",
        formats=("pdf", "png")
    )

    # 5. Save legacy combined 4-panel dashboard on Test Set
    dashboard_path = str(results_dir / "detection_dashboard_test.png")
    plot_detection_dashboard(
        d0_scores=test_d0,
        abs_d0_scores=test_abs_d0,
        delta01_scores=test_delta01,
        labels=test_labels,
        metrics_d0=metrics_test_d0,
        output_path=dashboard_path,
        model_name="Standard ZED (Test 30%)"
    )
    print(f"📊 Saved Test Evaluation Dashboard plot: {dashboard_path}")
    print(f"\n✨ All done! Split evaluation output saved in: {results_dir.resolve()}")

if __name__ == "__main__":
    main()
