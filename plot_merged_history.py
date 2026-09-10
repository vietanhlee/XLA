"""
Script: plot_merged_history.py
Mục đích: Tự động quét đệ quy tất cả các file JSON (training_history.json)
từ các session/run tải từ Kaggle về, gộp lại theo thứ tự Epoch (tự khử trùng lặp nếu có resume),
lưu file JSON tổng hợp và vẽ biểu đồ Loss chuẩn publication (PNG + PDF).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


def find_and_load_history_files(root_dir: str) -> List[Path]:
    """Tìm tất cả các file JSON chứa training history trong folder và các folder con."""
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {root_dir}")

    candidates = list(root_path.rglob("*.json"))
    valid_files = []

    for file_path in candidates:
        if file_path.name.startswith("merged_"):
            continue  # Bỏ qua file gộp đã tạo trước đó
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Kiểm tra xem có cấu trúc của training_history hay không
            if isinstance(data, dict) and "epoch" in data and "train_loss" in data:
                if len(data["epoch"]) > 0:
                    valid_files.append(file_path)
        except Exception:
            continue

    # Sắp xếp theo đường dẫn hoặc số thứ tự folder (1, 2, 3...)
    def sort_key(p: Path):
        for part in p.parts:
            if part.isdigit():
                return int(part)
        return str(p)

    valid_files.sort(key=sort_key)
    return valid_files


def merge_histories(json_files: List[Path]) -> Dict[str, List[Any]]:
    """
    Gộp dữ liệu từ nhiều file JSON theo thứ tự epoch.
    Nếu có trùng lặp epoch (do resume từ checkpoint giữa chừng), ưu tiên lấy từ run sau.
    """
    epoch_map: Dict[int, Dict[str, Any]] = {}
    all_keys = set()

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        epochs = data.get("epoch", [])
        num_items = len(epochs)
        keys = [k for k in data.keys() if isinstance(data[k], list) and len(data[k]) == num_items]
        all_keys.update(keys)

        for i, ep in enumerate(epochs):
            if ep not in epoch_map:
                epoch_map[ep] = {}
            for k in keys:
                epoch_map[ep][k] = data[k][i]

    # Sắp xếp theo số epoch tăng dần
    sorted_epochs = sorted(epoch_map.keys())

    merged_data: Dict[str, List[Any]] = {k: [] for k in all_keys}
    merged_data["epoch"] = sorted_epochs

    for ep in sorted_epochs:
        row = epoch_map[ep]
        for k in all_keys:
            if k == "epoch":
                continue
            merged_data[k].append(row.get(k, None))

    return merged_data


def plot_comprehensive_curves(
    history: Dict[str, List[Any]],
    output_png: str,
    output_pdf: Optional[str] = None,
    title: str = "Standard ZED Model - Training Loss & Convergence Dashboard"
):
    """Vẽ dashboard biểu đồ loss gồm 3 subplot chuẩn publication."""
    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    l0_loss = history.get("level_0_loss", [])
    l1_loss = history.get("level_1_loss", [])
    l2_loss = history.get("level_2_loss", [])
    epoch_times = history.get("epoch_time", [])

    if not epochs or not train_loss:
        print("⚠️ Không có dữ liệu để vẽ biểu đồ!")
        return

    # Thiết lập giao diện matplotlib hiện đại, tinh gọn
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 14
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.suptitle(title, fontweight="bold", y=1.02)

    # -------------------------------------------------------------
    # Subplot 1: Total NLL Loss (Train vs Val)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, label="Train NLL Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4.5, alpha=0.95)

    if val_loss and any(v is not None for v in val_loss):
        valid_val = [(ep, v) for ep, v in zip(epochs, val_loss) if v is not None]
        v_eps, v_vals = zip(*valid_val)
        ax1.plot(v_eps, v_vals, label="Validation NLL Loss", color="#ff7f0e", linewidth=2.2, linestyle="--", marker="s", markersize=4.5, alpha=0.95)

        min_val_idx = int(np.argmin(v_vals))
        best_epoch = v_eps[min_val_idx]
        best_val = v_vals[min_val_idx]

        ax1.scatter([best_epoch], [best_val], color="#d62728", s=90, zorder=6, label=f"Best Val: {best_val:.4f} (Ep {best_epoch})")
        ax1.axvline(x=best_epoch, color="#d62728", linestyle=":", alpha=0.6, linewidth=1.5)
        ax1.annotate(
            f"Best Val: {best_val:.4f}\n(Epoch {best_epoch})",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 0.8, best_val + 2.5),
            arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1, headwidth=6),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="#d62728", alpha=0.85)
        )

    ax1.set_title("Total Negative Log-Likelihood (NLL) Loss", fontweight="bold", pad=8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (bits / pixel)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#d0d0d0")
    if len(epochs) <= 30:
        ax1.set_xticks(epochs)

    # -------------------------------------------------------------
    # Subplot 2: Multi-Scale Breakdown (Level 0, 1, 2)
    # -------------------------------------------------------------
    ax2 = axes[1]
    has_levels = False
    if l0_loss and any(v is not None for v in l0_loss):
        ax2.plot(epochs, l0_loss, label="Level 0 (256x256 - Fine Details)", color="#2ca02c", linewidth=2.0, marker="o", markersize=4)
        has_levels = True
    if l1_loss and any(v is not None for v in l1_loss):
        ax2.plot(epochs, l1_loss, label="Level 1 (128x128 - Mid Texture)", color="#9467bd", linewidth=2.0, marker="s", markersize=4)
        has_levels = True
    if l2_loss and any(v is not None for v in l2_loss):
        ax2.plot(epochs, l2_loss, label="Level 2 (64x64 - Coarse Context)", color="#8c564b", linewidth=2.0, marker="^", markersize=4)
        has_levels = True

    if has_levels:
        ax2.set_title("Multi-Scale Pyramid NLL Loss Breakdown", fontweight="bold", pad=8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss per Level (bits / pixel)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#d0d0d0")
        if len(epochs) <= 30:
            ax2.set_xticks(epochs)
    else:
        ax2.text(0.5, 0.5, "Không có dữ liệu chi tiết từng Level", ha="center", va="center")

    plt.tight_layout()

    # Lưu ảnh PNG độ nét cao
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    print(f"✅ Đã lưu biểu đồ PNG chất lượng cao tại: {output_png}")

    # Lưu ảnh PDF vector cho báo cáo/bài báo nếu có
    if output_pdf:
        plt.savefig(output_pdf, bbox_inches="tight")
        print(f"✅ Đã lưu biểu đồ vector PDF tại: {output_pdf}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Gộp và vẽ biểu đồ training history từ các folder Kaggle.")
    parser.add_argument(
        "--dir",
        type=str,
        default=r"C:\Users\levie\Downloads\zed_cnn",
        help="Đường dẫn đến thư mục chứa các thư mục con kết quả từ Kaggle."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="zed_cnn_output",
        help="Đường dẫn thư mục lưu file kết quả (mặc định lưu tại zed_cnn_output)."
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Standard ZED Model - Multi-Scale Density Training Convergence",
        help="Tiêu đề biểu đồ."
    )
    args = parser.parse_args()

    root_dir = args.dir
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"🔍 Bắt đầu quét thư mục: {root_dir}")
    print(f"📁 Thư mục lưu kết quả: {out_dir}")

    json_files = find_and_load_history_files(root_dir)
    if not json_files:
        print(f"❌ Không tìm thấy file JSON training history nào trong {root_dir}")
        sys.exit(1)

    print(f"📁 Tìm thấy {len(json_files)} file lịch sử huấn luyện:")
    for i, p in enumerate(json_files, 1):
        print(f"   [{i}] {p}")

    merged = merge_histories(json_files)
    epochs = merged.get("epoch", [])
    print(f"\n📊 Đã gộp thành công tổng cộng {len(epochs)} Epochs: {epochs}")

    # Lưu file JSON gộp tại thư mục hiện tại
    out_merged_json = os.path.join(out_dir, "merged_training_history.json")
    with open(out_merged_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"💾 Đã lưu JSON tổng hợp tại: {out_merged_json}")

    # Đường dẫn ảnh xuất ra tại thư mục hiện tại
    out_png = os.path.join(out_dir, "training_curves_merged.png")
    out_pdf = os.path.join(out_dir, "training_curves_merged.pdf")

    # Vẽ biểu đồ
    plot_comprehensive_curves(merged, out_png, out_pdf, title=args.title)

    # In bảng tóm tắt kết quả
    train_loss = merged.get("train_loss", [])
    val_loss = merged.get("val_loss", [])
    epoch_times = merged.get("epoch_time", [])

    print("\n" + "=" * 60)
    print("📈 TỔNG KẾT QUÁ TRÌNH HUẤN LUYỆN (CONVERGENCE SUMMARY)")
    print("=" * 60)
    print(f"• Tổng số Epoch đã huấn luyện: {len(epochs)} (từ Epoch {epochs[0]} đến {epochs[-1]})")
    if train_loss:
        print(f"• Initial Train Loss (Epoch {epochs[0]}): {train_loss[0]:.4f} -> Final: {train_loss[-1]:.4f}")
    if val_loss and any(v is not None for v in val_loss):
        valid_val = [(ep, v) for ep, v in zip(epochs, val_loss) if v is not None]
        best_ep, best_val = min(valid_val, key=lambda x: x[1])
        print(f"• Best Validation Loss : {best_val:.4f} (đạt được tại Epoch {best_ep})")
        print(f"• Initial Val Loss     : {valid_val[0][1]:.4f} -> Final: {valid_val[-1][1]:.4f}")
    if epoch_times and any(t is not None for t in epoch_times):
        valid_times = [t for t in epoch_times if t is not None]
        total_hours = sum(valid_times) / 3600.0
        avg_mins = (sum(valid_times) / len(valid_times)) / 60.0
        print(f"• Tổng thời gian train : {total_hours:.2f} giờ (~{avg_mins:.1f} phút / epoch)")
    print("=" * 60)


if __name__ == "__main__":
    main()
