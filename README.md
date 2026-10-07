# ZED, Advanced ZED & Modern ZED: Zero-Shot Detection of AI-Generated Images

Mã nguồn triển khai hoàn chỉnh các thế hệ mô hình phát hiện ảnh AI Zero-Shot:
1. **Standard ZED**: Mô hình nguyên bản theo bài báo Cozzolino et al. (CNN ResBlocks).
2. **Advanced ZED**: Mô hình mở rộng tích hợp 2D Haar Wavelet DWT và Spatial Self-Attention.
3. **Modern ZED (Khuyên dùng)**: Mô hình tối ưu hóa hiện đại tích hợp **ConvNeXt-SReC Encoder (7x7 Depthwise Conv + Inverted Bottleneck + ECA Channel Attention)**, **Density-Preserving Transform (Native Crop + D4 Symmetry)** và thống kê phát hiện **Top-10% Patch Anomaly**.

---

## 🔬 Vì Sao ViT / Spatial Self-Attention Kém Hiệu Quả Trong ZED?

1. **Sai lệch Inductive Bias**: Bài toán SReC là ước lượng mật độ xác suất có điều kiện vi mô ($3 \times 3$ đến $7 \times 7$) nhằm đếm chính xác bit entropy trên từng điểm ảnh 8-bit. Ngữ cảnh toàn cục đã được cung cấp sẵn ở tầng pyramid thấp hơn ($y^{(l+1)}$). ViT loại bỏ tính cảm thụ cục bộ nên làm loãng gradient khi ước lượng phân phối sub-pixel.
2. **Attention là bộ lọc thông thấp (Low-pass Filter)**: Softmax Attention làm nhẵn đặc trưng, triệt tiêu các vết nhiễu bất thường ở dải tần số cao vốn là dấu vết đặc trưng của ảnh AI.
3. **Cạm bẫy Data Augmentation**: Các phép biến đổi mạnh (Bilinear Resize, Gaussian Noise) làm sai lệch hàm mật độ phân phối thật $P_{\text{real}}(x)$ của cảm biến camera, phá hỏng khả năng phân biệt Zero-Shot.
4. **Giải pháp Modern ZED**: Giữ vững inductive bias cục bộ bằng **ConvNeXt 7x7 Depthwise Conv**, sử dụng **ECA (Efficient Channel Attention)** chú ý trên kênh màu/bộ lọc thay vì làm loạn tọa độ không gian, và áp dụng **D4 Dihedral Symmetry** (quay $90^\circ$, lật gương - bảo toàn 100% giá trị rời rạc 8-bit).

---

## 📁 Cấu trúc Thư mục Code

```
g:/XLA/
├── config.py                     # Cấu hình siêu tham số (Model, Train, Detect)
├── requirements.txt              # Danh sách phụ thuộc (PyTorch, torchvision, v.v.)
├── README.md                     # Hướng dẫn chi tiết
├── train_modern.py               # [MỚI] Script huấn luyện Modern ZED (ConvNeXt-SReC + D4 Transform)
├── detect_modern.py              # [MỚI] Script suy luận Zero-Shot cho Modern ZED (Top-10% Patch Anomaly)
├── train.py                      # Script huấn luyện ZED Nguyên bản Gốc (Standard ZED)
├── train_advanced.py             # Script huấn luyện Advanced ZED
├── detect_advanced.py            # Script suy luận Zero-Shot cho Advanced ZED
├── zed/
│   ├── augmentations.py          # Pipeline D4 Symmetry & Native Crop bảo toàn mật độ
│   ├── dataset.py                # DataLoader đệ quy subfolders & train/val split
│   ├── utils.py                  # EarlyStopping, Checkpoint, ROC-AUC, Top-10% metrics
│   ├── trainer.py                # Reusable Training Loop (AMP, Gradient Clip, Resume)
│   ├── detect.py                 # Script suy luận mô hình ZED tiêu chuẩn
│   └── models/
│       ├── logistic_mixture.py   # Discretized Logistic Mixture (NLL & exact Entropy)
│       ├── cnn_encoder.py        # SReC CNN tiêu chuẩn
│       ├── zed_model.py          # Mô hình ZED tiêu chuẩn
│       ├── convnext_encoder.py   # [MỚI] ConvNeXt-SReC Encoder (7x7 DW Conv + ECA)
│       ├── modern_zed_model.py   # [MỚI] Mô hình Modern ZED đa độ phân giải
│       ├── wavelet.py            # 2D Haar Wavelet Decomposition (DWT)
│       ├── attention.py          # Spatial Self-Attention Block
│       ├── advanced_cnn_encoder.py # Advanced Encoder (Spatial + DWT + Attention)
│       └── advanced_zed_model.py # Mô hình Advanced ZED
```

---

## 🚀 Hướng dẫn Chạy Lệnh trên Command Prompt (CMD / Terminal)

*(Lưu ý: Bạn hãy tự chạy các lệnh CMD này trên terminal của bạn theo Rule 2 nhé!)*

### 1. Huấn luyện Mô hình Khuyên Dùng: Modern ZED (ConvNeXt-SReC + D4)
```cmd
python train_modern.py --data_dir "C:\Users\levie\Downloads\img\images" --val_split 0.15 --epochs 50 --batch_size 16 --lr 0.0001 --patience 7
```
- Sử dụng **ConvNeXt-SReC** và **DensityPreservingTransform** (Native Crop + D4 Symmetry).
- Tự động lưu checkpoint tốt nhất tại `checkpoints/zed_modern_best.pth`.

---

### 2. Suy luận & Đánh giá Zero-Shot với Modern ZED
```cmd
python detect_modern.py --checkpoint checkpoints/zed_modern_best.pth --real_dir data/test/real --fake_dir data/test/fake
```
- Đánh giá toàn diện 4 chỉ số thống kê:
  - $D^{(0)}$ Mean Coding Cost
  - $D^{(0)}_{\text{top10}}$ Top-10% Patch Anomaly (cực nhạy với ảnh AI bị lỗi chi tiết cục bộ)
  - $|D^{(0)}|$ Magnitude
  - $|\Delta^{01}|$ Multi-Scale Residual
- Xuất dashboard 4 panel tại `results/detection_dashboard_modern_zed.png` và JSON số liệu tại `results/detection_metrics_modern_zed.json`.

---

### 3. Huấn luyện Mô hình ZED Tiêu Chuẩn (Standard Baseline)
```cmd
python train.py --data_dir "C:\Users\levie\Downloads\img\images" --val_split 0.15 --epochs 50 --batch_size 16 --lr 0.0001 --patience 7
```
- Tự động lưu checkpoint tốt nhất tại `checkpoints/zed_best.pth`.

---

### 4. Tiếp tục Huấn luyện từ Checkpoint dở dang (--resume)

```cmd
# Tiếp tục train Modern ZED từ checkpoint dở dang zed_modern_last.pth
python train_modern.py --data_dir "C:\Users\levie\Downloads\img\images" --epochs 100 --resume checkpoints/zed_modern_last.pth

# Tiếp tục train Standard ZED từ checkpoint dở dang zed_last.pth
python train.py --data_dir "C:\Users\levie\Downloads\img\images" --epochs 100 --resume checkpoints/zed_last.pth
```
