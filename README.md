# ZED & Advanced ZED: Zero-Shot Detection of AI-Generated Images

Mã nguồn triển khai hoàn chỉnh mô hình **ZED tiêu chuẩn** và **Advanced ZED nâng cấp** (tích hợp Wavelet DWT, Spatial Self-Attention và Robust Augmentation Pipeline).

---

## 🌟 Các Nâng Cấp Vượt Trội trong `AdvancedZEDModel`

1. **Phân tích Miền Tần số 2D Haar Wavelet (`zed/models/wavelet.py`):**
   - Trích xuất dải tần số cao (LH, HL, HH) nhằm phát hiện các nhiễu vi mô (spectral artifacts) đặc trưng của GAN & Latent Diffusion (SDXL, Midjourney).
2. ** Spatial Self-Attention Block (`zed/models/attention.py`):**
   - Mô hình hóa mối quan hệ phụ thuộc không gian diện rộng (long-range dependencies) giữa các vùng ảnh xa nhau.
3. **Pipeline Augmentation Chống Báo Động Giả (`zed/augmentations.py`):**
   - Tích hợp nén JPEG ngẫu nhiên (Quality 50-95), co giãn tỷ lệ và nhiễu cảm biến trong quá trình nạp ảnh thật, giúp bộ nén học được tính chất của ảnh internet thực tế.

---

## 📁 Cấu trúc Thư mục Code

```
g:/XLA/
├── config.py                     # Cấu hình siêu tham số (Model, Train, Detect)
├── requirements.txt              # Danh sách phụ thuộc (PyTorch, torchvision, v.v.)
├── README.md                     # Hướng dẫn chi tiết
├── train_advanced.py             # Script huấn luyện mô hình Advanced ZED (Khuyên dùng)
├── detect_advanced.py            # Script suy luận Zero-Shot cho Advanced ZED
├── zed/
│   ├── augmentations.py          # Robust Data Augmentation Pipeline
│   ├── dataset.py                # DataLoader cho ảnh thật & tập test
│   ├── utils.py                  # Tiện ích tính ROC-AUC, Accuracy, Threshold
│   ├── train.py                  # Script huấn luyện mô hình ZED tiêu chuẩn
│   ├── detect.py                 # Script suy luận mô hình ZED tiêu chuẩn
│   └── models/
│       ├── logistic_mixture.py   # Discretized Logistic Mixture (NLL & Entropy)
│       ├── cnn_encoder.py        # SReC CNN tiêu chuẩn
│       ├── zed_model.py          # Mô hình ZED tiêu chuẩn
│       ├── wavelet.py            # 2D Haar Wavelet Decomposition (DWT)
│       ├── attention.py          # Spatial Self-Attention Block
│       ├── advanced_cnn_encoder.py # Advanced Encoder (Spatial + DWT + Attention)
│       └── advanced_zed_model.py # Mô hình Advanced ZED đa độ phân giải
```

---

## 🚀 Hướng dẫn Chạy trên Command Prompt (CMD)

*(Lưu ý: Bạn hãy tự thực thi các lệnh dưới đây trong cửa sổ CMD/Terminal)*

### 1. Huấn luyện Mô hình Advanced ZED (Nâng cấp - Khuyên dùng)
```bash
python train_advanced.py --data_dir "C:\Users\levie\Downloads\img\images" --epochs 50 --batch_size 16 --lr 0.0001
```
Checkpoint tốt nhất sẽ được tự động lưu tại `checkpoints/zed_advanced_best.pth`.

### 2. Suy luận & Đánh giá Zero-Shot với Advanced ZED
```bash
python detect_advanced.py --checkpoint checkpoints/zed_advanced_best.pth --real_dir data/test/real --fake_dir data/test/fake
```

### 3. (Tùy chọn) Huấn luyện Mô hình ZED Tiêu chuẩn (Standard ZED)
```bash
python zed/train.py --data_dir "C:\Users\levie\Downloads\img\images" --epochs 50 --batch_size 16 --lr 0.0001
```
