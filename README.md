# ZED & Advanced ZED: Zero-Shot Detection of AI-Generated Images

Mã nguồn triển khai hoàn chỉnh mô hình **ZED tiêu chuẩn** (Standard ZED) và **Advanced ZED nâng cấp** (tích hợp 2D Haar Wavelet, Spatial Self-Attention, Robust Augmentation, quét đệ quy thư mục ảnh, phân chia Train/Val split tự động và Early Stopping).

---

## 🌟 Các Nâng Cấp Vượt Trội trong Pipeline & Kiến trúc

1. **Quét Đệ quy & Phân chia Train/Val (`zed/dataset.py`):**
   - Tự động nhận vào một thư mục gốc bất kỳ, quét đệ quy qua **tất cả các thư mục con (subfolders)** để lấy ra toàn bộ ảnh (`.jpg`, `.png`, `.webp`, `.bmp`, `.tif`, v.v.).
   - Phân chia ngẫu nhiên (reproducible seed=42) thành tập **Train** và tập **Validation** (mặc định `--val_split 0.15`).
2. **Cơ chế Early Stopping (`zed/utils.py`):**
   - Theo dõi sát sao loss NLL trên tập Validation (`val_loss`).
   - Nếu `val_loss` không giảm sau số epoch kiên nhẫn (`--patience 7`), tiến trình sẽ ngắt sớm để chống overfit và lưu lại checkpoint tối ưu nhất (`zed_best.pth` / `zed_advanced_best.pth`).
3. **Phân tích Miền Tần số 2D Haar Wavelet (`zed/models/wavelet.py`):**
   - Trích xuất dải tần số cao (LH, HL, HH) nhằm phát hiện các nhiễu vi mô (spectral artifacts) đặc trưng của GAN & Latent Diffusion (SDXL, Midjourney).
4. **Spatial Self-Attention Block (`zed/models/attention.py`):**
   - Mô hình hóa mối quan hệ phụ thuộc không gian diện rộng (long-range dependencies) giữa các vùng ảnh xa nhau.
5. **Pipeline Augmentation Chống Báo Động Giả (`zed/augmentations.py`):**
   - Tích hợp nén JPEG ngẫu nhiên (Quality 50-95), co giãn tỷ lệ và nhiễu cảm biến trong quá trình nạp ảnh thật, giúp bộ nén học được tính chất của ảnh internet thực tế.

---

## 📁 Cấu trúc Thư mục Code

```
g:/XLA/
├── config.py                     # Cấu hình siêu tham số (Model, Train, Detect)
├── requirements.txt              # Danh sách phụ thuộc (PyTorch, torchvision, v.v.)
├── README.md                     # Hướng dẫn chi tiết
├── train.py                      # Script huấn luyện ZED Nguyên bản Gốc (Standard ZED)
├── train_advanced.py             # Script huấn luyện Advanced ZED (Mô hình Nâng cấp)
├── detect_advanced.py            # Script suy luận Zero-Shot cho Advanced ZED
├── zed/
│   ├── augmentations.py          # Robust Data Augmentation Pipeline
│   ├── dataset.py                # DataLoader đệ quy subfolders & train/val split
│   ├── utils.py                  # EarlyStopping, Checkpoint, ROC-AUC metrics
│   ├── train.py                  # Script huấn luyện ZED nguyên bản gốc (Standard ZED)
│   ├── detect.py                 # Script suy luận mô hình ZED tiêu chuẩn
│   └── models/
│       ├── logistic_mixture.py   # Discretized Logistic Mixture (NLL & Entropy)
│       ├── cnn_encoder.py        # SReC CNN tiêu chuẩn
│       ├── zed_model.py          # Mô hình ZED tiêu chuẩn
│       ├── wavelet.py            # 2D Haar Wavelet Decomposition (DWT)
      ├── attention.py          # Spatial Self-Attention Block
│       ├── advanced_cnn_encoder.py # Advanced Encoder (Spatial + DWT + Attention)
│       └── advanced_zed_model.py # Mô hình Advanced ZED đa độ phân giải
```

---

## 🚀 Hướng dẫn Chạy Lệnh trên Command Prompt (CMD / Terminal)

*(Lưu ý: Bạn hãy tự chạy các lệnh CMD này trên terminal của bạn theo Rule 2 nhé!)*

### 1. Huấn luyện Mô hình ZED Nguyên bản Gốc (Standard ZED)
```cmd
python train.py --data_dir "C:\Users\levie\Downloads\img\images" --val_split 0.15 --epochs 50 --batch_size 16 --lr 0.0001 --patience 7
```
- Thuật toán sẽ quét đệ quy toàn bộ thư mục `images` và mọi thư mục con bên trong.
- Tự động chia 85% Train, 15% Validation.
- Bật **Early Stopping**: Tự động dừng huấn luyện khi validation loss không còn cải thiện sau 7 epoch liên tiếp và lưu checkpoint tốt nhất tại `checkpoints/zed_best.pth`.

---

### 2. Huấn luyện Mô hình Advanced ZED (Nâng cấp)
```cmd
python train_advanced.py --data_dir "C:\Users\levie\Downloads\img\images" --val_split 0.15 --epochs 50 --batch_size 16 --lr 0.0001 --patience 7
```
- Tích hợp thêm **2D Haar Wavelet**, **Spatial Self-Attention** và **Robust Data Augmentations**.
- Tự động lưu checkpoint tốt nhất tại `checkpoints/zed_advanced_best.pth`.

---

### 3. Tiếp tục Huấn luyện từ Checkpoint dở dang (--resume)

Nếu quá trình huấn luyện bị ngắt giữa chừng, bạn có thể truyền thêm tham số `--resume` để tiếp tục train từ epoch cũ (nạp lại đầy đủ weights, optimizer state và scheduler state):

```cmd
# Tiếp tục train Standard ZED từ checkpoint dở dang zed_last.pth
python train.py --data_dir "C:\Users\levie\Downloads\img\images" --epochs 100 --resume checkpoints/zed_last.pth

# Tiếp tục train Advanced ZED từ checkpoint dở dang zed_advanced_last.pth
python train_advanced.py --data_dir "C:\Users\levie\Downloads\img\images" --epochs 100 --resume checkpoints/zed_advanced_last.pth
```

---

### 4. Suy luận Zero-Shot & Đánh giá (Detection)

#### Cho Mô hình Standard ZED:
```cmd
python zed/detect.py --checkpoint checkpoints/zed_best.pth --real_dir data/test/real --fake_dir data/test/fake
```

#### Cho Mô hình Advanced ZED:
```cmd
python detect_advanced.py --checkpoint checkpoints/zed_advanced_best.pth --real_dir data/test/real --fake_dir data/test/fake
```
