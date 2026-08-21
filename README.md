# ZED: Zero-Shot Detection of AI-Generated Images (PyTorch Production Code)

Dự án này mã hóa hoàn chỉnh kiến trúc và luồng huấn luyện cho bài báo khoa học **"Zero-Shot Detection of AI-Generated Images" (ZED)** (Cozzolino et al.).

---

## 🌟 Tổng quan về Phương pháp (Methodology)

1. **Nguyên lý Zero-Shot:**
   * Không sử dụng bất kỳ ảnh AI/Fake nào trong quá trình huấn luyện.
   * Chỉ học phân bố mật độ xác suất của **ảnh thật (Real Images)** bằng bộ nén không mất thông tin SReC (Super-resolution based Lossless Compressor).

2. **Cơ chế phát hiện (Detection Mechanism):**
   * **NLL (Negative Log-Likelihood):** Chi phí mã hóa thực tế để nén một bức ảnh.
   * **Entropy ($H$):** Độ hỗn loạn kỳ vọng dự đoán từ mô hình ảnh thật.
   * **Chênh lệch chi phí mã hóa ($D^{(l)} = \text{NLL}^{(l)} - H^{(l)}$):**
     * Ảnh thật: $NLL \approx H \Rightarrow D \approx 0$
     * Ảnh do AI tạo ra (DALL-E 2/3, Midjourney, SDXL, GANs...): Mismatch phân bố $\Rightarrow D^{(0)} > 0$ hoặc xuất hiện dị biệt rõ rệt.

---

## 📁 Cấu trúc Thư mục Code

```
g:/XLA/
├── config.py                     # Cấu hình siêu tham số (Model, Train, Detect)
├── requirements.txt              # Danh sách phụ thuộc (PyTorch, torchvision, v.v.)
├── README.md                     # Hướng dẫn sử dụng chi tiết
├── zed/
│   ├── __init__.py
│   ├── dataset.py                # DataLoader cho ảnh thật & tập test
│   ├── utils.py                  # Hàm tính ROC-AUC, Accuracy, Threshold, Checkpoint
│   ├── train.py                  # Script huấn luyện mật độ ảnh thật (Real-only)
│   ├── detect.py                 # Script suy luận Zero-Shot (Real vs AI)
│   └── models/
│       ├── __init__.py
│       ├── logistic_mixture.py   # Hỗn hợp Logistic rời rạc (NLL & Entropy toán học)
│       ├── cnn_encoder.py        # Mạng SReC CNN dự đoán tham số phân bố
│       └── zed_model.py          # Mô hình ZED đa độ phân giải (Level 0, 1, 2, 3)
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy trên Command Prompt (CMD)

*(Lưu ý: Bạn hãy tự thực thi các lệnh dưới đây trong cửa sổ CMD/Terminal trên máy của bạn)*

### 1. Cài đặt môi trường & thư viện
```bash
# Di chuyển tới thư mục dự án
cd /d g:\XLA

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
Tạo cấu trúc thư mục dữ liệu như sau:
```
g:/XLA/data/
├── real_images/         # Thư mục chứa ảnh THẬT để HUẤN LUYỆN (JPG, PNG)
└── test/
    ├── real/            # Thư mục chứa ảnh THẬT để KIỂM THỬ
    └── fake/            # Thư mục chứa ảnh AI/FAKE để KIỂM THỬ
```

### 3. Chạy Huấn luyện (Training)
Mô hình sẽ **chỉ học trên tập ảnh thật** (`data/real_images`):
```bash
python zed/train.py --data_dir data/real_images --epochs 50 --batch_size 16 --lr 0.0001
```
Checkpoint tốt nhất sẽ tự động được lưu tại `checkpoints/zed_best.pth`.

### 4. Chạy Suy luận Phát hiện Ảnh AI (Zero-Shot Detection)
Đánh giá độ chính xác ROC-AUC và tìm ngưỡng phân loại tối ưu:
```bash
python zed/detect.py --checkpoint checkpoints/zed_best.pth --real_dir data/test/real --fake_dir data/test/fake
```

---

## 🔬 Điểm nổi bật về mặt Mã Nguồn (Code Quality)
- **Chuẩn toán học:** Tính NLL & Entropy chính xác theo phân bố `Discretized Logistic Mixture` ($K=10$), xử lý mượt mà biên 8-bit $[0, 255]$.
- **Đa độ phân giải (Multi-resolution Pyramid):** Tự động xử lý downsampling 2x2 pooling và tính toán $D^{(0)}, D^{(1)}, D^{(2)}, \Delta^{01}$.
- **Hiệu năng cao:** Hỗ trợ Mixed Precision (`torch.cuda.amp`), Cosine Annealing Scheduler và Gradient Clipping.
