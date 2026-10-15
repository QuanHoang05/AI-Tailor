# 🧵 AI-TAILOR: HỆ THỐNG TRÍCH XUẤT SỐ ĐO MAY MẶC 3D TỰ ĐỘNG (ISO 8559)

> **AI-Tailor** là giải pháp thị giác máy tính 3D (3D Computer Vision & Computational Geometry) giúp tự động trích xuất đầy đủ **8 thông số nhân trắc học may mặc** theo tiêu chuẩn quốc tế **ISO 8559-1** từ một bức ảnh chụp toàn thân:
> 
> **Chiều cao • Vòng ngực • Vòng eo • Vòng hông/mông • Chiều rộng vai • Chiều dài tay áo • Chiều dài chân trong (Inseam) • Cân nặng ước tính**
> 
> Đồng thời hệ thống tự động xuất **File 3D Avatar (.obj)** để nhập trực tiếp vào các phần mềm thiết kế thời trang 3D (**CLO3D, Marvelous Designer, Browzwear, Blender**) và tự động tư vấn kích cỡ trang phục (**Size S/M/L/XL, Quần âu theo Inch, Độ cử động Ease Allowance**).

---

## ⚡ ĐIỂM ĐẶC BIỆT CỦA PHIÊN BẢN NÀY (PURE PYTORCH 2.X)
* **100% PyTorch 2.x Thuần:** Đã loại bỏ hoàn toàn các phụ thuộc biên dịch C++/CUDA phức tạp (`mesh_mesh_intersect_cuda`, `mmcv.ops.roi_align`).
* **Độc lập nền tảng:** Chạy mượt mà trên **Google Colab**, **Máy tính cá nhân Windows/Linux**, hoặc **Máy chủ thuê đám mây (Cloud Server / Docker)** mà không lo xung đột thư viện.
* **Tự động chuyển đổi CPU/GPU:** Tự nhận diện GPU NVIDIA CUDA; nếu máy không có card rời, hệ thống tự động chạy ở chế độ CPU mà không bị sập.

---

## 🎯 HƯỚNG DẪN CHẠY TRÊN CÁC MÔI TRƯỜNG

---

### 🟢 TRƯỜNG HỢP 1: TẢI NGUYÊN THƯ MỤC TỪ GOOGLE DRIVE (CHẠY LOCAL MÁY BẠN BÈ)

> **Dành cho:** Bạn gửi cả thư mục dự án lên Google Drive rồi gửi link cho bạn bè/đối tác tải về chạy trực tiếp trên máy của họ (Windows / Linux) mà không cần cấu hình rườm rà.

Vì thư mục trên Google Drive đã chứa **đầy đủ cả mã nguồn lẫn 5 file trọng số mô hình (~800MB)**, người nhận chỉ cần làm theo 3 bước sau:

#### Bước 1: Giải nén thư mục và mở Terminal
Mở ứng dụng Terminal / PowerShell hoặc Command Prompt tại thư mục vừa tải về:
```powershell
cd duong_dan_toi_thu_muc/Focused-Clean-Colab
```

#### Bước 2: Cài đặt môi trường tự động (1 chạm)
* **Trên Windows:** Chỉ cần **nhấp đúp chuột vào file `setup_windows.bat`**.  
  *(Script sẽ tự động cài PyTorch, NumPy 1.23.5 và các thư viện cần thiết).*
* **Hoặc dùng Conda (Windows/Linux/Mac):**
  ```bash
  conda env create -f environment.yml
  conda activate ai-tailor
  ```

#### Bước 3: Chạy đo đạc ngay lập tức
```powershell
# Chạy đo từ 1 ảnh (ví dụ cao 163 cm, nặng 64 kg):
python run_measurement.py --image duong_dan_anh.jpg --height_cm 163 --weight_kg 64 --save_mesh avatar.obj
```

#### Bước 4 (Tùy chọn): Chạy REST API Server để tích hợp Web / App
```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Mở trình duyệt truy cập `http://localhost:8000/docs` để sử dụng giao diện Swagger UI thử nghiệm API.

---

### 🔵 TRƯỜNG HỢP 2: CLONE CODE TỪ GITHUB HOẶC THUÊ MÁY CHỦ ĐÁM MÂY (SERVER / DOCKER)

> **Dành cho:** Lập trình viên clone mã nguồn từ GitHub: `https://github.com/QuanHoang05/AI-Tailor.git` hoặc triển khai lên máy chủ thuê (RunPod, Vast.ai, AWS EC2, VPS Ubuntu 22.04).

⚠️ **LƯU Ý QUAN TRỌNG VỀ TRỌNG SỐ MÔ HÌNH (> 100MB):**  
Vì chính sách của GitHub không cho phép đẩy file lớn hơn 100MB, khi clone code về máy mới, bạn cần đặt các file mô hình vào đúng các thư mục như bảng hướng dẫn bên dưới.

#### Bước 1: Clone mã nguồn từ GitHub
```bash
git clone https://github.com/QuanHoang05/AI-Tailor.git
cd AI-Tailor
```

#### Bước 2: Tải và đặt các file mô hình vào đúng thư mục
Bạn hãy tải hoặc copy 5 file trọng số mô hình và đặt chính xác vào các thư mục sau:

| Tên File Trọng Số | Kích thước | Đặt chính xác vào đường dẫn thư mục | Nguồn tải về |
| :--- | :---: | :--- | :--- |
| **`smpler_x_s32.pth.tar`** | 384 MB | `pretrained_models/smpler_x_s32.pth.tar` | [Tải từ HuggingFace](https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_s32.pth.tar) hoặc Google Drive |
| **`vitpose_small.pth`** | 97 MB | `pretrained_models/vitpose_small.pth` | [Tải từ HuggingFace](https://huggingface.co/caizhongang/SMPLer-X/resolve/main/vitpose_small.pth) hoặc Google Drive |
| **`SMPLX_NEUTRAL.npz`** | 108 MB | `common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz` | Đăng ký miễn phí tại [SMPL-X Portal](https://smpl-x.is.tue.mpg.de) hoặc copy từ Drive |
| **`SMPLX_MALE.npz`** | 108 MB | `common/utils/human_model_files/smplx/SMPLX_MALE.npz` | [SMPL-X Portal](https://smpl-x.is.tue.mpg.de) hoặc copy từ Drive |
| **`SMPLX_FEMALE.npz`** | 108 MB | `common/utils/human_model_files/smplx/SMPLX_FEMALE.npz` | [SMPL-X Portal](https://smpl-x.is.tue.mpg.de) hoặc copy từ Drive |

*(Gợi ý: Nếu bạn có sẵn thư mục trên máy tính hoặc Google Drive, chỉ cần nén 5 file trên thành file `models.zip` rồi dùng lệnh `scp` hoặc `gdown` tải lên server một lần duy nhất).*

#### Bước 3: Kiểm tra tính toàn vẹn của mô hình
Chạy lệnh kiểm tra tự động:
```bash
python download_models.py
```
Khi màn hình hiển thị đủ 5 dấu tích xanh `[✓] SẴN SÀNG`, hệ thống đã sẵn sàng 100%!

#### Bước 4: Triển khai trên Server

* **Cách A: Dùng Docker (Chuẩn Doanh nghiệp - Khuyên dùng):**
  ```bash
  # Build Docker Image:
  docker build -t ai-tailor:latest .

  # Khởi chạy Container phục vụ API 24/7 trên cổng 8000:
  docker run --gpus all -p 8000:8000 -d --name ai_tailor_service ai-tailor:latest python -m uvicorn app:app --host 0.0.0.0 --port 8000
  ```

* **Cách B: Cài đặt Native trên Ubuntu 22.04:**
  ```bash
  chmod +x setup_colab.sh
  ./setup_colab.sh

  # Chạy đo thử nghiệm:
  python run_measurement.py --image duong_dan_anh.jpg --height_cm 163 --weight_kg 64
  ```

---

### 🟡 TRƯỜNG HỢP 3: CHẠY TRÊN GOOGLE COLAB (GPU T4 MIỄN PHÍ)

1. Mở notebook **`run_on_colab.ipynb`** trên Google Colab.
2. Chọn Menu **Runtime -> Change runtime type -> T4 GPU**.
3. **Chạy Cell 1 (Mount Drive):** Notebook sẽ tự động tìm đúng thư mục dự án trên Google Drive của bạn.
4. **Chạy Cell 2 (`./setup_colab.sh`):** Cài đặt môi trường tự động (mất ~1-2 phút).
5. **Chạy Cell 3 & 4:** Tải ảnh lên và bấm chạy trích xuất số đo trong 1-2 giây.

---

## 📖 BẢNG THAM SỐ DÒNG LỆNH (`run_measurement.py`)

| Tham số | Kiểu dữ liệu | Mặc định | Ý nghĩa & Mô tả |
| :--- | :---: | :---: | :--- |
| **`--image`** | `str` | *Bắt buộc* | Đường dẫn tới file ảnh toàn thân (.jpg/.png). |
| **`--height_cm`** | `float` | `None` | Chiều cao thực tế (cm). **Khuyên dùng** để chuẩn hóa tỉ lệ người 1:1. |
| **`--weight_kg`** | `float` | `None` | Cân nặng thực tế (kg) để tinh chỉnh độ dày cơ thể bằng Adam optimizer. |
| **`--gender`** | `str` | `neutral` | Giới tính hình thể: `neutral` (trung tính), `male` (nam), `female` (nữ). |
| **`--save_mesh`** | `str` | `None` | Đường dẫn lưu file 3D Avatar (`.obj` hoặc `.ply`) để kéo vào CLO3D/Blender. |
| **`--output_json`**| `str` | `output_measurements.json` | Đường dẫn lưu file kết quả JSON. |
| **`--device`** | `str` | `cuda:0` / `cpu` | Thiết bị tính toán (tự nhận diện GPU CUDA nếu có). |

### Ví dụ câu lệnh đầy đủ:
```bash
python run_measurement.py \
    --image anh_khach_hang.jpg \
    --height_cm 163.0 \
    --weight_kg 64.0 \
    --gender male \
    --save_mesh avatar_163cm.obj \
    --output_json ket_qua.json
```

---

## 📏 CÔNG CỤ ĐỐI CHUẨN THỰC NGHIỆM VỚI THƯỚC DÂY (`evaluate_benchmark.py`)

Khi bạn đo người thật bằng thước dây may mặc, bạn có thể chạy công cụ này để kiểm chứng độ chính xác:
```bash
python evaluate_benchmark.py \
    --pred output_measurements.json \
    --real_height 163.0 \
    --real_chest 95.0 \
    --real_waist 80.0 \
    --real_hips 87.0 \
    --real_inseam 72.0 \
    --real_weight 64.0
```

Hệ thống sẽ tự động tính sai số trung bình tuyệt đối (**MAE**) và tỷ lệ % sai số, đối chiếu với tiêu chuẩn may đo công nghiệp (MTM) và may sẵn (RTW).

---

## 📸 NGUYÊN TẮC CHỤP ẢNH CHUẨN MAY MẶC (A-POSE)
Để sai số đạt ngưỡng xuất sắc ($< 1.5\text{ cm}$):
1. **Tư thế chữ A (A-pose):** Đứng thẳng, hai chân mở rộng bằng vai (thấy rõ đũng quần), hai tay dang nhẹ tạo góc $30^\circ - 40^\circ$ với thân người (không ép bắp tay vào mạng sườn).
2. **Góc đặt máy:** Đặt điện thoại ở độ cao **ngang rốn / thắt lưng** ($90 - 100\text{ cm}$ so với mặt đất), giữ máy thẳng đứng $90^\circ$ (không chếch chúc đầu máy xuống).
3. **Khoảng cách:** Đứng cách camera từ **2.5 mét đến 3.0 mét**.
4. **Trang phục:** Mặc đồ mỏng ôm sát (đồ gym, legging, áo thun ôm).
