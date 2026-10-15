# 🚀 HƯỚNG DẪN TRIỂN KHAI VẬN HÀNH THỰC TẾ (PRODUCTION DEPLOYMENT GUIDE)
## DỰ ÁN: AI-TAILOR — TRÍCH XUẤT SỐ ĐO MAY MẶC 3D (ISO 8559)

> Tài liệu này hướng dẫn chi tiết cách chạy dự án ngay lập tức trên 3 môi trường:
> 1. **Google Colab (Miễn phí / T4 GPU)**
> 2. **Máy tính Cá nhân (Local Windows / Linux)**
> 3. **Thuê Máy chủ Đám mây (Cloud Server / RunPod / AWS / Vast.ai / Docker)**

---

## 🟢 MÔI TRƯỜNG 1: GOOGLE COLAB (ĐÃ KIỂM THỬ THÀNH CÔNG 100%)

### Tình trạng: SẴN SÀNG CHẠY NGAY
Bạn đã chạy thử nghiệm thành công trên Google Colab với kết quả chuẩn ISO 8559 (Ngực: 95.3cm, Eo: 80.9cm, Inseam: 71.9cm).

### Các bước thực hiện:
1. Mở file `run_on_colab.ipynb` trên Google Colab.
2. **Cell 1 (Mount Google Drive):** Chạy để liên kết với thư mục chứa mã nguồn và các file mô hình có sẵn trên Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Hệ thống sẽ tự động tìm và cd vào thư mục Focused-Clean-Colab trên Drive của bạn
   ```
3. **Cell 2 (Cài môi trường):** Chạy `!./setup_colab.sh` (mất khoảng 1–2 phút cho lần đầu tiên).
4. **Cell 3 (Chạy đo):**
   ```bash
   !python run_measurement.py \
       --image "duong_dan_anh.jpg" \
       --gender "neutral" \
       --height_cm 163.0 \
       --weight_kg 64.0 \
       --save_mesh "output_avatar.obj"
   ```
5. Kết quả số đo xuất ra màn hình và lưu vào file `output_measurements.json`.

---

## 🟡 MÔI TRƯỜNG 2: MÁY TÍNH CÁ NHÂN (LOCAL WINDOWS)

### Yêu cầu cấu hình máy:
* Hệ điều hành: Windows 10 / 11 (64-bit).
* Python 3.10 (khuyên dùng Anaconda hoặc Miniconda).
* Card màn hình NVIDIA (GTX 1060 / 1660 / RTX 2060 trở lên) để chạy GPU; nếu không có card rời, hệ thống sẽ tự động chạy ở chế độ CPU (mất khoảng 8-10 giây/ảnh).

### Các bước thực hiện:
1. Mở thư mục dự án trên máy:
   ```powershell
   cd d:\Focused-human-body-measurementTestProject\Focused-Clean-Colab_2\Focused-Clean-Colab
   ```
2. **Cài đặt tự động:**
   * Cách A (Khuyên dùng): Nhấp đúp chuột vào file `setup_windows.bat`.
   * Cách B (Dùng Conda):
     ```powershell
     conda env create -f environment.yml
     conda activate ai-tailor
     ```
3. **Kiểm tra file mô hình:**
   ```powershell
   python download_models.py
   ```
   *(Toàn bộ 5 file mô hình đã có sẵn trên máy của bạn).*
4. **Chạy đo thử nghiệm:**
   ```powershell
   python run_measurement.py --image ..\..\imageTest\ten_anh.jpg --height_cm 163 --weight_kg 64 --save_mesh avatar.obj
   ```
5. **Chạy REST API Server (để các ứng dụng khác gọi):**
   ```powershell
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   Mở trình duyệt truy cập: `http://localhost:8000/docs` để thử nghiệm giao diện API.

---

## 🔵 MÔI TRƯỜNG 3: THUÊ MÁY CHỦ ĐÁM MÂY (CLOUD VPS / RUNPOD / AWS / DOCKER)

### Lựa chọn cấu hình khi thuê server:
* **Nền tảng khuyên dùng:** RunPod.io, Vast.ai, hoặc AWS EC2 (g4dn.xlarge).
* **Giá thuê:** Khoảng **0.2$ - 0.4$ / giờ** (rất rẻ).
* **Hệ điều hành:** Ubuntu 22.04 LTS có GPU NVIDIA (T4, A10, RTX 3090, RTX 4090).

### Cách 1: Triển khai qua Docker (Chuẩn Doanh nghiệp - 1 Lệnh duy nhất)
1. Clone mã nguồn từ GitHub:
   ```bash
   git clone https://github.com/QuanHoang05/AI-Tailor.git
   cd AI-Tailor
   ```
2. Sao chép các file trọng số mô hình lớn (>100MB) vào thư mục tương ứng:
   - File `smpler_x_s32.pth.tar` vào `pretrained_models/`
   - File `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz` vào `common/utils/human_model_files/smplx/`
   *(Bạn có thể upload từ Google Drive qua lệnh `gdown` hoặc copy qua SCP/FileZilla).*
3. Build và khởi chạy Docker:
   ```bash
   docker build -t ai-tailor:latest .
   docker run --gpus all -p 8000:8000 -d --name ai_tailor_service ai-tailor:latest python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```
4. Server API chạy liên tục 24/7, sẵn sàng phục vụ hàng ngàn yêu cầu đo đạc mỗi ngày qua cổng `8000`.

### Cách 2: Triển khai Native trên Linux (Không dùng Docker)
1. Clone repo và chạy cài đặt tự động:
   ```bash
   git clone https://github.com/QuanHoang05/AI-Tailor.git
   cd AI-Tailor
   chmod +x setup_colab.sh
   ./setup_colab.sh
   ```
2. Đặt các file mô hình vào đúng thư mục rồi chạy:
   ```bash
   python run_measurement.py --image anh_test.jpg --height_cm 163 --weight_kg 64
   ```

---

## 📦 LƯU Ý SỐNG CÒN VỀ CÁC FILE MÔ HÌNH NẶNG (> 100MB)

Vì GitHub giới hạn kích thước mỗi file không vượt quá 100MB, mã nguồn đẩy lên GitHub chỉ chứa code, không chứa 4 file sau:
1. `pretrained_models/smpler_x_s32.pth.tar` (384MB)
2. `common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz` (108MB)
3. `common/utils/human_model_files/smplx/SMPLX_MALE.npz` (108MB)
4. `common/utils/human_model_files/smplx/SMPLX_FEMALE.npz` (108MB)

* **Trên Colab (Google Drive):** Đã có sẵn 100% trong Drive của bạn.
* **Trên Máy Local (ổ D:):** Đã có sẵn 100% trong máy của bạn.
* **Khi thuê Server mới:** Bạn chỉ cần nén 4 file này thành 1 file zip (`models.zip` khoảng 600MB), upload lên Google Drive hoặc dùng `scp` đẩy lên server 1 lần duy nhất là xong.
