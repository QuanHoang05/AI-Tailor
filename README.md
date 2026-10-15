# 🧵 AI-Tailor: Hệ Thống Trích Xuất Số Đo May Mặc Tự Động Từ Ảnh 2D (ISO 8559)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![SMPLer-X](https://img.shields.io/badge/3D%20Model-SMPLer--X-green.svg)](https://github.com/caizhongang/SMPLer-X)
[![Standard](https://img.shields.io/badge/Standard-ISO%208559-purple.svg)](https://www.iso.org/standard/64515.html)

**AI-Tailor** là giải pháp thị giác máy tính và hình học 3D tái tạo vóc dáng người hoàn chỉnh từ **1 bức ảnh 2D duy nhất**, tự động đo đạc 8 thông số nhân trắc học may mặc chính xác theo **tiêu chuẩn may đo quốc tế ISO 8559**.

Hệ thống được thiết kế theo chuẩn **Production-Ready**:
- **100% PyTorch thuần & Torchvision**: Đã loại bỏ hoàn toàn các phụ thuộc nhị phân C++ phức tạp của MMDetection/MMCV, miễn nhiễm vĩnh viễn với các lỗi lệch chuẩn ABI (`SetDeviceEi`, `bool subtraction`).
- **Khử nhiễu cánh tay T-pose (Arm Filtering)**: Thuật toán độc quyền ngăn ngừa hiện tượng chu vi vòng ngực bị phình to do dính tay mà không cắt lẹm vòng eo/hông của người to béo.
- **Hỗ trợ hình thể giới tính chuyên biệt**: Phân loại theo `male` (nam - dáng V-shape), `female` (nữ - dáng đồng hồ cát), và `neutral` (trung tính).

---

## 📐 8 Chỉ Số May Mặc Tiêu Chuẩn (ISO 8559)

| STT | Chỉ số đo | Mã ISO 8559 | Đơn vị | Phương pháp trích xuất |
|:---:|:---|:---:|:---:|:---|
| 1 | **Chiều cao toàn thân** | ISO 8559-1 (4.1.1) | cm | Khoảng cách đỉnh đầu (Vertex) $\to$ Mặt đất |
| 2 | **Vòng ngực (Bust / Chest)** | ISO 8559-1 (4.2.1) | cm | Thiết diện cắt ngang 3D qua điểm mốc nhũ hoa (Nipples) + Lọc tay |
| 3 | **Vòng eo (Waist)** | ISO 8559-1 (4.2.3) | cm | Thiết diện qua rốn / vùng hẹp nhất giữa lồng ngực và xương chậu |
| 4 | **Vòng mông / hông (Hips)** | ISO 8559-1 (4.2.5) | cm | Thiết diện cắt ngang qua đỉnh nở lớn nhất của cơ mông lớn |
| 5 | **Chiều rộng vai (Shoulder)** | ISO 8559-1 (4.3.1) | cm | Khoảng cách đường cong giải phẫu mỏm cùng vai trái $\to$ phải (Acromion) |
| 6 | **Chiều dài tay áo (Sleeve)** | ISO 8559-1 (4.3.7) | cm | Mỏm cùng vai $\to$ Khuỷu tay $\to$ Mỏm trâm cổ tay (Stylion) |
| 7 | **Chiều dài chân trong (Inseam)**| ISO 8559-1 (4.3.4) | cm | Đáy đũng quần (Crotch level) $\to$ Mặt sàn gót chân |
| 8 | **Cân nặng ước tính** | Anthropometric BMI | kg | Tích phân thể tích 3D lưới mesh $\times$ Khối lượng riêng $1.06 \text{ kg/dm}^3$ |

---

## 📂 Cấu Trúc Thư Mục

```text
AI-Tailor/
├── run_measurement.py          # Script chạy 1 chạm chính (BBox -> SMPLer-X -> ISO 8559)
├── run_on_colab.ipynb          # Notebook Google Colab tương tác trực quan
├── setup_colab.sh              # Script tự động cấu hình môi trường chuẩn
├── requirements.txt            # Danh sách gói phụ thuộc Python tối giản
├── architecture.md             # Tài liệu kiến trúc chi tiết hệ thống
├── report.md                   # Báo cáo chuyên gia đa góc nhìn (DevOps, 3D Vision, Tailor ISO)
├── common/                     # Thư viện mạng SMPLer-X & ma trận biến đổi PyTorch
├── data/SHAPY/                 # Module tính toán lát cắt 3D & bao lồi ConvexHull
└── main/                       # ViTPose Backbone & Configs
```

---

## 🚀 Hướng Dẫn Sử Dụng Nhanh (Google Colab / Linux)

### 1. Chạy trên Google Colab (GPU T4 miễn phí)
1. Tải repository này lên Google Drive hoặc clone trực tiếp vào Colab.
2. Mở file `run_on_colab.ipynb` và chọn GPU T4 (Runtime -> Change runtime type -> T4 GPU).
3. Chạy lệnh đo đạc:

```bash
python run_measurement.py \
    --image "path/to/anh_chup_khach_hang.jpg" \
    --gender "neutral" \
    --output_json "output_measurements.json"
```

### 2. Chuẩn hóa may đo 100% (Nếu khách cung cấp chiều cao / cân nặng)
```bash
python run_measurement.py \
    --image "path/to/anh_khach_hang.jpg" \
    --gender "male" \
    --height_cm 172.0 \
    --weight_kg 65.0 \
    --output_json "output_measurements.json"
```

---

## 📊 Kết Quả Đầu Ra Mẫu (output_measurements.json)

```json
{
  "thong_tin_nhap": {
    "anh_dau_vao": "anh_test.jpg",
    "gioi_tinh": "male",
    "chieu_cao_cung_cap_cm": 172.0,
    "can_nang_cung_cap_kg": 65.0
  },
  "so_do_may_mac_cm": {
    "chieu_cao": 172.0,
    "vong_nguc": 102.5,
    "vong_eo": 88.4,
    "vong_hong": 99.8,
    "rong_vai": 38.2,
    "dai_tay_ao": 52.1,
    "dai_chan_inseam": 76.5,
    "can_nang_uoc_tinh_kg": 65.0
  },
  "trang_thai": "Thành công"
}
```

---

## 📜 Trích Dẫn & Bản Quyền
- Dựa trên nền tảng tái tạo người 3D **SMPLer-X** (CVPR 2024 / 2025).
- Định nghĩa mốc nhân trắc học theo tiêu chuẩn **ISO 8559-1:2017 & ISO 8559-2:2017**.
- Giấy phép: MIT License.
