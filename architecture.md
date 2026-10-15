# 🏗️ KIẾN TRÚC DỰ ÁN: Focused-Clean-Colab

> Hệ thống trích xuất số đo cơ thể may mặc từ ảnh 2D, dựa trên SMPLer-X (NeurIPS 2023) + SHAPY.  
> Luồng chính: **Ảnh 2D → Phát hiện người → Ước tính 3D SMPL-X → Tối ưu hóa hình thể → Số đo ISO 8559**

---

## 📁 Cấu trúc thư mục tổng quan

```
Focused-Clean-Colab/
│
├── run_measurement.py          ← ENTRY POINT – Script chính, chạy toàn bộ pipeline
├── setup_colab.sh              ← Script cài đặt môi trường Colab (conda/pip)
├── run_on_colab.ipynb          ← Notebook Google Colab step-by-step
├── requirements.txt            ← Danh sách thư viện Python cần cài
├── README.md                   ← Hướng dẫn nhanh
├── architecture.md             ← File này
│
├── main/                       ← Model SMPLer-X (Encoder + Decoder + Head)
├── common/                     ← Utilities dùng chung (base class, preprocessing, SMPL-X wrapper)
├── data/                       ← Module đo lường SHAPY + dataset adapter
└── pretrained_models/          ← Trọng số (checkpoint) của các model
```

---

## 🔑 FILES QUAN TRỌNG NHẤT

### `run_measurement.py` — Entry Point & Pipeline Chính
**Nhiệm vụ:** Kết nối tất cả modules lại thành một luồng duy nhất.

| Hàm | Vai trò |
|-----|---------|
| `parse_args()` | Nhận tham số CLI: `--image`, `--height_cm`, `--weight_kg`, `--device`, ... |
| `load_detector()` | Khởi tạo Faster R-CNN (mmdet) để phát hiện người trong ảnh |
| `load_smpler_x_model()` | Load checkpoint Focused SMPLer-X, trả về `Demoer` object |
| `detect_person()` | Chạy detector, trả về bounding-box `[x, y, w, h]` của người lớn nhất |
| `extract_3d_mesh()` | Crop ảnh → normalize → forward qua SMPLer-X → trích `betas` (10 giá trị hình thể) |
| `compute_measurements()` | **Hàm cốt lõi**: tạo lưới SMPL-X T-pose → [tối ưu betas] → đo chu vi & chiều cao |
| `main()` | Gọi tuần tự 4 bước trên, in bảng kết quả, lưu JSON |

**Luồng dữ liệu bên trong `compute_measurements()`:**
```
betas (1×10)
  → smplx_layer(betas, zero_poses) → vertices (1×10475×3)
  → [NẾU có height/weight] Adam optimizer 35 steps tinh chỉnh betas
  → smplx_layer(opt_betas) → final vertices
  → BodyMeasurements(triangles) → chest, waist, hips, height, mass
  → vertex indices hardcoded → shoulder_width, arm_length, inseam
  → scale_factor = height_input / est_height (nếu có)
  → JSON output
```

---

## 📂 `main/` — Model SMPLer-X

### `main/SMPLer_X.py`
**Nhiệm vụ:** Định nghĩa kiến trúc mạng SMPLer-X đầy đủ.

| Class/Hàm | Vai trò |
|-----------|---------|
| `class Model(nn.Module)` | Toàn bộ mô hình: encoder ViTPose + body/hand/face regressor |
| `Model.forward()` | Forward pass: ảnh → feature → joints → SMPL-X params (pose, shape, expr) |
| `get_model(mode)` | Factory function, khởi tạo `Model` với các sub-networks |
| `Model.smplx_layer` | SMPL-X layer nhúng trong model để tính loss khi training |

**Output key của `Model.forward()` ở mode `'test'`:**
- `out['smplx_shape']` → **betas** (1×10) — dùng trong `run_measurement.py`

### `main/config.py`
**Nhiệm vụ:** Class `Config` — đọc file `.py` config model, quản lý đường dẫn.

| Phương thức | Vai trò |
|-------------|---------|
| `get_config_fromfile(path)` | Load config từ `main/config/config_*.py` via mmcv |
| `update_test_config(...)` | Set checkpoint path, testset name, eval options |
| `update_config(num_gpus, exp_name)` | Tạo thư mục output, backup code |

### `main/config/config_smpler_x_s32.py`
**Nhiệm vụ:** Siêu tham số của model variant `smpler_x_s32`.
- `input_img_shape = (512, 384)` — kích thước ảnh đầu vào
- `encoder_pretrained_model_path` → `vitpose_small.pth`

### `main/transformer_utils/`
**Nhiệm vụ:** Config và wrapper cho ViTPose encoder. Không cần chỉnh sửa khi chỉ inference.

---

## 📂 `common/` — Utilities Dùng Chung

### `common/base.py`
**Nhiệm vụ:** Base classes cho train/test/demo pipeline.

| Class | Vai trò |
|-------|---------|
| `class Base` | Abstract base với timer, logger |
| `class Trainer(Base)` | (**Không dùng trong inference**) Pipeline training |
| `class Tester(Base)` | (**Không dùng**) Pipeline evaluation benchmark |
| `class Demoer(Base)` | **DÙNG** — Load model checkpoint, expose `self.model` |
| `Demoer._make_model()` | Load weights từ `.pth.tar`, remap key names, set eval mode |

> **Lưu ý:** Các hàm `_make_batch_generator()` và `_evaluate()` trong `Demoer` là legacy
> từ demo script gốc — **không được gọi** trong `run_measurement.py`.

### `common/utils/human_models.py`
**Nhiệm vụ:** Wrapper cho SMPL-X model, expose singleton `smpl_x`.

| Attribute | Giá trị / Vai trò |
|-----------|-------------------|
| `smpl_x.layer['neutral']` | SMPL-X nn.Module cho giới tính trung tính |
| `smpl_x.face` | Face indices (20908×3) để tạo triangles |
| `smpl_x.vertex_num` | 10475 vertices |
| `smpl_x.J_regressor` | Ma trận chiếu vertices → joints |

### `common/utils/preprocessing.py`
- `process_bbox(bbox, w, h)` — normalize & expand bounding box
- `generate_patch_image(img, bbox, ...)` — crop + affine transform vào `input_img_shape`

### `common/utils/inference_utils.py`
- `process_mmdet_results(results, cat_id, multi_person)` — lọc detection, trả về list bbox

### `common/utils/smplx/`
**Nhiệm vụ:** Thư viện SMPL-X Python được nhúng trực tiếp để tránh dependency conflict.

### `common/nets/`
- `smpler_x.py` — PositionNet, BoxNet, HandRoI, BodyRotationNet, HandRotationNet, FaceRegressor
- `loss.py` — CoordLoss, ParamLoss, CELoss (**chỉ dùng khi training**)
- `layer.py` — Helper layers (linear, conv wrappers)

---

## 📂 `data/` — Module Đo Lường & Dataset Adapter

### `data/SHAPY/mesh-mesh-intersection/body_measurements/body_measurements.py`
**Nhiệm vụ:** Module đo chu vi và khối lượng cơ thể 3D theo ISO 8559. **Đây là module output cuối.**

| Phương thức | Vai trò |
|-------------|---------|
| `compute_height(triangles)` | Đo khoảng cách heel → head_top theo trục Y |
| `compute_mass(triangles)` | Thể tích lưới × density 985 kg/m³ |
| `compute_peripheries(triangles)` | MeshMeshIntersection + ConvexHull → chu vi ngực/eo/hông |
| `forward(triangles)` | Gọi 3 hàm trên, trả về `{'measurements': {...}}` |

**Output keys từ `forward()`:** `chest`, `waist`, `hips`, `height`, `mass`  
(mỗi key có sub-key `['tensor']`, đơn vị: **mét** và **kg**)

### `data/SHAPY/mesh-mesh-intersection/data/`
| File | Nội dung |
|------|---------|
| `measurement_defitions.yaml` | CW_p=chest, BW_p=waist, IW_p=hips, H=height, ... |
| `smpl_measurement_vertices.yaml` | Barycentric coords: NippleRight, BellyButton, Crotch, HeelLeft, HeadTop |

### `data/dataset.py`
**Nhiệm vụ:** `MultipleDatasets` — kết hợp nhiều dataset khi training. **Không dùng trong inference.**

---

## 📂 `pretrained_models/` — Model Weights

| File | Kích thước | Vai trò |
|------|-----------|---------|
| `smpler_x_s32.pth.tar` | ~366 MB | Checkpoint Focused SMPLer-X (ViTPose-Small, batch 32) |
| `vitpose_small.pth` | ~93 MB | Pre-trained ViTPose-Small encoder |
| `mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth` | ~160 MB | Faster R-CNN COCO |
| `mmdet/mmdet_faster_rcnn_r50_fpn_coco.py` | ~6 KB | Config mmdet detector |

---

## 📂 `common/utils/human_model_files/smplx/` — SMPL-X Models

| File | Kích thước | Vai trò |
|------|-----------|---------|
| `SMPLX_NEUTRAL.npz` | ~104 MB | **Khuôn mẫu chính** dùng cho đo lường |
| `SMPLX_MALE.npz` | ~104 MB | Khuôn mẫu nam (optional) |
| `SMPLX_FEMALE.npz` | ~104 MB | Khuôn mẫu nữ (optional) |
| `SMPLX_to_J14.pkl` | ~4.5 MB | Regressor vertices → 14 joints |
| `MANO_SMPLX_vertex_ids.pkl` | ~13 KB | Hand vertex mapping |
| `SMPL-X__FLAME_vertex_ids.npy` | ~39 KB | Face vertex mapping |

**Trạng thái:** ✅ Tất cả đã tải đủ và đúng thư mục.

---

## 🔄 Luồng Dữ Liệu End-to-End

```
📷 input_image.jpg
    │
    ▼  load_detector()
┌──────────────────────────────────────────────┐
│  mmdet Faster R-CNN                          │
│  → detect person bboxes (cat_id=0)           │
│  → pick largest/most confident box           │
└──────────────┬───────────────────────────────┘
               │ box_xywh [x, y, w, h]
               ▼  extract_3d_mesh()
┌──────────────────────────────────────────────┐
│  process_bbox() → expand 1.2×               │
│  generate_patch_image() → 512×384 crop       │
│  normalize pixels → [0, 1] float32           │
│  → SMPLer-X forward() [ViTPose + regressors] │
│  → out['smplx_shape'] → betas (1×10)         │
└──────────────┬───────────────────────────────┘
               │ betas (10-dim shape vector)
               ▼  compute_measurements()
┌──────────────────────────────────────────────┐
│  smplx_layer(betas, zero_poses)              │
│  → vertices (10475×3) → triangles (20908×3×3)│
│                                              │
│  [IF height/weight provided]:                │
│    Adam optimizer, lr=0.04, 35 steps         │
│    L = 150×(h-h*)² + 2×(m-m*)² + 0.8×‖Δβ‖²│
│    → opt_betas                               │
│                                              │
│  BodyMeasurements(triangles):                │
│    height (m), mass (kg),                   │
│    chest, waist, hips (m)                   │
│                                              │
│  Vertex indices (T-pose topology):           │
│    [3865]↔[18186] → shoulder width          │
│    [3865]→[3363]  → arm length              │
│    [6194]→[15605] → inseam (Y-delta)        │
│                                              │
│  scale_factor = h_input / h_estimated        │
│  → scale all measurements                   │
└──────────────┬───────────────────────────────┘
               │
               ▼
📊 output_measurements.json
{
  "thong_tin_nhap": { "chieu_cao_nhap_cm", "can_nang_nhap_kg", "scale_factor" },
  "so_do_may_mac_cm": {
    "chieu_cao", "vong_nguc", "vong_eo", "vong_hong",
    "rong_vai", "dai_tay_ao", "dai_chan_inseam",
    "can_nang_uoc_tinh_kg"
  },
  "smplx_betas": [10 floats]
}
```

---

## ⚠️ Điểm Quan Trọng Khi Vận Hành

### 1. Dependency C++ cần compile
```bash
cd data/SHAPY/mesh-mesh-intersection
pip install -e .
```
Module `mesh_mesh_intersection` là C++ CUDA extension, **phải build trước khi chạy**.

### 2. Vertex Indices Cứng Nhắc
```python
shoulder_width = verts[3865]  vs  verts[18186]   # Acromion trái-phải
arm_length     = verts[3865]  vs  verts[3363]     # Acromion → wrist
inseam         = verts[6194]  vs  verts[15605]    # Crotch → heel (Y)
```
Dựa trên topology SMPL-X cố định (10475 vertices). Không thay đổi giữa các cá nhân.

### 3. T-pose vs Predicted Pose
Phép đo luôn dùng **T-pose trung tính** (zero body/hand/jaw poses), chỉ dùng `betas` từ inference. Đảm bảo tính nhất quán của số đo hình học.

### 4. Scale Factor
Khi có `--height_cm`, mọi số đo đều nhân với `scale_factor = h_input / h_estimated`. Kết quả: chiều cao khớp tuyệt đối, các số đo khác co giãn tỉ lệ.

---

## 🚀 Cách Chạy

```bash
# Cơ bản
python run_measurement.py --image person.jpg

# Khuyến nghị: cung cấp chiều cao thực tế
python run_measurement.py --image person.jpg --height_cm 170.0

# Tốt nhất: cả chiều cao + cân nặng
python run_measurement.py --image person.jpg --height_cm 170.0 --weight_kg 65.0

# Chỉ định thiết bị và file output
python run_measurement.py --image person.jpg --height_cm 170.0 --device cuda:0 --output_json results.json
```
