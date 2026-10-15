#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
HỆ THỐNG TRÍCH XUẤT SỐ ĐO MAY MẶC 1 CHẠM (END-TO-END BODY MEASUREMENT)
-------------------------------------------------------------------------------
[BẢNG CHỈ MỤC (INDEX) CÁC HÀM TRONG FILE]:
1. parse_args()           : Nhận tham số dòng lệnh (--image, --height_cm, --weight_kg...).
2. load_detector()        : Nạp mô hình Faster R-CNN (MMDetection) để tìm người trong ảnh 2D.
3. load_smpler_x_model()  : Nạp mô hình SMPLer-X (ViTPose + Regressors) từ checkpoint .pth.tar.
4. detect_person()        : Tìm khung hình chữ nhật (Bounding Box) chứa người lớn nhất trong ảnh.
5. extract_3d_mesh()      : Cắt ảnh Affine (512x384) -> Chạy SMPLer-X -> Xuất 10 tham số hình thể (betas).
6. get_landmark_3d()      : [ĐÃ SỬA LỖI] Nội suy tọa độ 3D chuẩn từ face_idx và trọng số bc (tránh IndexError).
7. compute_measurements() : Tối ưu hóa betas theo chiều cao/cân nặng -> Đo chu vi 3D & khoảng cách giải phẫu.
8. main()                 : Luồng điều phối chính (Ảnh -> BBox -> 3D Mesh -> Số đo -> JSON).

[LƯU Ý KHI CHỈNH SỬA (CONFLICT WARNINGS)]:
- Không lấy trực tiếp `face_idx` trong smplx_measurements.yaml để truy xuất vào `vertices`
  vì `vertices` chỉ có 10.475 đỉnh trong khi `triangles` có 20.908 mặt. Luôn dùng hàm `get_landmark_3d()`.
- Khi chạy tối ưu hóa betas (Adam 35 bước), chỉ tối ưu trên tư thế T-pose trung tính (zero_pose)
  để các mặt phẳng cắt ngang ngực/eo/hông không bị nghiêng theo tư thế đứng của người trong ảnh.
===============================================================================
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
try:
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass
import sys
import os.path as osp
import glob
import ctypes

# Tự động nạp libcudart.so.12 vào global symbol table nếu thiếu trong LD_LIBRARY_PATH (khắc phục lỗi MMCV trên Colab)
cuda_candidates = (
    glob.glob('/usr/local/cuda*/lib64/libcudart.so.12*') +
    glob.glob('/usr/local/cuda*/lib64/libcudart.so*') +
    glob.glob('/usr/lib/x86_64-linux-gnu/libcudart.so.12*') +
    glob.glob('/usr/lib/libcudart.so.12*') +
    glob.glob('/opt/py310/**/libcudart.so.12*', recursive=True)
)
for lib_p in cuda_candidates:
    try:
        ctypes.CDLL(lib_p, mode=ctypes.RTLD_GLOBAL)
        break
    except Exception:
        pass

if '/usr/local/cuda/lib64' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import argparse
import json
import yaml
import numpy as np
import torch
import cv2

# Thiết lập đường dẫn hệ thống ưu tiên các module nội bộ của dự án
CURRENT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(CURRENT_DIR, 'main'))
sys.path.insert(0, osp.join(CURRENT_DIR, 'common'))
sys.path.insert(0, osp.join(CURRENT_DIR, 'data'))
sys.path.insert(0, osp.join(CURRENT_DIR, 'data', 'SHAPY', 'mesh-mesh-intersection'))

from config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Trích xuất số đo cơ thể từ ảnh 2D theo chuẩn ISO 8559")
    parser.add_argument('--image', '--img_path', dest='image', type=str, required=True,
                        help="Đường dẫn tới file ảnh toàn thân (jpg/png)")
    parser.add_argument('--output_json', type=str, default='output_measurements.json',
                        help="Đường dẫn lưu file kết quả JSON")
    parser.add_argument('--height_cm', type=float, default=None,
                        help="Chiều cao thực tế (cm) - Khuyên dùng để chuẩn hóa tỉ lệ chuẩn 100%%")
    parser.add_argument('--weight_kg', type=float, default=None,
                        help="Cân nặng thực tế (kg) - Tùy chọn để tinh chỉnh độ dày cơ thể")
    parser.add_argument('--gender', type=str, default='neutral',
                        help="Giới tính hình thể (neutral, male/nam, female/nu - mặc định: neutral)")
    parser.add_argument('--pretrained_model', type=str, default='smpler_x_s32',
                        help="Tên model checkpoint (mặc định: smpler_x_s32)")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help="Thiết bị tính toán (cuda:0 hoặc cpu)")
    parser.add_argument('--save_mesh', type=str, default=None,
                        help="Đường dẫn lưu file 3D avatar (.obj hoặc .ply) để import vào CLO3D/Blender")
    args = parser.parse_args()
    if not args.gender or str(args.gender).strip().lower() in ['', '""', "''", 'none', 'null']:
        args.gender = 'neutral'
    return args


def load_detector(device):
    """
    Bước 1: Khởi tạo Detector phát hiện người trong ảnh.
    Sử dụng Torchvision Faster R-CNN (100% PyTorch thuần, miễn nhiễm hoàn toàn lỗi MMCV C++ ABI) hoặc Auto-Crop.
    """
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        print(f"[*] Đang nạp Detector người chuẩn PyTorch (Torchvision Faster R-CNN) trên {device}...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        detector = fasterrcnn_resnet50_fpn(weights=weights)
        detector.eval()
        if 'cuda' in device and torch.cuda.is_available():
            detector = detector.to(device)
        return ('torchvision', detector)
    except Exception as e:
        print(f"[!] Torchvision detector ({e}). Sẽ sử dụng Auto-Crop toàn thân.")
        return ('autocrop', None)


def load_smpler_x_model(pretrained_model_name, device):
    """Bước 2: Nạp cấu hình và trọng số mô hình 3D SMPLer-X."""
    config_path = osp.join(CURRENT_DIR, 'main', 'config', f'config_{pretrained_model_name}.py')
    ckpt_path = osp.join(CURRENT_DIR, 'pretrained_models', f'{pretrained_model_name}.pth.tar')

    if not osp.exists(config_path) or not osp.exists(ckpt_path):
        raise FileNotFoundError(f"Thiếu config {config_path} hoặc checkpoint {ckpt_path}")

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config('EHF', 'na', shapy_eval_split=None, pretrained_model_path=ckpt_path, use_cache=False)
    cfg.update_config(1, 'output/measurement_run')

    print(f"[*] Đang nạp mô hình 3D SMPLer-X ({pretrained_model_name})...")
    from base import Demoer
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()
    if 'cuda' in device and torch.cuda.is_available():
        demoer.model = demoer.model.cuda()
    return demoer


def detect_person(detector, img_input):
    """Lấy tọa độ khung hình chữ nhật [x, y, w, h] của người trong ảnh (hỗ trợ đường dẫn file hoặc mảng numpy)."""
    import cv2
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    elif isinstance(img_input, np.ndarray):
        img = img_input
    else:
        raise ValueError(f"Đầu vào ảnh không hợp lệ: {type(img_input)}")

    if img is None:
        raise ValueError(f"Không thể đọc file ảnh: {img_input}")
    h, w = img.shape[:2]

    det_type = detector[0] if isinstance(detector, tuple) else 'autocrop'
    model = detector[1] if isinstance(detector, tuple) else None

    if det_type == 'torchvision' and model is not None:
        try:
            from torchvision.transforms import functional as TF
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = TF.to_tensor(img_rgb)
            dev = next(model.parameters()).device
            tensor = tensor.to(dev).unsqueeze(0)
            with torch.no_grad():
                preds = model(tensor)[0]
            person_mask = (preds['labels'] == 1) & (preds['scores'] > 0.5)
            boxes = preds['boxes'][person_mask].cpu().numpy()
            if len(boxes) > 0:
                best_box = boxes[0]
                box_xywh = np.array([
                    float(best_box[0]),
                    float(best_box[1]),
                    float(abs(best_box[2] - best_box[0])),
                    float(abs(best_box[3] - best_box[1]))
                ], dtype=np.float32)
                print(f"[✓] Đã phát hiện người: [x={box_xywh[0]:.1f}, y={box_xywh[1]:.1f}, w={box_xywh[2]:.1f}, h={box_xywh[3]:.1f}]")
                return box_xywh
        except Exception as e:
            print(f"[!] Torchvision inference gặp lỗi ({e}). Chuyển sang Auto-Crop.")

    # Fallback tự động: Ảnh chụp người toàn thân, lấy bounding box căn chuẩn giữa ảnh
    box_xywh = np.array([0.05 * w, 0.02 * h, 0.90 * w, 0.96 * h], dtype=np.float32)
    print(f"[*] Auto-Crop: Tự động định vị khung người toàn thân [{w}x{h}] -> {box_xywh}")
    return box_xywh


def extract_3d_mesh(demoer, original_img, box_xywh, device):
    """Cắt ảnh theo phép biến đổi Affine 512x384 và suy luận vector hình thể betas (1x10)."""
    from utils.preprocessing import process_bbox, generate_patch_image

    h, w = original_img.shape[:2]
    bbox = process_bbox(box_xywh, w, h)
    img_patch, _, _ = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)

    img_tensor = torch.from_numpy(img_patch.astype(np.float32) / 255.0).permute(2, 0, 1)[None, :, :, :]
    if 'cuda' in device and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    inputs = {'img': img_tensor}
    with torch.no_grad():
        out = demoer.model(inputs, {}, {}, 'test')

    betas = out['smplx_shape'].reshape(1, 10)
    return betas, bbox


def get_landmark_3d(triangles_np, landmark_dict, name):
    """
    Tính tọa độ điểm mốc 3D chuẩn xác từ chỉ số mặt tam giác (face_idx) và tọa độ trọng tâm (bc).
    Khắc phục triệt để lỗi IndexError (18186 > 10475) khi nhầm face_idx với vertex_idx.
    """
    face_idx = landmark_dict[name]['face_idx']
    bc = np.array(landmark_dict[name]['bc'], dtype=np.float32).reshape(3, 1)
    tri_verts = triangles_np[face_idx]  # (3, 3) - 3 đỉnh của mặt tam giác thứ face_idx
    return (tri_verts * bc).sum(axis=0)


def compute_measurements(betas, height_cm_input=None, weight_kg_input=None, gender='neutral', device='cuda:0', save_mesh_path=None):
    """Tạo lưới SMPL-X T-pose, tối ưu hóa betas và trích xuất toàn bộ số đo may mặc ISO 8559."""
    from utils.human_models import smpl_x
    from body_measurements import BodyMeasurements

    gender_map = {
        'male': 'male', 'nam': 'male', 'm': 'male',
        'female': 'female', 'nu': 'female', 'nữ': 'female', 'f': 'female',
        'neutral': 'neutral', 'trung_tinh': 'neutral'
    }
    active_gender = gender_map.get(str(gender).lower().strip(), 'neutral')

    use_cuda = ('cuda' in device and torch.cuda.is_available())
    smplx_layer = smpl_x.layer[active_gender]
    if use_cuda:
        smplx_layer = smplx_layer.cuda()
        betas = betas.cuda()

    # Khởi tạo tư thế T-pose chuẩn (tất cả góc khớp = 0) để đo đạc nhất quán
    zero_pose_body = torch.zeros((1, 63), device=betas.device, dtype=torch.float32)
    zero_pose_root = torch.zeros((1, 3), device=betas.device, dtype=torch.float32)
    zero_hand = torch.zeros((1, 45), device=betas.device, dtype=torch.float32)
    zero_jaw = torch.zeros((1, 3), device=betas.device, dtype=torch.float32)
    zero_eye = torch.zeros((1, 3), device=betas.device, dtype=torch.float32)
    zero_expr = torch.zeros((1, 10), device=betas.device, dtype=torch.float32)

    faces_tensor = torch.tensor(smpl_x.face.astype(np.int64), device=betas.device)

    meas_def_path = osp.join(CURRENT_DIR, 'data', 'SHAPY', 'mesh-mesh-intersection', 'data', 'measurement_defitions.yaml')
    meas_verts_path = osp.join(CURRENT_DIR, 'data', 'SHAPY', 'mesh-mesh-intersection', 'data', 'smplx_measurements.yaml')
    meas_cfg = {
        'meas_definition_path': meas_def_path,
        'meas_vertices_path': meas_verts_path,
    }
    meas_module = BodyMeasurements(meas_cfg)
    if use_cuda:
        meas_module = meas_module.cuda()

    has_height = (height_cm_input is not None and height_cm_input > 0)
    has_weight = (weight_kg_input is not None and weight_kg_input > 0)

    # Tối ưu hóa 10 tham số hình thể (betas) nếu người dùng nhập chiều cao hoặc cân nặng thật
    if has_height or has_weight:
        mode_str = []
        if has_height:
            mode_str.append(f"Chiều cao {height_cm_input:.1f} cm")
        if has_weight:
            mode_str.append(f"Cân nặng {weight_kg_input:.1f} kg")
        print(f"[*] Đang tinh chỉnh hình thể 3D (Betas Optimization) theo: {', '.join(mode_str)}...")

        target_h_m = (height_cm_input / 100.0) if has_height else None
        target_m_kg = weight_kg_input if has_weight else None

        init_betas = betas.clone().detach()
        opt_betas = betas.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt_betas], lr=0.04)

        for _ in range(35):
            optimizer.zero_grad()
            out_opt = smplx_layer(
                betas=opt_betas, body_pose=zero_pose_body, global_orient=zero_pose_root,
                right_hand_pose=zero_hand, left_hand_pose=zero_hand, jaw_pose=zero_jaw,
                leye_pose=zero_eye, reye_pose=zero_eye, expression=zero_expr, return_verts=True
            )
            tris_opt = out_opt.vertices[:, faces_tensor]

            loss = 0.0
            if has_height:
                cur_h = meas_module.compute_height(tris_opt)[0]
                loss = loss + 150.0 * (cur_h - target_h_m).pow(2).sum()
            if has_weight:
                cur_m = meas_module.compute_mass(tris_opt)
                loss = loss + 2.0 * (cur_m - target_m_kg).pow(2).sum()

            loss = loss + 0.8 * (opt_betas - init_betas).pow(2).sum()
            loss.backward()
            optimizer.step()

        betas = opt_betas.detach()
        print("[✓] Hoàn tất tinh chỉnh hình thể 3D.")

    # Tạo lưới 3D cuối cùng từ betas đã tinh chỉnh
    output = smplx_layer(
        betas=betas, body_pose=zero_pose_body, global_orient=zero_pose_root,
        right_hand_pose=zero_hand, left_hand_pose=zero_hand, jaw_pose=zero_jaw,
        leye_pose=zero_eye, reye_pose=zero_eye, expression=zero_expr, return_verts=True
    )
    vertices = output.vertices
    triangles = vertices[:, faces_tensor]

    # Đo chu vi (Ngực, Eo, Hông), Chiều cao và Cân nặng bằng mặt phẳng cắt ngang 3D
    meas_raw = meas_module(triangles)['measurements']
    est_height_m = meas_raw['height']['tensor'].item()
    est_chest_m = meas_raw['chest']['tensor'].item()
    est_waist_m = meas_raw['waist']['tensor'].item()
    est_hips_m = meas_raw['hips']['tensor'].item()
    est_mass_kg = meas_raw['mass']['tensor'].item()

    # Đo khoảng cách giải phẫu (Rộng vai, Dài tay, Dài chân trong) từ tọa độ trọng tâm mặt tam giác
    with open(meas_verts_path, 'r') as f:
        landmarks_yaml = yaml.safe_load(f)

    tris_np = triangles.detach().cpu().numpy()[0]  # (20908, 3, 3)
    pt_shoulder_l = get_landmark_3d(tris_np, landmarks_yaml, 'ShoulderLeft')
    pt_shoulder_r = get_landmark_3d(tris_np, landmarks_yaml, 'ShoulderRight')
    pt_wrist_l = get_landmark_3d(tris_np, landmarks_yaml, 'WristLeft')
    pt_crotch = get_landmark_3d(tris_np, landmarks_yaml, 'Crotch')
    pt_heel_l = get_landmark_3d(tris_np, landmarks_yaml, 'HeelLeft')

    shoulder_width_m = float(np.linalg.norm(pt_shoulder_l - pt_shoulder_r))
    arm_length_m = float(np.linalg.norm(pt_shoulder_l - pt_wrist_l))
    leg_inseam_m = float(abs(pt_crotch[1] - pt_heel_l[1]))

    # Chuẩn hóa tỉ lệ (Scale Factor) nếu có chiều cao thực tế
    scale_factor = (height_cm_input / 100.0) / est_height_m if has_height else 1.0

    final_mass_kg = weight_kg_input if has_weight else (est_mass_kg * (scale_factor ** 3))

    if save_mesh_path:
        try:
            import trimesh
            scaled_verts = (vertices[0] * scale_factor).detach().cpu().numpy()
            mesh_faces = smpl_x.face.astype(np.int64)
            mesh_obj = trimesh.Trimesh(vertices=scaled_verts, faces=mesh_faces, process=False)
            mesh_obj.export(save_mesh_path)
            print(f"[✓] Đã xuất thành công mô hình 3D avatar (Mannequin 3D) vào: {save_mesh_path}")
        except Exception as e:
            print(f"[!] Không thể xuất file 3D mesh: {e}")

    return {
        "thong_tin_nhap": {
            "gioi_tinh": active_gender,
            "chieu_cao_nhap_cm": height_cm_input,
            "can_nang_nhap_kg": weight_kg_input,
            "scale_factor_hieu_chinh": round(float(scale_factor), 4)
        },
        "so_do_may_mac_cm": {
            "chieu_cao": round(est_height_m * scale_factor * 100.0, 1),
            "vong_nguc": round(est_chest_m * scale_factor * 100.0, 1),
            "vong_eo": round(est_waist_m * scale_factor * 100.0, 1),
            "vong_hong": round(est_hips_m * scale_factor * 100.0, 1),
            "rong_vai": round(shoulder_width_m * scale_factor * 100.0, 1),
            "dai_tay_ao": round(arm_length_m * scale_factor * 100.0, 1),
            "dai_chan_inseam": round(leg_inseam_m * scale_factor * 100.0, 1),
            "can_nang_uoc_tinh_kg": round(float(final_mass_kg), 1)
        },
        "smplx_betas": [round(float(x), 5) for x in betas.detach().cpu().numpy().tolist()[0]]
    }


def main():
    args = parse_args()
    img_path = osp.abspath(args.image)
    if not osp.exists(img_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {img_path}")

    print("\n=======================================================")
    print("   HỆ THỐNG TRÍCH XUẤT SỐ ĐO MAY MẶC TỰ ĐỘNG (ISO 8559)")
    print("=======================================================")
    print(f"Ảnh đầu vào: {img_path}")

    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Không thể đọc file ảnh: {img_path}")

    detector = load_detector(args.device)
    box_xywh = detect_person(detector, img_path)
    print(f"[*] Đã phát hiện vị trí người (BBox: {box_xywh.astype(int).tolist()})")

    demoer = load_smpler_x_model(args.pretrained_model, args.device)
    betas, _ = extract_3d_mesh(demoer, original_img, box_xywh, args.device)
    print("[*] Đã tái tạo thành công hình thể 3D SMPL-X.")

    print(f"[*] Đang tính toán thiết diện cắt ngang và khoảng cách giải phẫu (Giới tính: {args.gender})...")
    results = compute_measurements(
        betas,
        height_cm_input=args.height_cm,
        weight_kg_input=args.weight_kg,
        gender=args.gender,
        device=args.device,
        save_mesh_path=args.save_mesh
    )

    sd = results["so_do_may_mac_cm"]
    tt = results["thong_tin_nhap"]
    print("\n" + "=" * 50)
    print("     KẾT QUẢ SỐ ĐO MAY MẶC (ĐƠN VỊ: CM / KG)")
    print("=" * 50)
    print(f"  * Giới tính mô hình    : {tt['gioi_tinh'].upper()}")
    print(f"  1. Chiều cao toàn thân : {sd['chieu_cao']} cm")
    print(f"  2. Vòng ngực           : {sd['vong_nguc']} cm")
    print(f"  3. Vòng eo             : {sd['vong_eo']} cm")
    print(f"  4. Vòng hông / mông    : {sd['vong_hong']} cm")
    print(f"  5. Chiều rộng vai      : {sd['rong_vai']} cm")
    print(f"  6. Chiều dài tay áo    : {sd['dai_tay_ao']} cm")
    print(f"  7. Chiều dài chân trong: {sd['dai_chan_inseam']} cm (Đáy quần -> gót)")
    print(f"  8. Cân nặng ước tính   : {sd['can_nang_uoc_tinh_kg']} kg")
    print("=" * 50)

    out_json = osp.abspath(args.output_json)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"[✓] Đã lưu kết quả chi tiết vào: {out_json}\n")


if __name__ == '__main__':
    main()
