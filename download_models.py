#!/usr/bin/env python3
# ==============================================================================
# download_models.py: Script kiểm tra và hỗ trợ tải trọng số mô hình cho AI-Tailor
# Tải các checkpoints: SMPLer-X, ViTPose, và mô hình nhân trắc học SMPL-X (10.475 đỉnh)
# ==============================================================================

import os
import sys
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_INFO = [
    {
        "name": "Mô hình SMPLer-X S32 Backbone (384MB)",
        "path": os.path.join(BASE_DIR, "pretrained_models", "smpler_x_s32.pth.tar"),
        "url": "https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_s32.pth.tar",
        "description": "Trọng số chính của mạng SMPLer-X hồi quy khung xương và hình thể 3D"
    },
    {
        "name": "Mô hình ViTPose Small Encoder (97MB)",
        "path": os.path.join(BASE_DIR, "pretrained_models", "vitpose_small.pth"),
        "url": "https://huggingface.co/caizhongang/SMPLer-X/resolve/main/vitpose_small.pth",
        "description": "Mạng Transformer Encoder trích xuất đặc trưng tư thế người"
    },
    {
        "name": "Mô hình SMPL-X Neutral (108MB)",
        "path": os.path.join(BASE_DIR, "common", "utils", "human_model_files", "smplx", "SMPLX_NEUTRAL.npz"),
        "url": "https://smpl-x.is.tue.mpg.de",
        "description": "Lưới 3D nhân trắc học trung tính (yêu cầu giấy phép Max Planck)"
    },
    {
        "name": "Mô hình SMPL-X Male (108MB)",
        "path": os.path.join(BASE_DIR, "common", "utils", "human_model_files", "smplx", "SMPLX_MALE.npz"),
        "url": "https://smpl-x.is.tue.mpg.de",
        "description": "Lưới 3D nhân trắc học nam giới"
    },
    {
        "name": "Mô hình SMPL-X Female (108MB)",
        "path": os.path.join(BASE_DIR, "common", "utils", "human_model_files", "smplx", "SMPLX_FEMALE.npz"),
        "url": "https://smpl-x.is.tue.mpg.de",
        "description": "Lưới 3D nhân trắc học nữ giới"
    }
]


def check_and_report_models():
    print("=" * 75)
    print("   KIỂM TRA TÌNH TRẠNG TRỌNG SỐ MÔ HÌNH AI-TAILOR (CHECKPOINTS AUDIT)")
    print("=" * 75)

    all_exist = True
    missing_models = []

    for item in MODELS_INFO:
        exists = os.path.exists(item["path"])
        size_mb = (os.path.getsize(item["path"]) / (1024 * 1024)) if exists else 0
        status_icon = "[✓]" if exists else "[✗]"
        print(f" {status_icon} {item['name']:<45} : {'SẴN SÀNG (' + f'{size_mb:.1f} MB)' if exists else 'THIẾU'}")
        if not exists:
            all_exist = False
            missing_models.append(item)

    print("=" * 75)
    if all_exist:
        print("[✓] CHÚC MỪNG: Toàn bộ 5 mô hình cốt lõi đã sẵn sàng! Bạn có thể chạy ngay.")
        return True

    print(f"[!] Phát hiện {len(missing_models)} mô hình chưa có trên máy trạm này.")
    print("    Đối với mô hình SMPLer-X & ViTPose: có thể tải trực tiếp từ HuggingFace.")
    print("    Đối với mô hình SMPL-X (*.npz): Vui lòng đăng ký tài khoản miễn phí tại")
    print("    https://smpl-x.is.tue.mpg.de và đặt file vào: common/utils/human_model_files/smplx/")
    print("=" * 75)
    return False


if __name__ == "__main__":
    check_and_report_models()
