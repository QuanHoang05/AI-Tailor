#!/usr/bin/env python3
# ==============================================================================
# evaluate_benchmark.py: Công cụ đối chuẩn & thẩm định sai số giữa AI và Thước dây
# Đánh giá theo tiêu chuẩn may mặc quốc tế ISO 8559 và dung sai may công nghiệp
# ==============================================================================

import os
import sys
import json
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá sai số số đo AI so với Thước dây thực tế")
    parser.add_argument('--pred', type=str, default='output_measurements.json',
                        help="File JSON kết quả từ AI (mặc định: output_measurements.json)")
    parser.add_argument('--real_height', type=float, default=None, help="Chiều cao đo thực tế (cm)")
    parser.add_argument('--real_chest', type=float, default=None, help="Vòng ngực đo thực tế (cm)")
    parser.add_argument('--real_waist', type=float, default=None, help="Vòng eo đo thực tế (cm)")
    parser.add_argument('--real_hips', type=float, default=None, help="Vòng hông/mông đo thực tế (cm)")
    parser.add_argument('--real_shoulder', type=float, default=None, help="Chiều rộng vai đo thực tế (cm)")
    parser.add_argument('--real_sleeve', type=float, default=None, help="Dài tay đo thực tế (cm)")
    parser.add_argument('--real_inseam', type=float, default=None, help="Dài chân trong (inseam) thực tế (cm)")
    parser.add_argument('--real_weight', type=float, default=None, help="Cân nặng thực tế (kg)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.pred):
        print(f"[!] Không tìm thấy file kết quả dự đoán: {args.pred}")
        return

    with open(args.pred, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meas_ai = data.get("so_do_may_mac_cm", data)

    ground_truth = {
        "chieu_cao": ("Chiều cao toàn thân (cm)", args.real_height, 1.5),
        "vong_nguc": ("Vòng ngực (cm)", args.real_chest, 2.0),
        "vong_eo": ("Vòng eo (cm)", args.real_waist, 2.0),
        "vong_hong": ("Vòng hông / mông (cm)", args.real_hips, 2.0),
        "rong_vai": ("Chiều rộng vai (cm)", args.real_shoulder, 1.5),
        "dai_tay_ao": ("Chiều dài tay áo (cm)", args.real_sleeve, 1.5),
        "dai_chan_inseam": ("Dài chân trong (Inseam) (cm)", args.real_inseam, 1.5),
        "can_nang_uoc_tinh_kg": ("Cân nặng ước tính (kg)", args.real_weight, 2.0)
    }

    print("\n" + "=" * 90)
    print("      BẢNG ĐỐI CHUẨN ĐỘ CHÍNH XÁC: AI-TAILOR vs THƯỚC DÂY MAY THỰC TẾ (ISO 8559)")
    print("=" * 90)
    print(f" {'Chỉ số đo':<28} | {'Số đo AI':<10} | {'Thước dây':<10} | {'Lệch (cm/kg)':<14} | {'Sai số %':<10} | {'Đánh giá'}")
    print("-" * 90)

    errors = []
    compared_count = 0

    for key, (label, real_val, tol) in ground_truth.items():
        ai_val = meas_ai.get(key)
        if ai_val is None:
            continue

        if real_val is not None and real_val > 0:
            diff = abs(ai_val - real_val)
            pct = (diff / real_val) * 100.0
            errors.append(diff)
            compared_count += 1

            if diff <= 1.0:
                grade = "[✓✓] XUẤT SẮC"
            elif diff <= tol:
                grade = "[✓] ĐẠT CHUẨN MAY"
            elif diff <= tol + 1.5:
                grade = "[~] CHẤP NHẬN ĐƯỢC"
            else:
                grade = "[✗] CẦN KIỂM TRA"

            diff_str = f"{diff:+.1f} cm" if "kg" not in key else f"{diff:+.1f} kg"
            print(f" {label:<28} | {ai_val:<10.1f} | {real_val:<10.1f} | {diff_str:<14} | {pct:<9.1f}% | {grade}")
        else:
            print(f" {label:<28} | {ai_val:<10.1f} | {'Chưa nhập':<10} | {'-':<14} | {'-':<10} | [i] AI Baseline")

    print("=" * 90)

    if compared_count > 0:
        mae = sum(errors) / compared_count
        print(f"[*] TỔNG KẾT: Đã đối chiếu {compared_count} chỉ số.")
        print(f"[*] Sai số trung bình tuyệt đối (MAE): {mae:.2f} cm")
        if mae <= 1.5:
            print("[✓] KẾT LUẬN: ĐẠT TIÊU CHUẨN THƯƠNG MẠI MAY ĐO (Vượt ngưỡng chấp nhận của ngành may)")
        elif mae <= 2.5:
            print("[✓] KẾT LUẬN: ĐẠT TIÊU CHUẨN THỜI TRANG MAY SẴN & GỢI Ý SIZE E-COMMERCE")
        else:
            print("[!] KẾT LUẬN: Cần rà soát lại tư thế đứng (A-pose) hoặc góc chụp camera")
    else:
        print("[i] Gợi ý: Hãy nhập thêm các cờ số đo thực tế để tính bảng sai số MAE. Ví dụ:")
        print("    python evaluate_benchmark.py --pred output_measurements.json --real_chest 95.0 --real_waist 80.0 --real_inseam 72.0")
    print("=" * 90 + "\n")


if __name__ == '__main__':
    main()
