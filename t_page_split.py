import os
import sys
import re
import time
import argparse
import numpy as np
from PIL import Image

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from module.seeit import draw_box
from module.ocr import OCR
from module import init_in_out

# Cấu hình thiết bị chạy CPU/GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def filter_target_boxes(raw_bxs, crop_y_offset=0):
    """Lọc đúng 2 trường: Số hiệu và Tiêu ngữ / Quốc hiệu"""
    patterns = {
        "so_hieu": r"(SỐ|SO)\s*[:.]?\s*[\d\w/_\-]+",
        "tieu_ngu": r"(ĐỘC\s*LẬP|DOC\s*LAP|HẠNH\s*PHÚC|HANH\s*PHUC|CỘNG\s*HÒA|CONG\s*HOA)"
    }
    target_boxes = []

    for line in raw_bxs:
        coords, (text, score) = line
        text_clean = text.strip().upper()

        # Bù tọa độ Y theo ảnh gốc
        bbox = [
            coords[0][0],
            coords[0][1] + crop_y_offset,
            coords[1][0],
            coords[-1][1] + crop_y_offset
        ]

        for label, pattern in patterns.items():
            if re.search(pattern, text_clean):
                target_boxes.append({
                    "text": text,
                    "bbox": bbox,
                    "type": label,
                    "score": score
                })
                break

    return target_boxes

def main(args):
    ocr = OCR()
    images, outputs = init_in_out(args)

    if not images:
        print("Không tìm thấy ảnh hoặc trang PDF nào để xử lý!")
        return

    # Chỉ lấy 1 trang/ảnh đầu tiên để kiểm tra
    img = images[0]
    out_path = outputs[0]

    print("=== BẮT ĐẦU TEST 1 ẢNH ===")
    start_time = time.time()
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # 1. Cắt 20% phần đầu trang (Header) để tăng tốc và tránh bắt nhầm phần thân
    header_h = int(h * 0.20)
    header_crop = img_np[0:header_h, 0:w]

    # 2. Chạy OCR chỉ trên vùng Header
    raw_bxs = ocr(header_crop, 0)

    # 3. Lọc lấy các khung khớp Số hiệu & Tiêu ngữ
    target_bxs = filter_target_boxes(raw_bxs, crop_y_offset=0)

    # 4. Hiển thị kết quả kiểm tra
    print("\n--- KẾT QUẢ NHẬN DIỆN ---")
    detected_labels = [b["type"] for b in target_bxs]
    
    if "tieu_ngu" in detected_labels or "so_hieu" in detected_labels:
        print("=> ĐÂY LÀ: [TRANG ĐẦU CÔNG VĂN]")
    else:
        print("=> ĐÂY LÀ: [TRANG NỐI TIẾP / KHÔNG PHẢI TRANG ĐẦU]")

    for item in target_bxs:
        print(f"   [{item['type'].upper()}] : '{item['text']}' | BBox: {item['bbox']}")

    # 5. Vẽ khung nhận diện lên ảnh gốc và lưu file
    draw_labels = ["so_hieu", "tieu_ngu"]
    img_result = draw_box(img, target_bxs, draw_labels, 0.5)
    img_result.save(out_path, quality=95)

    elapsed = time.time() - start_time
    print(f"\nThời gian xử lý: {elapsed:.2f} giây")
    print(f"Ảnh kết quả debug đã lưu tại: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True, help="Đường dẫn đến 1 file ảnh (jpg, png) hoặc file PDF")
    parser.add_argument('--output_dir', default="./ocr_outputs", help="Thư mục lưu ảnh kết quả")
    args = parser.parse_args()
    main(args)