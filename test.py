import cv2
import numpy as np
import os
import pytesseract

# CHÚ Ý: Cập nhật đường dẫn tesseract.exe trên máy của bạn (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def segment_and_ocr_mono(paragraph_img_path, output_dir="paragraph_chars", output_txt_path="ketqua_ocr.txt"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(paragraph_img_path)
    if img is None:
        print(f"Không thể mở ảnh tại {paragraph_img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Nhị phân hóa ảnh: Chữ trắng, nền đen (Dùng để tìm contours/bounding box)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Tạo một bản lật ngược: Chữ đen, nền trắng (Để đưa vào OCR)
    thresh_for_ocr = cv2.bitwise_not(thresh)
    
    h_doc, w_doc = thresh.shape

    # ========================================================
    # BƯỚC 1: TÁCH ĐOẠN THÀNH TỪNG DÒNG (Line Segmentation)
    # ========================================================
    h_proj = np.sum(thresh, axis=1)
    
    lines = []
    in_line = False
    start_y = 0
    
    for y in range(h_doc):
        if h_proj[y] > 0 and not in_line:
            in_line = True
            start_y = y
        elif h_proj[y] == 0 and in_line:
            in_line = False
            if y - start_y > 8:
                lines.append((start_y, y))
                
    if in_line:
        lines.append((start_y, h_doc))

    print(f"-> Tìm thấy {len(lines)} dòng trong đoạn văn.\n")

    # ========================================================
    # BƯỚC 2: CẮT TỪNG KÝ TỰ & TIỀN XỬ LÝ OCR
    # ========================================================
    total_chars = 0
    full_text = ""

    for line_idx, (y1, y2) in enumerate(lines):
        pad_y = 3
        y_min = max(0, y1 - pad_y)
        y_max = min(h_doc, y2 + pad_y)
        
        line_thresh = thresh[y_min:y_max, :]
        line_ocr_src = thresh_for_ocr[y_min:y_max, :] # Dùng ảnh chữ đen nền trắng
        
        h_line, w_line = line_thresh.shape

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_thresh, connectivity=8)
        
        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 8:  
                components.append({'bbox': (x, y, w, h), 'cx': centroids[i][0]})

        if not components:
            continue

        components = sorted(components, key=lambda c: c['bbox'][0])

        char_boxes = []
        visited = set()

        for i in range(len(components)):
            if i in visited:
                continue
            x1_c, y1_c, w1_c, h1_c = components[i]['bbox']
            cx1, cy1, cx2, cy2 = x1_c, y1_c, x1_c + w1_c, y1_c + h1_c
            visited.add(i)

            for j in range(i + 1, len(components)):
                if j in visited:
                    continue
                x2_c, y2_c, w2_c, h2_c = components[j]['bbox']
                
                overlap_x = max(0, min(cx2, x2_c + w2_c) - max(cx1, x2_c))
                if overlap_x / min(w1_c, w2_c) > 0.35:
                    cx1 = min(cx1, x2_c)
                    cy1 = min(cy1, y2_c)
                    cx2 = max(cx2, x2_c + w2_c)
                    cy2 = max(cy2, y2_c + h2_c)
                    visited.add(j)

            char_boxes.append((cx1, cy1, cx2 - cx1, cy2 - cy1))

        char_boxes = sorted(char_boxes, key=lambda b: b[0])
        avg_w = sum([b[2] for b in char_boxes]) / len(char_boxes) if char_boxes else 0

        # ====================================================
        # BƯỚC 3: XỬ LÝ ẢNH CHỮ, LƯU VÀ CHẠY OCR
        # ====================================================
        line_folder = os.path.join(output_dir, f"line_{line_idx+1:02d}")
        os.makedirs(line_folder, exist_ok=True)
        
        line_text = ""
        prev_x_end = None
        
        # psm 10: Nhận diện 1 ký tự, psm 7: Nhận diện 1 dòng (với ảnh 1 chữ đôi khi psm 7/8 chạy ổn hơn)
        tesseract_config = r'--oem 3 --psm 10' 

        for char_idx, (x, y, w, h) in enumerate(char_boxes):
            if prev_x_end is not None:
                gap = x - prev_x_end
                if gap > avg_w * 0.5: 
                    line_text += " "

            pad = 2
            c_ymin = max(0, y - pad)
            c_ymax = min(h_line, y + h + pad)
            c_xmin = max(0, x - pad)
            c_xmax = min(w_line, x + w + pad)

            # 1. Cắt ảnh chữ đen nền trắng
            char_crop = line_ocr_src[c_ymin:c_ymax, c_xmin:c_xmax]
            
            # 2. Phóng to ảnh lên gấp 3 lần (Upscale) để Tesseract dễ đọc
            char_crop = cv2.resize(char_crop, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            
            # 3. Thêm viền trắng (Padding) dày xung quanh chữ
            char_crop = cv2.copyMakeBorder(char_crop, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

            char_filename = os.path.join(line_folder, f"char_{char_idx+1:03d}.png")
            cv2.imwrite(char_filename, char_crop)
            
            # Nhận diện OCR
            try:
                char_str = pytesseract.image_to_string(char_crop, config=tesseract_config, lang='vie')
                char_str = char_str.strip()
                if char_str:
                    # Tesseract đôi khi trả về chuỗi dài do nhiễu, ta chỉ lấy ký tự đầu tiên
                    line_text += char_str[0]
            except Exception as e:
                pass
            
            prev_x_end = x + w
            total_chars += 1
            
        full_text += line_text + "\n"

    # ========================================================
    # BƯỚC 4: LƯU VĂN BẢN RA FILE TXT
    # ========================================================
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Hoàn thành! Đã xử lý {total_chars} ký tự từ {len(lines)} dòng.")
    print(f"File text lưu tại: {output_txt_path}")
    print("=" * 40)
    print("NỘI DUNG VĂN BẢN:")
    print("-" * 40)
    print(full_text)

if __name__ == "__main__":
    segment_and_ocr_mono("test_pdf/font-typewriter-SVN-Erika.jpg")