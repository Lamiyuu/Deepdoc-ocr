import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import hog
from sklearn.svm import SVC
import joblib

# ========================================================
# 1. BẢNG KÝ TỰ TIẾNG VIỆT ĐẦY ĐỦ CẦN NHẬN DIỆN
# ========================================================
CHARACTERS = (
    # Chữ thường
    "aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệghiíìỉĩịklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvxyýỳỷỹỵ"
    # Chữ hoa
    "AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬBCDĐEÉÈẺẼẸÊẾỀỂỄỆGHIÍÌỈĨỊKLMNOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢPQRSTUÚÙỦŨỤƯỨỪỬỮỰVXYÝỲỶỸỴ"
    # Số và ký tự đặc biệt
    "0123456789.,:;!?-()/\"' "
)

IMG_SIZE = (32, 32)

def extract_hog_features(img_cv2):
    """Tiền xử lý và trích xuất vector đặc trưng HOG"""
    if len(img_cv2.shape) == 3:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv2

    # Nhị phân hóa Otsu để tách rõ nét mực
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Đưa về kích thước chuẩn 32x32
    resized = cv2.resize(thresh, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Tính HOG descriptor
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    return features

# ========================================================
# 2. SINH DỮ LIỆU HUẤN LUYỆN TỰ ĐỘNG BẰNG FONT MONO
# ========================================================
def generate_training_data():
    """Tạo ảnh mẫu giả lập từ các font Mono có sẵn trong máy"""
    print("-> Đang sinh dữ liệu huấn luyện giả lập từ Font...")
    
    # Danh sách font Mono phổ biến (Windows/Linux)
    font_candidates = [
        "C:/Windows/Fonts/cour.ttf",       # Courier New
        "C:/Windows/Fonts/consola.ttf",    # Consolas
        "C:/Windows/Fonts/lucon.ttf",      # Lucida Console
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    ]
    
    available_fonts = [f for f in font_candidates if os.path.exists(f)]
    if not available_fonts:
        available_fonts = [ImageFont.load_default()]
    else:
        available_fonts = [ImageFont.truetype(f, size=24) for f in available_fonts]

    X, y_labels = [], [] # Đổi tên list chứa nhãn thành y_labels để không bị trùng

    for char in CHARACTERS:
        for font in available_fonts:
            # Tạo ảnh 32x32 chứa ký tự
            img = Image.new("RGB", IMG_SIZE, color=(255, 255, 255))
            draw = ImageDraw.Draw(img)

            if isinstance(font, ImageFont.FreeTypeFont):
                bbox = draw.textbbox((0, 0), char, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pos_x = (IMG_SIZE[0] - w) // 2 - bbox[0]  # Đổi thành pos_x
                pos_y = (IMG_SIZE[1] - h) // 2 - bbox[1]  # Đổi thành pos_y
                draw.text((pos_x, pos_y), char, fill=(0, 0, 0), font=font)
            else:
                draw.text((8, 4), char, fill=(0, 0, 0), font=font)

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # 1. Ảnh gốc
            X.append(extract_hog_features(img_cv))
            y_labels.append(char)

            # 2. Data Augmentation: Thêm nhiễu giãn nở nét
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            dilated = cv2.erode(gray, np.ones((2, 2), np.uint8), iterations=1)
            X.append(extract_hog_features(dilated))
            y_labels.append(char)

    return np.array(X), np.array(y_labels)

# ========================================================
# 3. HUẤN LUYỆN MODEL SVM
# ========================================================
def train_svm_classifier(model_save_path="char_svm_model.pkl"):
    # Đảm bảo lấy X, y chuẩn từ hàm trên
    X, y_labels = generate_training_data()
    print(f"-> Đang huấn luyện SVM trên {len(X)} mẫu đặc trưng HOG...")
    
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    clf.fit(X, y_labels)

    joblib.dump(clf, model_save_path)
    print(f"-> Đã lưu mô hình tại '{model_save_path}'!")
    return clf

# ========================================================
# 4. KHÔI PHỤC VĂN BẢN TỪ THƯ MỤC paragraph_chars
# ========================================================
def predict_and_reconstruct(root_dir="paragraph_chars", model_path="char_svm_model.pkl", output_txt="svm_result.txt"):
    # Tải hoặc huấn luyện mô hình
    if os.path.exists(model_path):
        print(f"-> Tải mô hình đã lưu từ {model_path}...")
        clf = joblib.load(model_path)
    else:
        clf = train_svm_classifier(model_path)

    line_dirs = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("line_")
    ])

    if not line_dirs:
        print(f"Không tìm thấy thư mục dòng nào trong '{root_dir}'.")
        return

    full_text_lines = []
    print("\n=== BẮT ĐẦU NHẬN DIỆN KÝ TỰ BẰNG HOG + SVM ===")

    for line_dir in line_dirs:
        line_path = os.path.join(root_dir, line_dir)
        char_files = sorted(glob.glob(os.path.join(line_path, "char_*.png")))

        line_chars = []
        for char_file in char_files:
            img = cv2.imread(char_file)
            if img is None:
                continue

            # Trích xuất vector HOG
            feat = extract_hog_features(img)
            
            # Phân loại ký tự
            pred_char = clf.predict([feat])[0]
            line_chars.append(pred_char)

        line_text = "".join(line_chars)
        full_text_lines.append(line_text)
        print(f"[{line_dir.upper()}]: {line_text}")

    # Ghi kết quả
    full_output = "\n".join(full_text_lines)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(full_output)

    print("\n" + "=" * 50)
    print("KẾT QUẢ KHÔI PHỤC HOÀN CHỈNH:")
    print("=" * 50)
    print(full_output)
    print(f"\nĐã lưu kết quả ra file: {output_txt}")

if __name__ == "__main__":
    predict_and_reconstruct(
        root_dir="paragraph_chars",
        model_path="char_svm_model.pkl",
        output_txt="svm_result.txt"
    )