import numpy as np
from PyPDF2 import PdfReader,PdfWriter
from pdf2image import convert_from_path
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os
import cv2
from PIL import Image

############### Chuyển các file pdf trong folder sang ảnh ######################

# Folder paths
input_folder = '/content/drive/MyDrive/sample_intern1/sample'
img_output_folder = '/content/drive/MyDrive/sample_intern1/output/img_first_pages'
os.makedirs(img_output_folder, exist_ok=True)

# Loop through PDF files
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.pdf'):
        file_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]

        # Convert first page to image only
        images = convert_from_path(file_path, first_page=1, last_page=1)
        img_output_path = os.path.join(img_output_folder, f'{base_name}_page1.jpg')
        images[0].save(img_output_path, 'JPEG')
        print(f'Converted to image: {img_output_path}')

################################ CROP ẢNH ######################################

# Đường dẫn đến thư mục chứa ảnh
crop_folder = "/content/drive/MyDrive/sample_intern1/output/test/"
os.makedirs(crop_folder, exist_ok=True)

# Tải mô hình YOLO
model_crop = YOLO("/content/drive/MyDrive/sample_intern1/title_crop.pt")

# Biến để đánh số thứ tự các ảnh crop
crop_count = 0

# Lặp qua tất cả các file ảnh trong thư mục
for filename in os.listdir(img_output_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(img_output_folder, filename)
        image = cv2.imread(image_path)
        results = model_crop(image)

        # Duyệt qua tất cả các kết quả dự đoán
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                object_image = image[y1:y2, x1:x2]

                output_filename = os.path.join(crop_folder, f"crop_{crop_count}.jpg")
                cv2.imwrite(output_filename, object_image)
                crop_count += 1

######### Text detection (YOLO) và text recognition (vietocr) ##################

# Load YOLO model 
yolo_model = YOLO("/content/drive/MyDrive/sample_intern1/text_det_best.pt")

# Khởi tạo VietOCR cho recognition
config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained'] = False
config['device'] = 'cuda:0'
vietocr_predictor = Predictor(config)

# Thư mục
result_folder = '/content/drive/MyDrive/sample_intern1/output/result_images'
os.makedirs(result_folder, exist_ok=True)

# Danh sách ảnh
image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(crop_folder) if f.lower().endswith(image_extensions)]

recognized_text_dict = {}

for img_name in image_files:
    img_path = os.path.join(crop_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"[SKIPPED] Không thể đọc ảnh: {img_path}")
        continue

    print(f"[PROCESSING] {img_name}")

    # YOLO detect
    results = yolo_model(img_path)  # Trả về list results
    recognized_text_list = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)  # [x_min, y_min, x_max, y_max]
        for idx, (x_min, y_min, x_max, y_max) in enumerate(boxes):
            # Cắt ảnh
            cropped = image[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                print(f"[SKIP CROP] Empty crop at {img_name} box {idx}")
                continue

            try:
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(cropped_rgb)
                recognized_text = vietocr_predictor.predict(img_pil)
            except Exception as e:
                recognized_text = "[ERROR]"
                print(f"Recognition error: {e}")

            recognized_text_list.append(recognized_text)

            # Vẽ box + text
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(image, recognized_text, (x_min, max(0, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Lưu ảnh kết quả
    result_img_path = os.path.join(result_folder, f"result_{img_name}")
    cv2.imwrite(result_img_path, image)
    recognized_text_dict[img_name] = recognized_text_list
    print(f"[DONE] {img_name} -> {result_img_path}")

################################ PHÂN LOẠI #####################################

van_ban_dict = {
    'HIẾN PHÁP': [],
    'BỘ LUẬT': [],
    'LUẬT': [],
    'PHÁP LỆNH': [],
    'LỆNH': [],
    'NGHỊ QUYẾT': [],
    'NGHỊ QUYẾT LIÊN TỊCH': [],
    'NGHỊ ĐỊNH': [],
    'QUYẾT ĐỊNH': [],
    'THÔNG TƯ': [],
    'THÔNG TƯ LIÊN TỊCH': [],
    'BÁO CÁO': [],
    'CHỈ THỊ': [],
    'QUY CHẾ': [],
    'QUY ĐỊNH': [],
    'THÔNG CÁO': [],
    'THÔNG BÁO': [],
    'HƯỚNG DẪN': [],
    'CHƯƠNG TRÌNH': [],
    'KẾ HOẠCH': [],
    'PHƯƠNG ÁN': [],
    'ĐỀ ÁN': [],
    'DỰ ÁN': [],
    'BIÊN BẢN': [],
    'TỜ TRÌNH': [],
    'HỢP ĐỒNG': [],
    'CÔNG ĐIỆN': [],
    'BẢN GHI NHỚ': [],
    'BẢN THỎA THUẬN': [],
    'GIẤY ỦY QUYỀN': [],
    'GIẤY MỜI': [],
    'GIẤY GIỚI THIỆU': [],
    'GIẤY NGHỈ PHÉP': [],
    'PHIẾU GỬI': [],
    'PHIẾU CHUYỂN': [],
    'PHIẾU BÁO': [],
    'BẢN SAO Y': [],
    'BẢN TRÍCH SAO': [],
    'BẢN SAO LỤC': [],
    'MẶC ĐỊNH': []
}

for img_name, texts in recognized_text_dict.items():
  classified = False
  for text in texts:
      if text.isupper():
        if text in van_ban_dict:
          van_ban_dict[text].append(img_name)
          classified = True
          break
  if not classified:
    van_ban_dict['MẶC ĐỊNH'].append(img_name)