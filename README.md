# Text Classification from Legal Documents (YOLO + VietOCR)
Dự án này thực hiện phân loại văn bản hành chính/luật từ file PDF bằng cách:
1. Chuyển PDF sang ảnh
2. Phát hiện tiêu đề văn bản bằng mô hình YOLO
3. Crop tiêu đề thành ảnh riêng biệt
4. Nhận dạng văn bản (OCR) bằng VietOCR
5. Phân loại văn bản dựa trên tập nhãn định nghĩa sẵn (Luật, Nghị định, Thông tư, ...)

# Pipeline xử lý
### 1. Chuyển PDF sang ảnh
* Lặp qua thư mục ```sample/```
* Chuyển trang đầu tiên của mỗi file PDF sang ảnh JPG
* Lưu kết quả tại ```output/img_first_pages/```

### 2. Crop ảnh bằng YOLO
* Dùng mô hình YOLO (title_crop.pt) để phát hiện vùng tiêu đề trong ảnh
* Cắt tiêu đề thành ảnh nhỏ hơn và lưu tại ```output/test/```
* Output ảnh:

    ![output](/output/mophong.png)

### 3. Nhận dạng văn bản
* Dùng mô hình YOLO ```text_det_best.pt``` để detect các vùng chứa chữ
* Nhận dạng text bằng VietOCR
* Lưu kết quả (ảnh có bounding box + text) tại ```output/result_images/```
* Lưu dict chứa text OCR theo từng ảnh

### 4. Phân loại văn bản
* So khớp text OCR với danh sách nhãn (ví dụ: LUẬT, NGHỊ ĐỊNH, THÔNG TƯ...)
* Nếu không khớp, gán vào nhóm "MẶC ĐỊNH"

Ví dụ đầu ra:
```
{
  'LUẬT': ['crop_1.jpg'],
  'NGHỊ ĐỊNH': ['crop_2.jpg'],
  'THÔNG TƯ': ['crop_3.jpg'],
  'VĂN BẢN KHÁC': ['crop_4.jpg', 'crop_5.jpg']
}
```

# Cài đặt
```
pip install ultralytics vietocr pdf2image PyPDF2 numpy==2.0
sudo apt-get install poppler-utils
```
*Cần phải cài poppler để có thể chạy 2 thư viện pdf2image và PyPDF2*

# Các nhãn phân loại 
Danh sách các loại nhãn có sẵn:
```
[
 'HIẾN PHÁP', 'BỘ LUẬT', 'LUẬT', 'PHÁP LỆNH', 'LỆNH',
 'NGHỊ QUYẾT', 'NGHỊ ĐỊNH', 'QUYẾT ĐỊNH', 'THÔNG TƯ',
 'BÁO CÁO', 'CHỈ THỊ', 'KẾ HOẠCH', 'HỢP ĐỒNG',
 'GIẤY MỜI', 'PHIẾU GỬI', 'BẢN SAO Y', 'MẶC ĐỊNH', ...
]
```
