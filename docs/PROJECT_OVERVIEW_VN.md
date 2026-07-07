# Chuyên đề nghiên cứu: Phân tích dữ liệu khách hàng rời mạng viễn thông

- Tên người làm: Vũ Trung Kiên
- Mã học viên: 250104011
- Chuyên đề: Phân tích dữ liệu - IS6502.CH201
- Giảng viên hướng dẫn: Thầy Nguyễn Đình Thuận

## Mục tiêu nghiên cứu
Dự án này xây dựng quy trình phân tích dữ liệu và học máy nhằm dự đoán khách hàng rời mạng (customer churn) trong ngành viễn thông. Ngoài việc dự đoán, hệ thống còn tích hợp Explainable AI (SHAP) và phân cụm để làm rõ nguyên nhân và phân nhóm hành vi khách hàng, đồng thời chuyển những hiểu biết này thành các đề xuất hành động giữ chân.

## Điểm mới của đề tài
Đề tài không dừng ở việc huấn luyện một mô hình phân loại. Điểm mới nằm ở việc xây dựng một framework hybrid gồm ba tầng:
1. Prediction layer: dự đoán khả năng rời mạng bằng mô hình Random Forest.
2. Explanation layer: dùng SHAP để xác định các đặc trưng có ảnh hưởng lớn nhất đối với quyết định rời mạng.
3. Action layer: nhóm khách hàng theo giá trị SHAP bằng K-Means để tạo persona và từ đó đề xuất chính sách giữ chân phù hợp.

Nhờ vậy, kết quả nghiên cứu không chỉ là accuracy hay F1-score, mà còn có ý nghĩa kinh doanh và có thể sử dụng cho bài báo, hội nghị và đề xuất chiến lược thực tiễn. Đây là một đóng góp có giá trị vì kết nối trực tiếp việc dự đoán với việc giải thích và tối ưu chính sách giữ chân khách hàng.

## Luồng xử lý chính
1. Tiền xử lý dữ liệu: xử lý giá trị thiếu, mã hóa dữ liệu phân loại, tạo đặc trưng mới.
2. Huấn luyện mô hình: so sánh nhiều mô hình học máy và chọn mô hình phù hợp cho phân tích giải thích.
3. Giải thích mô hình: dùng SHAP để phân tích ảnh hưởng của từng đặc trưng.
4. Gom cụm personas: nhóm khách hàng theo đặc trưng nguyên nhân rời mạng.

## Cấu trúc thư mục chính
- data/: tập dữ liệu và các file train/test đã tiền xử lý.
- src/: mã nguồn tiền xử lý, huấn luyện, giải thích.
- reports/: bảng kết quả và hình ảnh SHAP.
- docs/: tài liệu mô tả dự án.
- latex/: mẫu LaTeX dùng cho bài báo hội nghị.

## Hướng dẫn chạy nhanh
```bash
pip install -r requirements.txt
python src/demo.py
```

## Kết quả mong đợi
Sau khi chạy xong, hệ thống sẽ tạo ra:
- các file dữ liệu train/test trong thư mục data/
- file mô hình tốt nhất tại src/best_model.pkl
- các báo cáo SHAP và hình ảnh trong reports/
