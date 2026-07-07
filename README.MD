# Telco Customer Churn Prediction - Explainable AI (XAI) & Clustering

Dự án này sử dụng học máy kết hợp với Explainable AI để dự đoán và phân tích nguyên nhân khách hàng rời mạng trong ngành viễn thông.

- Tên người làm: Vũ Trung Kiên
- Mã học viên: 250104011
- Chuyên đề: IS6502.CH201
- Giảng viên hướng dẫn: Thầy Nguyễn Đình Thuận

## 🌟 Tính mới của dự án
Đây không chỉ là một dự án dùng thư viện để dự đoán churn. Nghiên cứu xây dựng một framework mới và có tính ứng dụng cao:
1. Dự đoán khách hàng rời mạng bằng mô hình Random Forest.
2. Dùng SHAP để giải thích tại mức đặc trưng, không chỉ đưa ra điểm số mà còn chỉ ra yếu tố nào làm khách hàng có xu hướng rời mạng.
3. Chuyển các giá trị SHAP thành các persona khách hàng bằng K-Means, rồi từ đó sinh ra các hành động giữ chân có ý nghĩa cho doanh nghiệp.
4. Tạo ra các đầu ra nghiên cứu có thể dùng trực tiếp cho paper: báo cáo SHAP, bảng persona, đề xuất chiến lược giữ chân.

Điểm mới cốt lõi là việc kết nối ba tầng: prediction → explanation → retention action. Đây là phần làm nên giá trị thực sự của đề tài, khác với các bài làm chỉ dừng ở việc huấn luyện mô hình. Nó thể hiện một cách tiếp cận có tính nghiên cứu: từ dự đoán, đến diễn giải, rồi tới đề xuất hành động kinh doanh.

## 📂 Cấu trúc thư mục chính
- src/preprocess.py: tiền xử lý dữ liệu và tạo tập train/test.
- src/model.py: huấn luyện và so sánh nhiều mô hình.
- src/explain.py: tính SHAP và gom cụm personas.
- docs/PROJECT_OVERVIEW_VN.md: tài liệu mô tả nội dung nghiên cứu.
- latex/: mẫu LaTeX cho bài báo hội nghị.

## 🚀 Cài đặt và chạy nhanh
```bash
pip install -r requirements.txt
python demo.py
```

### Chạy từng bước riêng lẻ
```bash
python src/preprocess.py
python src/model.py
python src/explain.py
```

## 📌 Kết quả đầu ra
Sau khi chạy xong, hệ thống sẽ tạo:
- dữ liệu train/test trong thư mục data/
- mô hình tốt nhất tại src/best_model.pkl
- bảng so sánh tại reports/model_comparison.csv
- hình ảnh SHAP tại reports/figures/shap_summary.png
- báo cáo personas tại reports/customer_personas_shap.csv và reports/cluster_shap_mean.csv

## 📝 Ghi chú cho bài báo và hội nghị
Một bộ mẫu LaTeX đã được chuẩn bị tại latex/main.tex để bạn chỉnh sửa tiếp cho bài báo nộp hội nghị.
