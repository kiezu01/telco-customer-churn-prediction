# Thư giải trình (Cover Letter)
**Kính gửi:** Giảng viên phụ trách môn học Phân tích dữ liệu (IS6502.CH201) / Ban biên tập Tạp chí, Hội thảo

**Chủ đề:** Nộp bài báo cáo cuối kỳ: "An Explainable AI Framework for Telecom Customer Churn Prediction and SHAP-Based Retention Personas"

Chúng tôi xin trân trọng gửi đến bài nghiên cứu về việc dự đoán và phân tích nguyên nhân khách hàng rời mạng trong lĩnh vực viễn thông.

**1. Tính mới (Novelty) và Đóng góp của Nghiên cứu:**
Bài báo không chỉ dừng lại ở việc áp dụng các thuật toán học máy để phân lớp nhị phân (Churn / Non-Churn). Điểm nổi bật của nghiên cứu là xây dựng một **khung giải thích kết hợp (Hybrid Explainable Framework)** gồm 3 bước:
- **Tối ưu hóa khả năng dự đoán:** So sánh nhiều mô hình học máy trên bộ dữ liệu Telco Customer Churn và chọn mô hình nền tảng phù hợp cho giải thích.
- **Explainable AI (XAI):** Ứng dụng SHAP để xác định các đặc trưng có ảnh hưởng lớn nhất đến quyết định churn, với `Contract` là yếu tố quan trọng nhất.
- **SHAP-based Persona Clustering:** Gom cụm trên không gian SHAP theme để xác định các persona khách hàng dựa trên cơ chế rời mạng và chuyển thành khuyến nghị giữ chân cụ thể.

Các kết quả mới nhất trong paper cho thấy:
- Logistic Regression đạt accuracy cao nhất: 0.7903.
- Random Forest được dùng làm mô hình nền tảng cho giải thích với accuracy 0.7846, F1-score 0.5524, AUC 0.8148.
- Gom cụm trên SHAP theme đạt silhouette 0.3157 với 5 cụm.
- Kết quả này cho phép xây dựng các chiến lược giữ chân khác nhau cho từng persona thay vì chỉ đưa ra một dự đoán chung.

**2. Tài nguyên đính kèm:**
- **Bài báo học thuật & Slides:** Đính kèm trong hồ sơ nộp.
- **Mã nguồn (Source Code):** Được tổ chức lại theo cấu trúc chuẩn với các bước tiền xử lý, huấn luyện, giải thích và gom cụm.
- **Hình minh họa kết quả:** SHAP summary, ablation study và các bảng persona đã được chuẩn bị trong thư mục `reports/`.

Chúng tôi hy vọng rằng những đóng góp này sẽ đem lại góc nhìn ứng dụng thực tế và khoa học cho lĩnh vực phân tích dữ liệu kinh doanh. Xin chân thành cảm ơn sự xem xét và đánh giá của quý vị.

Trân trọng,
[Tên Nhóm / Tên Học viên]
Học viên Lớp IS6502.CH201
