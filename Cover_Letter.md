# Thư giải trình (Cover Letter)
**Kính gửi:** Giảng viên phụ trách môn học Phân tích dữ liệu (IS6502.CH201) / Ban biên tập Tạp chí, Hội thảo

**Chủ đề:** Nộp bài báo cáo cuối kỳ: "An Explainable Machine Learning Framework for Profiling and Predicting Telecommunication Customer Churn"

Chúng tôi xin trân trọng gửi đến bài nghiên cứu về việc dự đoán và phân tích nguyên nhân khách hàng rời mạng (Customer Churn) trong lĩnh vực viễn thông.

**1. Tính mới (Novelty) và Đóng góp của Nghiên cứu:**
Bài báo không chỉ dừng lại ở việc áp dụng các thuật toán Học máy (Machine Learning) truyền thống để phân lớp nhị phân (Churn / Non-Churn) như phần lớn các tài liệu hướng dẫn trên Kaggle. Điểm đột phá của nghiên cứu nằm ở việc xây dựng một **Khung giải thích kết hợp (Hybrid Explainable Framework)** bao gồm 3 bước:
- **Tối ưu hóa khả năng dự đoán:** Sử dụng mô hình Random Forest (Ensemble Learning) để đạt được độ chính xác và F1-Score tối ưu cho bài toán dữ liệu mất cân bằng.
- **Explainable AI (XAI):** Ứng dụng SHAP (SHapley Additive exPlanations) TreeExplainer nhằm bóc tách tầm quan trọng của từng đặc trưng (feature) ở mức độ vi mô (từng khách hàng) thay vì chỉ đánh giá tổng thể mô hình. Điều này giúp giải quyết bài toán "hộp đen" (black-box) của Học máy.
- **Gom cụm Khách hàng theo SHAP (SHAP-based Persona Clustering):** Đây là đóng góp chính thức và có giá trị nhất. Bằng cách áp dụng thuật toán K-Means lên chính không gian của các giá trị SHAP (không phải không gian đặc trưng ban đầu), nghiên cứu đã xác định được các "Persona" (kiểu người dùng) dựa trên **cơ chế rời mạng của họ**. Thay vì gom cụm khách hàng vì họ "trẻ tuổi" hay "dùng cáp quang", chúng tôi gom cụm khách hàng vì "Họ rời đi do giá cước cao" hoặc "Họ rời đi do dịch vụ kỹ thuật kém".

Từ đó, bài báo đưa ra các đề xuất về chính sách kinh doanh, giữ chân khách hàng (Retention Policies) mang tính cá nhân hóa và sát với thực tế quản trị viễn thông.

**2. Tài nguyên đính kèm:**
- **Bài báo học thuật & Slides:** Đính kèm trong hồ sơ nộp.
- **Mã nguồn (Source Code):** Toàn bộ mã nguồn đã được tổ chức lại theo cấu trúc chuẩn và đẩy lên kho lưu trữ Github (bao gồm Tiền xử lý, Huấn luyện, Giải thích và Gom cụm).

Chúng tôi hy vọng rằng những đóng góp này sẽ đem lại góc nhìn ứng dụng thực tế và khoa học cho lĩnh vực Phân tích dữ liệu kinh doanh. Xin chân thành cảm ơn sự xem xét và đánh giá của quý vị.

Trân trọng,
[Tên Nhóm / Tên Học viên]
Học viên Lớp IS6502.CH201
