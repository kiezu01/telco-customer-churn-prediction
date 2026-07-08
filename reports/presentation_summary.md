# Tóm tắt trình bày đồ án cuối kỳ

## Đề tài
Explainable AI for Telecom Customer Churn Prediction and Persona-Based Retention.

## Mục tiêu
- Dự đoán khách hàng rời mạng.
- Giải thích nguyên nhân churn bằng SHAP.
- Xây dựng persona khách hàng và đề xuất hành động giữ chân.

## Phương pháp chính
1. Tiền xử lý dữ liệu và chia train/test.
2. Huấn luyện mô hình Random Forest.
3. Tính SHAP values và phân tích feature importance.
4. Gom cụm bằng K-Means trên không gian SHAP.
5. So sánh ablation: raw feature, SHAP value, SHAP theme.
6. Kiểm định thống kê bằng Kruskal-Wallis với effect size.

## Kết quả nổi bật
- Accuracy: 0.7846
- F1-score: 0.5524
- AUC: 0.8148
- SHAP feature quan trọng: Contract, tenure, OnlineSecurity, TechSupport.
- Persona được xây dựng từ SHAP và hỗ trợ chiến lược giữ chân.

## Điểm mới
- Không chỉ dự đoán churn mà còn giải thích và chuyển thành hành động kinh doanh.
- Kết hợp prediction, explanation và retention action trong một pipeline.
