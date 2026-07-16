# Tóm tắt trình bày đồ án cuối kỳ

## Đề tài
Explainable AI for Telecom Customer Churn Prediction and Persona-Based Retention.

## Mục tiêu
- Dự đoán khách hàng rời mạng.
- Giải thích nguyên nhân churn bằng SHAP.
- Xây dựng persona khách hàng và đề xuất hành động giữ chân.

## Phương pháp chính
1. Tiền xử lý dữ liệu và chia train/test.
2. So sánh nhiều mô hình phân loại trên tập kiểm tra.
3. Dùng Random Forest làm mô hình nền tảng cho SHAP.
4. Tính SHAP values, xếp hạng feature importance và phân tích theo theme.
5. Gom cụm bằng K-Means trên ba không gian: raw feature, SHAP value, SHAP theme.
6. Kiểm định thống kê bằng Kruskal-Wallis với effect size.

## Kết quả nổi bật
- Accuracy tốt nhất: 0.7903 với Logistic Regression.
- Mô hình nền tảng cho giải thích: Random Forest, accuracy 0.7846, F1-score 0.5524, AUC 0.8148.
- SHAP feature quan trọng nhất: Contract, tiếp theo là tenure, OnlineSecurity, TechSupport, TotalCharges.
- Ablation tốt nhất: SHAP theme, silhouette 0.3157, cải thiện 5.16% so với raw feature.
- Persona thu được: 5 cụm với kích thước 230, 582, 237, 225 và 133.
- Các theme đều có khác biệt thống kê có ý nghĩa giữa các persona.

## Điểm mới
- Không chỉ dự đoán churn mà còn giải thích và chuyển thành hành động kinh doanh.
- Kết hợp prediction, explanation và retention action trong một pipeline.

## Minh họa từ paper
### SHAP summary
![SHAP summary](figures/shap_summary.png)

### Ablation study
![Ablation study](figures/ablation_raw_vs_shap.png)
