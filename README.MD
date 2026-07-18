# Telco Customer Churn Prediction - Explainable AI and Retention Personas

Dự án này xây dựng một pipeline giải thích được để dự đoán khách hàng rời mạng trong ngành viễn thông, sau đó chuyển kết quả giải thích thành persona và khuyến nghị giữ chân có thể hành động.

- Tên người làm: Vũ Trung Kiên
- Mã học viên: 250104011
- Chuyên đề: IS6502.CH201
- Giảng viên hướng dẫn: Thầy Nguyễn Đình Thuân

## Tóm tắt ngắn
Quy trình nghiên cứu gồm 3 tầng:
1. Dự đoán churn bằng nhiều mô hình học máy.
2. Giải thích mô hình bằng SHAP.
3. Gom cụm theo SHAP theme để tạo persona và gợi ý chiến lược giữ chân.

Điểm nhấn của đề tài là không dừng ở accuracy hay F1-score, mà kết nối trực tiếp:
`prediction -> explanation -> retention action`.

## Kết quả mới nhất
- Bộ dữ liệu: 7.043 khách hàng, churn rate 26,54%.
- Mô hình so sánh tốt nhất theo accuracy: Logistic Regression với 0.7903.
- Mô hình nền tảng cho giải thích: Random Forest với accuracy 0.7846, F1-score 0.5524, AUC 0.8148.
- Đặc trưng quan trọng nhất theo SHAP: `Contract`, sau đó là `tenure`, `OnlineSecurity`, `TechSupport`, `TotalCharges`.
- Ablation study tốt nhất: `shap_theme`, silhouette 0.3157, cải thiện 5.16% so với `raw_feature`.
- Persona cuối cùng: 5 cụm với kích thước 230, 582, 237, 225, 133.

## Minh họa kết quả
### SHAP summary
![SHAP Summary](reports/figures/shap_summary.png)

### Ablation study
![Ablation Study](reports/figures/ablation_raw_vs_shap.png)

## Cấu trúc thư mục chính
- `data/`: dữ liệu gốc và dữ liệu train/test đã tiền xử lý.
- `src/`: mã nguồn tiền xử lý, huấn luyện và giải thích.
- `reports/`: bảng kết quả, persona và hình SHAP.
- `docs/`: tài liệu mô tả dự án.
- `latex/`: bản thảo paper tiếng Việt và tiếng Anh.

## Các file quan trọng
- `latex/main_vn.tex`: paper tiếng Việt.
- `latex/main.tex`: paper tiếng Anh.
- `reports/model_comparison.csv`: bảng so sánh mô hình.
- `reports/shap_feature_importance.csv`: top feature theo SHAP.
- `reports/persona_retention_recommendations.csv`: persona và khuyến nghị giữ chân.
- `reports/presentation_summary.md`: tóm tắt trình bày.

## Chạy nhanh
```bash
pip install -r requirements.txt
python demo.py
```

### Chạy từng bước
```bash
python src/preprocess.py
python src/model.py
python src/explain.py
```

## Đầu ra sinh ra
- `data/X_train.csv`, `data/X_test.csv`, `data/y_train.csv`, `data/y_test.csv`
- `src/best_model.pkl`
- `reports/model_comparison.csv`
- `reports/figures/shap_summary.png`
- `reports/figures/ablation_raw_vs_shap.png`
- `reports/customer_personas_shap.csv`
- `reports/cluster_shap_mean.csv`
- `reports/persona_retention_recommendations.csv`

## Ghi chú
Bản thảo paper đã được đồng bộ theo kết quả mới nhất trong `reports/`. Nếu bạn cần phần trình bày, xem thêm `reports/presentation_summary.md`.
