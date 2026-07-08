# Persona-based retention recommendations

## Ablation Study: SHAP-space vs Raw-feature-space Clustering

- Silhouette tốt nhất (raw feature space, k=3): 0.3002
- Silhouette tốt nhất (SHAP value space, k=2): 0.3073
- Mức cải thiện: 2.36%
- 100.0% feature quan trọng có khác biệt ý nghĩa thống kê (Kruskal-Wallis, p<0.05) giữa các persona trên SHAP space.

Ghi chú quan trọng: hai persona hiện tại có xu hướng giống nhau vì tín hiệu churn trong tập dữ liệu chủ yếu bị chi phối bởi cùng một nhóm đặc trưng liên quan đến hợp đồng và thời gian sử dụng. Trong thực tế, điều này cho thấy phân tách giữa các nhóm chưa đủ rõ ràng, nên cần thêm dữ liệu, đặc trưng mới hoặc điều chỉnh cách gom cụm để tăng tính phân biệt.

## Persona 0
- Nhóm: Contract/tenure-sensitive
- Đặc trưng chi phối: Contract
- Diễn giải: Nhóm này nhạy cảm với thời hạn cam kết và sự ổn định lâu dài.
- Hành động giữ chân: Đề xuất khuyến mãi ký hợp đồng dài hạn và chương trình giữ chân theo giai đoạn.

## Persona 1
- Nhóm: Contract/tenure-sensitive
- Đặc trưng chi phối: Contract
- Diễn giải: Nhóm này nhạy cảm với thời hạn cam kết và sự ổn định lâu dài.
- Hành động giữ chân: Đề xuất khuyến mãi ký hợp đồng dài hạn và chương trình giữ chân theo giai đoạn.

