# Persona-based retention recommendations

## Ablation Study: raw_feature vs shap_value vs shap_theme

- Silhouette tốt nhất (raw_feature, k=3): 0.3002
- Silhouette tốt nhất (shap_value, k=2): 0.3073 (cải thiện 2.36% so với raw_feature)
- Silhouette tốt nhất (shap_theme, k=2): 0.3918 (cải thiện 30.5% so với raw_feature)
- shap_theme được chọn làm không gian cluster chính thức vì cho persona đa dạng chủ đề hơn so với per-feature SHAP-space (vốn bị chi phối bởi đa cộng tuyến giữa Contract/tenure/AvgMonthlyCharge).
- 100.0% theme có khác biệt ý nghĩa thống kê (Kruskal-Wallis, p<0.05) giữa các persona.

Ghi chú quan trọng: các persona hiện tại vẫn trùng chủ đề chi phối dù đã gộp theo theme. Cần cân nhắc tăng k, thêm feature engineering mới, hoặc thu thập thêm dữ liệu hành vi (usage pattern, complaint log) để tăng khả năng phân tách giữa các nhóm.

## Persona 0
- Nhóm: Nhạy cảm chi phí (Cost-sensitive)
- Chủ đề chi phối: Cost_Pressure (Cohen's d = -0.4818, small)
- Feature nổi bật nhất trong chủ đề: TotalCharges
- Diễn giải: Nhóm này bị thúc đẩy mạnh bởi chi phí và áp lực thanh toán.
- Hành động giữ chân: Ưu tiên chương trình giảm phí, voucher hoặc gói cước linh hoạt cho nhóm này.
- Số lượng khách hàng: 961

## Persona 1
- Nhóm: Nhạy cảm chi phí (Cost-sensitive)
- Chủ đề chi phối: Cost_Pressure (Cohen's d = 1.0381, large)
- Feature nổi bật nhất trong chủ đề: TotalCharges
- Diễn giải: Nhóm này bị thúc đẩy mạnh bởi chi phí và áp lực thanh toán.
- Hành động giữ chân: Ưu tiên chương trình giảm phí, voucher hoặc gói cước linh hoạt cho nhóm này.
- Số lượng khách hàng: 446

