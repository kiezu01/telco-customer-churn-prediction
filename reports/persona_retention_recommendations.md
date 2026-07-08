# Persona-based retention recommendations

## Ablation Study: raw_feature vs shap_value vs shap_theme

- Silhouette tốt nhất (raw_feature, k=3): 0.3002
- Silhouette tốt nhất (shap_value, k=2): 0.3073 (cải thiện 2.36% so với raw_feature)
- Silhouette tốt nhất (shap_theme, k=5): 0.3157 (cải thiện 5.16% so với raw_feature)
- shap_theme được chọn làm không gian cluster chính thức vì cho persona đa dạng chủ đề hơn so với per-feature SHAP-space (vốn bị chi phối bởi đa cộng tuyến giữa Contract/tenure/AvgMonthlyCharge).
- 100.0% theme có khác biệt ý nghĩa thống kê (Kruskal-Wallis, p<0.05) giữa các persona.

## Persona 0
- Nhóm: Nhạy cảm chất lượng dịch vụ (Service-quality-sensitive)
- Chủ đề chi phối: Service_Quality (Cohen's d = -0.9818, large)
- Feature nổi bật nhất trong chủ đề: OnlineSecurity
- Diễn giải: Nhóm này bị ảnh hưởng bởi chất lượng dịch vụ và trải nghiệm kỹ thuật.
- Hành động giữ chân: Tăng cường hỗ trợ kỹ thuật, cải thiện chất lượng mạng và chăm sóc sau bán hàng.
- Số lượng khách hàng: 230

## Persona 1
- Nhóm: Nhạy cảm hợp đồng/thời hạn (Contract/tenure-sensitive)
- Chủ đề chi phối: Commitment (Cohen's d = -0.9883, large)
- Feature nổi bật nhất trong chủ đề: Contract
- Diễn giải: Nhóm này nhạy cảm với thời hạn cam kết và sự ổn định lâu dài.
- Hành động giữ chân: Đề xuất khuyến mãi ký hợp đồng dài hạn và chương trình giữ chân theo giai đoạn.
- Số lượng khách hàng: 582

## Persona 2
- Nhóm: Nhạy cảm chất lượng dịch vụ (Service-quality-sensitive)
- Chủ đề chi phối: Service_Quality (Cohen's d = 0.8349, large)
- Feature nổi bật nhất trong chủ đề: InternetService
- Diễn giải: Nhóm này bị ảnh hưởng bởi chất lượng dịch vụ và trải nghiệm kỹ thuật.
- Hành động giữ chân: Tăng cường hỗ trợ kỹ thuật, cải thiện chất lượng mạng và chăm sóc sau bán hàng.
- Số lượng khách hàng: 237

## Persona 3
- Nhóm: Nhạy cảm gói dịch vụ giải trí (Lifestyle-bundle-sensitive)
- Chủ đề chi phối: Lifestyle_Bundle (Cohen's d = 1.5304, large)
- Feature nổi bật nhất trong chủ đề: None
- Diễn giải: Nhóm này nhạy cảm với các gói dịch vụ giải trí/tiện ích đi kèm.
- Hành động giữ chân: Đề xuất bundle ưu đãi cho các dịch vụ streaming/multiple lines đi kèm.
- Số lượng khách hàng: 225

## Persona 4
- Nhóm: Nhạy cảm chi phí (Cost-sensitive)
- Chủ đề chi phối: Cost_Pressure (Cohen's d = 2.0427, large)
- Feature nổi bật nhất trong chủ đề: TotalCharges
- Diễn giải: Nhóm này bị thúc đẩy mạnh bởi chi phí và áp lực thanh toán.
- Hành động giữ chân: Ưu tiên chương trình giảm phí, voucher hoặc gói cước linh hoạt cho nhóm này.
- Số lượng khách hàng: 133

