import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from scipy.stats import kruskal

def _prepare_shap_values(model, X_test):
    """Compute SHAP values with a tree explainer when possible and fall back to KernelExplainer."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        background = X_test.sample(min(100, len(X_test)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = np.asarray(shap_values[1])
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return np.asarray(shap_values)


def _select_top_features(shap_values, feature_names, top_n=10):
    """Rank features by mean absolute SHAP values."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked_idx = np.argsort(mean_abs)[::-1]
    top_idx = ranked_idx[:top_n]
    return top_idx, [feature_names[i] for i in top_idx], mean_abs[top_idx]


def _save_global_plots(shap_values, X_test, top_idx, top_features, reports_dir):
    """Save clearer SHAP summary plots for paper-style reporting."""
    fig_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values[:, top_idx],
        X_test.iloc[:, top_idx],
        show=False,
        plot_type='dot',
        max_display=len(top_features)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values[:, top_idx],
        X_test.iloc[:, top_idx],
        show=False,
        plot_type='bar',
        max_display=len(top_features)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'shap_bar.png'), bbox_inches='tight')
    plt.close()


def _choose_cluster_count(shap_matrix, min_clusters=2, max_clusters=5):
    """Pick a reasonable number of clusters using silhouette score."""
    if len(shap_matrix) < 2 * min_clusters:
        return 3

    best_k = 3
    best_score = -1.0
    scaler = StandardScaler()
    scaled = scaler.fit_transform(shap_matrix)

    for n_clusters in range(min_clusters, min(max_clusters, len(shap_matrix) - 1) + 1):
        if n_clusters >= len(shap_matrix):
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_score = score
            best_k = n_clusters

    return best_k
# thêm mới đoạn này
def _cluster_and_score(data, n_clusters, random_state=42):
    """Chuẩn hoá dữ liệu, chạy KMeans, trả về nhãn cụm + 2 chỉ số đánh giá độc lập:
    - Silhouette: càng cao càng tốt (cụm compact & tách biệt), range [-1, 1]
    - Davies-Bouldin: càng thấp càng tốt (tỷ lệ within/between-cluster distance)
    Dùng cả 2 chỉ số vì Silhouette có thể bias khi cluster lệch kích thước,
    trong khi Davies-Bouldin nhạy với cluster overlap hơn.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(scaled)
    sil = silhouette_score(scaled, labels)
    db = davies_bouldin_score(scaled, labels)
    return labels, sil, db


def compare_clustering_spaces(X_test, shap_top, top_idx, k_range=range(2, 6)):
    """ABLATION STUDY - đóng góp chính của đề tài.

    So sánh chất lượng phân cụm khách hàng khi thực hiện trên 2 không gian:
    (1) raw_feature : giá trị feature thô sau chuẩn hoá - cách làm PHỔ BIẾN
        trong các nghiên cứu customer segmentation truyền thống.
    (2) shap_value   : đóng góp SHAP của từng feature tới xác suất churn -
        ĐỀ XUẤT của đề tài này, vì nó phản ánh "feature này đẩy khách hàng
        về phía churn/không-churn bao nhiêu" thay vì chỉ là giá trị thô,
        giúp cụm phản ánh đúng NGUYÊN NHÂN hành vi thay vì chỉ đặc điểm nhân khẩu học.

    Với mỗi k trong k_range, chạy KMeans độc lập trên từng không gian,
    ghi nhận đầy đủ Silhouette/Davies-Bouldin để vẽ biểu đồ so sánh theo k,
    đồng thời chọn ra k tốt nhất (theo Silhouette) riêng cho từng không gian
    để so sánh kiểu "best-of-each" công bằng.
    """
    raw_data = X_test.iloc[:, top_idx]

    rows = []
    best = {}

    for space_name, data in [('raw_feature', raw_data), ('shap_value', shap_top)]:
        best[space_name] = None
        for k in k_range:
            if k >= len(data):
                continue
            labels, sil, db = _cluster_and_score(data, k)
            rows.append({
                'Space': space_name,
                'k': k,
                'Silhouette': round(sil, 4),
                'Davies_Bouldin': round(db, 4)
            })
            if best[space_name] is None or sil > best[space_name]['Silhouette']:
                best[space_name] = {
                    'k': k, 'Silhouette': sil,
                    'Davies_Bouldin': db, 'labels': labels
                }

    comparison_df = pd.DataFrame(rows)
    return comparison_df, best


def validate_cluster_separation(feature_df, labels, feature_names, alpha=0.05):
    """Kiểm định Kruskal-Wallis H-test cho từng feature giữa các cụm.

    H0: phân phối feature giống nhau giữa các cụm (cụm KHÔNG thực sự khác biệt
        về feature đó - chỉ là artifact của thuật toán clustering).
    Nếu p-value < alpha => bác bỏ H0 => cụm khác biệt có ý nghĩa thống kê.

    Dùng Kruskal-Wallis (phi tham số) thay vì ANOVA vì SHAP values sau
    chuẩn hoá thường không thỏa giả định phân phối chuẩn / phương sai đồng nhất.
    Đây là bằng chứng ĐỊNH LƯỢNG bắt buộc phải có trong bài báo để chứng minh
    persona không phải là "cụm giả" (như trường hợp 2 persona đều dominant
    bởi cùng 1 feature Contract ở phiên bản pipeline cũ).
    """
    results = []
    df = feature_df.copy()
    df['_cluster'] = labels
    for feat in feature_names:
        groups = [g[feat].values for _, g in df.groupby('_cluster')]
        stat, p = kruskal(*groups)
        results.append({
            'Feature': feat,
            'H_stat': round(stat, 4),
            'p_value': round(p, 6),
            'Significant_at_0.05': p < alpha
        })
    result_df = pd.DataFrame(results)
    pct_significant = round(result_df['Significant_at_0.05'].mean() * 100, 1)
    return result_df, pct_significant


def _save_ablation_plot(comparison_df, reports_dir):
    """Vẽ biểu đồ Silhouette theo k cho 2 không gian, dùng trực tiếp trong
    slide trình bày và phần Kết quả thực nghiệm của bài báo."""
    fig_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for space_name, marker in [('raw_feature', 'o'), ('shap_value', 's')]:
        subset = comparison_df[comparison_df['Space'] == space_name]
        axes[0].plot(subset['k'], subset['Silhouette'], marker=marker, label=space_name)
        axes[1].plot(subset['k'], subset['Davies_Bouldin'], marker=marker, label=space_name)

    axes[0].set_title('Silhouette Score theo k (càng cao càng tốt)')
    axes[0].set_xlabel('Số cụm (k)')
    axes[0].set_ylabel('Silhouette')
    axes[0].legend()

    axes[1].set_title('Davies-Bouldin Index theo k (càng thấp càng tốt)')
    axes[1].set_xlabel('Số cụm (k)')
    axes[1].set_ylabel('Davies-Bouldin')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'ablation_raw_vs_shap.png'), bbox_inches='tight')
    plt.close()
# end thêm mới đoạn này

def _label_cluster(feature_name):
    """Create readable cluster labels from dominant SHAP features."""
    lower = feature_name.lower()
    if any(token in lower for token in ['charge', 'monthly', 'total']):
        return 'Cost-sensitive'
    if 'contract' in lower or 'tenure' in lower:
        return 'Contract/tenure-sensitive'
    if any(token in lower for token in ['internet', 'service', 'tech', 'support']):
        return 'Service-quality-sensitive'
    return 'Behavioral-sensitive'


def perform_shap_and_clustering(data_path, models_dir, reports_dir):
    """Run SHAP analysis and cluster customers based on the most influential SHAP features."""
    print("Loading data and model...")
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))

    print("Calculating SHAP values...")
    shap_values = _prepare_shap_values(model, X_test)

    feature_names = list(X_test.columns)
    top_idx, top_features, importance_scores = _select_top_features(shap_values, feature_names, top_n=10)

    _save_global_plots(shap_values, X_test, top_idx, top_features, reports_dir)

    # print("Performing KMeans clustering on the most important SHAP features...")
    # shap_top = shap_values[:, top_idx]
    # n_clusters = _choose_cluster_count(shap_top)
    # scaler = StandardScaler()
    # scaled = scaler.fit_transform(shap_top)

    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    # clusters = kmeans.fit_predict(scaled)
    
    shap_top = shap_values[:, top_idx]

    # ===== ABLATION STUDY: SHAP-space vs Raw-feature-space clustering =====
    print("Chạy ablation study: SHAP-space vs Raw-feature-space clustering...")
    comparison_df, best = compare_clustering_spaces(X_test, shap_top, top_idx, k_range=range(2, 6))
    comparison_df.to_csv(os.path.join(reports_dir, 'ablation_raw_vs_shap.csv'), index=False)
    _save_ablation_plot(comparison_df, reports_dir)

    improvement_pct = round(
        (best['shap_value']['Silhouette'] - best['raw_feature']['Silhouette'])
        / abs(best['raw_feature']['Silhouette']) * 100, 2
    )
    print(f"Best silhouette - raw_feature: {best['raw_feature']['Silhouette']:.4f} (k={best['raw_feature']['k']})")
    print(f"Best silhouette - shap_value : {best['shap_value']['Silhouette']:.4f} (k={best['shap_value']['k']})")
    print(f"Cải thiện của SHAP-space so với raw-feature-space: {improvement_pct}%")

    # Dùng kết quả tốt nhất từ SHAP-space (đề xuất của đề tài) làm persona chính thức
    n_clusters = best['shap_value']['k']
    clusters = best['shap_value']['labels']

    # ===== VALIDATION: kiểm định thống kê cho cụm SHAP đã chọn =====
    shap_top_df = pd.DataFrame(shap_top, columns=top_features)
    validation_df, pct_significant = validate_cluster_separation(
        shap_top_df, clusters, top_features
    )
    validation_df.to_csv(os.path.join(reports_dir, 'cluster_validation_kruskal.csv'), index=False)
    print(f"{pct_significant}% feature có khác biệt ý nghĩa thống kê (p<0.05) giữa các cụm SHAP.")
    # =========================================================================

    # old here
    X_test_clustered = X_test.copy()
    X_test_clustered['Persona_Cluster'] = clusters
    X_test_clustered.to_csv(os.path.join(reports_dir, 'customer_personas_shap.csv'), index=False)

    print("Extracting persona insights...")
    shap_df = pd.DataFrame(shap_top, columns=top_features)
    shap_df['Persona_Cluster'] = clusters

    cluster_summary = []
    for cluster_id in sorted(shap_df['Persona_Cluster'].unique()):
        subset = shap_df[shap_df['Persona_Cluster'] == cluster_id]
        cluster_mean = subset.drop(columns=['Persona_Cluster']).mean()
        dominant_feature = cluster_mean.abs().sort_values(ascending=False).index[0]
        cluster_summary.append({
            'Persona_Cluster': cluster_id,
            'Cluster_Size': int(len(subset)),
            'Dominant_Feature': dominant_feature,
            'Cluster_Label': _label_cluster(dominant_feature),
            'Mean_SHAP_Contribution': round(float(cluster_mean[dominant_feature]), 4),
        })

    cluster_summary_df = pd.DataFrame(cluster_summary)
    cluster_summary_df.to_csv(os.path.join(reports_dir, 'cluster_shap_mean.csv'), index=False)

    cluster_note = ""
    if cluster_summary_df['Cluster_Label'].nunique() <= 1 and cluster_summary_df['Dominant_Feature'].nunique() <= 1:
        cluster_note = (
            "Ghi chú quan trọng: hai persona hiện tại có xu hướng giống nhau vì tín hiệu churn trong tập dữ liệu chủ yếu bị chi phối bởi cùng một nhóm đặc trưng liên quan đến hợp đồng và thời gian sử dụng. "
            "Trong thực tế, điều này cho thấy phân tách giữa các nhóm chưa đủ rõ ràng, nên cần thêm dữ liệu, đặc trưng mới hoặc điều chỉnh cách gom cụm để tăng tính phân biệt."
        )

    persona_recommendations = []
    for _, row in cluster_summary_df.iterrows():
        dominant_feature = row['Dominant_Feature']
        dominant_lower = dominant_feature.lower()
        if any(token in dominant_lower for token in ['monthlycharges', 'totalcharges', 'avgmonthlycharge', 'charge']):
            reason = 'Nhóm này bị thúc đẩy mạnh bởi chi phí và áp lực thanh toán.'
            action = 'Ưu tiên chương trình giảm phí, voucher hoặc gói cước linh hoạt cho nhóm này.'
        elif 'contract' in dominant_lower or 'tenure' in dominant_lower:
            reason = 'Nhóm này nhạy cảm với thời hạn cam kết và sự ổn định lâu dài.'
            action = 'Đề xuất khuyến mãi ký hợp đồng dài hạn và chương trình giữ chân theo giai đoạn.'
        elif any(token in dominant_lower for token in ['internet', 'service', 'support', 'tech']):
            reason = 'Nhóm này bị ảnh hưởng bởi chất lượng dịch vụ và trải nghiệm kỹ thuật.'
            action = 'Tăng cường hỗ trợ kỹ thuật, cải thiện chất lượng mạng và chăm sóc sau bán hàng.'
        else:
            reason = 'Nhóm này thể hiện dấu hiệu rời mạng đa chiều và hành vi không ổn định.'
            action = 'Triển khai chương trình chăm sóc cá nhân hóa và nhắc nhở kịp thời.'

        persona_recommendations.append({
            'Persona_Cluster': row['Persona_Cluster'],
            'Cluster_Label': row['Cluster_Label'],
            'Dominant_Feature': dominant_feature,
            'Interpretation': reason,
            'Retention_Action': action,
            'Cluster_Size': row['Cluster_Size'],
            'Mean_SHAP_Contribution': row['Mean_SHAP_Contribution'],
        })

    persona_recommendations_df = pd.DataFrame(persona_recommendations)
    persona_recommendations_df.to_csv(
        os.path.join(reports_dir, 'persona_retention_recommendations.csv'), index=False
    )

    with open(os.path.join(reports_dir, 'persona_retention_recommendations.md'), 'w', encoding='utf-8') as fh:
        fh.write('# Persona-based retention recommendations\n\n')
        fh.write('## Ablation Study: SHAP-space vs Raw-feature-space Clustering\n\n')
        fh.write(f"- Silhouette tốt nhất (raw feature space, k={best['raw_feature']['k']}): "
                 f"{best['raw_feature']['Silhouette']:.4f}\n")
        fh.write(f"- Silhouette tốt nhất (SHAP value space, k={best['shap_value']['k']}): "
                 f"{best['shap_value']['Silhouette']:.4f}\n")
        fh.write(f"- Mức cải thiện: {improvement_pct}%\n")
        fh.write(f"- {pct_significant}% feature quan trọng có khác biệt ý nghĩa thống kê "
                 f"(Kruskal-Wallis, p<0.05) giữa các persona trên SHAP space.\n\n")
        if cluster_note:
            fh.write(f"{cluster_note}\n\n")
        for _, row in persona_recommendations_df.iterrows():
            fh.write(f"## Persona {int(row['Persona_Cluster'])}\n")
            fh.write(f"- Nhóm: {row['Cluster_Label']}\n")
            fh.write(f"- Đặc trưng chi phối: {row['Dominant_Feature']}\n")
            fh.write(f"- Diễn giải: {row['Interpretation']}\n")
            fh.write(f"- Hành động giữ chân: {row['Retention_Action']}\n\n")

    importance_df = pd.DataFrame({
        'Feature': top_features,
        'MeanAbsSHAP': importance_scores,
        'Rank': range(1, len(top_features) + 1)
    })
    importance_df.sort_values('MeanAbsSHAP', ascending=False).to_csv(
        os.path.join(reports_dir, 'shap_feature_importance.csv'), index=False
    )

    print(f"Selected cluster count: {n_clusters}")
    print("SHAP analysis & clustering completed!")
    print(f"Reports and figures saved at: {reports_dir}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'src')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

    perform_shap_and_clustering(DATA_PATH, MODELS_DIR, REPORTS_DIR)
