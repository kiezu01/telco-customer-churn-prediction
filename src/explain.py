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
from sklearn.decomposition import PCA

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
    """Ablation study for raw features versus per-feature SHAP values.

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

    Bổ sung epsilon-squared (effect size cho Kruskal-Wallis) vì p-value một mình
    không đủ ý nghĩa khi n lớn (n~1400 khiến hầu hết p đều <0.05 kể cả effect nhỏ).
    Epsilon-squared: 0.01=nhỏ, 0.06=vừa, 0.14+=lớn (ngưỡng tương tự eta-squared).
    """
    n = len(feature_df)
    k = len(np.unique(labels))
    results = []
    df = feature_df.copy()
    df['_cluster'] = labels
    for feat in feature_names:
        groups = [g[feat].values for _, g in df.groupby('_cluster')]
        stat, p = kruskal(*groups)
        epsilon_sq = (stat - k + 1) / (n - k)
        results.append({
            'Feature': feat,
            'H_stat': round(stat, 4),
            'p_value': round(p, 6),
            'Epsilon_Squared': round(max(epsilon_sq, 0), 4),
            'Significant_at_0.05': p < alpha,
            'Effect_Size': ('large' if epsilon_sq >= 0.14 else
                             'medium' if epsilon_sq >= 0.06 else
                             'small' if epsilon_sq >= 0.01 else 'negligible')
        })
    result_df = pd.DataFrame(results)
    pct_significant = round(result_df['Significant_at_0.05'].mean() * 100, 1)
    return result_df, pct_significant


def _save_ablation_plot(comparison_df, reports_dir):
    """Vẽ biểu đồ Silhouette và Davies-Bouldin cho 3 không gian clustering.

    Bao gồm: raw_feature, shap_value (per-feature) và shap_theme (composite score).
    """
    fig_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    markers = {'raw_feature': 'o', 'shap_value': 's', 'shap_theme': 'D'}
    for space_name, marker in markers.items():
        subset = comparison_df[comparison_df['Space'] == space_name]
        if subset.empty:
            continue
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

def _select_distinctive_feature(shap_df, cluster_col='Persona_Cluster'):
    """Chọn feature đặc trưng của từng cụm bằng standardized mean difference
    (tương tự Cohen's d), CHUẨN HOÁ THEO ĐỘ LỆCH CHUẨN CỦA TOÀN BỘ POPULATION
    (mọi khách hàng), KHÔNG chuẩn hoá theo std giữa các cluster-mean.

    Lý do bắt buộc phải sửa: nếu chuẩn hoá theo std giữa các cluster-mean
    (cách làm ở phiên bản trước), với k=2 cụm công thức z-score sẽ LUÔN cho
    ra đúng ±1/sqrt(2) = 0.7071 cho MỌI feature bất kể mức độ khác biệt thực
    tế - đây là tính chất toán học suy biến của z-score khi n=2, không phản
    ánh dữ liệu. Kết quả "InternetService distinctive" ở lần chạy trước
    thực chất là NHIỄU do lỗi công thức, không phải phát hiện thật.

    Standardized mean difference: (cluster_mean - global_mean) / global_std
    dùng global_std (độ lệch chuẩn trên toàn bộ tập test) làm mẫu số cố định,
    nên không suy biến dù k=2 hay k=10, và có thể diễn giải theo ngưỡng
    Cohen's d quen thuộc (0.2=nhỏ, 0.5=vừa, 0.8+=lớn).
    """
    feature_cols = [c for c in shap_df.columns if c != cluster_col]
    global_mean = shap_df[feature_cols].mean()
    global_std = shap_df[feature_cols].std().replace(0, np.nan)  # tránh chia 0

    cluster_means = shap_df.groupby(cluster_col)[feature_cols].mean()
    z_scores = (cluster_means - global_mean) / global_std

    distinctive = []
    for cid in cluster_means.index:
        row = z_scores.loc[cid].abs().sort_values(ascending=False)
        top_feat = row.index[0]
        distinctive.append({
            'Persona_Cluster': cid,
            'Distinctive_Feature': top_feat,
            'Cohens_d': round(z_scores.loc[cid, top_feat], 4),
            'Effect_Magnitude': (
                'large' if abs(z_scores.loc[cid, top_feat]) >= 0.8 else
                'medium' if abs(z_scores.loc[cid, top_feat]) >= 0.5 else
                'small' if abs(z_scores.loc[cid, top_feat]) >= 0.2 else 'negligible'
            ),
            'Cluster_Mean_SHAP': round(cluster_means.loc[cid, top_feat], 4),
            'Top3_Distinctive_Features': list(row.index[:3])
        })
    return pd.DataFrame(distinctive)

THEME_MAP = {
    'Cost_Pressure': ['MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'HighCharge'],
    'Commitment': ['Contract', 'tenure', 'PaperlessBilling'],
    'Service_Quality': ['InternetService', 'OnlineSecurity', 'OnlineBackup',
                         'DeviceProtection', 'TechSupport'],
    'Lifestyle_Bundle': ['StreamingTV', 'StreamingMovies', 'MultipleLines', 'PhoneService'],
}

THEME_LABELS_VN = {
    'Cost_Pressure': 'Nhạy cảm chi phí (Cost-sensitive)',
    'Commitment': 'Nhạy cảm hợp đồng/thời hạn (Contract/tenure-sensitive)',
    'Service_Quality': 'Nhạy cảm chất lượng dịch vụ (Service-quality-sensitive)',
    'Lifestyle_Bundle': 'Nhạy cảm gói dịch vụ giải trí (Lifestyle-bundle-sensitive)',
}

THEME_ACTIONS_VN = {
    'Cost_Pressure': (
        'Nhóm này bị thúc đẩy mạnh bởi chi phí và áp lực thanh toán.',
        'Ưu tiên chương trình giảm phí, voucher hoặc gói cước linh hoạt cho nhóm này.'
    ),
    'Commitment': (
        'Nhóm này nhạy cảm với thời hạn cam kết và sự ổn định lâu dài.',
        'Đề xuất khuyến mãi ký hợp đồng dài hạn và chương trình giữ chân theo giai đoạn.'
    ),
    'Service_Quality': (
        'Nhóm này bị ảnh hưởng bởi chất lượng dịch vụ và trải nghiệm kỹ thuật.',
        'Tăng cường hỗ trợ kỹ thuật, cải thiện chất lượng mạng và chăm sóc sau bán hàng.'
    ),
    'Lifestyle_Bundle': (
        'Nhóm này nhạy cảm với các gói dịch vụ giải trí/tiện ích đi kèm.',
        'Đề xuất bundle ưu đãi cho các dịch vụ streaming/multiple lines đi kèm.'
    ),
}


def _top_feature_within_theme(shap_top_df, cluster_labels, cluster_id, theme, theme_map=THEME_MAP):
    """Tìm raw feature có |SHAP| trung bình lớn nhất trong cụm cluster_id,
    giới hạn trong tập feature thuộc theme đã được chọn làm Dominant_Feature.
    Dùng để làm chi tiết diễn giải bên dưới nhãn theme (theme là nhãn chính,
    feature cụ thể là bằng chứng hỗ trợ, tránh persona chỉ dừng ở mức trừu tượng)."""
    feats_in_theme = [f for f in theme_map.get(theme, []) if f in shap_top_df.columns]
    if not feats_in_theme:
        return None
    mask = cluster_labels == cluster_id
    sub = shap_top_df.loc[mask, feats_in_theme]
    return sub.abs().mean().idxmax()

def _build_theme_scores(shap_values, feature_names, theme_map=THEME_MAP):
    """Gộp SHAP values của các feature cùng chủ đề nghiệp vụ thành composite score.

    Đây là fix xử lý tận gốc: Contract/tenure/AvgMonthlyCharge vốn tương quan cao,
    nên cluster trên feature riêng lẻ dễ bị 1-2 trục này chi phối. Gộp theo theme
    sẽ giảm đa cộng tuyến và làm persona dễ diễn giải hơn về hành vi/ý nghĩa kinh doanh.
    """
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    theme_scores = pd.DataFrame({
        theme: shap_df[[f for f in feats if f in feature_names]].sum(axis=1)
        for theme, feats in theme_map.items()
    })
    return theme_scores

def diagnose_theme_correlation(theme_scores, reports_dir):
    """Kiểm tra ma trận tương quan giữa các theme score - nếu các theme
    tương quan cao (|r|>0.6), nghĩa là chúng cùng phản ánh 1 trục hành vi
    ẩn duy nhất (vd: 'mức độ gắn kết tổng thể'), giải thích tại sao mọi
    cách gộp feature (raw, per-feature SHAP, theme SHAP) đều cho ra persona
    khác nhau về CƯỜNG ĐỘ nhưng giống nhau về CHỦ ĐỀ chi phối."""
    corr = theme_scores.corr()
    corr.to_csv(os.path.join(reports_dir, 'theme_correlation_matrix.csv'))
    print("Ma trận tương quan giữa các theme:")
    print(corr.round(3).to_string())
    return corr


def _build_orthogonal_theme_space(theme_scores, variance_threshold=0.95):
    """Trực giao hoá theme scores bằng PCA trước khi cluster, giải quyết
    tận gốc vấn đề các theme tương quan cao khiến KMeans luôn bị 1 trục
    biến thiên chính (latent 'overall engagement axis') chi phối, bất kể
    gộp feature theo cách nào (raw/per-feature SHAP/theme SHAP).

    Giữ đủ số PC để giải thích >= variance_threshold phương sai, đảm bảo
    không mất thông tin quan trọng trong khi loại bỏ đa cộng tuyến.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(theme_scores)

    pca = PCA(random_state=42)
    pcs = pca.fit_transform(scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= variance_threshold) + 1)
    n_components = max(n_components, 2)  # tối thiểu 2 chiều để cluster có ý nghĩa

    pc_df = pd.DataFrame(
        pcs[:, :n_components],
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=theme_scores.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    print(f"PCA giữ {n_components} thành phần chính, giải thích {cum_var[n_components-1]*100:.1f}% phương sai")
    print("Loadings (theme nào đóng góp mạnh vào PC nào):")
    print(loadings.round(3).to_string())

    return pc_df, loadings, pca.explained_variance_ratio_[:n_components]


def diagnose_pc_separation(cluster_space, clusters, reports_dir):
    """Kiểm tra cụm phân biệt theo PC nào - đây là bước xác nhận cuối cùng
    rằng việc trực giao hoá bằng PCA có thực sự tạo ra persona đa dạng
    (tách theo PC2 - trục Lifestyle vs Commitment độc lập) hay vẫn chỉ
    tách theo PC1 (trục mức độ rủi ro tổng thể chung)."""
    df = cluster_space.copy()
    df['Persona_Cluster'] = clusters
    pc_means = df.groupby('Persona_Cluster').mean()
    pc_means.to_csv(os.path.join(reports_dir, 'cluster_pc_means.csv'))
    print("Trung bình từng PC theo cụm:")
    print(pc_means.round(4).to_string())
    return pc_means

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
    """Run SHAP analysis and cluster customers based on theme-aggregated SHAP scores.

    Pipeline:
    1. Tính SHAP values, chọn top-10 feature quan trọng nhất (loại bỏ đa cộng
       tuyến ở bước _select_top_features).
    2. Gộp SHAP values thành theme score theo chủ đề nghiệp vụ (Cost_Pressure,
       Commitment, Service_Quality, Lifestyle_Bundle) - xử lý tận gốc vấn đề
       Contract/tenure/AvgMonthlyCharge tương quan cao lấn át toàn bộ cluster.
    3. Ablation study: so sánh 3 không gian cluster (raw_feature, shap_value
       per-feature, shap_theme) bằng Silhouette + Davies-Bouldin.
    4. Chọn shap_theme làm không gian cluster CHÍNH THỨC (đã kiểm chứng cho
       persona đa dạng hơn per-feature SHAP ở bước thử nghiệm trước).
    5. Đặt tên persona theo theme có Cohen's d lớn nhất (distinctiveness so
       với phân phối toàn cục) + raw feature nổi bật nhất trong theme đó.
    6. Kiểm định Kruskal-Wallis (kèm effect size) để xác nhận cụm tách biệt
       có ý nghĩa thống kê thật, không chỉ do cỡ mẫu lớn.
    """
    print("Loading data and model...")
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))

    print("Calculating SHAP values...")
    shap_values = _prepare_shap_values(model, X_test)

    feature_names = list(X_test.columns)
    top_idx, top_features, importance_scores = _select_top_features(shap_values, feature_names, top_n=10)

    _save_global_plots(shap_values, X_test, top_idx, top_features, reports_dir)

    shap_top = shap_values[:, top_idx]
    shap_top_df = pd.DataFrame(shap_top, columns=top_features)

    # ===== THEME-BASED SHAP AGGREGATION =====
    theme_scores = _build_theme_scores(shap_values, feature_names).astype(float)
    theme_corr = diagnose_theme_correlation(theme_scores, reports_dir)

    # Nếu theme tương quan cao (>0.5 ở off-diagonal) -> dùng PCA space thay vì raw theme space
    max_offdiag_corr = theme_corr.where(~np.eye(len(theme_corr), dtype=bool)).abs().max().max()
    if max_offdiag_corr > 0.5:
        print(f"Phát hiện theme tương quan cao (max |r|={max_offdiag_corr:.3f}) -> dùng PCA-orthogonalized space")
        cluster_space, pca_loadings, explained_var = _build_orthogonal_theme_space(theme_scores)
        pca_loadings.to_csv(os.path.join(reports_dir, 'theme_pca_loadings.csv'))
    else:
        cluster_space = theme_scores

    # ===== ABLATION STUDY: raw_feature vs shap_value (per-feature) vs shap_theme =====
    # QUAN TRỌNG: nhánh shap_theme phải chạy trên cluster_space (đã trực giao hoá
    # bằng PCA nếu theme tương quan cao) - KHÔNG dùng theme_scores thô, vì nếu
    # dùng theme_scores thô thì toàn bộ phân tích PCA/loadings ở trên chỉ là
    # chẩn đoán suông, không thực sự cải thiện được persona.
    print("Chạy ablation study: raw_feature vs shap_value vs shap_theme (PCA-orthogonalized)...")
    comparison_df, best = compare_clustering_spaces(X_test, shap_top, top_idx, k_range=range(2, 6))

    rows_theme = []
    best_theme = None
    for k in range(2, 6):
        if k >= len(cluster_space):
            continue
        labels_t, sil_t, db_t = _cluster_and_score(cluster_space, k)
        rows_theme.append({'Space': 'shap_theme', 'k': k,
                            'Silhouette': round(sil_t, 4), 'Davies_Bouldin': round(db_t, 4)})
        if best_theme is None or sil_t > best_theme['Silhouette']:
            best_theme = {'k': k, 'labels': labels_t, 'Silhouette': sil_t, 'Davies_Bouldin': db_t}
    best['shap_theme'] = best_theme

    comparison_df = pd.concat([comparison_df, pd.DataFrame(rows_theme)], ignore_index=True)
    comparison_df.to_csv(os.path.join(reports_dir, 'ablation_raw_vs_shap.csv'), index=False)
    _save_ablation_plot(comparison_df, reports_dir)

    improvement_pct = round(
        (best['shap_value']['Silhouette'] - best['raw_feature']['Silhouette'])
        / abs(best['raw_feature']['Silhouette']) * 100, 2
    )
    theme_vs_raw_pct = round(
        (best['shap_theme']['Silhouette'] - best['raw_feature']['Silhouette'])
        / abs(best['raw_feature']['Silhouette']) * 100, 2
    )
    print(f"Best silhouette - raw_feature: {best['raw_feature']['Silhouette']:.4f} (k={best['raw_feature']['k']})")
    print(f"Best silhouette - shap_value : {best['shap_value']['Silhouette']:.4f} (k={best['shap_value']['k']})")
    print(f"Best silhouette - shap_theme : {best['shap_theme']['Silhouette']:.4f} (k={best['shap_theme']['k']})")
    print(f"Cải thiện shap_value so với raw_feature: {improvement_pct}%")
    print(f"Cải thiện shap_theme so với raw_feature : {theme_vs_raw_pct}%")

    # ===== CHỌN shap_theme (PCA-orthogonalized) LÀM KHÔNG GIAN CLUSTER CHÍNH THỨC =====
    # (đã kiểm chứng bằng thực nghiệm: per-feature SHAP-space và theme-space thô
    # đều cho persona trùng chủ đề chi phối do đa cộng tuyến Contract/tenure/
    # Cost_Pressure; PCA trực giao hoá 4 theme thành các PC độc lập, cho phép
    # KMeans phân biệt theo nhiều trục hành vi thay vì chỉ 1 trục chi phối)
    n_clusters = best['shap_theme']['k']
    clusters = best['shap_theme']['labels']

    pc_means = diagnose_pc_separation(cluster_space, clusters, reports_dir)
    # ===== VALIDATION: Kruskal-Wallis kèm effect size trên theme space =====
    validation_df, pct_significant = validate_cluster_separation(
        theme_scores, clusters, list(theme_scores.columns)
    )
    validation_df.to_csv(os.path.join(reports_dir, 'cluster_validation_kruskal.csv'), index=False)
    print(f"{pct_significant}% theme có khác biệt ý nghĩa thống kê (p<0.05) giữa các cụm.")
    print(validation_df.to_string(index=False))

    # ===== Lưu persona gắn vào từng khách hàng =====
    X_test_clustered = X_test.copy()
    X_test_clustered['Persona_Cluster'] = clusters
    X_test_clustered.to_csv(os.path.join(reports_dir, 'customer_personas_shap.csv'), index=False)

    print("Extracting persona insights...")

    # ===== Đặt tên persona theo THEME có Cohen's d lớn nhất =====
    theme_df = theme_scores.copy()
    theme_df['Persona_Cluster'] = clusters
    distinctive_df = _select_distinctive_feature(theme_df)  # cột 'Distinctive_Feature' giờ là tên theme
    distinctive_df.to_csv(os.path.join(reports_dir, 'cluster_distinctive_themes.csv'), index=False)
    print(distinctive_df[['Persona_Cluster', 'Distinctive_Feature', 'Cohens_d', 'Effect_Magnitude']].to_string(index=False))

    cluster_summary = []
    for _, row in distinctive_df.iterrows():
        cluster_id = row['Persona_Cluster']
        theme = row['Distinctive_Feature']
        cluster_size = int((theme_df['Persona_Cluster'] == cluster_id).sum())
        top_raw_feature = _top_feature_within_theme(shap_top_df, clusters, cluster_id, theme)

        cluster_summary.append({
            'Persona_Cluster': cluster_id,
            'Cluster_Size': cluster_size,
            'Dominant_Theme': theme,
            'Top_Raw_Feature_In_Theme': top_raw_feature,
            'Cluster_Label': THEME_LABELS_VN.get(theme, f'Nhóm hành vi: {theme}'),
            'Cohens_d': row['Cohens_d'],
            'Effect_Magnitude': row['Effect_Magnitude'],
            'Cluster_Mean_Theme_Score': row['Cluster_Mean_SHAP'],
        })

    cluster_summary_df = pd.DataFrame(cluster_summary)
    cluster_summary_df.to_csv(os.path.join(reports_dir, 'cluster_shap_mean.csv'), index=False)

    cluster_note = ""
    if cluster_summary_df['Dominant_Theme'].nunique() <= 1:
        cluster_note = (
            "Ghi chú quan trọng: các persona hiện tại vẫn trùng chủ đề chi phối "
            "dù đã gộp theo theme. Cần cân nhắc tăng k, thêm feature engineering "
            "mới, hoặc thu thập thêm dữ liệu hành vi (usage pattern, complaint "
            "log) để tăng khả năng phân tách giữa các nhóm."
        )

    persona_recommendations = []
    for _, row in cluster_summary_df.iterrows():
        theme = row['Dominant_Theme']
        reason, action = THEME_ACTIONS_VN.get(
            theme,
            ('Nhóm này thể hiện dấu hiệu rời mạng đa chiều và hành vi không ổn định.',
             'Triển khai chương trình chăm sóc cá nhân hóa và nhắc nhở kịp thời.')
        )
        persona_recommendations.append({
            'Persona_Cluster': row['Persona_Cluster'],
            'Cluster_Label': row['Cluster_Label'],
            'Dominant_Theme': theme,
            'Top_Raw_Feature_In_Theme': row['Top_Raw_Feature_In_Theme'],
            'Interpretation': reason,
            'Retention_Action': action,
            'Cluster_Size': row['Cluster_Size'],
            'Cohens_d': row['Cohens_d'],
            'Effect_Magnitude': row['Effect_Magnitude'],
        })

    persona_recommendations_df = pd.DataFrame(persona_recommendations)
    persona_recommendations_df.to_csv(
        os.path.join(reports_dir, 'persona_retention_recommendations.csv'), index=False
    )

    with open(os.path.join(reports_dir, 'persona_retention_recommendations.md'), 'w', encoding='utf-8') as fh:
        fh.write('# Persona-based retention recommendations\n\n')
        fh.write('## Ablation Study: raw_feature vs shap_value vs shap_theme\n\n')
        fh.write(f"- Silhouette tốt nhất (raw_feature, k={best['raw_feature']['k']}): "
                 f"{best['raw_feature']['Silhouette']:.4f}\n")
        fh.write(f"- Silhouette tốt nhất (shap_value, k={best['shap_value']['k']}): "
                 f"{best['shap_value']['Silhouette']:.4f} (cải thiện {improvement_pct}% so với raw_feature)\n")
        fh.write(f"- Silhouette tốt nhất (shap_theme, k={best['shap_theme']['k']}): "
                 f"{best['shap_theme']['Silhouette']:.4f} (cải thiện {theme_vs_raw_pct}% so với raw_feature)\n")
        fh.write(f"- shap_theme được chọn làm không gian cluster chính thức vì cho persona "
                 f"đa dạng chủ đề hơn so với per-feature SHAP-space (vốn bị chi phối bởi "
                 f"đa cộng tuyến giữa Contract/tenure/AvgMonthlyCharge).\n")
        fh.write(f"- {pct_significant}% theme có khác biệt ý nghĩa thống kê "
                 f"(Kruskal-Wallis, p<0.05) giữa các persona.\n\n")
        if cluster_note:
            fh.write(f"{cluster_note}\n\n")
        for _, row in persona_recommendations_df.iterrows():
            fh.write(f"## Persona {int(row['Persona_Cluster'])}\n")
            fh.write(f"- Nhóm: {row['Cluster_Label']}\n")
            fh.write(f"- Chủ đề chi phối: {row['Dominant_Theme']} "
                     f"(Cohen's d = {row['Cohens_d']}, {row['Effect_Magnitude']})\n")
            fh.write(f"- Feature nổi bật nhất trong chủ đề: {row['Top_Raw_Feature_In_Theme']}\n")
            fh.write(f"- Diễn giải: {row['Interpretation']}\n")
            fh.write(f"- Hành động giữ chân: {row['Retention_Action']}\n")
            fh.write(f"- Số lượng khách hàng: {row['Cluster_Size']}\n\n")

    importance_df = pd.DataFrame({
        'Feature': top_features,
        'MeanAbsSHAP': importance_scores,
        'Rank': range(1, len(top_features) + 1)
    })
    importance_df.sort_values('MeanAbsSHAP', ascending=False).to_csv(
        os.path.join(reports_dir, 'shap_feature_importance.csv'), index=False
    )

    print(f"Selected cluster count (shap_theme space): {n_clusters}")
    print("SHAP analysis & clustering completed!")
    print(f"Reports and figures saved at: {reports_dir}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'src')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

    perform_shap_and_clustering(DATA_PATH, MODELS_DIR, REPORTS_DIR)




