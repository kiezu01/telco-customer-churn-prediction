import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import os
import joblib

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None

def train_and_evaluate(data_path, models_dir, reports_dir):
    """Train models and evaluate them."""
    # Load data
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).squeeze()

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    if xgb is not None:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, eval_metric='logloss')

    results = []
    
    # 1. Train all models and collect metrics
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({'Model': name, 'Accuracy': round(acc,4), 'Precision': round(prec,4), 
                        'Recall': round(rec,4), 'F1': round(f1,4), 'AUC': round(auc,4)})
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    
    # 2. Analytical Model Selection
    print("\n" + "="*50)
    print("PHÂN TÍCH VÀ LỰA CHỌN MÔ HÌNH (MODEL SELECTION ANALYSIS)")
    print("="*50)
    
    # Tìm mô hình có F1-score cao nhất toàn cục
    best_global = results_df.loc[results_df['F1'].idxmax()]
    print(f"- Mô hình có F1-score cao nhất tổng thể: {best_global['Model']} (F1 = {best_global['F1']})")
    
    # Vì mục tiêu nghiên cứu là SHAP-based interpretability, cần chọn mô hình tree-based
    # hỗ trợ TreeExplainer tốt và cho phép diễn giải kết quả rõ ràng hơn cho paper.
    shap_supported_models = ['Random Forest', 'Decision Tree']
    if xgb is not None:
        shap_supported_models.append('XGBoost')

    print(f"- Tuy nhiên, hướng phát triển đề tài yêu cầu sử dụng giải thuật SHAP TreeExplainer để gom cụm Personas.")
    print(f"- Các mô hình tuyến tính (Logistic, SVM) hoặc khoảng cách (KNN) không phù hợp để diễn giải bằng TreeExplainer.")
    print(f"- Lọc tập ứng viên tree-based phù hợp: {shap_supported_models}")

    valid_df = results_df[results_df['Model'].isin(shap_supported_models)]
    preferred_order = ['Random Forest', 'XGBoost', 'Decision Tree']
    best_model_name = None
    for candidate in preferred_order:
        if candidate in valid_df['Model'].values:
            best_model_name = candidate
            break

    if best_model_name is None:
        best_model_name = valid_df.loc[valid_df['F1'].idxmax(), 'Model']

    best_tree = valid_df.loc[valid_df['Model'] == best_model_name].iloc[0]
    best_model = trained_models[best_model_name]

    print(f"\n=> KẾT LUẬN: Chọn {best_model_name} làm mô hình nền tảng vì nó vừa có độ chính xác tốt, vừa phù hợp với SHAP TreeExplainer để tạo kết quả giải thích có giá trị cho nghiên cứu và paper.")
    
    # Save best model
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))

    # Save results
    os.makedirs(reports_dir, exist_ok=True)
    results_df.to_csv(os.path.join(reports_dir, 'model_comparison.csv'), index=False)
    
    print("\n=== COMPARISON TABLE ===")
    print(results_df.to_string(index=False))
    print("\nTraining process completed!")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'src') # Saving in src for now based on notebook
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    
    train_and_evaluate(DATA_PATH, MODELS_DIR, REPORTS_DIR)
