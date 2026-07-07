import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(data_path):
    """Load the raw Telco customer churn dataset."""
    file_path = os.path.join(data_path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the dataset (handling missing values, dropping columns)."""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    df = df.drop(['customerID'], axis=1)
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.drop('Churn')
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df, le_dict

def feature_engineering(df):
    """Create new features."""
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['HighCharge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
    df['LowTenureHighCharge'] = ((df['tenure'] < 12) & (df['HighCharge'] == 1)).astype(int)
    
    if 'TenureGroup' in df.columns:
        df = pd.get_dummies(df, columns=['TenureGroup'], drop_first=True)
        
    return df

def split_and_scale(df, data_path):
    """Split data into train and test sets, scale numeric columns, and save."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge']
    scaler = StandardScaler()
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Save the processed data
    X_train.to_csv(os.path.join(data_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(data_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_path, 'y_test.csv'), index=False)
    
    print(f"Data preprocessing and splitting completed. Files saved to {data_path}")
    print(f"Train shape: {X_train.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    
    df_raw = load_data(DATA_PATH)
    df_processed, _ = preprocess_data(df_raw)
    df_featured = feature_engineering(df_processed)
    split_and_scale(df_featured, DATA_PATH)
