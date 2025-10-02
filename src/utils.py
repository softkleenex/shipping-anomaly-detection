"""
유틸리티 함수들
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time


def load_data(data_path='../open/'):
    """데이터 로드"""
    train = pd.read_csv(f'{data_path}train.csv')
    test = pd.read_csv(f'{data_path}test.csv')
    submission = pd.read_csv(f'{data_path}sample_submission.csv')

    return train, test, submission


def get_feature_columns(df):
    """피처 컬럼 추출"""
    return [col for col in df.columns if col not in ['ID', 'target']]


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name="Model"):
    """모델 평가"""
    start_time = time.time()

    # 학습
    model.fit(X_train, y_train)

    # 예측
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Macro F1 Score 계산
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')

    # 각 클래스별 F1 Score
    class_f1 = f1_score(y_val, y_val_pred, average=None)

    elapsed_time = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Train Macro F1: {train_f1:.4f}")
    print(f"Val Macro F1: {val_f1:.4f}")
    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"\nClass-wise F1 scores:")
    for i, f1 in enumerate(class_f1):
        print(f"  Class {i}: {f1:.4f}")

    return model, val_f1, class_f1


def plot_feature_importance(model, feature_names, top_n=20):
    """피처 중요도 시각화"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()

        return importance_df
    else:
        print("This model doesn't have feature_importances_ attribute")
        return None


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Confusion Matrix 시각화"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return cm


def stratified_kfold_cv(model, X, y, n_splits=5, random_state=42):
    """Stratified K-Fold 교차 검증"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    print(f"\n{n_splits}-Fold Cross Validation:")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

        # 모델 클론
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train_fold, y_train_fold)

        # 예측 및 평가
        y_pred = model_clone.predict(X_val_fold)
        score = f1_score(y_val_fold, y_pred, average='macro')
        scores.append(score)

        print(f"  Fold {fold}: {score:.4f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"\nMean CV Score: {mean_score:.4f} (±{std_score:.4f})")

    return scores, mean_score, std_score


def save_submission(predictions, submission_df, filename=None):
    """제출 파일 저장"""
    from datetime import datetime

    submission_df['target'] = predictions

    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'../submissions/submission_{timestamp}.csv'

    submission_df.to_csv(filename, index=False)
    print(f"Submission saved to: {filename}")

    return submission_df


def print_target_distribution(y, title="Target Distribution"):
    """타겟 분포 출력"""
    value_counts = pd.Series(y).value_counts().sort_index()
    value_ratios = pd.Series(y).value_counts(normalize=True).sort_index()

    print(f"\n{title}:")
    print("="*50)
    for cls in value_counts.index:
        count = value_counts[cls]
        ratio = value_ratios[cls]
        print(f"Class {cls}: {count:6d} ({ratio*100:5.2f}%)")
    print("="*50)

    # 클래스 불균형 비율
    imbalance_ratio = value_counts.max() / value_counts.min()
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")

    return value_counts, value_ratios


def check_missing_values(df):
    """결측치 확인"""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    missing_table = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    missing_table = missing_table[missing_table['Missing_Count'] > 0]
    missing_table = missing_table.sort_values('Missing_Count', ascending=False)

    if len(missing_table) == 0:
        print("No missing values found!")
    else:
        print(f"Missing values in {len(missing_table)} columns:")
        print(missing_table)

    return missing_table


def memory_usage(df):
    """메모리 사용량 확인"""
    memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage: {memory:.2f} MB")
    return memory