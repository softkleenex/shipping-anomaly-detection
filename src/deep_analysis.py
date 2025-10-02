"""
Deep Data Analysis and Strategy Development

Goal: Achieve Macro-F1 > 0.9

This script performs comprehensive analysis to identify:
- Low variance features
- Highly correlated feature pairs
- Top discriminative features
- Class-wise performance gaps
- Baseline model performance

Output:
- Performance metrics and improvement opportunities
- Feature importance rankings
- Problem classes identification
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_data():
    """데이터 로드"""
    train = pd.read_csv('../open/train.csv')
    test = pd.read_csv('../open/test.csv')

    feature_cols = [col for col in train.columns if col not in ['ID', 'target']]
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]

    return X_train, y_train, X_test, train, test

def analyze_target_distribution(y_train):
    """타겟 분포 분석"""
    print("\n" + "="*70)
    print("1. TARGET DISTRIBUTION ANALYSIS")
    print("="*70)

    target_counts = y_train.value_counts().sort_index()
    target_ratio = y_train.value_counts(normalize=True).sort_index()

    print(f"\nTotal classes: {y_train.nunique()}")
    print(f"Total samples: {len(y_train)}")
    print(f"\nClass distribution:")

    for cls in sorted(target_counts.index):
        print(f"  Class {cls:2d}: {target_counts[cls]:5d} ({target_ratio[cls]*100:5.2f}%)")

    imbalance_ratio = target_counts.max() / target_counts.min()
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 2:
        print("  [WARNING] Significant class imbalance detected!")
    else:
        print("  [OK] Classes are balanced")

    return target_counts

def analyze_feature_statistics(X_train, y_train):
    """피처 통계 분석"""
    print("\n" + "="*70)
    print("2. FEATURE STATISTICS ANALYSIS")
    print("="*70)

    # 기본 통계
    print("\nBasic statistics:")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Missing values: {X_train.isnull().sum().sum()}")

    # 피처별 분산
    variances = X_train.var().sort_values(ascending=False)
    print(f"\nTop 10 features by variance:")
    for i, (feat, var) in enumerate(variances.head(10).items(), 1):
        print(f"  {i}. {feat}: {var:.4f}")

    # Low variance features (거의 변하지 않는 피처)
    low_var_features = variances[variances < 0.01]
    if len(low_var_features) > 0:
        print(f"\n[WARNING] {len(low_var_features)} low variance features found:")
        print(f"  {list(low_var_features.index)}")
    else:
        print("\n[OK] No low variance features")

    # 스케일 차이
    feature_scales = X_train.max() - X_train.min()
    print(f"\nFeature scale range:")
    print(f"  Min: {feature_scales.min():.4f}")
    print(f"  Max: {feature_scales.max():.4f}")
    print(f"  Ratio: {feature_scales.max() / feature_scales.min():.2f}")

    return variances, low_var_features

def analyze_feature_importance_by_class(X_train, y_train):
    """클래스별 피처 중요도 분석"""
    print("\n" + "="*70)
    print("3. CLASS-WISE FEATURE IMPORTANCE")
    print("="*70)

    # 각 클래스별 평균값
    class_means = X_train.groupby(y_train).mean()

    # 클래스 간 차이가 큰 피처 찾기
    feature_discriminability = class_means.std(axis=0).sort_values(ascending=False)

    print("\nTop 15 discriminative features (highest class variance):")
    for i, (feat, std) in enumerate(feature_discriminability.head(15).items(), 1):
        print(f"  {i:2d}. {feat}: {std:.4f}")

    return feature_discriminability

def analyze_correlations(X_train):
    """상관관계 분석"""
    print("\n" + "="*70)
    print("4. CORRELATION ANALYSIS")
    print("="*70)

    # 높은 상관관계 쌍 찾기
    corr_matrix = X_train.corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

    if high_corr_pairs:
        print(f"\n[WARNING] {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9):")
        for pair in high_corr_pairs[:10]:
            print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")
    else:
        print("\n[OK] No highly correlated feature pairs")

    return high_corr_pairs

def quick_model_benchmark(X_train, y_train):
    """빠른 모델 벤치마크"""
    print("\n" + "="*70)
    print("5. QUICK MODEL BENCHMARK")
    print("="*70)

    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier

    # 데이터 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_train
    )

    # 스케일링
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    # 간단한 모델
    model = LGBMClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    )

    print("\nTraining quick baseline model...")
    model.fit(X_tr_scaled, y_tr)

    # 예측
    y_pred = model.predict(X_val_scaled)

    # 평가
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    class_f1 = f1_score(y_val, y_pred, average=None)

    print(f"\nValidation Macro F1: {macro_f1:.4f}")
    print("\nClass-wise F1 scores:")

    for cls, f1 in enumerate(class_f1):
        print(f"  Class {cls:2d}: {f1:.4f}")

    # 가장 성능이 낮은 클래스 찾기
    worst_classes = np.argsort(class_f1)[:5]
    print(f"\nWorst performing classes:")
    for cls in worst_classes:
        print(f"  Class {cls}: F1={class_f1[cls]:.4f}")

    return macro_f1, class_f1, worst_classes

def generate_improvement_strategy(macro_f1, class_f1, worst_classes,
                                  feature_discriminability, high_corr_pairs):
    """개선 전략 생성"""
    print("\n" + "="*70)
    print("6. IMPROVEMENT STRATEGY")
    print("="*70)

    gap = 0.9 - macro_f1

    print(f"\nCurrent Baseline: {macro_f1:.4f}")
    print(f"Target: 0.9000")
    print(f"Gap: {gap:.4f} ({gap*100:.2f}%)")

    print("\n[PRIORITY 1] Feature Engineering:")
    print("  - Create interaction features from top discriminative features")
    print(f"    Top features: {list(feature_discriminability.head(5).index)}")
    print("  - Polynomial features (degree 2)")
    print("  - Frequency domain features (FFT)")
    print("  - Clustering features (KMeans, DBSCAN)")

    if len(high_corr_pairs) > 0:
        print("\n[PRIORITY 2] Feature Selection:")
        print(f"  - Remove or combine {len(high_corr_pairs)} highly correlated pairs")
        print("  - Use mutual information for feature selection")

    print("\n[PRIORITY 3] Class Imbalance Handling:")
    print("  - Apply SMOTE or ADASYN for minority classes")
    print(f"  - Focus on worst classes: {[int(c) for c in worst_classes]}")
    print("  - Adjust class weights dynamically")

    print("\n[PRIORITY 4] Model Optimization:")
    print("  - Deep hyperparameter tuning (Optuna 200+ trials)")
    print("  - Test different objective functions")
    print("  - Try focal loss for hard classes")

    print("\n[PRIORITY 5] Ensemble Techniques:")
    print("  - Stacking with meta-learner")
    print("  - Weighted voting based on class performance")
    print("  - Blending multiple models")

    print("\n[PRIORITY 6] Post-processing:")
    print("  - Threshold optimization per class")
    print("  - Calibration (Platt scaling, isotonic)")
    print("  - Test Time Augmentation (TTA)")

    # 예상 개선치
    print("\n" + "="*70)
    print("EXPECTED IMPROVEMENTS:")
    print("="*70)
    print(f"  Feature Engineering:     +0.02 ~ +0.04")
    print(f"  Hyperparameter Tuning:   +0.01 ~ +0.03")
    print(f"  Class Balancing:         +0.01 ~ +0.02")
    print(f"  Advanced Ensemble:       +0.02 ~ +0.05")
    print(f"  Post-processing:         +0.01 ~ +0.02")
    print(f"  " + "-"*40)
    print(f"  Total Expected:          +0.07 ~ +0.16")
    print(f"  Target Score:            0.90+")

def main():
    """메인 실행"""
    print("\n" + "="*70)
    print("DEEP ANALYSIS FOR MACRO-F1 > 0.9")
    print("="*70)

    # 데이터 로드
    print("\nLoading data...")
    X_train, y_train, X_test, train, test = load_data()

    # 1. 타겟 분포 분석
    target_counts = analyze_target_distribution(y_train)

    # 2. 피처 통계 분석
    variances, low_var_features = analyze_feature_statistics(X_train, y_train)

    # 3. 클래스별 피처 중요도
    feature_discriminability = analyze_feature_importance_by_class(X_train, y_train)

    # 4. 상관관계 분석
    high_corr_pairs = analyze_correlations(X_train)

    # 5. 빠른 벤치마크
    macro_f1, class_f1, worst_classes = quick_model_benchmark(X_train, y_train)

    # 6. 개선 전략 생성
    generate_improvement_strategy(
        macro_f1, class_f1, worst_classes,
        feature_discriminability, high_corr_pairs
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Implement advanced feature engineering")
    print("2. Run deep hyperparameter optimization")
    print("3. Build stacking ensemble")
    print("4. Test and submit improved model")

if __name__ == '__main__':
    main()
