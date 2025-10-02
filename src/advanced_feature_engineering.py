"""
Advanced Feature Engineering for Anomaly Detection

Goal: Improve Macro-F1 from 0.8004 to 0.90+

Strategy:
1. Interaction features from top discriminative features
2. Clustering-based features (KMeans)
3. Statistical transformations (mean, std, skew, kurtosis, quantiles)
4. Polynomial features
5. PCA dimensionality reduction
6. Remove highly correlated features

Input: 52 original sensor features
Output: 149 engineered features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# 분석에서 발견한 핵심 피처
TOP_DISCRIMINATIVE = ['X_19', 'X_37', 'X_40', 'X_11', 'X_28']
LOW_VARIANCE_FEATURES = ['X_49', 'X_29', 'X_35', 'X_10', 'X_17', 'X_13', 'X_43',
                         'X_21', 'X_12', 'X_24', 'X_46', 'X_45', 'X_06', 'X_04',
                         'X_39', 'X_27', 'X_15', 'X_31', 'X_36', 'X_14', 'X_48',
                         'X_44', 'X_52', 'X_34', 'X_50', 'X_02', 'X_23', 'X_03', 'X_01']

HIGHLY_CORRELATED_PAIRS = [
    ('X_04', 'X_39'), ('X_05', 'X_25'), ('X_06', 'X_45'), ('X_07', 'X_33')
]

def create_interaction_features(df):
    """상위 피처 간 상호작용 생성"""
    df_new = df.copy()

    print("Creating interaction features...")

    # Top discriminative features 간 상호작용
    for i, feat1 in enumerate(TOP_DISCRIMINATIVE):
        if feat1 in df.columns:
            for feat2 in TOP_DISCRIMINATIVE[i+1:]:
                if feat2 in df.columns:
                    # 곱셈
                    df_new[f'{feat1}_mul_{feat2}'] = df[feat1] * df[feat2]
                    # 차이
                    df_new[f'{feat1}_sub_{feat2}'] = df[feat1] - df[feat2]
                    # 비율 (안전하게)
                    df_new[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-10)

    print(f"  Added {len(df_new.columns) - len(df.columns)} interaction features")
    return df_new

def create_statistical_features(df):
    """고급 통계 피처"""
    df_new = df.copy()

    print("Creating statistical features...")

    # 행별 통계 (더 다양하게)
    df_new['row_mean'] = df.mean(axis=1)
    df_new['row_std'] = df.std(axis=1)
    df_new['row_median'] = df.median(axis=1)
    df_new['row_max'] = df.max(axis=1)
    df_new['row_min'] = df.min(axis=1)
    df_new['row_range'] = df_new['row_max'] - df_new['row_min']
    df_new['row_skew'] = df.skew(axis=1)
    df_new['row_kurt'] = df.kurtosis(axis=1)

    # 분위수
    df_new['row_q10'] = df.quantile(0.1, axis=1)
    df_new['row_q25'] = df.quantile(0.25, axis=1)
    df_new['row_q75'] = df.quantile(0.75, axis=1)
    df_new['row_q90'] = df.quantile(0.9, axis=1)
    df_new['row_iqr'] = df_new['row_q75'] - df_new['row_q25']

    # 비율 피처
    df_new['mean_to_std'] = df_new['row_mean'] / (df_new['row_std'] + 1e-10)
    df_new['max_to_min'] = df_new['row_max'] / (df_new['row_min'] + 1e-10)
    df_new['range_to_mean'] = df_new['row_range'] / (df_new['row_mean'] + 1e-10)

    # 카운트 피처
    df_new['positive_count'] = (df > 0).sum(axis=1)
    df_new['negative_count'] = (df < 0).sum(axis=1)
    df_new['zero_count'] = (df == 0).sum(axis=1)

    # 극값 피처
    df_new['abs_max'] = df.abs().max(axis=1)
    df_new['abs_min'] = df.abs().min(axis=1)
    df_new['abs_mean'] = df.abs().mean(axis=1)
    df_new['abs_std'] = df.abs().std(axis=1)

    print(f"  Added {len(df_new.columns) - len(df.columns)} statistical features")
    return df_new

def create_clustering_features(df_train, df_test=None):
    """클러스터링 기반 피처"""
    print("Creating clustering features...")

    # 스케일링
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(df_train)

    if df_test is not None:
        test_scaled = scaler.transform(df_test)

    # KMeans 클러스터링 (여러 개수)
    for n_clusters in [5, 10, 15, 21]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        df_train[f'kmeans_{n_clusters}'] = kmeans.fit_predict(train_scaled)

        if df_test is not None:
            df_test[f'kmeans_{n_clusters}'] = kmeans.predict(test_scaled)

        # 클러스터 중심까지의 거리
        distances = kmeans.transform(train_scaled).min(axis=1)
        df_train[f'kmeans_{n_clusters}_dist'] = distances

        if df_test is not None:
            test_distances = kmeans.transform(test_scaled).min(axis=1)
            df_test[f'kmeans_{n_clusters}_dist'] = test_distances

    print(f"  Added clustering features")

    if df_test is not None:
        return df_train, df_test
    return df_train

def create_pca_features(df_train, df_test=None, n_components=30):
    """PCA 피처 (더 많은 컴포넌트)"""
    print(f"Creating PCA features (n_components={n_components})...")

    # 원본 피처만 사용
    original_features = [col for col in df_train.columns if col.startswith('X_')]

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    train_pca = pca.fit_transform(df_train[original_features])

    for i in range(n_components):
        df_train[f'pca_{i}'] = train_pca[:, i]

    if df_test is not None:
        test_pca = pca.transform(df_test[original_features])
        for i in range(n_components):
            df_test[f'pca_{i}'] = test_pca[:, i]

    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    if df_test is not None:
        return df_train, df_test
    return df_train

def create_polynomial_features(df, degree=2):
    """다항식 피처 (Top features만)"""
    print(f"Creating polynomial features (degree={degree})...")

    available_top = [f for f in TOP_DISCRIMINATIVE if f in df.columns]

    if len(available_top) > 0:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(df[available_top])

        # 새로운 피처만 추가 (원본 제외)
        n_original = len(available_top)
        for i in range(n_original, poly_features.shape[1]):
            df[f'poly_{i}'] = poly_features[:, i]

    print(f"  Added {len([c for c in df.columns if c.startswith('poly_')])} polynomial features")
    return df

def remove_highly_correlated(df):
    """고도로 상관된 피처 제거"""
    print("Removing highly correlated features...")

    features_to_drop = []
    for feat1, feat2 in HIGHLY_CORRELATED_PAIRS:
        if feat1 in df.columns and feat2 in df.columns:
            # feat2를 제거 (feat1 유지)
            features_to_drop.append(feat2)

    if features_to_drop:
        df = df.drop(columns=features_to_drop)
        print(f"  Removed {len(features_to_drop)} highly correlated features")

    return df

def remove_low_variance(df):
    """Low variance 피처 제거 (선택적)"""
    print("Checking low variance features...")

    available_low_var = [f for f in LOW_VARIANCE_FEATURES if f in df.columns]

    if available_low_var:
        # 실제 분산 재확인
        variances = df[available_low_var].var()
        really_low = variances[variances < 0.001].index.tolist()

        if really_low:
            df = df.drop(columns=really_low)
            print(f"  Removed {len(really_low)} very low variance features")
        else:
            print(f"  Kept all features (variance acceptable)")

    return df

def engineer_features(train_df, test_df, y_train=None):
    """전체 Feature Engineering 파이프라인"""
    print("="*70)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*70)
    print(f"\nOriginal features: {len(train_df.columns)}")

    # 1. 상호작용 피처
    train_df = create_interaction_features(train_df)
    test_df = create_interaction_features(test_df)

    # 2. 통계 피처
    train_df = create_statistical_features(train_df)
    test_df = create_statistical_features(test_df)

    # 3. 다항식 피처
    train_df = create_polynomial_features(train_df)
    test_df = create_polynomial_features(test_df)

    # 4. 클러스터링 피처
    train_df, test_df = create_clustering_features(train_df, test_df)

    # 5. PCA 피처
    train_df, test_df = create_pca_features(train_df, test_df, n_components=30)

    # 6. 고도로 상관된 피처 제거
    train_df = remove_highly_correlated(train_df)
    test_df = remove_highly_correlated(test_df)

    # 7. Low variance 피처 제거 (선택적)
    # train_df = remove_low_variance(train_df)
    # test_df = remove_low_variance(test_df)

    print(f"\nFinal features: {len(train_df.columns)}")
    print(f"Features added: {len(train_df.columns) - 52}")

    return train_df, test_df

def main():
    """테스트 실행"""
    print("\nLoading data...")
    train = pd.read_csv('../open/train.csv')
    test = pd.read_csv('../open/test.csv')

    feature_cols = [col for col in train.columns if col not in ['ID', 'target']]
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]

    # Feature Engineering
    X_train_fe, X_test_fe = engineer_features(X_train, X_test, y_train)

    # 저장
    print("\nSaving engineered features...")
    X_train_fe.to_csv('../data/train_advanced_fe.csv', index=False)
    X_test_fe.to_csv('../data/test_advanced_fe.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)

    print("\n✅ Feature engineering complete!")
    print(f"   Train: {X_train_fe.shape}")
    print(f"   Test: {X_test_fe.shape}")

if __name__ == '__main__':
    main()
