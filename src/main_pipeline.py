"""
메인 파이프라인 - 이상신호 감지 기반 비정상 작동 진단
목표: Macro-F1 Score > 0.9
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import time
from datetime import datetime
import os
import sys

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def create_features(df):
    """Feature Engineering"""
    df_new = df.copy()

    # 통계적 특징
    df_new['row_mean'] = df.mean(axis=1)
    df_new['row_std'] = df.std(axis=1)
    df_new['row_max'] = df.max(axis=1)
    df_new['row_min'] = df.min(axis=1)
    df_new['row_median'] = df.median(axis=1)
    df_new['row_range'] = df_new['row_max'] - df_new['row_min']
    df_new['row_skew'] = df.skew(axis=1)
    df_new['row_kurt'] = df.kurtosis(axis=1)

    # 비율 특징
    df_new['mean_to_std_ratio'] = df_new['row_mean'] / (df_new['row_std'] + 1e-10)
    df_new['max_to_min_ratio'] = df_new['row_max'] / (df_new['row_min'] + 1e-10)

    # 분위수 특징
    df_new['row_q25'] = df.quantile(0.25, axis=1)
    df_new['row_q75'] = df.quantile(0.75, axis=1)
    df_new['row_iqr'] = df_new['row_q75'] - df_new['row_q25']

    # 카운트 특징
    df_new['positive_count'] = (df > 0).sum(axis=1)
    df_new['negative_count'] = (df < 0).sum(axis=1)
    df_new['zero_count'] = (df == 0).sum(axis=1)

    # 극값 특징
    df_new['abs_max'] = df.abs().max(axis=1)
    df_new['abs_min'] = df.abs().min(axis=1)
    df_new['abs_mean'] = df.abs().mean(axis=1)

    return df_new


def add_pca_features(X_train, X_test, n_components=20):
    """PCA 특징 추가"""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)

    X_train_pca = pca.fit_transform(X_train.iloc[:, :52])  # 원본 피처만
    X_test_pca = pca.transform(X_test.iloc[:, :52])

    # PCA 컴포넌트를 데이터프레임에 추가
    for i in range(n_components):
        X_train[f'pca_{i}'] = X_train_pca[:, i]
        X_test[f'pca_{i}'] = X_test_pca[:, i]

    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    return X_train, X_test


def select_features(X_train, y_train, X_test, importance_threshold=0):
    """Feature Selection based on importance"""

    # LightGBM으로 피처 중요도 계산
    lgb = LGBMClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    )
    lgb.fit(X_train, y_train)

    # 중요도가 threshold 이상인 피처 선택
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': lgb.feature_importances_
    }).sort_values('importance', ascending=False)

    selected_features = feature_importance[
        feature_importance['importance'] > importance_threshold
    ]['feature'].tolist()

    print(f"Selected features: {len(selected_features)} / {len(X_train.columns)}")

    return X_train[selected_features], X_test[selected_features]


def get_optimized_models():
    """최적화된 모델 반환"""

    models = {
        'lgb': LGBMClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=50,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            objective='multiclass',
            metric='multi_logloss',
            class_weight='balanced'
        ),
        'xgb': XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            gamma=0.1,
            min_child_weight=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        'cat': CatBoostClassifier(
            iterations=800,
            depth=10,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=False,
            auto_class_weights='Balanced',
            task_type='CPU',
            l2_leaf_reg=5
        )
    }

    return models


def train_ensemble(X_train, y_train, X_val, y_val):
    """앙상블 모델 학습"""

    models = get_optimized_models()

    # 개별 모델 학습
    print("\n개별 모델 학습 중...")
    for name, model in models.items():
        print(f"  - {name.upper()} 학습 중...")
        model.fit(X_train, y_train)

        if X_val is not None:
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='macro')
            print(f"    Validation Macro F1: {score:.4f}")

    # Voting Ensemble
    voting_clf = VotingClassifier(
        estimators=list(models.items()),
        voting='soft',
        n_jobs=-1
    )

    print("\nVoting Ensemble 학습 중...")
    voting_clf.fit(X_train, y_train)

    if X_val is not None:
        y_pred = voting_clf.predict(X_val)
        score = f1_score(y_val, y_pred, average='macro')
        print(f"  Voting Ensemble Validation Macro F1: {score:.4f}")

    return voting_clf


def cross_validate_model(model, X, y, n_splits=5):
    """교차 검증"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    print(f"\n{n_splits}-Fold 교차 검증 중...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)

        y_pred = model_clone.predict(X_val)
        score = f1_score(y_val, y_pred, average='macro')
        scores.append(score)
        print(f"  Fold {fold}: {score:.4f}")

    print(f"\nCV Mean: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    return scores


def main():
    """메인 파이프라인"""

    print("="*60)
    print("이상신호 감지 기반 비정상 작동 진단")
    print("목표: Macro-F1 Score > 0.9")
    print("="*60)

    # 1. 데이터 로드
    print("\n[1/6] 데이터 로드 중...")
    train = pd.read_csv('../open/train.csv')
    test = pd.read_csv('../open/test.csv')
    submission = pd.read_csv('../open/sample_submission.csv')

    # 피처와 타겟 분리
    feature_cols = [col for col in train.columns if col not in ['ID', 'target']]
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]

    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Classes: {sorted(y_train.unique())}")

    # 2. Feature Engineering
    print("\n[2/6] Feature Engineering 중...")
    X_train = create_features(X_train)
    X_test = create_features(X_test)
    print(f"  Features after engineering: {X_train.shape[1]}")

    # PCA 특징 추가
    X_train, X_test = add_pca_features(X_train, X_test, n_components=20)
    print(f"  Features after PCA: {X_train.shape[1]}")

    # 3. Feature Selection
    print("\n[3/6] Feature Selection 중...")
    X_train, X_test = select_features(X_train, y_train, X_test)

    # 4. 스케일링
    print("\n[4/6] 데이터 스케일링 중...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # 5. 모델 학습
    print("\n[5/6] 모델 학습 중...")

    # Validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    # 앙상블 모델 학습
    ensemble_model = train_ensemble(X_tr, y_tr, X_val, y_val)

    # 교차 검증 (옵션)
    if input("\n교차 검증을 수행하시겠습니까? (y/n): ").lower() == 'y':
        cv_scores = cross_validate_model(ensemble_model, X_train_scaled, y_train)

    # 전체 데이터로 최종 학습
    print("\n최종 모델 학습 중...")
    final_model = train_ensemble(X_train_scaled, y_train, None, None)

    # 6. 예측 및 제출
    print("\n[6/6] 예측 및 제출 파일 생성 중...")
    predictions = final_model.predict(X_test_scaled)

    # 예측 분포 확인
    print("\n예측 분포:")
    pred_dist = pd.Series(predictions).value_counts().sort_index()
    for cls, count in pred_dist.items():
        print(f"  Class {cls}: {count:6d} ({count/len(predictions)*100:5.2f}%)")

    # 제출 파일 생성
    submission['target'] = predictions

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = f'../submissions/submission_ensemble_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ 제출 파일 생성 완료: {submission_path}")
    print("\n" + "="*60)
    print("파이프라인 완료!")
    print("="*60)

    return submission


if __name__ == "__main__":
    submission = main()