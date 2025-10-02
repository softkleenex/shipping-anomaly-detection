"""
최종 모델 실행 스크립트
빠른 실행을 위한 최적화된 버전
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from datetime import datetime
import sys
import os

# 현재 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_data, get_feature_columns, save_submission, print_target_distribution

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def create_quick_features(df):
    """빠른 Feature Engineering (핵심 피처만)"""
    df_new = df.copy()

    # 핵심 통계 특징만
    df_new['row_mean'] = df.mean(axis=1)
    df_new['row_std'] = df.std(axis=1)
    df_new['row_max'] = df.max(axis=1)
    df_new['row_min'] = df.min(axis=1)
    df_new['row_range'] = df_new['row_max'] - df_new['row_min']

    # 비율 특징
    df_new['mean_to_std_ratio'] = df_new['row_mean'] / (df_new['row_std'] + 1e-10)

    # 카운트 특징
    df_new['positive_count'] = (df > 0).sum(axis=1)
    df_new['negative_count'] = (df < 0).sum(axis=1)

    return df_new


def get_final_ensemble():
    """검증된 최종 앙상블 모델"""

    # 최적화된 하이퍼파라미터
    lgb = LGBMClassifier(
        n_estimators=600,
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
        class_weight='balanced'
    )

    xgb = XGBClassifier(
        n_estimators=600,
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
        use_label_encoder=False
    )

    cat = CatBoostClassifier(
        iterations=800,
        depth=10,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=False,
        auto_class_weights='Balanced',
        l2_leaf_reg=5
    )

    # Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('lgb', lgb),
            ('xgb', xgb),
            ('cat', cat)
        ],
        voting='soft',
        n_jobs=-1
    )

    return ensemble


def main():
    """메인 실행 함수"""

    print("="*70)
    print("  🚀 이상신호 감지 기반 비정상 작동 진단 - 최종 모델")
    print("  목표: Macro-F1 Score > 0.9")
    print("="*70)

    # 1. 데이터 로드
    print("\n[STEP 1] 데이터 로드...")
    train, test, submission = load_data()

    # 피처와 타겟 분리
    feature_cols = get_feature_columns(train)
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]

    print(f"  ✓ Train: {X_train.shape}")
    print(f"  ✓ Test: {X_test.shape}")

    # 타겟 분포 확인
    print_target_distribution(y_train, "Train Target Distribution")

    # 2. Feature Engineering
    print("\n[STEP 2] Feature Engineering...")
    X_train = create_quick_features(X_train)
    X_test = create_quick_features(X_test)
    print(f"  ✓ Features created: {X_train.shape[1]}")

    # 3. 스케일링
    print("\n[STEP 3] 데이터 스케일링...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  ✓ Scaling completed")

    # 4. 모델 학습
    print("\n[STEP 4] 앙상블 모델 학습...")
    ensemble = get_final_ensemble()

    print("  모델 학습 중... (약 2-3분 소요)")
    ensemble.fit(X_train_scaled, y_train)
    print("  ✓ Training completed")

    # 5. 예측
    print("\n[STEP 5] 테스트 데이터 예측...")
    predictions = ensemble.predict(X_test_scaled)
    print("  ✓ Predictions completed")

    # 예측 분포 확인
    print_target_distribution(predictions, "Prediction Distribution")

    # 6. 제출 파일 생성
    print("\n[STEP 6] 제출 파일 생성...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = f'../submissions/final_ensemble_{timestamp}.csv'
    save_submission(predictions, submission, submission_path)

    print("\n" + "="*70)
    print("  ✅ 파이프라인 완료!")
    print("  제출 파일: " + submission_path)
    print("="*70)

    return predictions


if __name__ == "__main__":
    predictions = main()

    # 추가 정보 출력
    print("\n[추가 정보]")
    print("- 모델: LightGBM + XGBoost + CatBoost Voting Ensemble")
    print("- Feature Engineering: 통계적 특징 + 비율 특징")
    print("- Scaling: RobustScaler (이상치에 강함)")
    print("- Class Weight: Balanced (클래스 불균형 처리)")
    print("\n제출 전 리더보드를 확인하고 점수를 기록하세요!")
    print("목표: Macro-F1 Score > 0.9")