"""
GPU 버전 - 최종 모델 실행 스크립트
CUDA가 설치되어 있어야 실행 가능
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

# GPU 사용 가능 여부 확인
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"[OK] CUDA Available! GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("[WARNING] CUDA not available. Running on CPU.")

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


def get_final_ensemble_gpu():
    """GPU 최적화된 앙상블 모델"""

    # LightGBM GPU 설정
    lgb_params = {
        'n_estimators': 600,
        'max_depth': 10,
        'learning_rate': 0.05,
        'num_leaves': 50,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'objective': 'multiclass',
        'class_weight': 'balanced'
    }

    # GPU 사용 가능하면 GPU 설정 추가
    if CUDA_AVAILABLE:
        lgb_params['device'] = 'gpu'
        lgb_params['gpu_platform_id'] = 0
        lgb_params['gpu_device_id'] = 0
        print("  LightGBM: GPU 모드")
    else:
        lgb_params['device'] = 'cpu'
        print("  LightGBM: CPU 모드")

    lgb = LGBMClassifier(**lgb_params)

    # XGBoost GPU 설정
    xgb_params = {
        'n_estimators': 600,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'gamma': 0.1,
        'min_child_weight': 3,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'objective': 'multi:softprob',
        'use_label_encoder': False
    }

    if CUDA_AVAILABLE:
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['predictor'] = 'gpu_predictor'
        xgb_params['gpu_id'] = 0
        print("  XGBoost: GPU 모드")
    else:
        xgb_params['tree_method'] = 'hist'
        print("  XGBoost: CPU 모드")

    xgb = XGBClassifier(**xgb_params)

    # CatBoost GPU 설정
    cat_params = {
        'iterations': 800,
        'depth': 10,
        'learning_rate': 0.05,
        'random_state': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'l2_leaf_reg': 5
    }

    if CUDA_AVAILABLE:
        cat_params['task_type'] = 'GPU'
        cat_params['devices'] = '0'
        print("  CatBoost: GPU 모드")
    else:
        cat_params['task_type'] = 'CPU'
        print("  CatBoost: CPU 모드")

    cat = CatBoostClassifier(**cat_params)

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
    print("  [GPU VERSION] Abnormal Signal Detection Model")
    print("  Target: Macro-F1 Score > 0.9")
    print("="*70)

    # 1. 데이터 로드
    print("\n[STEP 1] 데이터 로드...")
    train, test, submission = load_data()

    # 피처와 타겟 분리
    feature_cols = get_feature_columns(train)
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]

    print(f"  [v] Train: {X_train.shape}")
    print(f"  [v] Test: {X_test.shape}")

    # 타겟 분포 확인
    print_target_distribution(y_train, "Train Target Distribution")

    # 2. Feature Engineering
    print("\n[STEP 2] Feature Engineering...")
    X_train = create_quick_features(X_train)
    X_test = create_quick_features(X_test)
    print(f"  [v] Features created: {X_train.shape[1]}")

    # 3. 스케일링
    print("\n[STEP 3] 데이터 스케일링...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  [v] Scaling completed")

    # 4. 모델 학습
    print("\n[STEP 4] 앙상블 모델 학습...")
    ensemble = get_final_ensemble_gpu()

    if CUDA_AVAILABLE:
        print("  [GPU] Training model on GPU... (30s~1min)")
    else:
        print("  [CPU] Training model on CPU... (2-3min)")

    import time
    start_time = time.time()
    ensemble.fit(X_train_scaled, y_train)
    elapsed_time = time.time() - start_time

    print(f"  [v] Training completed in {elapsed_time:.2f} seconds")

    # 5. 예측
    print("\n[STEP 5] 테스트 데이터 예측...")
    predictions = ensemble.predict(X_test_scaled)
    print("  [v] Predictions completed")

    # 예측 분포 확인
    print_target_distribution(predictions, "Prediction Distribution")

    # 6. 제출 파일 생성
    print("\n[STEP 6] 제출 파일 생성...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = f'../submissions/final_ensemble_gpu_{timestamp}.csv'
    save_submission(predictions, submission, submission_path)

    print("\n" + "="*70)
    print("  [COMPLETE] Pipeline finished!")
    print("  제출 파일: " + submission_path)
    print("="*70)

    return predictions


if __name__ == "__main__":
    predictions = main()

    # 추가 정보 출력
    print("\n[실행 정보]")
    if CUDA_AVAILABLE:
        print(f"- GPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"- CUDA Version: {torch.version.cuda}")
        print("- 속도: CPU 대비 약 3-10배 빠름")
    else:
        print("- CPU 사용 (CUDA 미설치)")

    print("\n[모델 정보]")
    print("- 모델: LightGBM + XGBoost + CatBoost Voting Ensemble")
    print("- Feature Engineering: 통계적 특징 + 비율 특징")
    print("- Scaling: RobustScaler (이상치에 강함)")
    print("- Class Weight: Balanced (클래스 불균형 처리)")

    print("\nPlease check the leaderboard and record your score!")
    print("Target: Macro-F1 Score > 0.9")