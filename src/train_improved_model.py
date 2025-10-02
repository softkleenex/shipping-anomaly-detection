"""
개선된 모델 학습 (GPU)
목표: 0.8004 -> 0.90+
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import time
from datetime import datetime
import torch

RANDOM_STATE = 42
CUDA_AVAILABLE = torch.cuda.is_available()

print("="*70)
print("IMPROVED MODEL TRAINING (GPU)" if CUDA_AVAILABLE else "IMPROVED MODEL TRAINING (CPU)")
print("="*70)
print(f"GPU Available: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*70)

def load_engineered_data():
    """엔지니어링된 데이터 로드"""
    print("\nLoading engineered features...")
    X_train = pd.read_csv('../data/train_advanced_fe.csv')
    X_test = pd.read_csv('../data/test_advanced_fe.csv')
    y_train = pd.read_csv('../data/y_train.csv').squeeze()

    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]}")

    return X_train, X_test, y_train

def get_optimized_models():
    """최적화된 모델 생성"""

    # LightGBM
    lgb_params = {
        'n_estimators': 800,
        'max_depth': 12,
        'learning_rate': 0.03,
        'num_leaves': 64,
        'min_child_samples': 15,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'class_weight': 'balanced'
    }

    if CUDA_AVAILABLE:
        lgb_params['device'] = 'gpu'
        print("  LightGBM: GPU mode")
    else:
        lgb_params['device'] = 'cpu'

    lgb = LGBMClassifier(**lgb_params)

    # XGBoost
    xgb_params = {
        'n_estimators': 800,
        'max_depth': 12,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'gamma': 0.2,
        'min_child_weight': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'objective': 'multi:softprob',
        'use_label_encoder': False
    }

    if CUDA_AVAILABLE:
        xgb_params['tree_method'] = 'hist'  # Updated for newer XGBoost
        xgb_params['device'] = 'cuda'
        print("  XGBoost: GPU mode")
    else:
        xgb_params['tree_method'] = 'hist'

    xgb = XGBClassifier(**xgb_params)

    # CatBoost (CPU only - GPU out of memory on RTX 4060)
    cat_params = {
        'iterations': 1000,
        'depth': 12,
        'learning_rate': 0.03,
        'random_state': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'l2_leaf_reg': 5,
        'border_count': 254,
        'task_type': 'CPU',  # Force CPU to avoid OOM
        'thread_count': -1
    }

    print("  CatBoost: CPU mode (GPU OOM prevention)")

    cat = CatBoostClassifier(**cat_params)

    return {'lgb': lgb, 'xgb': xgb, 'cat': cat}

def train_and_evaluate():
    """학습 및 평가"""

    # 데이터 로드
    X_train, X_test, y_train = load_engineered_data()

    # 스케일링
    print("\nScaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5-Fold Cross Validation
    print("\n5-Fold Cross Validation:")
    print("-"*70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = get_optimized_models()

    # 각 모델별 CV 점수
    model_cv_scores = {name: [] for name in models.keys()}
    model_class_scores = {name: [] for name in models.keys()}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
        print(f"\nFold {fold}/5:")

        X_tr = X_train_scaled[train_idx]
        X_val = X_train_scaled[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        fold_start = time.time()

        for name, model in models.items():
            # 학습
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_tr, y_tr)

            # 예측
            y_pred = model_clone.predict(X_val)

            # 평가
            macro_f1 = f1_score(y_val, y_pred, average='macro')
            class_f1 = f1_score(y_val, y_pred, average=None)

            model_cv_scores[name].append(macro_f1)
            model_class_scores[name].append(class_f1)

            print(f"  {name.upper()}: {macro_f1:.4f}")

        fold_time = time.time() - fold_start
        print(f"  Fold time: {fold_time:.1f}s")

    # CV 결과 요약
    print("\n" + "="*70)
    print("Cross Validation Results:")
    print("="*70)

    for name in models.keys():
        scores = model_cv_scores[name]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{name.upper()}: {mean_score:.4f} (+/- {std_score:.4f})")

    # 최고 모델 선택
    best_model_name = max(model_cv_scores.keys(),
                          key=lambda x: np.mean(model_cv_scores[x]))
    best_score = np.mean(model_cv_scores[best_model_name])

    print(f"\nBest Model: {best_model_name.upper()} ({best_score:.4f})")

    # 문제 클래스 분석
    print("\n" + "="*70)
    print("Problem Classes Analysis:")
    print("="*70)

    problem_classes = [0, 3, 9, 15, 19]
    for name in models.keys():
        all_class_scores = np.array(model_class_scores[name])
        mean_class_scores = all_class_scores.mean(axis=0)

        print(f"\n{name.upper()} - Problem classes:")
        for cls in problem_classes:
            print(f"  Class {cls}: {mean_class_scores[cls]:.4f}")

    # 전체 데이터로 최종 학습
    print("\n" + "="*70)
    print("Training final models on full data...")
    print("="*70)

    final_models = {}
    for name, model in models.items():
        print(f"\nTraining {name.upper()}...")
        start_time = time.time()

        model.fit(X_train_scaled, y_train)
        final_models[name] = model

        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.1f}s")

    # Voting Ensemble
    print("\nCreating Voting Ensemble...")
    voting_ensemble = VotingClassifier(
        estimators=list(final_models.items()),
        voting='soft',
        n_jobs=-1
    )
    voting_ensemble.fit(X_train_scaled, y_train)

    # 예측
    print("\nPredicting on test data...")
    predictions = voting_ensemble.predict(X_test_scaled)

    # 예측 분포
    pred_dist = pd.Series(predictions).value_counts().sort_index()
    print("\nPrediction distribution:")
    for cls, count in pred_dist.items():
        print(f"  Class {cls}: {count:5d} ({count/len(predictions)*100:5.2f}%)")

    # 제출 파일 생성
    print("\nCreating submission file...")
    submission = pd.read_csv('../open/sample_submission.csv')
    submission['target'] = predictions

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'../submissions/improved_ensemble_cv{best_score:.4f}_{timestamp}.csv'
    submission.to_csv(filename, index=False)

    print(f"\n[OK] Submission saved: {filename}")

    return best_score, filename

def main():
    """메인 실행"""
    start_time = time.time()

    try:
        best_cv_score, submission_file = train_and_evaluate()

        total_time = time.time() - start_time

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best CV Score: {best_cv_score:.4f}")
        print(f"Target Score: 0.9000")
        print(f"Gap: {0.9 - best_cv_score:.4f}")
        print(f"Total Training Time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"\nSubmission file: {submission_file}")

        if best_cv_score >= 0.9:
            print("\n[SUCCESS] Target achieved!")
        else:
            print(f"\n[INFO] Need +{0.9 - best_cv_score:.4f} improvement")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
