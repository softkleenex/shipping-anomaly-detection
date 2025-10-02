"""
긴급 빠른 학습 및 제출 (마감 임박!)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from datetime import datetime
import time

RANDOM_STATE = 42

print("="*70)
print("QUICK TRAINING - URGENT SUBMISSION")
print("="*70)

# 데이터 로드
print("\nLoading data...")
start = time.time()

try:
    # 엔지니어링된 데이터 사용
    X_train = pd.read_csv('../data/train_advanced_fe.csv')
    X_test = pd.read_csv('../data/test_advanced_fe.csv')
    y_train = pd.read_csv('../data/y_train.csv').squeeze()
    print(f"  Using advanced features: {X_train.shape[1]} features")
except:
    # 기본 데이터로 폴백
    train = pd.read_csv('../open/train.csv')
    test = pd.read_csv('../open/test.csv')
    feature_cols = [col for col in train.columns if col not in ['ID', 'target']]
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]
    print(f"  Using original features: {X_train.shape[1]} features")

# Train/Val split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_train
)

# 스케일링
print("Scaling...")
scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 빠른 모델들 (iterations 줄임)
print("\nTraining models...")

lgb = LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
    device='gpu',
    class_weight='balanced'
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    device='cuda',
    tree_method='hist'
)

cat = CatBoostClassifier(
    iterations=300,
    depth=10,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    verbose=False,
    task_type='CPU',
    auto_class_weights='Balanced'
)

# 학습
models = {'lgb': lgb, 'xgb': xgb, 'cat': cat}

for name, model in models.items():
    print(f"\n  Training {name.upper()}...")
    t = time.time()
    model.fit(X_tr_scaled, y_tr)
    y_pred = model.predict(X_val_scaled)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"    Val F1: {f1:.4f} ({time.time()-t:.1f}s)")

# Ensemble
print("\n  Creating ensemble...")
voting = VotingClassifier(
    estimators=list(models.items()),
    voting='soft',
    n_jobs=-1
)

print("  Training on full data...")
voting.fit(X_train_scaled, y_train)

# 예측
print("\nPredicting...")
predictions = voting.predict(X_test_scaled)

# 제출
submission = pd.read_csv('../open/sample_submission.csv')
submission['target'] = predictions

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'../submissions/quick_ensemble_{timestamp}.csv'
submission.to_csv(filename, index=False)

total_time = time.time() - start

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"Time: {total_time:.1f}s ({total_time/60:.1f}min)")
print(f"File: {filename}")
print("\nNow submit using Dacon API...")
