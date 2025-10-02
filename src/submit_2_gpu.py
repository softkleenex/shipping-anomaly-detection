"""
제출 2/5: 개선된 Feature Engineering (149 피처) - GPU 최적화
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from datetime import datetime
import time

RANDOM_STATE = 42

print("="*70)
print("SUBMISSION 2/5: Advanced Features (149) - GPU Optimized")
print("="*70)

start_time = time.time()

# 엔지니어링된 데이터 로드
print("\nLoading advanced features...")
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_train = pd.read_csv(os.path.join(base_dir, 'data', 'train_advanced_fe.csv'))
X_test = pd.read_csv(os.path.join(base_dir, 'data', 'test_advanced_fe.csv'))
y_train = pd.read_csv(os.path.join(base_dir, 'data', 'y_train.csv')).squeeze()

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Features: {X_train.shape[1]}")

# 스케일링
print("\nScaling...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GPU 최적화 모델
print("\nTraining models (GPU optimized)...")

# LightGBM - GPU
print("  Training LightGBM (GPU)...")
lgb = LGBMClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=50,
    random_state=RANDOM_STATE,
    device='gpu',
    gpu_platform_id=0,
    gpu_device_id=0,
    verbose=-1,
    class_weight='balanced'
)
t1 = time.time()
lgb.fit(X_train_scaled, y_train)
print(f"    Done in {time.time()-t1:.1f}s")

# XGBoost - GPU
print("  Training XGBoost (GPU)...")
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    tree_method='hist',
    device='cuda',
    eval_metric='mlogloss'
)
t2 = time.time()
xgb.fit(X_train_scaled, y_train)
print(f"    Done in {time.time()-t2:.1f}s")

# CatBoost - CPU (GPU OOM 방지)
print("  Training CatBoost (CPU)...")
cat = CatBoostClassifier(
    iterations=500,
    depth=10,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    verbose=False,
    task_type='CPU',
    thread_count=-1,
    auto_class_weights='Balanced'
)
t3 = time.time()
cat.fit(X_train_scaled, y_train)
print(f"    Done in {time.time()-t3:.1f}s")

# 예측
print("\nPredicting...")
pred_lgb = lgb.predict(X_test_scaled)
pred_xgb = xgb.predict(X_test_scaled)
pred_cat = cat.predict(X_test_scaled)

# Soft Voting (수동 구현)
pred_lgb_proba = lgb.predict_proba(X_test_scaled)
pred_xgb_proba = xgb.predict_proba(X_test_scaled)
pred_cat_proba = cat.predict_proba(X_test_scaled)

avg_proba = (pred_lgb_proba + pred_xgb_proba + pred_cat_proba) / 3
predictions = np.argmax(avg_proba, axis=1)

# 제출
submission = pd.read_csv(os.path.join(base_dir, 'open', 'sample_submission.csv'))
submission['target'] = predictions

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = os.path.join(base_dir, 'submissions', f'submit2_advanced_149feat_{timestamp}.csv')
submission.to_csv(filename, index=False)

total_time = time.time() - start_time

print("\n" + "="*70)
print(f"[COMPLETE] Submission 2/5 saved: {filename}")
print(f"Features: 149 (52 original + 97 engineered)")
print(f"Models: LightGBM(GPU) + XGBoost(GPU) + CatBoost(CPU)")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
print("="*70)
