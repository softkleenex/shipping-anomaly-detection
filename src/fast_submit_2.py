"""
제출 2/5: 개선된 Feature Engineering (149 피처) - CPU 고속
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

RANDOM_STATE = 42

print("="*70)
print("SUBMISSION 2/5: Advanced Features (149) - CPU Fast")
print("="*70)

# 엔지니어링된 데이터 로드
print("\nLoading advanced features...")
X_train = pd.read_csv('../data/train_advanced_fe.csv')
X_test = pd.read_csv('../data/test_advanced_fe.csv')
y_train = pd.read_csv('../data/y_train.csv').squeeze()

print(f"Features: {X_train.shape[1]}")

# 스케일링
print("Scaling...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CPU 최적화 모델 (빠른 학습)
print("\nTraining models (CPU optimized)...")

lgb = LGBMClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
    class_weight='balanced'
)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist'
)

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

# Voting Ensemble
voting = VotingClassifier(
    estimators=[('lgb', lgb), ('xgb', xgb), ('cat', cat)],
    voting='soft',
    n_jobs=-1
)

print("Training ensemble...")
voting.fit(X_train_scaled, y_train)

# 예측
print("Predicting...")
predictions = voting.predict(X_test_scaled)

# 제출
submission = pd.read_csv('../open/sample_submission.csv')
submission['target'] = predictions

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'../submissions/submit2_advanced_fe_{timestamp}.csv'
submission.to_csv(filename, index=False)

print(f"\n[COMPLETE] Submission 2/5 saved: {filename}")
print(f"Features: 149 (52 original + 97 engineered)")
print(f"Models: LightGBM + XGBoost + CatBoost")
