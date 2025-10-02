"""
빠른 제출용 - 기본 피처 + CPU 최적화
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from datetime import datetime
import time
import os

start = time.time()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading data...")
train = pd.read_csv(os.path.join(base_dir, 'open', 'train.csv'))
test = pd.read_csv(os.path.join(base_dir, 'open', 'test.csv'))

feature_cols = [col for col in train.columns if col not in ['ID', 'target']]
X_train = train[feature_cols]
y_train = train['target']
X_test = test[feature_cols]

print("Scaling...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training LightGBM (CPU)...")
model = LGBMClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

print("Predicting...")
predictions = model.predict(X_test_scaled)

submission = pd.read_csv(os.path.join(base_dir, 'open', 'sample_submission.csv'))
submission['target'] = predictions

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = os.path.join(base_dir, 'submissions', f'submit2_fast_cpu_{timestamp}.csv')
submission.to_csv(filename, index=False)

print(f"Complete! {time.time()-start:.1f}s - {filename}")
