# 스마트 해운물류 x AI 미션 챌린지 : 이상신호 감지 기반 비정상 작동 진단

해양수산부 주최 장비 센서 데이터 기반 고장 진단 머신러닝 프로젝트

## 프로젝트 개요

장비에서 수집된 52개 센서 신호 데이터를 분석하여 21가지 정상/비정상 작동 유형을 분류하는 머신러닝 모델을 개발했습니다. 블랙박스 환경에서 도메인 지식 없이 순수 데이터 분석과 Feature Engineering만으로 성능을 최적화하는 것이 핵심 과제였습니다.

### 주요 성과

- **최종 순위**: 236등 / 1,000+명
- **최종 점수**: Macro-F1 0.75515
- **평가지표**: Macro-F1 Score
- **데이터**: 21,693개 학습 샘플, 15,004개 테스트 샘플
- **특징**: 52개 센서 신호 (X_01 ~ X_52)
- **목표**: 21개 클래스 분류

## 기술 스택

- **Python 3.12**
- **ML Framework**: scikit-learn, LightGBM, XGBoost, CatBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **GPU Acceleration**: CUDA 12.6, RTX 4060

## 프로젝트 구조

```
Shipping_logistics_abnormal_signal/
├── notebooks/                          # 탐색적 분석 및 모델링
│   ├── 01_EDA.ipynb                   # 데이터 탐색 및 시각화
│   ├── 02_baseline_model.ipynb        # 베이스라인 모델 구축
│   └── 03_feature_engineering_ensemble.ipynb  # 고급 피처 엔지니어링
│
├── src/                                # 실행 가능한 스크립트
│   ├── advanced_feature_engineering.py # 149개 고급 피처 생성
│   ├── deep_analysis.py               # 성능 분석 및 개선점 도출
│   ├── train_improved_model.py        # GPU 최적화 모델 학습
│   └── multi_submit.py                # Dacon API 자동 제출
│
├── data/                               # 데이터 저장소
├── submissions/                        # 제출 파일
├── requirements.txt                    # 의존성 패키지
└── README.md
```

## 주요 기능

### 1. 탐색적 데이터 분석 (EDA)

- 52개 센서 신호의 분포 및 상관관계 분석
- 클래스 불균형 확인 (완벽하게 균형잡힌 21개 클래스)
- 이상치 및 결측치 탐지
- 저분산 피처 및 고상관 피처 식별

**주요 발견**:
- 29개 저분산 피처 발견
- 29쌍의 고상관 피처 쌍 (r > 0.95)
- 주요 판별 피처: X_19, X_37, X_40, X_11, X_28

### 2. Feature Engineering

기본 52개 피처에서 149개 고급 피처로 확장:

```python
# Interaction Features (30개)
- 주요 판별 피처 간 곱셈, 뺄셈, 나눗셈 조합

# Statistical Features (23개)
- 행 단위 평균, 표준편차, 최대/최소, 왜도, 첨도
- 분위수 (Q25, Q75, IQR)

# Polynomial Features (10개)
- 주요 피처의 2차 다항식

# Clustering Features (54개)
- KMeans (k=5, 10, 15, 21)
- 클러스터 레이블 및 거리

# Dimensionality Reduction (30개)
- PCA 30 components

# Feature Selection
- 고상관 피처 4개 제거 → 최종 149개
```

### 3. 모델 앙상블

**GPU 최적화 Voting Ensemble**:

```python
# LightGBM (GPU)
- n_estimators: 800
- max_depth: 12
- device: 'gpu'
- class_weight: 'balanced'

# XGBoost (GPU)
- n_estimators: 800
- max_depth: 12
- tree_method: 'hist'
- device: 'cuda'

# CatBoost (CPU)
- iterations: 1000
- depth: 10
- task_type: 'CPU'  # GPU OOM 방지
- auto_class_weights: 'Balanced'
```

**Soft Voting**: 3개 모델의 확률 평균

### 4. 성능 분석

```
Baseline F1 Score: 0.8004
Target F1 Score: 0.9000
Gap: 0.0996

Class-wise Performance:
- Best: Class 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20 (F1 > 0.90)
- Worst: Class 9 (0.2056), 0 (0.2489), 15 (0.2511), 19 (0.4474), 3 (0.5105)
```

## 설치 및 실행

### 환경 설정

```bash
# 저장소 클론
git clone https://github.com/yourusername/shipping-anomaly-detection.git
cd shipping-anomaly-detection

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 데이터 준비

```bash
# Dacon에서 데이터 다운로드
# https://dacon.io/competitions/official/236590/data

# 데이터 배치
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 실행 방법

```bash
# 1. 고급 피처 생성
python src/advanced_feature_engineering.py

# 2. 모델 학습 및 예측
python src/train_improved_model.py

# 3. Dacon 제출 (선택)
python src/multi_submit.py
```

### Jupyter 노트북 실행

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

## 핵심 인사이트

### 1. 블랙박스 환경에서의 전략

도메인 지식이 차단된 상황에서:
- 통계적 패턴 분석에 집중
- 피처 간 상호작용 탐색
- 클러스터링을 통한 숨겨진 패턴 발견

### 2. GPU 메모리 최적화

```python
# CatBoost GPU OOM 문제 해결
# 149 피처 × 21,693 샘플 → 11.8GB 요구
# 해결: CatBoost만 CPU 모드, 나머지는 GPU
```

### 3. 불균형 처리

완벽하게 균형잡힌 데이터셋이지만:
- 클래스별 성능 편차 큼 (F1: 0.20 ~ 0.95)
- `class_weight='balanced'` 적용
- Macro-F1로 평가하여 모든 클래스 동등하게 중요

### 4. Feature Engineering의 중요성

```
52개 피처 → 149개 피처
Baseline F1: 0.8004 → Target: 0.9000
개선 여지: 상호작용 피처, 클러스터링, PCA
```

## 개선 방향

- [ ] 문제 클래스(0, 3, 9, 15, 19) 타겟 학습
- [ ] Stacking Ensemble 시도
- [ ] Optuna 하이퍼파라미터 튜닝
- [ ] Pseudo Labeling
- [ ] TabNet 등 Neural Network 모델

## 배운 점

1. **데이터 중심 접근**: 도메인 지식 없이도 데이터 분석만으로 성능 개선 가능
2. **GPU 최적화**: 메모리 제약 상황에서 하이브리드 전략 (GPU+CPU)
3. **앙상블의 힘**: 3개 모델 조합으로 안정적 성능
4. **Feature Engineering**: 피처 수 증가가 항상 성능 향상을 보장하지 않음 (선택이 중요)

## 라이선스

MIT License

## 참고 자료

- [대회 링크](https://dacon.io/competitions/official/236590/overview/description)
- [Macro-F1 Score](https://en.wikipedia.org/wiki/F-score)
- [LightGBM GPU 튜토리얼](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)

---

**개발 기간**: 2025.09.08 - 2025.10.02
**주최**: 해양수산부, 울산항만공사, 한국정보산업연합회
**운영**: 데이콘
