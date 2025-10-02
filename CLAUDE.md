# 이상신호 감지 기반 비정상 작동 진단 경진대회

## 대회 개요
- **대회 URL**: https://dacon.io/competitions/official/236590/overview/description
- **주최**: 해양수산부
- **주관**: 울산항만공사/한국정보산업연합회
- **운영**: 데이콘

## 대회 목표
장비 센서 데이터를 기반으로 장비의 정상/비정상 작동 유형을 분류하는 모델 개발
- 센서 간 관계와 미세한 변화 포착
- 신속한 점검·보전을 돕는 현장 활용형 진단기 설계

## 데이터 특징
- **블랙박스 환경**: 도메인 의미가 차단된 비식별화 데이터
- **특징**: X_01 ~ X_52 (52개의 센서/제어 신호)
- **목표**: 정상/비정상 유형 분류 (다중 클래스 분류)
- **참고**: 각 행은 동일 시점의 상태, 타임스탬프·시퀀스 정보 미제공

## 평가 지표
- **Metric**: Macro-F1 Score
- **Public Score**: 테스트 데이터의 30% 샘플
- **Private Score**: 테스트 데이터의 100%

## 주요 일정
- **대회 시작**: 2025년 09월 08일(월) 10:00
- **팀 병합 마감**: 2025년 09월 25일(목) 23:59
- **대회 종료**: 2025년 10월 02일(목) 10:00
- **코드 및 PPT 제출 마감**: 2025년 10월 10일(금)
- **코드 검증**: 2025년 10월 13일(월) ~ 10월 17일(금)
- **본선 진출자 발표**: 2025년 10월 20일(월) 10:00

## 제출 규칙
- **일일 최대 제출 횟수**: 5회
- **사용 가능 언어**: Python
- **최종 제출**: 제출 파일 중 1개 선택

## 코드 제출 양식
- 데이터 입/출력 경로: `/data`
- 파일 확장자: `.py`, `.ipynb`
- 인코딩: UTF-8
- 전체 프로세스를 하나의 파일로 정리
- 개발 환경 및 라이브러리 버전 명시
- 모델 파일(`.pth` 등) 제공
- 솔루션 PPT 필수 제출

## 외부 데이터 및 모델 규칙
### 허용
- 공식 공개된 사전학습 모델 (MIT, Apache 2.0 등 오픈소스 라이선스)
- 로컬 환경에서 실행 가능한 모델

### 금지
- 외부 데이터 사용
- API 기반 모델 (OpenAI API, Gemini API 등)
- 원격 서버 의존 모델

## 작업 가이드라인

### 1. 프로젝트 구조
```
Shipping_logistics_abnormal_signal/
├── open/                    # 원본 데이터
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── data/                    # 처리된 데이터
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── EDA.ipynb
│   └── modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── utils.py
├── models/
│   └── saved_models/
├── submissions/
│   └── submission_files/
└── requirements.txt
```

### 2. 주요 작업 단계
1. **EDA (Exploratory Data Analysis)**
   - 데이터 분포 확인
   - 결측치 및 이상치 탐지
   - 센서 간 상관관계 분석
   - 클래스 불균형 확인

2. **Feature Engineering**
   - 통계적 특징 생성 (mean, std, min, max, etc.)
   - 시계열 특징 추출 (lag, rolling statistics)
   - 센서 간 상호작용 특징
   - 차원 축소 (PCA, t-SNE)

3. **Model Development**
   - Baseline 모델 구축
   - 다양한 알고리즘 실험 (RF, XGBoost, LightGBM, CatBoost, Neural Networks)
   - 하이퍼파라미터 튜닝
   - 앙상블 방법 적용

4. **Validation Strategy**
   - Stratified K-Fold Cross Validation
   - Time-based validation (if applicable)
   - Macro-F1 최적화

5. **Post-processing**
   - Threshold optimization
   - Prediction calibration

### 3. 코드 실행 명령어
```bash
# 데이터 전처리
python src/preprocessing.py

# 특징 생성
python src/feature_engineering.py

# 모델 학습
python src/model.py --train

# 예측 생성
python src/model.py --predict

# 제출 파일 생성
python src/utils.py --create-submission
```

### 4. 주의사항
- 데이터 누수(Data Leakage) 방지
- 재현 가능한 코드 작성 (random seed 고정)
- 명확한 주석과 문서화
- 코드 모듈화 및 함수화
- 제출 전 코드 검증

## 참고 자료
- [Macro-F1 Score 이해](https://en.wikipedia.org/wiki/F-score)
- [Imbalanced Classification 기법](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- [Time Series Feature Engineering](https://www.kaggle.com/learn/time-series)

## 데이터 정보

### 파일 구조
- **train.csv**: 학습 데이터
  - ID: 샘플별 고유 ID
  - X_01 ~ X_52: 센서/제어 신호 (비식별화)
  - target: 고장 진단 라벨 (의미 비공개)

- **test.csv**: 테스트 데이터
  - ID: 샘플별 고유 ID
  - X_01 ~ X_52: 센서/제어 신호

- **sample_submission.csv**: 제출 양식
  - ID: 샘플별 고유 ID
  - target: 예측할 고장 진단 라벨

### 데이터 특이사항
- 스케일·분포는 피처별로 상이할 수 있음
- Excel로 열람 시 비정상적 표시 가능 (Pandas 사용 권장)
- 센서/제어 신호가 혼재되어 있으나 구체적 매핑은 비공개

## 프로젝트 현황

### 완료된 작업
- [x] 데이터 다운로드 및 확인
- [x] 프로젝트 구조 설정
- [x] EDA 노트북 생성 (01_EDA.ipynb)
- [x] Baseline 모델 구축 (02_baseline_model.ipynb)
- [x] Feature Engineering 및 앙상블 (03_feature_engineering_ensemble.ipynb)
- [x] 실행 스크립트 생성 (src/main_pipeline.py, src/run_final_model.py)

### 개발된 모델
1. **Baseline Models**: RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost
2. **Advanced Models**: 하이퍼파라미터 최적화 모델
3. **Ensemble**: Voting Classifier, Stacking Classifier

### 주요 Feature Engineering
- 통계적 특징 (mean, std, max, min, range, skew, kurtosis)
- 비율 특징 (mean/std, max/min)
- 분위수 특징 (Q25, Q75, IQR)
- 카운트 특징 (positive, negative, zero counts)
- PCA 특징 (20 components)

## 빠른 실행 가이드

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 필요 패키지 설치
pip install -r requirements.txt
```

### 2. 최종 모델 실행
```bash
cd src
python run_final_model.py
```

### 3. 전체 파이프라인 실행 (선택적)
```bash
cd src
python main_pipeline.py
```

### 4. Jupyter 노트북 실행
```bash
cd notebooks
jupyter notebook
# 01_EDA.ipynb → 02_baseline_model.ipynb → 03_feature_engineering_ensemble.ipynb 순서로 실행
```

## 성능 목표
- **목표**: Macro-F1 Score > 0.9
- **현재 예상 성능**: 0.85 ~ 0.92 (데이터 특성에 따라 변동)

## 개선 방향
1. 추가 Feature Engineering (시간 도메인, 클러스터링)
2. Pseudo Labeling
3. Test Time Augmentation
4. Neural Network 모델 추가 (TabNet)
5. Post-processing 최적화

## Dacon API 제출 가이드

**자세한 가이드**: [DACON_SUBMIT_GUIDE.md](./DACON_SUBMIT_GUIDE.md)

### 빠른 사용법
```python
from dacon_submit_api import dacon_submit_api

result = dacon_submit_api.post_submission_file(
    '파일경로',
    'YOUR_TOKEN',
    '236590',
    '팀이름',
    '메모'
)
```

### API 설치
```bash
curl -L "https://bit.ly/3gMPScE" -o dacon_submit_api.whl
unzip dacon_submit_api.whl
pip install dacon_submit_api-0.0.4-py3-none-any.whl
```

## 체크리스트
- [x] 데이터 다운로드 및 확인
- [x] EDA 완료
- [x] Feature Engineering 완료
- [x] Baseline 모델 구축
- [x] 모델 개선 및 앙상블
- [x] 코드 정리 및 문서화
- [x] 최종 제출 파일 생성
- [x] **Dacon API로 제출 완료** ✅
- [ ] 리더보드 점수 확인
- [ ] PPT 작성
- [ ] 코드 검증 준비