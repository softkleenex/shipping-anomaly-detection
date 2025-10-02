# Data Directory

이 폴더는 데이터 파일을 저장합니다.

## 필수 파일

Dacon 대회 페이지에서 다운로드:
https://dacon.io/competitions/official/236590/data

```
data/
├── train.csv              # 학습 데이터 (21,693 rows, 54 columns)
├── test.csv               # 테스트 데이터 (15,004 rows, 53 columns)
└── sample_submission.csv  # 제출 양식
```

## 생성되는 파일

Feature Engineering 스크립트 실행 후:

```
data/
├── train_advanced_fe.csv  # 149개 고급 피처 (학습용)
├── test_advanced_fe.csv   # 149개 고급 피처 (테스트용)
└── y_train.csv           # 타겟 변수
```

## 주의사항

- 모든 CSV 파일은 `.gitignore`에 포함되어 GitHub에 업로드되지 않습니다
- 데이터는 Dacon에서 직접 다운로드해야 합니다
