# Submissions Directory

이 폴더는 Dacon 제출용 CSV 파일을 저장합니다.

## 제출 파일 형식

```csv
ID,target
TEST_00000,5
TEST_00001,12
...
```

- **ID**: 테스트 샘플 고유 ID
- **target**: 예측한 클래스 (0-20)

## 주의사항

- 모든 제출 파일은 `.gitignore`에 포함되어 GitHub에 업로드되지 않습니다
- 일일 최대 5회 제출 가능
- Macro-F1 Score로 평가됩니다
