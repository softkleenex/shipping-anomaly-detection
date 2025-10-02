# 🚀 Dacon API 제출 - 초간단 가이드

## 1️⃣ 설치 (한 번만)

```bash
curl -L "https://bit.ly/3gMPScE" -o dacon_submit_api.whl
unzip dacon_submit_api.whl
pip install dacon_submit_api-0.0.4-py3-none-any.whl
```

## 2️⃣ 제출 코드 (복사해서 사용)

```python
from dacon_submit_api import dacon_submit_api

result = dacon_submit_api.post_submission_file(
    './submission.csv',    # 파일 경로
    'YOUR_TOKEN_HERE',     # Token (https://dacon.io 마이페이지)
    '236590',              # 대회 ID (URL에서 확인)
    '',                    # 팀 이름 (선택)
    'My submission'        # 메모 (선택)
)

print(result)
# 성공: {'isSubmitted': True, 'detail': 'Success'}
```

## 3️⃣ Token 발급

1. https://dacon.io 로그인
2. 마이페이지 → API Token 발급
3. Token 복사 (1회만 표시됨!)

## 4️⃣ 대회 ID 확인

URL에서 숫자 부분:
```
https://dacon.io/competitions/official/236590/overview
                                      ^^^^^^
```

## ✅ 제출 확인

- 내 제출: https://dacon.io/competitions/official/236590/mysubmit
- 리더보드: https://dacon.io/competitions/official/236590/leaderboard

---

**더 자세한 가이드**: [DACON_SUBMIT_GUIDE.md](./DACON_SUBMIT_GUIDE.md)
