"""
Google Drive 업로드 스크립트
실행 전 필요: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import pickle

# Google Drive API 스코프
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def upload_to_drive(file_path, folder_name='Dacon_Submissions'):
    """파일을 Google Drive에 업로드"""

    creds = None
    token_path = 'token.pickle'

    # 저장된 토큰 로드
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # 유효하지 않으면 로그인
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # credentials.json 파일이 필요함 (Google Cloud Console에서 다운로드)
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # 토큰 저장
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    # Drive API 서비스 생성
    service = build('drive', 'v3', credentials=creds)

    # 폴더 찾기 또는 생성
    folder_id = None
    results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        spaces='drive',
        fields='files(id, name)'
    ).execute()

    items = results.get('files', [])
    if items:
        folder_id = items[0]['id']
    else:
        # 폴더 생성
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')

    # 파일 업로드
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()

    print(f"✅ 업로드 완료!")
    print(f"파일 ID: {file.get('id')}")
    print(f"링크: {file.get('webViewLink')}")

    return file.get('webViewLink')

if __name__ == '__main__':
    submission_file = '../submissions/final_ensemble_gpu_20250928_154041.csv'

    print("Google Drive 업로드 시작...")
    print(f"파일: {submission_file}")

    try:
        link = upload_to_drive(submission_file)
        print(f"\n공유 링크: {link}")
    except FileNotFoundError:
        print("❌ credentials.json 파일이 필요합니다.")
        print("Google Cloud Console에서 OAuth 2.0 클라이언트 ID를 생성하고")
        print("credentials.json 파일을 다운로드하여 src/ 폴더에 저장하세요.")
        print("\n참고: https://developers.google.com/drive/api/quickstart/python")
    except Exception as e:
        print(f"❌ 오류: {e}")
