# 고객 상담 및 Solved 티켓 KPI 관리 시스템

## 📋 개요
고객 상담 데이터를 관리하고 KPI를 추적하는 Streamlit 기반 CRM 시스템입니다.

## 🚀 로컬 실행

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 앱 실행
```bash
streamlit run customer_CRM_&_ticket_system.py
```

### 3. 자동 폴더 스캔
로컬 Windows 환경에서는 다음 폴더가 자동으로 스캔됩니다:
- `C:\Users\Admin\Downloads\Updated_streamlit_app_files\customer data histories via streamlits`
- `C:\Users\Admin\OneDrive\ドキュメント\Yeonwoo_streamlit_app_test\customer data histories via streamlits (practicing)`

## 🌐 GitHub 배포 (Streamlit Cloud)

### 1. 환경 변수 설정
GitHub 배포 시 로컬 절대 경로는 작동하지 않으므로 환경 변수를 사용해야 합니다.

#### 방법 1: Streamlit Cloud Secrets
1. Streamlit Cloud 대시보드에서 앱 선택
2. Settings → Secrets → Add new secret
3. 다음 추가:
```
CRM_DATA_FOLDERS = /mount/data/folder1;/mount/data/folder2
```

#### 방법 2: 환경 변수
배포 플랫폼에서 환경 변수 설정:
```bash
export CRM_DATA_FOLDERS="/path/to/folder1;/path/to/folder2"
```

### 2. 데이터 폴더 마운트 (Streamlit Cloud)
Streamlit Cloud에서 데이터 폴더를 마운트하려면:
1. Settings → General → Mount data directory
2. 마운트 경로 설정 (예: `/mount/data`)

### 3. 폴더 경로 설정
마운트된 폴더 경로를 Secrets에 설정:
```
CRM_DATA_FOLDERS = /mount/data/customer_histories
```

## 📁 폴더 구조
```
Updated_streamlit_app_files/
├── customer_CRM_&_ticket_system.py  # 메인 앱
├── crm_manager.py                    # DB 관리 모듈
├── file_parser.py                    # 파일 파싱 모듈
├── file_importer.py                  # 파일 임포트 모듈
├── requirements.txt                   # 패키지 목록
├── data/                             # 데이터 저장 폴더
│   ├── crm_db.json                  # CRM 데이터베이스
│   └── scanned_files.json           # 스캔된 파일 추적
└── .streamlit/
    └── secrets.toml.example         # Secrets 설정 예제
```

## ⚙️ 설정 파일

### config_example.env
환경 변수 설정 예제 파일입니다. `.env`로 복사하여 사용하세요.

### .streamlit/secrets.toml.example
Streamlit Cloud Secrets 설정 예제입니다. `.streamlit/secrets.toml`로 복사하여 사용하세요.

## 🔧 주요 기능

1. **자동 카운팅**: 앱 실행 시 지정된 폴더를 자동으로 스캔
2. **파일 파싱**: PDF, Word, PPTX, JSON, CSV 파일 지원
3. **중복 방지**: 이미 스캔한 파일은 건너뛰기
4. **KPI 추적**: Solved 티켓 수, CSAT 점수, 고객별 통계

## 📝 주의사항

- 로컬 환경과 GitHub 배포 환경의 폴더 경로가 다릅니다
- GitHub 배포 시 반드시 환경 변수나 Secrets를 설정해야 합니다
- 절대 경로는 로컬 Windows 환경에서만 작동합니다





