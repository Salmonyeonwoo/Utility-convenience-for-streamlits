"""
설정 및 상수 정의 모듈
경로 설정, 다국어 설정, 기본 상수 등을 포함합니다.
"""
import os
from typing import Dict, Any

# ========================================
# 기본 경로/로컬 DB 설정
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "local_db")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
RAG_INDEX_DIR = os.path.join(DATA_DIR, "rag_index")

VOICE_META_FILE = os.path.join(DATA_DIR, "voice_records.json")
SIM_META_FILE = os.path.join(DATA_DIR, "simulation_histories.json")

# 디렉토리 생성은 안전하게 처리 (권한 문제 방지)
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(RAG_INDEX_DIR, exist_ok=True)
except (OSError, PermissionError):
    pass

# ========================================
# 다국어 설정
# ========================================
DEFAULT_LANG = "ko"

# LANG 딕셔너리는 너무 크므로 별도 파일로 분리하는 것을 권장합니다.
# 여기서는 기본 구조만 정의하고, 실제 다국어 데이터는 utils/i18n.py로 이동 가능합니다.



































