"""
설정 및 상수 정의 모듈
"""
import os

# ========================================
# 기본 경로/로컬 DB 설정
# ========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "local_db")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
RAG_INDEX_DIR = os.path.join(DATA_DIR, "rag_index")

VOICE_META_FILE = os.path.join(DATA_DIR, "voice_records.json")
SIM_META_FILE = os.path.join(DATA_DIR, "simulation_histories.json")
VIDEO_MAPPING_DB_FILE = os.path.join(DATA_DIR, "video_mapping_database.json")
FAQ_DB_FILE = os.path.join(DATA_DIR, "faq_database.json")
PRODUCT_IMAGE_CACHE_FILE = os.path.join(DATA_DIR, "product_image_cache.json")
PRODUCT_IMAGE_DIR = os.path.join(DATA_DIR, "product_images")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRODUCT_IMAGE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RAG_INDEX_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ========================================
# API 설정
# ========================================

SUPPORTED_APIS = {
    "openai": {
        "label": "OpenAI API Key",
        "secret_key": "OPENAI_API_KEY",
        "session_key": "user_openai_key",
        "placeholder": "sk-proj-**************************",
    },
    "gemini": {
        "label": "Google Gemini API Key",
        "secret_key": "GEMINI_API_KEY",
        "session_key": "user_gemini_key",
        "placeholder": "AIza***********************************",
    },
    "hyperclova": {
        "label": "Hyperclova API Key",
        "secret_key": "HYPERCLOVA_API_KEY",
        "session_key": "user_hyperclova_key",
        "placeholder": "hyperclova-**************************",
    },
    "nvidia": {
        "label": "NVIDIA NIM API Key",
        "secret_key": "NVIDIA_API_KEY",
        "session_key": "user_nvidia_key",
        "placeholder": "nvapi-**************************",
    },
    "claude": {
        "label": "Anthropic Claude API Key",
        "secret_key": "CLAUDE_API_KEY",
        "session_key": "user_claude_key",
        "placeholder": "sk-ant-api-**************************",
    },
    "groq": {
        "label": "Groq API Key",
        "secret_key": "GROQ_API_KEY",
        "session_key": "user_groq_key",
        "placeholder": "gsk_**************************",
    },
}

DEFAULT_LANG = "ko"









