# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
설정 및 상수 모듈
경로 설정, API 설정, 기본값 등을 관리합니다.
"""

import os
try:
    from dotenv import load_dotenv
    _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_file):
        load_dotenv(_env_file, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass

# ⭐ OpenMP 라이브러리 충돌 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ========================================
# 기본 경로/로컬 DB 설정
# ========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "local_db")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
RAG_INDEX_DIR = os.path.join(DATA_DIR, "rag_index")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
PRODUCT_IMAGE_DIR = os.path.join(DATA_DIR, "product_images")

# 파일 경로
VOICE_META_FILE = os.path.join(DATA_DIR, "voice_records.json")
SIM_META_FILE = os.path.join(DATA_DIR, "simulation_histories.json")
VIDEO_MAPPING_DB_FILE = os.path.join(DATA_DIR, "video_mapping_database.json")
FAQ_DB_FILE = os.path.join(DATA_DIR, "faq_database.json")
PRODUCT_IMAGE_CACHE_FILE = os.path.join(DATA_DIR, "product_image_cache.json")

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
    "github": {
        "label": "GitHub Personal Access Token",
        "secret_key": "GITHUB_API_KEY",
        "session_key": "user_github_key",
        "placeholder": "ghp_**************************",
    },
}

# 기본 언어 설정
DEFAULT_LANG = "ko"

# ========================================
# Streamlit 설정 함수들 (참고용 app.py와의 호환성)
# ========================================
import streamlit as st


def init_page_config():
    """페이지 설정 초기화"""
    st.set_page_config(
        page_title="AI 고객응대 시뮬레이터",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def init_session_state():
    """세션 상태 초기화"""
    if 'user_type' not in st.session_state:
        st.session_state.user_type = "operator"  # 기본값: 상담원 모드
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"  # 기본값: 홈 페이지
    if 'selected_customer_id' not in st.session_state:
        st.session_state.selected_customer_id = None
    if 'last_message_id' not in st.session_state:
        st.session_state.last_message_id = {}
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'language' not in st.session_state:
        st.session_state.language = 'ko'  # 기본 언어: 한국어


def get_api_key(api_name="openai"):
    """API 키 가져오기 (Streamlit Secrets > 환경변수 > 세션 상태 순서)"""
    # ⭐ 1. Streamlit Secrets (.streamlit/secrets.toml) - 최우선
    try:
        if hasattr(st, "secrets"):
            secret_key_name = f"{api_name.upper()}_API_KEY"
            if secret_key_name in st.secrets:
                return st.secrets[secret_key_name]
            # OpenAI API 키도 확인 (fallback)
            if api_name != "openai" and "OPENAI_API_KEY" in st.secrets:
                return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    
    # 2. Environment Variable (os.environ)
    env_key = os.getenv(f"{api_name.upper()}_API_KEY", "")
    if not env_key and api_name != "openai":
        # 다른 API의 경우 OPENAI_API_KEY도 확인 (fallback)
        env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    
    # 3. User Input (Session State)
    session_key = st.session_state.get(f"user_{api_name}_key", "")
    if session_key:
        return session_key
    
    # 4. llm_client에서 가져오기 시도
    try:
        from llm_client import get_api_key as llm_get_api_key
        return llm_get_api_key(api_name)
    except ImportError:
        pass
    
    return ""


def get_css_styles():
    """CSS 스타일 반환 (Chatstack 스타일)"""
    return """
    <style>
        /* Chatstack 스타일: 다크 사이드바 및 고객 목록 */
        .dark-sidebar {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            min-height: 100vh;
        }
        .dark-sidebar h3, .dark-sidebar h4 {
            color: white;
        }
        .dark-sidebar .stButton>button {
            background-color: #34495e;
            color: white;
            border: none;
            width: 100%;
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
            transition: background-color 0.3s;
        }
        .dark-sidebar .stButton>button:hover {
            background-color: #4a5f7a;
        }
        .customer-list-dark {
            background-color: #34495e;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .customer-list-dark h4 {
            color: #ecf0f1;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
            margin-bottom: 15px;
        }
        .customer-item-dark {
            background-color: #2c3e50;
            padding: 12px;
            margin: 8px 0;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s;
            color: white;
        }
        .customer-item-dark:hover {
            background-color: #3d5468;
        }
        .customer-item-dark.selected {
            background-color: #3498db;
        }
        .unread-badge {
            background-color: #e74c3c;
            color: white;
            border-radius: 50%;
            padding: 2px 6px;
            font-size: 0.75em;
            margin-left: 8px;
        }
        .agent-profile {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #4a5f7a;
        }
        .agent-profile .status-online {
            color: #2ecc71;
        }
        
        /* 채팅 메시지 스타일 (Chatstack 스타일) */
        .message-operator {
            background-color: #e8e8e8;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            text-align: left;
            max-width: 70%;
            margin-left: auto;
            margin-right: 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            clear: both;
            float: right;
            position: relative;
        }
        .message-operator::before {
            content: '';
            position: absolute;
            right: -8px;
            bottom: 12px;
            width: 0;
            height: 0;
            border-style: solid;
            border-width: 8px 0 8px 8px;
            border-color: transparent transparent transparent #e8e8e8;
        }
        .message-customer {
            background-color: #e3f2fd;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            text-align: left;
            max-width: 70%;
            margin-left: 0;
            margin-right: auto;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            clear: both;
            float: left;
            position: relative;
        }
        .message-customer::before {
            content: '';
            position: absolute;
            left: -8px;
            bottom: 12px;
            width: 0;
            height: 0;
            border-style: solid;
            border-width: 8px 8px 8px 0;
            border-color: transparent #e3f2fd transparent transparent;
        }
        .chat-container {
            overflow-y: auto;
            padding: 20px;
            background-color: #ffffff;
            min-height: 400px;
        }
        .chat-container::after {
            content: "";
            display: table;
            clear: both;
        }
        .message-ai-suggestion {
            background-color: #fff3cd;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 70%;
            border-left: 4px solid #ffc107;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* 채팅 입력창 스타일 */
        .chat-input-container {
            padding: 15px;
            background-color: #f8f9fa;
            border-top: 1px solid #e0e0e0;
        }
        .chat-input-icons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .chat-icon-button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s;
        }
        .chat-icon-button:hover {
            background-color: #2980b9;
        }
        
        /* 고객 정보 패널 스타일 */
        .customer-info-panel {
            background-color: #ffffff;
            padding: 20px;
            border-left: 1px solid #e0e0e0;
        }
        .customer-info-item {
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .customer-info-item:last-child {
            border-bottom: none;
        }
        .customer-info-label {
            font-size: 0.85em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .customer-info-value {
            color: #2c3e50;
            font-weight: 500;
        }
    </style>
    """
