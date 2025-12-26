# 세션 초기화와 사이드바를 모듈로 추출하는 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 세션 초기화 부분 찾기 (444줄부터 시작, 약 250줄)
session_start = 444
session_end = 790

# 사이드바 부분 찾기 (790줄부터 시작, 약 210줄)
sidebar_start = 790
sidebar_end = 1000

# 세션 초기화 코드 추출
session_code = ''.join(lines[session_start-1:sidebar_start-1])

session_module = f"""# ========================================
# _pages/_session_init.py
# 세션 상태 초기화 모듈
# ========================================

import streamlit as st
from datetime import datetime, timedelta
import uuid
import hashlib
from lang_pack import LANG, DEFAULT_LANG as LANG_DEFAULT
from config import SUPPORTED_APIS, DEFAULT_LANG
from llm_client import get_api_key, get_llm_client, init_openai_audio_client
from _pages._classes import CallHandler, AppAudioHandler, CustomerDataManager

def init_session_state():
    \"\"\"세션 상태 초기화 함수\"\"\"
{session_code}
"""

with open('_pages/_session_init.py', 'w', encoding='utf-8') as f:
    f.write(session_module)
print(f"OK: Session init module created: _pages/_session_init.py ({sidebar_start-session_start} lines)")

# 사이드바 코드 추출
sidebar_code = ''.join(lines[sidebar_start-1:sidebar_end])

sidebar_module = f"""# ========================================
# _pages/_sidebar.py
# 사이드바 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from llm_client import get_api_key
from config import SUPPORTED_APIS

def render_sidebar():
    \"\"\"사이드바 렌더링 함수\"\"\"
{sidebar_code}
"""

with open('_pages/_sidebar.py', 'w', encoding='utf-8') as f:
    f.write(sidebar_module)
print(f"OK: Sidebar module created: _pages/_sidebar.py ({sidebar_end-sidebar_start} lines)")

print("Session init and sidebar modules extracted successfully!")








