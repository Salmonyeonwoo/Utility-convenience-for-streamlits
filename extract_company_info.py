# 회사 정보 탭을 _pages/_company_info.py로 추출하는 스크립트
import re

with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 회사 정보 탭 범위: 866-1917줄
start_line = 866
end_line = 1917

# 회사 정보 탭 코드 추출
company_info_code = ''.join(lines[start_line-1:end_line])

# 모듈 파일 생성
module_content = f"""# ========================================
# _pages/_company_info.py
# 회사 정보 탭 모듈
# ========================================

import streamlit as st
import os
import base64
import tempfile
import requests
from lang_pack import LANG
from faq_manager import (
    load_faq_database, save_faq_database, get_company_info_faq,
    get_product_image_url, generate_company_info_with_llm
)
from llm_client import get_api_key, run_llm
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai


def render_company_info():
    \"\"\"회사 정보 탭 렌더링 함수\"\"\"
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])

{company_info_code}
"""

with open('_pages/_company_info.py', 'w', encoding='utf-8') as f:
    f.write(module_content)

print(f"OK: Company info module created: _pages/_company_info.py ({end_line-start_line+1} lines)")

