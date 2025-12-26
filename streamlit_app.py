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

# ========================================
# streamlit_app.py (메인 진입점)
# ========================================

import streamlit as st
import os
from streamlit_app_imports import (
    CHAT_SIMULATOR_AVAILABLE, COMPANY_INFO_AVAILABLE, PHONE_SIMULATOR_AVAILABLE,
    RAG_AVAILABLE, CONTENT_AVAILABLE, SIDEBAR_AVAILABLE,
    render_chat_simulator, render_company_info, render_phone_simulator,
    render_rag, render_content, render_sidebar
)
from streamlit_app_session_init import init_all_session_state
from config import DATA_DIR, PRODUCT_IMAGE_DIR, AUDIO_DIR, RAG_INDEX_DIR, VIDEO_DIR, DEFAULT_LANG
from lang_pack import LANG

# ========================================
# Streamlit 페이지 설정
# ========================================
st.set_page_config(
    page_title="AI Study Coach & Customer Service Simulator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# ========================================
# 기본 경로/로컬 DB 설정
# ========================================
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRODUCT_IMAGE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RAG_INDEX_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ========================================
# 세션 상태 초기화
# ========================================
init_all_session_state()

# ========================================
# 사이드바 렌더링
# ========================================
if SIDEBAR_AVAILABLE:
    render_sidebar()

# ========================================
# 메인 라우팅
# ========================================
# 언어 키 안전하게 가져오기
current_lang = st.session_state.get("language", "ko")
if current_lang not in ["ko", "en", "ja"]:
    current_lang = "ko"
L = LANG.get(current_lang, LANG["ko"])

# 타이틀과 설명을 한 줄로 간결하게 표시
feature_selection = st.session_state.get(
    "feature_selection", L.get("home_tab", "홈"))

# 참고용 app.py 구조 통합: 홈 페이지 추가
if feature_selection == L.get("home_tab", "홈"):
    try:
        from _pages._reference_home import render_home_page
        if 'selected_customer_id' not in st.session_state:
            st.session_state.selected_customer_id = None
        if 'last_message_id' not in st.session_state:
            st.session_state.last_message_id = {}
        if 'ai_suggestion' not in st.session_state:
            st.session_state.ai_suggestion = {}
        render_home_page()
    except ImportError:
        st.title("📊 대시보드")
        st.info("홈 페이지 모듈을 불러올 수 없습니다.")

elif feature_selection == L.get("chat_email_tab", "채팅/이메일"):
    if CHAT_SIMULATOR_AVAILABLE:
        render_chat_simulator()
    else:
        try:
            from _pages._app_chat_page import render_chat_page
            if 'selected_customer_id' not in st.session_state:
                st.session_state.selected_customer_id = None
            if 'last_message_id' not in st.session_state:
                st.session_state.last_message_id = {}
            if 'ai_suggestion' not in st.session_state:
                st.session_state.ai_suggestion = {}
            render_chat_page()
        except ImportError as e:
            st.error(f"채팅 페이지 모듈을 불러올 수 없습니다: {str(e)}")

elif feature_selection == L.get("phone_tab", "전화"):
    st.markdown(f"### 📞 {L.get('phone_tab', '전화')}")
    st.caption(L.get('sim_tab_phone_desc', '전화 시뮬레이터 기능입니다.'))
    if PHONE_SIMULATOR_AVAILABLE:
        render_phone_simulator()
    else:
        st.error("전화 시뮬레이터 탭 모듈을 찾을 수 없습니다.")

elif feature_selection == L.get("customer_data_tab", "고객 데이터"):
    st.markdown(f"### 📋 고객 데이터 관리")
    st.caption("고객 정보를 등록, 수정, 조회하고 JSON/CSV 형식으로 관리합니다.")
    try:
        from _pages._customer_data import render_customer_data_page
        render_customer_data_page()
    except ImportError:
        st.error("고객 데이터 관리 모듈을 찾을 수 없습니다.")
