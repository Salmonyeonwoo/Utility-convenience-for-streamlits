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

# ⭐ OpenMP 라이브러리 충돌 해결
from visualization import (
    visualize_customer_profile_scores, visualize_similarity_cases,
    visualize_case_trends, visualize_customer_characteristics
)

# 채팅 시뮬레이터 모듈
try:
    from _pages._chat_simulator import render_chat_simulator
    CHAT_SIMULATOR_AVAILABLE = True
except ImportError:
    CHAT_SIMULATOR_AVAILABLE = False
    def render_chat_simulator():
        st.error("채팅 시뮬레이터 모듈을 찾을 수 없습니다.")

# 회사 정보 탭 모듈
try:
    from _pages._company_info import render_company_info
    COMPANY_INFO_AVAILABLE = True
except ImportError:
    COMPANY_INFO_AVAILABLE = False
    def render_company_info():
        st.error("회사 정보 탭 모듈을 찾을 수 없습니다.")

# 전화 시뮬레이터 탭 모듈
try:
    from _pages._phone_simulator import render_phone_simulator
    PHONE_SIMULATOR_AVAILABLE = True
except ImportError:
    PHONE_SIMULATOR_AVAILABLE = False
    def render_phone_simulator():
        st.error("전화 시뮬레이터 탭 모듈을 찾을 수 없습니다.")

# RAG 탭 모듈
try:
    from _pages._rag_page import render_rag_page as render_rag
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    def render_rag():
        st.error("RAG 탭 모듈을 찾을 수 없습니다.")

# 콘텐츠 탭 모듈
try:
    from _pages._content import render_content
    CONTENT_AVAILABLE = True
except ImportError:
    CONTENT_AVAILABLE = False
    def render_content():
        st.error("콘텐츠 탭 모듈을 찾을 수 없습니다.")

# 사이드바 모듈
try:
    from ui.sidebar import render_sidebar
    SIDEBAR_AVAILABLE = True
except ImportError:
    SIDEBAR_AVAILABLE = False
    def render_sidebar():
        st.error("사이드바 모듈을 찾을 수 없습니다.")

# 세션 초기화 모듈
try:
    from _pages._session_init import init_session_state
    SESSION_INIT_AVAILABLE = True
except ImportError:
    SESSION_INIT_AVAILABLE = False
    def init_session_state():
        pass  # 기본 초기화는 이미 streamlit_app.py에서 수행됨

# simulation_handler에서 핵심 함수들 import
from simulation_handler import (
    generate_realtime_hint,
    generate_agent_response_draft, generate_outbound_call_summary,
    get_chat_history_for_prompt, generate_customer_reaction,
    summarize_history_with_ai, generate_customer_reaction_for_call,
    generate_customer_reaction_for_first_greeting, summarize_history_for_call,
    generate_customer_closing_response, generate_agent_first_greeting,
)

# utils 모듈에서 분리된 함수들 import
from utils.translation import translate_text_with_llm
from utils.history_handler import (
    load_simulation_histories_local, generate_chat_summary,
    save_simulation_history_local, export_history_to_word,
    export_history_to_pptx, export_history_to_pdf,
    delete_all_history_local, generate_daily_customer_guide,
    save_daily_customer_guide, recommend_guideline_for_customer,
    get_daily_data_statistics,
)
from utils.audio_handler import (
    transcribe_bytes_with_whisper, transcribe_audio, synthesize_tts,
    render_tts_button, load_voice_records, save_voice_records,
    save_audio_record_local, delete_audio_record_local, get_audio_bytes_local,
    TTS_VOICES,
)
from utils.customer_verification import (
    mask_email, verify_customer_info, check_if_login_related_inquiry,
    check_if_customer_provided_verification_info,
)
from utils.customer_analysis import (
    detect_text_language, analyze_customer_profile, find_similar_cases,
    generate_guideline_from_past_cases, generate_initial_advice,
    _generate_initial_advice,  # 별칭 (하위 호환성)
)
from rag_handler import (
    load_documents, split_documents, get_embedding_model,
    get_embedding_function, build_rag_index, load_rag_index,
    rag_answer, load_or_train_lstm
)
from video_handler import (
    analyze_text_for_video_selection, get_video_path_by_avatar,
    load_video_mapping_database, save_video_mapping_database,
    add_video_mapping_feedback, get_recommended_video_from_database,
    render_synchronized_video, generate_virtual_human_video,
    get_virtual_human_config
)
from faq_manager import (
    load_faq_database, save_faq_database, get_company_info_faq,
    visualize_company_data, load_product_image_cache, save_product_image_cache,
    generate_product_image_prompt, generate_product_image_with_ai,
    get_product_image_url, search_faq, get_common_product_faqs,
    generate_company_info_with_llm
)
from llm_client import get_api_key, get_llm_client, run_llm, init_openai_audio_client
from lang_pack import LANG, DEFAULT_LANG as LANG_DEFAULT
from utils import _load_json, _save_json
from config import (
    BASE_DIR, DATA_DIR, AUDIO_DIR, RAG_INDEX_DIR, VIDEO_DIR,
    VOICE_META_FILE, SIM_META_FILE, VIDEO_MAPPING_DB_FILE,
    FAQ_DB_FILE, PRODUCT_IMAGE_CACHE_FILE, PRODUCT_IMAGE_DIR,
    SUPPORTED_APIS, DEFAULT_LANG
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from streamlit_mic_recorder import mic_recorder
from anthropic import Anthropic
from openai import OpenAI
import requests
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
import random
import hashlib
import tempfile
import base64
import uuid
import time
import re
import json
import io
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 표준 라이브러리

# 서드파티 라이브러리

# LangChain 관련
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("❌ 'langchain-text-splitters' 패키지가 설치되지 않았습니다.")
try:
    try:
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        try:
            from langchain_classic.memory import ConversationBufferMemory
        except ImportError:
            from langchain_core.memory import ConversationBufferMemory
except ImportError:
    raise ImportError("❌ 'langchain' 패키지가 설치되지 않았습니다.")
try:
    try:
        from langchain.chains import ConversationChain
    except ImportError:
        try:
            from langchain_classic.chains import ConversationChain
        except ImportError:
            ConversationChain = None
except ImportError:
    ConversationChain = None


# Word, PPTX, PDF 생성 라이브러리
try:
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    IS_DOCX_AVAILABLE = True
except ImportError:
    IS_DOCX_AVAILABLE = False

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    IS_PPTX_AVAILABLE = True
except ImportError:
    IS_PPTX_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.colors import black
    IS_REPORTLAB_AVAILABLE = True
except ImportError:
    IS_REPORTLAB_AVAILABLE = False

# Plotly 시각화
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False

# 임베딩 모델
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    IS_GEMINI_EMBEDDING_AVAILABLE = True
except ImportError:
    IS_GEMINI_EMBEDDING_AVAILABLE = False

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    IS_NVIDIA_EMBEDDING_AVAILABLE = True
except ImportError:
    IS_NVIDIA_EMBEDDING_AVAILABLE = False

# ========================================
# 모듈 Import
# ========================================

# 클래스 정의 import
try:
    from call_handler import CallHandler
    CALL_HANDLER_AVAILABLE = True
except ImportError:
    try:
        from _pages._classes import CallHandler
        CALL_HANDLER_AVAILABLE = True
    except ImportError:
        CALL_HANDLER_AVAILABLE = False
        class CallHandler:
            pass

try:
    from audio_handler import AudioHandler as AppAudioHandler
    APP_AUDIO_HANDLER_AVAILABLE = True
except ImportError:
    try:
        from _pages._classes import AppAudioHandler
        APP_AUDIO_HANDLER_AVAILABLE = True
    except ImportError:
        APP_AUDIO_HANDLER_AVAILABLE = False
        class AppAudioHandler:
            pass

try:
    from customer_data import CustomerDataManager
    CUSTOMER_DATA_AVAILABLE = True
except ImportError:
    try:
        from _pages._classes import CustomerDataManager
        CUSTOMER_DATA_AVAILABLE = True
    except ImportError:
        CUSTOMER_DATA_AVAILABLE = False
        class CustomerDataManager:
            pass


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
# 0. 기본 경로/로컬 DB 설정
# ========================================


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRODUCT_IMAGE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RAG_INDEX_DIR, exist_ok=True)

# 비디오 디렉토리도 초기화 시 생성
os.makedirs(VIDEO_DIR, exist_ok=True)


# ----------------------------------------
# JSON Helper는 utils.py로 이동됨
# ----------------------------------------


# ========================================
# 1. 다국어 설정 (전화 발신 관련 텍스트 추가)
# ========================================


# ========================================
# 1-1. Session State 초기화 (전화 발신 관련 상태 추가)
# ========================================
# ⭐ 사이드바 버튼은 사이드바 블록 안으로 이동해야 함
# 여기서는 세션 상태만 초기화

if "language" not in st.session_state:
    st.session_state.language = DEFAULT_LANG
# is_llm_ready는 _session_init.py에서 초기화됨 (여기서는 기본값만 설정)
if "is_llm_ready" not in st.session_state:
    st.session_state.is_llm_ready = False
if "llm_init_error_msg" not in st.session_state:
    st.session_state.llm_init_error_msg = ""
if "uploaded_files_state" not in st.session_state:
    st.session_state.uploaded_files_state = None
if "is_rag_ready" not in st.session_state:
    st.session_state.is_rag_ready = False
if "rag_vectorstore" not in st.session_state:
    st.session_state.rag_vectorstore = None
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if "agent_input" not in st.session_state:
    st.session_state.agent_input = ""
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "simulator_messages" not in st.session_state:
    st.session_state.simulator_messages = []
if "simulator_memory" not in st.session_state:
    st.session_state.simulator_memory = ConversationBufferMemory(
        memory_key="chat_history")
if "simulator_chain" not in st.session_state:
    st.session_state.simulator_chain = None
if "initial_advice_provided" not in st.session_state:
    st.session_state.initial_advice_provided = False
if "is_chat_ended" not in st.session_state:
    st.session_state.is_chat_ended = False
if "show_delete_confirm" not in st.session_state:
    st.session_state.show_delete_confirm = False
if "customer_query_text_area" not in st.session_state:
    st.session_state.customer_query_text_area = ""
if "agent_response_area_text" not in st.session_state:
    st.session_state.agent_response_area_text = ""
if "reset_agent_response_area" not in st.session_state:
    st.session_state.reset_agent_response_area = False
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "sim_audio_bytes" not in st.session_state:
    st.session_state.sim_audio_bytes = None
if "chat_state" not in st.session_state:
    st.session_state.chat_state = "idle"
    # idle → initial_customer → supervisor_advice → agent_turn → customer_turn
    # → closing
if "openai_client" not in st.session_state:
    st.session_state.openai_client = None
if "openai_init_msg" not in st.session_state:
    st.session_state.openai_init_msg = ""
if "sim_stage" not in st.session_state:
    st.session_state.sim_stage = "WAIT_FIRST_QUERY"
    # WAIT_FIRST_QUERY (초기 문의 입력)
    # AGENT_TURN (에이전트 응답 입력)
    # CUSTOMER_TURN (고객 반응 생성 요청)
    # WAIT_CLOSING_CONFIRMATION_FROM_AGENT (고객이 감사, 에이전트가 종료 확인 메시지 보내기 대기)
    # WAIT_CUSTOMER_CLOSING_RESPONSE (종료 확인 메시지 보냄, 고객의 마지막 응답 대기)
    # FINAL_CLOSING_ACTION (최종 종료 버튼 대기)
    # CLOSING (채팅 종료)
    # ⭐ 추가: OUTBOUND_CALL_IN_PROGRESS (전화 발신 진행 중)
if "start_time" not in st.session_state:  # AHT 타이머 시작 시간
    st.session_state.start_time = None
if "is_solution_provided" not in st.session_state:  # 솔루션 제공 여부 플래그
    st.session_state.is_solution_provided = False
if "transfer_summary_text" not in st.session_state:  # 이관 시 번역된 요약
    st.session_state.transfer_summary_text = ""
if "translation_success" not in st.session_state:  # 번역 성공 여부 추적
    st.session_state.translation_success = True
if "language_transfer_requested" not in st.session_state:  # 고객의 언어 이관 요청 여부
    st.session_state.language_transfer_requested = False
if "customer_attachment_file" not in st.session_state:  # 고객 첨부 파일 정보
    st.session_state.customer_attachment_file = None
if "customer_data_manager" not in st.session_state:  # 고객 데이터 관리자
    st.session_state.customer_data_manager = CustomerDataManager()
if "customer_data" not in st.session_state:  # 현재 고객 데이터
    st.session_state.customer_data = None
if "show_agent_response_ui" not in st.session_state:  # 에이전트 응답 UI 표시 여부
    st.session_state.show_agent_response_ui = False
if "show_customer_data_ui" not in st.session_state:  # 고객 데이터 가져오기 UI 표시 여부
    st.session_state.show_customer_data_ui = False
if "language_at_transfer" not in st.session_state:  # 현재 언어와 비교를 위한 변수
    st.session_state.language_at_transfer = st.session_state.language
if "language_at_transfer_start" not in st.session_state:  # 번역 재시도를 위한 원본 언어
    st.session_state.language_at_transfer_start = st.session_state.language
if "transfer_retry_count" not in st.session_state:
    st.session_state.transfer_retry_count = 0
if "customer_type_sim_select" not in st.session_state:  # FIX: Attribute Error 해결
    # LANG이 정의되기 전이므로 기본값을 직접 설정
    default_customer_type = "까다로운 고객"  # 한국어 기본값
    if st.session_state.language == "en":
        default_customer_type = "Difficult Customer"
    elif st.session_state.language == "ja":
        default_customer_type = "難しい顧客"
    st.session_state.customer_type_sim_select = default_customer_type
if "customer_email" not in st.session_state:  # FIX: customer_email 초기화
    st.session_state.customer_email = ""
if "customer_phone" not in st.session_state:  # FIX: customer_phone 초기화
    st.session_state.customer_phone = ""
if "agent_response_input_box_widget" not in st.session_state:  # FIX: customer_phone 초기화
    st.session_state.agent_response_input_box_widget = ""
# 고객 검증 관련 상태
if "is_customer_verified" not in st.session_state:
    st.session_state.is_customer_verified = False
if "verification_info" not in st.session_state:  # 시스템 내부 검증 정보 (confidential)
    st.session_state.verification_info = {
        "receipt_number": "",
        "card_last4": "",
        "customer_name": "",
        "customer_email": "",
        "customer_phone": "",
        "verification_attempts": 0
    }
# WAIT_VERIFICATION, VERIFICATION_IN_PROGRESS, VERIFIED, VERIFICATION_FAILED
if "verification_stage" not in st.session_state:
    st.session_state.verification_stage = "WAIT_VERIFICATION"
if "sim_instance_id" not in st.session_state:  # FIX: DuplicateWidgetID 방지용 인스턴스 ID 초기화
    st.session_state.sim_instance_id = str(uuid.uuid4())
if "sim_attachment_context_for_llm" not in st.session_state:
    st.session_state.sim_attachment_context_for_llm = ""
if "realtime_hint_text" not in st.session_state:
    st.session_state.realtime_hint_text = ""
# ⭐ 추가: 전화 발신 관련 상태
if "sim_call_outbound_summary" not in st.session_state:
    st.session_state.sim_call_outbound_summary = ""
if "sim_call_outbound_target" not in st.session_state:
    st.session_state.sim_call_outbound_target = None
# ----------------------------------------------------------------------
# ⭐ 전화 기능 관련 상태 추가 (app.py 스타일)
if "call_handler" not in st.session_state:
    st.session_state.call_handler = CallHandler()
if "audio_handler" not in st.session_state:
    st.session_state.audio_handler = AppAudioHandler()
if "call_active" not in st.session_state:
    st.session_state.call_active = False
# 세션 상태 초기화
if SESSION_INIT_AVAILABLE:
    init_session_state()

# 사이드바 렌더링
if SIDEBAR_AVAILABLE:
    render_sidebar()

# 메인 타이틀
# ⭐ L 변수가 정의되어 있는지 확인 (사이드바에서 이미 정의됨)
if "language" not in st.session_state:
    st.session_state.language = "ko"
# 언어 키 안전하게 가져오기
current_lang = st.session_state.get("language", "ko")
if current_lang not in ["ko", "en", "ja"]:
    current_lang = "ko"
L = LANG.get(current_lang, LANG["ko"])

# ⭐ 타이틀과 설명을 한 줄로 간결하게 표시
feature_selection = st.session_state.get(
    "feature_selection", L["sim_tab_chat_email"])

if feature_selection == L["sim_tab_chat_email"]:
    st.markdown(f"### 📧 {L['sim_tab_chat_email']}")
    st.caption(L['sim_tab_chat_email_desc'])
    if CHAT_SIMULATOR_AVAILABLE:
        render_chat_simulator()
    else:
        st.error("채팅 시뮬레이터 모듈을 찾을 수 없습니다.")

elif feature_selection == L["sim_tab_phone"]:
    st.markdown(f"### 📞 {L['sim_tab_phone']}")
    st.caption(L['sim_tab_phone_desc'])
    if PHONE_SIMULATOR_AVAILABLE:
        render_phone_simulator()
    else:
        st.error("전화 시뮬레이터 탭 모듈을 찾을 수 없습니다.")

elif feature_selection == L["rag_tab"]:
    st.markdown(f"### 📚 {L['rag_tab']}")
    st.caption(L['rag_tab_desc'])
    if RAG_AVAILABLE:
        render_rag()
    else:
        st.error("RAG 탭 모듈을 찾을 수 없습니다.")

elif feature_selection == L["content_tab"]:
    st.markdown(f"### 📝 {L['content_tab']}")
    st.caption(L['content_tab_desc'])
    if CONTENT_AVAILABLE:
        render_content()
    else:
        st.error("콘텐츠 탭 모듈을 찾을 수 없습니다.")

elif feature_selection == L["lstm_tab"]:
    st.markdown(f"### 📊 {L['lstm_tab']}")
    st.caption(L['lstm_tab_desc'])

elif feature_selection == L["voice_rec_header"]:
    st.markdown(f"### 🎤 {L['voice_rec_header']}")
    st.caption(L['voice_rec_header_desc'])

elif feature_selection == L["company_info_tab"]:
    st.markdown(f"#### 📋 {L['company_info_tab']}")
    st.caption(L['company_info_tab_desc'])
    if COMPANY_INFO_AVAILABLE:
        render_company_info()
    else:
        st.error("회사 정보 탭 모듈을 찾을 수 없습니다.")
