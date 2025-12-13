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
# streamlit_app.py (모듈화된 버전)
# ========================================

# ⭐ OpenMP 라이브러리 충돌 해결
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 표준 라이브러리
import io
import json
import time
import uuid
import base64
import tempfile
import hashlib
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Tuple

# 서드파티 라이브러리
import google.generativeai as genai
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import requests
from openai import OpenAI
from anthropic import Anthropic
from streamlit_mic_recorder import mic_recorder

# LangChain 관련
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
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

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

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
from config import (
    BASE_DIR, DATA_DIR, AUDIO_DIR, RAG_INDEX_DIR, VIDEO_DIR,
    VOICE_META_FILE, SIM_META_FILE, VIDEO_MAPPING_DB_FILE,
    FAQ_DB_FILE, PRODUCT_IMAGE_CACHE_FILE, PRODUCT_IMAGE_DIR,
    SUPPORTED_APIS, DEFAULT_LANG
)
from utils import _load_json, _save_json
from lang_pack import LANG, DEFAULT_LANG as LANG_DEFAULT
from llm_client import get_api_key, get_llm_client, run_llm, init_openai_audio_client
from faq_manager import (
    load_faq_database, save_faq_database, get_company_info_faq,
    visualize_company_data, load_product_image_cache, save_product_image_cache,
    generate_product_image_prompt, generate_product_image_with_ai,
    get_product_image_url, search_faq, get_common_product_faqs,
    generate_company_info_with_llm
)
from audio_handler import (
    transcribe_bytes_with_whisper, transcribe_audio, synthesize_tts,
    render_tts_button, load_voice_records, save_voice_records,
    save_audio_record_local, delete_audio_record_local, get_audio_bytes_local,
    TTS_VOICES
)
from video_handler import (
    analyze_text_for_video_selection, get_video_path_by_avatar,
    load_video_mapping_database, save_video_mapping_database,
    add_video_mapping_feedback, get_recommended_video_from_database,
    render_synchronized_video, generate_virtual_human_video,
    get_virtual_human_config
)
from rag_handler import (
    load_documents, split_documents, get_embedding_model,
    get_embedding_function, build_rag_index, load_rag_index,
    rag_answer, load_or_train_lstm
)
from simulation_handler import (
    translate_text_with_llm, generate_realtime_hint,
    generate_agent_response_draft, generate_outbound_call_summary,
    load_simulation_histories_local, generate_chat_summary,
    save_simulation_history_local, export_history_to_word,
    export_history_to_pptx, export_history_to_pdf,
    get_chat_history_for_prompt, generate_customer_reaction,
    summarize_history_with_ai, generate_customer_reaction_for_call,
    generate_customer_reaction_for_first_greeting, summarize_history_for_call,
    generate_customer_closing_response, generate_agent_first_greeting,
    detect_text_language, analyze_customer_profile, find_similar_cases,
    generate_guideline_from_past_cases, _generate_initial_advice
)
from visualization import (
    visualize_customer_profile_scores, visualize_similarity_cases,
    visualize_case_trends, visualize_customer_characteristics
)


# ========================================
# Streamlit 페이지 설정
# ========================================
st.set_page_config(
    page_title="AI Study Coach & Customer Service Simulator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
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
    st.session_state.simulator_memory = ConversationBufferMemory(memory_key="chat_history")
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
    # idle → initial_customer → supervisor_advice → agent_turn → customer_turn → closing
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
# ⭐ 전화 기능 관련 상태 추가
if "call_sim_stage" not in st.session_state:
    st.session_state.call_sim_stage = "WAITING_CALL"  # WAITING_CALL, RINGING, IN_CALL, CALL_ENDED
if "call_sim_mode" not in st.session_state:
    st.session_state.call_sim_mode = "INBOUND"  # INBOUND or OUTBOUND
if "incoming_phone_number" not in st.session_state:
    st.session_state.incoming_phone_number = "+82 10-1234-5678"
if "is_on_hold" not in st.session_state:
    st.session_state.is_on_hold = False
if "hold_start_time" not in st.session_state:
    st.session_state.hold_start_time = None
if "total_hold_duration" not in st.session_state:
    st.session_state.total_hold_duration = timedelta(0)
if "current_customer_audio_text" not in st.session_state:
    st.session_state.current_customer_audio_text = ""
if "current_agent_audio_text" not in st.session_state:
    st.session_state.current_agent_audio_text = ""
if "agent_response_input_box_widget_call" not in st.session_state:  # 전화 탭 전용 입력창
    st.session_state.agent_response_input_box_widget_call = ""
if "call_initial_query" not in st.session_state:  # 전화 탭 전용 초기 문의
    st.session_state.call_initial_query = ""
if "call_website_url" not in st.session_state:  # 전화 탭 전용 홈페이지 주소
    st.session_state.call_website_url = ""
# ⭐ 추가: 통화 요약 및 초기 고객 음성 저장소
if "call_summary_text" not in st.session_state:
    st.session_state.call_summary_text = ""
if "customer_initial_audio_bytes" not in st.session_state:  # 고객의 첫 음성 (TTS 결과) 저장
    st.session_state.customer_initial_audio_bytes = None
if "supervisor_policy_context" not in st.session_state:
    # Supervisor가 업로드한 예외 정책 텍스트를 저장합니다.
    st.session_state.supervisor_policy_context = ""
if "agent_policy_attachment_content" not in st.session_state:
    # 에이전트가 업로드한 정책 파일 객체(또는 내용)를 저장합니다.
    st.session_state.agent_policy_attachment_content = ""
if "customer_attachment_b64" not in st.session_state:
    st.session_state.customer_attachment_b64 = ""
if "customer_history_summary" not in st.session_state:
    st.session_state.customer_history_summary = ""
if "customer_avatar" not in st.session_state:
    st.session_state.customer_avatar = {
        "gender": "male",  # 기본값
        "state": "NEUTRAL",  # 기본 아바타 상태
    }
# ⭐ 추가: 비디오 동기화 관련 세션 상태
if "current_customer_video" not in st.session_state:
    st.session_state.current_customer_video = None  # 현재 재생 중인 고객 비디오 경로
if "current_customer_video_bytes" not in st.session_state:
    st.session_state.current_customer_video_bytes = None  # 현재 재생 중인 고객 비디오 바이트
if "is_video_sync_enabled" not in st.session_state:
    st.session_state.is_video_sync_enabled = True  # 비디오 동기화 활성화 여부
if "video_male_neutral" not in st.session_state:
    st.session_state.video_male_neutral = None  # 남자 중립 비디오 경로
if "video_male_happy" not in st.session_state:
    st.session_state.video_male_happy = None
if "video_male_angry" not in st.session_state:
    st.session_state.video_male_angry = None
if "video_male_asking" not in st.session_state:
    st.session_state.video_male_asking = None
if "video_male_sad" not in st.session_state:
    st.session_state.video_male_sad = None
if "video_female_neutral" not in st.session_state:
    st.session_state.video_female_neutral = None  # 여자 중립 비디오 경로
if "video_female_happy" not in st.session_state:
    st.session_state.video_female_happy = None
if "video_female_angry" not in st.session_state:
    st.session_state.video_female_angry = None
if "video_female_asking" not in st.session_state:
    st.session_state.video_female_asking = None
if "video_female_sad" not in st.session_state:
    st.session_state.video_female_sad = None
# ⭐ 추가: 전사할 오디오 바이트 임시 저장소
if "bytes_to_process" not in st.session_state:
    st.session_state.bytes_to_process = None

# 언어 키 안전하게 가져오기
current_lang = st.session_state.get("language", "ko")
if current_lang not in ["ko", "en", "ja"]:
    current_lang = "ko"
L = LANG.get(current_lang, LANG["ko"])

# ⭐ 2-A. Gemini 키 초기화 (잘못된 키 잔존 방지)
if "user_gemini_key" in st.session_state and st.session_state["user_gemini_key"].startswith("AIza"):
    pass

# ========================================
# 0. 세션 상태 초기화
# ========================================

# 세션 초기화 (SUPPORTED_APIS는 config에서 import됨)
for api, cfg in SUPPORTED_APIS.items():
    if cfg["session_key"] not in st.session_state:
        st.session_state[cfg["session_key"]] = ""

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "openai_gpt4"


# ========================================
# 1. Sidebar UI: API Key 입력 제거
# ========================================
# API Key 입력 UI는 제거하고, 환경변수와 Streamlit Secrets만 사용하도록 함.


# ========================================
# 2. LLM 클라이언트 라우팅 & 실행
# ========================================
# ========================================
# 2-A. Whisper / TTS 용 OpenAI Client 별도로 초기화
# ========================================

if "openai_client" not in st.session_state or st.session_state.openai_client is None:
    try:
        st.session_state.openai_client = init_openai_audio_client()
    except Exception as e:
        st.session_state.openai_client = None
        print(f"OpenAI 클라이언트 초기화 중 오류 (무시됨): {e}")

# LLM 준비 상태 캐싱 (API 키 변경 시에만 재확인)
# ⭐ 수정: 초기화 시 블로킹 방지를 위해 try-except 추가
if "is_llm_ready" not in st.session_state or "llm_ready_checked" not in st.session_state:
    try:
        probe_client, _ = get_llm_client()
        st.session_state.is_llm_ready = probe_client is not None
    except Exception as e:
        # 초기화 실패 시에도 앱이 계속 실행되도록 False로 설정
        st.session_state.is_llm_ready = False
        print(f"LLM 초기화 중 오류 (무시됨): {e}")
    st.session_state.llm_ready_checked = True

# API 키 변경 감지를 위한 해시 체크
current_api_keys_hash = hashlib.md5(
    f"{get_api_key('openai')}{get_api_key('gemini')}{get_api_key('claude')}{get_api_key('groq')}".encode()
).hexdigest()

if "api_keys_hash" not in st.session_state:
    st.session_state.api_keys_hash = current_api_keys_hash
elif st.session_state.api_keys_hash != current_api_keys_hash:
    # API 키가 변경된 경우만 재확인
    # ⭐ 수정: 초기화 시 블로킹 방지를 위해 try-except 추가
    try:
        probe_client, _ = get_llm_client()
        st.session_state.is_llm_ready = probe_client is not None
    except Exception as e:
        st.session_state.is_llm_ready = False
        print(f"LLM 재초기화 중 오류 (무시됨): {e}")
    st.session_state.api_keys_hash = current_api_keys_hash
    # OpenAI 클라이언트도 재초기화
    try:
        st.session_state.openai_client = init_openai_audio_client()
    except Exception as e:
        st.session_state.openai_client = None
        print(f"OpenAI 클라이언트 재초기화 중 오류 (무시됨): {e}")

if st.session_state.openai_client:
    # 키를 찾았고 클라이언트 객체는 생성되었으나, 실제 인증은 API 호출 시 이루어짐 (401 오류는 여기서 발생)
    st.session_state.openai_init_msg = "✅ OpenAI TTS/Whisper 클라이언트 준비 완료 (Key 확인됨)"
else:
    # 키를 찾지 못한 경우
    st.session_state.openai_init_msg = L["openai_missing"]

if not st.session_state.is_llm_ready:
    st.session_state.llm_init_error_msg = L["simulation_no_key_warning"]
else:
    st.session_state.llm_init_error_msg = ""


# ----------------------------------------
# LLM 번역 함수는 simulation_handler.py로 이동됨
# ----------------------------------------

# ========================================
# 3. Whisper / TTS Helper는 audio_handler.py로 이동됨
# ========================================

# ========================================
# 비디오 동기화 관련 함수는 video_handler.py로 이동됨
# 시뮬레이션 관련 함수는 simulation_handler.py로 이동됨
# ========================================

# ========================================
# 8. LLM (ChatOpenAI) for Simulator / Content
# (RAG와 동일하게 run_llm으로 통합)
# ========================================

# ConversationChain 대신 run_llm을 사용하여 메모리 기능을 수동으로 구현
# st.session_state.simulator_memory는 유지하여 대화 기록을 관리합니다.

def visualize_customer_profile_scores(customer_profile: Dict[str, Any], current_lang_key: str):
    """고객 프로필 점수를 시각화 (감정 점수, 긴급도)"""
    if not IS_PLOTLY_AVAILABLE:
        return None

    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    sentiment_score = customer_profile.get("sentiment_score", 50)
    urgency_map = {"low": 25, "medium": 50, "high": 75}
    urgency_level = customer_profile.get("urgency_level", "medium")
    urgency_score = urgency_map.get(urgency_level.lower(), 50)

    # 게이지 차트 생성
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=(
            L.get("sentiment_score_label", "고객 감정 점수"),
            L.get("urgency_score_label", "긴급도 점수")
        )
    )

    # 감정 점수 게이지
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': L.get("sentiment_score_label", "감정 점수")},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )

    # 긴급도 점수 게이지
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=urgency_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': L.get("urgency_score_label", "긴급도")},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ],
            }
        ),
        row=1, col=2
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def visualize_similarity_cases(similar_cases: List[Dict[str, Any]], current_lang_key: str):
    """유사 케이스 추천을 시각화"""
    if not IS_PLOTLY_AVAILABLE or not similar_cases:
        return None

    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    case_labels = []
    similarity_scores = []
    sentiment_scores = []
    satisfaction_scores = []

    for idx, similar_case in enumerate(similar_cases, 1):
        summary = similar_case["summary"]
        similarity = similar_case["similarity_score"]
        case_labels.append(f"Case {idx}")
        similarity_scores.append(similarity)
        sentiment_scores.append(summary.get("customer_sentiment_score", 50))
        satisfaction_scores.append(summary.get("customer_satisfaction_score", 50))

    # 유사도 차트
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            L.get("similarity_chart_title", "유사 케이스 유사도"),
            L.get("scores_comparison_title",
                  "감정 및 만족도 점수 비교")
        ),
        vertical_spacing=0.15
    )

    # 유사도 바 차트
    fig.add_trace(
        go.Bar(
            x=case_labels,
            y=similarity_scores,
            name=L.get("similarity_score_label", "유사도"),
            marker_color='lightblue',
            text=[f"{s:.1f}%" for s in similarity_scores],
            textposition='outside'
        ),
        row=1, col=1
    )

    # 감정 및 만족도 점수 비교
    fig.add_trace(
        go.Bar(
            x=case_labels,
            y=sentiment_scores,
            name=L.get("sentiment_score_label", "감정 점수"),
            marker_color='lightcoral'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=case_labels,
            y=satisfaction_scores,
            name=L.get("satisfaction_score_label", "만족도"),
            marker_color='lightgreen'
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20),
        barmode='group'
    )
    fig.update_yaxes(title_text="점수", row=2, col=1)
    fig.update_yaxes(title_text="유사도 (%)", row=1, col=1)

    return fig


def visualize_case_trends(histories: List[Dict[str, Any]], current_lang_key: str):
    """과거 성공 사례 트렌드를 시각화"""
    if not IS_PLOTLY_AVAILABLE or not histories:
        return None

    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    # 요약 데이터가 있는 케이스만 필터링
    cases_with_summary = [
        h for h in histories
        if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
    ]

    if not cases_with_summary:
        return None

    # 날짜별로 정렬
    cases_with_summary.sort(key=lambda x: x.get("timestamp", ""))

    dates = []
    sentiment_scores = []
    satisfaction_scores = []

    for case in cases_with_summary:
        summary = case.get("summary", {})
        timestamp = case.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(timestamp)
            dates.append(dt)
            sentiment_scores.append(summary.get("customer_sentiment_score", 50))
            satisfaction_scores.append(summary.get("customer_satisfaction_score", 50))
        except Exception:
            continue

    if not dates:
        return None

    # 트렌드 라인 차트
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=sentiment_scores,
        mode='lines+markers',
        name=L.get("sentiment_trend_label", "감정 점수 추이"),
        line=dict(color='lightcoral', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=satisfaction_scores,
        mode='lines+markers',
        name=L.get("satisfaction_trend_label", "만족도 점수 추이"),
        line=dict(color='lightgreen', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=L.get("case_trends_title", "과거 케이스 점수 추이"),
        xaxis_title=L.get("date_label", "날짜"),
        yaxis_title=L.get("score_label", "점수 (0-100)"),
        height=400,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def visualize_customer_characteristics(summary: Dict[str, Any], current_lang_key: str):
    """고객 특성을 시각화 (언어, 문화권, 지역 등)"""
    if not IS_PLOTLY_AVAILABLE or not summary:
        return None

    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    characteristics = summary.get("customer_characteristics", {})
    privacy_info = summary.get("privacy_info", {})

    # 특성 데이터 준비
    labels = []
    values = []

    # 언어 정보
    language = characteristics.get("language", "unknown")
    if language != "unknown":
        labels.append(L.get("language_label", "언어"))
        lang_map = {"ko": "한국어", "en": "English", "ja": "日本語"}
        values.append(lang_map.get(language, language))

    # 개인정보 제공 여부
    if privacy_info.get("has_email"):
        labels.append(L.get("email_provided_label", "이메일 제공"))
        values.append("Yes")
    if privacy_info.get("has_phone"):
        labels.append(L.get("phone_provided_label", "전화번호 제공"))
        values.append("Yes")

    # 지역 정보
    region = privacy_info.get("region_hint", characteristics.get("region", "unknown"))
    if region != "unknown":
        labels.append(L.get("region_label", "지역"))
        values.append(region)

    if not labels:
        return None

    # 파이 차트
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=[1] * len(labels),
        hole=0.4,
        marker_colors=px.colors.qualitative.Set3[:len(labels)]
    )])

    fig.update_layout(
        title=L.get("customer_characteristics_title",
                    "고객 특성 분포"),
        height=300,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


[Case {idx}] (Similarity: {similarity:.1f}%)
- Inquiry: {summary.get('main_inquiry', 'N/A')}
- Customer Sentiment: {summary.get('customer_sentiment_score', 50)}/100
- Customer Satisfaction: {summary.get('customer_satisfaction_score', 50)}/100
- Key Responses: {', '.join(summary.get('key_responses', [])[:3])}
- Summary: {summary.get('summary', 'N/A')[:200]}
"""

    guideline_prompt = f"""
You are an AI Customer Support Supervisor analyzing past successful cases to provide guidance.

Based on the following similar past cases and their successful resolution strategies, provide actionable guidelines for handling the current customer inquiry.

Current Customer Inquiry:
{customer_query}

Current Customer Profile:
- Gender: {customer_profile.get('gender', 'unknown')}
- Sentiment Score: {customer_profile.get('sentiment_score', 50)}/100
- Communication Style: {customer_profile.get('communication_style', 'unknown')}
- Urgency: {customer_profile.get('urgency_level', 'medium')}
- Predicted Type: {customer_profile.get('predicted_customer_type', 'normal')}

Similar Past Cases (Successful Resolutions):
{past_cases_text}

Provide a concise guideline in {lang_name} that:
1. Identifies what worked well in similar past cases
2. Suggests specific approaches based on successful patterns
3. Warns about potential pitfalls based on past experiences
4. Recommends response strategies that led to high customer satisfaction

Guideline (in {lang_name}):
def _generate_initial_advice(customer_query, customer_type_display, customer_email, customer_phone, current_lang_key,
                             customer_attachment_file):
    """Supervisor 가이드라인과 초안을 생성하는 함수 (저장된 데이터 활용)"""
    # 입력 텍스트의 언어를 자동 감지 (오류 발생 시 안전하게 처리)
    try:
        detected_lang = detect_text_language(customer_query)
    except Exception as e:
        print(f"Language detection failed in _generate_initial_advice: {e}")
        detected_lang = current_lang_key if current_lang_key else "ko"
    
    # 감지된 언어를 우선 사용하되, current_lang_key가 명시적으로 제공되면 그것을 사용
    lang_key_to_use = detected_lang if detected_lang else current_lang_key
    # lang_key_to_use가 유효한지 확인
    if lang_key_to_use not in ["ko", "en", "ja"]:
        lang_key_to_use = current_lang_key if current_lang_key else "ko"
    
    # 언어 키 검증
    if lang_key_to_use not in ["ko", "en", "ja"]:
        lang_key_to_use = st.session_state.get("language", "ko")
        if lang_key_to_use not in ["ko", "en", "ja"]:
            lang_key_to_use = "ko"
    L = LANG.get(lang_key_to_use, LANG["ko"])
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[lang_key_to_use]

    contact_info_block = ""
    if customer_email or customer_phone:
        contact_info_block = (
            f"\n\n[Customer contact info for reference (DO NOT use these in your reply draft!)]"
            f"\n- Email: {customer_email or 'N/A'}"
            f"\n- Phone: {customer_phone or 'N/A'}"
        )

    attachment_block = ""
    if customer_attachment_file:
        file_name = customer_attachment_file.name
        attachment_block = f"\n\n[ATTACHMENT NOTE]: {L['attachment_info_llm'].format(filename=file_name)}"

    # 고객 프로필 분석 (감지된 언어 사용)
    customer_profile = analyze_customer_profile(customer_query, lang_key_to_use)

    # 유사 케이스 찾기 (감지된 언어 사용)
    similar_cases = find_similar_cases(customer_query, customer_profile, lang_key_to_use, limit=5)

    # 과거 케이스 기반 가이드라인 생성 (감지된 언어 사용)
    past_cases_guideline = ""
    if similar_cases:
        past_cases_guideline = generate_guideline_from_past_cases(
            customer_query, customer_profile, similar_cases, lang_key_to_use
        )

    # 고객 프로필 정보
    gender_display = customer_profile.get('gender', 'unknown')
    profile_block = f"""
[Customer Profile Analysis]
- Gender: {gender_display}
- Sentiment Score: {customer_profile.get('sentiment_score', 50)}/100
- Communication Style: {customer_profile.get('communication_style', 'unknown')}
- Urgency Level: {customer_profile.get('urgency_level', 'medium')}
- Predicted Type: {customer_profile.get('predicted_customer_type', 'normal')}
- Key Concerns: {', '.join(customer_profile.get('key_concerns', []))}
- Tone: {customer_profile.get('tone_analysis', 'unknown')}
"""

    # 과거 케이스 기반 가이드라인 블록
    past_cases_block = ""
    if past_cases_guideline:
        past_cases_block = f"""
[Guidelines Based on {len(similar_cases)} Similar Past Cases]
{past_cases_guideline}
"""
    elif similar_cases:
        past_cases_block = f"""
[Note: Found {len(similar_cases)} similar past cases, but unable to generate detailed guidelines.
Consider reviewing past cases manually for patterns.]
"""

    # Output ALL text (guidelines and draft) STRICTLY in {lang_name}. <--- 강력한 언어 강제 지시
    initial_prompt = f"""
Output ALL text (guidelines and draft) STRICTLY in {lang_name}.

You are an AI Customer Support Supervisor. Your role is to analyze the following customer inquiry
from a **{st.session_state.customer_type_sim_select}** and provide:

1) A detailed **response guideline for the human agent** (step-by-step).
2) A **ready-to-send draft reply** in {lang_name}.

[FORMAT]
- Use the exact markdown headers:
  - "### {L['simulation_advice_header']}"
  - "### {L['simulation_draft_header']}"

[CRITICAL GUIDELINE RULES]
1. **Initial Information Collection (Req 3):** The first step in the guideline MUST be to request the necessary initial diagnostic information (e.g., device compatibility, local status/location, order number) BEFORE attempting to troubleshoot or solve the problem.
2. **Empathy for Difficult Customers (Req 5):** If the customer type is 'Difficult Customer' or 'Highly Dissatisfied Customer', the guideline MUST emphasize extreme politeness, empathy, and apologies, even if the policy (e.g., no refund) must be enforced.
3. **24-48 Hour Follow-up (Req 6):** If the issue cannot be solved immediately or requires confirmation from a local partner/supervisor, the guideline MUST state the procedure:
   - Acknowledge the issue.
   - Inform the customer they will receive a definite answer within 24 or 48 hours.
   - Request the customer's email or phone number for follow-up contact. (Use provided contact info if available)
4. **Past Cases Learning:** If past cases guidelines are provided, incorporate successful strategies from those cases into your recommendations.

Customer Inquiry:
{customer_query}
{contact_info_block}
{attachment_block}
{profile_block}
{past_cases_block}
# ========================================
# 9. 사이드바
# ========================================

with st.sidebar:
    # 언어 키 안전하게 가져오기
    if "language" not in st.session_state:
        st.session_state.language = "ko"
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # 회사 목록 초기화 (회사 정보 탭에서 사용)
    if "company_language_priority" not in st.session_state:
        st.session_state.company_language_priority = {
            "default": ["ko", "en", "ja"],
            "companies": {}
        }
    
    st.markdown("---")
    
    # 언어 선택
    if "language" not in st.session_state:
        st.session_state.language = "ko"
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    lang_priority = st.session_state.company_language_priority["default"]
    
    selected_lang_key = st.selectbox(
        L["lang_select"],
        options=lang_priority,
        index=lang_priority.index(st.session_state.language) if st.session_state.language in lang_priority else 0,
        format_func=lambda x: {"ko": "한국어", "en": "English", "ja": "日本語"}[x],
    )

    # 🔹 언어 변경 감지
    if selected_lang_key != st.session_state.language:
        st.session_state.language = selected_lang_key
        # 채팅/전화 공통 상태 초기화
        st.session_state.simulator_messages = []
        # ⭐ 안전한 메모리 초기화
        try:
            if hasattr(st.session_state, 'simulator_memory') and st.session_state.simulator_memory is not None:
                st.session_state.simulator_memory.clear()
        except Exception:
            # 메모리 초기화 실패 시 새로 생성
            try:
                st.session_state.simulator_memory = ConversationBufferMemory(memory_key="chat_history")
            except Exception:
                pass  # 초기화 실패해도 계속 진행
        st.session_state.initial_advice_provided = False
        st.session_state.is_chat_ended = False
        # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로 플래그 사용
        st.session_state.reset_agent_response_area = True
        st.session_state.customer_query_text_area = ""
        st.session_state.last_transcript = ""
        st.session_state.sim_audio_bytes = None
        st.session_state.sim_stage = "WAIT_FIRST_QUERY"
        st.session_state.customer_attachment_file = []  # 언어 변경 시 첨부 파일 초기화
        st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
        st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
        # 전화 시뮬레이터 상태 초기화
        st.session_state.call_sim_stage = "WAITING_CALL"
        st.session_state.call_sim_mode = "INBOUND"
        st.session_state.is_on_hold = False
        st.session_state.total_hold_duration = timedelta(0)
        st.session_state.hold_start_time = None
        st.session_state.current_customer_audio_text = ""
        st.session_state.current_agent_audio_text = ""
        st.session_state.agent_response_input_box_widget_call = ""
        st.session_state.call_initial_query = ""
        # 전화 발신 관련 상태 초기화
        st.session_state.sim_call_outbound_summary = ""
        st.session_state.sim_call_outbound_target = None
        # ⭐ 언어 변경 시 재실행 - 무한 루프 방지를 위해 플래그 사용
        if "language_changed" not in st.session_state or not st.session_state.language_changed:
            st.session_state.language_changed = True
        else:
            # 이미 한 번 재실행했으면 플래그 초기화
            st.session_state.language_changed = False

    # 언어 키 안전하게 가져오기
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])

    st.title(L["sidebar_title"])
    st.markdown("---")

    # ⭐ API Key 설정 섹션 추가
    st.subheader("🔑 API Key 설정")
    
    # LLM 선택
    llm_options = {
        "openai_gpt4": "OpenAI GPT-4",
        "openai_gpt35": "OpenAI GPT-3.5",
        "gemini_pro": "Google Gemini Pro",
        "gemini_flash": "Google Gemini Flash",
        "claude": "Anthropic Claude",
        "groq": "Groq",
        "nvidia": "NVIDIA NIM"
    }
    
    current_llm = st.session_state.get("selected_llm", "openai_gpt4")
    selected_llm = st.selectbox(
        "LLM 모델 선택",
        options=list(llm_options.keys()),
        format_func=lambda x: llm_options[x],
        index=list(llm_options.keys()).index(current_llm) if current_llm in llm_options else 0,
        key="sidebar_llm_select"
    )
    if selected_llm != current_llm:
        st.session_state.selected_llm = selected_llm
    
    # API Key 매핑
    api_key_map = {
        "openai_gpt4": "openai",
        "openai_gpt35": "openai",
        "gemini_pro": "gemini",
        "gemini_flash": "gemini",
        "claude": "claude",
        "groq": "groq",
        "nvidia": "nvidia"
    }
    
    api_name = api_key_map.get(selected_llm, "openai")
    api_config = SUPPORTED_APIS.get(api_name, {})
    
    if api_config:
        # 현재 API Key 확인
        current_key = get_api_key(api_name)
        if not current_key:
            # 수동 입력 필드
            session_key = api_config.get("session_key", "")
            manual_key = st.text_input(
                api_config.get("label", "API Key"),
                value=st.session_state.get(session_key, ""),
                type="password",
                placeholder=api_config.get("placeholder", "API Key를 입력하세요"),
                key=f"manual_api_key_{selected_llm}"
            )
            if manual_key and manual_key != st.session_state.get(session_key, ""):
                st.session_state[session_key] = manual_key
        else:
            st.success(f"✅ {api_config.get('label', 'API Key')} 설정됨")
    
    st.markdown("---")

    # ⭐ 기능 선택 - 기본값을 AI 챗 시뮬레이터로 설정
    if "feature_selection" not in st.session_state:
        st.session_state.feature_selection = L["sim_tab_chat_email"]

    # ⭐ 핵심 기능과 더보기 기능 분리 (회사 정보 및 FAQ 추가)
    core_features = [L["sim_tab_chat_email"], L["sim_tab_phone"], L["company_info_tab"]]
    other_features = [L["rag_tab"], L["content_tab"], L["lstm_tab"], L["voice_rec_header"]]
    
    # 모든 기능을 하나의 리스트로 통합 (하나만 선택 가능하도록)
    all_features = core_features + other_features
    
    # 현재 선택된 기능
    current_selection = st.session_state.get("feature_selection", L["sim_tab_chat_email"])
    
    # 현재 선택의 인덱스 찾기
    try:
        current_index = all_features.index(current_selection) if current_selection in all_features else 0
    except (ValueError, AttributeError):
        current_index = 0
    
    # ⭐ 하나의 통합된 선택 로직 (하나만 선택 가능) - 설명 제거
    selected_feature = st.radio(
        "기능 선택",
        all_features,
        index=current_index,
        key="unified_feature_selection",
        label_visibility="hidden"
    )
    
    # 선택된 기능 업데이트
    if selected_feature != current_selection:
        st.session_state.feature_selection = selected_feature
    
    feature_selection = st.session_state.get("feature_selection", L["sim_tab_chat_email"])

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
feature_selection = st.session_state.get("feature_selection", L["sim_tab_chat_email"])
if feature_selection == L["sim_tab_chat_email"]:
    st.markdown(f"### 📧 {L['sim_tab_chat_email']}")
    st.caption(L['sim_tab_chat_email_desc'])
elif feature_selection == L["sim_tab_phone"]:
    st.markdown(f"### 📞 {L['sim_tab_phone']}")
    st.caption(L['sim_tab_phone_desc'])
elif feature_selection == L["rag_tab"]:
    st.markdown(f"### 📚 {L['rag_tab']}")
    st.caption(L['rag_tab_desc'])
elif feature_selection == L["content_tab"]:
    st.markdown(f"### 📝 {L['content_tab']}")
    st.caption(L['content_tab_desc'])
elif feature_selection == L["lstm_tab"]:
    st.markdown(f"### 📊 {L['lstm_tab']}")
    st.caption(L['lstm_tab_desc'])
elif feature_selection == L["voice_rec_header"]:
    st.markdown(f"### 🎤 {L['voice_rec_header']}")
    st.caption(L['voice_rec_header_desc'])
elif feature_selection == L["company_info_tab"]:
    # 공백 축소: 제목과 설명을 한 줄로 간결하게 표시
    st.markdown(f"#### 📋 {L['company_info_tab']}")
    st.caption(L['company_info_tab_desc'])

# ========================================
# 10. 기능별 페이지
# ========================================

# -------------------- Company Info & FAQ Tab --------------------
if feature_selection == L["company_info_tab"]:
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # FAQ 데이터베이스 로드
    faq_data = load_faq_database()
    companies = list(faq_data.get("companies", {}).keys())
    
    # 회사명 검색 입력 (상단에 배치) - 입력란은 글로벌 기업 영문명 고려하여 원래 크기 유지
    col_search_header, col_search_input, col_search_btn = st.columns([0.5, 1.2, 0.2])
    with col_search_header:
        st.write(f"**{L['search_company']}**")
    with col_search_input:
        company_search_input = st.text_input(
            "",
            placeholder=L["company_search_placeholder"],
            key="company_search_input",
            value=st.session_state.get("searched_company", ""),
            label_visibility="collapsed"
        )
    with col_search_btn:
        search_button = st.button(f"🔍 {L['company_search_button']}", key="company_search_btn", type="primary", use_container_width=True)
    
    # 검색된 회사 정보 저장
    searched_company = st.session_state.get("searched_company", "")
    searched_company_data = st.session_state.get("searched_company_data", None)
    
    # 검색 버튼 클릭 시 LLM으로 회사 정보 생성
    if search_button and company_search_input:
        with st.spinner(f"{company_search_input} {L['generating_company_info']}"):
            generated_data = generate_company_info_with_llm(company_search_input, current_lang)
            st.session_state.searched_company = company_search_input
            st.session_state.searched_company_data = generated_data
            searched_company = company_search_input
            searched_company_data = generated_data
            
            # 생성된 데이터를 데이터베이스에 저장
            if company_search_input not in faq_data.get("companies", {}):
                faq_data.setdefault("companies", {})[company_search_input] = {
                    f"info_{current_lang}": generated_data.get("company_info", ""),
                    "info_ko": generated_data.get("company_info", ""),
                    "info_en": "",
                    "info_ja": "",
                    "popular_products": generated_data.get("popular_products", []),
                    "trending_topics": generated_data.get("trending_topics", []),
                    "faqs": generated_data.get("faqs", []),
                    "interview_questions": generated_data.get("interview_questions", []),
                    "ceo_info": generated_data.get("ceo_info", {})
                }
                save_faq_database(faq_data)
    
    # 검색된 회사가 있으면 해당 데이터 사용, 없으면 기존 회사 선택
    if searched_company and searched_company_data:
        display_company = searched_company
        display_data = searched_company_data
        # 데이터베이스에도 저장되어 있으면 업데이트
        if display_company in faq_data.get("companies", {}):
            faq_data["companies"][display_company].update({
                f"info_{current_lang}": display_data.get("company_info", ""),
                "popular_products": display_data.get("popular_products", []),
                "trending_topics": display_data.get("trending_topics", []),
                "faqs": display_data.get("faqs", []),
                "interview_questions": display_data.get("interview_questions", []),
                "ceo_info": display_data.get("ceo_info", {})
            })
            save_faq_database(faq_data)
    elif companies:
        display_company = st.selectbox(
            L["select_company"],
            options=companies,
            key="company_select_display"
        )
        company_db_data = faq_data["companies"][display_company]
        display_data = {
            "company_info": company_db_data.get(f"info_{current_lang}", company_db_data.get("info_ko", "")),
            "popular_products": company_db_data.get("popular_products", []),
            "trending_topics": company_db_data.get("trending_topics", []),
            "faqs": company_db_data.get("faqs", []),
            "interview_questions": company_db_data.get("interview_questions", []),
            "ceo_info": company_db_data.get("ceo_info", {})
        }
    else:
        display_company = None
        display_data = None
    
    # 탭 생성 (FAQ 검색 탭 제거, FAQ 탭에 통합) - 공백 축소
    tab1, tab2, tab3 = st.tabs([
        L["company_info"], 
        L["company_faq"], 
        L["button_add_company"]
    ])
    
    # 탭 1: 회사 소개 및 시각화
    with tab1:
        if display_company and display_data:
            # 제목을 더 간결하게 표시
            st.markdown(f"#### {display_company} - {L['company_info']}")
            
            # 회사 소개 표시
            if display_data.get("company_info"):
                st.markdown(display_data["company_info"])
            
            # 시각화 차트 표시
            if display_data.get("popular_products") or display_data.get("trending_topics"):
                charts = visualize_company_data(
                    {
                        "popular_products": display_data.get("popular_products", []),
                        "trending_topics": display_data.get("trending_topics", [])
                    },
                    current_lang
                )
                
                if charts:
                    # 막대 그래프 표시 - 공백 축소
                    st.markdown(f"#### 📊 {L['visualization_chart']}")
                    col1_bar, col2_bar = st.columns(2)
                    
                    if "products_bar" in charts:
                        with col1_bar:
                            st.plotly_chart(charts["products_bar"], use_container_width=True)
                    
                    if "topics_bar" in charts:
                        with col2_bar:
                            st.plotly_chart(charts["topics_bar"], use_container_width=True)
                    
                    # 선형 그래프 표시
                    col1_line, col2_line = st.columns(2)
                    
                    if "products_line" in charts:
                        with col1_line:
                            st.plotly_chart(charts["products_line"], use_container_width=True)
                    
                    if "topics_line" in charts:
                        with col2_line:
                            st.plotly_chart(charts["topics_line"], use_container_width=True)
            
            # 인기 상품 목록 (이미지 포함) - 공백 축소
            if display_data.get("popular_products"):
                st.markdown(f"#### {L['popular_products']}")
                # 상품을 그리드 형태로 표시
                product_cols = st.columns(min(3, len(display_data["popular_products"])))
                for idx, product in enumerate(display_data["popular_products"]):
                    product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                    product_score = product.get("score", 0)
                    product_image_url = product.get("image_url", "")
                    
                    with product_cols[idx % len(product_cols)]:
                        # 이미지 표시 - 상품명 기반으로 동적 이미지 검색
                        if not product_image_url:
                            # 모든 언어 버전의 상품명을 확인하여 이미지 URL 생성
                            # 우선순위: 현재 언어 > 한국어 > 영어 > 일본어
                            image_found = False
                            for lang_key in [current_lang, "ko", "en", "ja"]:
                                check_text = product.get(f"text_{lang_key}", "")
                                if check_text:
                                    check_url = get_product_image_url(check_text)
                                    if check_url:
                                        product_image_url = check_url
                                        image_found = True
                                        break
                            
                            # 모든 언어에서 이미지를 찾지 못한 경우 기본 이미지 사용
                            if not image_found:
                                product_image_url = get_product_image_url(product_text)
                        
                        # 이미지 표시 시도 (로컬 파일 및 URL 모두 지원)
                        image_displayed = False
                        if product_image_url:
                            try:
                                # 로컬 파일 경로인 경우
                                if os.path.exists(product_image_url):
                                    st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                                    image_displayed = True
                                # URL인 경우
                                elif product_image_url.startswith("http://") or product_image_url.startswith("https://"):
                                    try:
                                        # HEAD 요청으로 이미지 존재 여부 확인 (타임아웃 2초)
                                        response = requests.head(product_image_url, timeout=2, allow_redirects=True)
                                        if response.status_code == 200:
                                            st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                                            image_displayed = True
                                        else:
                                            image_displayed = False
                                    except Exception:
                                        # HEAD 요청 실패 시에도 이미지 표시 시도 (일부 서버는 HEAD를 지원하지 않음)
                                        try:
                                            st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                                            image_displayed = True
                                        except Exception:
                                            image_displayed = False
                                else:
                                    # 기타 경로 시도
                                    try:
                                        st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                                        image_displayed = True
                                    except Exception:
                                        image_displayed = False
                            except Exception as img_error:
                                # 이미지 로딩 실패
                                image_displayed = False
                        
                        # 이미지 표시 실패 시 이모지 카드 표시
                        if not image_displayed:
                            product_emoji = "🎫" if "티켓" in product_text or "ticket" in product_text.lower() else \
                                          "🎢" if "테마파크" in product_text or "theme" in product_text.lower() or "디즈니" in product_text or "유니버셜" in product_text or "스튜디오" in product_text else \
                                          "✈️" if "항공" in product_text or "flight" in product_text.lower() else \
                                          "🏨" if "호텔" in product_text or "hotel" in product_text.lower() else \
                                          "🍔" if "음식" in product_text or "food" in product_text.lower() else \
                                          "🌏" if "여행" in product_text or "travel" in product_text.lower() or "사파리" in product_text else \
                                          "📦"
                            st.markdown(
                                f"""
                                <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 10px; color: white; min-height: 200px; display: flex; flex-direction: column; justify-content: center;'>
                                    <h1 style='font-size: 64px; margin: 0;'>{product_emoji}</h1>
                                    <p style='font-size: 16px; margin-top: 15px; font-weight: bold;'>{product_text[:25]}</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        st.write(f"**{product_text}**")
                        st.caption(f"{L.get('popularity', '인기도')}: {product_score}")
                        st.markdown("---")
            
            # 화제의 소식 목록 (상세 내용 포함) - 공백 축소
            if display_data.get("trending_topics"):
                st.markdown(f"#### {L['trending_topics']}")
                for idx, topic in enumerate(display_data["trending_topics"], 1):
                    topic_text = topic.get(f"text_{current_lang}", topic.get("text_ko", ""))
                    topic_score = topic.get("score", 0)
                    topic_detail = topic.get(f"detail_{current_lang}", topic.get("detail_ko", ""))
                    
                    with st.expander(f"{idx}. **{topic_text}** ({L.get('trend_score', '화제도')}: {topic_score})"):
                        if topic_detail:
                            st.write(topic_detail)
                        else:
                            # 상세 내용이 없으면 LLM으로 생성
                            if display_company:
                                try:
                                    # 언어별 프롬프트
                                    detail_prompts = {
                                        "ko": f"{display_company}의 '{topic_text}'에 대한 상세 내용을 200자 이상 작성해주세요.",
                                        "en": f"Please write detailed content of at least 200 characters about '{topic_text}' from {display_company}.",
                                        "ja": f"{display_company}の「{topic_text}」に関する詳細内容を200文字以上で作成してください。"
                                    }
                                    detail_prompt = detail_prompts.get(current_lang, detail_prompts["ko"])
                                    generated_detail = run_llm(detail_prompt)
                                    if generated_detail and not generated_detail.startswith("❌"):
                                        st.write(generated_detail)
                                        # 생성된 상세 내용을 데이터베이스에 저장
                                        if display_company in faq_data.get("companies", {}):
                                            topic_idx = idx - 1
                                            if topic_idx < len(faq_data["companies"][display_company].get("trending_topics", [])):
                                                faq_data["companies"][display_company]["trending_topics"][topic_idx][f"detail_{current_lang}"] = generated_detail
                                                save_faq_database(faq_data)
                                    else:
                                        st.write(L.get("generating_detail", "상세 내용을 생성하는 중입니다..."))
                                except Exception as e:
                                    st.write(L.get("checking_additional_info", "상세 내용: {topic}에 대한 추가 정보를 확인 중입니다.").format(topic=topic_text))
                            else:
                                st.write(L.get("checking_additional_info", "상세 내용: {topic}에 대한 추가 정보를 확인 중입니다.").format(topic=topic_text))
            
            # CEO/대표이사 정보 표시
            if display_data.get("ceo_info"):
                ceo_info = display_data["ceo_info"]
                ceo_name = ceo_info.get(f"name_{current_lang}", ceo_info.get("name_ko", ""))
                ceo_position = ceo_info.get(f"position_{current_lang}", ceo_info.get("position_ko", ""))
                ceo_bio = ceo_info.get(f"bio_{current_lang}", ceo_info.get("bio_ko", ""))
                ceo_tenure = ceo_info.get(f"tenure_{current_lang}", ceo_info.get("tenure_ko", ""))
                ceo_education = ceo_info.get(f"education_{current_lang}", ceo_info.get("education_ko", ""))
                ceo_career = ceo_info.get(f"career_{current_lang}", ceo_info.get("career_ko", ""))
                
                if ceo_name or ceo_position:
                    st.markdown(f"#### 👔 {L.get('ceo_info', 'CEO/대표이사 정보')}")
                    st.markdown("---")
                    
                    # CEO 정보 카드 형태로 표시
                    col_ceo_left, col_ceo_right = st.columns([1, 2])
                    
                    with col_ceo_left:
                        # CEO 이름과 직책
                        if ceo_name:
                            st.markdown(f"### {ceo_name}")
                        if ceo_position:
                            st.markdown(f"**{L.get('position', '직책')}:** {ceo_position}")
                        if ceo_tenure:
                            st.markdown(f"**{L.get('tenure', '재임 기간')}:** {ceo_tenure}")
                    
                    with col_ceo_right:
                        # 상세 소개
                        if ceo_bio:
                            st.markdown(f"**{L.get('ceo_bio', '소개')}**")
                            st.markdown(ceo_bio)
                    
                    # 학력 및 경력 정보
                    if ceo_education or ceo_career:
                        st.markdown("---")
                        col_edu, col_career = st.columns(2)
                        
                        with col_edu:
                            if ceo_education:
                                st.markdown(f"**{L.get('education', '학력')}**")
                                st.markdown(ceo_education)
                        
                        with col_career:
                            if ceo_career:
                                st.markdown(f"**{L.get('career', '주요 경력')}**")
                                st.markdown(ceo_career)
                    
                    st.markdown("---")
            
            # 면접 질문 목록 표시
            if display_data.get("interview_questions"):
                st.markdown(f"#### 💼 {L.get('interview_questions', '면접 예상 질문')}")
                st.markdown(f"*{L.get('interview_questions_desc', '면접에서 나올 만한 핵심 질문들과 상세한 답변입니다. 면접 준비와 회사 이해에 도움이 됩니다.')}*")
                st.markdown("---")
                
                # 카테고리별로 그룹화
                interview_by_category = {}
                for idx, iq in enumerate(display_data["interview_questions"]):
                    question = iq.get(f"question_{current_lang}", iq.get("question_ko", ""))
                    answer = iq.get(f"answer_{current_lang}", iq.get("answer_ko", ""))
                    category = iq.get(f"category_{current_lang}", iq.get("category_ko", L.get("interview_category_other", "기타")))
                    
                    if category not in interview_by_category:
                        interview_by_category[category] = []
                    interview_by_category[category].append({
                        "question": question,
                        "answer": answer,
                        "index": idx + 1
                    })
                
                # 카테고리별로 표시
                for category, questions in interview_by_category.items():
                    with st.expander(f"📋 **{category}** ({len(questions)}{L.get('items', '개')})"):
                        for item in questions:
                            st.markdown(f"**{item['index']}. {item['question']}**")
                            st.markdown(item['answer'])
                            st.markdown("---")
        else:
            st.info(L["company_search_or_select"])
    
    # 탭 2: 자주 묻는 질문 (FAQ) - 검색 기능 포함
    with tab2:
        if display_company and display_data:
            # 제목을 더 간결하게 표시
            st.markdown(f"#### {display_company} - {L['company_faq']}")
            
            # FAQ 검색 기능 (탭 내부에 통합) - 검색 범위 확대, 공백 축소
            col_search_faq, col_btn_faq = st.columns([3.5, 1])
            with col_search_faq:
                faq_search_query = st.text_input(
                    L["faq_search_placeholder"],
                    key="faq_search_in_tab",
                    placeholder=L.get("faq_search_placeholder_extended", L["faq_search_placeholder"])
                )
            with col_btn_faq:
                faq_search_btn = st.button(L["button_search_faq"], key="faq_search_btn_in_tab")
            
            faqs = display_data.get("faqs", [])
            popular_products = display_data.get("popular_products", [])
            trending_topics = display_data.get("trending_topics", [])
            company_info = display_data.get("company_info", "")
            
            # 검색 관련 변수 초기화
            matched_products = []
            matched_topics = []
            matched_info = False
            
            # 검색어가 있으면 확장된 검색 (FAQ, 상품, 화제 소식, 회사 소개 모두 검색)
            if faq_search_query and faq_search_btn:
                query_lower = faq_search_query.lower()
                filtered_faqs = []
                
                # 1. FAQ 검색 (기본 FAQ + 상품명 관련 FAQ)
                for faq in faqs:
                    question = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                    answer = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                    if query_lower in question.lower() or query_lower in answer.lower():
                        filtered_faqs.append(faq)
                
                # 2. 상품명으로 FAQ 검색 (상품명이 검색어와 일치하거나 포함되는 경우)
                # 검색어가 상품명에 포함되면 해당 상품과 관련된 FAQ를 찾아서 표시
                for product in popular_products:
                    product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                    product_text_lower = product_text.lower()
                    
                    # 검색어가 상품명에 포함되는 경우
                    if query_lower in product_text_lower:
                        # 해당 상품명이 FAQ 질문/답변에 포함된 경우 찾기
                        product_related_faqs = []
                        for faq in faqs:
                            question = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                            answer = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                            # 상품명이 FAQ에 언급되어 있으면 추가
                            if product_text_lower in question.lower() or product_text_lower in answer.lower():
                                if faq not in filtered_faqs:
                                    filtered_faqs.append(faq)
                                    product_related_faqs.append(faq)
                        
                        # 상품명이 매칭되었지만 관련 FAQ가 없는 경우, 상품 정보만 표시
                        if not product_related_faqs:
                            matched_products.append(product)
                
                # 2. 인기 상품 검색
                for product in popular_products:
                    product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                    if query_lower in product_text.lower():
                        matched_products.append(product)
                
                # 3. 화제의 소식 검색
                for topic in trending_topics:
                    topic_text = topic.get(f"text_{current_lang}", topic.get("text_ko", ""))
                    if query_lower in topic_text.lower():
                        matched_topics.append(topic)
                
                # 4. 회사 소개 검색
                if query_lower in company_info.lower():
                    matched_info = True
                
                # 검색 결과가 있으면 표시
                if filtered_faqs or matched_products or matched_topics or matched_info:
                    # 매칭된 상품 표시 (FAQ가 없는 경우에만)
                    if matched_products and not filtered_faqs:
                        st.subheader(f"🔍 {L.get('related_products', '관련 상품')} ({len(matched_products)}{L.get('items', '개')})")
                        st.info(L.get("no_faq_for_product", "해당 상품과 관련된 FAQ를 찾을 수 없습니다. 상품 정보만 표시됩니다."))
                        for idx, product in enumerate(matched_products, 1):
                            product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                            product_score = product.get("score", 0)
                            st.write(f"• **{product_text}** ({L.get('popularity', '인기도')}: {product_score})")
                        st.markdown("---")
                    
                    # 매칭된 화제 소식 표시
                    if matched_topics:
                        st.subheader(f"🔍 {L.get('related_trending_news', '관련 화제 소식')} ({len(matched_topics)}{L.get('items', '개')})")
                        for idx, topic in enumerate(matched_topics, 1):
                            topic_text = topic.get(f"text_{current_lang}", topic.get("text_ko", ""))
                            topic_score = topic.get("score", 0)
                            st.write(f"• **{topic_text}** ({L.get('trend_score', '화제도')}: {topic_score})")
                        st.markdown("---")
                    
                    # 매칭된 회사 소개 표시
                    if matched_info:
                        st.subheader(f"🔍 {L.get('related_company_info', '관련 회사 소개 내용')}")
                        # 검색어가 포함된 부분 강조하여 표시
                        info_lower = company_info.lower()
                        query_pos = info_lower.find(query_lower)
                        if query_pos != -1:
                            start = max(0, query_pos - 100)
                            end = min(len(company_info), query_pos + len(query_lower) + 100)
                            snippet = company_info[start:end]
                            if start > 0:
                                snippet = "..." + snippet
                            if end < len(company_info):
                                snippet = snippet + "..."
                            # 검색어 강조
                            highlighted = snippet.replace(
                                query_lower, 
                                f"**{query_lower}**"
                            )
                            st.write(highlighted)
                        st.markdown("---")
                    
                    # FAQ 결과
                    faqs = filtered_faqs
                else:
                    faqs = []
            
            # FAQ 목록 표시
            if faqs:
                if faq_search_query and faq_search_btn:
                    st.subheader(f"🔍 {L.get('related_faq', '관련 FAQ')} ({len(faqs)}{L.get('items', '개')})")
                else:
                    st.subheader(f"{L['company_faq']} ({len(faqs)}{L.get('items', '개')})")
                for idx, faq in enumerate(faqs, 1):
                    question = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                    answer = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                    with st.expander(f"{L['faq_question_prefix'].format(num=idx)} {question}"):
                        st.write(f"**{L['faq_answer']}:** {answer}")
            else:
                if faq_search_query and faq_search_btn:
                    # 검색 결과가 없을 때만 메시지 표시 (위에서 이미 관련 상품/소식 등이 표시되었을 수 있음)
                    if not (matched_products or matched_topics or matched_info):
                        st.info(L["no_faq_results"])
                else:
                    st.info(L.get("no_faq_for_company", f"{display_company}의 FAQ가 없습니다.").format(company=display_company))
        else:
            st.info(L.get("no_company_selected", "회사명을 검색하거나 선택해주세요."))
    
    # 탭 3: 고객 문의 재확인 (에이전트용)
    with tab3:
        # 제목과 설명을 한 줄로 간결하게 표시
        st.markdown(f"#### {L['customer_inquiry_review']}")
        st.caption(L.get("customer_inquiry_review_desc", "에이전트가 상사들에게 고객 문의 내용을 재확인하고, AI 답안 및 힌트를 생성할 수 있는 기능입니다."))
        
        # 세션 상태 초기화
        if "generated_ai_answer" not in st.session_state:
            st.session_state.generated_ai_answer = None
        if "generated_hint" not in st.session_state:
            st.session_state.generated_hint = None
        
        # 회사 선택 (선택사항)
        selected_company_for_inquiry = None
        if companies:
            all_option = L.get("all_companies", "전체")
            selected_company_for_inquiry = st.selectbox(
                f"{L['select_company']} ({L.get('optional', '선택사항')})",
                options=[all_option] + companies,
                key="inquiry_company_select"
            )
            if selected_company_for_inquiry == all_option:
                selected_company_for_inquiry = None
        
        # 고객 문의 내용 입력
        customer_inquiry = st.text_area(
            L["inquiry_question_label"],
            placeholder=L["inquiry_question_placeholder"],
            key="customer_inquiry_input",
            height=150
        )
        
        # 고객 첨부 파일 업로드
        uploaded_file = st.file_uploader(
            L.get("inquiry_attachment_label", "📎 고객 첨부 파일 업로드 (사진/스크린샷)"),
            type=["png", "jpg", "jpeg", "pdf"],
            key="customer_inquiry_attachment",
            help=L.get("inquiry_attachment_help", "특히 취소 불가 여행상품의 비행기 지연, 여권 이슈 등 불가피한 사유의 경우, 반드시 사진이나 스크린샷을 첨부해주세요.")
        )
        
        # 업로드된 파일 정보 저장
        attachment_info = ""
        uploaded_file_info = None
        file_content_extracted = ""
        file_content_translated = ""
        
        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_type = uploaded_file.type
            file_size = len(uploaded_file.getvalue())
            st.success(L.get("inquiry_attachment_uploaded", "✅ 첨부 파일이 업로드되었습니다: {filename}").format(filename=file_name))
            
            # 파일 정보 저장
            uploaded_file_info = {
                "name": file_name,
                "type": file_type,
                "size": file_size
            }
            
            # 파일 내용 추출 (PDF, TXT, 이미지 파일인 경우)
            if file_name.lower().endswith(('.pdf', '.txt', '.png', '.jpg', '.jpeg')):
                try:
                    with st.spinner(L.get("extracting_file_content", "파일 내용 추출 중...")):
                        if file_name.lower().endswith('.pdf'):
                            import tempfile
                            import os
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                            tmp.write(uploaded_file.getvalue())
                            tmp.flush()
                            tmp.close()
                            try:
                                loader = PyPDFLoader(tmp.name)
                                file_docs = loader.load()
                                file_content_extracted = "\n".join([doc.page_content for doc in file_docs])
                            finally:
                                try:
                                    os.remove(tmp.name)
                                except:
                                    pass
                        elif file_name.lower().endswith('.txt'):
                            uploaded_file.seek(0)  # 파일 포인터를 처음으로 이동
                            file_content_extracted = uploaded_file.read().decode("utf-8", errors="ignore")
                        elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # 이미지 파일의 경우 OCR을 사용하여 텍스트 추출
                            uploaded_file.seek(0)
                            image_bytes = uploaded_file.getvalue()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            
                            # Gemini Vision API를 사용하여 이미지에서 텍스트 추출
                            ocr_prompt = """이 이미지에 있는 모든 텍스트를 정확히 추출해주세요. 
이미지에 한국어, 일본어, 영어 등 어떤 언어의 텍스트가 있든 모두 추출하고, 
텍스트의 구조와 순서를 유지해주세요. 
이미지에 텍스트가 없으면 "텍스트 없음"이라고 답변하세요.

추출된 텍스트:"""
                            
                            try:
                                # Gemini Vision API 호출
                                gemini_key = get_api_key("gemini")
                                if gemini_key:
                                    import google.generativeai as genai
                                    genai.configure(api_key=gemini_key)
                                    model = genai.GenerativeModel('gemini-2.0-flash-exp')
                                    
                                    # 이미지와 프롬프트를 함께 전송
                                    response = model.generate_content([
                                        {
                                            "mime_type": file_type,
                                            "data": image_bytes
                                        },
                                        ocr_prompt
                                    ])
                                    file_content_extracted = response.text if response.text else ""
                                else:
                                    # Gemini 키가 없으면 LLM에 base64 이미지를 전송하여 OCR 요청
                                    ocr_llm_prompt = f"""{ocr_prompt}

이미지는 base64로 인코딩되어 전송되었습니다. 이미지에서 텍스트를 추출해주세요."""
                                    # LLM이 이미지를 직접 처리할 수 없으므로, 사용자에게 안내
                                    file_content_extracted = ""
                                    st.info(L.get("ocr_requires_manual", "이미지 OCR을 위해서는 Gemini API 키가 필요합니다. 이미지의 텍스트를 수동으로 입력해주세요."))
                            except Exception as ocr_error:
                                error_msg = L.get("ocr_error", "이미지 텍스트 추출 중 오류: {error}")
                                st.warning(error_msg.format(error=str(ocr_error)))
                                file_content_extracted = ""
                        
                        # 파일 내용이 추출된 경우 언어 감지 및 번역 (일본어/영어 버전에서 한국어 파일 번역)
                        if file_content_extracted and current_lang in ["ja", "en"]:
                            # 한국어 내용인지 확인하고 번역
                            with st.spinner(L.get("detecting_language", "언어 감지 중...")):
                                # 언어 감지 프롬프트 (현재 언어에 맞춤)
                                detect_prompts = {
                                    "ja": f"""次のテキストの言語を検出してください。韓国語、日本語、英語のいずれかで答えてください。

テキスト:
{file_content_extracted[:500]}

言語:""",
                                    "en": f"""Detect the language of the following text. Answer with only one of: Korean, Japanese, or English.

Text:
{file_content_extracted[:500]}

Language:""",
                                    "ko": f"""다음 텍스트의 언어를 감지해주세요. 한국어, 일본어, 영어 중 하나로만 답변하세요.

텍스트:
{file_content_extracted[:500]}

언어:"""
                                }
                                detect_prompt = detect_prompts.get(current_lang, detect_prompts["ko"])
                                detected_lang = run_llm(detect_prompt).strip().lower()
                                
                                # 한국어로 감지된 경우 현재 언어로 번역
                                if "한국어" in detected_lang or "korean" in detected_lang or "ko" in detected_lang:
                                    with st.spinner(L.get("translating_content", "파일 내용 번역 중...")):
                                        # 번역 프롬프트 (현재 언어에 맞춤)
                                        translate_prompts = {
                                            "ja": f"""次の韓国語テキストを日本語に翻訳してください。原文の意味とトーンを正確に維持しながら、自然な日本語で翻訳してください。

韓国語テキスト:
{file_content_extracted}

日本語翻訳:""",
                                            "en": f"""Please translate the following Korean text into English. Maintain the exact meaning and tone of the original text while translating into natural English.

Korean text:
{file_content_extracted}

English translation:"""
                                        }
                                        translate_prompt = translate_prompts.get(current_lang)
                                        if translate_prompt:
                                            file_content_translated = run_llm(translate_prompt)
                                            if file_content_translated and not file_content_translated.startswith("❌"):
                                                st.info(L.get("file_translated", "✅ 파일 내용이 번역되었습니다."))
                                            else:
                                                file_content_translated = ""
                except Exception as e:
                    error_msg = L.get("file_extraction_error", "파일 내용 추출 중 오류가 발생했습니다: {error}")
                    st.warning(error_msg.format(error=str(e)))
            
            # 언어별 파일 정보 텍스트 생성
            file_content_to_include = file_content_translated if file_content_translated else file_content_extracted
            content_section = ""
            if file_content_to_include:
                content_section = f"\n\n[파일 내용]\n{file_content_to_include[:2000]}"  # 최대 2000자만 포함
                if len(file_content_to_include) > 2000:
                    content_section += "\n...(내용이 길어 일부만 표시됨)"
            
            attachment_info_by_lang = {
                "ko": f"\n\n[고객 첨부 파일 정보]\n- 파일명: {file_name}\n- 파일 타입: {file_type}\n- 파일 크기: {file_size} bytes\n- 참고: 고객이 {file_name} 파일을 첨부했습니다. 이 파일은 비행기 지연, 여권 이슈, 질병 등 불가피한 사유로 인한 취소 불가 여행상품 관련 증빙 자료일 수 있습니다. 파일 내용을 참고하여 응대하세요.{content_section}",
                "en": f"\n\n[Customer Attachment Information]\n- File name: {file_name}\n- File type: {file_type}\n- File size: {file_size} bytes\n- Note: The customer has attached the file {file_name}. This file may be evidence related to non-refundable travel products due to unavoidable reasons such as flight delays, passport issues, illness, etc. Please refer to the file content when responding.{content_section}",
                "ja": f"\n\n[顧客添付ファイル情報]\n- ファイル名: {file_name}\n- ファイルタイプ: {file_type}\n- ファイルサイズ: {file_size} bytes\n- 参考: 顧客が{file_name}ファイルを添付しました。このファイルは、飛行機の遅延、パスポートの問題、病気などやむを得ない理由によるキャンセル不可の旅行商品に関連する証拠資料である可能性があります。ファイルの内容を参照して対応してください。{content_section}"
            }
            attachment_info = attachment_info_by_lang.get(current_lang, attachment_info_by_lang["ko"])
            
            # 이미지 파일인 경우 미리보기 표시
            if file_type and file_type.startswith("image/"):
                st.image(uploaded_file, caption=file_name, use_container_width=True)
        
        col_ai_answer, col_hint = st.columns(2)
        
        # AI 답안 생성
        with col_ai_answer:
            if st.button(L["button_generate_ai_answer"], key="generate_ai_answer_btn", type="primary"):
                if customer_inquiry:
                    with st.spinner(L["generating_ai_answer"]):
                        # 회사 정보가 있으면 포함하여 답안 생성
                        company_context = ""
                        if selected_company_for_inquiry and selected_company_for_inquiry in faq_data.get("companies", {}):
                            company_data = get_company_info_faq(selected_company_for_inquiry, current_lang)
                            company_info_label = L.get("company_info", "회사 정보")
                            company_context = f"\n\n{company_info_label}: {company_data.get('info', '')}"
                            # 관련 FAQ도 포함
                            related_faqs = company_data.get("faqs", [])[:5]  # 상위 5개만
                            if related_faqs:
                                faq_label = L.get("company_faq", "자주 나오는 질문")
                                faq_context = f"\n\n{faq_label}:\n"
                                for faq in related_faqs:
                                    q = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                                    a = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                                    faq_context += f"Q: {q}\nA: {a}\n"
                                company_context += faq_context
                        
                        # 언어별 프롬프트
                        lang_prompts_inquiry = {
                            "ko": f"""다음 고객 문의에 대한 전문적이고 친절한 답안을 작성해주세요.

고객 문의: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

답안은 다음을 포함해야 합니다:
1. 고객의 문의에 대한 명확한 답변
2. 필요한 경우 추가 정보나 안내
3. 친절하고 전문적인 톤
4. 첨부 파일이 있는 경우, 해당 파일 내용을 참고하여 응대하세요. 특히 취소 불가 여행상품의 비행기 지연, 여권 이슈 등 불가피한 사유의 경우, 첨부된 증빙 자료를 확인하고 적절히 대응하세요.

답안:""",
                            "en": f"""Please write a professional and friendly answer to the following customer inquiry.

Customer Inquiry: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

The answer should include:
1. Clear answer to the customer's inquiry
2. Additional information or guidance if needed
3. Friendly and professional tone
4. If there is an attachment, please reference the file content in your response. For non-refundable travel products with unavoidable reasons (flight delays, passport issues, etc.), review the attached evidence and respond appropriately.

Answer:""",
                            "ja": f"""次の顧客問い合わせに対する専門的で親切な回答を作成してください。

顧客問い合わせ: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

回答には以下を含める必要があります:
1. 顧客の問い合わせに対する明確な回答
2. 必要に応じて追加情報や案内
3. 親切で専門的なトーン
4. 添付ファイルがある場合は、そのファイルの内容を参照して対応してください。特にキャンセル不可の旅行商品で、飛行機の遅延、パスポートの問題などやむを得ない理由がある場合は、添付された証拠資料を確認し、適切に対応してください。

回答:"""
                        }
                        prompt = lang_prompts_inquiry.get(current_lang, lang_prompts_inquiry["ko"])
                        
                        ai_answer = run_llm(prompt)
                        st.session_state.generated_ai_answer = ai_answer
                        st.success(f"✅ {L.get('ai_answer_generated', 'AI 답안이 생성되었습니다.')}")
                else:
                    st.warning(L.get("warning_enter_inquiry", "고객 문의 내용을 입력해주세요."))
        
        # 응대 힌트 생성
        with col_hint:
            if st.button(L["button_generate_hint"], key="generate_hint_btn", type="primary"):
                if customer_inquiry:
                    with st.spinner(L["generating_hint"]):
                        # 회사 정보가 있으면 포함하여 힌트 생성
                        company_context = ""
                        if selected_company_for_inquiry and selected_company_for_inquiry in faq_data.get("companies", {}):
                            company_data = get_company_info_faq(selected_company_for_inquiry, current_lang)
                            company_info_label = L.get("company_info", "회사 정보")
                            company_context = f"\n\n{company_info_label}: {company_data.get('info', '')}"
                        
                        # 언어별 프롬프트
                        lang_prompts_hint = {
                            "ko": f"""다음 고객 문의에 대한 응대 힌트를 작성해주세요.

고객 문의: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

응대 힌트는 다음을 포함해야 합니다:
1. 고객 문의의 핵심 포인트
2. 응대 시 주의사항
3. 권장 응대 방식
4. 추가 확인이 필요한 사항 (있는 경우)
5. 첨부 파일이 있는 경우, 해당 파일을 확인하고 증빙 자료로 활용하세요. 특히 취소 불가 여행상품의 경우, 첨부된 사진이나 스크린샷을 통해 불가피한 사유를 확인하고 적절한 조치를 취하세요.

응대 힌트:""",
                            "en": f"""Please write response hints for the following customer inquiry.

Customer Inquiry: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

Response hints should include:
1. Key points of the customer inquiry
2. Precautions when responding
3. Recommended response method
4. Items that need additional confirmation (if any)
5. If there is an attachment, review the file and use it as evidence. For non-refundable travel products, verify unavoidable reasons through attached photos or screenshots and take appropriate action.

Response Hints:""",
                            "ja": f"""次の顧客問い合わせに対する対応ヒントを作成してください。

顧客問い合わせ: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

対応ヒントには以下を含める必要があります:
1. 顧客問い合わせの核心ポイント
2. 対応時の注意事項
3. 推奨対応方法
4. 追加確認が必要な事項（ある場合）
5. 添付ファイルがある場合は、そのファイルを確認し、証拠資料として活用してください。特にキャンセル不可の旅行商品の場合、添付された写真やスクリーンショットを通じてやむを得ない理由を確認し、適切な措置を取ってください。

対応ヒント:"""
                        }
                        prompt = lang_prompts_hint.get(current_lang, lang_prompts_hint["ko"])
                        
                        hint = run_llm(prompt)
                        st.session_state.generated_hint = hint
                        st.success(f"✅ {L.get('hint_generated', '응대 힌트가 생성되었습니다.')}")
                else:
                    st.warning(L.get("warning_enter_inquiry", "고객 문의 내용을 입력해주세요."))
        
        # 생성된 결과 표시
        if st.session_state.get("generated_ai_answer"):
            st.markdown("---")
            st.subheader(L["ai_answer_header"])
            
            answer_text = st.session_state.generated_ai_answer
            
            # 답안을 선택 가능한 텍스트로 표시 (폰트 크기 확대)
            import html as html_escape
            answer_escaped = html_escape.escape(answer_text)
            st.markdown(f"""
            <div style="font-size: 18px; line-height: 1.8; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Malgun Gothic', '맑은 고딕', 'Noto Sans JP', sans-serif; margin: 0; font-size: 18px; color: #212529;">{answer_escaped}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            # 다운로드 버튼 추가 (더 안정적인 복사 방법)
            col_copy, col_download = st.columns(2)
            with col_copy:
                st.info(L.get("copy_instruction", "💡 위 텍스트를 선택하고 Ctrl+C (Mac: Cmd+C)로 복사하세요."))
            with col_download:
                st.download_button(
                    label=f"📥 {L.get('button_download_answer', '답안 다운로드')}",
                    data=answer_text.encode('utf-8'),
                    file_name=f"ai_answer_{st.session_state.get('copy_answer_id', 0)}.txt",
                    mime="text/plain",
                    key="download_answer_btn"
                )
        
        if st.session_state.get("generated_hint"):
            st.markdown("---")
            st.subheader(L["hint_header"])
            
            hint_text = st.session_state.generated_hint
            
            # 힌트를 선택 가능한 텍스트로 표시 (폰트 크기 확대)
            import html as html_escape
            hint_escaped = html_escape.escape(hint_text)
            st.markdown(f"""
            <div style="font-size: 18px; line-height: 1.8; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Malgun Gothic', '맑은 고딕', 'Noto Sans JP', sans-serif; margin: 0; font-size: 18px; color: #212529;">{hint_escaped}</pre>
            </div>
    # =========================
    # 0. 전체 이력 삭제
    # =========================
    col_del, _ = st.columns([1, 4])
    with col_del:
        if st.button(L["delete_history_button"], key="trigger_delete_hist"):
            st.session_state.show_delete_confirm = True

    if st.session_state.show_delete_confirm:
        with st.container():
            st.warning(L["delete_confirm_message"])
            c_yes, c_no = st.columns(2)
            if c_yes.button(L["delete_confirm_yes"], key="confirm_del_yes"):
                with st.spinner(L["deleting_history_progress"]):
                    delete_all_history_local()
                    st.session_state.simulator_messages = []
                    st.session_state.simulator_memory.clear()
                    st.session_state.show_delete_confirm = False
                    st.session_state.is_chat_ended = False
                    st.session_state.sim_stage = "WAIT_FIRST_QUERY"
                    st.session_state.customer_attachment_file = []  # 첨부 파일 초기화
                    st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
                    st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
                    st.success(L["delete_success"])
            if c_no.button(L["delete_confirm_no"], key="confirm_del_no"):
                st.session_state.show_delete_confirm = False

    # =========================
    # 1. 이전 이력 로드 (검색/필터링 기능 개선)
    # =========================
    with st.expander(L["history_expander_title"]):
        # Always load all available histories for the current language (sorted by recency)
        histories = load_simulation_histories_local(current_lang)

        # 전체 통계 및 트렌드 대시보드 (요약 데이터가 있는 경우만)
        cases_with_summary = [
            h for h in histories
            if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
               and not h.get("is_call", False)  # 전화 이력 제외
        ]

        if cases_with_summary:
            st.markdown("---")
            st.subheader("📈 과거 케이스 트렌드 대시보드")

            # 트렌드 차트 표시
            trend_chart = visualize_case_trends(histories, current_lang)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            else:
                # Plotly가 없을 경우 텍스트로 표시
                avg_sentiment = np.mean(
                    [h["summary"].get("customer_sentiment_score", 50) for h in cases_with_summary if h.get("summary")])
                avg_satisfaction = np.mean(
                    [h["summary"].get("customer_satisfaction_score", 50) for h in cases_with_summary if
                     h.get("summary")])
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("평균 감정 점수", f"{avg_sentiment:.1f}/100", f"총 {len(cases_with_summary)}건")
                with col2:
                    st.metric("평균 만족도", f"{avg_satisfaction:.1f}/100", f"총 {len(cases_with_summary)}건")

            st.markdown("---")

        # ⭐ 검색 폼 제거 및 독립된 위젯 사용
        col_search, col_btn = st.columns([4, 1])

        with col_search:
            # st.text_input은 Enter 키 입력 시 앱을 재실행합니다.
            search_query = st.text_input(L["search_history_label"], key="sim_hist_search_input_new")

        with col_btn:
            # 검색 버튼: 누르면 앱을 강제 재실행하여 검색/필터링 로직을 다시 타도록 합니다.
            st.markdown("<br>", unsafe_allow_html=True)  # Align button vertically
            search_clicked = st.button(L["history_search_button"], key="apply_search_btn_new")

        # 날짜 범위 필터
        today = datetime.now().date()
        date_range_value = [today - timedelta(days=7), today]
        dr = st.date_input(
            L["date_range_label"],
            value=date_range_value,
            key="sim_hist_date_range_actual",
        )

        # --- Filtering Logic ---
        current_search_query = search_query.strip()

        if histories:
            start_date = min(dr)
            end_date = max(dr)

            filtered = []
            for h in histories:
                # 전화 이력은 제외 (채팅/이메일 탭이므로)
                if h.get("is_call", False):
                    continue

                ok_search = True
                if current_search_query:
                    q = current_search_query.lower()
                    # 검색 대상: 초기 문의, 고객 유형, 요약 데이터
                    text = (h["initial_query"] + " " + h["customer_type"]).lower()

                    # 요약 데이터가 있으면 요약 내용도 검색 대상에 포함
                    summary = h.get("summary")
                    if summary and isinstance(summary, dict):
                        summary_text = summary.get("main_inquiry", "") + " " + summary.get("summary", "")
                        text += " " + summary_text.lower()

                    # Check if query matches in initial query, customer type, or summary
                    if q not in text:
                        ok_search = False

                ok_date = True
                ts = h.get("timestamp")
                if ts:
                    try:
                        d = datetime.fromisoformat(ts).date()
                        # Apply date filtering
                        if not (start_date <= d <= end_date):
                            ok_date = False
                    except Exception:
                        pass  # Ignore histories with invalid timestamp

                if ok_search and ok_date:
                    filtered.append(h)
        else:
            filtered = []

        # Determine the list for display (⭐ 요청 사항: 검색어/필터가 없으면 최근 10건만 표시)
        is_searching_or_filtering = bool(current_search_query) or dr != date_range_value

        if not is_searching_or_filtering:
            # 검색/필터 조건이 없으면, 전체 이력 중 최신 10건만 표시
            filtered_for_display = filtered[:10]  # 필터링된 목록(전화 제외) 중 10개
        else:
            # 검색/필터 조건이 있으면, 필터링된 모든 결과를 표시
            filtered_for_display = filtered

        # --- Display Logic ---

        if filtered_for_display:
            def _label(h):
                try:
                    t = datetime.fromisoformat(h["timestamp"])
                    t_str = t.strftime("%m-%d %H:%M")
                except Exception:
                    t_str = h.get("timestamp", "")

                # 요약 데이터가 있으면 요약 정보 표시, 없으면 초기 문의 표시
                summary = h.get("summary")
                if summary and isinstance(summary, dict):
                    main_inquiry = summary.get("main_inquiry", h["initial_query"][:30])
                    sentiment = summary.get("customer_sentiment_score", 50)
                    satisfaction = summary.get("customer_satisfaction_score", 50)
                    q = main_inquiry[:30].replace("\n", " ")
                    # 첨부 파일 여부 표시 추가
                    attachment_icon = "📎" if h.get("attachment_context") else ""
                    # 요약 데이터 표시 (감정/만족도 점수 포함)
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} | 감정:{sentiment} 만족:{satisfaction} - {q}..."
                else:
                    q = h["initial_query"][:30].replace("\n", " ")
                    attachment_icon = "📎" if h.get("attachment_context") else ""
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} - {q}..."


            options_map = {_label(h): h for h in filtered_for_display}

            # Show a message indicating what is displayed if filters were applied
            if is_searching_or_filtering:
                st.caption(f"🔎 총 {len(filtered_for_display)}개 이력 검색됨 (전화 이력 제외)")
            else:
                st.caption(f"⭐ 최근 {len(filtered_for_display)}개 이력 표시 중 (전화 이력 제외)")

            sel_key = st.selectbox(L["history_selectbox_label"], options=list(options_map.keys()))

            if st.button(L["history_load_button"], key="load_hist_btn"):
                h = options_map[sel_key]
                st.session_state.customer_query_text_area = h["initial_query"]

                # 메시지가 비어있고 요약 데이터가 있는 경우, 요약을 기반으로 최소한의 메시지 재구성
                if not h.get("messages") and h.get("summary"):
                    summary = h["summary"]
                    # 요약 데이터를 기반으로 기본 메시지 구조 생성
                    reconstructed_messages = [
                        {"role": "customer", "content": h["initial_query"]}
                    ]
                    # 요약에서 핵심 응답 추가
                    if summary.get("key_responses"):
                        for response in summary.get("key_responses", [])[:3]:  # 최대 3개만
                            reconstructed_messages.append({"role": "agent_response", "content": response})
                    # 요약 정보를 supervisor 메시지로 추가
                    summary_text = f"**요약된 상담 이력**\n\n"
                    summary_text += f"주요 문의: {summary.get('main_inquiry', 'N/A')}\n"
                    summary_text += f"고객 감정 점수: {summary.get('customer_sentiment_score', 50)}/100\n"
                    summary_text += f"고객 만족도: {summary.get('customer_satisfaction_score', 50)}/100\n"
                    summary_text += f"\n전체 요약:\n{summary.get('summary', 'N/A')}"
                    reconstructed_messages.append({"role": "supervisor", "content": summary_text})
                    st.session_state.simulator_messages = reconstructed_messages

                    # 요약 데이터 시각화
                    st.markdown("---")
                    st.subheader("📊 로드된 케이스 분석")

                    # 요약 데이터를 프로필 형식으로 변환
                    loaded_profile = {
                        "sentiment_score": summary.get("customer_sentiment_score", 50),
                        "urgency_level": "medium",  # 기본값
                        "predicted_customer_type": h.get("customer_type", "normal")
                    }

                    # 프로필 점수 차트
                    profile_chart = visualize_customer_profile_scores(loaded_profile, current_lang)
                    if profile_chart:
                        st.plotly_chart(profile_chart, use_container_width=True)
                    else:
                        # Plotly가 없을 경우 텍스트로 표시
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(L.get("sentiment_score_label", "감정 점수"),
                                      f"{summary.get('customer_sentiment_score', 50)}/100")
                        with col2:
                            st.metric(L.get("urgency_score_label", "긴급도"), f"50/100")
                        with col3:
                            st.metric(L.get("customer_type_label", "고객 유형"), h.get("customer_type", "normal"))

                    # 고객 특성 시각화
                    if summary.get("customer_characteristics") or summary.get("privacy_info"):
                        characteristics_chart = visualize_customer_characteristics(summary, current_lang)
                        if characteristics_chart:
                            st.plotly_chart(characteristics_chart, use_container_width=True)
                else:
                    # 기존 메시지가 있는 경우 그대로 사용
                    st.session_state.simulator_messages = h.get("messages", [])

                st.session_state.initial_advice_provided = True
                st.session_state.is_chat_ended = h.get("is_chat_ended", False)
                st.session_state.sim_attachment_context_for_llm = h.get("attachment_context", "")  # 컨텍스트 로드
                st.session_state.customer_attachment_file = []  # 로드된 이력에는 파일 객체 대신 컨텍스트 문자열만 사용
                st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화

                # 상태 복원
                if st.session_state.is_chat_ended:
                    st.session_state.sim_stage = "CLOSING"
                else:
                    messages = st.session_state.simulator_messages
                    last_role = messages[-1]["role"] if messages else None
                    if last_role == "agent_response":
                        st.session_state.sim_stage = "CUSTOMER_TURN"
                    elif last_role == "customer_rebuttal":
                        st.session_state.sim_stage = "AGENT_TURN"
                    elif last_role == "supervisor" and messages and messages[-1]["content"] == L[
                        "customer_closing_confirm"]:
                        st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                    else:
                        st.session_state.sim_stage = "AGENT_TURN"

                st.session_state.simulator_memory.clear()  # 메모리 초기화
        else:
            st.info(L["no_history_found"])

    # =========================
    # AHT 타이머 (화면 최상단)
    # =========================
    if st.session_state.sim_stage not in ["WAIT_FIRST_QUERY", "CLOSING", "idle"]:
        elapsed_placeholder = st.empty()

        if st.session_state.start_time is not None:
            # 실시간 업데이트를 위해 페이지 로드 시마다 현재 시간 계산
            elapsed_time = datetime.now() - st.session_state.start_time
            total_seconds = elapsed_time.total_seconds()

            # Hold 시간 제외 (채팅/이메일은 Hold 없음, 전화 탭과 로직 통일 위해 유지)
            # total_seconds -= st.session_state.total_hold_duration.total_seconds()

            # 시간 형식 포맷팅
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            # 경고 기준
            if total_seconds > 900:  # 15분
                delta_str = L["timer_info_risk"]
                delta_color = "inverse"
            elif total_seconds > 600:  # 10분
                delta_str = L["timer_info_warn"]
                delta_color = "off"
            else:
                delta_str = L["timer_info_ok"]
                delta_color = "normal"

            elapsed_placeholder.metric(
                L["timer_metric"],
                time_str,
                delta=delta_str,
                delta_color=delta_color
            )

            # ⭐ 수정: 3초마다 재실행하여 AHT 실시간성 확보
            if seconds % 3 == 0 and total_seconds < 1000:
                time.sleep(1)

        st.markdown("---")

    # =========================
    # 2. LLM 준비 체크 & 채팅 종료 상태
    # =========================
    if not st.session_state.is_llm_ready:
        st.warning(L["simulation_no_key_warning"])

    if st.session_state.sim_stage == "CLOSING":
        st.success(L["survey_sent_confirm"])
        st.info(L["new_simulation_ready"])
        
        # ⭐ 추가: 현재 세션 이력 다운로드 기능
        st.markdown("---")
        st.markdown("**📥 현재 세션 이력 다운로드**")
        download_col1, download_col2, download_col3 = st.columns(3)
        
        # 현재 세션의 이력을 생성
        current_session_history = None
        if st.session_state.simulator_messages:
            try:
                customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
                current_session_summary = generate_chat_summary(
                    st.session_state.simulator_messages,
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.language
                )
                current_session_history = [{
                    "id": f"session_{st.session_state.sim_instance_id}",
                    "timestamp": datetime.now().isoformat(),
                    "initial_query": st.session_state.customer_query_text_area,
                    "customer_type": customer_type_display,
                    "language_key": st.session_state.language,
                    "messages": st.session_state.simulator_messages,
                    "summary": current_session_summary,
                    "is_chat_ended": True,
                    "attachment_context": st.session_state.sim_attachment_context_for_llm
                }]
            except Exception as e:
                st.warning(f"이력 생성 중 오류 발생: {e}")
        
        # 다운로드 버튼들을 직접 표시
        if current_session_history:
            # 현재 언어 가져오기
            current_lang = st.session_state.get("language", "ko")
            if current_lang not in ["ko", "en", "ja"]:
                current_lang = "ko"
            
            with download_col1:
                try:
                    filepath_word = export_history_to_word(current_session_history, lang=current_lang)
                    with open(filepath_word, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_word", "📥 이력 다운로드 (Word)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_word),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_word_file"
                        )
                except Exception as e:
                    st.error(f"Word 다운로드 오류: {e}")
            
            with download_col2:
                try:
                    filepath_pptx = export_history_to_pptx(current_session_history, lang=current_lang)
                    with open(filepath_pptx, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pptx", "📥 이력 다운로드 (PPTX)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pptx),
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key="download_pptx_file"
                        )
                except Exception as e:
                    st.error(f"PPTX 다운로드 오류: {e}")
            
            with download_col3:
                try:
                    filepath_pdf = export_history_to_pdf(current_session_history, lang=current_lang)
                    with open(filepath_pdf, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pdf", "📥 이력 다운로드 (PDF)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pdf),
                            mime="application/pdf",
                            key="download_pdf_file"
                        )
                except Exception as e:
                    st.error(f"PDF 다운로드 오류: {e}")
        else:
            st.warning("다운로드할 이력이 없습니다.")
        
        st.markdown("---")
        
        if st.button(L["new_simulation_button"], key="new_simulation_btn"):
            # 초기화 로직
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.initial_advice_provided = False
            st.session_state.is_chat_ended = False
            st.session_state.agent_response_area_text = ""
            st.session_state.customer_query_text_area = ""
            st.session_state.last_transcript = ""
            st.session_state.sim_audio_bytes = None
            st.session_state.sim_stage = "WAIT_FIRST_QUERY"
            st.session_state.customer_attachment_file = []  # 첨부 파일 초기화
            st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
            st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
            st.session_state.start_time = None
            # 전화 발신 관련 상태 초기화
            st.session_state.sim_call_outbound_summary = ""
            st.session_state.sim_call_outbound_target = None
        # st.stop()

    # =========================
    # 5-A. 전화 발신 진행 중 (OUTBOUND_CALL_IN_PROGRESS)
    # =========================
    elif st.session_state.sim_stage == "OUTBOUND_CALL_IN_PROGRESS":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        target = st.session_state.get("sim_call_outbound_target", "대상")
        st.warning(L["call_outbound_loading"])

        # LLM 호출 및 요약 생성
        with st.spinner(L["call_outbound_loading"]):
            # 1. LLM 호출하여 통화 요약 생성
            summary = generate_outbound_call_summary(
                st.session_state.customer_query_text_area,
                st.session_state.language,
                target
            )

            # 2. 시스템 메시지 (전화 시도) 추가
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": L["call_outbound_system_msg"].format(target=target)}
            )

            # 3. 요약 메시지 (결과) 추가
            summary_markdown = f"### {L['call_outbound_summary_header']}\n\n{summary}"
            st.session_state.simulator_messages.append(
                {"role": "supervisor", "content": summary_markdown}
            )

            # 4. Agent Turn으로 복귀
            st.session_state.sim_stage = "AGENT_TURN"
            st.session_state.sim_call_outbound_summary = summary_markdown  # Save for display/reference
            st.session_state.sim_call_outbound_target = None  # Reset target

            # 5. 이력 저장 (전화 발신 후 상태 저장)
            customer_type_display = st.session_state.get("customer_type_sim_select", "")
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display + f" (Outbound Call to {target})",
                st.session_state.simulator_messages, is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

        st.success(f"✅ {L['call_outbound_simulation_header']}가 완료되었습니다. 요약을 확인하고 고객에게 회신하세요.")

    # ========================================
    # 3. 초기 문의 입력 (WAIT_FIRST_QUERY)
    # ========================================
    if st.session_state.sim_stage == "WAIT_FIRST_QUERY":
        customer_query = st.text_area(
            L["customer_query_label"],
            key="customer_query_text_area",
            height=150,
            placeholder=L["initial_query_sample"],
        )

        # --- 필수 입력 필드 (요청 3 반영: UI 텍스트 변경) ---
        customer_email = st.text_input(
            L["customer_email_label"],
            key="customer_email_input",
            value=st.session_state.customer_email,
        )
        customer_phone = st.text_input(
            L["customer_phone_label"],
            key="customer_phone_input",
            value=st.session_state.customer_phone,
        )
        # 세션 상태 업데이트
        st.session_state.customer_email = customer_email
        st.session_state.customer_phone = customer_phone
        # --------------------------------------------------

        customer_type_options = L["customer_type_options"]
        # st.session_state.customer_type_sim_select는 이미 초기화됨
        default_idx = customer_type_options.index(
            st.session_state.customer_type_sim_select) if st.session_state.customer_type_sim_select in customer_type_options else 0

        # Selectbox는 자체적으로 세션 상태를 업데이트하므로, 여기에 value를 설정할 필요 없음
        st.session_state.customer_type_sim_select = st.selectbox(
            L["customer_type_label"],
            customer_type_options,
            index=default_idx,
            key="customer_type_sim_select_widget",
        )

        # --- 첨부 파일 업로더 추가 ---
        customer_attachment_widget = st.file_uploader(
            L["attachment_label"],
            type=["png", "jpg", "jpeg", "pdf"],
            key="customer_attachment_file_uploader",
            help=L["attachment_placeholder"],
            accept_multiple_files=False  # 채팅/이메일은 단일 파일만 허용
        )

        # 파일 정보 저장 및 LLM 컨텍스트 생성
        if customer_attachment_widget:
            st.session_state.customer_attachment_file = customer_attachment_widget
            st.session_state.sim_attachment_context_for_llm = L["attachment_status_llm"].format(
                filename=customer_attachment_widget.name, filetype=customer_attachment_widget.type
            )
        else:
            st.session_state.customer_attachment_file = None
            st.session_state.sim_attachment_context_for_llm = ""
        # --------------------------

        if st.button(L["button_simulate"], key=f"btn_simulate_initial_{st.session_state.sim_instance_id}"):  # 고유 키 사용
            if not customer_query.strip():
                st.warning(L["simulation_warning_query"])
                # st.stop()

            # --- 필수 입력 필드 검증 (요청 3 반영: 검증 로직 추가) ---
            if not st.session_state.customer_email.strip() or not st.session_state.customer_phone.strip():
                st.error(L["error_mandatory_contact"])
                # st.stop()
            # ------------------------------------------

            # 초기 상태 리셋
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.is_chat_ended = False
            st.session_state.initial_advice_provided = False
            st.session_state.is_solution_provided = False  # 솔루션 플래그 리셋
            st.session_state.language_transfer_requested = False  # 언어 요청 플래그 리셋
            st.session_state.transfer_summary_text = ""  # 이관 요약 리셋
            st.session_state.start_time = None  # AHT 타이머 초기화 (첫 고객 반응 후 시작)
            st.session_state.sim_instance_id = str(uuid.uuid4())  # 새 시뮬레이션 ID 할당
            # 전화 발신 관련 상태 초기화
            st.session_state.sim_call_outbound_summary = ""
            st.session_state.sim_call_outbound_target = None

            # 1) 고객 첫 메시지 추가
            st.session_state.simulator_messages.append(
                {"role": "customer", "content": customer_query}
            )

            # 2) Supervisor 가이드 + 초안 생성
            # 입력 텍스트의 언어를 자동 감지 (오류 발생 시 안전하게 처리)
            try:
                detected_lang = detect_text_language(customer_query)
                # 감지된 언어가 유효한지 확인
                if detected_lang not in ["ko", "en", "ja"]:
                    detected_lang = current_lang
                else:
                    # 언어가 감지되었고 현재 언어와 다르면 자동으로 언어 설정 업데이트
                    if detected_lang != current_lang:
                        st.session_state.language = detected_lang
                        st.info(f"🌐 입력 언어가 감지되어 언어 설정이 '{detected_lang}'로 자동 변경되었습니다.")
            except Exception as e:
                print(f"Language detection failed: {e}")
                detected_lang = current_lang  # 기본값으로 폴백
            
            # 고객 프로필 분석 (시각화를 위해 먼저 수행, 감지된 언어 사용)
            customer_profile = analyze_customer_profile(customer_query, detected_lang)
            similar_cases = find_similar_cases(customer_query, customer_profile, detected_lang, limit=5)

            # 시각화 차트 표시
            st.markdown("---")
            st.subheader("📊 고객 프로필 분석")

            # 고객 프로필 점수 차트 (감지된 언어 사용)
            profile_chart = visualize_customer_profile_scores(customer_profile, detected_lang)
            if profile_chart:
                st.plotly_chart(profile_chart, use_container_width=True)
            else:
                # Plotly가 없을 경우 텍스트로 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    gender_display = customer_profile.get("gender", "unknown")
                    if gender_display == "male":
                        gender_display = "남자"
                    elif gender_display == "female":
                        gender_display = "여자"
                    else:
                        gender_display = "알 수 없음"
                    st.metric(
                        "성별",
                        gender_display
                    )
                with col2:
                    st.metric(
                        L.get("sentiment_score_label", "감정 점수"),
                        f"{customer_profile.get('sentiment_score', 50)}/100"
                    )
                with col3:
                    urgency_map = {"low": 25, "medium": 50, "high": 75}
                    urgency_score = urgency_map.get(customer_profile.get("urgency_level", "medium").lower(), 50)
                    st.metric(
                        L.get("urgency_score_label", "긴급도"),
                        f"{urgency_score}/100"
                    )
                with col4:
                    st.metric(
                        L.get("customer_type_label", "고객 유형"),
                        customer_profile.get("predicted_customer_type", "normal")
                    )

            # 유사 케이스 시각화
            if similar_cases:
                st.markdown("---")
                st.subheader("🔍 유사 케이스 추천")
                similarity_chart = visualize_similarity_cases(similar_cases, detected_lang)
                if similarity_chart:
                    st.plotly_chart(similarity_chart, use_container_width=True)

                # 유사 케이스 요약 표시
                with st.expander(f"💡 {len(similar_cases)}개 유사 케이스 상세 정보"):
                    for idx, similar_case in enumerate(similar_cases, 1):
                        case = similar_case["case"]
                        summary = similar_case["summary"]
                        similarity = similar_case["similarity_score"]
                        st.markdown(f"### 케이스 {idx} (유사도: {similarity:.1f}%)")
                        st.markdown(f"**문의 내용:** {summary.get('main_inquiry', 'N/A')}")
                        st.markdown(f"**감정 점수:** {summary.get('customer_sentiment_score', 50)}/100")
                        st.markdown(f"**만족도 점수:** {summary.get('customer_satisfaction_score', 50)}/100")
                        if summary.get("key_responses"):
                            st.markdown("**핵심 응답:**")
                            for response in summary.get("key_responses", [])[:3]:
                                st.markdown(f"- {response[:100]}...")
                        st.markdown("---")

            # 초기 조언 생성 (감지된 언어 사용)
            text = _generate_initial_advice(
                customer_query,
                st.session_state.customer_type_sim_select,
                st.session_state.customer_email,
                st.session_state.customer_phone,
                detected_lang,  # 감지된 언어 사용
                st.session_state.customer_attachment_file
            )
            st.session_state.simulator_messages.append({"role": "supervisor", "content": text})

            st.session_state.initial_advice_provided = True
            save_simulation_history_local(
                customer_query,
                st.session_state.customer_type_sim_select,
                st.session_state.simulator_messages,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
                is_chat_ended=False,
            )
            st.session_state.sim_stage = "AGENT_TURN"

    # =========================
    # 4. 대화 로그 표시 (공통)
    # =========================
    
    # 피드백 저장 콜백 함수
    def save_feedback(index):
        """에이전트 응답에 대한 고객 피드백을 저장"""
        feedback_key = f"feedback_{st.session_state.sim_instance_id}_{index}"
        if feedback_key in st.session_state:
            feedback_value = st.session_state[feedback_key]
            # 메시지에 피드백 정보 저장
            if index < len(st.session_state.simulator_messages):
                st.session_state.simulator_messages[index]["feedback"] = feedback_value
    
    for idx, msg in enumerate(st.session_state.simulator_messages):
        role = msg["role"]
        content = msg["content"]
        avatar = {"customer": "🙋", "supervisor": "🤖", "agent_response": "🧑‍💻", "customer_rebuttal": "✨",
                  "system_end": "📌", "system_transfer": "📌"}.get(role, "💬")
        tts_role = "customer" if role.startswith("customer") or role == "customer_rebuttal" else (
            "agent" if role == "agent_response" else "supervisor")

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            # 인덱스를 render_tts_button에 전달하여 고유 키 생성에 사용
            render_tts_button(content, st.session_state.language, role=tts_role, prefix=f"{role}_", index=idx)
            
            # ⭐ 에이전트 응답에 대한 피드백 위젯 추가
            if role == "agent_response":
                feedback_key = f"feedback_{st.session_state.sim_instance_id}_{idx}"
                # 기존 피드백 값 가져오기
                existing_feedback = msg.get("feedback", None)
                if existing_feedback is not None:
                    st.session_state[feedback_key] = existing_feedback
                
                # 피드백 위젯 표시
                st.feedback(
                    "thumbs",
                    key=feedback_key,
                    disabled=existing_feedback is not None,
                    on_change=save_feedback,
                    args=[idx],
                )

            # ⭐ [새로운 로직] 고객 첨부 파일 렌더링 (첫 번째 메시지인 경우)
            if idx == 0 and role == "customer" and st.session_state.customer_attachment_b64:
                mime = st.session_state.customer_attachment_mime or "image/png"
                data_url = f"data:{mime};base64,{st.session_state.customer_attachment_b64}"

                # 이미지 파일만 표시 (PDF 등은 아직 처리하지 않음)
                if mime.startswith("image/"):
                    st.image(data_url, caption=f"첨부된 증거물 ({st.session_state.customer_attachment_file.name})",
                             use_column_width=True)
                elif mime == "application/pdf":
                    # PDF 파일일 경우, 파일 이름과 함께 다운로드 링크 또는 경고 표시
                    st.warning(
                        f"첨부된 PDF 파일 ({st.session_state.customer_attachment_file.name})은 현재 인라인 미리보기가 지원되지 않습니다.")

    # 이관 요약 표시 (이관 후에만) - 루프 밖으로 이동하여 한 번만 표시
    if st.session_state.transfer_summary_text or (st.session_state.language != st.session_state.language_at_transfer_start and st.session_state.language_at_transfer_start):
                st.markdown("---")
                st.markdown(f"**{L['transfer_summary_header']}**")
                st.info(L["transfer_summary_intro"])

                # 번역이 실패했을 경우 확인 (번역 성공 여부 플래그 사용)
                is_translation_failed = not st.session_state.get("translation_success", True) or not st.session_state.transfer_summary_text

                if is_translation_failed:
                    # 번역 실패 시에도 원본 텍스트가 표시되므로 오류 메시지 없이 원본 텍스트만 표시
                    # (오류 메시지를 표시하지 않아도 원본 텍스트로 계속 진행 가능)
                    if st.session_state.transfer_summary_text:
                        st.info(st.session_state.transfer_summary_text)
                    # 번역 재시도 버튼 추가 (선택적)
                    if st.button(L.get("button_retry_translation", "번역 다시 시도"),
                                 key=f"btn_retry_translation_{st.session_state.sim_instance_id}"):  # 고유 키 사용
                        # 재시도 로직 실행
                        with st.spinner(L.get("transfer_loading", "번역 중...")):
                            source_lang = st.session_state.language_at_transfer_start
                            target_lang = st.session_state.language

                            # 이전 대화 내용 재가공
                            history_text = get_chat_history_for_prompt(include_attachment=False)
                            for msg in st.session_state.simulator_messages:
                                role = "Customer" if msg["role"].startswith("customer") or msg[
                                    "role"] == "initial_query" else "Agent"
                                if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response",
                                                   "customer_closing_response"]:
                                    history_text += f"{role}: {msg['content']}\n"

                            # ⭐ 수정: 먼저 핵심 포인트만 요약한 후 번역
                            lang_name_source = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(source_lang, "Korean")
                            summary_prompt = f"""
You are an AI assistant that summarizes customer service conversations. 
Extract ONLY the key points from the conversation below. Keep it concise and focused on:
1. Customer's main inquiry/question
2. Key information provided by the agent
3. Important decisions or outcomes
4. Any unresolved issues

Write the summary in {lang_name_source}. Maximum 200 words. Be brief and to the point.

--- Conversation ---
{history_text}
---

Key Points Summary:
    # =========================
    # 5. 에이전트 입력 단계 (AGENT_TURN)
    # =========================
    if st.session_state.sim_stage == "AGENT_TURN":
        st.markdown(f"### {L['agent_response_header']}")

        # --- 실시간 응대 힌트 영역 ---
        hint_cols = st.columns([4, 1])
        with hint_cols[0]:
            st.info(L["hint_placeholder"] + st.session_state.realtime_hint_text)

        with hint_cols[1]:
            # 힌트 요청 버튼
            if st.button(L["button_request_hint"], key=f"btn_request_hint_{st.session_state.sim_instance_id}"):
                with st.spinner(L["response_generating"]):
                    # 채팅/이메일 탭이므로 is_call=False
                    hint = generate_realtime_hint(current_lang, is_call=False)
                    st.session_state.realtime_hint_text = hint

        # --- 언어 이관 요청 강조 표시 ---
        if st.session_state.language_transfer_requested:
            st.error("🚨 고객이 언어 전환(이관)을 요청했습니다. 즉시 응대하거나 이관을 진행하세요。")

        # --- 고객 첨부 파일 정보 재표시 ---
        if st.session_state.sim_attachment_context_for_llm:
            st.info(
                f"📎 최초 문의 시 첨부된 파일 정보:\n\n{st.session_state.sim_attachment_context_for_llm.replace('[ATTACHMENT STATUS]', '').strip()}")

        # --- AI 응답 초안 생성 버튼 (요청 1 반영) ---
        if st.button(L["button_generate_draft"], key=f"btn_generate_ai_draft_{st.session_state.sim_instance_id}"):
            if not st.session_state.is_llm_ready:
                st.warning(L["simulation_no_key_warning"])
            else:
                with st.spinner(L["draft_generating"]):
                    # 초안 생성 함수 호출
                    ai_draft = generate_agent_response_draft(current_lang)
                    if ai_draft and not ai_draft.startswith("❌"):
                        st.session_state.agent_response_area_text = ai_draft
                        st.success(L["draft_success"])
                    else:
                        st.error(ai_draft if ai_draft else L.get("draft_error", "응답 초안 생성에 실패했습니다."))

        # --- 전화 발신 버튼 추가 (요청 2 반영) ---
        st.markdown("---")
        st.subheader(L["button_call_outbound"])
        call_cols = st.columns(2)

        with call_cols[0]:
            if st.button(L["button_call_outbound_to_provider"], key="btn_call_outbound_partner", use_container_width=True):
                # 전화 발신 시뮬레이션: 현지 업체
                st.session_state.sim_call_outbound_target = "현지 업체/파트너"
                st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"

        with call_cols[1]:
            if st.button(L["button_call_outbound_to_customer"], key="btn_call_outbound_customer", use_container_width=True):
                # 전화 발신 시뮬레이션: 고객
                st.session_state.sim_call_outbound_target = "고객"
                st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"

        st.markdown("---")
        # --- 전화 발신 버튼 추가 끝 ---

        st.markdown("### 🚨 Supervisor 정책/지시 사항 업로드 (예외 처리 방침)")

        # --- Supervisor 정책 업로더 추가 ---
        supervisor_attachment_widget = st.file_uploader(
            "Supervisor 지시 사항/스크린샷 업로드 (예외 정책 포함)",
            type=["png", "jpg", "jpeg", "pdf", "txt"],
            key="supervisor_policy_uploader",
            help="비행기 지연, 질병 등 예외적 상황에 대한 Supervisor의 최신 지시 사항을 업로드하세요。",
            accept_multiple_files=False
        )

        # 파일 정보 저장 및 LLM 컨텍스트 생성
        if supervisor_attachment_widget:
            # 텍스트 파일 또는 PDF/이미지 파일의 텍스트 컨텐츠를 추출하여 policy_context에 저장해야 함
            # 여기서는 파일 이름과 타입만 컨텍스트로 전달하고, LLM이 이것이 '예외 정책'임을 알도록 유도
            file_name = supervisor_attachment_widget.name
            st.session_state.supervisor_policy_context = f"[Supervisor Policy Attached] Filename: {file_name}, Filetype: {supervisor_attachment_widget.type}. This file contains a CRITICAL, temporary policy update regarding exceptions (e.g., flight delays, illness, natural disasters). Analyze and prioritize this policy in the response."
            st.success(f"✅ Supervisor 정책 파일: **{file_name}**이(가) 응대 가이드에 반영됩니다.")
        elif st.session_state.supervisor_policy_context:
            st.info("⭐ 현재 적용 중인 Supervisor 정책이 있습니다.")
        else:
            st.session_state.supervisor_policy_context = ""

        # --- 에이전트 첨부 파일 업로더 (다중 파일 허용) ---
        agent_attachment_files = st.file_uploader(
            L["agent_attachment_label"],
            type=["png", "jpg", "jpeg", "pdf"],
            key="agent_attachment_file_uploader",
            help=L["agent_attachment_placeholder"],
            accept_multiple_files=True
        )

        if agent_attachment_files:
            st.session_state.agent_attachment_file = [
                {"name": f.name, "type": f.type, "size": f.size} for f in agent_attachment_files
            ]
            file_names = ", ".join([f["name"] for f in
                                    st.session_state.agent_attachment_file])  # 수정: file_infos 대신 st.session_state.agent_attachment_file 사용
            st.info(f"✅ {len(agent_attachment_files)}개 에이전트 첨부 파일 준비 완료: {file_names}")
        else:
            st.session_state.agent_attachment_file = []

        # --- 입력 필드 및 버튼 ---
        col_mic, col_text = st.columns([1, 2])

        # --- 마이크 녹음 ---
        with col_mic:
            mic_audio = mic_recorder(
                start_prompt=L["button_mic_input"],
                stop_prompt=L["button_mic_stop"],
                just_once=False,
                format="wav",
                use_container_width=True,
                key="sim_mic_recorder",
            )

        if mic_audio and mic_audio.get("bytes"):
            st.session_state.sim_audio_bytes = mic_audio["bytes"]
            # 언어 키 안전하게 가져오기
            current_lang = st.session_state.get("language", "ko")
            if current_lang not in ["ko", "en", "ja"]:
                current_lang = "ko"
            L = LANG.get(current_lang, LANG["ko"])
            st.info(L["recording_complete_press_transcribe"])

        if st.session_state.sim_audio_bytes:
            col_audio, col_transcribe, col_del = st.columns([3, 1, 1])

            # 1. 오디오 플레이어
            # Streamlit 문서: bytes 데이터를 직접 전달 가능
            with col_audio:
                try:
                    st.audio(st.session_state.sim_audio_bytes, format="audio/wav", autoplay=False)
                except Exception as e:
                    st.error(f"오디오 재생 오류: {e}")

            # 2. 녹음 삭제 버튼 (추가 요청 반영)
            with col_del:
                st.markdown("<br>", unsafe_allow_html=True)  # 버튼 수직 정렬
                if st.button(L["delete_mic_record"], key="btn_delete_sim_audio_call"):
                    # 오디오 및 관련 상태 초기화
                    st.session_state.sim_audio_bytes = None
                    st.session_state.last_transcript = ""
                    # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로 플래그 사용
                    st.session_state.reset_agent_response_area = True
                    st.success("녹음이 삭제되었습니다. 다시 녹음해 주세요.")

            # 3. 전사(Whisper) 버튼 (기존 로직 대체)
            col_tr, _ = st.columns([1, 2])
            if col_tr.button(L["transcribe_btn"], key="sim_transcribe_btn"):
                if st.session_state.sim_audio_bytes is None:
                    st.warning("먼저 마이크로 녹음을 완료하세요.")
                else:
                    # ⭐ 수정: OpenAI 또는 Gemini API 키 체크
                    has_openai = st.session_state.openai_client is not None
                    has_gemini = bool(get_api_key("gemini"))
                    
                    if not has_openai and not has_gemini:
                        st.error(L["whisper_client_error"] + " (OpenAI 또는 Gemini API Key 필요)")
                    else:
                        with st.spinner(L["whisper_processing"]):
                            # transcribe_bytes_with_whisper 함수를 사용하도록 수정
                            # 자동 언어 감지 사용 (입력 언어와 관계없이 정확한 전사)
                            transcribed_text = transcribe_bytes_with_whisper(
                                st.session_state.sim_audio_bytes,
                                "audio/wav",
                                lang_code=None,
                                auto_detect=True,
                            )
                            if transcribed_text.startswith("❌"):
                                st.error(transcribed_text)
                                st.session_state.last_transcript = ""
                            else:
                                st.session_state.last_transcript = transcribed_text.strip()
                                # ⭐ 수정: 전사된 텍스트를 입력창의 세션 상태 변수에 반영
                                st.session_state.agent_response_area_text = transcribed_text.strip()
                                st.session_state.agent_response_input_box_widget = transcribed_text.strip()

                                snippet = transcribed_text[:50].replace("\n", " ")
                                if len(transcribed_text) > 50:
                                    snippet += "..."
                                st.success(L["whisper_success"] + f"\n\n**인식 내용:** *{snippet}*")

        col_text, col_button = st.columns([4, 1])

        # --- 입력 필드 및 버튼 ---
        with col_text:
            # ⭐ 수정: 위젯 생성 전에 초기화 플래그를 확인하여 값을 초기화합니다.
            if st.session_state.get("reset_agent_response_area", False):
                st.session_state.agent_response_area_text = ""
                st.session_state.reset_agent_response_area = False
            
            # st.text_area의 값을 읽어 세션 상태를 직접 업데이트하는 on_change를 제거하고
            # st.text_area 위젯 자체의 키를 사용하여 send_clicked 시 최신 값을 읽도록 합니다.
            # (Streamlit 기본 동작: 버튼 클릭 시 위젯의 최종 값이 세션 상태에 반영됨)
            # ⭐ 수정: key를 agent_response_area_text로 통일하여 세션 상태와 동기화
            agent_response_input = st.text_area(
                L["agent_response_placeholder"],
                value=st.session_state.agent_response_area_text,
                key="agent_response_area_text",  # 세션 상태 키와 동일하게 설정하여 동기화 보장
                height=150,
            )

            # 솔루션 제공 체크박스
            st.session_state.is_solution_provided = st.checkbox(
                L["solution_check_label"],
                value=st.session_state.is_solution_provided,
                key="solution_checkbox_widget",
            )

        with col_button:
            send_clicked = st.button(L["send_response_button"], key="send_agent_response_btn")

        if send_clicked:
            # ⭐ 수정: st.session_state.agent_response_area_text에서 최신 입력값을 가져옴 (key와 동일)
            agent_response = st.session_state.agent_response_area_text.strip()

            if not agent_response:
                st.warning(L["empty_response_warning"])
                # st.stop()

            # AHT 타이머 시작
            if st.session_state.start_time is None and len(st.session_state.simulator_messages) >= 1:
                st.session_state.start_time = datetime.now()

            # --- 에이전트 첨부 파일 처리 (다중 파일 처리) ---
            final_response_content = agent_response
            if st.session_state.agent_attachment_file:
                file_infos = st.session_state.agent_attachment_file
                file_names = ", ".join([f["name"] for f in file_infos])
                attachment_msg = L["agent_attachment_status"].format(
                    filename=file_names, filetype=f"총 {len(file_infos)}개 파일"
                )
                final_response_content = f"{agent_response}\n\n---\n{attachment_msg}"

            # 로그 업데이트
            st.session_state.simulator_messages.append(
                {"role": "agent_response", "content": final_response_content}
            )

            # ⭐ 추가: 에이전트 응답에 메일 끝인사가 포함되어 있는지 확인
            email_closing_patterns = [
                "추가 문의사항이 있으면 언제든지 연락", "추가 문의 사항이 있으면 언제든지 연락",
                "추가 문의사항이 있으시면", "추가 문의 사항이 있으시면",
                "언제든지 연락", "언제든지 연락 주세요",
                "additional inquiries", "any additional questions", "any further questions",
                "feel free to contact", "please feel free to contact",
                "please don't hesitate to contact", "don't hesitate to contact",
                "please let me know", "let me know", "let me know if",
                "please let me know so", "let me know so",
                "if you have any questions", "if you have any further questions",
                "if you need any assistance", "if you need further assistance",
                "if you encounter any issues", "if you still have", "if you remain unclear",
                "I can assist further", "I can help further", "I can assist",
                "so I can assist", "so I can help", "so I can assist further",
                "追加のご質問", "追加のお問い合わせ", "ご質問がございましたら", "お問い合わせがございましたら"
            ]
            is_email_closing_in_response = any(pattern.lower() in final_response_content.lower() for pattern in email_closing_patterns)
            if is_email_closing_in_response:
                st.session_state.has_email_closing = True  # 플래그 설정

            # 입력창/오디오/첨부 파일 초기화
            # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로,
            # rerun 후 위젯이 다시 생성될 때 초기값이 적용되도록 플래그를 사용합니다.
            st.session_state.sim_audio_bytes = None
            st.session_state.agent_attachment_file = []  # 첨부 파일 초기화
            st.session_state.language_transfer_requested = False
            st.session_state.realtime_hint_text = ""  # 힌트 초기화
            st.session_state.sim_call_outbound_summary = ""  # 전화 발신 요약 초기화

            # ⭐ 수정: agent_response_area_text는 rerun 후 위젯이 다시 생성될 때 초기화되도록
            # 플래그를 설정합니다. 위젯 생성 전에 이 플래그를 확인하여 값을 초기화합니다.
            st.session_state.reset_agent_response_area = True
            
            # ⭐ 수정: 응답 전송 시 바로 고객 반응 자동 생성
            if st.session_state.is_llm_ready:
                # LLM이 준비된 경우 바로 고객 반응 생성
                with st.spinner(L["generating_customer_response"]):
                    customer_response = generate_customer_reaction(st.session_state.language, is_call=False)
                
                # 고객 반응을 메시지에 추가
                st.session_state.simulator_messages.append(
                    {"role": "customer", "content": customer_response}
                )
                
                # ⭐ 추가: 메일 끝인사가 포함된 경우 고객 응답 확인 및 설문 조사 버튼 활성화
                if st.session_state.get("has_email_closing", False):
                    # 고객의 긍정 반응 확인
                    positive_keywords = [
                        "No, that will be all", "no more", "없습니다", "감사합니다", "Thank you", "ありがとう",
                        "추가 문의 사항 없습니다", "추가 문의사항 없습니다", "no additional", "追加の質問はありません",
                        "알겠습니다", "알겠어요", "ok", "okay", "네", "yes", "좋습니다", "good", "fine", "괜찮습니다"
                    ]
                    is_positive = any(keyword.lower() in customer_response.lower() for keyword in positive_keywords)
                    
                    if is_positive or L.get('customer_no_more_inquiries', '') in customer_response:
                        # 설문 조사 버튼 활성화를 위해 WAIT_CUSTOMER_CLOSING_RESPONSE 단계로 이동
                        st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                        st.rerun()
            else:
                # LLM이 없는 경우 플래그 설정하여 CUSTOMER_TURN 단계에서 수동 생성 가능하도록
                st.session_state.need_customer_response = True
            
            # ⭐ 수정: 고객 반응 생성 후 CUSTOMER_TURN 단계로 이동하고 UI 업데이트
            st.session_state.sim_stage = "CUSTOMER_TURN"
            st.rerun()
            

        # --- 언어 이관 버튼 ---
        st.markdown("---")
        st.markdown(f"**{L['transfer_header']}**")
        transfer_cols = st.columns(len(LANG) - 1)

        languages = list(LANG.keys())
        languages.remove(current_lang)


        def transfer_session(target_lang: str, current_messages: List[Dict[str, str]]):
            """언어 이관 시스템 메시지를 추가하고 세션 언어를 변경합니다."""

            # API 키 체크는 run_llm 내부에서 처리되지만, 명시적으로 Gemini 키를 요구함
            if not get_api_key("gemini"):
                st.error(LANG[current_lang]["simulation_no_key_warning"].replace('API Key', 'Gemini API Key'))
                # st.stop()
                return

            current_lang_at_start = st.session_state.language  # Source language

            # AHT 타이머 중지
            st.session_state.start_time = None

            # 1. 로딩 시작 (시간 양해 메시지 시뮬레이션)
            with st.spinner(L["transfer_loading"]):
                # 실제 대기 시간 5~10초 (3~10분 시뮬레이션)
                time.sleep(np.random.uniform(5, 10))

                # 2. 대화 기록을 번역할 텍스트로 가공
                history_text = ""
                for msg in current_messages:
                    role = "Customer" if msg["role"].startswith("customer") or msg[
                        "role"] == "initial_query" else "Agent"
                    if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response",
                                       "customer_closing_response"]:
                        history_text += f"{role}: {msg['content']}\n"

                # ⭐ 수정: 먼저 핵심 포인트만 요약한 후 번역
                # 요약 프롬프트 생성
                lang_name_source = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(current_lang_at_start, "Korean")
                summary_prompt = f"""
You are an AI assistant that summarizes customer service conversations. 
Extract ONLY the key points from the conversation below. Keep it concise and focused on:
1. Customer's main inquiry/question
2. Key information provided by the agent
3. Important decisions or outcomes
4. Any unresolved issues

Write the summary in {lang_name_source}. Maximum 200 words. Be brief and to the point.

--- Conversation ---
{history_text}
---

Key Points Summary:
    # =========================
    # 6. 고객 반응 생성 단계 (CUSTOMER_TURN)
    # =========================
    elif st.session_state.sim_stage == "CUSTOMER_TURN":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
        st.info(L["customer_turn_info"])

        # 1. 고객 반응 생성
        # 이미 고객 반응이 생성되어 있는지 확인
        last_customer_message = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "customer" and msg.get("content"):
                last_customer_message = msg.get("content", "")
                break
        
        if last_customer_message is None:
            # 고객 반응이 없는 경우에만 생성
            with st.spinner(L["generating_customer_response"]):
                customer_response = generate_customer_reaction(st.session_state.language, is_call=False)

            # 2. 대화 로그 업데이트
            st.session_state.simulator_messages.append(
                {"role": "customer", "content": customer_response}
            )
            
            # 3. 생성 직후 바로 다음 단계 결정
            positive_closing_phrases = [L["customer_positive_response"], L["customer_no_more_inquiries"]]
            is_positive_closing = any(phrase in customer_response for phrase in positive_closing_phrases)
            
            # 다음 단계 결정
            if L["customer_positive_response"] in customer_response:
                if st.session_state.is_solution_provided:
                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                else:
                    st.session_state.sim_stage = "AGENT_TURN"
            elif is_positive_closing:
                if L['customer_no_more_inquiries'] in customer_response:
                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                else:
                    if st.session_state.is_solution_provided:
                        st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                    else:
                        st.session_state.sim_stage = "AGENT_TURN"
            elif customer_response.startswith(L["customer_escalation_start"]):
                st.session_state.sim_stage = "ESCALATION_REQUIRED"
            else:
                # 고객이 추가 질문하거나 정보 제공한 경우 -> 에이전트 턴으로 이동
                st.session_state.sim_stage = "AGENT_TURN"
            
            # UI 업데이트를 위해 rerun
            st.rerun()
        else:
            customer_response = last_customer_message

        # 3. 종료 조건 검토 (이미 고객 반응이 있는 경우)
        positive_closing_phrases = [L["customer_positive_response"], L["customer_no_more_inquiries"]]
        is_positive_closing = any(phrase in customer_response for phrase in positive_closing_phrases)

        # ⭐ 추가: 메일 응대 종료 문구 확인 (플래그 또는 에이전트의 마지막 응답 확인)
        # 먼저 플래그 확인 (에이전트 응답 전송 시 설정됨)
        is_email_closing = st.session_state.get("has_email_closing", False)
        
        # 플래그가 없으면 에이전트의 마지막 응답에서 직접 확인
        if not is_email_closing:
            last_agent_response = None
            for msg in reversed(st.session_state.simulator_messages):
                if msg.get("role") == "agent_response" and msg.get("content"):
                    last_agent_response = msg.get("content", "")
                    break
            
            # 메일 끝인사 문구 패턴 (다국어 지원) - 더 포괄적인 패턴 추가
            email_closing_patterns = [
                "추가 문의사항이 있으면 언제든지 연락", "추가 문의 사항이 있으면 언제든지 연락",
                "추가 문의사항이 있으시면", "추가 문의 사항이 있으시면",
                "언제든지 연락", "언제든지 연락 주세요",
                "additional inquiries", "any additional questions", "any further questions",
                "feel free to contact", "please feel free to contact",
                "please don't hesitate to contact", "don't hesitate to contact",
                "please let me know", "let me know", "let me know if",
                "please let me know so", "let me know so",
                "if you have any questions", "if you have any further questions",
                "if you need any assistance", "if you need further assistance",
                "if you encounter any issues", "if you still have", "if you remain unclear",
                "I can assist further", "I can help further", "I can assist",
                "so I can assist", "so I can help", "so I can assist further",
                "追加のご質問", "追加のお問い合わせ", "ご質問がございましたら", "お問い合わせがございましたら"
            ]
            
            if last_agent_response:
                is_email_closing = any(pattern.lower() in last_agent_response.lower() for pattern in email_closing_patterns)
                if is_email_closing:
                    st.session_state.has_email_closing = True  # 플래그 업데이트

        # ⭐ 수정: 메일 끝인사가 포함된 경우, 고객의 긍정 반응이나 "추가 문의 사항 없습니다" 답변을 인식하면 설문 조사 링크 전송 버튼 자동 활성화
        if is_email_closing:
            # 고객의 긍정 반응 또는 "추가 문의 사항 없습니다" 답변 확인
            no_more_keywords = [
                L['customer_no_more_inquiries'],
                "No, that will be all",
                "no more",
                "없습니다",
                "감사합니다",
                "Thank you",
                "ありがとう",
                "追加 문의 사항 없습니다",
                "추가 문의사항 없습니다",
                "no additional",
                "追加の質問はありません",
                "알겠습니다",
                "알겠어요",
                "ok",
                "okay",
                "네",
                "yes"
            ]
            has_no_more_inquiry = any(keyword.lower() in customer_response.lower() for keyword in no_more_keywords)
            
            # 긍정 반응 키워드 추가 (더 포괄적인 인식)
            positive_keywords = [
                "알겠습니다", "알겠어요", "네", "yes", "ok", "okay", "감사합니다", "thank you", "ありがとう",
                "좋습니다", "good", "fine", "괜찮습니다", "알겠습니다 감사합니다"
            ]
            is_positive_response = any(keyword.lower() in customer_response.lower() for keyword in positive_keywords)
            
            # 긍정 반응이 있거나 "추가 문의 사항 없습니다" 답변이 있으면 설문 조사 링크 전송 버튼 활성화
            if is_positive_closing or has_no_more_inquiry or L['customer_no_more_inquiries'] in customer_response or is_positive_response:
                # 에이전트 감사 인사가 아직 추가되지 않은 경우에만 추가
                agent_closing_added = False
                for msg in reversed(st.session_state.simulator_messages):
                    if msg.get("role") == "agent_response":
                        agent_msg_content = msg.get("content", "")
                        if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                            agent_closing_added = True
                        break
                
                if not agent_closing_added:
                    # 에이전트가 감사 인사 메시지 전송
                    agent_name = st.session_state.get("agent_name", "000")
                    if current_lang == "ko":
                        agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                    elif current_lang == "en":
                        agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                    else:  # ja
                        agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"
                    
                    # 에이전트 감사 인사를 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "agent_response", "content": agent_closing_msg}
                    )
                
                # 설문 조사 링크 전송 버튼 활성화를 위해 WAIT_CUSTOMER_CLOSING_RESPONSE 단계로 이동
                # (실제로는 고객 응답이 이미 있으므로 바로 설문 조사 버튼 표시)
                st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                st.rerun()
            else:
                # 메일 끝인사가 있지만 고객이 추가 질문을 한 경우
                st.session_state.sim_stage = "AGENT_TURN"
                st.rerun()
        # ⭐ 수정: 고객이 "알겠습니다. 감사합니다"라고 답변했을 때, 솔루션이 제공된 경우에만 추가 문의 여부 확인 단계로 이동
        # 정확한 문자열 비교가 아닌 포함 여부로 확인 (LLM 응답이 약간 다를 수 있음)
        elif L["customer_positive_response"] in customer_response:
            # 솔루션이 제공된 경우에만 추가 문의 여부 확인 단계로 이동
            if st.session_state.is_solution_provided:
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                # 솔루션이 제공되지 않은 경우 에이전트 턴으로 유지
                st.session_state.sim_stage = "AGENT_TURN"
        elif is_positive_closing:
            # 긍정 종료 응답 처리
            if L['customer_no_more_inquiries'] in customer_response:
                # ⭐ 수정: "없습니다. 감사합니다" 답변 시 에이전트가 감사 인사를 한 후 종료하도록 변경
                # 바로 종료하지 않고 WAIT_CLOSING_CONFIRMATION_FROM_AGENT 단계로 이동하여 에이전트가 감사 인사 후 종료
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                # "알겠습니다. 감사합니다"와 유사한 긍정 응답인 경우, 솔루션 제공 여부 확인
                if st.session_state.is_solution_provided:
                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                else:
                    st.session_state.sim_stage = "AGENT_TURN"


        # ⭐ 수정: 고객이 아직 솔루션에 만족하지 않거나 추가 질문을 한 경우 (일반적인 턴)
        elif customer_response.startswith(L["customer_escalation_start"]):
            st.session_state.sim_stage = "ESCALATION_REQUIRED"  # 에스컬레이션 필요
        else:
            # 에이전트 턴으로 유지 (고객이 추가 질문하거나 정보 제공)
            st.session_state.sim_stage = "AGENT_TURN"

        st.session_state.is_solution_provided = False  # 종료 단계 진입 후 플래그 리셋

        # 이력 저장 (종료되지 않은 경우에만 저장)
        # ⭐ 수정: "없습니다. 감사합니다" 답변 시에는 이미 이력 저장을 했으므로 중복 저장 방지
        if st.session_state.sim_stage != "CLOSING":
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

        st.session_state.realtime_hint_text = ""  # 힌트 초기화
        
        # 다음 단계로 이동했으므로 UI 업데이트를 위해 rerun
        st.rerun()


    # =========================
    # 7. 종료 확인 메시지 대기 (WAIT_CLOSING_CONFIRMATION_FROM_AGENT)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CLOSING_CONFIRMATION_FROM_AGENT":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        st.success(L["customer_positive_solution_reaction"])

        col_chat_end, col_email_end = st.columns(2)  # 버튼을 나란히 배치

        # [1] 채팅 - 추가 문의 확인 메시지 보내기 버튼
        with col_chat_end:
            # [수정 1] 다국어 레이블 사용
            if st.button(L["send_closing_confirm_button"],
                         key=f"btn_send_closing_confirm_{st.session_state.sim_instance_id}"):
                # ⭐ 수정: 에이전트가 감사 인사를 포함한 종료 메시지 전송
                # 언어별 감사 인사 메시지 생성
                agent_name = st.session_state.get("agent_name", "000")
                if current_lang == "ko":
                    closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. {L['customer_closing_confirm']} 즐거운 하루 되세요."
                elif current_lang == "en":
                    closing_msg = f"Thank you for contacting us. This was {agent_name}. {L['customer_closing_confirm']} Have a great day!"
                else:  # ja
                    closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。{L['customer_closing_confirm']} 良い一日をお過ごしください。"

                # 에이전트 응답으로 로그 기록
                st.session_state.simulator_messages.append(
                    {"role": "agent_response", "content": closing_msg}
                )

                # [추가] TTS 버튼 렌더링을 위해 sleep/rerun 강제
                time.sleep(0.1)
                st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                st.rerun()

        # [2] 이메일 - 상담 종료 버튼 (즉시 종료)
        with col_email_end:
            # [수정 1] 다국어 레이블 사용
            if st.button(L["button_email_end_chat"], key=f"btn_email_end_chat_{st.session_state.sim_instance_id}"):
                # AHT 타이머 정지
                st.session_state.start_time = None

                # [수정 1] 다국어 레이블 사용
                end_msg = L["prompt_survey"]
                st.session_state.simulator_messages.append(
                    {"role": "system_end", "content": "(시스템: 이메일 상담 종료) " + end_msg}
                )

                # [추가] TTS 버튼 렌더링을 위해 sleep/rerun 강제
                time.sleep(0.1)
                st.session_state.is_chat_ended = True
                st.session_state.sim_stage = "CLOSING"  # 바로 CLOSING으로 전환

    # =========================
    # 8. 고객 최종 응답 생성 및 처리 (WAIT_CUSTOMER_CLOSING_RESPONSE)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CUSTOMER_CLOSING_RESPONSE":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
        
        # ⭐ 추가: 메일 응대 종료 문구 확인 (에이전트의 마지막 응답에 "추가 문의사항이 있으면 언제든지 연락 주세요" 같은 문구가 포함되어 있는지 확인)
        last_agent_response = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "agent_response" and msg.get("content"):
                last_agent_response = msg.get("content", "")
                break
        
        # 메일 끝인사 문구 패턴 (다국어 지원) - 더 포괄적인 패턴 추가
        email_closing_patterns = [
            "추가 문의사항이 있으면 언제든지 연락",
            "추가 문의 사항이 있으면 언제든지 연락",
            "추가 문의사항이 있으시면 언제든지 연락",
            "추가 문의 사항이 있으시면 언제든지 연락",
            "추가 문의사항이 있으시면",
            "추가 문의 사항이 있으시면",
            "추가 문의사항이 있으면",
            "추가 문의 사항이 있으면",
            "언제든지 연락",
            "언제든지 연락 주세요",
            "언제든지 연락 주시기 바랍니다",
            "additional inquiries",
            "any additional questions",
            "any further questions",
            "feel free to contact",
            "please feel free to contact",
            "please don't hesitate to contact",
            "don't hesitate to contact",
            "追加のご質問",
            "追加のお問い合わせ",
            "ご質問がございましたら",
            "お問い合わせがございましたら"
        ]
        
        is_email_closing = False
        if last_agent_response:
            is_email_closing = any(pattern.lower() in last_agent_response.lower() for pattern in email_closing_patterns)
        
        # ⭐ 수정: 이미 고객 응답이 생성되어 있는지 확인
        last_customer_message = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "customer_rebuttal":
                last_customer_message = msg.get("content", "")
                break
            # ⭐ 추가: customer 역할의 메시지도 확인 (메일 끝인사가 포함된 경우 CUSTOMER_TURN에서 이미 고객 응답이 생성되었을 수 있음)
            elif msg.get("role") == "customer" and is_email_closing:
                last_customer_message = msg.get("content", "")
                break
        
        # 고객 응답이 아직 생성되지 않은 경우에만 생성
        if last_customer_message is None:
            # 고객 답변 자동 생성 (LLM Key 검증 포함)
            if not st.session_state.is_llm_ready:
                st.warning(L["llm_key_missing_customer_response"])
                if st.button(L["customer_generate_response_button"], key="btn_generate_final_response"):
                    st.session_state.sim_stage = "AGENT_TURN"
                    st.rerun()
                st.stop()
            
            # LLM이 준비된 경우 고객 응답 생성
            st.info(L["agent_confirmed_additional_inquiry"])
            with st.spinner(L["generating_customer_response"]):
                final_customer_reaction = generate_customer_closing_response(st.session_state.language)

            # 로그 기록
            st.session_state.simulator_messages.append(
                {"role": "customer_rebuttal", "content": final_customer_reaction}
            )
            last_customer_message = final_customer_reaction
        
        # 고객 응답에 따라 처리 (생성 직후 또는 이미 있는 경우 모두 처리)
        if last_customer_message is None:
            # 고객 응답이 없는 경우 (이미 생성했는데도 None인 경우는 에러)
            st.warning(L["customer_response_generation_failed"])
        else:
            final_customer_reaction = last_customer_message
            
            # (A) "없습니다. 감사합니다" 경로 -> 에이전트가 감사 인사 후 버튼 표시
            # 더 유연한 매칭을 위해 키워드 체크 추가
            no_more_keywords = [
                L['customer_no_more_inquiries'],
                "No, that will be all",
                "no more",
                "없습니다",
                "감사합니다",
                "結構です",
                "ありがとう",
                "추가 문의 사항 없습니다",
                "추가 문의사항 없습니다",
                "no additional",
                "追加の質問はありません"
            ]
            has_no_more_inquiry = any(keyword.lower() in final_customer_reaction.lower() for keyword in no_more_keywords)
            
            # ⭐ 추가: 메일 끝인사가 포함된 경우, 고객의 긍정 반응이나 "추가 문의 사항 없습니다" 답변을 인식하면 설문 조사 링크 전송 버튼 자동 활성화
            # 긍정 반응 키워드 추가
            positive_keywords = [
                "알겠습니다", "알겠어요", "네", "yes", "ok", "okay", "감사합니다", "thank you", "ありがとう"
            ]
            is_positive_response = any(keyword.lower() in final_customer_reaction.lower() for keyword in positive_keywords)
            
            if is_email_closing and (has_no_more_inquiry or L['customer_no_more_inquiries'] in final_customer_reaction or is_positive_response):
                # 에이전트 감사 인사가 아직 추가되지 않은 경우에만 추가
                agent_closing_added = False
                for msg in reversed(st.session_state.simulator_messages):
                    if msg.get("role") == "agent_response":
                        agent_msg_content = msg.get("content", "")
                        if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                            agent_closing_added = True
                        break
                
                if not agent_closing_added:
                    # 에이전트가 감사 인사 메시지 전송
                    agent_name = st.session_state.get("agent_name", "000")
                    if current_lang == "ko":
                        agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                    elif current_lang == "en":
                        agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                    else:  # ja
                        agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"
                    
                    # 에이전트 감사 인사를 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "agent_response", "content": agent_closing_msg}
                    )
                
                # 설문 조사 링크 전송 버튼 표시
                st.markdown("---")
                st.success(L["no_more_inquiries_confirmed"])
                st.markdown(f"### {L['consultation_end_header']}")
                st.info(L["click_survey_button_to_end"])
                st.markdown("---")
                
                # 버튼을 중앙에 크게 표시
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    end_chat_button = st.button(
                        L["sim_end_chat_button"], 
                        key="btn_final_end_chat_email_closing", 
                        use_container_width=True, 
                        type="primary"
                    )
                
                if end_chat_button:
                    # AHT 타이머 정지
                    st.session_state.start_time = None

                    # 설문 조사 링크 전송 메시지 추가
                    end_msg = L["prompt_survey"]
                    st.session_state.simulator_messages.append(
                        {"role": "system_end", "content": end_msg}
                    )

                    # 채팅 종료 처리
                    st.session_state.is_chat_ended = True
                    st.session_state.sim_stage = "CLOSING"
                    
                    # 이력 저장
                    save_simulation_history_local(
                        st.session_state.customer_query_text_area, customer_type_display,
                        st.session_state.simulator_messages, is_chat_ended=True,
                        attachment_context=st.session_state.sim_attachment_context_for_llm,
                    )
                    
                    st.session_state.realtime_hint_text = ""  # 힌트 초기화
                    st.rerun()  # 버튼 클릭 후 UI 업데이트
            # 메일 끝인사가 포함된 경우 여기서 처리 완료, 다른 로직은 실행하지 않음
            elif L['customer_no_more_inquiries'] in final_customer_reaction or has_no_more_inquiry:
                # ⭐ 수정: 에이전트 감사 인사가 아직 추가되지 않은 경우에만 추가
                agent_closing_added = False
                for msg in reversed(st.session_state.simulator_messages):
                    if msg.get("role") == "agent_response":
                        # 이미 에이전트 감사 인사가 있는지 확인
                        agent_msg_content = msg.get("content", "")
                        if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                            agent_closing_added = True
                        break
                
                if not agent_closing_added:
                    # ⭐ 추가: 에이전트가 감사 인사 메시지 전송
                    agent_name = st.session_state.get("agent_name", "000")
                    if current_lang == "ko":
                        agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                    elif current_lang == "en":
                        agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                    else:  # ja
                        agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"
                    
                    # 에이전트 감사 인사를 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "agent_response", "content": agent_closing_msg}
                    )
                
                # ⭐ 수정: 현재 단계에서 바로 버튼 표시 (FINAL_CLOSING_ACTION으로 이동하지 않음)
                st.markdown("---")
                st.success(L["no_more_inquiries_confirmed"])
                st.markdown(f"### {L['consultation_end_header']}")
                st.info(L["click_survey_button_to_end"])
                st.markdown("---")
                
                # 버튼을 중앙에 크게 표시
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    end_chat_button = st.button(
                        L["sim_end_chat_button"], 
                        key="btn_final_end_chat_in_wait", 
                        use_container_width=True, 
                        type="primary"
                    )
                
                if end_chat_button:
                    # AHT 타이머 정지
                    st.session_state.start_time = None

                    # 설문 조사 링크 전송 메시지 추가
                    end_msg = L["prompt_survey"]
                    st.session_state.simulator_messages.append(
                        {"role": "system_end", "content": end_msg}
                    )

                    # 채팅 종료 처리
                    st.session_state.is_chat_ended = True
                    st.session_state.sim_stage = "CLOSING"
                    
                    # 이력 저장
                    save_simulation_history_local(
                        st.session_state.customer_query_text_area, customer_type_display,
                        st.session_state.simulator_messages, is_chat_ended=True,
                        attachment_context=st.session_state.sim_attachment_context_for_llm,
                    )
                    
                    st.session_state.realtime_hint_text = ""  # 힌트 초기화
                    st.rerun()  # 버튼 클릭 후 UI 업데이트
            # (B) "추가 문의 사항도 있습니다" 경로 -> AGENT_TURN으로 복귀
            elif L['customer_has_additional_inquiries'] in final_customer_reaction:
                st.session_state.sim_stage = "AGENT_TURN"
                save_simulation_history_local(
                    st.session_state.customer_query_text_area, customer_type_display,
                    st.session_state.simulator_messages, is_chat_ended=False,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                )
                st.session_state.realtime_hint_text = ""
                st.rerun()  # AGENT_TURN으로 이동 후 UI 업데이트
            else:
                # 고객 응답이 생성되었지만 조건에 맞지 않는 경우에도 버튼 표시
                # (기본적으로 "없습니다. 감사합니다"로 간주)
                # ⭐ 수정: fallback 경로에서도 에이전트 감사 인사 메시지 추가
                agent_closing_added = False
                for msg in reversed(st.session_state.simulator_messages):
                    if msg.get("role") == "agent_response":
                        # 이미 에이전트 감사 인사가 있는지 확인
                        agent_msg_content = msg.get("content", "")
                        if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                            agent_closing_added = True
                        break
                
                if not agent_closing_added:
                    # ⭐ 추가: 에이전트가 감사 인사 메시지 전송
                    agent_name = st.session_state.get("agent_name", "000")
                    if current_lang == "ko":
                        agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                    elif current_lang == "en":
                        agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                    else:  # ja
                        agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"
                    
                    # 에이전트 감사 인사를 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "agent_response", "content": agent_closing_msg}
                    )
                
                st.markdown("---")
                st.success(L["no_more_inquiries_confirmed"])
                st.markdown(f"### {L['consultation_end_header']}")
                st.info(L["click_survey_button_to_end"])
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    end_chat_button = st.button(
                        L["sim_end_chat_button"], 
                        key="btn_final_end_chat_fallback", 
                        use_container_width=True, 
                        type="primary"
                    )
                
                if end_chat_button:
                    # AHT 타이머 정지
                    st.session_state.start_time = None
                    
                    # 설문 조사 링크 전송 메시지 추가
                    end_msg = L["prompt_survey"]
                    st.session_state.simulator_messages.append(
                        {"role": "system_end", "content": end_msg}
                    )
                    
                    # 채팅 종료 처리
                    st.session_state.is_chat_ended = True
                    st.session_state.sim_stage = "CLOSING"
                    
                    # 이력 저장
                    save_simulation_history_local(
                        st.session_state.customer_query_text_area, customer_type_display,
                        st.session_state.simulator_messages, is_chat_ended=True,
                        attachment_context=st.session_state.sim_attachment_context_for_llm,
                    )
                    
                    st.session_state.realtime_hint_text = ""  # 힌트 초기화
                    st.rerun()  # 버튼 클릭 후 UI 업데이트

    # =========================
    # 9. 최종 종료 행동 (FINAL_CLOSING_ACTION)
    # =========================
    elif st.session_state.sim_stage == "FINAL_CLOSING_ACTION":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        
        # ⭐ 수정: 명확한 안내 메시지와 함께 버튼 표시
        st.markdown("---")
        st.success(L["no_more_inquiries_confirmed"])
        st.markdown(f"### {L['consultation_end_header']}")
        st.info(L["click_survey_button_to_end"])
        st.markdown("---")
        
        # 버튼을 중앙에 크게 표시
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            end_chat_button = st.button(
                L["sim_end_chat_button"], 
                key="btn_final_end_chat", 
                use_container_width=True, 
                type="primary"
            )
        
        if end_chat_button:
            # AHT 타이머 정지
            st.session_state.start_time = None

            # 설문 조사 링크 전송 메시지 추가
            end_msg = L["prompt_survey"]
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": end_msg}
            )

            # 채팅 종료 처리
            st.session_state.is_chat_ended = True
            st.session_state.sim_stage = "CLOSING"
            
            # 이력 저장
            customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )
            
            st.session_state.realtime_hint_text = ""  # 힌트 초기화

# ========================================
# 전화 시뮬레이터 로직
# ========================================

elif feature_selection == L["sim_tab_phone"]:
    st.header(L["phone_header"])
    st.markdown(L["simulator_desc"])

    current_lang = st.session_state.language
    L = LANG[current_lang]



    # ========================================
    # AHT 타이머 (IN_CALL 상태에서만 동작)
    # ========================================
    if st.session_state.call_sim_stage == "IN_CALL":
        # AHT 타이머 계산 로직
        col_timer, col_duration = st.columns([1, 4])

        if st.session_state.start_time is not None:
            now = datetime.now()

            # Hold 중이라면, Hold 상태가 된 이후의 시간을 현재 total_hold_duration에 더하지 않음 (Resume 시 정산)
            if st.session_state.is_on_hold and st.session_state.hold_start_time:
                # Hold 중이지만 AHT 타이머는 계속 흘러가야 하므로, Hold 시간은 제외하지 않고 최종 AHT 계산에만 사용
                elapsed_time_total = now - st.session_state.start_time
            else:
                elapsed_time_total = now - st.session_state.start_time

            # ⭐ AHT는 통화 시작부터 현재까지의 총 경과 시간입니다.
            total_seconds = elapsed_time_total.total_seconds()
            total_seconds = max(0, total_seconds)  # 음수 방지

            # 시간 형식 포맷팅
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            # 경고 기준
            if total_seconds > 900:  # 15분
                delta_str = L["timer_info_risk"]
                delta_color = "inverse"
            elif total_seconds > 600:  # 10분
                delta_str = L["timer_info_warn"]
                delta_color = "off"
            else:
                delta_str = L["timer_info_ok"]
                delta_color = "normal"

                with col_timer:
                    # AHT 타이머 표시
                    st.metric(L["timer_metric"], time_str, delta=delta_str, delta_color=delta_color)

                # ⭐ 수정: AHT 타이머 실시간 갱신을 위한 강제 재실행 로직 추가
                # 통화 중이고, Hold 상태가 아닐 때만 1초마다 업데이트하여 실시간성을 확보
                if not st.session_state.is_on_hold and total_seconds < 1000:
                    time.sleep(1)

        # ========================================
        # 화면 구분 (애니메이션 / CC)
        # ========================================
    col_video, col_cc = st.columns([1, 2])

    with col_video:
        st.subheader(f"📺 {L['customer_video_simulation']}")

        if st.session_state.call_sim_stage == "WAITING_CALL":
            st.info("통화 수신 대기 중...")

        elif st.session_state.call_sim_stage == "CALL_ENDED":
            st.info("통화 종료")

        else:
            # ⭐ 비디오 파일 업로드 옵션 추가 (로컬 경로 지원)
            # 항상 펼쳐진 상태로 표시하여 비디오를 쉽게 확인할 수 있도록 함
            with st.expander(L["video_upload_expander"], expanded=True):
                # 비디오 동기화 활성화 여부
                st.session_state.is_video_sync_enabled = st.checkbox(
                    L["video_sync_enable"],
                    value=st.session_state.is_video_sync_enabled,
                    key="video_sync_checkbox"
                )
                
                # OpenAI/Gemini 기반 영상 RAG 설명
                st.markdown("---")
                st.markdown(f"**{L['video_rag_title']}**")
                st.success(L["video_rag_desc"])
                
                # 가상 휴먼 기술은 현재 비활성화 (OpenAI/Gemini 기반 영상 RAG 사용)
                st.session_state.virtual_human_enabled = False
                
                # 성별 및 감정 상태별 비디오 업로드
                st.markdown(f"**{L['video_gender_emotion_setting']}**")
                col_gender_video, col_emotion_video = st.columns(2)
                
                with col_gender_video:
                    video_gender = st.radio(L["video_gender_label"], [L["video_gender_male"], L["video_gender_female"]], key="video_gender_select", horizontal=True)
                    gender_key = "male" if video_gender == L["video_gender_male"] else "female"
                
                with col_emotion_video:
                    video_emotion = st.selectbox(
                        L["video_emotion_label"],
                        ["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"],
                        key="video_emotion_select"
                    )
                    emotion_key = video_emotion.lower()
                
                # 해당 조합의 비디오 업로드
                video_key = f"video_{gender_key}_{emotion_key}"
                uploaded_video = st.file_uploader(
                    L["video_upload_label"].format(gender=video_gender, emotion=video_emotion),
                    type=["mp4", "webm", "ogg"],
                    key=f"customer_video_uploader_{gender_key}_{emotion_key}"
                )
                
                # ⭐ Gemini 제안: 바이트 데이터를 세션 상태에 직접 저장 (파일 저장은 옵션)
                upload_key = f"last_uploaded_video_{gender_key}_{emotion_key}"
                video_bytes_key = f"video_bytes_{gender_key}_{emotion_key}"  # 바이트 데이터 저장 키
                
                if uploaded_video is not None:
                    # 파일이 새로 업로드되었는지 확인 (파일명으로 비교)
                    current_upload_name = uploaded_video.name if hasattr(uploaded_video, 'name') else None
                    last_upload_info = st.session_state.get(upload_key, None)
                    # 딕셔너리인 경우 'name' 키에서 파일명 가져오기
                    if isinstance(last_upload_info, dict):
                        last_upload_name = last_upload_info.get('name', None)
                    else:
                        last_upload_name = last_upload_info
                    
                    # 새 파일이거나 이전과 다른 파일인 경우에만 저장
                    if current_upload_name != last_upload_name:
                        try:
                            # 업로드된 비디오를 즉시 읽기 (rerun 전에 처리)
                            video_bytes = uploaded_video.read()
                            current_upload_size = len(video_bytes)
                            
                            if not video_bytes or len(video_bytes) == 0:
                                st.error(L["video_empty_error"])
                            else:
                                # 파일명 및 확장자 결정
                                uploaded_filename = uploaded_video.name if hasattr(uploaded_video, 'name') else f"{gender_key}_{emotion_key}.mp4"
                                file_ext = os.path.splitext(uploaded_filename)[1].lower() if uploaded_filename else ".mp4"
                                if file_ext not in ['.mp4', '.webm', '.ogg', '.mpeg4']:
                                    file_ext = ".mp4"
                                
                                # MIME 타입 결정
                                mime_type = uploaded_video.type if hasattr(uploaded_video, 'type') else f"video/{file_ext.lstrip('.')}"
                                if not mime_type or mime_type == "application/octet-stream":
                                    mime_type = f"video/{file_ext.lstrip('.')}"
                                
                                # ⭐ 1차 해결책: 바이트 데이터를 세션 상태에 직접 저장 (가장 안정적)
                                st.session_state[video_bytes_key] = video_bytes
                                st.session_state[video_key] = video_bytes_key  # 경로 대신 바이트 키 저장
                                st.session_state[upload_key] = {
                                    'name': current_upload_name,
                                    'size': current_upload_size,
                                    'mime': mime_type,
                                    'ext': file_ext
                                }
                                
                                file_size_mb = current_upload_size / (1024 * 1024)
                                st.success(L["video_bytes_saved"].format(name=current_upload_name, size=f"{file_size_mb:.2f}"))
                                
                                # ⭐ 즉시 미리보기 (바이트 데이터 직접 사용)
                                try:
                                    st.video(video_bytes, format=mime_type, autoplay=False, loop=False, muted=False)
                                except Exception as video_error:
                                    st.warning(f"⚠️ {L.get('video_preview_error', '비디오 미리보기 오류')}: {video_error}")
                                    # MIME 타입을 기본값으로 재시도
                                    try:
                                        st.video(video_bytes, format=f"video/{file_ext.lstrip('.')}", autoplay=False, loop=False, muted=False)
                                    except:
                                        st.error(L["video_playback_error"])
                                
                                # ⭐ 옵션: 파일 저장도 시도 (백업용, 실패해도 바이트는 이미 저장됨)
                                try:
                                    video_dir = os.path.join(DATA_DIR, "videos")
                                    os.makedirs(video_dir, exist_ok=True)
                                    video_filename = f"{gender_key}_{emotion_key}{file_ext}"
                                    video_path = os.path.join(video_dir, video_filename)
                                    
                                    # 파일 저장 시도 (권한 문제가 있어도 바이트는 이미 저장됨)
                                    try:
                                        with open(video_path, "wb") as f:
                                            f.write(video_bytes)
                                            f.flush()
                                        st.info(f"📂 파일도 저장됨: {video_path}")
                                    except Exception as save_error:
                                        st.info(f"💡 파일 저장은 건너뛰었습니다 (바이트 데이터는 메모리에 저장됨): {save_error}")
                                except:
                                    pass  # 파일 저장 실패해도 바이트는 이미 저장됨
                                
                        except Exception as e:
                            st.error(L["video_upload_error"].format(error=str(e)))
                            import traceback
                            st.code(traceback.format_exc())
                
                # 업로드된 비디오가 있으면 현재 선택된 조합의 비디오 표시
                st.markdown("---")
                st.markdown(f"**{L['video_current_selection'].format(gender=video_gender, emotion=video_emotion)}**")
                
                # ⭐ Gemini 제안: 세션 상태에서 바이트 데이터 직접 조회
                video_bytes_key = f"video_bytes_{gender_key}_{emotion_key}"
                current_video_bytes = st.session_state.get(video_bytes_key, None)
                
                if current_video_bytes:
                    # 바이트 데이터가 있으면 직접 사용 (가장 안정적)
                    upload_info = st.session_state.get(upload_key, {})
                    mime_type = upload_info.get('mime', 'video/mp4')
                    file_ext = upload_info.get('ext', '.mp4')
                    
                    st.success(f"✅ 비디오 바이트 데이터 발견: {upload_info.get('name', '업로드된 비디오')}")
                    try:
                        st.video(current_video_bytes, format=mime_type, autoplay=False, loop=False, muted=False)
                        st.caption(L["video_auto_play_info"].format(gender=video_gender, emotion=video_emotion))
                    except Exception as e:
                        st.warning(f"비디오 재생 오류: {e}")
                        # MIME 타입을 기본값으로 재시도
                        try:
                            st.video(current_video_bytes, format=f"video/{file_ext.lstrip('.')}", autoplay=False, loop=False, muted=False)
                        except:
                            st.error(L["video_playback_error"])
                else:
                    # 바이트 데이터가 없으면 파일 경로로 시도 (하위 호환성)
                    current_video_path = get_video_path_by_avatar(
                        gender_key,
                        video_emotion,
                        is_speaking=False,
                        gesture="NONE"
                    )
                    
                    if current_video_path and os.path.exists(current_video_path):
                        st.success(f"✅ 비디오 파일 발견: {os.path.basename(current_video_path)}")
                        try:
                            with open(current_video_path, "rb") as f:
                                existing_video_bytes = f.read()
                            st.video(existing_video_bytes, format="video/mp4", autoplay=False, loop=False, muted=False)
                            st.caption(L["video_auto_play_info"].format(gender=video_gender, emotion=video_emotion))
                        except Exception as e:
                            st.warning(f"비디오 재생 오류: {e}")
                    else:
                        st.info(L["video_upload_prompt"].format(filename=f"{gender_key}_{emotion_key}.mp4"))
                    
                    # 디버깅 정보: 비디오 디렉토리와 파일 목록 표시
                    video_dir = os.path.join(DATA_DIR, "videos")
                    st.caption(L["video_save_path"] + f" {video_dir}")
                    
                    if os.path.exists(video_dir):
                        all_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.webm', '.ogg'))]
                        if all_videos:
                            st.caption(f"{L['video_uploaded_files']} ({len(all_videos)}개):")
                            for vid in all_videos:
                                st.caption(f"  - {vid}")
                            
                            # 비슷한 비디오 파일이 있는지 확인
                            similar_videos = [
                                f for f in all_videos
                                if f.startswith(f"{gender_key}_") and f.endswith(('.mp4', '.webm', '.ogg'))
                            ]
                            if similar_videos:
                                st.caption(f"📁 {L.get('video_similar_gender', '같은 성별의 다른 비디오')}: {', '.join(similar_videos[:3])}")
                                st.caption(L.get("video_rename_hint", "💡 위 비디오 중 하나를 사용하려면 파일명을 변경하거나 새로 업로드하세요."))
                        else:
                            st.caption(L["video_directory_empty"])
                    else:
                        st.caption(L["video_directory_not_exist"].format(path=video_dir))
                
                # 또는 로컬 파일 경로 입력 및 복사
                video_path_input = st.text_input(
                    L["video_local_path_input"],
                    placeholder=L["video_local_path_placeholder"],
                    key="video_path_input"
                )
                
                if video_path_input:
                    try:
                        # ⭐ Gemini 제안: 절대 경로 검증 강화
                        if not os.path.isabs(video_path_input):
                            st.error("❌ 로컬 경로 입력 시 반드시 **절대 경로**를 사용해주세요 (예: C:\\Users\\...\\video.mp4).")
                            st.error("💡 Streamlit 앱이 실행되는 서버 환경과 파일 시스템이 다르면 접근할 수 없습니다.")
                            st.stop()
                        
                        source_video_path = video_path_input
                        
                        if not os.path.exists(source_video_path):
                            st.error(f"❌ 파일을 찾을 수 없습니다: {source_video_path}")
                            st.error("💡 파일 경로를 확인하고, Streamlit 앱이 실행되는 서버에서 접근 가능한 경로인지 확인해주세요.")
                            st.stop()
                        
                        # 원본 파일 읽기
                        with open(source_video_path, "rb") as f:
                            video_bytes = f.read()
                        
                        if len(video_bytes) == 0:
                            st.error("❌ 파일이 비어있습니다.")
                            st.stop()
                        
                        # 파일명 및 확장자 결정
                        source_filename = os.path.basename(source_video_path)
                        file_ext = os.path.splitext(source_filename)[1].lower()
                        if file_ext not in ['.mp4', '.webm', '.ogg', '.mpeg4']:
                            file_ext = ".mp4"
                        
                        mime_type = f"video/{file_ext.lstrip('.')}"
                        
                        # ⭐ 바이트 데이터를 세션 상태에 직접 저장 (파일 복사는 옵션)
                        video_bytes_key = f"video_bytes_{gender_key}_{emotion_key}"
                        st.session_state[video_bytes_key] = video_bytes
                        st.session_state[video_key] = video_bytes_key
                        st.session_state[upload_key] = {
                            'name': source_filename,
                            'size': len(video_bytes),
                            'mime': mime_type,
                            'ext': file_ext
                        }
                        
                        file_size_mb = len(video_bytes) / (1024 * 1024)
                        st.success(f"✅ 비디오 바이트 로드 완료: {source_filename} ({file_size_mb:.2f} MB)")
                        
                        # 비디오 미리보기 (바이트 데이터 직접 사용)
                        try:
                            st.video(video_bytes, format=mime_type, autoplay=False, loop=False, muted=False)
                        except Exception as video_error:
                            st.warning(f"⚠️ 비디오 미리보기 오류: {video_error}")
                        
                        # ⭐ 옵션: 파일 복사도 시도 (백업용)
                        try:
                            video_dir = os.path.join(DATA_DIR, "videos")
                            os.makedirs(video_dir, exist_ok=True)
                            video_filename = f"{gender_key}_{emotion_key}{file_ext}"
                            target_video_path = os.path.join(video_dir, video_filename)
                            
                            with open(target_video_path, "wb") as f:
                                f.write(video_bytes)
                                f.flush()
                            st.info(f"📂 파일도 복사됨: {target_video_path}")
                        except Exception as copy_error:
                            st.info(f"💡 파일 복사는 건너뛰었습니다 (바이트 데이터는 메모리에 저장됨): {copy_error}")
                        
                        # 입력 필드 초기화
                        st.session_state.video_path_input = ""
                        
                    except Exception as e:
                        st.error(f"❌ 비디오 파일 로드 오류: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # 상태 선택 및 비디오 표시
            st.markdown("---")
            st.markdown(f"**{L['video_current_avatar']}**")
            
            if st.session_state.is_on_hold:
                avatar_state = "HOLD"
            else:
                avatar_state = st.session_state.customer_avatar.get("state", "NEUTRAL")
            
            customer_gender = st.session_state.customer_avatar.get("gender", "male")
            
            # get_video_path_by_avatar 함수를 사용하여 비디오 경로 찾기
            video_path = get_video_path_by_avatar(
                customer_gender, 
                avatar_state, 
                is_speaking=False,  # 미리보기는 자동 재생하지 않음
                gesture="NONE"
            )
            
            # 비디오 표시
            if video_path and os.path.exists(video_path):
                try:
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    
                    # 비디오 정보 표시
                    avatar_emoji = {
                        "NEUTRAL": "😐",
                        "HAPPY": "😊",
                        "ANGRY": "😠",
                        "ASKING": "🤔",
                        "SAD": "😢",
                        "HOLD": "⏸️"
                    }.get(avatar_state, "😐")
                    
                    st.markdown(f"### {avatar_emoji} {customer_gender.upper()} - {avatar_state}")
                    st.caption(f"비디오: {os.path.basename(video_path)}")
                    
                    # 현재 말하는 중이면 자동 재생, 아니면 수동 재생
                    is_speaking = bool(
                        st.session_state.get("customer_initial_audio_bytes") or 
                        st.session_state.get("current_customer_audio_text")
                    )
                    
                    autoplay_video = st.session_state.is_video_sync_enabled and is_speaking
                    st.video(video_bytes, format="video/mp4", autoplay=autoplay_video, loop=False, muted=False)
                    
                except Exception as e:
                    st.warning(f"비디오 재생 오류: {e}")
                    avatar_emoji = {
                        "NEUTRAL": "😐",
                        "HAPPY": "😊",
                        "ANGRY": "😠",
                        "ASKING": "🤔",
                        "SAD": "😢",
                        "HOLD": "⏸️"
                    }.get(avatar_state, "😐")
                    st.markdown(f"### {avatar_emoji} {L['customer_avatar']}")
                    st.info(L.get("avatar_status_info", "상태: {state} | 성별: {gender}").format(state=avatar_state, gender=customer_gender))
            else:
                # 비디오가 없으면 이모지로 표시
                avatar_emoji = {
                    "NEUTRAL": "😐",
                    "HAPPY": "😊",
                    "ANGRY": "😠",
                    "ASKING": "🤔",
                    "SAD": "😢",
                    "HOLD": "⏸️"
                }.get(avatar_state, "😐")
                
                st.markdown(f"### {avatar_emoji} 고객 아바타")
                st.info(L.get("avatar_status_info", "상태: {state} | 성별: {gender}").format(state=avatar_state, gender=customer_gender))
                st.warning(L["video_avatar_upload_prompt"].format(filename=f"{customer_gender}_{avatar_state.lower()}.mp4"))
                
                # 업로드된 비디오 목록 표시
                video_dir = os.path.join(DATA_DIR, "videos")
                if os.path.exists(video_dir):
                    uploaded_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.webm', '.ogg'))]
                    if uploaded_videos:
                        st.caption(f"{L['video_uploaded_files']}: {', '.join(uploaded_videos[:5])}")
                        if len(uploaded_videos) > 5:
                            st.caption(L.get("video_more_files", f"... 외 {len(uploaded_videos) - 5}개").format(count=len(uploaded_videos) - 5))

    with col_cc:
        # ⭐ 수정: "전화 수신 중" 메시지는 통화 중일 때만 표시
        if st.session_state.call_sim_stage == "IN_CALL":
            if st.session_state.call_sim_mode == "INBOUND":
                st.markdown(
                    f"## {L['call_status_ringing'].format(number=st.session_state.incoming_phone_number)}"
                )
            else:
                st.markdown(
                    f"## {L['button_call_outbound']} ({st.session_state.incoming_phone_number})"
                )
        st.markdown("---")

    # ========================================
    # WAITING / RINGING 상태
    # ========================================
    if st.session_state.call_sim_stage in ["WAITING_CALL", "RINGING"]:

        if "call_sim_mode" not in st.session_state:
            st.session_state.call_sim_mode = "INBOUND"  # INBOUND or OUTBOUND

        if st.session_state.call_sim_mode == "INBOUND":
            st.subheader(L["call_status_waiting"])
        else:
            st.subheader(L["button_call_outbound"])

        # 홈페이지 웹 주소 입력 (선택사항)
        st.session_state.call_website_url = st.text_input(
            L.get("website_url_label", "홈페이지 웹 주소 (선택사항)"),
            key="call_website_url_input",
            value=st.session_state.call_website_url,
            placeholder=L.get("website_url_placeholder", "https://example.com (홈페이지 주소가 있으면 입력하세요)"),
        )

        # 초기 문의 입력 (고객이 전화로 말할 내용)
        st.session_state.call_initial_query = st.text_area(
            L["customer_query_label"],
            key="call_initial_query_text_area",
            height=100,
            placeholder=L["call_query_placeholder"],
        )

        # 가상 전화번호 표시
        st.session_state.incoming_phone_number = st.text_input(
            "Incoming/Outgoing Phone Number",
            key="incoming_phone_number_input",
            value=st.session_state.incoming_phone_number,
            placeholder=L["call_number_placeholder"],
        )

        # 고객 유형 선택
        customer_type_options = L["customer_type_options"]
        default_idx = customer_type_options.index(
            st.session_state.customer_type_sim_select) if st.session_state.customer_type_sim_select in customer_type_options else 0

        st.session_state.customer_type_sim_select = st.selectbox(
            L["customer_type_label"],
            customer_type_options,
            index=default_idx,
            key="call_customer_type_sim_select_widget",
        )

        # ⭐ 추가: 고객 성별 및 감정 상태 설정
        col_gender, col_emotion = st.columns(2)
        
        with col_gender:
            # 고객 성별 선택
            if "customer_gender" not in st.session_state:
                st.session_state.customer_gender = "male"
            
            # ⭐ 수정: 번역 키 사용
            gender_options = [L["gender_male"], L["gender_female"]]
            current_gender = st.session_state.customer_avatar.get("gender", "male")
            default_gender_idx = 0 if current_gender == "male" else 1
            
            selected_gender_display = st.radio(
                L["customer_gender_label"],
                gender_options,
                index=default_gender_idx,
                key="call_customer_gender_radio",
                horizontal=True
            )
            # 세션 상태에 저장 (영어로)
            st.session_state.customer_avatar["gender"] = "male" if selected_gender_display == L["gender_male"] else "female"
            st.session_state.customer_gender = st.session_state.customer_avatar["gender"]
        
        with col_emotion:
            # 고객 감정 상태 선택
            # ⭐ 수정: 번역 키 사용
            emotion_options = [
                L["emotion_happy"],
                L["emotion_dissatisfied"],
                L["emotion_angry"],
                L["emotion_sad"],
                L["emotion_neutral"]
            ]
            emotion_mapping = {
                L["emotion_happy"]: "HAPPY",
                L["emotion_dissatisfied"]: "ASKING",
                L["emotion_angry"]: "ANGRY",
                L["emotion_sad"]: "SAD",
                L["emotion_neutral"]: "NEUTRAL"
            }
            
            current_emotion_state = st.session_state.customer_avatar.get("state", "NEUTRAL")
            default_emotion_idx = 4  # 기본값: 중립
            for i, (emotion_display, emotion_state) in enumerate(emotion_mapping.items()):
                if emotion_state == current_emotion_state:
                    default_emotion_idx = i
                    break
            
            selected_emotion = st.selectbox(
                L["customer_emotion_label"],
                emotion_options,
                index=default_emotion_idx,
                key="call_customer_emotion_select",
            )
            # 세션 상태에 저장
            st.session_state.customer_avatar["state"] = emotion_mapping.get(selected_emotion, "NEUTRAL")

        st.markdown("---")

        col_in, col_out = st.columns(2)

        # 전화 응답 (수신)
        with col_in:
            if st.button(L["button_answer"], key=f"answer_call_btn_{st.session_state.sim_instance_id}"):
                # 입력 검증
                if not st.session_state.call_initial_query.strip():
                    st.warning(L["simulation_warning_query"])
                    # st.stop()

                # ⭐ 수정: OpenAI 또는 Gemini API 키 체크
                has_openai = st.session_state.openai_client is not None
                has_gemini = bool(get_api_key("gemini"))
                
                if not st.session_state.is_llm_ready or (not has_openai and not has_gemini):
                    st.error(L["simulation_no_key_warning"] + " (OpenAI 또는 Gemini API Key 필요)")
                    # st.stop()

                # INBOUND 모드 설정
                st.session_state.call_sim_mode = "INBOUND"

                # 시뮬레이션 초기화 및 시작
                st.session_state.call_sim_stage = "IN_CALL"
                st.session_state.is_call_ended = False
                st.session_state.is_on_hold = False
                st.session_state.total_hold_duration = timedelta(0)
                st.session_state.hold_start_time = None
                st.session_state.start_time = datetime.now()  # 통화 시작 시간 (AHT 시작)
                st.session_state.simulator_messages = []
                st.session_state.current_customer_audio_text = ""
                st.session_state.current_agent_audio_text = ""
                st.session_state.agent_response_input_box_widget_call = ""
                st.session_state.sim_instance_id = str(uuid.uuid4())
                st.session_state.call_summary_text = ""  # 요약 초기화
                st.session_state.customer_initial_audio_bytes = None  # 오디오 초기화
                st.session_state.customer_history_summary = ""  # AI 요약 초기화 (추가)
                st.session_state.sim_audio_bytes = None  # 녹음 파일 초기화 (추가)

                # ⭐ 수정: 자동 인사말 생성 제거 - 에이전트가 직접 녹음하도록 변경
                st.session_state.just_entered_call = False
                st.session_state.customer_turn_start = False  # 에이전트 인사말 완료 전까지 False

                # 고객의 첫 번째 음성 메시지 (시뮬레이션 시작 메시지)
                initial_query_text = st.session_state.call_initial_query.strip()
                st.session_state.current_customer_audio_text = initial_query_text

                # ⭐ 입력 텍스트의 언어를 자동 감지 및 언어 설정 업데이트
                try:
                    detected_lang = detect_text_language(initial_query_text)
                    if detected_lang in ["ko", "en", "ja"] and detected_lang != st.session_state.language:
                        st.session_state.language = detected_lang
                        st.info(f"🌐 입력 언어가 감지되어 언어 설정이 '{detected_lang}'로 자동 변경되었습니다.")
                except Exception as e:
                    print(f"Language detection failed in call: {e}")
                    detected_lang = st.session_state.language

                # ⭐ 고객의 첫 문의 TTS 음성 생성 및 저장 (감지된 언어 사용)
                with st.spinner(L["tts_status_generating"] + " (Initial Customer Query)"):
                    audio_bytes, msg = synthesize_tts(initial_query_text, st.session_state.language, role="customer")
                    if audio_bytes:
                        st.session_state.customer_initial_audio_bytes = audio_bytes
                    else:
                        st.error(f"❌ {msg}")
                        st.session_state.customer_initial_audio_bytes = None

                # ✅ 상태 변경 후 재실행하여 IN_CALL 상태로 전환
                # 에이전트가 인사말을 녹음할 수 있도록 안내 메시지 표시
                st.info(L["call_started_message"])

        # 전화 발신 (새로운 세션 시작)
        with col_out:
            st.markdown(f"### {L['button_call_outbound']}")
            call_targets = [
                L["call_target_customer"],
                L["call_target_partner"]
            ]

            call_target_selection = st.radio(
                L.get("call_target_select_label", "발신 대상 선택"),
                call_targets,
                key="outbound_call_target_radio",
                horizontal=True
            )

            # 선택된 대상에 따라 버튼 텍스트 변경
            if call_target_selection == L["call_target_customer"]:
                button_text = L["button_call_outbound_to_customer"]
            else:
                button_text = L["button_call_outbound_to_provider"]

            if st.button(button_text, key=f"outbound_call_start_btn_{st.session_state.sim_instance_id}", type="secondary", use_container_width=True):
                # 입력 검증
                if not st.session_state.call_initial_query.strip():
                    st.warning("전화 발신 목표 (고객 문의 내용)를 입력해 주세요。")
                    # st.stop()

                # ⭐ 수정: OpenAI 또는 Gemini API 키 체크
                has_openai = st.session_state.openai_client is not None
                has_gemini = bool(get_api_key("gemini"))
                
                if not st.session_state.is_llm_ready or (not has_openai and not has_gemini):
                    st.error(L["simulation_no_key_warning"] + " (OpenAI 또는 Gemini API Key 필요)")
                    # st.stop()

                # OUTBOUND 모드 설정 및 시뮬레이션 시작
                st.session_state.call_sim_mode = "OUTBOUND"

                # 시뮬레이션 초기화 및 시작
                st.session_state.call_sim_stage = "IN_CALL"
                st.session_state.is_call_ended = False
                st.session_state.is_on_hold = False
                st.session_state.total_hold_duration = timedelta(0)
                st.session_state.hold_start_time = None
                st.session_state.start_time = datetime.now()  # 통화 시작 시간 (AHT 시작)
                st.session_state.simulator_messages = []

                # ⭐ 수정: 자동 인사말 생성 제거 - 에이전트가 직접 녹음하도록 변경
                st.session_state.just_entered_call = False
                st.session_state.customer_turn_start = False

                initial_query_text = st.session_state.call_initial_query.strip()

                # 발신 시뮬레이션에서는 에이전트가 먼저 말해야 하므로, 고객 CC 텍스트는 안내 메시지로 설정
                st.session_state.current_customer_audio_text = f"📞 {L['button_call_outbound']} 성공! {call_target_selection}이(가) 받았습니다。 잠시 후 응답이 시작됩니다。 (문의 목표: {initial_query_text[:50]}...)"
                st.session_state.current_agent_audio_text = ""  # Agent speaks first
                st.session_state.agent_response_input_box_widget_call = ""
                st.session_state.sim_instance_id = str(uuid.uuid4())
                st.session_state.call_summary_text = ""
                st.session_state.customer_initial_audio_bytes = None
                st.session_state.customer_history_summary = ""
                st.session_state.sim_audio_bytes = None

                st.success(f"'{call_target_selection}'에게 전화 발신 시뮬레이션이 시작되었습니다. 아래 마이크 버튼을 눌러 인사말을 녹음하세요。")

        # ------------------
        # IN_CALL 상태 (통화 중)
        # ------------------
    elif st.session_state.call_sim_stage == "IN_CALL":
        # ⭐ 수정: 자동 인사말 생성 로직 제거 - 에이전트가 직접 녹음하도록 변경
        
        # ------------------------------
        # 전화 통화 제목 (통화 중일 때만 표시)
        # ------------------------------
        # ⭐ 수정: 제목은 이미 위에서 표시되므로 여기서는 제거
        # st.markdown(f"## {title}")
        # st.markdown("---")

        # ------------------------------
        # Hangup / Hold 버튼
        # ------------------------------
        col_hangup, col_hold = st.columns(2)

        with col_hangup:
            if st.button(L["button_hangup"], key="hangup_call_btn"):

                # Hold 정산
                if st.session_state.is_on_hold and st.session_state.hold_start_time:
                    st.session_state.total_hold_duration += datetime.now() - st.session_state.hold_start_time

                # 요약 생성
                with st.spinner("AI 요약 생성 중..."):
                    # ⭐ [수정 9] 함수명 통일: summarize_history_for_call로 변경 및 호출
                    summary = summarize_history_for_call(
                        st.session_state.simulator_messages,
                        st.session_state.call_initial_query,
                        st.session_state.language
                    )
                    st.session_state.call_summary_text = summary

                # 종료
                st.session_state.call_sim_stage = "CALL_ENDED"
                st.session_state.is_call_ended = True

                # ⭐ [수정 10] Hangup 후 UI 갱신을 위해 rerun 추가
                st.rerun()

        # ------------------------------
        # Hold / Resume
        # ------------------------------
        with col_hold:
            if st.session_state.is_on_hold:
                if st.button(L["button_resume"], key="resume_call_btn"):
                    # Hold 상태 해제 및 시간 정산
                    st.session_state.is_on_hold = False
                    if st.session_state.hold_start_time:
                        st.session_state.total_hold_duration += datetime.now() - st.session_state.hold_start_time
                        st.session_state.hold_start_time = None
            else:
                if st.button(L["button_hold"], key="hold_call_btn"):
                    st.session_state.is_on_hold = True
                    st.session_state.hold_start_time = datetime.now()

        # ------------------------------
        # Hold 표시
        # ------------------------------
        if st.session_state.is_on_hold:
            if st.session_state.hold_start_time:
                current_hold = datetime.now() - st.session_state.hold_start_time
            else:
                current_hold = timedelta(0)

            total_hold = st.session_state.total_hold_duration + current_hold
            hold_str = str(total_hold).split('.')[0]

            st.warning(L["hold_status"].format(duration=hold_str))
            time.sleep(1)

        # ------------------------------
        # (중략) - **이관, 힌트, 요약, CC, Whisper 전사, 고객 반응 생성**
        # ------------------------------
        def transfer_session(target_lang: str, current_messages: List[Dict[str, str]]):
            """언어 이관 시스템 메시지를 추가하고 세션 언어를 변경합니다."""

            current_lang = st.session_state.language  # 현재 언어 확인 (Source language)
            L = LANG[current_lang]

            # API 키 체크
            if not st.session_state.is_llm_ready:
                st.error(L["simulation_no_key_warning"].replace('API Key', 'LLM API Key'))
                return

            current_lang_at_start = st.session_state.language  # Source language

            # AHT 타이머 정지 (실제로 통화가 종료되는 것은 아니므로, AHT는 계속 흐름)
            # st.session_state.start_time = None

            # 1. 로딩 시작 (시간 양해 메시지 시뮬레이션)
            with st.spinner(L["transfer_loading"]):
                time.sleep(np.random.uniform(5, 10))

                # 2. 대화 기록을 번역할 텍스트로 가공
                history_text = ""
                for msg in current_messages:
                    role = "Customer" if msg["role"].startswith("customer") or msg[
                        "role"] == "initial_query" else "Agent"
                    if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response",
                                       "customer_closing_response", "phone_exchange"]:  # phone_exchange 추가
                        history_text += f"{role}: {msg['content']}\n"

                # ⭐ 수정: 먼저 핵심 포인트만 요약한 후 번역
                # 요약 프롬프트 생성
                lang_name_source = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(current_lang_at_start, "Korean")
                summary_prompt = f"""
You are an AI assistant that summarizes customer service conversations. 
Extract ONLY the key points from the conversation below. Keep it concise and focused on:
1. Customer's main inquiry/question
2. Key information provided by the agent
3. Important decisions or outcomes
4. Any unresolved issues

Write the summary in {lang_name_source}. Maximum 200 words. Be brief and to the point.

--- Conversation ---
{history_text}
---

Key Points Summary:
        # =========================
        # AI 요약 버튼 및 표시 로직 (추가된 기능)
        # =========================
        st.markdown("---")
        # ⭐ history_expander_title에서 괄호 안 내용만 제거 (예: (최근 10건))
        summary_title = L['history_expander_title'].split('(')[0].strip()
        st.markdown(f"### 📑 {summary_title} 요약")

        # 1. 요약/번역 재시도 버튼 영역
        col_sum_btn, col_trans_btn = st.columns(2)

        with col_sum_btn:
            # ⭐ [수정 FIX] 키 중복 오류 해결: 세션 ID를 키에 추가
            if st.button(L["btn_request_phone_summary"], key=f"btn_request_phone_summary_{st.session_state.sim_instance_id}"):
                # 요약 함수 호출
                st.session_state.customer_history_summary = summarize_history_with_ai(st.session_state.language)

        # 2. 이관 번역 재시도 버튼 (이관 후 번역이 실패했을 경우)
        if st.session_state.language != st.session_state.language_at_transfer_start and not st.session_state.transfer_summary_text:
            with col_trans_btn:
                # ⭐ [수정 FIX] 키 중복 오류 해결: 세션 ID와 언어 코드를 조합하여 고유 키 생성
                retry_key = f"btn_retry_translation_{st.session_state.language_at_transfer_start}_{st.session_state.language}_{st.session_state.sim_instance_id}"
                if st.button(L["button_retry_translation"], key=retry_key):
                    with st.spinner(L["transfer_loading"]):
                        # 이관 번역 로직 재실행 (기존 로직 유지)
                        translated_summary, is_success = translate_text_with_llm(
                            get_chat_history_for_prompt(include_attachment=False),
                            st.session_state.language,
                            st.session_state.language_at_transfer_start
                        )
                        st.session_state.transfer_summary_text = translated_summary
                        st.session_state.translation_success = is_success

        # 3. 요약 내용 표시
        if st.session_state.transfer_summary_text:
            st.subheader(f"🔍 {L['transfer_summary_header']}")
            st.info(st.session_state.transfer_summary_text)
            # ⭐ 이관 요약에 TTS 버튼 추가
            render_tts_button(
                st.session_state.transfer_summary_text,
                st.session_state.language,
                role="agent",
                prefix="trans_summary_tts_call",
                index=-1  # 고유 세션 ID 기반의 키를 생성하도록 지시
            )
        elif st.session_state.customer_history_summary:
            st.subheader("💡 AI 요약")
            st.info(st.session_state.customer_history_summary)

        st.markdown("---")

        # --- 실시간 응대 힌트 영역 ---
        hint_cols = st.columns([4, 1])
        with hint_cols[0]:
            st.info(L["hint_placeholder"] + st.session_state.realtime_hint_text)

        with hint_cols[1]:
            # 힌트 요청 버튼
            if st.button(L["button_request_hint"], key=f"btn_request_hint_call_{st.session_state.sim_instance_id}"):
                with st.spinner(L["response_generating"]):
                    # 전화 탭이므로 is_call=True
                    hint = generate_realtime_hint(current_lang, is_call=True)
                    st.session_state.realtime_hint_text = hint

        # =========================
        # CC 자막 / 음성 입력 및 제어 로직 (기존 로직)
        # =========================================

        # --- 실시간 CC 자막 / 전사 영역 ---
        st.subheader(L["cc_live_transcript"])

        if st.session_state.is_on_hold:
            st.text_area("Customer", value=L["customer_waiting_hold"], height=50, disabled=True, key="customer_live_cc_area")
            st.text_area("Agent", value=L["agent_hold_message"], height=50, disabled=True,
                         key="agent_live_cc_area")
        else:
            # 고객 CC (LLM 생성 텍스트 또는 초기 문의)
            # ⭐ 수정: 고객 문의가 비어있지 않으면 초기 문의를 표시
            customer_cc_text = st.session_state.current_customer_audio_text
            if not customer_cc_text and st.session_state.call_initial_query:
                customer_cc_text = st.session_state.call_initial_query
            st.text_area(
                "Customer",
                value=customer_cc_text,
                height=50,
                disabled=True,
                key="customer_live_cc_area",
            )

            # 에이전트 CC (마이크 전사)
            st.text_area(
                "Agent",
                value=st.session_state.current_agent_audio_text,
                height=50,
                disabled=True,
                key="agent_live_cc_area",
            )

        st.markdown("---")

        # --- 에이전트 음성 입력 / 녹음 ---
        st.subheader(L["mic_input_status"])

        # 음성 입력: 짧은 청크로 끊어서 전사해야 실시간 CC 모방 가능
        if st.session_state.is_on_hold:
            st.info(L["call_on_hold_message"])
            mic_audio = None
        else:
            # ✅ 마이크 위젯을 항상 렌더링하여 활성화 상태를 유지
            mic_audio = mic_recorder(
                start_prompt=L["agent_response_prompt"],
                stop_prompt=L["agent_response_stop_and_send"],
                just_once=True,
                format="wav",
                use_container_width=True,
                key="call_sim_mic_recorder",
            )

            # 녹음 완료 (mic_audio.get("bytes")가 채워짐) 시, 바이트를 저장하고 재실행
            # ⭐ 수정: 채팅/이메일 탭과 동일한 패턴으로 수정 - 조건 단순화
            if mic_audio and mic_audio.get("bytes"):
                # ⭐ 수정: 이미 처리 중인 경우 중복 처리 방지
                if "bytes_to_process" not in st.session_state or st.session_state.bytes_to_process is None:
                    st.session_state.bytes_to_process = mic_audio["bytes"]
                    st.session_state.current_agent_audio_text = L["recording_complete_transcribing"]
                    # ✅ 재실행하여 다음 실행 주기에서 전사 로직을 처리
                    st.rerun()

        # ⭐ 수정: 전사 로직을 마이크 위젯 렌더링 블록 밖으로 이동하여 실행 순서 보장
        # 전사 로직: bytes_to_process에 데이터가 있을 때만 실행
        if "bytes_to_process" in st.session_state and st.session_state.bytes_to_process is not None:
            # ⭐ 수정: OpenAI 또는 Gemini API 키가 있는지 확인
            has_openai = st.session_state.openai_client is not None
            has_gemini = bool(get_api_key("gemini"))
            
            if not has_openai and not has_gemini:
                st.error(L["openai_missing"] + " 또는 Gemini API Key가 필요합니다.")
                st.session_state.bytes_to_process = None
                # ⭐ 최적화: 에러 메시지 표시 후 불필요한 rerun 제거 (사용자가 API 키를 설정하면 자동으로 재실행됨)
            else:
                # ⭐ 전사 결과를 저장할 변수 초기화
                agent_response_transcript = None

                # ⭐ [수정]: Whisper 전사 로직 (채팅/이메일 탭과 동일한 패턴)
                # 전사 후 바이트 데이터 백업 (전사 전에 백업)
                audio_bytes_backup = st.session_state.bytes_to_process
                
                # 전사 후 바이트 데이터 즉시 삭제 (조건문 재평가 방지)
                st.session_state.bytes_to_process = None
                
                with st.spinner(L["whisper_processing"]):
                    try:
                        # 1) Whisper 전사 (자동 언어 감지 사용) - 채팅/이메일과 동일한 방식
                        agent_response_transcript = transcribe_bytes_with_whisper(
                            audio_bytes_backup,
                            "audio/wav",
                            lang_code=None,
                            auto_detect=True
                        )
                    except Exception as e:
                        agent_response_transcript = f"❌ 전사 오류: {e}"

                # 2) 전사 실패 처리 (채팅/이메일과 동일한 패턴)
                if not agent_response_transcript or agent_response_transcript.startswith("❌"):
                    error_msg = agent_response_transcript if agent_response_transcript else L["transcription_no_result"]
                    st.error(error_msg)
                    st.session_state.current_agent_audio_text = L["transcription_error"]
                    # ⭐ 최적화: 전사 실패 시에도 CC에 반영되지만 불필요한 rerun 제거 (Streamlit이 자동으로 재실행)
                elif not agent_response_transcript.strip(): # ⭐ 수정: 전사 결과가 비어 있거나 (공백만 있는 경우) 다음 단계로 진행하지 못하는 문제 해결
                    st.warning(L["transcription_empty_warning"])
                    st.session_state.current_agent_audio_text = ""
                    # ⭐ 최적화: 불필요한 rerun 제거
                elif agent_response_transcript.strip():
                    # 3) 전사 성공 - CC에 반영 (전사 결과를 먼저 CC 영역에 표시)
                    agent_response_transcript = agent_response_transcript.strip()
                    st.session_state.current_agent_audio_text = agent_response_transcript
                    
                    # 성공 메시지 표시 (채팅/이메일과 유사)
                    snippet = agent_response_transcript[:50].replace("\n", " ")
                    if len(agent_response_transcript) > 50:
                        snippet += "..."
                    st.success(L["whisper_success"] + f" **인식 내용:** *{snippet}*")

                    # ⭐ 수정: 첫 인사말인지 확인 (simulator_messages에 phone_exchange가 없으면 첫 인사말)
                    is_first_greeting = not any(
                        msg.get("role") == "phone_exchange" 
                        for msg in st.session_state.simulator_messages
                    )
                    
                    # ⭐ 수정: 전화 발신 모드 확인
                    is_outbound_call = st.session_state.get("call_sim_mode", "INBOUND") == "OUTBOUND"

                    if is_first_greeting:
                        # 첫 인사말인 경우: 로그에 기록하고 고객 문의 재생 준비
                        st.session_state.simulator_messages.append(
                            {"role": "agent", "content": agent_response_transcript}
                        )
                        # 아바타 표정 초기화
                        st.session_state.customer_avatar["state"] = "NEUTRAL"
                        
                        # ⭐ 수정: 전화 발신 모드에서 customer_initial_audio_bytes가 없으면 바로 고객 응답 생성
                        if is_outbound_call and not st.session_state.get("customer_initial_audio_bytes"):
                            # 전화 발신 모드이고 고객 문의 오디오가 없으면 바로 고객 응답 생성
                            st.session_state.current_agent_audio_text = agent_response_transcript
                            st.session_state.process_customer_reaction = True
                            st.session_state.pending_agent_transcript = agent_response_transcript
                            st.rerun()
                        else:
                            # ⭐ 수정: 고객 문의를 CC 자막에 미리 반영 (재생 전에 반영)
                            if st.session_state.call_initial_query:
                                st.session_state.current_customer_audio_text = st.session_state.call_initial_query
                            # ⭐ 수정: 고객 문의 재생을 바로 실행 (같은 실행 주기에서 처리)
                            # 고객 문의 재생 로직이 아래에 있으므로 플래그만 설정
                            st.session_state.customer_turn_start = True
                            # ⭐ 최적화: 플래그 설정 후 재실행하여 고객 문의 재생 로직 실행
                            st.rerun()
                    else:
                        # 이후 응답인 경우: 기존 로직대로 고객 반응 생성
                        # ⭐ 수정: 전화 발신 모드에서도 고객 반응이 생성되도록 보장
                        # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
                        # 🎯 아바타 표정 업데이트 (LLM 기반 영상 RAG)
                        # LLM이 에이전트 응답을 분석하여 고객의 예상 반응(감정)을 판단
                        # 이는 고객이 다음에 말할 때 어떤 비디오를 보여줄지 결정하는 데 사용됩니다.
                        try:
                            # LLM 기반 분석 (에이전트 응답에 대한 고객의 예상 반응)
                            # 에이전트가 "환불"을 언급하면 고객은 기쁠 것이고,
                            # "기다려"를 요청하면 고객은 질문할 것이고,
                            # "불가"를 말하면 고객은 화날 것입니다.
                            # ⭐ Gemini 제안: 에이전트 답변과 대화 컨텍스트를 전달하여 예측 정확도 향상
                            analysis_result = analyze_text_for_video_selection(
                                agent_response_transcript,
                                st.session_state.language,
                                agent_last_response=agent_response_transcript,
                                conversation_context=st.session_state.simulator_messages[-5:] if st.session_state.simulator_messages else None
                            )
                            # 고객의 예상 감정 상태 업데이트 (다음 고객 반응에 사용)
                            predicted_emotion = analysis_result.get("emotion", "NEUTRAL")
                            st.session_state.customer_avatar["state"] = predicted_emotion
                        except Exception as e:
                            # LLM 분석 실패 시 키워드 기반 폴백
                            print(f"LLM 분석 실패, 키워드 기반으로 폴백: {e}")
                            response_text = agent_response_transcript.lower()
                            if "refund" in response_text or "환불" in response_text:
                                st.session_state.customer_avatar["state"] = "HAPPY"
                            elif ("wait" in response_text or "기다려" in response_text or "잠시만" in response_text):
                                st.session_state.customer_avatar["state"] = "ASKING"
                            elif ("no" in response_text or "불가" in response_text or "안 됩니다" in response_text or "cannot" in response_text):
                                st.session_state.customer_avatar["state"] = "ANGRY"
                            else:
                                st.session_state.customer_avatar["state"] = "NEUTRAL"
                        # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

                        # ⭐ 수정: 전사 결과를 CC에 먼저 반영
                        st.session_state.current_agent_audio_text = agent_response_transcript

                        # ⭐ 수정: 전사 결과가 CC에 반영되도록 먼저 재실행
                        # 채팅과 동일하게 전사 결과를 먼저 화면에 표시한 후 고객 반응 생성
                        # 다음 실행 주기에서 고객 반응을 생성하도록 플래그 설정
                        st.session_state.process_customer_reaction = True
                        st.session_state.pending_agent_transcript = agent_response_transcript
                        # ⭐ 수정: 전사 완료 후 즉시 재실행하여 고객 반응 생성 단계로 진행
                        st.rerun()
                # ⭐ 수정: else 블록 제거 (이미 위에서 처리됨)

        # ⭐ 수정: 첫 인사말 후 고객 문의 재생 처리
        # customer_turn_start 플래그가 True일 때 고객 문의를 재생
        if st.session_state.get("customer_turn_start", False) and st.session_state.customer_initial_audio_bytes:
            # ⭐ 수정: 고객 문의 텍스트를 즉시 CC 영역에 반영 (재생 시작 전, 확실히 반영)
            st.session_state.current_customer_audio_text = st.session_state.call_initial_query
            
            # 고객 문의 재생 (비디오와 동기화) - LLM 기반 영상 RAG
            try:
                # 비디오 동기화가 활성화되어 있으면 비디오와 함께 재생
                if st.session_state.is_video_sync_enabled:
                    customer_gender = st.session_state.customer_avatar.get("gender", "male")
                    # ⭐ LLM 기반 텍스트 분석으로 감정/제스처 판단
                    # ⭐ Gemini 제안: 대화 컨텍스트 전달
                    agent_last_msg = None
                    if st.session_state.simulator_messages:
                        for msg in reversed(st.session_state.simulator_messages):
                            if msg.get("role") == "phone_exchange" and "Agent:" in msg.get("content", ""):
                                agent_last_msg = msg.get("content", "").split("Agent:")[-1].strip()
                                break
                    
                    analysis_result = analyze_text_for_video_selection(
                        st.session_state.call_initial_query,
                        st.session_state.language,
                        agent_last_response=agent_last_msg,
                        conversation_context=st.session_state.simulator_messages[-5:] if st.session_state.simulator_messages else None
                    )
                    avatar_state = analysis_result.get("emotion", st.session_state.customer_avatar.get("state", "NEUTRAL"))
                    gesture = analysis_result.get("gesture", "NONE")
                    context_keywords = analysis_result.get("context_keywords", [])  # ⭐ Gemini 제안
                    
                    # 분석 결과를 아바타 상태에 반영
                    st.session_state.customer_avatar["state"] = avatar_state
                    
                    # ⭐ Gemini 제안: 상황별 키워드를 고려한 비디오 선택
                    video_path = get_video_path_by_avatar(
                        customer_gender, 
                        avatar_state, 
                        is_speaking=True,
                        gesture=gesture,
                        context_keywords=context_keywords
                    )
                    
                    if video_path and os.path.exists(video_path):
                        with open(video_path, "rb") as f:
                            video_bytes = f.read()
                        # 비디오와 오디오를 함께 재생
                        st.video(video_bytes, format="video/mp4", autoplay=True, loop=False, muted=False)
                        st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                    else:
                        # 비디오가 없으면 오디오만 재생
                        st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                else:
                    # 비디오 동기화가 비활성화되어 있으면 오디오만 재생
                    st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                
                st.success(L["customer_query_playing"])
                st.info(f"{L['query_content_label']} {st.session_state.call_initial_query}")
                
                # ⭐ 수정: 재생 완료 대기 로직 완전 제거
                # 브라우저에서 자동으로 재생되므로 서버에서 기다릴 필요 없음
                # 재생은 백그라운드에서 계속 진행되며, CC 자막은 이미 반영됨
                
            except Exception as e:
                st.warning(L["auto_play_failed"].format(error=str(e)))
                st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=False)
                st.info(f"{L['query_content_label']} {st.session_state.call_initial_query}")
            
            # 플래그 초기화
            st.session_state.customer_turn_start = False
            
            # ⭐ 수정: 맞춤형 반응 생성을 같은 실행 주기에서 처리하되, 재생은 계속 진행되도록 함
            # 에이전트의 첫 인사말 가져오기
            agent_greeting = ""
            for msg in reversed(st.session_state.simulator_messages):
                if msg.get("role") == "agent":
                    agent_greeting = msg.get("content", "")
                    break
            
            if agent_greeting:
                # 맞춤형 고객 반응 생성 (재생과 동시에 진행)
                with st.spinner(L["generating_customized_response"]):
                    customer_reaction = generate_customer_reaction_for_first_greeting(
                        st.session_state.language,
                        agent_greeting,
                        st.session_state.call_initial_query
                    )
                    
                    # 고객 반응을 TTS로 재생 및 CC에 반영 (비디오와 동기화) - LLM 기반 영상 RAG
                    if not customer_reaction.startswith("❌"):
                        audio_bytes, msg = synthesize_tts(customer_reaction, st.session_state.language, role="customer")
                        if audio_bytes:
                            try:
                                # 비디오 동기화가 활성화되어 있으면 비디오와 함께 재생
                                if st.session_state.is_video_sync_enabled:
                                    customer_gender = st.session_state.customer_avatar.get("gender", "male")
                                    # ⭐ LLM 기반 텍스트 분석으로 감정/제스처 판단
                                    # ⭐ Gemini 제안: 에이전트 답변과 대화 컨텍스트 전달
                                    agent_last_msg = st.session_state.current_agent_audio_text if hasattr(st.session_state, 'current_agent_audio_text') else None
                                    analysis_result = analyze_text_for_video_selection(
                                        customer_reaction,
                                        st.session_state.language,
                                        agent_last_response=agent_last_msg,
                                        conversation_context=st.session_state.simulator_messages[-5:] if st.session_state.simulator_messages else None
                                    )
                                    avatar_state = analysis_result.get("emotion", st.session_state.customer_avatar.get("state", "NEUTRAL"))
                                    gesture = analysis_result.get("gesture", "NONE")
                                    context_keywords = analysis_result.get("context_keywords", [])  # ⭐ Gemini 제안
                                    
                                    # 분석 결과를 아바타 상태에 반영
                                    st.session_state.customer_avatar["state"] = avatar_state
                                    
                                    # ⭐ Gemini 제안: 상황별 키워드를 고려한 비디오 선택
                                    video_path = get_video_path_by_avatar(
                                        customer_gender, 
                                        avatar_state, 
                                        is_speaking=True,
                                        gesture=gesture,
                                        context_keywords=context_keywords
                                    )
                                    
                                    if video_path and os.path.exists(video_path):
                                        with open(video_path, "rb") as f:
                                            video_bytes = f.read()
                                        # 비디오와 오디오를 함께 재생
                                        st.video(video_bytes, format="video/mp4", autoplay=True, loop=False, muted=False)
                                        st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                                        
                                        # ⭐ Gemini 제안: 사용자 피드백 평가 UI 추가
                                        st.markdown("---")
                                        st.markdown("**💬 비디오 매칭 평가**")
                                        st.caption("이 비디오가 고객의 텍스트와 감정에 자연스럽게 매칭되었습니까?")
                                        
                                        feedback_key = f"video_feedback_call_{st.session_state.sim_instance_id}_{len(st.session_state.simulator_messages)}"
                                        
                                        col_rating, col_comment = st.columns([2, 3])
                                        with col_rating:
                                            rating = st.slider(
                                                "평가 점수 (1-5점)",
                                                min_value=1,
                                                max_value=5,
                                                value=3,
                                                key=f"{feedback_key}_rating",
                                                help="1점: 매우 부자연스러움, 5점: 매우 자연스러움"
                                            )
                                        
                                        with col_comment:
                                            comment = st.text_input(
                                                "의견 (선택사항)",
                                                key=f"{feedback_key}_comment",
                                                placeholder="예: 비디오가 텍스트와 잘 맞았습니다"
                                            )
                                        
                                        if st.button("피드백 제출", key=f"{feedback_key}_submit"):
                                            # 피드백을 데이터베이스에 저장
                                            add_video_mapping_feedback(
                                                customer_text=customer_reaction,
                                                selected_video_path=video_path,
                                                emotion=avatar_state,
                                                gesture=gesture,
                                                context_keywords=context_keywords,
                                                user_rating=rating,
                                                user_comment=comment
                                            )
                                            st.success(f"✅ 피드백이 저장되었습니다! (점수: {rating}/5)")
                                            st.info("💡 이 피드백은 향후 비디오 선택 정확도를 개선하는 데 사용됩니다.")
                                    else:
                                        # 비디오가 없으면 오디오만 재생
                                        st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                                else:
                                    # 비디오 동기화가 비활성화되어 있으면 오디오만 재생
                                    st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                                
                                st.success(L["customer_responded"].format(reaction=customer_reaction.strip()[:50] + "..."))
                            except Exception as e:
                                st.warning(L["auto_play_failed"].format(error=str(e)))
                                st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                                st.success(L["customer_responded"].format(reaction=customer_reaction.strip()[:50] + "..."))
                        else:
                            st.error(L["customer_voice_generation_error"].format(error=msg))
                        
                        # ⭐ 수정: 고객 반응을 CC 영역에 추가 (고객 문의는 유지)
                        # 고객 문의와 반응을 모두 표시
                        if st.session_state.current_customer_audio_text == st.session_state.call_initial_query:
                            # 고객 문의만 있는 경우 반응 추가
                            st.session_state.current_customer_audio_text = f"{st.session_state.call_initial_query}\n\n→ {customer_reaction.strip()}"
                        else:
                            # 이미 반응이 있는 경우 업데이트
                            st.session_state.current_customer_audio_text = customer_reaction.strip()
                        
                        # 이력 저장
                        log_entry = f"Agent: {agent_greeting} | Customer: {customer_reaction.strip()}"
                        st.session_state.simulator_messages.append(
                            {"role": "phone_exchange", "content": log_entry})
                    else:
                        st.error(customer_reaction)
            
            # ⭐ 수정: rerun 완전 제거 - 재생은 브라우저에서 자동으로 진행되므로 서버에서 기다릴 필요 없음

        # ⭐ 수정: 전사 후 고객 반응 생성 처리 (마이크 위젯 렌더링 이후에 위치)
        # 전사 결과가 CC에 먼저 표시된 후 고객 반응을 생성하도록 분리
        if st.session_state.get("process_customer_reaction") and st.session_state.get("pending_agent_transcript"):
            pending_transcript = st.session_state.pending_agent_transcript
            # 플래그 초기화
            st.session_state.process_customer_reaction = False
            del st.session_state.pending_agent_transcript

            # ⭐ 수정: 에이전트 응답을 먼저 CC에 반영
            if hasattr(st.session_state, 'current_agent_audio_text'):
                st.session_state.current_agent_audio_text = pending_transcript
            else:
                st.session_state.current_agent_audio_text = pending_transcript

            # 고객 반응 생성
            with st.spinner(L["generating_customer_response"]):
                customer_reaction = generate_customer_reaction_for_call(
                    st.session_state.language,
                    pending_transcript
                )

                # 고객 반응을 TTS로 재생 및 CC에 반영 (비디오와 동기화) - LLM 기반 영상 RAG
                if not customer_reaction.startswith("❌"):
                    audio_bytes, msg = synthesize_tts(customer_reaction, st.session_state.language, role="customer")
                    if audio_bytes:
                        # Streamlit 문서: autoplay는 브라우저 정책상 제한될 수 있음
                        try:
                            # 비디오 동기화가 활성화되어 있으면 비디오와 함께 재생
                            if st.session_state.is_video_sync_enabled:
                                customer_gender = st.session_state.customer_avatar.get("gender", "male")
                                # ⭐ LLM 기반 텍스트 분석으로 감정/제스처 판단
                                # ⭐ Gemini 제안: 에이전트 답변과 대화 컨텍스트 전달
                                agent_last_msg = st.session_state.current_agent_audio_text if hasattr(st.session_state, 'current_agent_audio_text') else None
                                analysis_result = analyze_text_for_video_selection(
                                    customer_reaction,
                                    st.session_state.language,
                                    agent_last_response=agent_last_msg,
                                    conversation_context=st.session_state.simulator_messages[-5:] if st.session_state.simulator_messages else None
                                )
                                avatar_state = analysis_result.get("emotion", st.session_state.customer_avatar.get("state", "NEUTRAL"))
                                gesture = analysis_result.get("gesture", "NONE")
                                context_keywords = analysis_result.get("context_keywords", [])  # ⭐ Gemini 제안
                                
                                # 분석 결과를 아바타 상태에 반영
                                st.session_state.customer_avatar["state"] = avatar_state
                                
                                # ⭐ Gemini 제안: 상황별 키워드를 고려한 비디오 선택
                                video_path = get_video_path_by_avatar(
                                    customer_gender, 
                                    avatar_state, 
                                    is_speaking=True,
                                    gesture=gesture,
                                    context_keywords=context_keywords
                                )
                                
                                if video_path and os.path.exists(video_path):
                                    with open(video_path, "rb") as f:
                                        video_bytes = f.read()
                                    # 비디오와 오디오를 함께 재생
                                    st.video(video_bytes, format="video/mp4", autoplay=True, loop=False, muted=False)
                                    st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                                    
                                    # ⭐ Gemini 제안: 사용자 피드백 평가 UI 추가
                                    st.markdown("---")
                                    st.markdown("**💬 비디오 매칭 평가**")
                                    st.caption("이 비디오가 고객의 텍스트와 감정에 자연스럽게 매칭되었습니까?")
                                    
                                    feedback_key = f"video_feedback_{st.session_state.sim_instance_id}_{len(st.session_state.simulator_messages)}"
                                    
                                    col_rating, col_comment = st.columns([2, 3])
                                    with col_rating:
                                        rating = st.slider(
                                            "평가 점수 (1-5점)",
                                            min_value=1,
                                            max_value=5,
                                            value=3,
                                            key=f"{feedback_key}_rating",
                                            help="1점: 매우 부자연스러움, 5점: 매우 자연스러움"
                                        )
                                    
                                    with col_comment:
                                        comment = st.text_input(
                                            "의견 (선택사항)",
                                            key=f"{feedback_key}_comment",
                                            placeholder="예: 비디오가 텍스트와 잘 맞았습니다"
                                        )
                                    
                                    if st.button("피드백 제출", key=f"{feedback_key}_submit"):
                                        # 피드백을 데이터베이스에 저장
                                        add_video_mapping_feedback(
                                            customer_text=customer_reaction,
                                            selected_video_path=video_path,
                                            emotion=avatar_state,
                                            gesture=gesture,
                                            context_keywords=context_keywords,
                                            user_rating=rating,
                                            user_comment=comment
                                        )
                                        st.success(f"✅ 피드백이 저장되었습니다! (점수: {rating}/5)")
                                        st.info("💡 이 피드백은 향후 비디오 선택 정확도를 개선하는 데 사용됩니다.")
                                else:
                                    # 비디오가 없으면 오디오만 재생
                                    st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                            else:
                                # 비디오 동기화가 비활성화되어 있으면 오디오만 재생
                                st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                            
                            st.success(L["customer_responded"].format(reaction=customer_reaction.strip()[:50] + "..."))
                            # ⭐ 수정: 고객 반응 재생 시간 확보를 위해 짧은 대기
                            time.sleep(0.5)
                        except Exception as e:
                            st.warning(L["auto_play_failed"].format(error=str(e)))
                            st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                            st.success(L["customer_responded"].format(reaction=customer_reaction.strip()[:50] + "..."))
                    else:
                        st.error(L["customer_voice_generation_error"].format(error=msg))

                    # 고객 반응 텍스트를 CC 영역에 반영
                    st.session_state.current_customer_audio_text = customer_reaction.strip()
                    
                    # ⭐ 수정: 고객 반응을 이력에 저장 (전화 발신 모드에서도 작동)
                    agent_response_text = st.session_state.get("current_agent_audio_text", pending_transcript)
                    log_entry = f"Agent: {agent_response_text} | Customer: {customer_reaction.strip()}"
                    st.session_state.simulator_messages.append(
                        {"role": "phone_exchange", "content": log_entry}
                    )

                    # ⭐ 수정: "없습니다. 감사합니다" 응답 처리 - 에이전트가 감사 인사 후 종료
                    if L['customer_no_more_inquiries'] in customer_reaction:
                        # ⭐ 수정: 이력 저장은 이미 위에서 처리되었으므로 중복 저장 방지
                        
                        # ⭐ 추가: 에이전트가 감사 인사 메시지 전송
                        agent_name = st.session_state.get("agent_name", "000")
                        current_lang_call = st.session_state.get("language", "ko")
                        if current_lang_call == "ko":
                            agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                        elif current_lang_call == "en":
                            agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                        else:  # ja
                            agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"
                        
                        st.session_state.simulator_messages.append(
                            {"role": "phone_exchange", "content": f"Agent: {agent_closing_msg}"}
                        )
                        
                        # 통화 요약 생성
                        with st.spinner("AI 요약 생성 중..."):
                            summary = summarize_history_for_call(
                                st.session_state.simulator_messages,
                                st.session_state.call_initial_query,
                                st.session_state.language
                            )
                            st.session_state.call_summary_text = summary
                        
                        # 통화 종료
                        st.session_state.call_sim_stage = "CALL_ENDED"
                        st.session_state.is_call_ended = True
                        
                        # 에이전트 입력 영역 초기화
                        st.session_state.current_agent_audio_text = ""
                        st.session_state.realtime_hint_text = ""
                        if "bytes_to_process" in st.session_state:
                            st.session_state.bytes_to_process = None
                        
                        st.success("✅ 고객이 추가 문의 사항이 없다고 확인했습니다. 에이전트가 감사 인사를 전송한 후 통화가 종료되었습니다.")
                        st.rerun()
                    # ⭐ 추가: "추가 문의 사항도 있습니다" 응답 처리 (통화 계속)
                    elif L['customer_has_additional_inquiries'] in customer_reaction:
                        # ⭐ 수정: 이력 저장은 이미 위에서 처리되었으므로 중복 저장 방지
                        
                        # 에이전트 입력 영역 초기화 (다음 녹음을 위해)
                        st.session_state.current_agent_audio_text = ""
                        st.session_state.realtime_hint_text = ""
                        if "bytes_to_process" in st.session_state:
                            st.session_state.bytes_to_process = None
                        
                        st.info("💡 고객이 추가 문의 사항이 있다고 했습니다. 다음 응답을 녹음하세요.")
                    else:
                        # 일반 고객 반응 처리
                        # ⭐ 수정: 이력 저장은 이미 위에서 처리되었으므로 중복 저장 방지

                        # 에이전트 입력 영역 초기화 (다음 녹음을 위해)
                        st.session_state.current_agent_audio_text = ""
                        st.session_state.realtime_hint_text = ""
                        # ⭐ 최적화: bytes_to_process도 초기화하여 다음 녹음을 준비
                        if "bytes_to_process" in st.session_state:
                            st.session_state.bytes_to_process = None

                    # ⭐ 수정: rerun 제거 - 재생은 브라우저에서 자동으로 진행되므로 서버에서 기다릴 필요 없음
                    # 첫 문의와 동일하게 rerun을 제거하여 재생이 끝까지 진행되도록 함


    # ========================================
    # CALL_ENDED 상태
    # ========================================
    elif st.session_state.call_sim_stage == "CALL_ENDED":
        st.success(L["call_end_message"])

        # AHT
        if st.session_state.start_time is not None:
            final_aht_seconds = max(0, (datetime.now() - st.session_state.start_time).total_seconds())
            final_aht_str = str(timedelta(seconds=final_aht_seconds)).split('.')[0]
            st.metric("Final AHT", final_aht_str)

            hold_str = str(st.session_state.total_hold_duration).split('.')[0]
            st.metric("Total Hold Time", hold_str)
        else:
            st.warning(L["aht_not_recorded"])

        st.markdown("---")

        # ⭐ 추가: 현재 세션 이력 다운로드 기능 (채팅/이메일과 동일)
        st.markdown("**📥 현재 세션 이력 다운로드**")
        download_col1, download_col2, download_col3 = st.columns(3)
        
        # 현재 세션의 이력을 생성
        current_session_history = None
        if st.session_state.simulator_messages:
            try:
                customer_type_display = st.session_state.get("customer_type_sim_select", "")
                # 전화 요약이 있으면 사용, 없으면 생성
                if st.session_state.call_summary_text:
                    # call_summary_text를 summary 형식으로 변환
                    summary_data = {
                        "main_inquiry": st.session_state.call_initial_query,
                        "key_responses": [],
                        "customer_sentiment_score": 50,  # 기본값
                        "customer_satisfaction_score": 50,  # 기본값
                        "customer_characteristics": {},
                        "privacy_info": {},
                        "summary": st.session_state.call_summary_text
                    }
                else:
                    # 요약 생성
                    summary_data = generate_chat_summary(
                        st.session_state.simulator_messages,
                        st.session_state.call_initial_query,
                        customer_type_display,
                        st.session_state.language
                    )
                
                current_session_history = [{
                    "id": f"call_session_{st.session_state.sim_instance_id}",
                    "timestamp": datetime.now().isoformat(),
                    "initial_query": st.session_state.call_initial_query,
                    "customer_type": customer_type_display,
                    "language_key": st.session_state.language,
                    "messages": st.session_state.simulator_messages,
                    "summary": summary_data,
                    "is_chat_ended": True,
                    "attachment_context": st.session_state.get("sim_attachment_context_for_llm", ""),
                    "is_call": True
                }]
            except Exception as e:
                st.warning(f"이력 생성 중 오류 발생: {e}")
        
        # 다운로드 버튼들을 직접 표시
        if current_session_history:
            # 현재 언어 가져오기
            current_lang = st.session_state.get("language", "ko")
            if current_lang not in ["ko", "en", "ja"]:
                current_lang = "ko"
            
            with download_col1:
                try:
                    filepath_word = export_history_to_word(current_session_history, lang=current_lang)
                    with open(filepath_word, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_word", "📥 이력 다운로드 (Word)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_word),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_call_word_file"
                        )
                except Exception as e:
                    st.error(f"Word 다운로드 오류: {e}")
            
            with download_col2:
                try:
                    filepath_pptx = export_history_to_pptx(current_session_history, lang=current_lang)
                    with open(filepath_pptx, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pptx", "📥 이력 다운로드 (PPTX)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pptx),
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key="download_call_pptx_file"
                        )
                except Exception as e:
                    st.error(f"PPTX 다운로드 오류: {e}")
            
            with download_col3:
                try:
                    filepath_pdf = export_history_to_pdf(current_session_history, lang=current_lang)
                    with open(filepath_pdf, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pdf", "📥 이력 다운로드 (PDF)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pdf),
                            mime="application/pdf",
                            key="download_call_pdf_file"
                        )
                except Exception as e:
                    st.error(f"PDF 다운로드 오류: {e}")
        else:
            st.warning("다운로드할 이력이 없습니다.")

        st.markdown("---")

        with st.expander("통화 기록 요약"):
            st.subheader("AI 통화 요약")

            if st.session_state.call_summary_text:
                st.info(st.session_state.call_summary_text)
            else:
                st.error("❌ 통화 요약 생성 실패")

            st.markdown("---")

            st.subheader("고객 최초 문의 (음성)")
            if st.session_state.customer_initial_audio_bytes:
                # Streamlit 문서: bytes 데이터를 직접 전달 가능
                try:
                    st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=False)
                except Exception as e:
                    st.error(f"오디오 재생 오류: {e}")
                st.caption(f"전사: {st.session_state.call_initial_query}")
            else:
                st.info("고객 최초 음성 없음")

            st.markdown("---")
            st.subheader("전체 교환 로그")
            for log in st.session_state.simulator_messages:
                st.write(log["content"])

        # 새 시뮬레이션
        if st.button(L["new_simulation_button"]):
            st.session_state.call_sim_stage = "WAITING_CALL"
            st.session_state.call_sim_mode = "INBOUND"
            st.session_state.is_on_hold = False
            st.session_state.total_hold_duration = timedelta(0)
            st.session_state.hold_start_time = None
            st.session_state.start_time = None
            st.session_state.current_customer_audio_text = ""
            st.session_state.current_agent_audio_text = ""
            st.session_state.agent_response_input_box_widget_call = ""
            st.session_state.call_initial_query = ""
            st.session_state.call_website_url = ""  # 홈페이지 주소 초기화
            st.session_state.simulator_messages = []
            st.session_state.call_summary_text = ""
            st.session_state.customer_initial_audio_bytes = None
            st.session_state.customer_history_summary = ""
            st.session_state.sim_audio_bytes = None


# -------------------- RAG Tab --------------------
elif feature_selection == L["rag_tab"]:
    st.header(L["rag_header"])
    st.markdown(L["rag_desc"])
    st.markdown("---")

    # ⭐ RAG 데이터 학습 기능 추가 - AI 고객 응대 시뮬레이터 데이터를 일일 파일로 학습
    st.subheader("📚 고객 가이드 자동 생성 (일일 학습)")
    
    if st.button("오늘 날짜 고객 가이드 생성", key="generate_daily_guide"):
        # 오늘 날짜로 파일명 생성 (예: 251130_고객가이드.TXT)
        today_str = datetime.now().strftime("%y%m%d")
        guide_filename = f"{today_str}_고객가이드.TXT"
        guide_filepath = os.path.join(DATA_DIR, guide_filename)
        
        # 최근 이력 로드
        all_histories = load_simulation_histories_local(st.session_state.language)
        recent_histories = all_histories[:50]  # 최근 50개 이력 사용
        
        if recent_histories:
            # LLM을 사용하여 고객 가이드 생성
            guide_prompt = f"""
당신은 CS 센터 교육 전문가입니다. 다음 고객 응대 이력 데이터를 분석하여 종합적인 고객 응대 가이드라인을 작성하세요.

분석할 이력 데이터:
{json.dumps([h.get('summary', {}) for h in recent_histories if h.get('summary')], ensure_ascii=False, indent=2)}

다음 내용을 포함하여 가이드라인을 작성하세요:
1. 고객 유형별 응대 전략 (일반/까다로운/매우 불만족)
2. 문화권별 응대 가이드 (언어, 문화적 배경 고려)
3. 주요 문의 유형별 해결 방법
4. 고객 감정 점수에 따른 응대 전략
5. 개인정보 처리 가이드
6. 효과적인 소통 스타일 권장사항

가이드라인을 한국어로 작성하세요.
"""
            
            if st.session_state.is_llm_ready:
                with st.spinner("고객 가이드 생성 중..."):
                    guide_content = run_llm(guide_prompt)
                    
                    # 파일 저장
                    with open(guide_filepath, "w", encoding="utf-8") as f:
                        f.write(f"고객 응대 가이드라인\n")
                        f.write(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"분석 이력 수: {len(recent_histories)}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(guide_content)
                    
                    st.success(f"✅ 고객 가이드가 생성되었습니다: {guide_filename}")
                    st.info(f"파일 위치: {guide_filepath}")
                    
                    # 생성된 파일을 자동으로 RAG에 추가할지 선택
                    if st.button("생성된 가이드를 RAG에 추가", key="add_guide_to_rag"):
                        # 파일을 업로드된 파일처럼 처리하여 RAG에 추가
                        st.info("RAG 인덱스 업데이트 중...")
                        # 실제로는 파일을 읽어서 RAG 인덱스에 추가하는 로직 필요
            else:
                st.error("LLM이 준비되지 않았습니다. API Key를 설정해주세요.")
        else:
            st.warning("분석할 이력이 없습니다. 먼저 고객 응대 시뮬레이션을 실행하세요.")
    
    st.markdown("---")

    # --- 파일 업로드 섹션 ---
    # ⭐ 수정된 부분: RAG 탭 전용 키 사용
    uploaded_files = st.file_uploader(
        L["file_uploader"],
        type=["pdf", "txt", "html"],
        key="rag_file_uploader", # RAG 전용 키
        accept_multiple_files=True
    )

    if uploaded_files:
        if uploaded_files != st.session_state.uploaded_files_state:
            # 파일이 변경되면 RAG 상태 초기화
            st.session_state.is_rag_ready = False
            st.session_state.rag_vectorstore = None
            st.session_state.uploaded_files_state = uploaded_files

        if not st.session_state.is_rag_ready:
            if st.button(L["button_start_analysis"]):
                if not st.session_state.is_llm_ready:
                    st.error(L["simulation_no_key_warning"])
                else:
                    with st.spinner(L["data_analysis_progress"]):
                        vectorstore, count = build_rag_index(uploaded_files)

                    if vectorstore:
                        st.session_state.rag_vectorstore = vectorstore
                        st.session_state.is_rag_ready = True
                        st.success(L["embed_success"].format(count=count))
                        st.session_state.rag_messages = [
                            {"role": "assistant", "content": f"✅ {len(uploaded_files)}개 파일 분석 완료. 질문해 주세요."}
                        ]
                    else:
                        st.error(L["embed_fail"])
                        st.session_state.is_rag_ready = False
    else:
        st.info(L["warning_no_files"])
        st.session_state.is_rag_ready = False
        st.session_state.rag_vectorstore = None
        st.session_state.rag_messages = []

    st.markdown("---")

    # --- 챗봇 섹션 ---
    if st.session_state.is_rag_ready and st.session_state.rag_vectorstore:
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = [{"role": "assistant", "content": "분석된 자료에 대해 질문해 주세요."}]

        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(L["rag_input_placeholder"]):
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(L["response_generating"]):
                    response = rag_answer(
                        prompt,
                        st.session_state.rag_vectorstore,
                        st.session_state.language
                    )
                    st.markdown(response)

            st.session_state.rag_messages.append({"role": "assistant", "content": response})
    else:
        st.warning(L["warning_rag_not_ready"])

# -------------------- Content Tab --------------------
elif feature_selection == L["content_tab"]:
    st.header(L["content_header"])
    st.markdown(L["content_desc"])
    st.markdown("---")

    if not st.session_state.is_llm_ready:
        st.warning(L["simulation_no_key_warning"])
        st.info("💡 API Key를 설정하면 콘텐츠 생성 기능을 사용할 수 있습니다.")
        # st.stop() 제거: UI는 표시하되 기능만 비활성화

    # 다국어 맵핑 변수는 그대로 사용
    level_map = {
        "초급": "Beginner",
        "중급": "Intermediate",
        "고급": "Advanced",
        "Beginner": "Beginner",
        "Intermediate": "Intermediate",
        "Advanced": "Advanced",
        "初級": "Beginner",
        "中級": "Intermediate",
        "上級": "Advanced",
    }
    content_map = {
        "핵심 요약 노트": "summary",
        "객관식 퀴즈 10문항": "quiz",
        "실습 예제 아이디어": "example",
        "Key Summary Note": "summary",
        "10 MCQ Questions": "quiz",
        "Practical Example Idea": "example",
        "核心要約ノート": "summary",
        "選択式クイズ10問": "quiz",
        "実践例のアイデア": "example",
    }

    topic = st.text_input(L["topic_label"])
    level_display = st.selectbox(L["level_label"], L["level_options"])
    content_display = st.selectbox(L["content_type_label"], L["content_options"])

    level = level_map.get(level_display, "Beginner")
    content_type = content_map.get(content_display, "summary")

    if st.button(L["button_generate"]):
        if not topic.strip():
            st.warning(L["warning_topic"])
            # st.stop() 제거: 경고만 표시하고 계속 진행
        elif not st.session_state.is_llm_ready:
            st.error("❌ LLM이 준비되지 않았습니다. API Key를 설정해주세요.")
            # st.stop() 제거: 에러만 표시하고 계속 진행
        else:
            target_lang = {"ko": "Korean", "en": "English", "ja": "Japanese"}[st.session_state.language]

            # 공통 프롬프트 설정 (퀴즈 형식을 포함하지 않는 기본 템플릿)
            system_prompt = f"""
            You are a professional AI coach. Generate learning content in {target_lang} for the topic '{topic}' at the '{level}' difficulty.
            The content format requested is: {content_display}.
            Output ONLY the raw content.
            """

            if content_type == "quiz":
                # 퀴즈 전용 프롬프트 및 JSON 구조 강제 (로직 유지)
                lang_instruction = {"ko": "한국어로", "en": "in English", "ja": "日本語で"}.get(st.session_state.language, "in Korean")
                quiz_prompt = f"""
                You are an expert quiz generator. Based on the topic '{topic}' and difficulty '{level}', generate 10 multiple-choice questions.
                IMPORTANT: All questions, options, and explanations must be written {lang_instruction}.
                Your output MUST be a **raw JSON object** containing a single key "quiz_questions" which holds an array of 10 questions.
                Each object in the array must strictly follow the required keys: 
                - "question" (string): The question text in {lang_instruction}
                - "options" (array of 4 strings): Four answer choices in {lang_instruction}
                - "answer" (integer): The correct answer index starting from 1 (1-4)
                - "explanation" (string): A DETAILED and COMPREHENSIVE explanation (at least 2-3 sentences, preferably 50-100 words) explaining:
                  * Why the correct answer is right
                  * Why other options are incorrect (briefly mention key differences)
                  * Additional context or background information that helps understanding
                  * Real-world examples or applications if relevant
                  Write the explanation in {lang_instruction} with clear, educational content.
                DO NOT include any explanation, introductory text, or markdown code blocks (e.g., ```json).
                Output ONLY the raw JSON object, starting with '{{' and ending with '}}'.
                Example structure:
                {{
                  "quiz_questions": [
                    {{
                      "question": "질문 내용",
                      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
                      "answer": 1,
                      "explanation": "정답인 이유를 상세히 설명하고, 다른 선택지가 왜 틀렸는지 간단히 언급하며, 관련 배경 지식이나 실제 사례를 포함한 충분히 긴 해설 내용 (최소 2-3문장, 50-100단어 정도)"
                    }}
                  ]
                }}
            def extract_json_from_text(text):
                """텍스트에서 JSON 객체를 추출하는 함수"""
                if not text:
                    return None
                
                text = text.strip()
                
                # 1. Markdown 코드 블록 제거
                if "```json" in text:
                    start = text.find("```json") + 7
                    end = text.find("```", start)
                    if end != -1:
                        text = text[start:end].strip()
                elif "```" in text:
                    start = text.find("```") + 3
                    end = text.find("```", start)
                    if end != -1:
                        text = text[start:end].strip()
                
                # 2. 첫 번째 '{' 부터 마지막 '}' 까지 추출
                first_brace = text.find('{')
                if first_brace == -1:
                    return None
                
                # 중괄호 매칭으로 JSON 객체 끝 찾기
                brace_count = 0
                last_brace = -1
                for i in range(first_brace, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_brace = i
                            break
                
                if last_brace != -1:
                    json_str = text[first_brace:last_brace + 1]
                    return json_str.strip()
                
                return None

            generated_json_text = None
            raw_response_text = None
            llm_attempts = []

            # 1순위: OpenAI (JSON mode가 가장 안정적)
            if get_api_key("openai"):
                llm_attempts.append(("openai", get_api_key("openai"), "gpt-4o"))
            # 2순위: Gemini (Fallback)
            if get_api_key("gemini"):
                llm_attempts.append(("gemini", get_api_key("gemini"), "gemini-2.5-flash"))

            with st.spinner(L["response_generating"]):
                for provider, api_key, model_name in llm_attempts:
                    try:
                        if provider == "openai":
                            client = OpenAI(api_key=api_key)
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": quiz_prompt}],
                                # JSON Mode 강제
                                response_format={"type": "json_object"},
                            )
                            raw_response_text = response.choices[0].message.content.strip()
                            # OpenAI는 JSON 객체를 반환하므로, 직접 사용 시도
                            generated_json_text = extract_json_from_text(raw_response_text) or raw_response_text
                            break

                        elif provider == "gemini":
                            # Gemini는 response_format을 지원하지 않으므로, run_llm을 통해 일반 텍스트로 호출
                            raw_response_text = run_llm(quiz_prompt)
                            generated_json_text = extract_json_from_text(raw_response_text)
                            
                            # JSON 추출 성공 시 시도 종료
                            if generated_json_text:
                                break

                    except Exception as e:
                        print(f"JSON generation failed with {provider}: {e}")
                        continue

            # --- START: JSON Parsing and Error Handling Logic ---
            parsed_obj = None
            quiz_data = None
            
            if generated_json_text:
                try:
                    # JSON 객체 파싱 시도
                    parsed_obj = json.loads(generated_json_text)

                    # 'quiz_questions' 키에서 배열 추출
                    quiz_data = parsed_obj.get("quiz_questions")

                    if not isinstance(quiz_data, list) or len(quiz_data) < 1:
                        raise ValueError("Missing 'quiz_questions' key or empty array.")

                    # 데이터 유효성 검사: 각 문제에 필수 필드가 있는지 확인
                    for i, q in enumerate(quiz_data):
                        if not isinstance(q, dict):
                            raise ValueError(f"Question {i+1} is not a valid object.")
                        if "question" not in q or "options" not in q or "answer" not in q:
                            raise ValueError(f"Question {i+1} is missing required fields (question, options, or answer).")
                        if not isinstance(q["options"], list) or len(q["options"]) != 4:
                            raise ValueError(f"Question {i+1} must have exactly 4 options.")
                        if not isinstance(q["answer"], int) or q["answer"] < 1 or q["answer"] > 4:
                            raise ValueError(f"Question {i+1} answer must be an integer between 1 and 4.")

                    # 파싱 성공 및 데이터 유효성 검사 후 상태 저장
                    st.session_state.quiz_data = quiz_data
                    st.session_state.current_question_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answers = [1] * len(quiz_data)
                    st.session_state.show_explanation = False
                    st.session_state.is_quiz_active = True
                    st.session_state.quiz_type_key = str(uuid.uuid4())

                    st.success(f"**{topic}** - {content_display} 생성 완료")

                except json.JSONDecodeError as e:
                    # JSON 파싱 오류
                    st.error(L["quiz_error_llm"])
                    st.caption(f"JSON 파싱 오류: {str(e)}")
                    st.subheader(L["quiz_original_response"])
                    st.code(raw_response_text or generated_json_text, language="text")
                    if generated_json_text:
                        st.caption("추출된 JSON 텍스트:")
                        st.code(generated_json_text, language="text")
                    
                except ValueError as e:
                    # 데이터 구조 오류
                    st.error(L["quiz_error_llm"])
                    st.caption(f"데이터 구조 오류: {str(e)}")
                    st.subheader(L["quiz_original_response"])
                    st.code(raw_response_text or generated_json_text, language="text")
                    if parsed_obj:
                        st.caption("파싱된 객체:")
                        st.json(parsed_obj)
                        
            else:
                # JSON 추출 실패
                st.error(L["quiz_error_llm"])
                st.caption("LLM 응답에서 JSON 객체를 찾을 수 없습니다.")
                if raw_response_text:
                    st.subheader(L["quiz_original_response"])
                    st.text_area("", raw_response_text, height=300)
                elif generated_json_text:
                    st.subheader(L["quiz_original_response"])
                    st.text_area("", generated_json_text, height=300)
                # --- END: JSON Parsing and Error Handling Logic ---

                else:  # 일반 텍스트 생성
                    st.session_state.is_quiz_active = False
                with st.spinner(L["response_generating"]):
                    content = run_llm(system_prompt)
                st.session_state.generated_content = content

                st.markdown("---")
                st.markdown(f"### {content_display}")
                st.markdown(st.session_state.generated_content)

    # --- 퀴즈/일반 콘텐츠 출력 로직 ---
    if st.session_state.get("is_quiz_active", False) and st.session_state.get("quiz_data"):
        # 퀴즈 진행 로직 (생략 - 기존 로직 유지)
        quiz_data = st.session_state.quiz_data
        idx = st.session_state.current_question_index

        # ⭐ 퀴즈 완료 시 IndexError 방지 로직 (idx >= len(quiz_data))
        if idx >= len(quiz_data):
            # 퀴즈 완료 시 최종 점수 표시
            st.success(L["quiz_complete"])
            total_questions = len(quiz_data)
            score = st.session_state.quiz_score
            incorrect_count = total_questions - score
            st.subheader(f"{L['score']}: {score} / {total_questions} ({(score / total_questions) * 100:.1f}%)")

            # 원형 차트로 맞은 문제/틀린 문제 표시
            if IS_PLOTLY_AVAILABLE:
                col1, col2 = st.columns([1, 2])
                with col1:
                    # 원형 차트 생성
                    fig = go.Figure(data=[go.Pie(
                        labels=[L["correct_questions"], L["incorrect_questions"]],
                        values=[score, incorrect_count],
                        hole=0.4,
                        marker_colors=['#28a745', '#dc3545'],
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    fig.update_layout(
                        title=L["question_result"],
                        showlegend=True,
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### " + L["question_result"])
                    # 문제별 정오 리스트 표시
                    for i, question_item in enumerate(quiz_data):
                        user_answer = st.session_state.quiz_answers[i] if i < len(st.session_state.quiz_answers) else None
                        is_correct = user_answer == 'Correctly Scored'
                        correct_answer_idx = question_item.get('answer', 1)
                        correct_answer_text = question_item['options'][correct_answer_idx - 1] if 0 < correct_answer_idx <= len(question_item['options']) else "N/A"
                        
                        # 사용자 답안 텍스트 가져오기
                        if is_correct:
                            user_answer_text = correct_answer_text
                            status_icon = "✅"
                            status_color = "green"
                        else:
                            if isinstance(user_answer, int) and 0 < user_answer <= len(question_item['options']):
                                user_answer_text = question_item['options'][user_answer - 1]
                            else:
                                user_answer_text = "미응답"
                            status_icon = "❌"
                            status_color = "red"
                        
                        # 문제별 결과 표시
                        with st.container():
                            st.markdown(f"""
                            <div style="border-left: 4px solid {status_color}; padding-left: 10px; margin-bottom: 15px;">
                                <strong>{status_icon} 문항 {i+1}:</strong> {question_item['question']}<br>
                                <span style="color: {status_color};">{L['your_answer']}: {user_answer_text}</span><br>
                                <span style="color: green;">{L['correct_answer_label']}: {correct_answer_text}</span>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                # Plotly가 없는 경우 텍스트로만 표시
                st.markdown(f"**{L['correct_questions']}:** {score}개")
                st.markdown(f"**{L['incorrect_questions']}:** {incorrect_count}개")
                st.markdown("### " + L["question_result"])
                for i, question_item in enumerate(quiz_data):
                    user_answer = st.session_state.quiz_answers[i] if i < len(st.session_state.quiz_answers) else None
                    is_correct = user_answer == 'Correctly Scored'
                    correct_answer_idx = question_item.get('answer', 1)
                    correct_answer_text = question_item['options'][correct_answer_idx - 1] if 0 < correct_answer_idx <= len(question_item['options']) else "N/A"
                    
                    if is_correct:
                        user_answer_text = correct_answer_text
                        status_icon = "✅"
                    else:
                        if isinstance(user_answer, int) and 0 < user_answer <= len(question_item['options']):
                            user_answer_text = question_item['options'][user_answer - 1]
                        else:
                            user_answer_text = "미응답"
                        status_icon = "❌"
                    
                    st.markdown(f"**{status_icon} 문항 {i+1}:** {question_item['question']}")
                    st.markdown(f"- {L['your_answer']}: {user_answer_text}")
                    st.markdown(f"- {L['correct_answer_label']}: {correct_answer_text}")
                    st.markdown("---")

            if st.button(L["retake_quiz"], key="retake_quiz_btn"):
                # 퀴즈 상태만 초기화 (퀴즈 데이터는 유지하여 같은 퀴즈를 다시 풀 수 있도록)
                st.session_state.current_question_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = [1] * len(quiz_data)  # 기본값으로 초기화
                st.session_state.show_explanation = False
                st.rerun()  # 페이지 새로고침하여 첫 번째 문제로 이동
            # st.stop() 제거: 퀴즈 완료 후에도 UI는 계속 표시
        else:
            # 퀴즈 진행 (현재 문항)
            question_data = quiz_data[idx]
            st.subheader(f"{L.get('question_label', '문항')} {idx + 1}/{len(quiz_data)}")
            st.markdown(f"**{question_data['question']}**")

            # 기존 퀴즈 진행 및 채점 로직 (변화 없음)
            current_selection_index = st.session_state.quiz_answers[idx]

            options = question_data['options']
            current_answer = st.session_state.quiz_answers[idx]

            if current_answer is None or not isinstance(current_answer, int) or current_answer <= 0:
                radio_index = 0
            else:
                radio_index = min(current_answer - 1, len(options) - 1)

            selected_option = st.radio(
                L["select_answer"],
                options,
                index=radio_index,
                key=f"quiz_radio_{st.session_state.quiz_type_key}_{idx}"
            )

            selected_option_index = options.index(selected_option) + 1 if selected_option in options else None

            check_col, next_col = st.columns([1, 1])

            if check_col.button(L["check_answer"], key=f"check_answer_btn_{idx}"):
                if selected_option_index is None:
                    st.warning("선택지를 선택해 주세요.")
                else:
                    # 점수 계산 로직
                    if st.session_state.quiz_answers[idx] != 'Correctly Scored':
                        correct_answer = question_data.get('answer')  # answer 키가 없을 경우 대비
                        if selected_option_index == correct_answer:
                            st.session_state.quiz_score += 1
                            st.session_state.quiz_answers[idx] = 'Correctly Scored'
                            st.success(L["correct_answer"])
                        else:
                            st.session_state.quiz_answers[idx] = selected_option_index  # 오답은 선택지 인덱스 저장
                            st.error(L["incorrect_answer"])

                    st.session_state.show_explanation = True

            # 정답 및 해설 표시
            if st.session_state.show_explanation:
                correct_index = question_data.get('answer', 1)
                correct_answer_text = question_data['options'][correct_index - 1] if 0 < correct_index <= len(
                    question_data['options']) else "N/A"

                st.markdown("---")
                st.markdown(f"**{L['correct_is']}:** {correct_answer_text}")
                with st.expander(f"**{L['explanation']}**", expanded=True):
                    st.info(question_data.get('explanation', '해설이 제공되지 않았습니다.'))

                # 다음 문항 버튼
                if next_col.button(L["next_question"], key=f"next_question_btn_{idx}"):
                    st.session_state.current_question_index += 1
                    st.session_state.show_explanation = False

            else:
                # 사용자가 이미 정답을 체크했고 (다시 로드된 경우), 다음 버튼을 바로 표시
                if st.session_state.quiz_answers[idx] == 'Correctly Scored' or (
                        isinstance(st.session_state.quiz_answers[idx], int) and st.session_state.quiz_answers[idx] > 0):
                    if next_col.button(L["next_question"], key=f"next_question_btn_after_check_{idx}"):
                        st.session_state.current_question_index += 1
                        st.session_state.show_explanation = False

    else:
        # 일반 콘텐츠 (핵심 요약 노트, 실습 예제 아이디어) 출력
        if st.session_state.get("generated_content"):
            content = st.session_state.generated_content  # Content를 다시 가져옴
            content_lines = content.split('\n')

            st.markdown("---")
            st.markdown(f"### {content_display}")

            # --- START: 효율성 개선 (상단 분석/하단 본문) ---

            st.subheader("💡 콘텐츠 분석 (Plotly 시각화)")

            if IS_PLOTLY_AVAILABLE:
                # 1. 키워드 빈도 시각화 (모의 데이터)

                # 콘텐츠를 텍스트 줄로 분할하여 모의 키워드 및 주요 문장 생성
                content = st.session_state.generated_content
                content_lines = content.split('\n')
                all_words = ' '.join(content_lines).replace('.', '').replace(',', '').split()

                # 모의 키워드 빈도 데이터 생성
                words = ['AI', '기술혁신', '고객경험', '데이터분석', '효율성', '여행산업']
                np.random.seed(42)
                counts = np.random.randint(5, 30, size=len(words))

                # 난이도에 따라 점수 가중치 (모의 감성 점수 변화)
                difficulty_score = {'Beginner': 60, 'Intermediate': 75, 'Advanced': 90}.get(level, 70)

                # --- 차트 1: 키워드 빈도 (Plotly Bar Chart) ---
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=words,
                        y=counts,
                        marker_color=px.colors.sequential.Plotly3,
                        name="키워드 빈도"
                    )
                ])
                fig_bar.update_layout(
                    title_text=f"주요 키워드 빈도 분석",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # --- 차트 2: 콘텐츠 감성 및 복잡도 추이 (Plotly Line Chart) ---
                # 모의 감성/복잡도 점수 추이 (5개 문단 모의)
                sections = ['도입부', '핵심1', '핵심2', '해결책', '결론']
                sentiment_scores = [difficulty_score - 10, difficulty_score + 5, difficulty_score,
                                    difficulty_score + 10, difficulty_score + 2]

                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=sections,
                    y=sentiment_scores,
                    mode='lines+markers',
                    name='감성/복잡도 점수',
                    line=dict(color='orange', width=2),
                    marker=dict(size=8)
                ))
                fig_line.update_layout(
                    title_text="콘텐츠 섹션별 감성 및 복잡도 추이 (모의)",
                    yaxis_range=[50, 100],
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_line, use_container_width=True)

            else:  # Plotly가 없을 경우 기존 텍스트 분석 모의 유지
                st.info("Plotly 라이브러리가 없어 시각화를 표시할 수 없습니다. 텍스트 분석 모의를 표시합니다.")
                all_words = ' '.join(content_lines).replace('.', '').replace(',', '').split()
                unique_words = sorted(set(all_words), key=len, reverse=True)[:5] if all_words else ["N/A"]
                key_sentences = [
                    content_lines[0].strip() if content_lines else "N/A",
                    content_lines[len(content_lines) // 2].strip() if len(content_lines) > 1 else "",
                    content_lines[-1].strip() if len(content_lines) > 1 else ""
                ]
                key_sentences = [s for s in key_sentences if s and s != "N/A"]

                col_keyword, col_sentences = st.columns([1, 1])

                with col_keyword:
                    st.markdown("**핵심 키워드/개념 (모의)**")
                    st.info(f"[{', '.join(unique_words)}...]")

                with col_sentences:
                    st.markdown("**주요 문장 요약 (모의)**")
                    for sentence in key_sentences[:2]:
                        st.write(f"• {sentence[:50]}...")

            st.markdown("---")

            # 2. 하단 본문 출력
            st.markdown(f"### 📝 원본 콘텐츠")
            st.markdown(content)

            # --- END: 효율성 개선 ---

            # --- START: 아이콘 버튼 활성화 ---
            st.markdown("---")

            # 1. 복사할 내용 정리 및 이스케이프
            content_for_js = json.dumps(content)

            # JavaScript 코드는 이스케이프된 중괄호 {{}}를 사용
            js_copy_script = """
               function copyToClipboard(text) {{
                   navigator.clipboard.writeText(text).then(function() {{
                       // Streamlit toast 호출 (모의)
                       const elements = window.parent.document.querySelectorAll('[data-testid="stToast"]');
                       if (elements.length === 0) {{
                           // Fallback UI update (use Streamlit's native mechanism if possible, or simple alert)
                           console.log("복사 완료: " + text.substring(0, 50) + "...");
                           }}
                       }}, function(err) {{
                           // Fallback: Copy via execCommand (deprecated but often works in Streamlit's iframe)
                           const textarea = document.createElement('textarea');
                           textarea.value = text;
                           document.body.appendChild(textarea);
                           textarea.select();
                           document.execCommand('copy');
                           document.body.removeChild(textarea);
                           alert("복사 완료!"); 
                       }});
                   }}
                   // f-string 대신 .format을 사용하여 JavaScript 코드에 주입
                   // content_for_js는 이미 Python에서 JSON 문자열로 안전하게 이스케이프됨
                   copyToClipboard(JSON.parse('{content_json_safe}'));
               """.format(content_json_safe=content_for_js)

            # --- JavaScript for SHARE Menu (Messenger Mock) ---
            # Streamlit은 현재 소셜 미디어 API를 직접 호출할 수 없으므로, URL 복사를 사용하고 UI에 메시지 옵션을 모의합니다.
            js_share_url_copy = """
               function copyShareUrl() {{
                   const url = window.location.href;
                   navigator.clipboard.writeText(url).then(function() {{
                       console.log('App URL copied');
                   }}, function(err) {{
                       // Fallback
                       const textarea = document.createElement('textarea');
                       textarea.value = url;
                       document.body.appendChild(textarea);
                       textarea.select();
                       document.execCommand('copy');
                       document.body.removeChild(textarea);
                   }});
               }}
            """

            # --- JavaScript for SHARE Menu (Messenger Mock) ---
            # Streamlit은 현재 소셜 미디어 API를 직접 호출할 수 없으므로, URL 복사를 사용하고 UI에 메시지 옵션을 모의합니다.
            js_native_share = """
               function triggerNativeShare(title, text, url) {{
                   if (navigator.share) {{
                       // 1. 네이티브 공유 API 지원 시 사용
                       navigator.share({{
                           title: title,
                           text: text,
                           url: url,
                       }}).then(() => {{
                           console.log('Successful share');
                       }}).catch((error) => {{
                           console.log('Error sharing', error);
                       }});
                       return true;
                   }} else {{
                      // 2. 네이티브 공유 API 미지원 시 (PC 환경 등)
                      return false;
                   }}
               }}
            def mock_download(file_type: str, file_name: str):
                """모의 다운로드 기능: 파일명과 함께 성공 토스트 메시지를 출력합니다."""
                st.toast(f"📥 {file_type} 파일을 생성하여 다운로드를 시작합니다: {file_name}")
                # 실제 다운로드 로직은 Streamlit 컴포넌트 환경에서는 복잡하여 생략합니다.


            col_like, col_dislike, col_share, col_copy, col_more = st.columns([1, 1, 1, 1, 6])
            current_content_id = str(uuid.uuid4())  # 동적 ID 생성

            # 1. 좋아요 버튼 (기능 활성화)
            if col_like.button("👍", key=f"content_like_{current_content_id}"):
                st.toast(L["toast_like"])

            # 2. 싫어요 버튼 (기능 활성화)
            if col_dislike.button("👎", key=f"content_dislike_{current_content_id}"):
                st.toast(L["toast_dislike"])

            # 3. 공유 버튼 (Web Share API 호출 통합)
            with col_share:
                share_clicked = st.button("🔗", key=f"content_share_{current_content_id}")

            if share_clicked:
                # 1단계: 네이티브 공유 API 호출 시도 (모바일 환경 대상)
                share_title = f"{content_display} ({topic})"
                share_text = content[:150] + "..."
                share_url = "https://utility-convenience-salmonyeonwoo.streamlit.app/"  # 실제 배포 URL로 가정

                # JavaScript 실행: 네이티브 공유 호출
                st.components.v1.html(
                    f"""
                    <script>{js_native_share}
                        const shared = triggerNativeShare('{share_title}', '{share_text}', '{share_url}');
                        if (shared) {{
                           // 네이티브 공유 성공 시 (토스트 메시지는 브라우저가 관리)
                            console.log("Native Share Attempted.");
                        }} else {{
                           // 네이티브 공유 미지원 시, 대신 URL 복사
                           const url = window.location.href;
                           const textarea = document.createElement('textarea');
                           textarea.value = url;
                           document.body.appendChild(textarea);
                           textarea.select();
                           document.execCommand('copy');
                           document.body.removeChild(textarea);
                           // PC 환경에서 URL 복사 완료 토스트 메시지 출력
                           const toastElement = window.parent.document.querySelector('[data-testid="stToast"]');
                           if (toastElement) {{
                               // 이미 토스트 메시지가 열려 있다면 갱신 (Streamlit의 toast 기능을 가정)
                           }} else {{
                              alert('URL이 클립보드에 복사되었습니다.');
                           }}
                        }}
                    </script>
                    """,
                    height=0,
                )

                # Streamlit의 toast 메시지는 네이티브 공유 성공 여부를 알 수 없으므로 URL 복사 완료를 알림
                st.toast(L["toast_share"])


            # 4. 복사 버튼 (기능 활성화 - 콘텐츠 텍스트 복사)
            if col_copy.button("📋", key=f"content_copy_{current_content_id}"):
                # JavaScript를 실행하여 복사 (execCommand 사용으로 안정화)
                st.components.v1.html(
