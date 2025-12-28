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
streamlit_app.py의 모든 import 문을 관리하는 모듈
"""

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
        import streamlit as st
        st.error("채팅 시뮬레이터 모듈을 찾을 수 없습니다.")

# 회사 정보 탭 모듈
try:
    from _pages._company_info import render_company_info
    COMPANY_INFO_AVAILABLE = True
except ImportError:
    COMPANY_INFO_AVAILABLE = False
    def render_company_info():
        import streamlit as st
        st.error("회사 정보 탭 모듈을 찾을 수 없습니다.")

# 전화 시뮬레이터 탭 모듈
try:
    from _pages._phone_simulator import render_phone_simulator
    PHONE_SIMULATOR_AVAILABLE = True
except ImportError:
    PHONE_SIMULATOR_AVAILABLE = False
    def render_phone_simulator():
        import streamlit as st
        st.error("전화 시뮬레이터 탭 모듈을 찾을 수 없습니다.")

# RAG 탭 모듈
try:
    from _pages._rag_page import render_rag_page as render_rag
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    def render_rag():
        import streamlit as st
        st.error("RAG 탭 모듈을 찾을 수 없습니다.")

# 콘텐츠 탭 모듈
try:
    from _pages._content import render_content
    CONTENT_AVAILABLE = True
except ImportError:
    CONTENT_AVAILABLE = False
    def render_content():
        import streamlit as st
        st.error("콘텐츠 탭 모듈을 찾을 수 없습니다.")

# 사이드바 모듈
try:
    from ui.sidebar import render_sidebar
    SIDEBAR_AVAILABLE = True
except ImportError:
    SIDEBAR_AVAILABLE = False
    def render_sidebar():
        import streamlit as st
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
    from customer_data_manager import CustomerDataManager
    CUSTOMER_DATA_AVAILABLE = True
except ImportError:
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





