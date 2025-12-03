# ========================================
# streamlit_app.py (전체 수정된 코드)
#
# 주요 개선 사항:
# 1. 채팅/이메일 탭에 '전화 발신 (현지 업체/고객)' 버튼 및 기능 추가 (예외 처리 대응)
# 2. 전화 탭에 '전화 발신' 버튼 추가 및 발신 통화 시뮬레이션 모드 지원
# 3. 관련 언어 팩 추가 및 세션 상태 업데이트
# 4. 퀴즈 기능의 정답 확인, 해설, 점수 표시 로직 완성
# 5. [BUG FIX] 언어 이관 시 '번역 다시 시도' 버튼의 DuplicateWidgetID 오류 해결
# 6. [BUG FIX] 콘텐츠 생성 탭의 LLM 응답 및 라디오 버튼 초기화 오류 해결
# ⭐ [전화 아바타 버그 수정]
# 7. 전화 응답 후 인사말 미출력 오류 수정 (just_entered_call 플래그 위치 수정)
# 8. 아바타 Lottie 파일 로딩 경로 수정 (업로드된 파일명 참조)
# ========================================

import os
import io
import json
import time
import uuid
import base64
import tempfile
import hashlib
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union
from typing import List, Dict, Any
import google.generativeai as genai
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import requests  # ⭐ 추가: requests 라이브러리 필요

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False

from openai import OpenAI
from anthropic import Anthropic

# mic_recorder (0.0.8) - returns dict with key "bytes"
from streamlit_mic_recorder import mic_recorder

# LangChain / RAG 관련
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError(
        "❌ 'langchain-text-splitters' 패키지가 설치되지 않았습니다.\n"
        "다음 명령어로 설치해주세요: pip install langchain-text-splitters\n"
        "또는 requirements.txt의 모든 패키지를 설치: pip install -r requirements.txt"
    )
from langchain_core.prompts import PromptTemplate
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    raise ImportError(
        "❌ 'langchain' 패키지가 설치되지 않았거나 'langchain.memory' 모듈을 찾을 수 없습니다.\n"
        "다음 명령어로 설치해주세요: pip install langchain\n"
        "또는 requirements.txt의 모든 패키지를 설치: pip install -r requirements.txt"
    )
try:
    from langchain.chains import ConversationChain
except ImportError:
    raise ImportError(
        "❌ 'langchain' 패키지가 설치되지 않았거나 'langchain.chains' 모듈을 찾을 수 없습니다.\n"
        "다음 명령어로 설치해주세요: pip install langchain\n"
        "또는 requirements.txt의 모든 패키지를 설치: pip install -r requirements.txt"
    )

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Word, PPTX, PDF 생성을 위한 라이브러리
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
# Streamlit 페이지 설정 (반드시 최상단에 위치)
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "local_db")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
RAG_INDEX_DIR = os.path.join(DATA_DIR, "rag_index")

VOICE_META_FILE = os.path.join(DATA_DIR, "voice_records.json")
SIM_META_FILE = os.path.join(DATA_DIR, "simulation_histories.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RAG_INDEX_DIR, exist_ok=True)




# ----------------------------------------
# JSON Helper
# ----------------------------------------
def _load_json(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)





# ========================================
# 1. 다국어 설정 (전화 발신 관련 텍스트 추가)
# ========================================
DEFAULT_LANG = "ko"

LANG: Dict[str, Dict[str, str]] = {
    "ko": {
        "title": "개인 맞춤형 AI 학습 코치 (음성 및 DB 통합)",
        "sidebar_title": "📚 AI Study Coach 설정",
        "file_uploader": "학습 자료 업로드 (PDF, TXT, HTML)",
        "button_start_analysis": "자료 분석 시작 (RAG Indexing)",
        "rag_tab": "RAG 지식 챗봇",
        "content_tab": "맞춤형 학습 콘텐츠 생성",
        "lstm_tab": "LSTM 성취도 예측 대시보드",
        "sim_tab_chat_email": "AI 고객 응대 시뮬레이터 (채팅/이메일)",
        "sim_tab_phone": "AI 고객 응대 시뮬레이터 (전화)",
        "simulator_tab": "AI 고객 응대 시뮬레이터",
        "rag_header": "RAG 지식 챗봇 (문서 기반 Q&A)",
        "rag_desc": "업로드된 문서 기반으로 질문에 답변합니다。",
        "rag_input_placeholder": "학습 자료에 대해 질문해 보세요",
        "llm_error_key": "⚠️ 경고: GEMINI API 키가 설정되지 않았습니다. Streamlit Secrets에 'GEMINI_API_KEY'를 설정해주세요。",
        "llm_error_init": "LLM 초기화 오류: API 키를 확인해 주세요。",
        "content_header": "맞춤형 학습 콘텐츠 생성",
        "content_desc": "학습 주제와 난이도에 맞춰 콘텐츠 생성",
        "topic_label": "학습 주제",
        "level_label": "난이도",
        "content_type_label": "콘텐츠 형식",
        "level_options": ["초급", "중급", "고급"],
        "content_options": ["핵심 요약 노트", "객관식 퀴즈 10문항", "실습 예제 아이디어"],
        "button_generate": "콘텐츠 생성",
        "warning_topic": "학습 주제를 입력해 주세요。",
        "lstm_header": "LSTM 기반 학습 성취도 예측 대시보드",
        "lstm_desc": "가상의 과거 퀴즈 점수 데이터를 바탕으로 LSTM 모델을 훈련하고 미래 성취도를 예측하여 보여줍니다。",
        "lang_select": "언어 선택",
        "embed_success": "총 {count}개 청크로 학습 DB 구축 완료!",
        "embed_fail": "임베딩 실패: 무료 티어 한도 초과 또는 네트워크 문제。",
        "warning_no_files": "먼저 학습 자료를 업로드하세요。",
        "warning_rag_not_ready": "RAG가 준비되지 않았습니다. 학습 자료를 업로드하고 분석하세요。",
        "quiz_fail_structure": "퀴즈 데이터 구조가 올바르지 않습니다。",
        "select_answer": "정답을 선택하세요",
        "check_answer": "정답 확인",
        "next_question": "다음 문항",
        "correct_answer": "정답입니다! 🎉",
        "incorrect_answer": "오답입니다。😞",
        "correct_is": "정답",
        "explanation": "해설",
        "quiz_complete": "퀴즈 완료!",
        "score": "점수",
        "retake_quiz": "퀴즈 다시 풀기",
        "quiz_error_llm": "퀴즈 생성 실패: LLM이 올바른 JSON 형식을 반환하지 않았습니다。",
        "quiz_original_response": "LLM 원본 응답",
        "firestore_loading": "데이터베이스에서 RAG 인덱스 로드 중...",
        "firestore_no_index": "데이터베이스에서 기존 RAG 인덱스를 찾을 수 없습니다. 파일을 업로드하여 새로 만드세요。",
        "db_save_complete": "(DB 저장 완료)",
        "data_analysis_progress": "자료 분석 및 학습 DB 구축 중...",
        "response_generating": "답변 생성 중...",
        "lstm_result_header": "학습 성취도 예측 결과",
        "lstm_score_metric": "현재 예측 성취도",
        "lstm_score_info": "다음 퀴즈 예상 점수는 약 **{predicted_score:.1f}점**입니다. 학습 성과를 유지하거나 개선하세요!",
        "lstm_rerun_button": "새로운 가상 데이터로 예측",

        # --- 토스트 메시지 추가 ---
        "toast_like": "🔥 컨텐츠가 맘에 드셨군요! (좋아요 카운트 +1)",
        "toast_dislike": "😔 더 나은 콘텐츠를 위해 피드백을 반영하겠습니다。",
        "toast_share": "🌐 콘텐츠 링크가 생성되었습니다。",
        "toast_copy": "✅ 콘텐츠가 클립보드에 복사되었습니다!",
        "toast_more": "ℹ️ 추가 옵션 (PDF, 인쇄본 저장 등)",
        "mock_pdf_save": "📥 PDF 저장",
        "mock_word_open": "📑 Word로 열기",
        "mock_print": "🖨 인쇄",

        # --- 시뮬레이터 ---
        "simulator_header": "AI 고객 응대 시뮬레이터",
        "simulator_desc": "까다로운 고객 문의에 AI의 응대 초안 및 가이드라인을 제공합니다。",
        "customer_query_label": "고객 문의 내용 (링크 포함 가능)",
        "customer_type_label": "고객 성향",
        "customer_type_options": ["일반적인 문의", "까다로운 고객", "매우 불만족스러운 고객"],
        "button_simulate": "응대 조언 요청",
        "customer_generate_response_button": "고객 반응 생성",
        "send_closing_confirm_button": "추가 문의 여부 확인 메시지 보내기",
        "simulation_warning_query": "고객 문의 내용을 입력해 주세요。",
        "simulation_no_key_warning": "⚠️ API Key가 없기 때문에 응답 생성은 실행되지 않습니다。",
        "simulation_advice_header": "AI의 응대 가이드라인",
        "simulation_draft_header": "추천 응대 초안",
        "button_listen_audio": "음성으로 듣기",
        "tts_status_ready": "음성으로 듣기 준비됨",
        "tts_status_generating": "오디오 생성 중...",
        "tts_status_success": "✅ 오디오 재생 완료!",
        "tts_status_error": "❌ TTS 오류 발생",
        "history_expander_title": "📝 이전 상담 이력 로드 (최근 10건)",
        "initial_query_sample": "프랑스 파리에 도착했는데, 클룩에서 구매한 eSIM이 활성화가 안 됩니다...",
        "button_mic_input": "🎙 음성 입력",
        "button_mic_stop": "⏹️ 녹음 종료",
        "prompt_customer_end": "고객님의 추가 문의 사항이 없어, 이 상담을 종료합니다。",
        "prompt_survey": "지금까지 상담원 000였습니다. 즐거운 하루 되시기 바랍니다. [설문 조사 링크]",
        "customer_closing_confirm": "다른 문의 사항은 없으십니까?",
        "customer_positive_response": "알겠습니다. 감사합니다。",
        "button_email_end_chat": "응대 종료 (설문 요청)",
        "error_mandatory_contact": "이메일과 전화번호 입력은 필수입니다。",
        "customer_attachment_label": "📎 고객 첨부 파일 업로드",
        "attachment_info_llm": "[고객 첨부 파일: {filename}이(가) 확인되었습니다. 이 파일을 참고하여 응대하세요.]",
        "button_retry_translation": "번역 다시 시도",
        "button_request_hint": "💡 응대 힌트 요청 (AHT 모니터링 중)",
        "button_generate_draft": "🤖 AI 응답 초안 생성",
        "draft_generating": "AI가 응답 초안을 생성 중입니다...",
        "draft_success": "✅ AI 응답 초안이 생성되었습니다. 아래에서 확인하고 수정하세요.",
        "hint_placeholder": "문의 응대에 대한 힌트:",
        "survey_sent_confirm": "📨 설문조사 링크가 전송되었으며, 이 상담은 종료되었습니다。",
        "new_simulation_ready": "새 시뮬레이션을 시작할 수 있습니다。",
        "agent_response_header": "✍️ 에이전트 응답",
        "agent_response_placeholder": "고객에게 응답하세요...",
        "send_response_button": "응답 전송",
        "customer_turn_info": "에이전트 응답 전송 완료. 고객 반응을 자동으로 생성 중입니다。",
        "generating_customer_response": "고객 반응 생성 중...",
        "customer_escalation_start": "상급자와 이야기하고 싶습니다",
        "request_rebuttal_button": "고객의 다음 반응 요청",
        "new_simulation_button": "새 시뮬레이션 시작",
        "history_selectbox_label": "로드할 이력을 선택하세요:",
        "history_load_button": "선택된 이력 로드",
        "delete_history_button": "❌ 모든 이력 삭제",
        "delete_confirm_message": "정말로 모든 상담 이력을 삭제하시겠습니까?",
        "delete_confirm_yes": "예, 삭제합니다",
        "download_history_word": "📥 이력 다운로드 (Word)",
        "download_history_pptx": "📥 이력 다운로드 (PPTX)",
        "download_history_pdf": "📥 이력 다운로드 (PDF)",
        "download_current_session": "📥 현재 세션 다운로드",
        "delete_confirm_no": "아니오, 유지합니다",
        "delete_success": "✅ 삭제 완료!",
        "deleting_history_progress": "이력 삭제 중...",
        "search_history_label": "이력 검색",
        "date_range_label": "날짜 범위 필터",
        "history_search_button": "🔍 검색",
        "no_history_found": "검색 조건에 맞는 이력이 없습니다。",
        "customer_email_label": "고객 이메일 (필수)",
        "customer_phone_label": "고객 연락처 / 전화번호 (필수)",
        "transfer_header": "언어 이관 요청 (다른 팀)",
        "transfer_to_en": "🇺🇸 영어 팀으로 이관",
        "transfer_to_ja": "🇯🇵 일본어 팀으로 이관",
        "transfer_to_ko": "🇰🇷 한국어 팀으로 이관",
        "transfer_system_msg": "📌 시스템 메시지: 고객 요청에 따라 상담 언어가 {target_lang} 팀으로 이관되었습니다. 새로운 상담원(AI)이 응대합니다。",
        "transfer_loading": "이관 처리 중: 이전 대화 이력 번역 및 검토 (고객님께 3~10분 양해 요청)",
        "transfer_summary_header": "🔍 이관된 상담원을 위한 요약 (번역됨)",
        "transfer_summary_intro": "고객님과의 이전 대화 이력입니다. 이 내용을 바탕으로 응대를 이어나가세요。",
        "llm_translation_error": "❌ 번역 실패: LLM 응답 오류",
        "timer_metric": "상담 경과 시간 (AHT)",
        "timer_info_ok": "AHT (15분 기준)",
        "timer_info_warn": "AHT (10분 초과)",
        "timer_info_risk": "🚨 15분 초과: 높은 리스크",
        "solution_check_label": "✅ 이 응답에 솔루션/해결책이 포함되어 있습니다。",
        "sentiment_score_label": "고객 감정 점수",
        "urgency_score_label": "긴급도 점수",
        "similarity_chart_title": "유사 케이스 유사도",
        "scores_comparison_title": "감정 및 만족도 점수 비교",
        "similarity_score_label": "유사도",
        "satisfaction_score_label": "만족도",
        "sentiment_trend_label": "감정 점수 추이",
        "satisfaction_trend_label": "만족도 점수 추이",
        "case_trends_title": "과거 케이스 점수 추이",
        "date_label": "날짜",
        "score_label": "점수 (0-100)",
        "customer_characteristics_title": "고객 특성 분포",
        "language_label": "언어",
        "email_provided_label": "이메일 제공",
        "phone_provided_label": "전화번호 제공",
        "region_label": "지역",
        "btn_request_phone_summary": "이력 요약 요청",

        # --- 추가된 전화 발신 기능 관련 ---
        "button_call_outbound": "전화 발신",
        "call_outbound_system_msg": "📌 시스템 메시지: 에이전트가 {target}에게 전화 발신을 시도했습니다。",
        "call_outbound_simulation_header": "📞 전화 발신 시뮬레이션 결과",
        "call_outbound_summary_header": "📞 현지 업체/고객과의 통화 요약",
        "call_outbound_loading": "전화 연결 및 통화 결과 정리 중... (LLM 호출)",
        "call_target_customer": "고객에게 발신",
        "call_target_partner": "현지 업체 발신",

        # --- 음성 기록 ---
        "voice_rec_header": "음성 기록 & 관리",
        "record_help": "마이크 버튼을 눌러 녹음하거나 파일을 업로드하세요。",
        "uploaded_file": "오디오 파일 업로드",
        "rec_list_title": "저장된 음성 기록",
        "transcribe_btn": "전사(Whisper)",
        "save_btn": "음성 기록 저장",
        "transcribing": "음성 전사 중...",
        "transcript_result": "전사 결과:",
        "transcript_text": "전사 텍스트",
        "openai_missing": "OpenAI API Key가 없습니다。",
        "whisper_client_error": "❌ Whisper API Client 초기화 실패",
        "whisper_auth_error": "❌ Whisper API 인증 실패",
        "whisper_format_error": "❌ 지원하지 않는 오디오 형식입니다。",
        "whisper_success": "✅ 음성 전사 완료!",
        "playback": "녹음 재생",
        "retranscribe": "재전사",
        "delete": "삭제",
        "no_records": "저장된 음성 기록이 없습니다。",
        "saved_success": "저장 완료!",
        "delete_confirm_rec": "정말 삭제하시겠습니까?",
        "gcs_not_conf": "GCS 미설정",
        "gcs_playback_fail": "오디오 재생 실패",
        "gcs_no_audio": "오디오 없음",
        "error": "오류:",
        "firestore_no_db_connect": "DB 연결 실패",
        "save_history_success": "상담 이력이 저장되었습니다。",
        "save_history_fail": "상담 이력 저장 실패",
        "delete_fail": "삭제 실패",
        "rec_header": "음성 입력 및 전사",
        "whisper_processing": "음성 전사 처리 중..",
        "empty_response_warning": "응답을 입력하세요.",
        "customer_no_more_inquiries": "없습니다. 감사합니다.",
        "customer_has_additional_inquiries": "추가 문의 사항도 있습니다.",
        "sim_end_chat_button": "설문 조사 링크 전송 및 응대 종료",
        "delete_mic_record": "❌ 녹음 삭제",

        # --- 첨부 파일 기능 추가 ---
        "attachment_label": "고객 첨부 파일 업로드 (스크린샷 등)",
        "attachment_placeholder": "파일을 첨부하여 상황을 설명하세요 (선택 사항)",
        "attachment_info_llm": "[고객 첨부 파일: {filename}이(가) 확인되었습니다. 이 파일을 참고하여 응대하세요.]",
        "agent_attachment_label": "에이전트 첨부 파일 (스크린샷 등)",
        "agent_attachment_placeholder": "응답에 첨부할 파일을 선택하세요 (선택 사항)",
        "agent_attachment_status": "📎 에이전트가 **{filename}** 파일을 응답에 첨부했습니다. (파일 타입: {filetype})",

        # --- RAG 오류 메시지 추가 ---
        "rag_embed_error_openai": "RAG 임베딩 실패: OpenAI API Key가 유효하지 않거나 설정되지 않았습니다。",
        "rag_embed_error_gemini": "RAG 임베딩 실패: Gemini API Key가 유효하지 않거나 설정되지 않았습니다。",
        "rag_embed_error_nvidia": "RAG 임베딩 실패: NVIDIA API Key가 유효하지 않거나 설정되지 않았습니다。",
        "rag_embed_error_none": "RAG 임베딩에 필요한 모든 키(OpenAI, Gemini, NVIDIA)가 유효하지 않습니다. 키를 설정해 주세요。",

        # --- 전화 기능 관련 추가 ---
        "phone_header": "AI 고객 응대 시뮬레이터 (전화)",
        "call_status_waiting": "수신 대기 중...",
        "call_status_ringing": "전화 수신 중: {number}",
        "button_answer": "📞 전화 응답",
        "button_hangup": "🔴 전화 끊기",
        "button_hold": "⏸️ Hold (소음 차단)",
        "button_resume": "▶️ 통화 재개",
        "hold_status": "통화 Hold 중 (누적 Hold 시간: {duration})",
        "cc_live_transcript": "🎤 실시간 CC 자막 / 전사",
        "mic_input_status": "🎙️ 에이전트 음성 입력",
        "customer_audio_playback": "🗣️ 고객 음성 재생",
        "agent_response_prompt": "고객에게 말할 응답을 녹음하세요。",
        "agent_response_stop_and_send": "⏹️ 녹음 종료 및 응답 전송",
        "call_end_message": "통화가 종료되었습니다. AHT 및 이력을 확인하세요。",
        "call_query_placeholder": "고객 문의 내용을 입력하세요。",
        "call_number_placeholder": "+82 10-xxxx-xxxx (가상 번호)",
        "call_summary_header": "AI 통화 요약",
        "customer_audio_header": "고객 최초 문의 (음성)",
        "aht_not_recorded": "⚠️ 통화 시작 시간이 기록되지 않아 AHT를 계산할 수 없습니다。",
        "no_audio_record": "고객의 최초 음성 기록이 없습니다。",
    },

    # --- ⭐ 영어 버전 (한국어 100% 매칭) ---
    "en": {
        "title": "Personalized AI Study Coach (Voice & Local DB)",
        "sidebar_title": "📚 AI Study Coach Settings",
        "file_uploader": "Upload Study Materials (PDF, TXT, HTML)",
        "button_start_analysis": "Start Analysis (RAG Indexing)",
        "rag_tab": "RAG Knowledge Chatbot",
        "content_tab": "Custom Content Generation",
        "lstm_tab": "LSTM Achievement Prediction Dashboard",
        "sim_tab_chat_email": "AI Customer Support Simulator (Chat / Email)",
        "sim_tab_phone": "AI Customer Support Simulator (Phone)",
        "simulator_tab": "AI Customer Support Simulator",
        "rag_header": "RAG Knowledge Chatbot (Document Q&A)",
        "rag_desc": "Answer questions based on uploaded documents.",
        "rag_input_placeholder": "Ask a question about your study materials",
        "llm_error_key": "⚠️ Warning: GEMINI_API_KEY is not set.",
        "llm_error_init": "LLM initialization error. Please check your API key.",
        "content_header": "Custom Learning Content Generation",
        "content_desc": "Generate content based on the topic and difficulty.",
        "topic_label": "Learning Topic",
        "level_label": "Difficulty",
        "content_type_label": "Content Type",
        "level_options": ["Beginner", "Intermediate", "Advanced"],
        "content_options": ["Key Summary Note", "10 MCQ Questions", "Practical Example Idea"],
        "button_generate": "Generate Content",
        "warning_topic": "Please enter a learning topic.",
        "lstm_header": "LSTM Achievement Prediction Dashboard",
        "lstm_desc": "Train an LSTM model on hypothetical quiz scores and predict performance.",
        "lang_select": "Select Language",
        "embed_success": "Learning DB built with {count} chunks!",
        "embed_fail": "Embedding failed: quota exceeded or network issue.",
        "warning_no_files": "Please upload study materials first.",
        "warning_rag_not_ready": "RAG is not ready. Upload materials and analyze.",
        "quiz_fail_structure": "Quiz data structure is invalid.",
        "select_answer": "Select answer",
        "check_answer": "Check answer",
        "next_question": "Next question",
        "correct_answer": "Correct! 🎉",
        "incorrect_answer": "Incorrect 😞",
        "correct_is": "Correct answer",
        "explanation": "Explanation",
        "quiz_complete": "Quiz Complete!",
        "score": "Score",
        "retake_quiz": "Retake Quiz",
        "quiz_error_llm": "Quiz generation failed: invalid JSON.",
        "quiz_original_response": "Original LLM Response",
        "firestore_loading": "Loading RAG index...",
        "firestore_no_index": "No existing RAG index found.",
        "db_save_complete": "(DB Save Complete)",
        "data_analysis_progress": "Analyzing materials and building DB...",
        "response_generating": "Generating response...",
        "lstm_result_header": "Achievement Prediction",
        "lstm_score_metric": "Predicted Achievement",
        "lstm_score_info": "Estimated next quiz score: **{predicted_score:.1f}**.",
        "lstm_rerun_button": "Predict with New Data",

        # --- 토스트 메시지 추가 ---
        "toast_like": "🔥 Content liked! (+1 Count Reflected)",
        "toast_dislike": "😔 Feedback recorded for better content.",
        "toast_share": "🌐 Content link generated.",
        "toast_copy": "✅ Content copied to clipboard!",
        "toast_more": "ℹ️ Additional options (Print, PDF Save, etc.)",
        "mock_pdf_save": "📥 Save as PDF",
        "mock_word_open": "📑 Open via Word",
        "mock_print": "🖨 Print",

        # --- 토스트 메시지 끝 ---

        # Simulator
        "simulator_header": "AI Customer Response Simulator",
        "simulator_desc": "AI generates draft responses and guidelines for customer inquiries.",
        "customer_query_label": "Customer Message (links allowed)",
        "customer_type_label": "Customer Type",
        "customer_type_options": ["General Inquiry", "Difficult Customer", "Highly Dissatisfied Customer"],
        "button_simulate": "Generate Response",
        "customer_generate_response_button": "Generate Customer Response",
        "send_closing_confirm_button": "Send Closing Confirmation",
        "simulation_warning_query": "Please enter the customer’s message.",
        "simulation_no_key_warning": "⚠️ API Key missing. Simulation cannot proceed.",
        "simulation_advice_header": "AI Response Guidelines",
        "simulation_draft_header": "Recommended Response Draft",
        "button_listen_audio": "Play as Audio",
        "tts_status_ready": "Ready to generate audio",
        "tts_status_generating": "Generating audio...",
        "tts_status_success": "Audio ready!",
        "tts_status_error": "TTS error occurred",
        "history_expander_title": "📝 Load Previous Sessions (Last 10)",
        "initial_query_sample": "I arrived in Paris but my Klook eSIM won't activate…",
        "button_mic_input": "🎙 Voice Input",
        "button_mic_stop": "⏹️ Stop recording",
        "prompt_customer_end": "No further inquiries. Ending chat.",
        "prompt_survey": "This was Agent 000. Have a nice day. [Survey Link]",
        "customer_closing_confirm": "Is there anything else we can assist you with?",
        "customer_positive_response": "I understand. Thank you.",
        "button_email_end_chat": "End supports (Survey Request)",
        "error_mandatory_contact": "Email and Phone number input are mandatory.",
        "customer_attachment_label": "📎 Customer Attachment Upload",
        "attachment_info_llm": "[Customer Attachment: {filename} is confirmed. Reference this file in your response.]",
        "button_retry_translation": "Retry Translation",
        "button_request_hint": "💡 Request Response Hint (AHT Monitored)",
        "button_generate_draft": "🤖 Generate AI Response Draft",
        "draft_generating": "AI is generating a response draft...",
        "draft_success": "✅ AI response draft has been generated. Please review and modify below.",
        "hint_placeholder": "Hints for responses",
        "survey_sent_confirm": "📨 The survey link has been sent. This chat session is now closed。",
        "new_simulation_ready": "You can now start a new simulation.",
        "agent_response_header": "✍️ Agent Response",
        "agent_response_placeholder": "Write a response...",
        "send_response_button": "Send Response",
        "customer_turn_info": "Agent response sent. Generating customer reaction automatically。",
        "generating_customer_response": "Generating customer response...",
        "customer_escalation_start": "I want to speak to a supervisor",
        "request_rebuttal_button": "Request Customer Reaction",
        "new_simulation_button": "Start New Simulation",
        "history_selectbox_label": "Choose a record to load:",
        "history_load_button": "Load Selected Record",
        "delete_history_button": "❌ Delete All History",
        "delete_confirm_message": "Are you sure you want to delete all records?",
        "delete_confirm_yes": "Yes, Delete",
        "delete_confirm_no": "Cancel",
        "download_history_word": "📥 Download History (Word)",
        "download_history_pptx": "📥 Download History (PPTX)",
        "download_history_pdf": "📥 Download History (PDF)",
        "download_current_session": "📥 Download Current Session",
        "delete_success": "Deleted successfully!",
        "deleting_history_progress": "Deleting history...",
        "search_history_label": "Search History",
        "date_range_label": "Date Filter",
        "history_search_button": "🔍 Search",
        "no_history_found": "No matching history found.",
        "customer_email_label": "Customer Email (Mandatory)",
        "customer_phone_label": "Customer Phone / WhatsApp (Mandatory)",
        "transfer_header": "Language Transfer Request (To Other Teams)",
        "transfer_to_en": "🇺🇸 English Team Transfer",
        "transfer_to_ja": "🇯🇵 Japanese Team Transfer",
        "transfer_to_ko": "🇰🇷 Korean Team Transfer",
        "transfer_system_msg": "📌 System Message: The session language has been transferred to the {target_lang} team per customer request. A new agent (AI) will now respond。",
        "transfer_loading": "Transferring: Translating and reviewing chat history (3-10 minute wait requested from customer)",
        "transfer_summary_header": "🔍 Summary for Transferred Agent (Translated)",
        "transfer_summary_intro": "This is the previous chat history. Please continue the support based on this summary。",
        "llm_translation_error": "❌ Translation failed: LLM response error",
        "timer_metric": "Elapsed Time",
        "timer_info_ok": "AHT (15 min standard)",
        "timer_info_warn": "AHT (Over 10 min)",
        "timer_info_risk": "🚨 Over 15 min: High Risk",
        "solution_check_label": "✅ This response includes a solution/fix。",
        "sentiment_score_label": "Customer Sentiment Score",
        "urgency_score_label": "Urgency Score",
        "similarity_chart_title": "Case Similarity",
        "scores_comparison_title": "Sentiment & Satisfaction Scores",
        "similarity_score_label": "Similarity",
        "satisfaction_score_label": "Satisfaction",
        "sentiment_trend_label": "Sentiment Trend",
        "satisfaction_trend_label": "Satisfaction Trend",
        "case_trends_title": "Case Score Trends",
        "date_label": "Date",
        "score_label": "Score (0-100)",
        "customer_characteristics_title": "Customer Characteristics",
        "language_label": "Language",
        "email_provided_label": "Email Provided",
        "phone_provided_label": "Phone Provided",
        "region_label": "Region",
        "btn_request_phone_summary": "Request to summarize histories",

        # --- 추가된 전화 발신 기능 관련 ---
        "button_call_outbound": "Call Outbound",
        "call_outbound_system_msg": "📌 System Message: Agent attempted an outbound call to {target}。",
        "call_outbound_simulation_header": "📞 Outbound Call Simulation Result",
        "call_outbound_summary_header": "📞 Summary of Call with Local Partner/Customer",
        "call_outbound_loading": "Connecting call and summarizing outcome... (LLM Call)",
        "call_target_customer": "Call Customer",
        "call_target_partner": "Call Local Partner",

        # --- 음성 기록 ---
        "voice_rec_header": "Voice Record & Management",
        "record_help": "Record using the microphone or upload a file。",
        "uploaded_file": "Upload Audio File",
        "rec_list_title": "Saved Voice Records",
        "transcribe_btn": "Transcribe (Whisper)",
        "save_btn": "Save Record",
        "transcribing": "Transcribing...",
        "transcript_result": "Transcription:",
        "transcript_text": "Transcribed Text",
        "openai_missing": "OpenAI API Key is missing。",
        "whisper_client_error": "❌ Error: Whisper API client not initialized。",
        "whisper_auth_error": "❌ Whisper API authentication failed。",
        "whisper_format_error": "❌ Error: Unsupported audio format。",
        "whisper_success": "✅ Voice Transcription Complete!",
        "playback": "Playback Recording",
        "retranscribe": "Re-transcribe",
        "delete": "Delete",
        "no_records": "No saved voice records。",
        "saved_success": "Saved successfully!",
        "delete_confirm_rec": "Are you sure you want to delete this voice record?",
        "gcs_not_conf": "GCS not configured or no audio available",
        "gcs_playback_fail": "Failed to play audio",
        "gcs_no_audio": "No audio file found",
        "error": "Error:",
        "firestore_no_db_connect": "DB connection failed",
        "save_history_success": "Saved successfully。",
        "save_history_fail": "Save failed。",
        "delete_fail": "Delete failed",
        "rec_header": "Voice Input & Transcription",
        "whisper_processing": "Processing...",
        "empty_response_warning": "Please enter a response。",
        "customer_no_more_inquiries": "No, that will be all, thank you。",
        "customer_has_additional_inquiries": "Yes, I have an additional question。",
        "sim_end_chat_button": "Send Survey Link and End Consultations",
        "delete_mic_record": "❌ Delete recordings",

        # --- 첨부 파일 기능 추가 ---
        "attachment_label": "Customer Attachment Upload (Screenshot, etc。)",
        "attachment_placeholder": "Attach a file to explain the situation (optional)",
        "attachment_info_llm": "[Customer Attachment: {filename} is confirmed. Reference this file in your response。]",
        "agent_attachment_label": "Agent Attachment (Screenshot, etc。)",
        "agent_attachment_placeholder": "Select a file to attach to the response (optional)",
        "agent_attachment_status": "📎 Agent attached **{filename}** file to the response。 (File type: {filetype})",

        # --- RAG 오류 메시지 추가 ---
        "rag_embed_error_openai": "RAG embedding failed: OpenAI API Key is invalid or not set。",
        "rag_embed_error_gemini": "RAG embedding failed: Gemini API Key is invalid or not set。",
        "rag_embed_error_nvidia": "RAG embedding failed: NVIDIA API Key is invalid or not set。",
        "rag_embed_error_none": "RAG embedding failed: All required keys (OpenAI, Gemini, NVIDIA) are invalid or not set。 Please configure a key。",

        # --- 전화 기능 관련 추가 ---
        "phone_header": "AI Customer Support Simulator (Phone)",
        "call_status_waiting": "Waiting for incoming call...",
        "call_status_ringing": "Incoming Call from: {number}",
        "button_answer": "📞 Answer Call",
        "button_hangup": "🔴 Hang Up",
        "button_hold": "⏸️ Hold (Mute)",
        "button_resume": "▶️ Resume Call",
        "hold_status": "On Hold (Total Hold Time: {duration})",
        "cc_live_transcript": "🎤 Live CC Transcript",
        "mic_input_status": "🎙️ Agent Voice Input",
        "customer_audio_playback": "🗣️ Customer Audio Playback",
        "agent_response_prompt": "Record your response to the customer。",
        "agent_response_stop_and_send": "⏹️ Stop and share recording to customers",
        "call_end_message": "Call ended. Check AHT and history。",
        "call_query_placeholder": "Enter customer's initial query。",
        "call_number_placeholder": "+1 (555) 123-4567 (Mock Number)",
        "call_summary_header": "AI Call Summary",
        "customer_audio_header": "Customer Initial Query (Voice)",
        "aht_not_recorded": "⚠️ Call start time not recorded。 Cannot calculate AHT。",
        "no_audio_record": "No initial customer voice record。",

    },

    # --- ⭐ 일본어 버전 (한국어 100% 매칭) ---
    "ja": {
        "title": "パーソナライズAI学習コーチ (音声・ローカルDB)",
        "sidebar_title": "📚 AI学習コーチ設定",
        "file_uploader": "学習資料をアップロード (PDF, TXT, HTML)",
        "button_start_analysis": "資料分析開始 (RAGインデックス作成)",
        "rag_tab": "RAG知識チャットボット",
        "content_tab": "カスタム学習コンテンツ生成",
        "lstm_tab": "LSTM達成度予測ダッシュボード",
        "sim_tab_chat_email": "AI顧客対応シミュレーター(チャット・メール)",
        "sim_tab_phone": "AI顧客対応シミュレーター(電話)",
        "simulator_tab": "AI顧客対応シミュレーター",
        "rag_header": "RAG知識チャットボット (ドキュメントQ&A)",
        "rag_desc": "アップロードされた資料に基づいて質問に回答します。",
        "rag_input_placeholder": "資料について質問してください",
        "llm_error_key": "⚠️ 注意: GEMINI_API_KEY が設定されていません。",
        "llm_error_init": "LLM 初期化エラー：APIキーを確認してください。",
        "content_header": "カスタム学習コンテンツ生成",
        "content_desc": "学習テーマと難易度に応じてコンテンツを生成します。",
        "topic_label": "学習テーマ",
        "level_label": "難易度",
        "content_type_label": "コンテンツ種類",
        "level_options": ["初級", "中級", "上級"],
        "content_options": ["要点サマリー", "選択式クイズ10問", "実践例アイデア"],
        "button_generate": "生成する",
        "warning_topic": "学習テーマを入力してください。",
        "lstm_header": "LSTM達成度予測ダッシュボード",
        "lstm_desc": "仮想クイズスコアを使用して達成度を予測します。",
        "lang_select": "言語選択",
        "embed_success": "{count}個のチャンクでDB構築完了!",
        "embed_fail": "埋め込み失敗：クォータ超過またはネットワーク問題。",
        "warning_no_files": "資料をアップロードしてください。",
        "warning_rag_not_ready": "RAGが準備できていません。",
        "quiz_fail_structure": "クイズデータの形式が正しくありません。",
        "select_answer": "回答を選択してください",
        "check_answer": "回答を確認",
        "next_question": "次の質問",
        "correct_answer": "正解！ 🎉",
        "incorrect_answer": "不正解 😞",
        "correct_is": "正解",
        "explanation": "解説",
        "quiz_complete": "クイズ完了!",
        "score": "スコア",
        "retake_quiz": "再挑戦",
        "quiz_error_llm": "퀴즈 생성 실패：JSON形式が正しくありません。",
        "quiz_original_response": "LLM 原本回答",
        "firestore_loading": "RAGインデックス読み込み中...",
        "firestore_no_index": "保存されたRAGインデックスが見つかりません。",
        "db_save_complete": "(DB保存完了)",
        "data_analysis_progress": "資料分析中...",
        "response_generating": "応答生成中...",
        "lstm_result_header": "達成度予測結果",
        "lstm_score_metric": "予測達成度",
        "lstm_score_info": "次のスコア予測: **{predicted_score:.1f}점**",
        "lstm_rerun_button": "新しいデータで再予測",

        # --- 토스트 메시지 추가 ---
        "toast_like": "🔥 コンテンツを気に入っていただけました！ (+1 カウント反映)",
        "toast_dislike": "😔 より良いコンテンツのためフィードバックを記録しました。",
        "toast_share": "🌐 コンテンツリンクが生成されました。",
        "toast_copy": "✅ コンテンツがクリップボードにコピーされました！",
        "toast_more": "ℹ️ その他のオプション（印刷、PDF保存など）",
        "mock_pdf_save": "📥 PDFで保存",
        "mock_word_open": "📑 Wordで開く",
        "mock_print": "🖨 印刷",
        # --- 토스트 메시지 끝 ---

        # --- Simulator ---
        "simulator_header": "AI顧客対応シミュレーター",
        "simulator_desc": "難しい顧客問い合わせに対するAIのガイドラインと草案を生成します。",
        "customer_query_label": "顧客からの問い合わせ内容 (リンク可)",
        "customer_type_label": "顧客タイプ",
        "customer_type_options": ["一般的な問い合わせ", "難しい顧客", "非常に不満な顧客"],
        "button_simulate": "応対ガイド生成",
        "customer_generate_response_button": "顧客の返信を生成",
        "send_closing_confirm_button": "追加のご質問有無を確認するメッセージを送信",
        "simulation_warning_query": "お問い合わせ内容を入力してください。",
        "simulation_no_key_warning": "⚠️ APIキー不足のため応対生成不可。",
        "simulation_advice_header": "AI対応ガイドライン",
        "simulation_draft_header": "推奨応対草案",
        "button_listen_audio": "音声で聞く",
        "tts_status_ready": "音声生成準備完了",
        "tts_status_generating": "音声生成中...",
        "tts_status_success": "音声準備完了！",
        "tts_status_error": "TTS エラーが発生しました",
        "history_expander_title": "📝 過去の対応履歴を読み込む (最新10件)",
        "initial_query_sample": "パリに到着しましたが、KlookのeSIMが使えません…",
        "button_mic_input": "🎙 音声入力",
        "button_mic_stop": "⏹️ 録音終了",
        "prompt_customer_end": "追加の質問がないためチャットを終了します。",
        "prompt_survey": "担当エージェント000でした。良い一日をお過ごしください。 [アンケートリンク]",
        "customer_closing_confirm": "他のお問合せはございませんでしょうか。",
        "customer_positive_response": "承知いたしました。ありがとうございます。",
        "button_email_end_chat": "応対終了（アンケート）",
        "error_mandatory_contact": "メールアドレスと電話番号の入力は必須です。",
        "customer_attachment_label": "📎 顧客添付ファイルアップロード",
        "attachment_info_llm": "[顧客添付ファイル: {filename}が確認されました。このファイルを参照して対応してください。]",
        "button_retry_translation": "翻訳を再試行",
        "button_request_hint": "💡 応対ヒントを要請 (AHT モニタリング中)",
        "button_generate_draft": "🤖 AI応答草案生成",
        "draft_generating": "AIが応答草案を生成中です...",
        "draft_success": "✅ AI応答草案が生成されました。以下で確認して修正してください。",
        "hint_placeholder": "お問合せの応対に対するヒント：",
        "new_simulation_ready": "新しいシミュレーションを開始できます。",
        "survey_sent_confirm": "📨 アンケートリンクを送信しました。このチャットは終了しました。",
        "agent_response_header": "✍️ エージェント応答",
        "agent_response_placeholder": "顧客へ返信内容を入力…",
        "send_response_button": "返信送信",
        "customer_turn_info": "エージェント応答送信完了。顧客の反応を自動生成中です。",
        "generating_customer_response": "顧客の反応を生成中...",
        "customer_escalation_start": "上級の担当者と話したい",
        "request_rebuttal_button": "顧客の反応を生成",
        "new_simulation_button": "新規シミュレーション",
        "history_selectbox_label": "履歴を選択:",
        "history_load_button": "履歴を読み込む",
        "delete_history_button": "❌ 全履歴削除",
        "delete_confirm_message": "すべての履歴を削除しますか？",
        "delete_confirm_yes": "はい、削除します。",
        "download_history_word": "📥 履歴ダウンロード (Word)",
        "download_history_pptx": "📥 履歴ダウンロード (PPTX)",
        "download_history_pdf": "📥 履歴ダウンロード (PDF)",
        "download_current_session": "📥 現在のセッションをダウンロード",
        "delete_confirm_no": "いいえ、維持します。",
        "delete_success": "削除完了！",
        "deleting_history_progress": "削除中...",
        "search_history_label": "履歴検索",
        "date_range_label": "日付フィルター",
        "history_search_button": "🔍 検索",
        "no_history_found": "該当する履歴はありません。",
        "customer_email_label": "顧客メールアドレス（必修）",
        "customer_phone_label": "顧客連絡先 / 電話番号（必修）",
        "transfer_header": "言語切り替え要請（他チームへ）",
        "transfer_to_en": "🇺🇸 英語チームへ転送",
        "transfer_to_ja": "🇯🇵 日本語チームへ転送",
        "transfer_to_ko": "🇰🇷 韓国語チームへ転送",
        "transfer_system_msg": "📌 システムメッセージ: 顧客の要請により、対応言語が {target_lang} チームへ切り替えられました。新しい担当者(AI)が対応します。",
        "transfer_loading": "転送中: 過去のチャット履歴を翻訳およびレビューしています (お客様には3〜10分のお時間をいただいています)",
        "transfer_summary_header": "🔍 転送された担当者向けの要約 (翻訳済み)",
        "transfer_summary_intro": "これが顧客との過去のチャット履歴です。この要約に基づいてサポートを続けてください。",
        "llm_translation_error": "❌ 翻訳失敗: LLM応答エラー",
        "timer_metric": "経過時間",
        "timer_info_ok": "AHT (15分基準)",
        "timer_info_warn": "AHT (10分経過)",
        "timer_info_risk": "🚨 15分経過: 高いリスク",
        "solution_check_label": "✅ この応答に解決策/対応策が含まれています。",
        "sentiment_score_label": "顧客の感情スコア",  # <--- 추가/수정
        "urgency_score_label": "緊急度スコア",
        "similarity_chart_title": "類似性ケースの比率",
        "scores_comparison_title": "感情及び満足度のスコア",
        "similarity_score_label": "類似性",
        "satisfaction_score_label": "満足度",
        "sentiment_trend_label": "感情のスコアの推測",
        "satisfaction_trend_label": "満足度のスコアの推測",
        "case_trends_title": "過去に推定されたスコア",
        "date_label": "日付",
        "score_label": "スコア (0-100)",
        "customer_characteristics_title": "顧客の性格",
        "language_label": "言語",
        "email_provided_label": "提供されたメールアドレス",
        "phone_provided_label": "提供された電話番号",
        "region_label": "地域",
        "btn_request_phone_summary": "履歴を要約する",

        # --- 추가된 전화 발신 기능 관련 ---
        "button_call_outbound": "電話発信",
        "call_outbound_system_msg": "📌 システムメッセージ: エージェントが{target}へ電話発信を試みました。",
        "call_outbound_simulation_header": "📞 電話発信シミュレーション結果",
        "call_outbound_summary_header": "📞 現地業者/顧客との通話要約",
        "call_outbound_loading": "電話接続と通話結果の整理中... (LLMコール)",
        "call_target_customer": "顧客へ電話発信",
        "call_target_partner": "現地業者へ電話発信",

        # --- Voice ---
        "voice_rec_header": "音声記録＆管理",
        "record_help": "録音するか音声ファイルをアップロードします。",
        "uploaded_file": "音声ファイルをアップロード",
        "rec_list_title": "保存された音声記録",
        "transcribe_btn": "転写 (Whisper)",
        "save_btn": "音声記録を保存",
        "transcribing": "音声を転写中...",
        "transcript_result": "転写結果:",
        "transcript_text": "転写テキスト",
        "openai_missing": "OpenAI APIキーがありません。",
        "whisper_client_error": "❌ エラー: Whisper APIクライアントが初期化されていません。",
        "whisper_auth_error": "❌ Whisper API認証に失敗しました。",
        "whisper_format_error": "❌ エラー: この音声形式はサポートされていません。",
        "whisper_success": "✅ 音声転写完了！",
        "playback": "録音再生",
        "retranscribe": "再転写",
        "delete": "削除",
        "no_records": "保存された音声記録はありません。",
        "saved_success": "保存しました！",
        "delete_confirm_rec": "この音声記録を削除しますか？",
        "gcs_not_conf": "GCSが設定されていないか、音声がありません。",
        "gcs_playback_fail": "音声の再生に失敗しました。",
        "gcs_no_audio": "音声ファイルがありません。",
        "error": "エラー:",
        "firestore_no_db_connect": "DB接続失敗",
        "save_history_success": "保存完了。",
        "save_history_fail": "保存失敗。",
        "delete_fail": "削除失敗",
        "rec_header": "音声入力＆転写",
        "whisper_processing": "処理中...",
        "empty_response_warning": "応答を入力してください。",
        "customer_no_more_inquiries": "いいえ、結構です。大丈夫です。有難う御座いました。",
        "customer_has_additional_inquiries": "はい、追加の問い合わせがあります。",
        "sim_end_chat_button": "アンケートリンクを送信して応対終了",
        "delete_mic_record": "録音を削除する",

        # --- 첨부 파일 기능 추가 ---
        "attachment_label": "顧客の添付ファイルアップロード (スクリーンショットなど)",
        "attachment_placeholder": "ファイルを添付して状況を説明してください（オプション）",
        "attachment_status_llm": "顧客が **{filename}** 파일을 첨부했습니다. 이 파일을 스크린샷이라고 가정하고 응대 초안과 가이드라인에 반영해주세요. (ファイルタイプ: {filetype})",
        "agent_attachment_label": "エージェント添付ファイル (スクリーンショットなど)",
        "agent_attachment_placeholder": "応答に添付するファイルを選択してください（オプション）",
        "agent_attachment_status": "📎 エージェントが **{filename}** ファイルを応答に添付しました。(ファイルタイプ: {filetype})",

        # --- RAG 오류 메시지 추가 ---
        "rag_embed_error_openai": "RAG embedding failed: OpenAI API Key is invalid or not set。",
        "rag_embed_error_gemini": "RAG embedding failed: Gemini API Key is invalid or not set。",
        "rag_embed_error_nvidia": "RAG embedding failed: NVIDIA API Key is invalid or not set。",
        "rag_embed_error_none": "RAG embedding failed: All required keys (OpenAI, Gemini, NVIDIA) are invalid or not set。 Please configure a key。",

        # --- 電話機能関連追加 ---
        "phone_header": "AI顧客対応シミュレーター(電話)",
        "call_status_waiting": "着信待ち...",
        "call_status_ringing": "着信中: {number}",
        "button_answer": "📞 電話に出る",
        "button_hangup": "🔴 電話を切る",
        "button_hold": "⏸️ 保留 (ノイズ遮断)",
        "button_resume": "▶️ 通話再開",
        "hold_status": "保留中 (累計保留時間: {duration})",
        "cc_live_transcript": "🎤 リアルタイムCC字幕 / 転写",
        "mic_input_status": "🎙️ エージェントの音声入力",
        "customer_audio_playback": "🗣️ 顧客の音声再生",
        "agent_response_prompt": "顧客への応答を録音してください。",
        "agent_response_stop_and_send": "⏹️録音を終了して、顧客へ転送する",
        "call_end_message": "通話が終了しました。AHTと履歴を確認してください。",
        "call_query_placeholder": "顧客からの最初の問い合わせ内容を入力してください。",
        "call_number_placeholder": "+81 90-xxxx-xxxx (仮想番号)",
        "call_summary_header": "AI 通話要約",
        "customer_audio_header": "顧客の最初の問い合わせ (音声)",
        "aht_not_recorded": "⚠️ 通話開始時間が記録されていないため、AHTを計算できません。",
        "no_audio_record": "顧客の最初の音声記録はありません。",
    }
}

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
# ⭐ 추가: 전사할 오디오 바이트 임시 저장소
if "bytes_to_process" not in st.session_state:
    st.session_state.bytes_to_process = None

L = LANG[st.session_state.language]

# ⭐ 2-A. Gemini 키 초기화 (잘못된 키 잔존 방지)
if "user_gemini_key" in st.session_state and st.session_state["user_gemini_key"].startswith("AIza"):
    pass

# ========================================
# 0. 멀티 모델 API Key 안전 구조 (Secrets + Env Var만 사용)
# ========================================

# 1) 지원하는 API 목록 정의
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

# 2) 세션 초기화
for api, cfg in SUPPORTED_APIS.items():
    if cfg["session_key"] not in st.session_state:
        st.session_state[cfg["session_key"]] = ""

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "openai_gpt4"


def get_api_key(api):
    cfg = SUPPORTED_APIS[api]

    # ⭐ 1. Streamlit Secrets (.streamlit/secrets.toml) - 최우선
    try:
        if hasattr(st, "secrets") and cfg["secret_key"] in st.secrets:
            return st.secrets[cfg["secret_key"]]
    except Exception:
        pass

    # 2. Environment Variable (os.environ)
    env_key = os.environ.get(cfg["secret_key"])
    if env_key:
        return env_key

    # 3. User Input (Session State - 제거됨)
    user_key = st.session_state.get(cfg["session_key"], "")
    if user_key:
        return user_key

    return ""


# ========================================
# 1. Sidebar UI: API Key 입력 제거
# ========================================
# API Key 입력 UI는 제거하고, 환경변수와 Streamlit Secrets만 사용하도록 함.


# ========================================
# 2. LLM 클라이언트 라우팅 & 실행
# ========================================
def get_llm_client():
    """선택된 모델에 맞는 클라이언트 + 모델코드 반환"""
    model_key = st.session_state.get("selected_llm", "openai_gpt4")

    # --- OpenAI ---
    if model_key.startswith("openai"):
        key = get_api_key("openai")
        if not key: return None, None
        try:
            client = OpenAI(api_key=key)
            model_name = "gpt-4o" if model_key == "openai_gpt4" else "gpt-3.5-turbo"
            return client, ("openai", model_name)
        except Exception:
            return None, None

    # --- Gemini ---
    if model_key.startswith("gemini"):
        key = get_api_key("gemini")
        if not key: return None, None
        try:
            genai.configure(api_key=key)
            model_name = "gemini-2.5-pro" if model_key == "gemini_pro" else "gemini-2.5-flash"
            return genai, ("gemini", model_name)
        except Exception:
            return None, None

    # --- Claude ---
    if model_key.startswith("claude"):
        key = get_api_key("claude")
        if not key: return None, None
        try:
            client = Anthropic(api_key=key)
            model_name = "claude-3-5-sonnet-latest"
            return client, ("claude", model_name)
        except Exception:
            return None, None

    # --- Groq ---
    if model_key.startswith("groq"):
        from groq import Groq
        key = get_api_key("groq")
        if not key: return None, None
        try:
            client = Groq(api_key=key)
            model_name = (
                "llama3-70b-8192"
                if "llama3" in model_key
                else "mixtral-8x7b-32768"
            )
            return client, ("groq", model_name)
        except Exception:
            return None, None

    return None, None


def run_llm(prompt: str) -> str:
    """선택된 LLM으로 프롬프트 실행 (Gemini 우선순위 변경 적용)"""
    client, info = get_llm_client()

    # Note: info는 사이드바에서 선택된 주력 모델의 정보를 담고 있습니다.
    provider, model_name = info if info else (None, None)

    # Fallback 순서를 정의합니다. (Gemini 우선)
    llm_attempts = []

    # 1. Gemini를 최우선 Fallback으로 시도 (Keys 확인)
    gemini_key = get_api_key("gemini")
    if gemini_key:
        llm_attempts.append(("gemini", gemini_key, "gemini-2.5-pro" if "pro" in model_name else "gemini-2.5-flash"))

    # 2. OpenAI를 2순위 Fallback으로 시도 (Keys 확인)
    openai_key = get_api_key("openai")
    if openai_key:
        llm_attempts.append(("openai", openai_key, "gpt-4o" if "4" in model_name else "gpt-3.5-turbo"))

    # 3. Claude를 3순위 Fallback으로 시도 (Keys 확인)
    claude_key = get_api_key("claude")
    if claude_key:
        llm_attempts.append(("claude", claude_key, "claude-3-5-sonnet-latest"))

    # 4. Groq를 4순위 Fallback으로 시도 (Keys 확인)
    groq_key = get_api_key("groq")
    if groq_key:
        groq_model = "llama3-70b-8192" if "llama3" in model_name else "mixtral-8x7b-32768"
        llm_attempts.append(("groq", groq_key, groq_model))

    # ⭐ 순서 조정: 주력 모델(사용자가 사이드바에서 선택한 모델)을 가장 먼저 시도합니다.
    # 만약 주력 모델이 Fallback 리스트에 포함되어 있다면, 그 모델을 첫 순서로 올립니다.
    if provider and provider in [attempt[0] for attempt in llm_attempts]:
        # 주력 모델을 리스트에서 찾아 제거
        primary_attempt = next((attempt for attempt in llm_attempts if attempt[0] == provider), None)
        if primary_attempt:
            llm_attempts.remove(primary_attempt)
            # 주력 모델이 Gemini나 OpenAI가 아니라면, Fallback 순서와 관계없이 가장 먼저 시도하도록 삽입
            llm_attempts.insert(0, primary_attempt)

    # LLM 순차 실행
    for provider, key, model in llm_attempts:
        if not key: continue

        try:
            if provider == "gemini":
                genai.configure(api_key=key)
                gen_model = genai.GenerativeModel(model)
                resp = gen_model.generate_content(prompt)
                return resp.text

            elif provider == "openai":
                o_client = OpenAI(api_key=key)
                resp = o_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

            elif provider == "claude":
                c_client = Anthropic(api_key=key)
                resp = c_client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text

            elif provider == "groq":
                from groq import Groq
                g_client = Groq(api_key=key)
                resp = g_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

        except Exception as e:
            # 해당 API가 실패하면 다음 API로 넘어갑니다.
            print(f"LLM {provider} ({model}) failed: {e}")
            continue

    # 모든 시도가 실패했을 때
    return "❌ 모든 LLM API 키가 작동하지 않거나 할당량이 소진되었습니다."


# ========================================
# 2-A. Whisper / TTS 용 OpenAI Client 별도로 초기화
# ========================================

def init_openai_audio_client():
    key = get_api_key("openai")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except:
        return None


# ⭐ 최적화: LLM 클라이언트 초기화 캐싱 (매번 재생성하지 않도록)
# OpenAI 클라이언트 캐싱
# ⭐ 수정: 초기화 시 블로킹 방지를 위해 try-except 추가
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
# LLM 번역 함수 (Gemini 클라이언트 의존성 제거 및 강화)
# ----------------------------------------
def translate_text_with_llm(text_content: str, target_lang_code: str, source_lang_code: str) -> str:
    """
    주어진 텍스트를 LLM을 사용하여 대상 언어로 번역합니다. (안정화된 텍스트 출력)
    **수정 사항:** LLM Fallback 순서를 OpenAI 우선으로 조정하고, 응답이 비어있을 경우 명시적인 오류 메시지를 반환하도록 변경
    """
    target_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(target_lang_code, "English")
    source_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(source_lang_code, "English")

    # 순수한 텍스트 번역 결과만 출력하도록 강제
    system_prompt = (
        f"You are a professional translation AI. Translate the entire following customer support chat history "
        f"from '{source_lang_name}' to '{target_lang_name}'. "
        f"You MUST translate the content to {target_lang_name} ONLY. "
        f"Do not include any mixed languages, the source text, or any introductory/concluding remarks. "
        f"Output ONLY the translated chat history text. "
    )
    prompt = f"Original Chat History:\n\n{text_content}"

    # LLM Fallback 순서: OpenAI -> Gemini -> Claude (OpenAI를 최우선으로 조정)
    llm_attempts = [
        ("openai", get_api_key("openai"), "gpt-4o"),  # 1순위: OpenAI (가장 안정적)
        ("gemini", get_api_key("gemini"), "gemini-2.5-flash"),  # 2순위
        ("claude", get_api_key("claude"), "claude-3-5-sonnet-latest"),  # 3순위
    ]

    last_error = ""

    for provider, key, model_name in llm_attempts:
        if not key: continue

        try:
            translated_text = ""

            if provider == "openai":
                o_client = OpenAI(api_key=key)
                resp = o_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    temperature=0.1
                )
                translated_text = resp.choices[0].message.content.strip()

            elif provider == "gemini":
                genai.configure(api_key=key)
                g_model = genai.GenerativeModel(model_name)
                resp = g_model.generate_content(
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.1)
                )
                translated_text = resp.text.strip()

            elif provider == "claude":
                from anthropic import Anthropic
                c_client = Anthropic(api_key=key)
                resp = c_client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    system=system_prompt
                )
                translated_text = resp.content[0].text.strip()

            # 번역 결과가 유효한지 확인
            if translated_text:
                return translated_text
            else:
                last_error = f"Translation failed: {provider} returned empty response."
                continue  # 다음 LLM 시도

        except Exception as e:
            last_error = f"Translation API call failed with {provider} ({model_name}): {e}"  # 모델명 추가
            print(last_error)
            continue  # 다음 LLM 시도

    # 모든 시도가 실패했을 때, 상세 오류 메시지 반환
    return f"❌ Translation Error: All LLM attempts failed. Last error: {last_error or 'No active API key found.'}"


# ----------------------------------------
# Realtime Hint Generation (요청 2 반영)
# ----------------------------------------
def generate_realtime_hint(current_lang_key: str, is_call: bool = False):
    """현재 대화 맥락을 기반으로 에이전트에게 실시간 응대 힌트(키워드/정책/액션)를 제공"""
    L = LANG[current_lang_key]
    # 채팅/전화 구분하여 이력 사용
    if is_call:
        # 전화 시뮬레이터에서는 현재 CC 영역에 표시된 텍스트와 초기 문의를 함께 사용
        history_text = (
            f"Initial Query: {st.session_state.call_initial_query}\n"
            f"Previous Customer Utterance: {st.session_state.current_customer_audio_text}\n"
            f"Previous Agent Utterance: {st.session_state.current_agent_audio_text}"
        )
    else:
        history_text = get_chat_history_for_prompt(include_attachment=True)

    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    hint_prompt = f"""
You are an AI Supervisor providing an **urgent, internal hint** to a human agent whose AHT is being monitored.
Analyze the conversation history, especially the customer's last message, which might be about complex issues like JR Pass, Universal Studio Japan (USJ), or a complex refund policy.

Provide ONE concise, actionable hint for the agent. The purpose is to save AHT time.

Output MUST be a single paragraph/sentence in {lang_name} containing actionable advice.
DO NOT use markdown headers or titles.
Do NOT direct the agent to check the general website.
Provide an actionable fact or the next specific step (e.g., check policy section, confirm coverage).

Examples of good hints (based on the content):
- Check the official JR Pass site for current exchange rates.
- The 'Universal Express Pass' is non-refundable; clearly cite policy section 3.2.
- Ask for the order confirmation number before proceeding with any action.
- The solution lies in the section of the Klook site titled '~'.

Conversation History:
{history_text}

HINT:
"""
    if not st.session_state.is_llm_ready:
        return "(Mock Hint: LLM Key is missing. Ask the customer for the booking number.)"

    with st.spinner(f"💡 {L['button_request_hint']}..."):
        try:
            return run_llm(hint_prompt).strip()
        except Exception as e:
            return f"❌ Hint Generation Error. (Try again or check API Key: {e})"


def generate_agent_response_draft(current_lang_key: str) -> str:
    """고객 응답을 기반으로 AI가 에이전트 응답 초안을 생성 (요청 1 반영)"""
    L = LANG[current_lang_key]
    history_text = get_chat_history_for_prompt(include_attachment=True)
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # 고객의 최신 문의 내용 추출 및 분석 (강화)
    latest_customer_message = ""
    initial_customer_query = st.session_state.get('customer_query_text_area', '')
    customer_query_analysis = ""
    
    # 모든 고객 메시지 수집
    all_customer_messages = []
    if st.session_state.simulator_messages:
        all_customer_messages = [msg["content"] for msg in st.session_state.simulator_messages 
                                if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]]
    
    # 초기 문의도 포함
    if initial_customer_query and initial_customer_query not in all_customer_messages:
        all_customer_messages.insert(0, initial_customer_query)
    
    if all_customer_messages:
        latest_customer_message = all_customer_messages[-1]
        
        # 문의 내용에서 핵심 정보 추출 (간단한 키워드 추출)
        inquiry_keywords = []
        inquiry_text = " ".join(all_customer_messages).lower()
        
        # 일반적인 문의 키워드 패턴
        important_patterns = [
            r'\b\d{4,}\b',  # 주문번호, 전화번호 등 숫자
            r'\b(주문|order|注文)\b',
            r'\b(환불|refund|返金)\b',
            r'\b(취소|cancel|キャンセル)\b',
            r'\b(배송|delivery|配送)\b',
            r'\b(변경|change|変更)\b',
            r'\b(문제|problem|issue|問題)\b',
            r'\b(도움|help|助け)\b',
        ]
        
        # 핵심 문의 내용 요약
        inquiry_summary = f"""
**CUSTOMER INQUIRY DETAILS:**

Initial Query: "{initial_customer_query if initial_customer_query else 'Not provided'}"

Latest Customer Message: "{latest_customer_message}"

All Customer Messages Context:
{chr(10).join([f"- {msg[:150]}..." if len(msg) > 150 else f"- {msg}" for msg in all_customer_messages[-3:]])}

**YOUR RESPONSE MUST DIRECTLY ADDRESS:**

1. **SPECIFIC ISSUE IDENTIFICATION**: 
   - What EXACT problem or question did the customer mention?
   - Extract and reference specific details: order numbers, dates, product names, locations, error messages, etc.
   - If multiple issues were mentioned, address EACH one specifically

2. **CONCRETE SOLUTION PROVIDED**:
   - Provide STEP-BY-STEP instructions tailored to their EXACT situation
   - Include specific actions they need to take (e.g., "Go to Settings > Account > Order History and click on order #12345")
   - Reference the exact products/services they mentioned
   - If they mentioned a location, reference it in your solution

3. **PERSONALIZATION**:
   - Use the customer's specific words/phrases when appropriate
   - Reference their exact situation (e.g., "Since you mentioned your eSIM isn't activating in Paris...")
   - Acknowledge their specific concern or frustration point

4. **COMPLETENESS**:
   - Answer ALL questions they asked
   - Address ALL problems they mentioned
   - If they asked "why", explain the specific reason for their situation
   - If they asked "how", provide detailed steps for their exact case

**CRITICAL REQUIREMENTS:**
- DO NOT use generic templates like "Thank you for contacting us" without addressing their specific issue
- DO NOT give vague answers like "Please check your settings" - be SPECIFIC about which settings and what to do
- DO NOT ignore specific details they mentioned (order numbers, dates, locations, etc.)
- Your response must read as if it was written SPECIFICALLY for this customer's exact inquiry
- If the customer mentioned "eSIM activation in Paris", your response MUST specifically address eSIM activation and Paris
- If the customer mentioned an order number, your response MUST reference that order number

**EXAMPLE OF GOOD RESPONSE:**
Bad: "Thank you for contacting us. We understand your concern and will help you resolve this issue."
Good: "I understand you're having trouble activating your eSIM in Paris. Let me help you resolve this step by step. First, please check if your phone's APN settings are configured correctly for the Paris network..."

**NOW GENERATE YOUR RESPONSE** following these requirements:
"""
        
        customer_query_analysis = inquiry_summary

    # 첨부 파일 컨텍스트 추가
    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        attachment_context = f"\n[고객 첨부 파일 정보: {attachment_context}]\n"
    else:
        attachment_context = ""

    # 고객 유형 및 반복 불만 패턴 분석
    customer_type = st.session_state.get('customer_type_sim_select', '일반적인 문의')
    is_difficult_customer = customer_type in ["까다로운 고객", "매우 불만족스러운 고객", "Difficult Customer",
                                              "Highly Dissatisfied Customer", "難しい顧客", "非常に不満な顧客"]

    # 고객 메시지 수 및 감정 분석
    customer_message_count = sum(
        1 for msg in st.session_state.simulator_messages if msg.get("role") in ["customer", "customer_rebuttal"])
    agent_message_count = sum(1 for msg in st.session_state.simulator_messages if msg.get("role") == "agent_response")

    # 이전 에이전트 응답들 추출 (다양성 확보를 위해)
    previous_agent_responses = [msg["content"] for msg in st.session_state.simulator_messages 
                                if msg.get("role") == "agent_response"]
    previous_responses_context = ""
    if previous_agent_responses:
        previous_responses_context = f"\n[이전 에이전트 응답들 (참고용, 동일하게 반복하지 말 것):\n"
        for i, prev_resp in enumerate(previous_agent_responses[-3:], 1):  # 최근 3개만
            prev_resp_preview = prev_resp[:200] + "..." if len(prev_resp) > 200 else prev_resp
            previous_responses_context += f"{i}. {prev_resp_preview}\n"
        previous_responses_context += "]\n"

    # 고객이 계속 따지거나 화내는 패턴 감지 (고객 메시지가 에이전트 메시지보다 많거나, 반복적인 불만 표현)
    is_repeating_complaints = False
    if customer_message_count > agent_message_count and customer_message_count >= 2:
        # 마지막 2개 고객 메시지 분석
        recent_customer_messages = [msg["content"].lower() for msg in st.session_state.simulator_messages if
                                    msg.get("role") in ["customer", "customer_rebuttal"]][-2:]
        complaint_keywords = ["왜", "이유", "설명", "말이 안", "이해가 안", "화나", "짜증", "불만", "왜", "why", "reason", "explain",
                              "angry", "frustrated", "complaint", "なぜ", "理由", "説明", "怒り", "不満"]
        if any(any(keyword in msg for keyword in complaint_keywords) for msg in recent_customer_messages):
            is_repeating_complaints = True

    # 대처법 포메이션 추가 여부 결정
    needs_coping_strategy = is_difficult_customer or (is_repeating_complaints and customer_message_count >= 2)

    # 대처법 가이드라인 생성
    coping_guidance = ""
    if needs_coping_strategy:
        coping_guidance = f"""

[CRITICAL: Handling Difficult Customer Situation]
The customer type is "{customer_type}" and the customer has sent {customer_message_count} messages while the agent has sent {agent_message_count} messages.
The customer may be showing signs of continued frustration or dissatisfaction.

**INCLUDE THE FOLLOWING COPING STRATEGY FORMAT IN YOUR RESPONSE:**

1. **Immediate Acknowledgment** (1-2 sentences):
   - Acknowledge their frustration/specific concern explicitly
   - Show deep empathy and understanding
   - Example formats:
     * "{'죄송합니다. 불편을 드려 정말 죄송합니다. 고객님의 상황을 충분히 이해하고 있습니다.' if current_lang_key == 'ko' else ('I sincerely apologize for the inconvenience. I fully understand your situation and frustration.' if current_lang_key == 'en' else '大変申し訳ございません。お客様の状況とご不便を十分に理解しております。')}"
     * "{'고객님의 소중한 의견을 잘 듣고 있습니다. 정말 답답하셨을 것 같습니다.' if current_lang_key == 'ko' else ('I hear your concerns clearly. This must have been very frustrating for you.' if current_lang_key == 'en' else 'お客様のご意見をしっかりと受け止めています。本当にお困りだったと思います。')}"

2. **Specific Solution Recap** (2-3 sentences):
   - Clearly restate the solution/step provided previously (if any)
   - Offer a NEW concrete action or alternative solution
   - Be specific and actionable
   - Example formats:
     * "{'앞서 안내드린 [구체적 해결책] 외에도, [새로운 대안/추가 조치]를 진행해드릴 수 있습니다.' if current_lang_key == 'ko' else ('In addition to the [specific solution] I mentioned earlier, I can also [new alternative/additional action] for you.' if current_lang_key == 'en' else '先ほどご案内した[具体的解決策]に加えて、[新しい代替案/追加措置]も進めることができます。')}"
     * "{'혹시 [구체적 문제점] 때문에 불편하셨다면, [구체적 해결 방법]을 바로 진행해드리겠습니다.' if current_lang_key == 'ko' else ('If you are experiencing [specific issue], I can immediately proceed with [specific solution].' if current_lang_key == 'en' else 'もし[具体的問題]でご不便でしたら、[具体的解決方法]をすぐに進めさせていただきます。')}"

3. **Escalation or Follow-up Offer** (1-2 sentences):
   - Offer to escalate to supervisor/higher level support
   - Promise immediate follow-up within specific time
   - Example formats:
     * "{'만약 여전히 불만이 해소되지 않으신다면, 즉시 상급 관리자에게 이관하여 더 나은 해결책을 찾아드리겠습니다.' if current_lang_key == 'ko' else ('If your concern is still not resolved, I can immediately escalate this to a supervisor to find a better solution.' if current_lang_key == 'en' else 'もしご不満が解消されない場合は、すぐに上級管理者にエスカレートして、より良い解決策を見つけさせていただきます。')}"
     * "{'24시간 이내에 [구체적 조치/결과]를 확인하여 고객님께 다시 연락드리겠습니다.' if current_lang_key == 'ko' else ('I will follow up with you within 24 hours regarding [specific action/result].' if current_lang_key == 'en' else '24時間以内に[具体的措置/結果]を確認し、お客様に再度ご連絡いたします。')}"

4. **Closing with Assurance** (1 sentence):
   - Reassure that their concern is being taken seriously
   - Example formats:
     * "{'고객님의 모든 문의사항을 최우선으로 처리하겠습니다.' if current_lang_key == 'ko' else ('I will prioritize resolving all of your concerns.' if current_lang_key == 'en' else 'お客様のすべてのご質問を最優先で処理いたします。')}"

**IMPORTANT NOTES:**
- DO NOT repeat the exact same solution that was already provided
- DO NOT sound dismissive or automated
- DO sound genuinely concerned and willing to go the extra mile
- If policy restrictions exist, acknowledge them but still offer alternatives
- Use warm, respectful tone while being firm about what can/cannot be done

**RESPONSE STRUCTURE:**
[Immediate Acknowledgment]
[Specific Solution Recap + New Action]
[Escalation/Follow-up Offer]
[Closing with Assurance]

Now generate the agent's response draft following this structure:
"""

    # 다양성 확보를 위한 추가 지시사항 (더 강화)
    diversity_instruction = ""
    if previous_agent_responses:
        # 이전 응답들의 주요 키워드/구문 추출 (반복 방지)
        previous_keywords = []
        for prev_resp in previous_agent_responses[-3:]:
            # 간단한 키워드 추출 (2-3단어 구문)
            words = prev_resp.split()[:20]  # 처음 20단어만
            for i in range(len(words) - 1):
                if len(words[i]) > 3 and len(words[i+1]) > 3:
                    previous_keywords.append(f"{words[i]} {words[i+1]}")
        
        keywords_warning = ""
        if previous_keywords:
            unique_keywords = list(set(previous_keywords))[:10]  # 최대 10개만
            keywords_warning = f"\n- AVOID using these exact phrases from previous responses: {', '.join(unique_keywords[:5])}"
        
        diversity_instruction = f"""
**CRITICAL DIVERSITY REQUIREMENT - STRICTLY ENFORCED:**
- You MUST generate a COMPLETELY DIFFERENT response from ALL previous agent responses shown above
- Use COMPLETELY DIFFERENT wording, phrasing, sentence structures, and vocabulary
- If similar solutions are needed, present them in a COMPLETELY DIFFERENT way or from a COMPLETELY DIFFERENT angle
- Vary your opening sentences, transition phrases, and closing statements - NO REPETITION
- DO NOT copy, paraphrase, or reuse ANY phrases from previous responses - be CREATIVE and UNIQUE while maintaining professionalism
- Consider COMPLETELY different approaches: 
  * If previous responses were formal, try a warmer, more personal tone (or vice versa)
  * If previous responses focused on one aspect, emphasize a COMPLETELY different aspect this time
  * Use different examples, analogies, or explanations
  * Change the order of information presentation
  * Use different sentence lengths and structures
{keywords_warning}
- IMPORTANT: Read ALL previous responses carefully and ensure your response is DISTINCTLY different in style, tone, structure, and content
- If you find yourself writing something similar to a previous response, STOP and rewrite it completely differently
"""

    # 랜덤 요소 추가를 위한 변형 지시사항
    variation_approaches = [
        "Start with a different greeting or acknowledgment style",
        "Use a different problem-solving approach or framework",
        "Present information in a different order",
        "Use different examples or analogies",
        "Vary the level of formality or warmth",
        "Focus on different aspects of the solution",
        "Use different transition words and phrases",
        "Change the length and complexity of sentences"
    ]
    selected_approaches = random.sample(variation_approaches, min(3, len(variation_approaches)))
    variation_note = "\n".join([f"- {approach}" for approach in selected_approaches])

    draft_prompt = f"""
You are an AI assistant helping a customer support agent write a professional, tailored response.

**PRIMARY OBJECTIVE:**
Generate a response draft that is SPECIFICALLY tailored to the customer's EXACT inquiry, providing concrete, actionable solutions that directly address their specific situation. The response must read as if it was written personally for this customer's unique case.

**CRITICAL REQUIREMENTS (IN ORDER OF PRIORITY):**
1. **MOST CRITICAL**: Address the customer's SPECIFIC inquiry/question with DETAILED, ACTIONABLE solutions
   - Extract and reference specific details from their message (order numbers, dates, product names, locations, error messages, etc.)
   - Provide step-by-step instructions tailored to their EXACT situation
   - Answer ALL questions they asked completely
   - Address ALL problems they mentioned specifically

2. The response MUST be in {lang_name}

3. Be professional, empathetic, and solution-oriented

4. If the customer asked a question, provide a COMPLETE and SPECIFIC answer - do NOT give vague or generic responses
   - Bad: "Please check your settings"
   - Good: "Please go to Settings > Mobile Network > APN Settings and ensure the APN is set to 'internet'"

5. If the customer mentioned a problem, acknowledge it SPECIFICALLY and provide STEP-BY-STEP solutions
   - Reference their exact problem description
   - Provide solutions that directly address their specific issue

6. Reference specific details from their inquiry (order numbers, dates, products, locations, etc.) if mentioned
   - If they mentioned "order #12345", your response MUST include "order #12345"
   - If they mentioned "Paris", your response should reference Paris specifically
   - If they mentioned "eSIM", address eSIM specifically, not just "SIM card"

7. Keep the tone appropriate for the customer type: {customer_type}

8. Do NOT include any markdown formatting, just plain text

9. {f'**FOLLOW THE COPING STRATEGY FORMAT BELOW**' if needs_coping_strategy else 'Use natural, conversational flow'}

10. **CRITICAL**: Generate a COMPLETELY UNIQUE and VARIED response - avoid repeating ANY similar phrases, structures, or approaches from previous responses

11. **CRITICAL**: Your response must be HIGHLY RELEVANT to the customer's specific inquiry - generic template responses are NOT acceptable
    - DO NOT start with generic greetings without immediately addressing their specific issue
    - DO NOT use placeholder text like "[specific solution]" - provide ACTUAL specific solutions
    - Your response should make the customer feel like you read and understood THEIR specific message

**VARIATION TECHNIQUES TO APPLY:**
{variation_note}

{customer_query_analysis}

**FULL CONVERSATION HISTORY:**
{history_text}
{attachment_context}

**PREVIOUS RESPONSES TO AVOID REPEATING:**
{previous_responses_context if previous_responses_context else "No previous responses to compare against."}

**DIVERSITY REQUIREMENTS:**
{diversity_instruction if diversity_instruction else "This is the first response, so no previous responses to avoid."}

{coping_guidance if needs_coping_strategy else ''}

**YOUR TASK:**
Generate the agent's response draft NOW. The response must:

1. **FIRST**: Read the customer inquiry analysis above CAREFULLY and identify:
   - What is their EXACT problem or question?
   - What specific details did they mention (order numbers, dates, locations, products)?
   - What do they need help with specifically?

2. **SECOND**: Write a response that:
   - Addresses their EXACT problem/question (not a generic version)
   - References the specific details they mentioned
   - Provides concrete, actionable steps tailored to their situation
   - Answers ALL their questions completely
   - Makes them feel understood

3. **THIRD**: Ensure the response is:
   - COMPLETELY DIFFERENT and UNIQUE from any previous responses
   - Professional, empathetic, and solution-oriented
   - Written in {lang_name}
   - Free of markdown formatting

**BEFORE YOU WRITE**: Ask yourself:
- "Does this response address the customer's SPECIFIC inquiry?"
- "Would a generic template response work here?" (If yes, rewrite it to be more specific)
- "Does this response reference specific details from the customer's message?"
- "Would the customer feel like I read and understood THEIR specific message?"

**NOW GENERATE THE RESPONSE:**
"""

    if not st.session_state.is_llm_ready:
        return ""

    try:
        draft = run_llm(draft_prompt).strip()
        # 마크다운 제거 (``` 등)
        if draft.startswith("```"):
            lines = draft.split("\n")
            draft = "\n".join(lines[1:-1]) if len(lines) > 2 else draft
        return draft
    except Exception as e:
        return f"❌ 응답 초안 생성 오류: {e}"


# ⭐ 새로운 함수: 전화 발신 시뮬레이션 요약 생성
def generate_outbound_call_summary(customer_query: str, current_lang_key: str, target: str) -> str:
    """
    Simulates an outbound call to a local partner or customer and generates a summary of the outcome.
    """
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # Get the current chat history for context
    history_text = get_chat_history_for_prompt(include_attachment=True)
    if not history_text:
        history_text = f"Initial Customer Query: {customer_query}"

    # Policy context (from supervisor) should be included to guide the outcome
    policy_context = st.session_state.supervisor_policy_context or ""

    summary_prompt = f"""
You are an AI simulating a quick, high-stakes phone call placed by the customer support agent to a '{target}' (either a local partner/vendor or the customer).

The purpose of the call is to resolve a complex, policy-restricted issue (like an exceptional refund for a non-refundable item, or urgent confirmation of an airport transfer change).

Analyze the conversation history, the initial query, and any provided supervisor policy.
Generate a concise summary of the OUTCOME of this simulated phone call.
The summary MUST be professional and strictly in {lang_name}.

[CRITICAL RULE]: For non-refundable items (e.g., Universal Studio Express Pass, non-refundable hotel/transfer), the local partner should only grant an exception IF the customer has provided strong, unavoidable proof (like a flight cancellation notice, doctor's note, or natural disaster notice). If no such proof is evident in the chat history, the outcome should usually be a denial or a request for more proof, but keep the tone professional.
If the customer's query is about Airport Transfer change, the outcome should be: 'Confirmation complete. Change is approved/denied based on partner policy.'

Conversation History:
{history_text}

Supervisor Policy Context (If any):
{policy_context}

Target of Call: {target}

Generate the phone call summary (Outcome ONLY):
"""
    if not st.session_state.is_llm_ready:
        return f"❌ LLM Key missing. (Simulated Outcome: The {target} requested the agent to send proof via email.)"

    try:
        summary = run_llm(summary_prompt).strip()
        # 마크다운 제거 (``` 등)
        if summary.startswith("```"):
            lines = summary.split("\n")
            summary = "\n".join(lines[1:-1]) if len(lines) > 2 else summary
        return summary
    except Exception as e:
        return f"❌ Phone call simulation error: {e}"


# ========================================
# 3. Whisper / TTS Helper
# ========================================

def transcribe_bytes_with_whisper(audio_bytes: bytes, mime_type: str = "audio/webm", lang_code: str = "ko") -> str:
    """
    OpenAI Whisper API를 사용하여 오디오 바이트를 텍스트로 전사합니다.
    """
    L = LANG[st.session_state.language]
    client = st.session_state.openai_client
    if client is None:
        return f"❌ {L['openai_missing']}"

    whisper_lang = {"ko": "ko", "en": "en", "ja": "ja"}.get(lang_code, "en")

    # 임시 파일 저장 (Whisper API 호환성)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()

    try:
        with open(tmp.name, "rb") as f:
            res = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language=whisper_lang,
            )
        # res.text 속성이 있는지 확인하고 없으면 res 자체를 문자열로 변환
        return res.text.strip() if hasattr(res, 'text') else str(res).strip()
    except Exception as e:
        # 파일 형식 오류 등 상세 오류 처리
        return f"❌ {L['error']} Whisper: {e}"
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def transcribe_audio(audio_bytes, filename="audio.wav"):
    client = st.session_state.openai_client

    # 1️⃣ OpenAI Whisper 시도
    if client:
        try:
            import io
            bio = io.BytesIO(audio_bytes)
            bio.name = filename
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=bio,
            )
            return resp.text
        except Exception as e:
            print("Whisper OpenAI failed:", e)

    # 2️⃣ Gemini STT fallback
    try:
        genai.configure(api_key=get_api_key("gemini"))
        model = genai.GenerativeModel("gemini-2.5-flash")
        text = model.generate_content("Transcribe this audio:").text
        return text or ""
    except Exception as e:
        print("Gemini STT failed:", e)

    return "❌ STT not available"


# 역할별 TTS 음성 스타일 설정
TTS_VOICES = {
    "customer": {
        "gender": "male",
        "voice": "alloy"  # Distinct Male, Generic/Customer
    },
    "agent": {
        "gender": "female",
        "voice": "shimmer"  # Distinct Female, Professional/Agent
    },
    "supervisor": {
        "gender": "female",
        "voice": "nova"  # Another Distinct Female, Informative/Supervisor
    }
}


def synthesize_tts(text: str, lang_key: str, role: str = "agent"):
    L = LANG[lang_key]
    client = st.session_state.openai_client
    if client is None:
        return None, L["openai_missing"]

    if role not in TTS_VOICES:
        role = "agent"

    voice_name = TTS_VOICES[role]["voice"]

    try:
        # tts-1 모델 사용 (안정성)
        resp = client.audio.speech.create(
            model="tts-1",
            voice=voice_name,
            input=text
            # format="mp3"은 기본값입니다.
        )
        return resp.read(), L["tts_status_success"]

    except Exception as e:
        return None, f"{L['tts_status_error']}: {e}"


# ----------------------------------------
# TTS Helper
# ----------------------------------------

def render_tts_button(text, lang_key, role="customer", prefix="", index: int = -1):
    L = LANG[lang_key]

    # ⭐ 수정: index=-1인 경우, UUID를 사용하여 safe_key 생성
    if index == -1:
        # 이관 요약처럼 인덱스가 고정되지 않는 경우, 텍스트 해시와 세션 인스턴스 ID를 조합
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()
        session_id_part = st.session_state.get('sim_instance_id', 'default_session')
        # ⭐ 수정: 이관 요약의 경우 안정적인 키를 생성 (time.time_ns() 제거하여 매번 같은 키 생성)
        # 언어 코드도 추가하여 이관 후 언어 변경 시에도 고유성 보장
        lang_code = st.session_state.get('language', lang_key)
        safe_key = f"{prefix}_SUMMARY_{session_id_part}_{lang_code}_{content_hash}"
    else:
        # 대화 로그처럼 인덱스가 존재하는 경우 (기존 로직 유지)
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()
        safe_key = f"{prefix}_{index}_{content_hash}"

    # 재생 버튼을 누를 때만 TTS 요청
    if st.button(L["button_listen_audio"], key=safe_key):
        if not st.session_state.openai_client:
            st.error(L["openai_missing"])
            return  # 키 없으면 종료

        with st.spinner(L["tts_status_generating"]):
            try:
                audio_bytes, msg = synthesize_tts(text, lang_key, role=role)
                if audio_bytes:
                    # ⭐ st.audio 호출 시 성공한 경우에만 재생 시간을 확보
                    # Streamlit 문서: autoplay는 브라우저 정책상 사용자 상호작용 없이는 작동하지 않을 수 있음
                    try:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                        st.success(msg)
                        # ⭐ 수정: 재생이 시작될 충분한 시간을 확보하기 위해 대기 시간을 3초로 늘림
                        time.sleep(3)
                    except Exception as e:
                        st.warning(f"오디오 재생 중 오류: {e}. 오디오 파일은 생성되었지만 자동 재생에 실패했습니다.")
                        st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                        st.success(msg)
                else:
                    st.error(msg)
                    time.sleep(1)  # 에러 발생 시도 잠시 대기
            except Exception as e:
                # TTS API 호출 자체에서 예외 발생 시 (네트워크 등)
                st.error(f"❌ TTS 생성 중 치명적인 오류 발생: {e}")
                time.sleep(1)

            # 버튼 클릭 이벤트 후, 불필요한 재실행을 막기 위해 여기서 함수 종료
            return
        # [중략: TTS Helper 끝]


# ========================================
# 4. 로컬 음성 기록 Helper
# ========================================

def load_voice_records() -> List[Dict[str, Any]]:
    return _load_json(VOICE_META_FILE, [])


def save_voice_records(records: List[Dict[str, Any]]):
    _save_json(VOICE_META_FILE, records)


def save_audio_record_local(
        audio_bytes: bytes,
        filename: str,
        transcript_text: str,
        mime_type: str = "audio/webm",
        meta: Dict[str, Any] = None,
) -> str:
    records = load_voice_records()
    rec_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    ext = filename.split(".")[-1] if "." in filename else "webm"
    audio_filename = f"{rec_id}.{ext}"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    rec = {
        "id": rec_id,
        "created_at": ts,
        "filename": filename,
        "audio_filename": audio_filename,
        "size": len(audio_bytes),
        "transcript": transcript_text,
        "mime_type": mime_type,
        "language": st.session_state.language,
        "meta": meta or {},
    }
    records.insert(0, rec)
    save_voice_records(records)
    return rec_id


def delete_audio_record_local(rec_id: str) -> bool:
    records = load_voice_records()
    idx = next((i for i, r in enumerate(records) if r.get("id") == rec_id), None)
    if idx is None:
        return False
    rec = records.pop(idx)
    audio_filename = rec.get("audio_filename")
    if audio_filename:
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        try:
            os.remove(audio_path)
        except FileNotFoundError:
            pass
    save_voice_records(records)
    return True


def get_audio_bytes_local(rec_id: str):
    records = load_voice_records()
    rec = next((r for r in records if r.get("id") == rec_id), None)
    if not rec:
        raise FileNotFoundError("record not found")
    audio_filename = rec["audio_filename"]
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    with open(audio_path, "rb") as f:
        b = f.read()
    return b, rec


# ========================================
# 5. 로컬 시뮬레이션 이력 Helper (요청 4 반영)
# ========================================

def load_simulation_histories_local(lang_key: str) -> List[Dict[str, Any]]:
    histories = _load_json(SIM_META_FILE, [])
    # 현재 언어와 메시지 리스트가 유효한 이력만 필터링
    return [
        h for h in histories
        if h.get("language_key") == lang_key and (isinstance(h.get("messages"), list) or h.get("summary"))
    ]


def generate_chat_summary(messages: List[Dict[str, Any]], initial_query: str, customer_type: str,
                          current_lang_key: str) -> Dict[str, Any]:
    """채팅 내용을 AI로 요약하여 주요 정보와 점수를 추출 (요청 4)"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # 대화 내용 추출
    conversation_text = f"Initial Query: {initial_query}\n\n"
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["customer", "customer_rebuttal", "phone_exchange"]:
            conversation_text += f"Customer: {content}\n"
        elif role == "agent_response" or role == "agent":
            conversation_text += f"Agent: {content}\n"
        # supervisor 메시지는 LLM에 전달하지 않아 역할 혼동 방지

    # 폰 교환 로그는 이미 "Agent: ... | Customer: ..." 형태로 기록되므로,
    # generate_summary_for_call 함수에서 별도로 처리할 필요 없이,
    # 여기서는 범용 채팅 요약 로직을 따르도록 메시지를 정제합니다.

    summary_prompt = f"""
You are an AI analyst summarizing a customer support conversation.

Analyze the conversation and provide a structured summary in JSON format (ONLY JSON, no markdown).

Extract and score:
1. Main inquiry topic (what the customer asked about)
2. Key responses provided by the agent (list of max 3 core actions/solutions)
3. Customer sentiment score (0-100, where 0=very negative, 50=neutral, 100=very positive)
4. Customer satisfaction score (0-100, based on final response)
5. Customer characteristics:
   - Language preference (if mentioned)
   - Cultural background hints (if any)
   - Location/region (if mentioned, but anonymize specific addresses)
   - Communication style (formal/casual, brief/detailed)
6. Privacy-sensitive information (anonymize: names, emails, phone numbers, specific addresses)
   - Extract patterns only (e.g., "email provided", "phone number provided", "resides in Asia region")

Output format (JSON only):
{{
  "main_inquiry": "brief description of main issue",
  "key_responses": ["response 1", "response 2"],
  "customer_sentiment_score": 75,
  "customer_satisfaction_score": 80,
  "customer_characteristics": {{
    "language": "ko/en/ja or unknown",
    "cultural_hints": "brief description or unknown",
    "region": "general region or unknown",
    "communication_style": "formal/casual/brief/detailed"
  }},
  "privacy_info": {{
    "has_email": true/false,
    "has_phone": true/false,
    "has_address": true/false,
    "region_hint": "general region or unknown"
  }},
  "summary": "overall conversation summary in {lang_name}"
}}

Conversation:
{conversation_text}

JSON Output:
"""

    if not st.session_state.is_llm_ready:
        # Fallback summary
        return {
            "main_inquiry": initial_query[:100],
            "key_responses": [],
            "customer_sentiment_score": 50,
            "customer_satisfaction_score": 50,
            "customer_characteristics": {
                "language": current_lang_key,
                "cultural_hints": "unknown",
                "region": "unknown",
                "communication_style": "unknown"
            },
            "privacy_info": {
                "has_email": False,
                "has_phone": False,
                "has_address": False,
                "region_hint": "unknown"
            },
            "summary": f"Customer inquiry about: {initial_query[:100]}"
        }

    try:
        summary_text = run_llm(summary_prompt).strip()
        # JSON 추출 (마크다운 코드 블록 제거)
        if "```json" in summary_text:
            summary_text = summary_text.split("```json")[1].split("```")[0].strip()
        elif "```" in summary_text:
            summary_text = summary_text.split("```")[1].split("```")[0].strip()

        import json
        summary_data = json.loads(summary_text)
        return summary_data
    except Exception as e:
        # JSON 파싱 실패 시 fallback
        st.warning(f"요약 생성 중 오류 발생: {e}")
        return {
            "main_inquiry": initial_query[:100],
            "key_responses": [],
            "customer_sentiment_score": 50,
            "customer_satisfaction_score": 50,
            "customer_characteristics": {
                "language": current_lang_key,
                "cultural_hints": "unknown",
                "region": "unknown",
                "communication_style": "unknown"
            },
            "privacy_info": {
                "has_email": False,
                "has_phone": False,
                "has_address": False,
                "region_hint": "unknown"
            },
            "summary": f"Customer inquiry about: {initial_query[:100]}"
        }
        # Fallback on error
        return {
            "main_inquiry": initial_query[:100],
            "key_responses": [],
            "customer_sentiment_score": 50,
            "customer_satisfaction_score": 50,
            "customer_characteristics": {
                "language": current_lang_key,
                "cultural_hints": "unknown",
                "region": "unknown",
                "communication_style": "unknown"
            },
            "privacy_info": {
                "has_email": False,
                "has_phone": False,
                "has_address": False,
                "region_hint": "unknown"
            },
            "summary": f"Error generating summary: {str(e)}"
        }


def save_simulation_history_local(initial_query: str, customer_type: str, messages: List[Dict[str, Any]],
                                  is_chat_ended: bool, attachment_context: str, is_call: bool = False):
    """AI 요약 데이터를 중심으로 이력을 저장 (요청 4 반영)"""
    histories = _load_json(SIM_META_FILE, [])
    doc_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    # AI 요약 생성 (채팅 종료 시 또는 충분한 대화가 있을 때)
    summary_data = None
    if is_chat_ended or len(messages) > 4 or is_call:  # 전화 통화는 바로 요약 시도
        summary_data = generate_chat_summary(messages, initial_query, customer_type, st.session_state.language)

    # 요약 데이터가 생성된 경우에만 저장 (요약 중심 저장)
    if summary_data:
        # 요약 데이터에 초기 문의와 핵심 정보 포함
        data = {
            "id": doc_id,
            "initial_query": initial_query,  # 초기 문의는 유지
            "customer_type": customer_type,
            "messages": [],  # 전체 메시지는 저장하지 않음 (요약만 저장)
            "summary": summary_data,  # AI 요약 데이터 (주요 저장 내용)
            "language_key": st.session_state.language,
            "timestamp": ts,
            "is_chat_ended": is_chat_ended,
            "attachment_context": attachment_context if attachment_context else "",  # 첨부 파일 컨텍스트
            "is_call": is_call,  # 전화 여부 플래그
        }
    else:
        # 요약이 아직 생성되지 않은 경우 (진행 중인 대화), 최소한의 정보만 저장
        data = {
            "id": doc_id,
            "initial_query": initial_query,
            "customer_type": customer_type,
            "messages": messages[:10] if len(messages) > 10 else messages,  # 최근 10개만 저장
            "summary": None,  # 요약 없음
            "language_key": st.session_state.language,
            "timestamp": ts,
            "is_chat_ended": is_chat_ended,
            "attachment_context": attachment_context if attachment_context else "",
            "is_call": is_call,
        }

    # 기존 이력에 추가 (최신순)
    histories.insert(0, data)
    # 너무 많은 이력 방지 (예: 100개로 증가 - 요약만 저장하므로 용량 부담 적음)
    _save_json(SIM_META_FILE, histories[:100])
    return doc_id


def delete_all_history_local():
    _save_json(SIM_META_FILE, [])


# ========================================
# DB 저장 기능 (Word/PPTX/PDF)
# ========================================
def export_history_to_word(histories: List[Dict[str, Any]], filename: str = None) -> str:
    """이력을 Word 파일로 저장"""
    if not IS_DOCX_AVAILABLE:
        raise ImportError("Word 저장을 위해 python-docx가 필요합니다: pip install python-docx")
    
    if filename is None:
        filename = f"customer_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    filepath = os.path.join(DATA_DIR, filename)
    
    doc = DocxDocument()
    
    # 제목 추가
    title = doc.add_heading('고객 응대 이력 요약', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 각 이력 추가
    for i, hist in enumerate(histories, 1):
        # 이력 제목
        doc.add_heading(f'이력 #{i}', level=1)
        
        # 기본 정보
        doc.add_paragraph(f'ID: {hist.get("id", "N/A")}')
        doc.add_paragraph(f'날짜: {hist.get("timestamp", "N/A")}')
        doc.add_paragraph(f'초기 문의: {hist.get("initial_query", "N/A")}')
        doc.add_paragraph(f'고객 유형: {hist.get("customer_type", "N/A")}')
        doc.add_paragraph(f'언어: {hist.get("language_key", "N/A")}')
        
        summary = hist.get('summary', {})
        if summary:
            # 요약 섹션
            doc.add_heading('요약', level=2)
            doc.add_paragraph(f'주요 문의: {summary.get("main_inquiry", "N/A")}')
            doc.add_paragraph(f'핵심 응답: {", ".join(summary.get("key_responses", []))}')
            doc.add_paragraph(f'고객 감정 점수: {summary.get("customer_sentiment_score", "N/A")}/100')
            doc.add_paragraph(f'고객 만족도 점수: {summary.get("customer_satisfaction_score", "N/A")}/100')
            
            # 고객 특성
            characteristics = summary.get('customer_characteristics', {})
            doc.add_heading('고객 특성', level=2)
            doc.add_paragraph(f'언어: {characteristics.get("language", "N/A")}')
            doc.add_paragraph(f'문화적 배경: {characteristics.get("cultural_hints", "N/A")}')
            doc.add_paragraph(f'지역: {characteristics.get("region", "N/A")}')
            doc.add_paragraph(f'소통 스타일: {characteristics.get("communication_style", "N/A")}')
            
            # 개인정보 요약
            privacy = summary.get('privacy_info', {})
            doc.add_heading('개인정보 요약', level=2)
            doc.add_paragraph(f'이메일 제공: {"예" if privacy.get("has_email") else "아니오"}')
            doc.add_paragraph(f'전화번호 제공: {"예" if privacy.get("has_phone") else "아니오"}')
            doc.add_paragraph(f'주소 제공: {"예" if privacy.get("has_address") else "아니오"}')
            doc.add_paragraph(f'지역 힌트: {privacy.get("region_hint", "N/A")}')
            
            # 전체 요약
            doc.add_paragraph(f'전체 요약: {summary.get("summary", "N/A")}')
        
        # 구분선
        if i < len(histories):
            doc.add_paragraph('-' * 80)
    
    doc.save(filepath)
    return filepath


def export_history_to_pptx(histories: List[Dict[str, Any]], filename: str = None) -> str:
    """이력을 PPTX 파일로 저장"""
    if not IS_PPTX_AVAILABLE:
        raise ImportError("PPTX 저장을 위해 python-pptx가 필요합니다: pip install python-pptx")
    
    if filename is None:
        filename = f"customer_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
    filepath = os.path.join(DATA_DIR, filename)
    
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # 제목 슬라이드
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "고객 응대 이력 요약"
    subtitle.text = f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # 각 이력에 대해 슬라이드 생성
    for i, hist in enumerate(histories, 1):
        # 제목 및 내용 레이아웃
        bullet_slide_layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(bullet_slide_layout)
        shapes = slide.shapes
        
        title_shape = shapes.title
        body_shape = shapes.placeholders[1]
        
        title_shape.text = f"이력 #{i}"
        
        tf = body_shape.text_frame
        tf.text = f"ID: {hist.get('id', 'N/A')}"
        
        p = tf.add_paragraph()
        p.text = f"날짜: {hist.get('timestamp', 'N/A')}"
        p.level = 0
        
        p = tf.add_paragraph()
        p.text = f"초기 문의: {hist.get('initial_query', 'N/A')}"
        p.level = 0
        
        p = tf.add_paragraph()
        p.text = f"고객 유형: {hist.get('customer_type', 'N/A')}"
        p.level = 0
        
        summary = hist.get('summary', {})
        if summary:
            p = tf.add_paragraph()
            p.text = f"주요 문의: {summary.get('main_inquiry', 'N/A')}"
            p.level = 0
            
            p = tf.add_paragraph()
            p.text = f"고객 감정 점수: {summary.get('customer_sentiment_score', 'N/A')}/100"
            p.level = 0
            
            p = tf.add_paragraph()
            p.text = f"고객 만족도 점수: {summary.get('customer_satisfaction_score', 'N/A')}/100"
            p.level = 0
    
    prs.save(filepath)
    return filepath


def export_history_to_pdf(histories: List[Dict[str, Any]], filename: str = None) -> str:
    """이력을 PDF 파일로 저장"""
    if not IS_REPORTLAB_AVAILABLE:
        raise ImportError("PDF 저장을 위해 reportlab이 필요합니다: pip install reportlab")
    
    if filename is None:
        filename = f"customer_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(DATA_DIR, filename)
    
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # 제목 스타일
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=black,
        spaceAfter=30,
        alignment=1  # 중앙 정렬
    )
    
    # 제목 추가
    story.append(Paragraph('고객 응대 이력 요약', title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # 각 이력 추가
    for i, hist in enumerate(histories, 1):
        # 이력 제목
        story.append(Paragraph(f'이력 #{i}', styles['Heading1']))
        story.append(Spacer(1, 0.1*inch))
        
        # 기본 정보
        story.append(Paragraph(f'ID: {hist.get("id", "N/A")}', styles['Normal']))
        story.append(Paragraph(f'날짜: {hist.get("timestamp", "N/A")}', styles['Normal']))
        story.append(Paragraph(f'초기 문의: {hist.get("initial_query", "N/A")}', styles['Normal']))
        story.append(Paragraph(f'고객 유형: {hist.get("customer_type", "N/A")}', styles['Normal']))
        story.append(Paragraph(f'언어: {hist.get("language_key", "N/A")}', styles['Normal']))
        
        summary = hist.get('summary', {})
        if summary:
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph('요약', styles['Heading2']))
            story.append(Paragraph(f'주요 문의: {summary.get("main_inquiry", "N/A")}', styles['Normal']))
            story.append(Paragraph(f'핵심 응답: {", ".join(summary.get("key_responses", []))}', styles['Normal']))
            story.append(Paragraph(f'고객 감정 점수: {summary.get("customer_sentiment_score", "N/A")}/100', styles['Normal']))
            story.append(Paragraph(f'고객 만족도 점수: {summary.get("customer_satisfaction_score", "N/A")}/100', styles['Normal']))
            
            characteristics = summary.get('customer_characteristics', {})
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph('고객 특성', styles['Heading2']))
            story.append(Paragraph(f'언어: {characteristics.get("language", "N/A")}', styles['Normal']))
            story.append(Paragraph(f'문화적 배경: {characteristics.get("cultural_hints", "N/A")}', styles['Normal']))
            story.append(Paragraph(f'지역: {characteristics.get("region", "N/A")}', styles['Normal']))
            story.append(Paragraph(f'소통 스타일: {characteristics.get("communication_style", "N/A")}', styles['Normal']))
            
            privacy = summary.get('privacy_info', {})
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph('개인정보 요약', styles['Heading2']))
            story.append(Paragraph(f'이메일 제공: {"예" if privacy.get("has_email") else "아니오"}', styles['Normal']))
            story.append(Paragraph(f'전화번호 제공: {"예" if privacy.get("has_phone") else "아니오"}', styles['Normal']))
            story.append(Paragraph(f'주소 제공: {"예" if privacy.get("has_address") else "아니오"}', styles['Normal']))
            story.append(Paragraph(f'지역 힌트: {privacy.get("region_hint", "N/A")}', styles['Normal']))
            
            story.append(Paragraph(f'전체 요약: {summary.get("summary", "N/A")}', styles['Normal']))
        
        # 구분선
        if i < len(histories):
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph('-' * 80, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
    
    doc.build(story)
    return filepath


# ========================================
# 6. RAG Helper (FAISS)
# ========================================
# RAG 관련 함수는 시뮬레이터와 무관하므로 기존 코드를 유지합니다.

def load_documents(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        name = f.name
        lower = name.lower()
        if lower.endswith(".pdf"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(f.read())
            tmp.flush()
            tmp.close()
            loader = PyPDFLoader(tmp.name)
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source"] = name
            docs.extend(file_docs)
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        elif lower.endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
        elif lower.endswith(".html") or lower.endswith(".htm"):
            text = f.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embedding_model():
    if get_api_key("openai"):
        try:
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except:
            pass
    if get_api_key("gemini"):
        try:
            return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        except:
            pass
    return None


def get_embedding_function():
    """
    RAG 임베딩에 사용할 임베딩 모델을 결정합니다.
    API 키 유효성 순서: OpenAI (사용자 설정 시) -> Gemini -> NVIDIA -> HuggingFace (fallback)
    API 인증 오류 발생 시 다음 모델로 이동하도록 처리합니다.
    """

    # 1. OpenAI 임베딩 시도 (사용자가 유효한 키를 설정했을 경우)
    openai_key = get_api_key("openai")
    if openai_key:
        try:
            st.info("🔹 RAG: OpenAI Embedding 사용 중")
            return OpenAIEmbeddings(openai_api_key=openai_key)
        except Exception as e:
            st.warning(f"OpenAI 임베딩 실패 → Gemini로 Fallback: {e}")

    # 2. Gemini 임베딩 시도
    gemini_key = get_api_key("gemini")
    if IS_GEMINI_EMBEDDING_AVAILABLE and gemini_key:
        try:
            st.info("🔹 RAG: Gemini Embedding 사용 중")
            # ⭐ 수정: 모델 이름 형식을 'models/model-name'으로 수정
            return GoogleGenerativeAIEmbeddings(google_api_key=gemini_key, model="models/text-embedding-004")
        except Exception as e:
            st.warning(f"Gemini 임베딩 실패 → NVIDIA로 Fallback: {e}")

    # 3. NVIDIA 임베딩 시도
    nvidia_key = get_api_key("nvidia")
    if IS_NVIDIA_EMBEDDING_AVAILABLE and nvidia_key:
        try:
            st.info("🔹 RAG: NVIDIA Embedding 사용 중")
            # NIM 모델 사용 (실제 키가 유효해야 함)
            return NVIDIAEmbeddings(api_key=nvidia_key, model="ai-embed-qa-4")
        except Exception as e:
            st.warning(f"NVIDIA 임베딩 실패 → HuggingFace Fallback: {e}")

    # 4. HuggingFace Embeddings (Local Fallback)
    try:
        st.info("🔹 RAG: Local HuggingFace Embedding 사용 중")
        # 경량 모델 사용
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"최종 Fallback 임베딩 실패: {e}")

    st.error("❌ RAG 임베딩 실패: 사용 가능한 API Key가 없습니다.")
    return None


def build_rag_index(files):
    L = LANG[st.session_state.language]
    if not files: return None, 0

    # 임베딩 함수를 시도하는 과정에서 에러 메시지가 발생할 수 있으므로 try-except로 감쌉니다.
    try:
        embeddings = get_embedding_function()
    except Exception as e:
        st.error(f"RAG 임베딩 함수 초기화 중 치명적인 오류 발생: {e}")
        return None, 0

    if embeddings is None:
        # 어떤 임베딩 모델도 초기화할 수 없음을 알림
        error_msg = L["rag_embed_error_none"]

        # 상세 오류 정보 구성 (실제 사용 가능한 임베딩 모델이 없는 경우)
        if not get_api_key("openai"):
            error_msg += f"\n- {L['rag_embed_error_openai']}"
        if not get_api_key("gemini"):
            error_msg += f"\n- {L['rag_embed_error_gemini']}"
        if not get_api_key("nvidia"):
            error_msg += f"\n- {L['rag_embed_error_nvidia']}"

        st.error(error_msg)
        return None, 0

    # 임베딩 객체 초기화 성공 후, 데이터 로드 및 분할
    docs = load_documents(files)
    if not docs: return None, 0

    chunks = split_documents(docs)
    if not chunks: return None, 0

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # 저장
        vectorstore.save_local(RAG_INDEX_DIR)
    except Exception as e:
        # API 인증 실패 등 실제 API 호출 오류 처리
        st.error(f"RAG 인덱스 생성 중 오류: {e}")
        return None, 0

    return vectorstore, len(chunks)


def load_rag_index():
    # RAG 인덱스 로드 시에도 유효한 임베딩 함수가 필요합니다.
    try:
        embeddings = get_embedding_function()
    except Exception:
        # get_embedding_function 내에서 에러 메시지를 처리하거나 스킵하므로 여기서는 조용히 처리
        return None

    if embeddings is None:
        return None

    try:
        # allow_dangerous_deserialization=True는 필수
        vs = FAISS.load_local(RAG_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception:
        return None


def rag_answer(question: str, vectorstore: FAISS, lang_key: str) -> str:
    # RAG Answer는 LLM 클라이언트 라우팅을 사용하도록 수정
    llm_client, info = get_llm_client()
    if llm_client is None:
        return LANG[lang_key]["simulation_no_key_warning"]

    # Langchain ChatOpenAI 대신 run_llm을 사용하기 위해 prompt를 직접 구성
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content[:1500] for d in docs)

    # ⭐ RAG 다국어 인식 오류 해결: 답변 생성 모델에게 질문 언어로 일관되게 답하도록 강력히 지시
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(lang_key, "English")

    prompt = (
            f"You are a helpful AI tutor. Answer the question using ONLY the provided context.\n"
            f"The answer MUST be STRICTLY in {lang_name}, which is the language of the question.\n"
            f"If you cannot find the answer in the context, say you don't know in {lang_name}.\n"
            f"Note: The context may be in a different language, but you must still answer in {lang_name}.\n\n"
            "Question:\n" + question + "\n\n"
                                       "Context:\n" + context + "\n\n"
                                                                f"Answer (in {lang_name}):"
    )
    return run_llm(prompt)


# ========================================
# 7. LSTM Helper (간단 Mock + 시각화)
# ========================================

def load_or_train_lstm():
    # 실제 LSTM 대신 랜덤 + sin 파형 기반 Mock
    np.random.seed(42)
    n_points = 50
    ts = 60 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_points)) + np.random.normal(0, 5, n_points)
    ts = np.clip(ts, 50, 100).astype(np.float32)
    return ts





# ========================================
# 8. LLM (ChatOpenAI) for Simulator / Content
# (RAG와 동일하게 run_llm으로 통합)
# ========================================

# ConversationChain 대신 run_llm을 사용하여 메모리 기능을 수동으로 구현
# st.session_state.simulator_memory는 유지하여 대화 기록을 관리합니다.

def get_chat_history_for_prompt(include_attachment=False):
    """메모리에서 대화 기록을 추출하여 프롬프트에 사용할 문자열 형태로 반환 (채팅용)"""
    history_str = ""
    for msg in st.session_state.simulator_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "customer" or role == "customer_rebuttal":
            history_str += f"Customer: {content}\n"
        elif role == "agent_response":
            history_str += f"Agent: {content}\n"
        # supervisor 메시지는 LLM에 전달하지 않아 역할 혼동 방지
    return history_str


def generate_customer_reaction(current_lang_key: str, is_call: bool = False) -> str:
    """
    고객의 다음 반응을 생성하는 LLM 호출 (채팅 전용)
    **수정 사항:** 에이전트 정보 요청 시 필수 정보 (주문번호, eSIM, 자녀 만 나이, 취소 사유) 제공 의무를 강화함.
    """
    history_text = get_chat_history_for_prompt()
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG[current_lang_key]

    # 첨부 파일 컨텍스트 추가
    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        # LLM에게 첨부 파일 컨텍스트를 제공하되, 에이전트에게 반복하지 않도록 주의
        attachment_context = f"[INITIAL ATTACHMENT CONTEXT (for customer reference only, do not repeat to agent)]\n{attachment_context}\n\n"
    else:
        attachment_context = ""

    next_prompt = f"""
{attachment_context}
You are now ROLEPLAYING as the CUSTOMER.

Read the following conversation and respond naturally in {lang_name}.

Conversation so far:
{history_text}

RULES:
1. You are only the customer. Do not write as the agent.
2. **[CRITICAL: Mandatory Information Submission for Problem Resolution]** If the agent requested any of the following critical information, you MUST provide it:
    - Order/Booking Number (e.g., ABC123, 123456)
    - eSIM related details (e.g., Host device compatibility, local status/location, time of activation)
    - Child-related product details (e.g., Child's Date of Birth or Current Age)
    - Exception/Refund Reason (e.g., flight cancellation/delay, illness, local natural disaster)
    - **If you are a difficult customer and the agent requests this information, you MUST still provide it, but you may express frustration or impatience while doing so.**
3. **[Crucial Rule for Repetition/New Inquiry]** After the agent has provided an attempt at a solution or answer:
    - If you are still confused or the problem is not fully solved, you MUST state the remaining confusion/problem clearly and briefly. DO NOT REPEAT THE INITIAL QUERY. Focus only on the unresolved aspect or the new inquiry.
4. **[CRITICAL: Solution Acknowledgment]** If the agent provided a clear and accurate solution/confirmation:
    - You MUST respond with appreciation and satisfaction, like "{L_local['customer_positive_response']}" or similar positive acknowledgment. This applies even if you are a difficult customer.
5. If the agent's LAST message was the closing confirmation: "{L_local['customer_closing_confirm']}"
    - If you have NO additional questions: You MUST reply with "{L_local['customer_no_more_inquiries']}".
    - If you DO have additional questions: You MUST reply with "{L_local['customer_has_additional_inquiries']}" AND MUST FOLLOW UP WITH THE NEW INQUIRY DETAILS IMMEDIATELY. DO NOT just repeat that you have an additional question.
6. Do NOT repeat your initial message or previous responses unless necessary.
7. Output ONLY the customer's next message.
"""
    try:
        reaction = run_llm(next_prompt)

        # ⭐ LLM이 응답했지만 내용이 너무 짧거나 비어있을 경우, 긍정 종료 문구를 반환
        if not reaction or len(reaction.strip()) < 5:
            print("LLM returned insufficient response. Using positive closing fallback.")
            return L_local['customer_positive_response']

        return reaction.strip()
    except Exception as e:
        # ⭐ LLM 호출 자체에서 오류 발생 시 (API 키, 할당량) 긍정 종료 문구를 강제 반환
        print(f"LLM Customer Reaction generation failed: {e}. Falling back to positive closing.")
        return L_local['customer_positive_response']  # 강제 안전장치


def summarize_history_with_ai(current_lang_key: str) -> str:
    """전화 통화 로그를 정리하여 LLM에 전달하고 요약 텍스트를 받는 함수."""
    # 전화 로그는 'phone_exchange' 역할을 가지거나, 'initial_query'에 포함되어 있음

    # 1. 로그 추출
    conversation_text = ""
    initial_query = st.session_state.get("call_initial_query", "N/A")
    if initial_query and initial_query != "N/A":
        conversation_text += f"Initial Query: {initial_query}\n"

    for msg in st.session_state.simulator_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "phone_exchange":
            # phone_exchange는 "Agent: ... | Customer: ..." 형태로 이미 정리되어 있음
            conversation_text += f"{content}\n"

    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    summary_prompt = f"""
You are an AI Analyst specialized in summarizing customer phone calls. 
Analyze the full conversation log below, identify the main issue, the steps taken by the agent, and the customer's sentiment.

Provide a concise, easy-to-read summary of the key exchange STRICTLY in {lang_name}.

--- Conversation Log ---
{conversation_text}
---

Summary:
"""
    if not st.session_state.is_llm_ready:
        return "LLM Key가 없어 요약 생성이 불가합니다."

    try:
        summary = run_llm(summary_prompt)
        return summary.strip()
    except Exception as e:
        return f"❌ AI 요약 생성 오류: {e}"


def generate_customer_reaction_for_call(current_lang_key: str, last_agent_response: str) -> str:
    """전화 시뮬레이터 전용 고객 반응 생성 (간결화)"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG[current_lang_key]  # ⭐ 수정: 함수 내에서 사용할 언어 팩

    # 전화 시뮬레이터에서는 전체 simulator_messages 대신,
    # st.session_state.call_initial_query와 st.session_state.current_customer_audio_text
    # 그리고 마지막 에이전트 응답(전사 텍스트)을 사용합니다.
    history_text = (
        f"Initial Query: {st.session_state.call_initial_query}\n"
        f"Previous Customer Utterance: {st.session_state.current_customer_audio_text}\n"
        f"Agent's Last Response (Transcribed): {last_agent_response}"
    )

    call_prompt = f"""
You are now ROLEPLAYING as the CUSTOMER in a PHONE CALL.
Your goal is to respond naturally and briefly (like a real person on the phone) in {lang_name}.

Conversation context:
{history_text}

RULES:
1. Respond to the Agent's Last Response. Your reply MUST be short and conversational.
2. If the agent's response is satisfactory: Acknowledge and state you are fine, or ask for closing confirmation (e.g., "{L_local['customer_positive_response']}").
3. If the agent requested information or provided an unsatisfactory answer: Briefly state the remaining problem or provide the requested information.
4. **NEVER** output the agent's response, supervisor advice, or full context. Output ONLY the next customer utterance.
5. If the agent said the call is on hold, you MUST wait silently or acknowledge briefly. (Simulate this by just outputting a very short confirmation like "Okay.")
6. If the agent's last response was the closing confirmation: You MUST reply with "{L_local['customer_no_more_inquiries']}" or "{L_local['customer_has_additional_inquiries']}" (followed by the new query).

Customer's next brief spoken response:
"""
    try:
        reaction = run_llm(call_prompt)
        return reaction.strip()
    except Exception as e:
        return f"❌ 고객 반응 생성 오류: {e}"


def summarize_history_for_call(call_logs: List[Dict[str, str]], initial_query: str, current_lang_key: str) -> str:
    """전화 통화 로그와 초기 문의를 바탕으로 요약본을 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # 로그 재구성 (phone_exchange 역할만 사용)
    full_log_text = f"--- Initial Customer Query ---\nCustomer: {initial_query}\n"
    for log in call_logs:
        if log["role"] == "phone_exchange":
            full_log_text += f"{log['content']}\n"
        elif log["role"] == "agent" and "content" in log:
            # 최초 에이전트 인사말은 여기에 포함
            full_log_text += f"Agent (Greeting): {log['content']}\n"

    summary_prompt = f"""
You are an AI Supervisor. Analyze the following telephone support conversation log.
Provide a concise, neutral summary of the key issue, the steps taken by the agent, and the final outcome.
The summary MUST be STRICTLY in {lang_name}.

--- Conversation Log ---
{full_log_text}
---

Summary:
"""
    if not st.session_state.is_llm_ready:
        return f"❌ LLM Key is missing. Cannot generate summary. Log length: {len(full_log_text.splitlines())}"

    try:
        summary = run_llm(summary_prompt)
        return summary.strip()
    except Exception as e:
        return f"❌ Summary Generation Error: {e}"


def generate_customer_closing_response(current_lang_key: str) -> str:
    """에이전트의 마지막 확인 질문에 대한 고객의 최종 답변 생성 (채팅용)"""
    history_text = get_chat_history_for_prompt()
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG[current_lang_key]  # ⭐ 수정: 함수 내에서 사용할 언어 팩

    # 마지막 메시지가 에이전트의 종료 확인 메시지인지 확인 (프롬프트에 포함)
    closing_msg = L_local['customer_closing_confirm']

    # 첨부 파일 컨텍스트 추가
    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        attachment_context = f"[INITIAL ATTACHMENT CONTEXT (for customer reference only, do not repeat to agent)]\n{attachment_context}\n\n"
    else:
        attachment_context = ""

    final_prompt = f"""
{attachment_context}
You are now ROLEPLAYING as the CUSTOMER.

The agent's final message was the closing confirmation: "{closing_msg}".
You MUST respond to this confirmation based on the overall conversation.

Conversation history:
{history_text}

RULES:
1. If the conversation seems resolved and you have NO additional questions:
    - You MUST reply with "{L_local['customer_no_more_inquiries']}".
2. If the conversation is NOT fully resolved and you DO have additional questions (or the agent provided a cancellation denial that you want to appeal):
    - You MUST reply with "{L_local['customer_has_additional_inquiries']}" AND MUST FOLLOW UP WITH THE NEW INQUIRY DETAILS. DO NOT just repeat that you have an additional question.
3. Your reply MUST be ONLY one of the two options above, in {lang_name}.
4. Output ONLY the customer's next message (must be one of the two rule options).
"""
    try:
        reaction = run_llm(final_prompt)
        # LLM의 출력이 규칙을 따르지 않을 경우를 대비하여 강제 적용
        reaction_text = reaction.strip()
        # "추가 문의 사항도 있습니다"가 포함되어 있으면 그대로 반환 (상세 내용 포함 가정)
        if L_local['customer_no_more_inquiries'] in reaction_text:
            return L_local['customer_no_more_inquiries']
        elif L_local['customer_has_additional_inquiries'] in reaction_text:
            return reaction_text
        else:
            # LLM이 규칙을 어겼을 경우, "추가 문의 사항이 있다"고 가정하고 에이전트 턴으로 넘김
            return L_local['customer_has_additional_inquiries']
    except Exception as e:
        st.error(f"고객 최종 반응 생성 오류: {e}")
        return L_local['customer_has_additional_inquiries']  # 오류 시 에이전트 턴으로 유도


# ----------------------------------------
# Initial Advice/Draft Generation (이관 후 재사용) (요청 4 반영)
# ----------------------------------------
def generate_agent_first_greeting(lang_key: str, initial_query: str) -> str:
    """전화 통화 시작 시 에이전트의 첫 인사말을 생성 (임시 함수)"""
    L_local = LANG[lang_key]
    # 문의 내용의 첫 10자만 사용 (too long)
    topic = initial_query.strip()[:15].replace('\n', ' ')
    if len(initial_query.strip()) > 15:
        topic += "..."

    if lang_key == 'ko':
        return f"안녕하세요, {topic} 관련 문의 주셨죠? 상담원 000입니다. 무엇을 도와드릴까요?"
    elif lang_key == 'en':
        return f"Hello, thank you for calling. I see you're calling about {topic}. My name is 000. How may I help you today?"
    elif lang_key == 'ja':
        return f"お電話ありがとうございます。{topic}の件ですね。担当の000と申します。どのようなご用件でしょうか?"
    return "Hello, how may I help you?"


def analyze_customer_profile(customer_query: str, current_lang_key: str) -> Dict[str, Any]:
    """신규 고객의 문의사항과 말투를 분석하여 고객성향 점수를 실시간으로 계산 (요청 4)"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    analysis_prompt = f"""
You are an AI analyst analyzing a customer's inquiry to determine their profile and sentiment.

Analyze the following customer inquiry and provide a structured analysis in JSON format (ONLY JSON, no markdown).

Analyze:
1. Customer sentiment score (0-100, where 0=very negative/angry, 50=neutral, 100=very positive/happy)
2. Communication style (formal/casual, brief/detailed, polite/direct)
3. Urgency level (low/medium/high)
4. Customer type prediction (normal/difficult/very_dissatisfied)
5. Language and cultural hints (if any)
6. Key concerns or pain points

Output format (JSON only):
{{
  "sentiment_score": 45,
  "communication_style": "brief, direct, slightly frustrated",
  "urgency_level": "high",
  "predicted_customer_type": "difficult",
  "cultural_hints": "unknown",
  "key_concerns": ["issue 1", "issue 2"],
  "tone_analysis": "brief description of tone"
}}

Customer Inquiry:
{customer_query}

JSON Output:
"""

    if not st.session_state.is_llm_ready:
        return {
            "sentiment_score": 50,
            "communication_style": "unknown",
            "urgency_level": "medium",
            "predicted_customer_type": "normal",
            "cultural_hints": "unknown",
            "key_concerns": [],
            "tone_analysis": "Unable to analyze"
        }

    try:
        analysis_text = run_llm(analysis_prompt).strip()
        # JSON 추출
        if "```json" in analysis_text:
            analysis_text = analysis_text.split("```json")[1].split("```")[0].strip()
        elif "```" in analysis_text:
            analysis_text = analysis_text.split("```")[1].split("```")[0].strip()

        import json
        analysis_data = json.loads(analysis_text)
        return analysis_data
    except Exception as e:
        return {
            "sentiment_score": 50,
            "communication_style": "unknown",
            "urgency_level": "medium",
            "predicted_customer_type": "normal",
            "cultural_hints": "unknown",
            "key_concerns": [],
            "tone_analysis": f"Analysis error: {str(e)}"
        }


def find_similar_cases(customer_query: str, customer_profile: Dict[str, Any], current_lang_key: str,
                       limit: int = 5) -> List[Dict[str, Any]]:
    """저장된 요약 데이터에서 유사한 케이스를 찾아 반환 (요청 4)"""
    histories = load_simulation_histories_local(current_lang_key)

    if not histories:
        return []

    # 요약 데이터가 있는 케이스만 필터링
    cases_with_summary = [
        h for h in histories
        if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
           and not h.get("is_call", False)  # 전화 이력 제외
    ]

    if not cases_with_summary:
        return []

    # 유사도 계산 (간단한 키워드 매칭 + 점수 유사도)
    similar_cases = []
    query_lower = customer_query.lower()
    customer_sentiment = customer_profile.get("sentiment_score", 50)
    customer_style = customer_profile.get("communication_style", "")

    for case in cases_with_summary:
        summary = case.get("summary", {})
        main_inquiry = summary.get("main_inquiry", "").lower()
        case_sentiment = summary.get("customer_sentiment_score", 50)
        case_satisfaction = summary.get("customer_satisfaction_score", 50)

        # 유사도 점수 계산
        similarity_score = 0

        # 1. 문의 내용 유사도 (키워드 매칭)
        query_words = set(query_lower.split())
        inquiry_words = set(main_inquiry.split())
        if query_words and inquiry_words:
            word_overlap = len(query_words & inquiry_words) / len(query_words | inquiry_words)
            similarity_score += word_overlap * 40

        # 2. 감정 점수 유사도
        sentiment_diff = abs(customer_sentiment - case_sentiment)
        sentiment_similarity = max(0, 1 - (sentiment_diff / 100)) * 30
        similarity_score += sentiment_similarity

        # 3. 만족도 점수 (높을수록 좋은 케이스)
        satisfaction_bonus = (case_satisfaction / 100) * 30
        similarity_score += satisfaction_bonus

        if similarity_score > 30:  # 최소 유사도 임계값
            similar_cases.append({
                "case": case,
                "similarity_score": similarity_score,
                "summary": summary
            })

    # 유사도 순으로 정렬
    similar_cases.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar_cases[:limit]


def visualize_customer_profile_scores(customer_profile: Dict[str, Any], current_lang_key: str):
    """고객 프로필 점수를 시각화 (감정 점수, 긴급도)"""
    if not IS_PLOTLY_AVAILABLE:
        return None

    L = LANG[current_lang_key]

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

    L = LANG[current_lang_key]

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

    L = LANG[current_lang_key]

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

    L = LANG[current_lang_key]

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


def generate_guideline_from_past_cases(customer_query: str, customer_profile: Dict[str, Any],
                                       similar_cases: List[Dict[str, Any]], current_lang_key: str) -> str:
    """과거 유사 케이스의 성공적인 해결 방법을 바탕으로 가이드라인 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    if not similar_cases:
        return ""

    # 유사 케이스 요약
    past_cases_text = ""
    for idx, similar_case in enumerate(similar_cases, 1):
        case = similar_case["case"]
        summary = similar_case["summary"]
        similarity = similar_case["similarity_score"]

        past_cases_text += f"""
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
"""

    if not st.session_state.is_llm_ready:
        return ""

    try:
        guideline = run_llm(guideline_prompt).strip()
        return guideline
    except Exception as e:
        return f"가이드라인 생성 오류: {str(e)}"


def _generate_initial_advice(customer_query, customer_type_display, customer_email, customer_phone, current_lang_key,
                             customer_attachment_file):
    """Supervisor 가이드라인과 초안을 생성하는 함수 (저장된 데이터 활용)"""
    L = LANG[current_lang_key]
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

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

    # 고객 프로필 분석
    customer_profile = analyze_customer_profile(customer_query, current_lang_key)

    # 유사 케이스 찾기
    similar_cases = find_similar_cases(customer_query, customer_profile, current_lang_key, limit=5)

    # 과거 케이스 기반 가이드라인 생성
    past_cases_guideline = ""
    if similar_cases:
        past_cases_guideline = generate_guideline_from_past_cases(
            customer_query, customer_profile, similar_cases, current_lang_key
        )

    # 고객 프로필 정보
    profile_block = f"""
[Customer Profile Analysis]
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
"""
    if not st.session_state.is_llm_ready:
        mock_text = (
            f"### {L['simulation_advice_header']}\n\n"
            f"- (Mock) {st.session_state.customer_type_sim_select} 유형 고객 응대 가이드입니다. (요청 3, 5, 6 반영)\n\n"
            f"### {L['simulation_draft_header']}\n\n"
            f"(Mock) 에이전트 응대 초안이 여기에 들어갑니다。\n\n"
        )
        return mock_text
    else:
        with st.spinner(L["response_generating"]):
            try:
                return run_llm(initial_prompt)
            except Exception as e:
                st.error(f"AI 조언 생성 중 오류 발생: {e}")
                return f"❌ AI Advice Generation Error: {e}"


# ========================================
# 9. 사이드바
# ========================================

with st.sidebar:
    # ⭐ 회사별 언어 우선순위 설정
    st.subheader("🌍 회사별 언어 설정")
    
    if "company_language_priority" not in st.session_state:
        st.session_state.company_language_priority = {
            "default": ["ko", "en", "ja"],
            "companies": {}
        }
    
    # 회사명 입력 및 추가
    with st.expander("회사 추가/관리"):
        new_company = st.text_input("회사명", key="new_company_input", placeholder="예: Company A")
        if st.button("회사 추가", key="add_company_btn") and new_company:
            if new_company not in st.session_state.company_language_priority["companies"]:
                st.session_state.company_language_priority["companies"][new_company] = ["ko", "en", "ja"]
                st.success(f"{new_company} 추가됨")
                # st.rerun()
    
    # 현재 회사 선택
    company_list = list(st.session_state.company_language_priority["companies"].keys())
    if company_list:
        selected_company = st.selectbox(
            "회사 선택",
            options=["기본 설정"] + company_list,
            key="selected_company"
        )
        
        if selected_company != "기본 설정":
            # 회사별 언어 우선순위 설정 (간단한 드래그 앤 드롭 대신 순서 선택)
            st.write(f"**{selected_company}** 언어 우선순위:")
            current_priority = st.session_state.company_language_priority["companies"].get(selected_company, ["ko", "en", "ja"])
            
            # 언어 순서를 multiselect로 설정 (첫 번째가 최우선)
            lang_order = st.multiselect(
                "언어 우선순위 (위에서 아래로)",
                options=["ko", "en", "ja"],
                default=current_priority,
                format_func=lambda x: {"ko": "한국어", "en": "English", "ja": "日本語"}[x],
                key=f"lang_order_{selected_company}"
            )
            
            if st.button("저장", key=f"save_priority_{selected_company}"):
                if len(lang_order) == 3:
                    st.session_state.company_language_priority["companies"][selected_company] = lang_order
                    st.success("언어 우선순위가 저장되었습니다.")
                    # st.rerun()
                else:
                    st.warning("모든 언어를 선택해주세요.")
    else:
        selected_company = None
    
    st.markdown("---")
    
    # 언어 선택 (회사별 우선순위 반영)
    # ⭐ L 변수를 먼저 정의 (기본 언어로)
    if "language" not in st.session_state:
        st.session_state.language = "ko"
    L = LANG[st.session_state.language]
    
    if selected_company and selected_company != "기본 설정" and selected_company in st.session_state.company_language_priority["companies"]:
        lang_priority = st.session_state.company_language_priority["companies"][selected_company]
    else:
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
            # st.rerun()
        else:
            # 이미 한 번 재실행했으면 플래그 초기화
            st.session_state.language_changed = False

    L = LANG[st.session_state.language]

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
        # st.rerun()
    
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
                # st.rerun()
        else:
            st.success(f"✅ {api_config.get('label', 'API Key')} 설정됨")
    
    st.markdown("---")

    st.subheader("클라이언트 초기화 상태")
    if st.session_state.llm_init_error_msg:
        st.error(st.session_state.llm_init_error_msg)
    elif st.session_state.is_llm_ready:
        st.success("✅ LLM 클라이언트 준비 완료")
    else:
        st.info("💡 API Key는 환경변수 또는 Streamlit Secrets에서 자동으로 로드됩니다.")

    if st.session_state.openai_client:
        st.success("✅ OpenAI TTS/Whisper 클라이언트 준비 완료")
    else:
        st.warning(L["openai_missing"])

    st.markdown("---")

    # ⭐ 기능 선택 - 기본값을 AI 챗 시뮬레이터로 설정
    if "feature_selection" not in st.session_state:
        st.session_state.feature_selection = L["sim_tab_chat_email"]

    # ⭐ 핵심 기능과 더보기 기능 분리 (RAG는 더보기로 이동)
    core_features = [L["sim_tab_chat_email"], L["sim_tab_phone"]]
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
    
    # 핵심 기능 표시 (시각적 구분)
    st.write("**핵심 기능**")
    for i, feature in enumerate(core_features):
        if i == current_index and current_selection in core_features:
            st.write(f"✓ {feature}")
        else:
            st.write(f"○ {feature}")
    
    # 더보기 기능
    with st.expander("더보기 기능", expanded=(current_selection in other_features)):
        for i, feature in enumerate(other_features, start=len(core_features)):
            if i == current_index and current_selection in other_features:
                st.write(f"✓ {feature}")
            else:
                st.write(f"○ {feature}")
    
    # ⭐ 하나의 통합된 선택 로직 (하나만 선택 가능)
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
L = LANG[st.session_state.language]

st.title(L["title"])

# ⭐ 프로젝트 목표 한줄 정리
st.info("🎯 **프로젝트 목표**: CS 센터 직원 교육용 AI 고객 응대 시뮬레이터 - 궁극적으로 CS 업무 시스템 대체재")

# ========================================
# 10. 기능별 페이지
# ========================================

# -------------------- Voice Record Tab --------------------
if feature_selection == L["voice_rec_header"]:
    # ... (기존 음성 기록 탭 로직 유지)
    st.header(L["voice_rec_header"])
    st.caption(L["record_help"])

    col_rec, col_list = st.columns([1, 1])

    # 녹음/업로드 + 전사 + 저장
    with col_rec:
        st.subheader(L["rec_header"])
        audio_file = st.file_uploader(
            L["uploaded_file"],
            type=["wav", "mp3", "m4a", "webm", "ogg"],
            key="voice_rec_uploader",
        )
        audio_bytes = None
        audio_mime = "audio/webm"

        if audio_file is not None:
            audio_bytes = audio_file.getvalue()
            audio_mime = audio_file.type or "audio/webm"

        # 재생
        # Streamlit 문서: bytes, 파일 경로, URL 모두 지원
        if audio_bytes:
            try:
                # MIME 타입이 올바른지 확인하고 기본값 설정
                valid_formats = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/webm", "audio/ogg", "audio/m4a"]
                if audio_mime not in valid_formats:
                    # MIME 타입이 유효하지 않으면 파일 확장자로 추정
                    audio_mime = "audio/wav"  # 기본값
                st.audio(audio_bytes, format=audio_mime)
            except Exception as e:
                st.error(f"오디오 재생 오류: {e}")
                # 기본 포맷으로 재시도
                try:
                    st.audio(audio_bytes, format="audio/wav")
                except:
                    st.error("오디오 파일을 재생할 수 없습니다.")

        # 전사 버튼
        if audio_bytes and st.button(L["transcribe_btn"]):
            if st.session_state.openai_client is None:
                st.error(L["openai_missing"])
            else:
                with st.spinner(L["transcribing"]):
                    text = transcribe_bytes_with_whisper(
                        audio_bytes, audio_mime, lang_code=st.session_state.language
                    )
                    st.session_state.last_transcript = text
                    snippet = text[:50].replace("\n", " ")
                    if len(text) > 50:
                        snippet += "..."
                    if text.startswith("❌"):
                        st.error(text)
                    else:
                        st.success(f"{L['transcript_result']} **{snippet}**")

        st.text_area(
            L["transcript_text"],
            value=st.session_state.last_transcript,
            height=150,
            key="voice_rec_transcript_area",
        )

        if audio_bytes and st.button(L["save_btn"]):
            try:
                ext = audio_mime.split("/")[-1] if "/" in audio_mime else "webm"
                filename = f"record_{int(time.time())}.{ext}"
                save_audio_record_local(
                    audio_bytes,
                    filename,
                    st.session_state.last_transcript,
                    mime_type=audio_mime,
                )
                st.success(L["saved_success"])
                st.session_state.last_transcript = ""
                # ⭐ 최적화: 버튼 클릭 후 Streamlit이 자동으로 재실행하므로 rerun 제거
            except Exception as e:
                st.error(f"{L['error']} {e}")

    # 저장된 기록 리스트
    with col_list:
        st.subheader(L["rec_list_title"])
        try:
            records = load_voice_records()
        except Exception as e:
            st.error(f"read error: {e}")
            records = []

        if not records:
            st.info(L["no_records"])
        else:
            for rec in records:
                rec_id = rec["id"]
                created_at = rec.get("created_at")
                try:
                    dt = datetime.fromisoformat(created_at)
                    created_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    created_str = str(created_at)

                transcript_snippet = (rec.get("transcript") or "")[:50].replace("\n", " ")
                if len(rec.get("transcript") or "") > 50:
                    transcript_snippet += "..."

                with st.expander(f"[{created_str}] {transcript_snippet}"):
                    st.write(f"**{L['transcript_text']}:** {rec.get('transcript') or 'N/A'}")
                    st.caption(
                        f"**Size:** {rec.get('size')} bytes | **File:** {rec.get('audio_filename')}"
                    )

                    col_p, col_r, col_d = st.columns([2, 1, 1])

                    if col_p.button(L["playback"], key=f"play_{rec_id}"):
                        try:
                            b, info = get_audio_bytes_local(rec_id)
                            mime = info.get("mime_type", "audio/webm")
                            # Streamlit 문서: bytes 데이터를 직접 전달 가능
                            # MIME 타입 검증
                            valid_formats = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/webm", "audio/ogg", "audio/m4a"]
                            if mime not in valid_formats:
                                mime = "audio/wav"  # 기본값
                            st.audio(b, format=mime, autoplay=False)
                        except Exception as e:
                            st.error(f"{L['gcs_playback_fail']}: {e}")

                    if col_r.button(L["retranscribe"], key=f"re_{rec_id}"):
                        if st.session_state.openai_client is None:
                            st.error(L["openai_missing"])
                        else:
                            with st.spinner(L["transcribing"]):
                                try:
                                    b, info = get_audio_bytes_local(rec_id)
                                    mime = info.get("mime_type", "audio/webm")
                                    new_text = transcribe_bytes_with_whisper(
                                        b, mime, lang_code=st.session_state.language
                                    )
                                    records = load_voice_records()
                                    for r in records:
                                        if r["id"] == rec_id:
                                            r["transcript"] = new_text
                                            break
                                    save_voice_records(records)
                                    st.success(L["retranscribe"] + " " + L["saved_success"])
                                    # ⭐ 최적화: 버튼 클릭 후 Streamlit이 자동으로 재실행하므로 rerun 제거
                                except Exception as e:
                                    st.error(f"{L['error']} {e}")

                    if col_d.button(L["delete"], key=f"del_{rec_id}"):
                        if st.session_state.get(f"confirm_del_{rec_id}", False):
                            ok = delete_audio_record_local(rec_id)
                            if ok:
                                st.success(L["delete_success"])
                            else:
                                st.error(L["delete_fail"])
                            st.session_state[f"confirm_del_{rec_id}"] = False
                            # ⭐ 최적화: 버튼 클릭 후 Streamlit이 자동으로 재실행하므로 rerun 제거
                        else:
                            st.session_state[f"confirm_del_{rec_id}"] = True
                            st.warning(L["delete_confirm_rec"])
                            st.write("sim_stage:", st.session_state.get("sim_stage"))
                            st.write("is_llm_ready:", st.session_state.get("is_llm_ready"))

# -------------------- Simulator (Chat/Email) Tab --------------------
elif feature_selection == L["sim_tab_chat_email"]:
    # ... (기존 채팅/이메일 시뮬레이터 로직 유지)
    st.header(L["simulator_header"])
    st.markdown(L["simulator_desc"])

    current_lang = st.session_state.language
    L = LANG[current_lang]  # 다시 L 업데이트

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
                    # ⭐ 최적화: 버튼 클릭 후 Streamlit이 자동으로 재실행하므로 rerun 제거
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
                # ⭐ 로드 후 UI 업데이트를 위해 재실행
                # st.rerun()
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
                # st.rerun()

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
            with download_col1:
                try:
                    filepath_word = export_history_to_word(current_session_history)
                    with open(filepath_word, "rb") as f:
                        st.download_button(
                            label=L["download_history_word"],
                            data=f.read(),
                            file_name=os.path.basename(filepath_word),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_word_file"
                        )
                except Exception as e:
                    st.error(f"Word 다운로드 오류: {e}")
            
            with download_col2:
                try:
                    filepath_pptx = export_history_to_pptx(current_session_history)
                    with open(filepath_pptx, "rb") as f:
                        st.download_button(
                            label=L["download_history_pptx"],
                            data=f.read(),
                            file_name=os.path.basename(filepath_pptx),
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key="download_pptx_file"
                        )
                except Exception as e:
                    st.error(f"PPTX 다운로드 오류: {e}")
            
            with download_col3:
                try:
                    filepath_pdf = export_history_to_pdf(current_session_history)
                    with open(filepath_pdf, "rb") as f:
                        st.download_button(
                            label=L["download_history_pdf"],
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
            # ⭐ 재실행
            st.rerun()
        # st.stop()

    # =========================
    # 5-A. 전화 발신 진행 중 (OUTBOUND_CALL_IN_PROGRESS)
    # =========================
    elif st.session_state.sim_stage == "OUTBOUND_CALL_IN_PROGRESS":
        L = LANG[st.session_state.language]
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
        # st.rerun()

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
            # 고객 프로필 분석 (시각화를 위해 먼저 수행)
            customer_profile = analyze_customer_profile(customer_query, current_lang)
            similar_cases = find_similar_cases(customer_query, customer_profile, current_lang, limit=5)

            # 시각화 차트 표시
            st.markdown("---")
            st.subheader("📊 고객 프로필 분석")

            # 고객 프로필 점수 차트
            profile_chart = visualize_customer_profile_scores(customer_profile, current_lang)
            if profile_chart:
                st.plotly_chart(profile_chart, use_container_width=True)
            else:
                # Plotly가 없을 경우 텍스트로 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        L.get("sentiment_score_label", "감정 점수"),
                        f"{customer_profile.get('sentiment_score', 50)}/100"
                    )
                with col2:
                    urgency_map = {"low": 25, "medium": 50, "high": 75}
                    urgency_score = urgency_map.get(customer_profile.get("urgency_level", "medium").lower(), 50)
                    st.metric(
                        L.get("urgency_score_label", "긴급도"),
                        f"{urgency_score}/100"
                    )
                with col3:
                    st.metric(
                        L.get("customer_type_label", "고객 유형"),
                        customer_profile.get("predicted_customer_type", "normal")
                    )

            # 유사 케이스 시각화
            if similar_cases:
                st.markdown("---")
                st.subheader("🔍 유사 케이스 추천")
                similarity_chart = visualize_similarity_cases(similar_cases, current_lang)
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

            # 초기 조언 생성
            text = _generate_initial_advice(
                customer_query,
                st.session_state.customer_type_sim_select,
                st.session_state.customer_email,
                st.session_state.customer_phone,
                current_lang,
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
            # ⭐ 재실행
            # st.rerun()

    # =========================
    # 4. 대화 로그 표시 (공통)
    # =========================
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

                # 번역이 실패했을 경우 (빈 문자열)
                # ⭐ 수정된 부분 1: DuplicateWidgetID 오류 해결을 위해 고유 키에 UUID 추가
                is_translation_failed = not st.session_state.transfer_summary_text or st.session_state.transfer_summary_text.startswith(
                    "❌ Translation Error")

                if is_translation_failed:
                    st.error(f"❌ LLM_TRANSLATION_ERROR (번역 실패). 상세 정보는 아래 요약 박스를 확인하세요.")
                    st.info(st.session_state.transfer_summary_text)
                    # 번역 재시도 버튼 추가
                    if st.button(L["button_retry_translation"],
                                 key=f"btn_retry_translation_{st.session_state.sim_instance_id}"):  # 고유 키 사용
                        # 재시도 로직 실행
                        with st.spinner(L["transfer_loading"]):
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

                            translated_summary = translate_text_with_llm(history_text, target_lang, source_lang)
                            st.session_state.transfer_summary_text = translated_summary
                            st.session_state.transfer_retry_count += 1

                            # ⭐ 재실행
                            # st.rerun()


                else:
                    # [수정 2] 번역 성공 시 내용 표시 및 TTS 버튼 추가
                    st.markdown(st.session_state.transfer_summary_text)
            # ⭐ 수정: 이관 요약의 경우 안정적인 키를 생성하도록 수정 (세션 ID와 언어 코드 조합)
                    render_tts_button(
                        st.session_state.transfer_summary_text,
                        st.session_state.language,
                        role="agent",
                        prefix="trans_summary_tts",
                        index=-1  # 고유 세션 ID 기반의 키를 생성하도록 지시
                    )
                st.markdown("---")

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
                    # ⭐ 재실행
                    # st.rerun()

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
                        # ⭐ 재실행하여 텍스트 영역 업데이트
                        st.rerun()
                    else:
                        st.error(ai_draft if ai_draft else L.get("draft_error", "응답 초안 생성에 실패했습니다."))

        # --- 전화 발신 버튼 추가 (요청 2 반영) ---
        st.markdown("---")
        st.subheader(L["button_call_outbound"])
        call_cols = st.columns(3)

        with call_cols[0]:
            if st.button(L["button_call_outbound"].replace("전화 발신", "현지 업체 전화 발신"), key="btn_call_outbound_partner"):
                # 전화 발신 시뮬레이션: 현지 업체
                st.session_state.sim_call_outbound_target = "현지 업체/파트너"
                st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"
                # st.rerun()

        with call_cols[1]:
            if st.button(L["button_call_outbound"].replace("전화 발신", "고객 전화 발신"), key="btn_call_outbound_customer"):
                # 전화 발신 시뮬레이션: 고객
                st.session_state.sim_call_outbound_target = "고객"
                st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"
                # st.rerun()

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
            st.info("✅ 녹음 완료! 아래 전사 버튼을 눌러 텍스트로 변환하세요.")

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
                    st.rerun()

            # 3. 전사(Whisper) 버튼 (기존 로직 대체)
            col_tr, _ = st.columns([1, 2])
            if col_tr.button(L["transcribe_btn"], key="sim_transcribe_btn"):
                if st.session_state.sim_audio_bytes is None:
                    st.warning("먼저 마이크로 녹음을 완료하세요.")
                elif st.session_state.openai_client is None:
                    st.error(L["whisper_client_error"])
                else:
                    with st.spinner(L["whisper_processing"]):
                        # transcribe_bytes_with_whisper 함수를 사용하도록 수정
                        transcribed_text = transcribe_bytes_with_whisper(
                            st.session_state.sim_audio_bytes,
                            "audio/wav",
                            lang_code=st.session_state.language,
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
                            # ⭐ 주석 처리: 전사 결과가 이미 세션 상태에 반영되었으므로 자동으로 입력창에 표시됨
                            # st.rerun()  # UI 업데이트

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

            # 입력창/오디오/첨부 파일 초기화
            # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로,
            # rerun 후 위젯이 다시 생성될 때 초기값이 적용되도록 플래그를 사용합니다.
            st.session_state.sim_audio_bytes = None
            st.session_state.agent_attachment_file = []  # 첨부 파일 초기화
            st.session_state.language_transfer_requested = False
            st.session_state.realtime_hint_text = ""  # 힌트 초기화
            st.session_state.sim_call_outbound_summary = ""  # 전화 발신 요약 초기화

            # ⭐ 수정: 고객 반응 생성 로직을 다음 단계에서 처리하도록 sim_stage 변경
            st.session_state.sim_stage = "CUSTOMER_TURN"
            
            # ⭐ 수정: agent_response_area_text는 rerun 후 위젯이 다시 생성될 때 초기화되도록
            # 플래그를 설정합니다. 위젯 생성 전에 이 플래그를 확인하여 값을 초기화합니다.
            st.session_state.reset_agent_response_area = True
            
            # ⭐ 재실행: 이 부분이 즉시 고객 반응을 생성하도록 유도합니다.
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

                # 3. LLM 번역 실행 (수정된 번역 함수 사용)
                translated_summary = translate_text_with_llm(history_text, target_lang,
                                                             current_lang_at_start)  # Use current_lang_at_start as source

                # 4. 세션 상태 업데이트
                st.session_state.transfer_summary_text = translated_summary
                st.session_state.language_at_transfer = target_lang  # Save destination language
                st.session_state.language_at_transfer_start = current_lang_at_start  # Save source language for retry
                st.session_state.language = target_lang  # Language switch

                # --- 기존 가이드라인 삭제 및 새 가이드라인 생성 (언어 통일성 확보) ---
                # 1. 기존 Supervisor Advice 메시지 삭제
                st.session_state.simulator_messages = [
                    msg for msg in st.session_state.simulator_messages
                    if msg['role'] != 'supervisor'
                ]

                # 2. 새로운 언어로 가이드라인/초안 재생성
                new_advice = _generate_initial_advice(
                    st.session_state.customer_query_text_area,
                    st.session_state.customer_type_sim_select,
                    st.session_state.customer_email,
                    st.session_state.customer_phone,
                    target_lang,  # 새로운 언어로 생성
                    st.session_state.customer_attachment_file
                )
                st.session_state.simulator_messages.append({"role": "supervisor", "content": new_advice})
                # -------------------------------------------------------------------

                st.session_state.is_solution_provided = False  # 새로운 응대를 위해 플래그 리셋
                st.session_state.language_transfer_requested = False  # 플래그 리셋
                st.session_state.sim_stage = "AGENT_TURN"

                # 5. 이력 저장
                customer_type_display = st.session_state.get("customer_type_sim_select", "")
                save_simulation_history_local(
                    st.session_state.customer_query_text_area,
                    customer_type_display + f" (Transferred from {current_lang_at_start} to {target_lang})",
                    st.session_state.simulator_messages,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                    is_chat_ended=False,
                )

            # 6. UI 재실행 (언어 변경 적용)
            st.success(f"✅ {LANG[target_lang]['transfer_summary_header']}가 준비되었습니다. 새로운 응대를 시작하세요.")
            st.rerun()


        for i, target_lang in enumerate(languages):
            button_label_key = f"transfer_to_{target_lang}"
            button_label = L.get(button_label_key, f"Transfer to {target_lang.capitalize()} Team")

            if transfer_cols[i].button(button_label, key=f"btn_transfer_{target_lang}"):
                transfer_session(target_lang, st.session_state.simulator_messages)

        st.markdown("---")

    # --- Language Transfer Buttons End ---

    # =========================
    # 6. 고객 반응 생성 단계 (CUSTOMER_TURN)
    # =========================
    elif st.session_state.sim_stage == "CUSTOMER_TURN":
        L = LANG[st.session_state.language]
        customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
        st.info(L["customer_turn_info"])

        # 1. 고객 반응 생성
        with st.spinner(L["generating_customer_response"]):
            customer_response = generate_customer_reaction(st.session_state.language, is_call=False)

        # 2. 대화 로그 업데이트
        st.session_state.simulator_messages.append(
            {"role": "customer", "content": customer_response}
        )

        # 3. 종료 조건 검토
        positive_closing_phrases = [L["customer_positive_response"], L["customer_no_more_inquiries"]]
        is_positive_closing = any(phrase in customer_response for phrase in positive_closing_phrases)

        # ⭐ 수정: 고객이 "알겠습니다. 감사합니다"라고 답변했을 때, 솔루션이 제공된 경우에만 추가 문의 여부 확인 단계로 이동
        # 정확한 문자열 비교가 아닌 포함 여부로 확인 (LLM 응답이 약간 다를 수 있음)
        if L["customer_positive_response"] in customer_response:
            # 솔루션이 제공된 경우에만 추가 문의 여부 확인 단계로 이동
            if st.session_state.is_solution_provided:
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                # 솔루션이 제공되지 않은 경우 에이전트 턴으로 유지
                st.session_state.sim_stage = "AGENT_TURN"
        elif is_positive_closing:
            # 긍정 종료 응답 처리
            if L['customer_no_more_inquiries'] in customer_response:
                # ⭐ 수정: "없습니다. 감사합니다" 답변 시 자동으로 설문 조사 링크 전송 및 응대 종료
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
                
                # 이력 저장 (종료 상태로 저장)
                save_simulation_history_local(
                    st.session_state.customer_query_text_area, customer_type_display,
                    st.session_state.simulator_messages, is_chat_ended=True,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                )
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

            # 4. 재실행
            # st.rerun()

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
        # ⭐ 재실행: 고객 반응이 추가되었으므로 상태 변경 반영
        st.rerun()


    # =========================
    # 7. 종료 확인 메시지 대기 (WAIT_CLOSING_CONFIRMATION_FROM_AGENT)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CLOSING_CONFIRMATION_FROM_AGENT":
        st.success("고객이 솔루션에 긍정적으로 반응했습니다. 추가 문의 여부를 확인해 주세요.")

        col_chat_end, col_email_end = st.columns(2)  # 버튼을 나란히 배치

        # [1] 채팅 - 추가 문의 확인 메시지 보내기 버튼
        with col_chat_end:
            # [수정 1] 다국어 레이블 사용
            if st.button(L["send_closing_confirm_button"],
                         key=f"btn_send_closing_confirm_{st.session_state.sim_instance_id}"):
                # [수정 1] 다국어 레이블 사용
                closing_msg = L["customer_closing_confirm"]

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
                st.rerun()

    # =========================
    # 8. 고객 최종 응답 생성 및 처리 (WAIT_CUSTOMER_CLOSING_RESPONSE)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CUSTOMER_CLOSING_RESPONSE":
        L = LANG[st.session_state.language]
        customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
        
        # ⭐ 수정: 이미 고객 응답이 생성되어 있는지 확인
        last_customer_message = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "customer_rebuttal":
                last_customer_message = msg.get("content", "")
                break
        
        # 고객 응답이 아직 생성되지 않은 경우에만 생성
        if last_customer_message is None:
            st.info("에이전트가 추가 문의 여부를 확인했습니다. 고객의 최종 답변을 자동으로 생성합니다.")

            # 고객 답변 자동 생성 (LLM Key 검증 포함)
            if not st.session_state.is_llm_ready:
                st.warning("LLM Key가 없어 고객 반응 자동 생성이 불가합니다. 수동으로 '고객 반응 생성' 버튼을 클릭하거나 AGENT_TURN으로 돌아가세요。")
                if st.button(L["customer_generate_response_button"], key="btn_generate_final_response"):
                    st.session_state.sim_stage = "AGENT_TURN"
                    st.rerun()
                st.stop()
            
            # LLM이 준비된 경우 고객 응답 생성
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
            st.warning("고객 응답을 생성할 수 없습니다. 다시 시도해주세요.")
        else:
            final_customer_reaction = last_customer_message
            
            # (A) "없습니다. 감사합니다" 경로 -> FINAL_CLOSING_ACTION 단계로 이동하여 버튼 표시
            if L['customer_no_more_inquiries'] in final_customer_reaction:
                # FINAL_CLOSING_ACTION 단계로 이동
                st.session_state.sim_stage = "FINAL_CLOSING_ACTION"
                st.session_state.realtime_hint_text = ""
                st.rerun()
            # (B) "추가 문의 사항도 있습니다" 경로 -> AGENT_TURN으로 복귀
            elif L['customer_has_additional_inquiries'] in final_customer_reaction:
                st.session_state.sim_stage = "AGENT_TURN"
                save_simulation_history_local(
                    st.session_state.customer_query_text_area, customer_type_display,
                    st.session_state.simulator_messages, is_chat_ended=False,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                )
                st.session_state.realtime_hint_text = ""
                st.rerun()

    # =========================
    # 9. 최종 종료 행동 (FINAL_CLOSING_ACTION)
    # =========================
    elif st.session_state.sim_stage == "FINAL_CLOSING_ACTION":
        L = LANG[st.session_state.language]
        st.success("고객이 더 이상 문의할 사항이 없다고 확인했습니다。")

        # ⭐ 수정: "설문 조사 링크 전송 및 응대 종료" 버튼 표시
        if st.button(L["sim_end_chat_button"], key="btn_final_end_chat"):
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
            st.rerun()

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
                    # st.rerun()  # 매 초마다 재실행하여 AHT 갱신

        # ========================================
        # 화면 구분 (애니메이션 / CC)
        # ========================================
    col_video, col_cc = st.columns([1, 2])

    with col_video:
        st.subheader("📺 고객 영상 시뮬레이션")

        if st.session_state.call_sim_stage == "WAITING_CALL":
            st.info("통화 수신 대기 중...")

        elif st.session_state.call_sim_stage == "CALL_ENDED":
            st.info("통화 종료")

        else:
            # ⭐ 비디오 파일 업로드 옵션 추가 (로컬 경로 지원)
            with st.expander("비디오 파일 업로드/로드", expanded=False):
                # 비디오 파일 업로드
                uploaded_video = st.file_uploader(
                    "비디오 파일 업로드 (MP4, WebM, OGG)",
                    type=["mp4", "webm", "ogg"],
                    key="customer_video_uploader"
                )
                
                # 또는 로컬 파일 경로 입력
                video_path_input = st.text_input(
                    "또는 로컬 파일 경로 입력",
                    placeholder="예: C:\\Users\\Admin\\Downloads\\video.mp4 또는 video.mp4",
                    key="video_path_input"
                )
                
                # 비디오 재생
                video_to_play = None
                video_format = "video/mp4"
                
                if uploaded_video:
                    # 업로드된 파일 사용
                    video_to_play = uploaded_video.read()
                    # 파일 확장자로 포맷 결정
                    file_ext = uploaded_video.name.split('.')[-1].lower()
                    video_format = {
                        'mp4': 'video/mp4',
                        'webm': 'video/webm',
                        'ogg': 'video/ogg'
                    }.get(file_ext, 'video/mp4')
                elif video_path_input:
                    # 로컬 파일 경로 사용
                    try:
                        # 절대 경로 또는 상대 경로 처리
                        if os.path.isabs(video_path_input):
                            video_path = video_path_input
                        else:
                            # 상대 경로는 작업 디렉토리 기준
                            video_path = os.path.join(os.getcwd(), video_path_input)
                        
                        if os.path.exists(video_path):
                            with open(video_path, "rb") as f:
                                video_to_play = f.read()
                            # 파일 확장자로 포맷 결정
                            file_ext = os.path.splitext(video_path)[1].lower().lstrip('.')
                            video_format = {
                                'mp4': 'video/mp4',
                                'webm': 'video/webm',
                                'ogg': 'video/ogg'
                            }.get(file_ext, 'video/mp4')
                        else:
                            st.error(f"파일을 찾을 수 없습니다: {video_path}")
                    except Exception as e:
                        st.error(f"비디오 파일 로드 오류: {e}")
                
                # 비디오 재생
                if video_to_play:
                    try:
                        # Streamlit 문서: bytes 데이터를 직접 전달 가능
                        st.video(video_to_play, format=video_format, autoplay=False, loop=False, muted=False)
                        st.success("✅ 비디오 로드 완료")
                    except Exception as e:
                        st.error(f"비디오 재생 오류: {e}")
                        st.info("💡 비디오가 H.264 코덱으로 인코딩되었는지 확인하세요. MP4V 코덱은 브라우저에서 지원되지 않을 수 있습니다.")
            
            # 상태 선택
            if st.session_state.is_on_hold:
                avatar_state = "HOLD"
            else:
                avatar_state = st.session_state.customer_avatar.get("state", "NEUTRAL")

            # ⭐ Lottie 제거: 로딩 문제로 인해 완전히 제거하고 간단한 텍스트로 대체
            avatar_emoji = {
                "NEUTRAL": "😐",
                "HAPPY": "😊",
                "ANGRY": "😠",
                "ASKING": "🤔",
                "HOLD": "⏸️"
            }.get(avatar_state, "😐")
            
            st.markdown(f"### {avatar_emoji} 고객 아바타")
            st.info(f"상태: {avatar_state}")

    with col_cc:
        st.markdown(
            f"## {L['call_status_ringing'].format(number=st.session_state.incoming_phone_number)}"
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

        st.markdown("---")

        col_in, col_out = st.columns(2)

        # 전화 응답 (수신)
        with col_in:
            if st.button(L["button_answer"], key=f"answer_call_btn_{st.session_state.sim_instance_id}"):
                # 입력 검증
                if not st.session_state.call_initial_query.strip():
                    st.warning(L["simulation_warning_query"])
                    # st.stop()

                if not st.session_state.is_llm_ready or st.session_state.openai_client is None:
                    st.error(L["simulation_no_key_warning"] + " " + L["openai_missing"])
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

                # ⭐ [수정 1-1] IN_CALL 진입 시 인사말 생성 플래그 활성화
                st.session_state.just_entered_call = True
                st.session_state.customer_turn_start = False

                # 고객의 첫 번째 음성 메시지 (시뮬레이션 시작 메시지)
                initial_query_text = st.session_state.call_initial_query.strip()
                st.session_state.current_customer_audio_text = initial_query_text

                # ⭐ 고객의 첫 문의 TTS 음성 생성 및 저장
                with st.spinner(L["tts_status_generating"] + " (Initial Customer Query)"):
                    audio_bytes, msg = synthesize_tts(initial_query_text, st.session_state.language, role="customer")
                    if audio_bytes:
                        st.session_state.customer_initial_audio_bytes = audio_bytes
                    else:
                        st.error(f"❌ {msg}")
                        st.session_state.customer_initial_audio_bytes = None

                # ✅ 상태 변경 후 재실행하여 IN_CALL 상태로 전환
                st.rerun()

        # 전화 발신 (새로운 세션 시작)
        with col_out:
            st.markdown(f"### {L['button_call_outbound']}")
            call_targets = [
                L["call_target_customer"],
                L["call_target_partner"]
            ]

            call_target_selection = st.radio(
                "발신 대상 선택",
                call_targets,
                key="outbound_call_target_radio",
                horizontal=True
            )

            if st.button(L["button_call_outbound"], key=f"outbound_call_start_btn_{st.session_state.sim_instance_id}", type="secondary"):
                # 입력 검증
                if not st.session_state.call_initial_query.strip():
                    st.warning("전화 발신 목표 (고객 문의 내용)를 입력해 주세요。")
                    # st.stop()

                if not st.session_state.is_llm_ready or st.session_state.openai_client is None:
                    st.error(L["simulation_no_key_warning"] + " " + L["openai_missing"])
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

                # ⭐ [수정 1-1] IN_CALL 진입 시 인사말 생성 플래그 활성화
                st.session_state.just_entered_call = True
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

                st.success(f"'{call_target_selection}'에게 전화 발신 시뮬레이션이 시작되었습니다. 에이전트의 첫 응답을 녹음하세요。")
                st.rerun()

        # ------------------
        # IN_CALL 상태 (통화 중)
        # ------------------
    elif st.session_state.call_sim_stage == "IN_CALL":
        # --------------------------------------------------------------------
        # ⭐ 1단계: 에이전트 인사말 재생 및 로그 기록 (just_entered_call=True)
        # --------------------------------------------------------------------
        if st.session_state.just_entered_call:
            initial_query = st.session_state.call_initial_query.strip()

            agent_greeting = generate_agent_first_greeting(
                st.session_state.language,
                initial_query
            )

            # 에이전트 첫 인사말 TTS (자동 재생)
            if st.session_state.openai_client and agent_greeting:
                with st.spinner(L["tts_status_generating"] + " (Agent Greeting)"):
                    audio_bytes, msg = synthesize_tts(
                        agent_greeting, st.session_state.language, role="agent"
                    )
                    if audio_bytes:
                        # Streamlit 문서: autoplay는 브라우저 정책상 제한될 수 있음
                        try:
                            st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                            st.success("✅ 에이전트 인사말 자동 재생 완료. 고객 문의 재생을 준비합니다.")
                        # ⭐ 수정: TTS 동기화 문제 방지를 위해 짧은 대기 후 rerun
                            time.sleep(1)
                        except Exception as e:
                            st.warning(f"자동 재생 실패 (브라우저 정책): {e}. 수동으로 재생해주세요.")
                            st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                            st.success("✅ 에이전트 인사말 생성 완료. 재생 버튼을 눌러주세요.")
                    else:
                        st.error(f"❌ TTS 오류: {msg}")

            # 1. CC에 에이전트 인사말 반영 및 로그 기록
            st.session_state.current_agent_audio_text = agent_greeting
            st.session_state.simulator_messages.append(
                {"role": "agent", "content": agent_greeting}
            )

            # 아바타 표정 초기화
            st.session_state.customer_avatar["state"] = "NEUTRAL"
            st.session_state.just_entered_call = False

            # 다음 단계(고객 문의 재생)로 전환
            st.session_state.customer_turn_start = True
            st.rerun()  # 다음 실행 주기에서 고객 문의가 재생되도록 유도

        # --------------------------------------------------------------------
        # ⭐ 2단계: 고객 문의 재생 및 CC 업데이트 (customer_turn_start=True)
        # --------------------------------------------------------------------
        elif st.session_state.customer_turn_start:
            customer_first_utterance = st.session_state.call_initial_query.strip()

            # 1. 고객의 첫 문의 TTS 음성 재생
            # INBOUND 시작 시 저장된 오디오를 사용하며, key 인수를 제거하여 TypeError 방지
            if st.session_state.customer_initial_audio_bytes:
                # Streamlit 문서: autoplay는 브라우저 정책상 제한될 수 있음
                try:
                    st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                except Exception as e:
                    st.warning(f"자동 재생 실패: {e}. 수동으로 재생해주세요.")
                    st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=False)
                st.success("✅ 고객의 최초 문의 재생 시작")
            else:
                st.error("❌ 고객 최초 문의 오디오를 찾을 수 없습니다. 텍스트만 표시합니다.")

            # 2. 고객의 첫 문의 CC에 반영 및 로그 기록
            st.session_state.current_customer_audio_text = customer_first_utterance
            st.session_state.current_agent_audio_text = ""  # 에이전트 CC를 비워 녹음 대기 상태로 만듦

            # 3. 로그에 교환 기록 (에이전트 인사말 + 고객 문의)
            agent_greeting = st.session_state.simulator_messages[-1]['content']
            log_entry = f"Agent (Greeting): {agent_greeting} | Customer (Initial Query): {customer_first_utterance}"
            # 기존의 "agent" 역할 메시지를 삭제하고, 교환 로그로 대체
            st.session_state.simulator_messages = [
                msg for msg in st.session_state.simulator_messages if msg.get("role") != "agent"
            ]
            st.session_state.simulator_messages.append({"role": "phone_exchange", "content": log_entry})

            # 4. 단계 종료
            st.session_state.customer_turn_start = False
            st.rerun()  # CC 반영 및 녹음 대기 상태로 최종 전환
        # ------------------------------
        # 전화 통화 제목
        # ------------------------------
        if st.session_state.call_sim_mode == "INBOUND":
            title = L['call_status_ringing'].format(number=st.session_state.incoming_phone_number)
        else:
            title = L['button_call_outbound'] + f" ({st.session_state.incoming_phone_number})"

        st.markdown(f"## {title}")
        st.markdown("---")

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
                    st.rerun()  # ⭐ [수정 10] Resume 후 UI 갱신을 위해 rerun 추가
            else:
                if st.button(L["button_hold"], key="hold_call_btn"):
                    st.session_state.is_on_hold = True
                    st.session_state.hold_start_time = datetime.now()
                    st.rerun()  # ⭐ [수정 10] Hold 후 UI 갱신을 위해 rerun 추가

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

                # 3. LLM 번역 실행 (수정된 번역 함수 사용)
                translated_summary = translate_text_with_llm(history_text, target_lang,
                                                             current_lang_at_start)  # Use current_lang_at_start as source

                # 4. 세션 상태 업데이트
                st.session_state.transfer_summary_text = translated_summary
                st.session_state.language_at_transfer = target_lang  # Save destination language
                st.session_state.language_at_transfer_start = current_lang_at_start  # Save source language for retry
                st.session_state.language = target_lang  # Language switch

                # --- 시스템 이관 메시지 추가 ---
                # 전화에서는 별도의 Supervisor 메시지 없이 로그에만 남김
                st.session_state.simulator_messages.append(
                    {"role": "system_transfer",
                     "content": LANG[target_lang]['transfer_system_msg'].format(target_lang=target_lang)})

                st.session_state.is_solution_provided = False
                st.session_state.language_transfer_requested = False

                # 이관 후 상태 전환: 통화 중인 상태는 유지
                st.session_state.call_sim_stage = "IN_CALL"

                # 5. 이력 저장
                customer_type_display = st.session_state.get("customer_type_sim_select", "")
                save_simulation_history_local(
                    st.session_state.call_initial_query,
                    customer_type_display + f" (Transferred from {current_lang_at_start} to {target_lang})",
                    st.session_state.simulator_messages,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                    is_chat_ended=False,
                    is_call=(st.session_state.call_sim_stage == "IN_CALL")  # 전화 이력임을 표시
                )

            # 6. UI 재실행 (언어 변경 적용)
            st.success(f"✅ {LANG[target_lang]['transfer_summary_header']}가 준비되었습니다. 새로운 응대를 시작하세요.")
            st.rerun()


        st.markdown("---")
        st.markdown(f"**{L['transfer_header']}**")
        transfer_cols = st.columns(len(LANG) - 1)

        languages = list(LANG.keys())
        languages.remove(current_lang)

        # transfer_session 함수를 재정의하지 않고, 기존의 transfer_session 함수를 호출합니다.
        for i, target_lang in enumerate(languages):
            button_label_key = f"transfer_to_{target_lang}"
            button_label = L.get(button_label_key, f"Transfer to {target_lang.capitalize()} Team")

            # ⭐ [수정 FIX] 키 중복 오류 해결: 세션 ID와 대상 언어를 조합하여 고유 키 생성
            if transfer_cols[i].button(button_label, key=f"btn_transfer_phone_{target_lang}_{st.session_state.sim_instance_id}"):
                # transfer_session 호출 시, 현재 통화 메시지(simulator_messages)를 넘겨줍니다.
                transfer_session(target_lang, st.session_state.simulator_messages)

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
                st.rerun()

        # 2. 이관 번역 재시도 버튼 (이관 후 번역이 실패했을 경우)
        if st.session_state.language != st.session_state.language_at_transfer_start and not st.session_state.transfer_summary_text:
            with col_trans_btn:
                # ⭐ [수정 FIX] 키 중복 오류 해결: 세션 ID와 언어 코드를 조합하여 고유 키 생성
                retry_key = f"btn_retry_translation_{st.session_state.language_at_transfer_start}_{st.session_state.language}_{st.session_state.sim_instance_id}"
                if st.button(L["button_retry_translation"], key=retry_key):
                    with st.spinner(L["transfer_loading"]):
                        # 이관 번역 로직 재실행 (기존 로직 유지)
                        translated_summary = translate_text_with_llm(
                            get_chat_history_for_prompt(include_attachment=False),
                            st.session_state.language,
                            st.session_state.language_at_transfer_start
                        )
                        st.session_state.transfer_summary_text = translated_summary
                        st.rerun()

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
                    st.rerun()

        # =========================
        # CC 자막 / 음성 입력 및 제어 로직 (기존 로직)
        # =========================================

        # --- 실시간 CC 자막 / 전사 영역 ---
        st.subheader(L["cc_live_transcript"])

        if st.session_state.is_on_hold:
            st.text_area("Customer", value="[고객: 잠시 대기 중입니다...]", height=50, disabled=True, key="customer_live_cc_area")
            st.text_area("Agent", value="[에이전트: Hold 중입니다. 통화 재개 버튼을 눌러주세요.]", height=50, disabled=True,
                         key="agent_live_cc_area")
        else:
            # 고객 CC (LLM 생성 텍스트)
            st.text_area(
                "Customer",
                value=st.session_state.current_customer_audio_text,
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

        # ⭐ 수정: 전사 후 고객 반응 생성 처리 (다음 실행 주기)
        # 전사 결과가 CC에 먼저 표시된 후 고객 반응을 생성하도록 분리
        if st.session_state.get("process_customer_reaction") and st.session_state.get("pending_agent_transcript"):
            pending_transcript = st.session_state.pending_agent_transcript
            # 플래그 초기화
            st.session_state.process_customer_reaction = False
            del st.session_state.pending_agent_transcript

            # 고객 반응 생성
            with st.spinner("고객 반응 생성 중..."):
                customer_reaction = generate_customer_reaction_for_call(
                    st.session_state.language,
                    pending_transcript
                )

                # 고객 반응을 TTS로 재생 및 CC에 반영
                if not customer_reaction.startswith("❌"):
                    audio_bytes, msg = synthesize_tts(customer_reaction, st.session_state.language, role="customer")
                    if audio_bytes:
                        # Streamlit 문서: autoplay는 브라우저 정책상 제한될 수 있음
                        try:
                            st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                            st.success(f"🗣️ 고객이 응답했습니다: {customer_reaction.strip()[:50]}...")
                        except Exception as e:
                            st.warning(f"자동 재생 실패: {e}. 수동으로 재생해주세요.")
                            st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                            st.success(f"🗣️ 고객이 응답했습니다: {customer_reaction.strip()[:50]}...")
                    else:
                        st.error(f"❌ 고객 음성 생성 오류: {msg}")

                    # 고객 반응 텍스트를 CC 영역에 반영
                    st.session_state.current_customer_audio_text = customer_reaction.strip()

                    # 이력 저장
                    log_entry = f"Agent: {st.session_state.current_agent_audio_text} | Customer: {st.session_state.current_customer_audio_text}"
                    st.session_state.simulator_messages.append(
                        {"role": "phone_exchange", "content": log_entry})

                    # 에이전트 입력 영역 초기화 (다음 녹음을 위해)
                    st.session_state.current_agent_audio_text = ""
                    st.session_state.realtime_hint_text = ""

                    # 고객 반응 후 재실행
                    st.rerun()

        # --- 에이전트 음성 입력 / 녹음 ---
        st.subheader(L["mic_input_status"])

        # 음성 입력: 짧은 청크로 끊어서 전사해야 실시간 CC 모방 가능
        if st.session_state.is_on_hold:
            st.info("통화가 Hold 중입니다. 통화 재개 후 녹음이 가능합니다.")
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
            if mic_audio and mic_audio.get("bytes") and "bytes_to_process" not in st.session_state:
                st.session_state.bytes_to_process = mic_audio["bytes"]
                st.session_state.current_agent_audio_text = "🎙️ 녹음 완료. 전사 처리 중..."  # 처리 중 메시지
                # ✅ 재실행하여 다음 실행 주기에서 전사 로직을 처리
                st.rerun()

            # ⭐ 전사 로직: bytes_to_process에 데이터가 있을 때만 실행
            if "bytes_to_process" in st.session_state and st.session_state.bytes_to_process:
                if not st.session_state.openai_client:
                    st.error(L["openai_missing"])
                    st.session_state.bytes_to_process = None
                    # ✅ 재실행
                    # st.rerun()

                if st.session_state.get("bytes_to_process"):
                    # ⭐ 전사 결과를 저장할 변수 초기화
                    agent_response_transcript = None

                    # ⭐ [수정 12]: Whisper 전사 로직에 스피너 추가
                    with st.spinner(L["whisper_processing"]):
                        # 1) Whisper 전사
                        agent_response_transcript = transcribe_bytes_with_whisper(
                            st.session_state.bytes_to_process,
                            "audio/wav",
                            lang_code=st.session_state.language
                        )

                        # 전사 후 바이트 데이터 삭제
                        del st.session_state.bytes_to_process

                        # 2) 전사 실패 처리
                    if agent_response_transcript and agent_response_transcript.startswith("❌"):
                        st.error(agent_response_transcript)
                        st.session_state.current_agent_audio_text = f"[ERROR: {L['error']} Whisper failed]"
                        # 전사 실패 시에도 CC에 반영되도록 재실행
                        st.rerun()
                    elif agent_response_transcript:
                        # 3) CC에 반영 (전사 결과를 먼저 CC 영역에 표시)
                        st.session_state.current_agent_audio_text = agent_response_transcript.strip()

                            # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
                            # 🎯 아바타 표정 업데이트 (최종 정리본)
                        response_text = agent_response_transcript.lower()
                            # ... (아바타 표정 업데이트 로직) ...
                        if "refund" in response_text or "환불" in response_text:
                            st.session_state.customer_avatar["state"] = "HAPPY"
                        elif ("wait" in response_text or "기다려" in response_text or "잠시만" in response_text):
                                st.session_state.customer_avatar["state"] = "ASKING"
                        elif ("no" in response_text or "불가" in response_text or "안 됩니다" in response_text or "cannot" in response_text):
                                st.session_state.customer_avatar["state"] = "ANGRY"
                        else:
                            st.session_state.customer_avatar["state"] = "NEUTRAL"
                            # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

                        # ⭐ 수정: 전사 결과가 CC에 반영되도록 먼저 재실행
                        # 채팅과 동일하게 전사 결과를 먼저 화면에 표시한 후 고객 반응 생성
                        # 다음 실행 주기에서 고객 반응을 생성하도록 플래그 설정
                        st.session_state.process_customer_reaction = True
                        st.session_state.pending_agent_transcript = agent_response_transcript.strip()
                        st.rerun()


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
            with download_col1:
                try:
                    filepath_word = export_history_to_word(current_session_history)
                    with open(filepath_word, "rb") as f:
                        st.download_button(
                            label=L["download_history_word"],
                            data=f.read(),
                            file_name=os.path.basename(filepath_word),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_call_word_file"
                        )
                except Exception as e:
                    st.error(f"Word 다운로드 오류: {e}")
            
            with download_col2:
                try:
                    filepath_pptx = export_history_to_pptx(current_session_history)
                    with open(filepath_pptx, "rb") as f:
                        st.download_button(
                            label=L["download_history_pptx"],
                            data=f.read(),
                            file_name=os.path.basename(filepath_pptx),
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key="download_call_pptx_file"
                        )
                except Exception as e:
                    st.error(f"PPTX 다운로드 오류: {e}")
            
            with download_col3:
                try:
                    filepath_pdf = export_history_to_pdf(current_session_history)
                    with open(filepath_pdf, "rb") as f:
                        st.download_button(
                            label=L["download_history_pdf"],
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
            st.session_state.simulator_messages = []
            st.session_state.call_summary_text = ""
            st.session_state.customer_initial_audio_bytes = None
            st.session_state.customer_history_summary = ""
            st.session_state.sim_audio_bytes = None
            st.rerun() # 새 시뮬레이션 시작 시 rerun


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
                    # st.stop()

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
                quiz_prompt = f"""
                You are an expert quiz generator. Based on the topic '{topic}' and difficulty '{level}', generate 10 multiple-choice questions.
                Your output MUST be a **raw JSON object** containing a single key "quiz_questions" which holds an array of 10 questions.
                Each object in the array must strictly follow the required keys: "question", "options" (array of 4 strings), and "answer" (an integer index starting from 1).
                DO NOT include any explanation, introductory text, or markdown code blocks (e.g., ```json).
                Output ONLY the raw JSON object, starting with '{{' and ending with '}}'.
                """

            generated_json_text = None
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
                            # OpenAI는 JSON 객체를 반환하므로, 펜스 제거 없이 바로 사용 가능해야 함
                            generated_json_text = response.choices[0].message.content.strip()
                            break

                        elif provider == "gemini":
                            # Gemini는 response_format을 지원하지 않으므로, run_llm을 통해 일반 텍스트로 호출
                            generated_json_text = run_llm(quiz_prompt)
                            # Markdown 펜스 제거 시도
                            raw_text = generated_json_text.strip()
                            if raw_text.startswith("```json"):
                                generated_json_text = raw_text.split("```json")[1].split("```")[0].strip()
                            elif raw_text.startswith("```"):
                                generated_json_text = raw_text.split("```")[1].split("```")[0].strip()

                            # Gemini의 응답이 JSON처럼 보이면 시도를 멈춤
                            if generated_json_text.startswith('{'):
                                break

                    except Exception as e:
                        print(f"JSON generation failed with {provider}: {e}")
                        continue

            # --- START: JSON Parsing and Error Handling Logic ---
            if generated_json_text and generated_json_text.startswith('{'):
                try:
                    # JSON 객체 파싱 시도 (최상위는 객체여야 함)
                    parsed_obj = json.loads(generated_json_text)

                    # 'quiz_questions' 키에서 배열 추출
                    quiz_data = parsed_obj.get("quiz_questions")

                    if not isinstance(quiz_data, list) or len(quiz_data) < 1:
                        raise ValueError("Missing 'quiz_questions' key or empty array.")

                    # 3. 파싱 성공 및 데이터 유효성 검사 후 상태 저장
                    st.session_state.quiz_data = quiz_data
                    st.session_state.current_question_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answers = [1] * len(quiz_data)
                    st.session_state.show_explanation = False
                    st.session_state.is_quiz_active = True
                    st.session_state.quiz_type_key = str(uuid.uuid4())

                    st.success(f"**{topic}** - {content_display} 생성 완료")
                    # st.rerun()  # 퀴즈 UI로 전환

                except (json.JSONDecodeError, ValueError) as e:
                    # 4. 파싱 실패 또는 데이터 구조 문제 시 에러 메시지 출력
                    st.error(L["quiz_error_llm"])
                    st.caption(f"Error Details: {type(e).__name__} - {e}")
                    st.subheader(L["quiz_original_response"])
                    st.code(generated_json_text, language="json")
                    # st.stop() 제거: 에러 표시 후 계속 진행
            else:
                st.error(L["quiz_error_llm"])
                if generated_json_text:
                    st.text_area(L["quiz_original_response"], generated_json_text, height=200)
                # st.stop() 제거: 에러 표시 후 계속 진행
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
            st.subheader(f"{L['score']}: {score} / {total_questions} ({(score / total_questions) * 100:.1f}%)")

            if st.button(L["retake_quiz"], key="retake_quiz_btn"):
                # 퀴즈 상태 초기화
                st.session_state.is_quiz_active = False
                st.session_state.quiz_data = None
                st.session_state.current_question_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = []
                st.session_state.show_explanation = False
                # st.rerun()  # 상태 초기화 후 즉시 재실행
            # st.stop() 제거: 퀴즈 완료 후에도 UI는 계속 표시

        # 퀴즈 진행 (현재 문항)
        question_data = quiz_data[idx]
        st.subheader(f"Question {idx + 1}/{len(quiz_data)}")
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
                # st.rerun()

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
                # st.rerun()

        else:
            # 사용자가 이미 정답을 체크했고 (다시 로드된 경우), 다음 버튼을 바로 표시
            if st.session_state.quiz_answers[idx] == 'Correctly Scored' or (
                    isinstance(st.session_state.quiz_answers[idx], int) and st.session_state.quiz_answers[idx] > 0):
                if next_col.button(L["next_question"], key=f"next_question_btn_after_check_{idx}"):
                    st.session_state.current_question_index += 1
                    st.session_state.show_explanation = False
                    # st.rerun()

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
            """


            # --- 더 보기 메뉴 (파일 다운로드/열기 모의) ---

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
                    f"""<script>{js_copy_script}</script>""",
                    height=0,
                )
                st.toast(L["toast_copy"])

            # 5. 더보기 버튼 (기능 활성화 - 파일 옵션 모의)
            with col_more:
                more_clicked = st.button("•••", key=f"content_more_{current_content_id}")

            if more_clicked:
                st.toast(L["toast_more"])

                # 파일 옵션 모의 출력 (버튼 배치)
                st.markdown("**문서 옵션 (모의):**")
                col_doc1, col_doc2, col_doc3 = st.columns(3)

                # 다국어 레이블 적용
                if col_doc1.button(L["mock_pdf_save"], key=f"mock_pdf_save_{current_content_id}"):  # 동적 ID 적용
                    mock_download("PDF", f"{topic}_summary.pdf")
                if col_doc2.button(L["mock_word_open"], key=f"mock_word_open_{current_content_id}"):  # 동적 ID 적용
                    mock_download("Word", f"{topic}_summary.docx")
                if col_doc3.button(L["mock_print"], key=f"mock_print_{current_content_id}"):  # 동적 ID 적용
                    st.toast("🖨 브라우저 인쇄 창이 열립니다.")

            # --- END: 효율성 개선 ---

            # --- END: 아이콘 버튼 추가 ---

# -------------------- LSTM Tab --------------------
elif feature_selection == L["lstm_tab"]:
    # ... (기존 LSTM 탭 로직 유지)
    st.header(L["lstm_header"])
    st.markdown(L["lstm_desc"])

    # ⭐ 최적화: 버튼 자체가 rerun을 유도하므로 명시적 rerun 제거 (버튼 클릭 시 자동 재실행)
    if st.button(L["lstm_rerun_button"]):
        # 버튼 클릭 시 Streamlit이 자동으로 재실행
        pass

    try:
        data = load_or_train_lstm()
        predicted_score = float(np.clip(data[-1] + np.random.uniform(-3, 5), 50, 100))

        st.markdown("---")
        st.subheader(L["lstm_result_header"])

        col_score, col_chart = st.columns([1, 2])

        with col_score:
            suffix = "점" if st.session_state.language == "ko" else ""
            st.metric(L["lstm_score_metric"], f"{predicted_score:.1f}{suffix}")
            st.info(L["lstm_score_info"].format(predicted_score=predicted_score))
        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(data, label="Past Scores", marker="o")
            ax.plot(len(data), predicted_score, marker="*", markersize=10)
            ax.set_title(L["lstm_header"])
            ax.set_xlabel("Time (attempts)")
            ax.set_ylabel("Score (0-100)")
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.info(f"LSTM 기능 에러: {e}")
