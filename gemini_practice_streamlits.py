# ========================================
# streamlit_app_last_correction.py
# 로컬 전용: RAG + 시뮬레이터 + 음성 기록 + LSTM + 콘텐츠
# Firebase/GCS 제거, local_db(JSON/파일)만 사용
# Python 3.9 / langchain>=1.0 / streamlit-mic-recorder 0.0.8 기준
# ========================================

import os
import io
import json
import time
import uuid
import base64
import tempfile
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
import google.generativeai as genai
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from openai import OpenAI

# mic_recorder (0.0.8) - returns dict with key "bytes"
from streamlit_mic_recorder import mic_recorder

# LangChain / RAG 관련
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

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
# 1. 다국어 설정
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
        "incorrect_answer": "오답입니다. 😞",
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
        "history_expander_title": "📝 이전 상담 이력 로드 (최근 10개)",
        "initial_query_sample": "프랑스 파리에 도착했는데, 클룩에서 구매한 eSIM이 활성화가 안 됩니다...",
        "button_mic_input": "🎙 음성 입력",
        "prompt_customer_end": "고객님의 추가 문의 사항이 없어, 이 상담을 종료합니다。",
        "prompt_survey": "지금까지 상담원 000였습니다. 즐거운 하루 되시기 바랍니다. [설문 조사 링크]",
        "customer_closing_confirm": "다른 문의 사항은 없으십니까?",
        "customer_positive_response": "알겠습니다. 감사합니다。",
        "button_end_chat": "응대 종료 (설문 요청)",
        "survey_sent_confirm": "📨 설문조사 링크가 전송되었으며, 이 상담은 종료되었습니다。",
        "new_simulation_ready": "새 시뮬레이션을 시작할 수 있습니다。",
        "agent_response_header": "✍️ 에이전트 응답",
        "agent_response_placeholder": "고객에게 응답하세요...",
        "send_response_button": "응답 전송",
        "request_rebuttal_button": "고객의 다음 반응 요청",
        "new_simulation_button": "새 시뮬레이션 시작",
        "history_selectbox_label": "로드할 이력을 선택하세요:",
        "history_load_button": "선택된 이력 로드",
        "delete_history_button": "❌ 모든 이력 삭제",
        "delete_confirm_message": "정말로 모든 상담 이력을 삭제하시겠습니까?",
        "delete_confirm_yes": "예, 삭제합니다",
        "delete_confirm_no": "아니오, 유지합니다",
        "delete_success": "✅ 삭제 완료!",
        "deleting_history_progress": "이력 삭제 중...",
        "search_history_label": "이력 검색",
        "date_range_label": "날짜 범위 필터",
        "no_history_found": "검색 조건에 맞는 이력이 없습니다。",
        "customer_email_label": "고객 이메일 (선택)",
        "customer_phone_label": "고객 연락처 / 전화번호 (선택)",
        "transfer_header": "언어 이관 요청 (다른 팀)",
        "transfer_to_en": "🇺🇸 영어 팀으로 이관",
        "transfer_to_ja": "🇯🇵 일본어 팀으로 이관",
        "transfer_to_ko": "🇰🇷 한국어 팀으로 이관",
        "transfer_system_msg": "📌 시스템 메시지: 고객 요청에 따라 상담 언어가 {target_lang} 팀으로 이관되었습니다. 새로운 상담원(AI)이 응대합니다。",
        "transfer_loading": "이관 처리 중: 이전 대화 이력 번역 및 검토 (고객님께 3~10분 양해 요청)",
        "transfer_summary_header": "🔍 이관된 상담원을 위한 요약 (번역됨)",
        "transfer_summary_intro": "고객님과의 이전 대화 이력입니다. 이 내용을 바탕으로 응대를 이어나가세요。",
        "llm_translation_error": "❌ 번역 실패: LLM 응답 오류",
        "timer_metric": "상담 경과 시간",
        "timer_info_ok": "AHT (15분 기준)",
        "timer_info_warn": "AHT (10분 초과)",
        "timer_info_risk": "🚨 15분 초과: 높은 리스크",
        "solution_check_label": "✅ 이 응답에 솔루션/해결책이 포함되어 있습니다.",

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
        "whisper_processing": "음성 전사 처리 중",
        "empty_response_warning": "응답을 입력하세요。",
        "customer_no_more_inquiries": "없습니다. 감사합니다。",
        "customer_has_additional_inquiries": "추가 문의 사항도 있습니다。",
        "sim_end_chat_button": "설문 조사 링크 전송 및 채팅 종료",
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
        "prompt_customer_end": "No further inquiries. Ending chat.",
        "prompt_survey": "This was Agent 000. Have a nice day. [Survey Link]",
        "customer_closing_confirm": "Is there anything else we can assist you with?",
        "customer_positive_response": "Noted with thanks.",
        "button_end_chat": "End Chat (Survey Request)",
        "survey_sent_confirm": "📨 The survey link has been sent. This chat session is now closed.",
        "new_simulation_ready": "You can now start a new simulation.",
        "agent_response_header": "✍️ Agent Response",
        "agent_response_placeholder": "Write a response...",
        "send_response_button": "Send Response",
        "request_rebuttal_button": "Request Customer Reaction",
        "new_simulation_button": "Start New Simulation",
        "history_selectbox_label": "Choose a record to load:",
        "history_load_button": "Load Selected Record",
        "delete_history_button": "❌ Delete All History",
        "delete_confirm_message": "Are you sure you want to delete all records?",
        "delete_confirm_yes": "Yes, Delete",
        "delete_confirm_no": "Cancel",
        "delete_success": "Deleted successfully!",
        "deleting_history_progress": "Deleting history...",
        "search_history_label": "Search History",
        "date_range_label": "Date Filter",
        "no_history_found": "No matching history found.",
        "customer_email_label": "Customer Email (optional)",
        "customer_phone_label": "Customer Phone / WhatsApp (optional)",
        "transfer_header": "Language Transfer Request (To Other Teams)",
        "transfer_to_en": "🇰🇷 Korean Team Transfer",
        "transfer_to_ja": "🇯🇵 Japanese Team Transfer",
        "transfer_to_ko": "🇺🇸 English Team Transfer",
        "transfer_system_msg": "📌 System Message: The session language has been transferred to the {target_lang} team per customer request. A new agent (AI) will now respond.",
        "transfer_loading": "Transferring: Translating and reviewing chat history (3-10 minute wait requested from customer)",
        "transfer_summary_header": "🔍 Summary for Transferred Agent (Translated)",
        "transfer_summary_intro": "This is the previous chat history. Please continue the support based on this summary.",
        "llm_translation_error": "❌ Translation failed: LLM response error",
        "timer_metric": "Elapsed Time",
        "timer_info_ok": "AHT (15 min standard)",
        "timer_info_warn": "AHT (Over 10 min)",
        "timer_info_risk": "🚨 Over 15 min: High Risk",
        "solution_check_label": "✅ This response includes a solution/fix.",

        # Voice
        "voice_rec_header": "Voice Record & Management",
        "record_help": "Record using the microphone or upload a file.",
        "uploaded_file": "Upload Audio File",
        "rec_list_title": "Saved Voice Records",
        "transcribe_btn": "Transcribe (Whisper)",
        "save_btn": "Save Record",
        "transcribing": "Transcribing...",
        "transcript_result": "Transcription:",
        "transcript_text": "Transcribed Text",
        "openai_missing": "Missing OPENAI_API_KEY",
        "whisper_client_error": "Whisper client initialization failed.",
        "whisper_auth_error": "Whisper authentication failed.",
        "whisper_format_error": "Unsupported audio format.",
        "whisper_success": "Transcription complete!",
        "playback": "Play Recording",
        "retranscribe": "Re-transcribe",
        "delete": "Delete",
        "transcribe_btn": "Transcribe (Whisper)",
        "save_btn": "Save Voice Record",
        "transcribing": "Transcribing voice...",
        "transcript_result": "Transcription Result:",
        "transcript_text": "Transcribed Text",
        "whisper_processing": "Processing voice transcription...",
        "whisper_success": "✅ Transcription complete! Please check the text below.",
        "openai_missing": "OpenAI API Key is missing. Please set OPENAI_API_KEY.",
        "whisper_client_error": "❌ Error: Whisper API client not initialized.",
        "whisper_auth_error": "❌ Whisper API authentication failed. Check your API Key.",
        "whisper_format_error": "❌ Error: Unsupported audio format.",
        "playback": "Playback Recording",
        "retranscribe": "Re-transcribe",
        "delete": "Delete",
        "no_records": "No saved voice records.",
        "saved_success": "Saved successfully!",
        "delete_confirm_rec": "Are you sure you want to delete this voice record?",
        "gcs_not_conf": "GCS not configured or no audio available",
        "gcs_playback_fail": "Failed to play audio",
        "gcs_no_audio": "No audio file found",
        "error": "Error:",
        "firestore_no_db_connect": "DB connection failed",
        "save_history_success": "Saved successfully.",
        "save_history_fail": "Save failed.",
        "delete_fail": "Delete failed",
        "rec_header": "Voice Input & Transcription",
        "whisper_processing": "Processing...",
        "empty_response_warning": "Please enter a response.",
        "customer_no_more_inquiries": "No, that will be all, thank you.",
        "customer_has_additional_inquiries": "Yes, I have an additional question.",
        "sim_end_chat_button": "Send Survey Link and End Chat",
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
        "quiz_error_llm": "クイズ生成失敗：JSON形式が正しくありません。",
        "quiz_original_response": "LLM 原本回答",
        "firestore_loading": "RAGインデックス読み込み中...",
        "firestore_no_index": "保存されたRAGインデックスが見つかりません。",
        "db_save_complete": "(DB保存完了)",
        "data_analysis_progress": "資料分析中...",
        "response_generating": "応答生成中...",
        "lstm_result_header": "達成度予測結果",
        "lstm_score_metric": "予測達成度",
        "lstm_score_info": "次のスコア予測: **{predicted_score:.1f}点**",
        "lstm_rerun_button": "新しいデータで再予測",

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
        "prompt_customer_end": "追加の質問がないためチャットを終了します。",
        "prompt_survey": "担当エージェント000でした。良い一日をお過ごしください。 [アンケートリンク]",
        "customer_closing_confirm": "他のお問合せはございませんでしょうか。",
        "customer_positive_response": "はい、承知いたしました。ありがとうございます。",
        "button_end_chat": "チャット終了（アンケート）",
        "new_simulation_ready": "新しいシミュレーションを開始できます。",
        "survey_sent_confirm": "📨 アンケートリンクを送信しました。このチャットは終了しました。",
        "agent_response_header": "✍️ エージェント応答",
        "agent_response_placeholder": "顧客へ返信内容を入力…",
        "send_response_button": "返信送信",
        "request_rebuttal_button": "顧客の反応を生成",
        "new_simulation_button": "新規シミュレーション",
        "history_selectbox_label": "履歴を選択:",
        "history_load_button": "履歴を読み込む",
        "delete_history_button": "❌ 全履歴削除",
        "delete_confirm_message": "すべての履歴を削除しますか？",
        "delete_confirm_yes": "はい、削除します。",
        "delete_confirm_no": "いいえ、維持します。",
        "delete_success": "削除完了！",
        "deleting_history_progress": "削除中...",
        "search_history_label": "履歴検索",
        "date_range_label": "日付フィルター",
        "no_history_found": "該当する履歴はありません。",
        "customer_email_label": "顧客メールアドレス（任意）",
        "customer_phone_label": "顧客連絡先 / 電話番号（任意）",
        "transfer_header": "言語切り替え要請（他チームへ）",
        "transfer_to_en": "🇺🇸 英語チームへ転送",
        "transfer_to_ko": "🇰🇷 韓国語チームへ転送",
        "transfer_system_msg": "📌 システムメッセージ: 顧客の要請により、対応言語が {target_lang} チームへ切り替えられました。新しい担当者(AI)が対応します。",
        "transfer_loading": "転送中: 過去のチャット履歴を翻訳およびレビューしています (お客様には3〜10分のお時間をいただいています)",
        "transfer_summary_header": "🔍 転送された担当者向けの要約 (翻訳済み)",
        "transfer_summary_intro": "これが顧客との過去のチャット履歴です。この要約に基づいてサポートを続けてください。",
        "llm_translation_error": "❌ 翻訳失敗: LLM応答エラー",
        "timer_metric": "経過時間",
        "timer_info_ok": "AHT (15분 기준)",
        "timer_info_warn": "AHT (10분 초과)",
        "timer_info_risk": "🚨 15분 초과: 高いリスク",
        "solution_check_label": "✅ この応答に解決策/対応策が含まれています。",

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
        "whisper_processing": "音声転写を処理中...",
        "whisper_success": "✅ 転写が完了しました！ 以下のテキストをご確認ください。",
        "openai_missing": "OpenAI APIキーがありません。OPENAI_API_KEYを設定してください。",
        "whisper_client_error": "❌ エラー: Whisper APIクライアントが初期化されていません。",
        "whisper_auth_error": "❌ Whisper API認証に失敗しました。APIキーをご確認ください。",
        "whisper_format_error": "❌ エラー: この音声形式はサポートされていません。",
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
        "sim_end_chat_button": "アンケートリンクを送信してチャット終了",
    }
}

# ========================================
# 1-1. Session State 초기화 (누락된 AHT/솔루션/이관 상태 추가)
# ========================================

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
if "start_time" not in st.session_state:  # AHT 타이머 시작 시간
    st.session_state.start_time = None
if "is_solution_provided" not in st.session_state:  # 솔루션 제공 여부 플래그
    st.session_state.is_solution_provided = False
if "transfer_summary_text" not in st.session_state:  # 이관 시 번역된 요약
    st.session_state.transfer_summary_text = ""
if "language_transfer_requested" not in st.session_state:  # 고객의 언어 이관 요청 여부
    st.session_state.language_transfer_requested = False

L = LANG[st.session_state.language]


# ========================================
# 2. LLM 클라이언트 라우팅 & 실행
# ... (생략)
# ========================================

# ... (Helper Functions - TTS, Whisper, RAG, etc. - are maintained)

# ========================================
# 8. LLM (ChatOpenAI) for Simulator / Content
# ... (get_chat_history_for_prompt, generate_customer_reaction, generate_customer_closing_response are maintained)
# ========================================

# ----------------------------------------
# LLM 번역 함수 (Gemini 클라이언트 의존성 제거 및 강화)
# ----------------------------------------
def translate_text_with_llm(text_content: str, target_lang_code: str, source_lang_code: str) -> str:
    """
    주어진 텍스트를 LLM(Gemini)을 사용하여 대상 언어로 번역합니다.
    """

    target_lang = {"ko": "Korean", "en": "English", "ja": "Japanese"}[st.session_state.language]

    prompt = f"""
    You are an AI educational content generator and helpful AI tutor.
    Generate the following content in {target_lang} ONLY.

    Topic: {topic}
    Difficulty: {level}
    Content Type: {content_type}
    """
    response = run_llm(prompt)

    # 1. Gemini API 키 확인 및 설정
    gemini_key = get_api_key("gemini")
    target_lang = LANG.get(target_lang_code, {})

    if not gemini_key:
        return f"❌ {target_lang.get('simulation_no_key_warning', 'API Key missing').replace('GEMINI_API_KEY', 'Translation API Key')}"

    try:
        # 2. Gemini 클라이언트 설정 (on-the-fly)
        client = genai
        client.configure(api_key=gemini_key)

        target_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(target_lang_code, "English")
        source_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(source_lang_code, "English")

        system_prompt = (
            f"You are a professional translation AI. Translate the following customer support chat history "
            f"from '{source_lang_name}' to '{target_lang_name}'. Preserve the original format, marking "
            f"each speaker (e.g., 'Customer:', 'Agent:'). Do not add any introductory or concluding remarks. "
            f"Translate the content accurately and neutrally."
        )

        prompt = f"Original Chat History:\n\n{text_content}"

        # 3. 번역 실행
        gen_model = client.GenerativeModel('gemini-2.5-flash')
        response = gen_model.generate_content(
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2
            ),
        )
        return response.text.strip()
    except Exception as e:
        # LLM 응답 오류 또는 기타 예외 처리
        st.error(f"{target_lang.get('llm_translation_error', 'Translation failed')}: {e}")
        return "❌ LLM_TRANSLATION_ERROR"


# ... (generate_customer_reaction, generate_customer_closing_response 그대로 유지)

# ========================================
# 9. 사이드바
# ... (생략)
# ========================================

# 메인 타이틀
st.title(L["title"])

# ========================================
# 10. 기능별 페이지
# ... (RAG, Content, LSTM, Voice Tabs are maintained)
# -------------------- Simulator Tab --------------------
elif feature_selection == L["simulator_tab"]:
st.header(L["simulator_header"])
st.markdown(L["simulator_desc"])

current_lang = st.session_state.language
L = LANG[current_lang]  # 다시 L 업데이트

# =========================
# 0. 전체 이력 삭제
# ... (생략)

# =========================
# 1. 이전 이력 로드 (기존 로직 유지)
# ... (생략)

# =========================
# AHT 타이머 (화면 최상단)
# ** Fix 2: 타이머 위젯을 고정된 컬럼에 배치하여 미표시 오류 해결 **
# =========================
if st.session_state.sim_stage not in ["WAIT_FIRST_QUERY", "CLOSING", "idle"]:
    col_timer, _ = st.columns([1, 4])

    # start_time이 있을 때만 계산 및 표시
    if st.session_state.start_time is not None:
        # 현재 시간 계산
        elapsed_time = datetime.now() - st.session_state.start_time
        total_seconds = elapsed_time.total_seconds()

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
            st.metric(
                L["timer_metric"],
                time_str,
                delta=delta_str,
                delta_color=delta_color
            )

    st.markdown("---")

# =========================
# 2. LLM 준비 체크 & 채팅 종료 상태
# ... (생략)

# =========================
# 3. 초기 문의 입력 (WAIT_FIRST_QUERY)
# ... (생략)

# =========================
# 4. 대화 로그 표시 (공통)
# ... (생략)

# 이관 요약 표시 (이관 후에만)
if st.session_state.transfer_summary_text:
    st.markdown("---")
    st.markdown(f"**{L['transfer_summary_header']}**")
    st.info(L["transfer_summary_intro"])
    st.markdown(st.session_state.transfer_summary_text)
    st.markdown("---")

# =========================
# 5. 에이전트 입력 단계 (AGENT_TURN)
# =========================
if st.session_state.sim_stage == "AGENT_TURN":
    st.markdown(f"### {L['agent_response_header']}")

    # --- 언어 이관 요청 강조 표시 ---
    if st.session_state.language_transfer_requested:
        st.error("🚨 고객이 언어 전환(이관)을 요청했습니다. 즉시 응대하거나 이관을 진행하세요.")

    col_mic, col_text = st.columns([1, 2])


    # ... (마이크 녹음 및 전사 로직)

    # --- 텍스트 입력 + 전송 버튼 ---

    # 1. 텍스트 입력 필드
    def update_agent_response():
        st.session_state.agent_response_area_text = st.session_state.agent_response_input_box_widget


    col_text, col_button = st.columns([4, 1])

    with col_text:
        st.text_area(
            L["agent_response_placeholder"],
            value=st.session_state.agent_response_area_text,
            key="agent_response_input_box_widget",
            on_change=update_agent_response,
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
        agent_response = st.session_state.agent_response_input_box_widget.strip()

        if not agent_response:
            st.warning(L["empty_response_warning"])
            st.stop()

        st.session_state.agent_response_area_text = agent_response  # 최종값 반영

        # 로그 업데이트 (솔루션 제공 여부는 이미 체크박스에서 상태 업데이트됨)
        st.session_state.simulator_messages.append(
            {"role": "agent_response", "content": agent_response}
        )

        # 입력창/오디오 초기화
        st.session_state.agent_response_area_text = ""
        st.session_state.sim_audio_bytes = None
        st.session_state.language_transfer_requested = False  # 처리되었으므로 플래그 리셋

        # 다음 단계: 고객 반응 생성 요청
        st.session_state.sim_stage = "CUSTOMER_TURN"
        # st.rerun()  # 주석 처리: 과도한 rerun 방지

    # --- 언어 이관 버튼 ---
    st.markdown("---")
    st.markdown(f"**{L['transfer_header']}**")
    transfer_cols = st.columns(len(LANG) - 1)

    languages = list(LANG.keys())
    languages.remove(current_lang)


    def transfer_session(target_lang: str, current_messages: List[Dict[str, str]]):
        """언어 이관 시스템 메시지를 추가하고 세션 언어를 변경합니다."""

        if not get_api_key("gemini"):
            st.error(LANG[current_lang]["simulation_no_key_warning"].replace('API Key', 'Gemini API Key'))
            st.stop()
            return

        # AHT 타이머 중지
        st.session_state.start_time = None

        # 1. 로딩 시작 (시간 양해 메시지 시뮬레이션)
        with st.spinner(L["transfer_loading"]):
            # 실제 대기 시간 5~10초 (3~10분 시뮬레이션)
            time.sleep(np.random.uniform(5, 10))

            # 2. 대화 기록을 번역할 텍스트로 가공
            history_text = ""
            for msg in current_messages:
                role = "Customer" if msg["role"].startswith("customer") or msg["role"] == "initial_query" else "Agent"
                if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response", "customer_closing_response"]:
                    history_text += f"{role}: {msg['content']}\n"

            # 3. LLM 번역 실행 (수정된 번역 함수 사용)
            translated_summary = translate_text_with_llm(history_text, target_lang, st.session_state.language)

            if translated_summary.startswith("❌"):
                st.session_state.transfer_summary_text = translated_summary
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
                return

            # 4. 세션 상태 업데이트
            st.session_state.transfer_summary_text = translated_summary

            # 시스템 메시지 추가 (이관 알림)
            target_lang_name = {"ko": "한국어", "en": "English", "ja": "日本語"}.get(target_lang, target_lang.capitalize())
            system_msg = L["transfer_system_msg"].format(target_lang=target_lang_name)
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": system_msg}
            )

            st.session_state.language = target_lang  # 언어 변경
            st.session_state.is_solution_provided = False  # 새로운 응대를 위해 플래그 리셋
            st.session_state.language_transfer_requested = False  # 플래그 리셋
            st.session_state.sim_stage = "AGENT_TURN"

            # 5. 이력 저장
            customer_type_display = st.session_state.get("customer_type_sim_select", "")
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display + f" (Transferred from {st.session_state.language} to {target_lang})",
                st.session_state.simulator_messages,
                is_chat_ended=False,
            )

        # 6. UI 재실행 (언어 변경 적용)
        st.success(f"✅ {LANG[target_lang]['transfer_summary_header']}가 준비되었습니다. 새로운 응대를 시작하세요.")
        # st.rerun()  # 주석 처리: 과도한 rerun 방지


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
if st.session_state.sim_stage == "CUSTOMER_TURN":
    st.info("에이전트 응답 전송 완료. 고객 반응 생성이 필요합니다.")

    if st.button(L["customer_generate_response_button"], key="sim_next_rebuttal_btn"):
        if not st.session_state.is_llm_ready:
            st.warning(L["simulation_no_key_warning"])
            st.stop()

        with st.spinner(L["response_generating"]):  # 로딩 표시
            reaction = generate_customer_reaction(st.session_state.language)

        if reaction.startswith("❌"):
            st.error(reaction)
            st.stop()

        st.session_state.simulator_messages.append(
            {"role": "customer_rebuttal", "content": reaction}
        )

        # --- AHT 타이머 시작 (고객 반응 생성 후, 즉 에이전트 응대가 시작되는 순간) ---
        if st.session_state.start_time is None:
            st.session_state.start_time = datetime.now()

        # 언어 이관 요청 키워드 확인 (요청 3 반영)
        lang_request_keywords = ["english", "japanese", "한국어", "英語", "日本語", "korean"]
        if any(k in reaction.lower() for k in lang_request_keywords):
            st.session_state.language_transfer_requested = True

        # 종료 의사 판별 (요청 7 반영: 감사 인사를 했는지)
        reaction_lower = reaction.lower()
        appreciation_signals = ["감사", "thank", "ありがとう", "noted"]
        has_appreciation = any(k in reaction_lower for k in appreciation_signals)

        is_additional_inquiry_signal = L['customer_has_additional_inquiries'] in reaction

        customer_type_display = st.session_state.get("customer_type_sim_select", "")

        # --- 핵심 로직 수정 (요청 1, 2 반영) ---
        # 1. 솔루션 제공 O, 고객 감사 O, 추가 문의 X -> 종료 확인 단계로
        if st.session_state.is_solution_provided and has_appreciation and not is_additional_inquiry_signal:
            st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            st.session_state.is_solution_provided = False  # 종료 단계 진입 후 플래그 리셋
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=False,
            )
        # 2. 솔루션 제공 X, 고객 반응 O/추가 문의 O -> 무조건 에이전트 턴 유지
        else:
            st.session_state.sim_stage = "AGENT_TURN"
            # 솔루션 제공 X 였더라도, AGENT_TURN으로 돌아가면 다음 응답 시 체크박스 상태를 유지해야 함.
            # 단, is_solution_provided 플래그는 이전 턴의 체크박스 상태를 반영하므로, 여기서 명시적으로 변경할 필요는 없음.
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=False,
            )

        # st.rerun()  # 주석 처리: 과도한 rerun 방지

# =========================
# 7. 종료 확인 메시지 대기 (WAIT_CLOSING_CONFIRMATION_FROM_AGENT)
# ** Fix 1: 이 상태에서는 버튼만 표시하고 입력 필드는 숨김 **
# =========================
if st.session_state.sim_stage == "WAIT_CLOSING_CONFIRMATION_FROM_AGENT":
    st.success("고객이 솔루션에 긍정적으로 반응했습니다. 추가 문의 여부를 확인해 주세요.")

    # 에이전트가 "추가 문의 여부 확인 메시지"를 보내는 버튼 (요청 1 반영)
    if st.button(L["send_closing_confirm_button"], key="btn_send_closing_confirm"):
        closing_msg = L["customer_closing_confirm"]

        # 에이전트 응답으로 로그 기록
        st.session_state.simulator_messages.append(
            {"role": "agent_response", "content": closing_msg}
        )

        # 다음 단계: 고객의 최종 답변 대기
        st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"

        customer_type_display = st.session_state.get("customer_type_sim_select", "")
        save_simulation_history_local(
            st.session_state.customer_query_text_area, customer_type_display,
            st.session_state.simulator_messages, is_chat_ended=False,
        )
        # st.rerun()  # 주석 처리: 과도한 rerun 방지

# =========================
# 8. 고객 최종 응답 생성 및 처리 (WAIT_CUSTOMER_CLOSING_RESPONSE)
# ... (생략)

# =========================
# 9. 최종 종료 행동 (FINAL_CLOSING_ACTION)
# ... (생략)

# -------------------- RAG Tab --------------------
elif feature_selection == L["rag_tab"]:
    st.header(L["rag_header"])
    st.markdown(L["rag_desc"])

    if not st.session_state.is_rag_ready or st.session_state.rag_vectorstore is None:
        if st.session_state.is_llm_ready:
            with st.spinner(L["firestore_loading"]):
                vs = load_rag_index()
                if vs is not None:
                    st.session_state.rag_vectorstore = vs
                    st.session_state.is_rag_ready = True
                else:
                    st.info(L["firestore_no_index"])
        else:
            st.warning(L["warning_rag_not_ready"])

    if st.session_state.is_rag_ready and st.session_state.rag_vectorstore is not None:
        for m in st.session_state.rag_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input(L["rag_input_placeholder"])
        if user_q:
            st.session_state.rag_messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner(L["response_generating"]):
                    try:
                        ans = rag_answer(user_q, st.session_state.rag_vectorstore, st.session_state.language)
                        st.markdown(ans)
                        st.session_state.rag_messages.append({"role": "assistant", "content": ans})
                    except Exception as e:
                        st.error(f"챗봇 오류: {e}")
                        msg = "오류 발생" if st.session_state.language == "ko" else "An error occurred"
                        st.session_state.rag_messages.append({"role": "assistant", "content": msg})
    else:
        st.warning(L["warning_rag_not_ready"])


# -------------------- LSTM Tab --------------------
elif feature_selection == L["lstm_tab"]:
    st.header(L["lstm_header"])
    st.markdown(L["lstm_desc"])

    if st.button(L["lstm_rerun_button"]):
        # st.rerun()  # 주석 처리: 과도한 rerun 방지
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

# -------------------- Voice Record Tab --------------------
elif feature_selection == L["voice_rec_header"]:
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
        if audio_bytes:
            st.audio(audio_bytes, format=audio_mime)

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
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
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
                            st.audio(b, format=mime)
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
                                    # st.rerun()  # 주석 처리: 과도한 rerun 방지
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
                            # st.rerun()  # 주석 처리: 과도한 rerun 방지
                        else:
                            st.session_state[f"confirm_del_{rec_id}"] = True
                            st.warning(L["delete_confirm_rec"])

    # =========================
    # 7. 종료 확인 메시지 대기 (WAIT_CLOSING_CONFIRMATION_FROM_AGENT)
    # ** Fix 1 & 2: 채팅/이메일 종료 분리 및 버튼 분리 **
    # =========================
    if st.session_state.sim_stage == "WAIT_CLOSING_CONFIRMATION_FROM_AGENT":
        st.success("고객이 솔루션에 긍정적으로 반응했습니다. 추가 문의 여부를 확인해 주세요.")

        col_chat_end, col_email_end = st.columns(2) # 버튼을 나란히 배치

        # [1] 채팅 - 추가 문의 확인 메시지 보내기 버튼 (기존 로직)
        with col_chat_end:
            if st.button(L["send_closing_confirm_button"], key="btn_send_closing_confirm"):
                # ... (기존 채팅 종료 확인 로직 유지)
                # st.rerun()  # 주석 처리: 과도한 rerun 방지

        # [2] 이메일 - 상담 종료 버튼 (요청 2 반영: 즉시 종료)
        with col_email_end:
            if st.button(L["button_email_end_chat"], key="btn_email_end_chat"):
                # 이메일은 끝인사에 문의 확인이 포함되므로, 바로 최종 종료 단계로 이동
                st.session_state.sim_stage = "FINAL_CLOSING_ACTION"
                st.session_state.simulator_messages.append(
                    {"role": "system_end", "content": "(시스템: 이메일 특성상, 즉시 최종 종료 단계로 진입합니다.)"}
                )
                # st.rerun()  # 주석 처리: 과도한 rerun 방지

                current_hold_duration = (now - st.session_state.hold_start_time) if st.session_state.is_on_hold and st.session_state.hold_start_time else timedelta(0)
    # =========================
    # 2. LLM 준비 체크 & 채팅 종료 상태
    # =========================

    # 기존 1차 가드 → 유지
    if not st.session_state.is_llm_ready:
        st.warning(L["simulation_no_key_warning"])
        return

    # 2차 실제 호출 기반 가드 → 추가
    resp = run_llm("ping")

    if resp is None or len(resp.strip()) == 0 or "❌" in resp:
        st.session_state.is_llm_ready = False
        st.warning(L["simulation_no_key_warning"])
        return


elif feature_selection == L["rag_tab"]:
    # ... (기존 RAG 탭 로직 유지)
    st.header(L["rag_header"])
    st.markdown(L["rag_desc"])

    # 학습 자료 업로드 (메인 컴포넌트로 이동)
    st.markdown("---")
    st.subheader("📚 학습 자료 업로드")
    uploaded_files_widget = st.file_uploader(
        L["file_uploader"], type=["pdf", "txt", "html"], accept_multiple_files=True,
        key="rag_file_uploader"
    )
    if uploaded_files_widget:
        st.session_state.uploaded_files_state = uploaded_files_widget

    files_to_process = st.session_state.uploaded_files_state or []

    # RAG 인덱싱 버튼
    if files_to_process and st.session_state.is_llm_ready:
        if st.button(L["button_start_analysis"], key="rag_start_analysis_btn"):
            with st.spinner(L["data_analysis_progress"]):
                vs, count = build_rag_index(files_to_process)
                if vs is not None:
                    st.session_state.rag_vectorstore = vs
                    st.session_state.is_rag_ready = True
                    st.success(L["embed_success"].format(count=count))
                    # ⭐ 재실행
                    # st.rerun()  # 주석 처리: 과도한 rerun 방지
                else:
                    st.session_state.is_rag_ready = False
    elif not files_to_process:
        st.info(L["warning_no_files"])

    st.markdown("---")

    if not st.session_state.is_rag_ready or st.session_state.rag_vectorstore is None:
        if st.session_state.is_llm_ready:
            with st.spinner(L["firestore_loading"]):
                # RAG 인덱스 로드 시에도 임베딩 함수를 사용하므로, 키 유효성 체크 필요
                vs = load_rag_index()
                if vs is not None:
                    st.session_state.rag_vectorstore = vs
                    st.session_state.is_rag_ready = True
                else:
                    st.info(L["firestore_no_index"])
        else:
            st.warning(L["warning_rag_not_ready"])

    if st.session_state.is_rag_ready and st.session_state.rag_vectorstore is not None:
        # 기존 대화 로그 표시
        for m in st.session_state.rag_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input(L["rag_input_placeholder"])
        if user_q:
            st.session_state.rag_messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner(L["response_generating"]):
                    try:
                        ans = rag_answer(user_q, st.session_state.rag_vectorstore, st.session_state.language)
                        st.markdown(ans)
                        st.session_state.rag_messages.append({"role": "assistant", "content": ans})
                    except Exception as e:
                        st.error(f"챗봇 오류: {e}")
                        msg = "오류 발생" if st.session_state.language == "ko" else "An error occurred"
                        st.session_state.rag_messages.append({"role": "assistant", "content": msg})
    else:
        st.warning(L["warning_rag_not_ready"])

# -------------------- Content Tab --------------------
elif feature_selection == L["content_tab"]:
    # ... (기존 콘텐츠 탭 로직 유지)
    st.header(L["content_header"])
    st.markdown(L["content_desc"])

    if not st.session_state.is_llm_ready:
        st.error(L["llm_error_init"])
    else:
        topic = st.text_input(L["topic_label"])
        level_display = st.selectbox(L["level_label"], L["level_options"])
        content_display = st.selectbox(L["content_type_label"], L["content_options"])

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

        level = level_map.get(level_display, "Beginner")
        content_type = content_map.get(content_display, "summary")

        if st.button(L["button_generate"]):
            if not topic.strip():
                st.warning(L["warning_topic"])
            else:
                target_lang = {"ko": "Korean", "en": "English", "ja": "Japanese"}[st.session_state.language]

                if content_type == "quiz":
                    system_prompt = (
                        "You are an expert quiz creator.\n"
                        "Generate EXACTLY 10 multiple-choice questions.\n"
                        "Return ONLY valid JSON wrapped inside ```json ... ```.\n"
                        "JSON structure:\n"
                        "{\n"
                        '  \"quiz_questions\": [\n'
                        "    {\n"
                        '      \"question\": \"...\",\n'
                        '      \"options\": [\"A\", \"B\", \"C\", \"D\"],\n'
                        '      \"correct_index\": 0,\n'
                        '      \"explanation\": \"...\"\n'
                        "    }\n"
                        "  ]\n"
                        "}\n"
                        f"The language of questions MUST be {target_lang}.\n"
                    )
                    user_msg = f"Topic: {topic} (level: {level})"
                    with st.spinner("퀴즈 생성 중..."):
                        try:
                            resp = run_llm(system_prompt + "\n\n" + user_msg)
                            # 단순 출력
                            st.success(f"**{topic}** - {content_display}")
                            st.code(resp, language="json")
                        except Exception as e:
                            st.error(f"Content Generation Error: {e}")
                else:
                    content_prompt = (
                        f"You are a professional AI coach at the {level} level.\n"
                        f"Generate clear and educational content in {target_lang}.\n"
                        f"Content type: {content_type}.\n"
                        f"Topic: {topic}\n"
                    )
                    with st.spinner("콘텐츠 생성 중..."):
                        try:
                            resp = run_llm(content_prompt)
                            st.success(f"**{topic}** - {content_display}")
                            st.markdown(resp)
                        except Exception as e:
                            st.error(f"Content Generation Error: {e}")

                            current_answer = st.session_state.quiz_answers[idx]

                            if current_answer is None or not isinstance(current_answer, int) or current_answer <= 0:
                                radio_index = -1
                            else:
                                radio_index = min(current_answer - 1, len(options) - 1)

                            selected_option = st.radio(
                                L["select_answer"],
                                options,
                                index=radio_index,
                                key=f"quiz_radio_{st.session_state.quiz_type_key}_{idx}"
                            )

# -------------------- LSTM Tab --------------------
elif feature_selection == L["lstm_tab"]:
    # ... (기존 LSTM 탭 로직 유지)
    st.header(L["lstm_header"])
    st.markdown(L["lstm_desc"])

    # ⭐ 최적화: 버튼 자체가 rerun을 유도하므로 명시적 rerun 제거 (버튼 클릭 시 자동 재실행)
    # if st.button(L["lstm_rerun_button"]):
    #     st.rerun()

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


elif st.session_state.sim_stage == "WAIT_CLOSING_CONFIRMATION_FROM_AGENT":
    st.success("고객이 솔루션에 긍정적으로 반응했습니다. 추가 문의 여부를 확인해 주세요.")

    col_chat_end, col_email_end = st.columns(2)  # 버튼을 나란히 배치

    # [1] 채팅 - 추가 문의 확인 메시지 보내기 버튼
    with col_chat_end:
        # 상태 전환 명확화: 이 버튼 클릭 시 다음 단계인 WAIT_CUSTOMER_CLOSING_RESPONSE로 반드시 넘어감
        if st.button(L["send_closing_confirm_button"],
                     key=f"btn_send_closing_confirm_{st.session_state.sim_instance_id}"):
            closing_msg = L["customer_closing_confirm"]

            # 에이전트 응답으로 로그 기록
            st.session_state.simulator_messages.append(
                {"role": "agent_response", "content": closing_msg}
            )

            # 다음 단계: 고객의 최종 답변 대기
            st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"

            # 이력 저장
            customer_type_display = st.session_state.get("customer_type_sim_select", "")
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )
            # ⭐ 재실행
            # st.rerun()  # 주석 처리: 과도한 rerun 방지

    # [2] 이메일 - 상담 종료 버튼 (즉시 종료)
    with col_email_end:
        if st.button(L["button_email_end_chat"], key=f"btn_email_end_chat_{st.session_state.sim_instance_id}"):
            # 이메일은 끝인사에 문의 확인이 포함되므로, 바로 최종 종료 단계로 이동

            # AHT 타이머 정지
            st.session_state.start_time = None

            # 최종 종료 메시지 (설문 조사 포함)
            end_msg = L["prompt_survey"]
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": "(시스템: 이메일 상담 종료) " + end_msg}
            )
            st.session_state.is_chat_ended = True
            st.session_state.sim_stage = "CLOSING"  # 바로 CLOSING으로 전환

            # 이력 저장
            customer_type_display = st.session_state.get("customer_type_sim_select", "")
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )
            # ⭐ 재실행
            # st.rerun()  # 주석 처리: 과도한 rerun 방지

    # =========================
    # 8. 고객 최종 응답 생성 및 처리 (WAIT_CUSTOMER_CLOSING_RESPONSE)
    # =========================
elif st.session_state.sim_stage == "WAIT_CUSTOMER_CLOSING_RESPONSE":
    L = LANG[st.session_state.language]
    st.info("에이전트가 추가 문의 여부를 확인했습니다. 고객의 최종 답변을 자동으로 생성합니다.")

    # 고객 답변 자동 생성 (LLM Key 검증 포함)
    if st.session_state.is_llm_ready:
        with st.spinner(L["generating_customer_response"]):
            # 고객의 최종 답변 생성 (채팅용)
            final_customer_reaction = generate_customer_closing_response(st.session_state.language)

        customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])

        # 로그 기록
        st.session_state.simulator_messages.append(
            {"role": "customer_rebuttal", "content": final_customer_reaction}
        )

        # (A) "없습니다. 감사합니다" 경로 -> FINAL_CLOSING_ACTION으로
        if L['customer_no_more_inquiries'] in final_customer_reaction:
            st.session_state.sim_stage = "FINAL_CLOSING_ACTION"
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )
        # (B) "추가 문의 사항도 있습니다" 경로 -> AGENT_TURN으로 복귀
        elif L['customer_has_additional_inquiries'] in final_customer_reaction:
            st.session_state.sim_stage = "AGENT_TURN"  # 다시 에이전트 응답 단계로
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

        st.session_state.realtime_hint_text = ""  # 힌트 초기화
        # ⭐ 필수 수정: 상태 변경 후 UI 업데이트를 위해 st.rerun() 추가
        # st.rerun()  # 주석 처리: 과도한 rerun 방지 (필요시 주석 해제)

    else:
        st.warning("LLM Key가 없어 고객 반응 자동 생성이 불가합니다. 수동으로 '고객 반응 생성' 버튼을 클릭하거나 AGENT_TURN으로 돌아가세요.")
        if st.button(L["customer_generate_response_button"], key="btn_generate_final_response"):
            # 수동 처리 시 AGENT_TURN으로 넘어가도록 처리
            st.session_state.sim_stage = "AGENT_TURN"
            # st.rerun()  # 주석 처리: 과도한 rerun 방지

    # =========================
    # 9. 최종 종료 행동 (FINAL_CLOSING_ACTION)
    # =========================
if st.session_state.sim_stage == "FINAL_CLOSING_ACTION":
    st.success("고객이 더 이상 문의할 사항이 없다고 확인했습니다.")

    if st.button(L["sim_end_chat_button"], key="btn_final_end_chat"):
        # AHT 타이머 정지
        st.session_state.start_time = None

        end_msg = L["prompt_survey"]
        st.session_state.simulator_messages.append(
            {"role": "system_end", "content": end_msg}
        )
        st.session_state.is_chat_ended = True
        st.session_state.sim_stage = "CLOSING"

        customer_type_display = st.session_state.get("customer_type_sim_select", "")
        save_simulation_history_local(
            st.session_state.customer_query_text_area, customer_type_display,
            st.session_state.simulator_messages, is_chat_ended=True,
            attachment_context=st.session_state.sim_attachment_context_for_llm,
        )

        # ⭐ 재실행
        # st.rerun()  # 주석 처리: 과도한 rerun 방지

elif feature_selection == L["sim_tab_phone"]:
    st.header(L["phone_header"])
    st.markdown(L["simulator_desc"])

    current_lang = st.session_state.language
    L = LANG[current_lang]

    # ========================================
    # 전화 시뮬레이터 로직
    # ========================================

    # ------------------
    # AHT 타이머 표시 (전화 시뮬레이션에서만)
    # ------------------
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
                # st.rerun()  # 주석 처리: 매 초마다 재실행은 성능 문제 유발 (과도한 rerun 방지)

    # ------------------
    # WAIT_FIRST_QUERY / WAITING_CALL 상태
    # ------------------
    if st.session_state.call_sim_stage in ["WAITING_CALL", "RINGING"]:
        st.subheader(L["call_status_waiting"])

        # 초기 문의 입력 (고객이 전화로 말할 내용)
        st.session_state.call_initial_query = st.text_area(
            L["customer_query_label"],
            key="call_initial_query_text_area",
            height=100,
            placeholder=L["call_query_placeholder"],
        )

        customer_type = st.radio(
            L["customer_type_label"],
            L["customer_type_options"],
            key=f"customer_type_sim_select_{st.session_state.sim_instance_id}"
        )
        st.session_state.customer_type_sim_select = customer_type

        # 가상 전화번호 표시
        st.session_state.incoming_phone_number = st.text_input(
            "Incoming Phone Number",
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

        if st.button(L["button_answer"], key="answer_call_btn"):
            if not st.session_state.call_initial_query.strip():
                st.warning(L["simulation_warning_query"])
                st.stop()

            if not st.session_state.is_llm_ready or st.session_state.openai_client is None:
                st.error(L["simulation_no_key_warning"] + " " + L["openai_missing"])
                st.stop()

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

            # 고객의 첫 번째 음성 메시지 (시뮬레이션 시작 메시지)
            initial_query_text = st.session_state.call_initial_query.strip()
            st.session_state.current_customer_audio_text = initial_query_text

            # ⭐ 고객의 첫 문의 TTS 음성 생성 및 저장
            with st.spinner(L["tts_status_generating"] + " (Initial Customer Query)"):
                audio_bytes, msg = synthesize_tts(initial_query_text, st.session_state.language, role="customer")
                if audio_bytes:
                    st.session_state.customer_initial_audio_bytes = audio_bytes
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                else:
                    st.error(f"❌ {msg}")
                    st.session_state.customer_initial_audio_bytes = None

            # ✅ 상태 변경 후 재실행하여 IN_CALL 상태로 전환
            # st.rerun()  # 주석 처리: 과도한 rerun 방지 (필요시 주석 해제)

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

# -------------------- Content Tab --------------------
elif feature_selection == L["content_tab"]:
    st.header(L["content_header"])
    st.markdown(L["content_desc"])
    st.markdown("---")

    if not st.session_state.is_llm_ready:
        st.warning(L["simulation_no_key_warning"])
        st.stop()

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
            st.stop()

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
                    # st.rerun()  # 주석 처리: 과도한 rerun 방지 (필요시 주석 해제)

                except (json.JSONDecodeError, ValueError) as e:
                    # 4. 파싱 실패 또는 데이터 구조 문제 시 에러 메시지 출력
                    st.error(L["quiz_error_llm"])
                    st.caption(f"Error Details: {type(e).__name__} - {e}")
                    st.subheader(L["quiz_original_response"])
                    st.code(generated_json_text, language="json")
                    st.stop()
            else:
                st.error(L["quiz_error_llm"])
                if generated_json_text:
                    st.text_area(L["quiz_original_response"], generated_json_text, height=200)
                st.stop()
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
                # st.rerun()  # 주석 처리: 과도한 rerun 방지 (필요시 주석 해제)
            st.stop()  # 퀴즈 완료 후 스크립트 실행을 완전히 중단

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
                # st.rerun()  # 주석 처리: 과도한 rerun 방지

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
                # st.rerun()  # 주석 처리: 과도한 rerun 방지

        else:
            # 사용자가 이미 정답을 체크했고 (다시 로드된 경우), 다음 버튼을 바로 표시
            if st.session_state.quiz_answers[idx] == 'Correctly Scored' or (
                    isinstance(st.session_state.quiz_answers[idx], int) and st.session_state.quiz_answers[idx] > 0):
                if next_col.button(L["next_question"], key=f"next_question_btn_after_check_{idx}"):
                    st.session_state.current_question_index += 1
                    st.session_state.show_explanation = False
                    # st.rerun()  # 주석 처리: 과도한 rerun 방지

    else:
        # 일반 콘텐츠 (핵심 요약 노트, 실습 예제 아이디어) 출력
        if st.session_state.get("generated_content"):
            content = st.session_state.generated_content

            st.markdown("---")
            st.markdown(f"### {content_display}")

            # --- START: 효율성 개선 (상단 분석/하단 본문) ---

            # 1. 상단 분석 영역: 시각화 대신 키워드/주요 문장 추출 모의 (중복 방지)
            st.subheader("💡 콘텐츠 분석 (시각화 모의)")

            # 콘텐츠를 텍스트 줄로 분할하여 모의 키워드 및 주요 문장 생성
            content_lines = content.split('\n')

            # 모의 키워드 추출 (가장 긴 3개 단어)
            all_words = ' '.join(content_lines).replace('.', '').replace(',', '').split()
            unique_words = sorted(set(all_words), key=len, reverse=True)[:5]

            # 모의 주요 문장 추출 (첫 번째, 가운데, 마지막 문장)
            key_sentences = [
                content_lines[0].strip() if content_lines else "N/A",
                content_lines[len(content_lines) // 2].strip() if len(content_lines) > 1 else "",
                content_lines[-1].strip() if len(content_lines) > 1 else ""
            ]
            key_sentences = [s for s in key_sentences if s]

            col_keyword, col_sentences = st.columns([1, 1])

            with col_keyword:
                st.markdown("**핵심 키워드/개념**")
                st.info(f"[{', '.join(unique_words)}...]")

            with col_sentences:
                st.markdown("**주요 문장 요약**")
                for sentence in key_sentences[:2]:
                    st.write(f"• {sentence[:50]}...")

            st.markdown("---")

            # 2. 하단 본문 출력
            st.markdown(f"### 📝 원본 콘텐츠")
            st.markdown(content)

            # --- END: 효율성 개선 ---

            # --- START: 아이콘 버튼 추가 ---
            st.markdown("---")
            # 콘텐츠를 복사하기 위해 JavaScript 사용 (Streamlit toast와 함께)
            js_copy_script = f"""
                function copyToClipboard(text) {{
                    navigator.clipboard.writeText(text).then(function() {{
                        // Streamlit toast 호출 (모의)
                        const elements = window.parent.document.querySelectorAll('[data-testid="stToast"]');
                        if (elements.length === 0) {{
                            // Fallback UI update (use Streamlit's native mechanism if possible, or simple alert)
                            alert("복사 완료!"); 
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
                copyToClipboard('{content.replace( / '/g, "\\'").replace(/\\n/g, " ")}');
                                                     """
                                       
                                                   col_like, col_dislike, col_share, col_copy, col_more = st.columns([1, 1, 1, 1, 6])
                                       
                                                   # 좋아요 버튼
                                                   if col_like.button("👍", key="content_like"):
                                                       st.toast("✅ '좋아요' 기능 활성화 예정")
                                       
                                                   # 싫어요 버튼
                                                   if col_dislike.button("👎", key="content_dislike"):
                                                       st.toast("✅ '싫어요' 기능 활성화 예정")
                                       
                                                   # 공유 버튼
                                                   if col_share.button("🔗", key="content_share"):
                                                       st.toast("✅ '공유' 기능 활성화 예정")
                                       
                                                   # 복사 버튼 (기능 활성화)
                                                   if col_copy.button("📋", key="content_copy"):
                                                       # Streamlit에서 직접 JavaScript를 실행하여 복사
                                                       st.components.v1.html(
                                                           f""" < script > {js_copy_script} </ script > """,
                    height=0,
                )
                st.toast("✅ 콘텐츠가 클립보드에 복사되었습니다!")

            # 더보기 버튼
            if col_more.button("•••", key="content_more"):
                st.toast("✅ '더보기' 기능 활성화 예정")
            # --- END: 아이콘 버튼 추가 ---
for idx, msg in enumerate(st.session_state.simulator_messages):
    role = msg["role"]
    content = msg["content"]
    avatar = {"customer": "🙋", "supervisor": "🤖", "agent_response": "🧑‍💻", "customer_rebuttal": "✨",
              "system_end": "📌"}.get(role, "💬")
    tts_role = "customer" if role.startswith("customer") or role == "customer_rebuttal" else (
       "agent" if role == "agent_response" else "supervisor")

    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        # 인덱스를 render_tts_button에 전달하여 고유 키 생성에 사용
        render_tts_button(content, st.session_state.language, role=tts_role, prefix=f"{role}_", index=idx)