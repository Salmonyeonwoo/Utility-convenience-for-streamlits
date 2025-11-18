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
        "prompt_survey": "문의해 주셔서 감사합니다. 필요하시면 언제든지 연락주세요。",
        "customer_closing_confirm": "또 다른 문의 사항은 없으신가요?",
        "customer_positive_response": "친절한 상담 감사드립니다。",
        "button_end_chat": "응대 종료 (설문 요청)",
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
        "openai_missing": "OpenAI API Key가 없습니다.",
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
        "simulator_tab": "AI Customer Response Simulator",
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
        "prompt_survey": "Thank you for contacting support.",
        "customer_closing_confirm": "Anything else I can help you with?",
        "customer_positive_response": "Thank you for your kind support.",
        "button_end_chat": "End Chat (Survey Request)",
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
        "empty_response_warning": "Please enter a response."
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
        "prompt_survey": "お問い合わせありがとうございました。",
        "customer_closing_confirm": "他のお問合せはございませんでしょうか。",
        "customer_positive_response": "ご丁寧な対応ありがとうございました。",
        "button_end_chat": "チャット終了（アンケート）",
        "agent_response_header": "✍️ エージェント応答",
        "agent_response_placeholder": "顧客へ返信内容を入力…",
        "send_response_button": "返信送信",
        "request_rebuttal_button": "顧客の反応を生成",
        "new_simulation_button": "新規シミュレーション",
        "history_selectbox_label": "履歴を選択:",
        "history_load_button": "履歴を読み込む",
        "delete_history_button": "❌ 全履歴削除",
        "delete_confirm_message": "すべての履歴を削除しますか？",
        "delete_confirm_yes": "はい、削除する",
        "delete_confirm_no": "キャンセル",
        "delete_success": "削除完了！",
        "deleting_history_progress": "削除中...",
        "search_history_label": "履歴検索",
        "date_range_label": "日付フィルター",
        "no_history_found": "該当する履歴はありません。",
        "customer_email_label": "顧客メールアドレス（任意）",
        "customer_phone_label": "顧客連絡先 / 電話番号（任意）",

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
        "empty_response_warning": "応答を入力してください。"
    }
}



# ========================================
# 1-1. Session State 초기화
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

if "openai_client" not in st.session_state:
    st.session_state.openai_client = None
if "openai_init_msg" not in st.session_state:
    st.session_state.openai_init_msg = ""

L = LANG[st.session_state.language]

# ========================================
# 2. OpenAI Client 초기화 (secrets 사용 안 함)
# ========================================

# @st.cache_resource
# ========================================
# 0-A. API Key 안전 구조 (Secrets + User Input)
# ========================================

# 1) Streamlit Cloud Secrets에서 우선 가져오기


secret_key = None

try:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        secret_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    secret_key = None

# 2) 사용자 입력 키 (세션에 저장)
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

# 3) UI 제공: 사용자가 직접 입력하는 백업 API Key
with st.sidebar:
    st.markdown("### 🔐 OpenAI API Key 설정")

    if secret_key:
        st.success("✔ Streamlit Secrets API Key 감지됨 (자동 적용)")
    else:
        st.warning("⚠ Streamlit Secrets에 API Key 없음 — 직접 입력 필요")

    user_key_input = st.text_input(
        "직접 OpenAI API Key 입력 (선택)",
        type="password",
        key="user_key_input_box",
        placeholder="sk-************************"
    )

    if st.button("API Key 적용"):
        if user_key_input.strip():
            st.session_state.user_api_key = user_key_input.strip()
            st.success("🔑 사용자 API Key 등록 완료! (세션 내 임시 저장)")
        else:
            st.warning("API Key를 입력하세요.")


# 4) 최종 API Key 선택 우선순위
def get_active_api_key():
    """
    1) Streamlit Cloud Secrets
    2) 사용자 입력 키
    3) 아무것도 없으면 None
    """
    if secret_key:
        return secret_key
    if st.session_state.user_api_key:
        return st.session_state.user_api_key
    return None


def init_openai_client():
    openai_key = get_active_api_key()
    if not openai_key:
        return None, LANG[DEFAULT_LANG]["openai_missing"]
    try:
        client = OpenAI(api_key=openai_key)
        return client, "✅ OpenAI 클라이언트 준비 완료"
    except Exception as e:
        return None, f"OpenAI client init error: {e}"


openai_client_obj, openai_msg = init_openai_client()
st.session_state.openai_client = openai_client_obj
st.session_state.openai_init_msg = openai_msg

# ========================================
# 3. Whisper / TTS Helper
# ========================================

def transcribe_bytes_with_whisper(audio_bytes: bytes, mime_type: str = "audio/webm", lang_code: str = "ko") -> str:
    L = LANG[st.session_state.language]
    client = st.session_state.openai_client
    if client is None:
        return f"❌ {L['openai_missing']}"

    # 언어 코드 매핑
    whisper_lang = {"ko": "ko", "en": "en", "ja": "ja"}.get(lang_code, "en")

    ext = "webm"
    if "/" in mime_type:
        ext = mime_type.split("/")[-1].lower()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
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
        return res.strip() if isinstance(res, str) else str(res)
    except Exception as e:
        return f"❌ {L['error']} Whisper: {e}"
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def synthesize_tts(text: str, lang_key: str):
    L = LANG[lang_key]
    client = st.session_state.openai_client

    if client is None:
        return None, f"❌ {L['tts_status_error']} (Client Missing)"

    try:
        # TTS 생성: format 파라미터 절대 넣지 말 것
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )

        # 모든 SDK 버전에서 작동하는 안전한 방식
        audio_bytes = response.read()

        return audio_bytes, f"✅ {L['tts_status_success']}"

    except Exception as e:
        return None, f"❌ {L['tts_status_error']} (OpenAI TTS Error: {e})"


def render_tts_button(text: str, lang_key: str, prefix: str = ""):
    L = LANG[lang_key]

    # 완전 고유 key 생성 (message, prefix, time 조합)
    unique_key = prefix + "_tts_" + hashlib.md5(
        (text + prefix + str(time.time())).encode("utf-8")
    ).hexdigest()

    if st.button(L["button_listen_audio"], key=unique_key):
        audio_bytes, msg = synthesize_tts(text, lang_key)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
        else:
            st.error(msg)


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
# 5. 로컬 시뮬레이션 이력 Helper
# ========================================

def load_simulation_histories_local(lang_key: str) -> List[Dict[str, Any]]:
    histories = _load_json(SIM_META_FILE, [])
    return [
        h for h in histories
        if h.get("language_key") == lang_key and isinstance(h.get("messages"), list)
    ]


def save_simulation_history_local(initial_query: str, customer_type: str, messages: List[Dict[str, Any]], is_chat_ended: bool):
    histories = _load_json(SIM_META_FILE, [])
    doc_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()
    data = {
        "id": doc_id,
        "initial_query": initial_query,
        "customer_type": customer_type,
        "messages": messages,
        "language_key": st.session_state.language,
        "timestamp": ts,
        "is_chat_ended": is_chat_ended,
    }
    histories.insert(0, data)
    _save_json(SIM_META_FILE, histories)
    return True


def delete_all_history_local():
    _save_json(SIM_META_FILE, [])

# ========================================
# 6. RAG Helper (FAISS)
# ========================================

def load_documents(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        name = f.name
        lower = name.lower()
        if lower.endswith(".pdf"):
            # UploadedFile -> temp 파일로
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


def build_rag_index(files, embeddings):
    if not files:
        st.warning(L["warning_no_files"])
        return None, 0

    docs = load_documents(files)
    if not docs:
        st.warning("문서를 불러오지 못했습니다.")
        return None, 0

    chunks = split_documents(docs)
    if not chunks:
        st.warning("문서 청크 분할에 실패했습니다.")
        return None, 0

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # 저장
        vectorstore.save_local(RAG_INDEX_DIR)
    except Exception as e:
        st.error(f"RAG 인덱스 생성 중 오류: {e}")
        return None, 0

    return vectorstore, len(chunks)


def load_rag_index(embeddings):
    try:
        vs = FAISS.load_local(RAG_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception:
        return None


def rag_answer(question: str, vectorstore: FAISS, lang_key: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return LANG[lang_key]["openai_missing"]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=api_key,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content[:1500] for d in docs)

    prompt_tmpl = PromptTemplate(
        template=(
            "You are a helpful AI tutor. Answer the question using ONLY the provided context.\n"
            "If you cannot find the answer in the context, say you don't know.\n\n"
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        ),
        input_variables=["question", "context"],
    )
    prompt = prompt_tmpl.format(question=question, context=context)
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)

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
# ========================================

LLM_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"

if "llm" not in st.session_state:
    if LLM_API_KEY:
        try:
            st.session_state.llm = ChatOpenAI(
                model=LLM_MODEL,
                temperature=0.7,
                openai_api_key=LLM_API_KEY,
            )
            st.session_state.embeddings = OpenAIEmbeddings(openai_api_key=LLM_API_KEY)
            st.session_state.is_llm_ready = True

            sim_prompt = PromptTemplate(
                template=(
                    "You are an AI customer who responds ONLY as the customer in the scenario.\n"
                    "Do NOT greet unless the agent greets first.\n"
                    "Do NOT repeat your initial message.\n"
                    "Always answer in the language used by the agent.\n\n"
                    "Rules:\n"
                    "- If the agent requests specific information, provide ONLY ONE detail.\n"
                    "- If the agent provides a solution, respond politely.\n"
                    "- If the conversation is nearing completion, optionally add a closing remark.\n"
                    "- DO NOT generate long formal greetings like 'Good morning'.\n"
                    "- DO NOT reset context.\n\n"
                    "{chat_history}\nHuman agent: {input}\nCustomer:"
                ),
                input_variables=["input", "chat_history"],
            )

            st.session_state.simulator_chain = ConversationChain(
                llm=st.session_state.llm,
                memory=st.session_state.simulator_memory,
                prompt=sim_prompt,
                input_key="input",
            )
        except Exception as e:
            st.session_state.llm_init_error_msg = f"{L['llm_error_init']} (OpenAI): {e}"
            st.session_state.is_llm_ready = False
    else:
        st.session_state.llm_init_error_msg = LANG[DEFAULT_LANG]["openai_missing"]
        st.session_state.is_llm_ready = False

# ========================================
# 9. 사이드바 (언어 선택 + 파일 업로드 + 분석 버튼)
# ========================================

with st.sidebar:
    selected_lang_key = st.selectbox(
        L["lang_select"],
        options=["ko", "en", "ja"],
        index=["ko", "en", "ja"].index(st.session_state.language),
        format_func=lambda x: {"ko": "한국어", "en": "English", "ja": "日本語"}[x],
    )

    # 🔹 언어 변경 감지
    if selected_lang_key != st.session_state.language:
        old_lang = st.session_state.language
        st.session_state.language = selected_lang_key
        L = LANG[st.session_state.language]

        # 🔹 시뮬레이터 관련 상태 초기화
        st.session_state.simulator_messages = []
        st.session_state.simulator_memory.clear()
        st.session_state.initial_advice_provided = False
        st.session_state.is_chat_ended = False
        st.session_state.agent_response_area_text = ""
        st.session_state.last_transcript = ""
        st.session_state.sim_audio_bytes = None
        st.session_state.sim_audio_bytes_raw = None

        # (원하면 RAG 채팅 이력도 언어별로 분리하고 싶을 때)
        # st.session_state.messages = []

        # st.rerun()


    L = LANG[st.session_state.language]

    st.title(L["sidebar_title"])
    st.markdown("---")

    st.subheader("클라이언트 초기화 상태")
    if st.session_state.llm_init_error_msg:
        st.error(st.session_state.llm_init_error_msg)
    elif st.session_state.is_llm_ready:
        st.success("✅ LLM 및 임베딩 클라이언트 준비 완료")

    if "✅" in st.session_state.openai_init_msg:
        st.success(st.session_state.openai_init_msg)
    else:
        st.warning(st.session_state.openai_init_msg)

    st.markdown("---")

    uploaded_files_widget = st.file_uploader(
        L["file_uploader"], type=["pdf", "txt", "html"], accept_multiple_files=True
    )
    if uploaded_files_widget:
        st.session_state.uploaded_files_state = uploaded_files_widget

    files_to_process = st.session_state.uploaded_files_state or []

    if files_to_process and st.session_state.is_llm_ready:
        if st.button(L["button_start_analysis"]):
            with st.spinner(L["data_analysis_progress"]):
                vs, count = build_rag_index(files_to_process, st.session_state.embeddings)
                if vs is not None:
                    st.session_state.rag_vectorstore = vs
                    st.session_state.is_rag_ready = True
                    st.success(L["embed_success"].format(count=count))
                else:
                    st.session_state.is_rag_ready = False
    elif not files_to_process:
        st.info(L["warning_no_files"])

    st.markdown("---")

    feature_selection = st.radio(
        "기능 선택",
        [L["rag_tab"], L["content_tab"], L["lstm_tab"], L["simulator_tab"], L["voice_rec_header"]],
    )

# 메인 타이틀
st.title(L["title"])

# ========================================
# 10. 기능별 페이지
# ========================================

# -------------------- Voice Record Tab --------------------
if feature_selection == L["voice_rec_header"]:
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
                # st.rerun()
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
                                    # st.rerun()
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
                            # st.rerun()
                        else:
                            st.session_state[f"confirm_del_{rec_id}"] = True
                            st.warning(L["delete_confirm_rec"])

# -------------------- Simulator Tab --------------------
elif feature_selection == L["simulator_tab"]:
    st.header(L["simulator_header"])
    st.markdown(L["simulator_desc"])

    st.markdown(
        f'<div style="padding:5px;text-align:center;border-radius:5px;background-color:#f0f0f0;margin-bottom:10px;">{L["tts_status_ready"]}</div>',
        unsafe_allow_html=True,
    )

    # 전체 이력 삭제
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
                    st.success(L["delete_success"])
                    # st.rerun()
            if c_no.button(L["delete_confirm_no"], key="confirm_del_no"):
                st.session_state.show_delete_confirm = False
                # st.rerun()

    current_lang = st.session_state.language

    # 이력 로드
    with st.expander(L["history_expander_title"]):
        histories = load_simulation_histories_local(current_lang)
        search_query = st.text_input(L["search_history_label"], key="sim_hist_search")

        today = datetime.now().date()
        dr = st.date_input(
            L["date_range_label"],
            value=[today - timedelta(days=7), today],
            key="sim_hist_date_range",
        )

        filtered = []
        if histories:
            if isinstance(dr, list) and len(dr) == 2:
                start_date = min(dr)
                end_date = max(dr)
            else:
                start_date = datetime.min.date()
                end_date = datetime.max.date()

            for h in histories:
                ok_search = True
                if search_query:
                    q = search_query.lower()
                    text = (h["initial_query"] + " " + h["customer_type"]).lower()
                    if q not in text:
                        ok_search = False

                ok_date = True
                ts = h.get("timestamp")
                if ts:
                    try:
                        d = datetime.fromisoformat(ts).date()
                        if not (start_date <= d <= end_date):
                            ok_date = False
                    except Exception:
                        pass

                if ok_search and ok_date:
                    filtered.append(h)

        if filtered:
            def _label(h):
                try:
                    t = datetime.fromisoformat(h["timestamp"])
                    t_str = t.strftime("%m-%d %H:%M")
                except Exception:
                    t_str = h.get("timestamp", "")
                q = h["initial_query"][:30].replace("\n", " ")
                return f"[{t_str}] {h['customer_type']} - {q}..."

            options_map = { _label(h): h for h in filtered }
            sel_key = st.selectbox(L["history_selectbox_label"], options=list(options_map.keys()))
            if st.button(L["history_load_button"], key="load_hist_btn"):
                h = options_map[sel_key]
                st.session_state.customer_query_text_area = h["initial_query"]
                st.session_state.simulator_messages = h["messages"]
                st.session_state.initial_advice_provided = True
                st.session_state.is_chat_ended = h.get("is_chat_ended", False)

                st.session_state.simulator_memory.clear()
                for msg in h["messages"]:
                    role = msg["role"]
                    if role in ["customer", "agent_response"]:
                        st.session_state.simulator_memory.chat_memory.add_user_message(msg["content"])
                    else:
                        st.session_state.simulator_memory.chat_memory.add_ai_message(msg["content"])

                st.rerun()
        else:
            st.info(L["no_history_found"])

    # LLM 없으면 시뮬레이터 제한
    if not st.session_state.is_llm_ready and not LLM_API_KEY:
        st.warning(L["simulation_no_key_warning"])

    if st.session_state.is_chat_ended:
        st.success(L["prompt_customer_end"] + " " + L["prompt_survey"])
        if st.button(L["new_simulation_button"], key="new_simulation_btn"):
            st.session_state.is_chat_ended = False
            st.session_state.initial_advice_provided = False
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.last_transcript = ""
            st.session_state.agent_response_area_text = ""
            st.session_state.customer_query_text_area = ""
            # st.rerun()
        st.stop()

    # 초기 문의 입력
    customer_query = st.text_area(
        L["customer_query_label"],
        key="customer_query_text_area",
        height=150,
        placeholder=L["initial_query_sample"],
        value=st.session_state.agent_response_area_text,
        disabled=st.session_state.initial_advice_provided,
    )

    # 🔹 새로 추가: 고객 연락처 (선택)
    customer_email = st.text_input(
        L.get("customer_email_label", "Customer email (optional)"),
        key="customer_email",
        disabled=st.session_state.initial_advice_provided,
    )
    customer_phone = st.text_input(
        L.get("customer_phone_label", "Customer phone / WhatsApp (optional)"),
        key="customer_phone",
        disabled=st.session_state.initial_advice_provided,
    )

    customer_type_options = L["customer_type_options"]
    default_idx = 1 if len(customer_type_options) > 1 else 0
    customer_type_display = st.selectbox(
        L["customer_type_label"],
        customer_type_options,
        index=default_idx,
        disabled=st.session_state.initial_advice_provided,
        key="customer_type_sim_select",
    )

    if st.button(L["button_simulate"], disabled=st.session_state.initial_advice_provided):
        if not customer_query.strip():
            st.warning(L["simulation_warning_query"])
            st.stop()

        st.session_state.simulator_memory.clear()
        st.session_state.simulator_messages = []
        st.session_state.is_chat_ended = False

        st.session_state.simulator_messages.append({"role": "customer", "content": customer_query})
        st.session_state.simulator_memory.chat_memory.add_user_message(customer_query)

        contact_info_block = ""
        if customer_email or customer_phone:
            contact_info_block = (
                f"\n\n[Customer contact info for your reference]"
                f"\n- Email: {customer_email or 'N/A'}"
                f"\n- Phone: {customer_phone or 'N/A'}"
            )

        current_lang_key = st.session_state.language

        initial_prompt = f"""
        You are an AI Customer Support Supervisor. Your role is to analyze the following customer inquiry
        from a **{customer_type_display}** and provide:

        1) A detailed **response guideline for the human agent** (step-by-step).
        2) A **ready-to-send draft reply** in {LANG[current_lang_key]['lang_select']}.


        [CRITICAL RULE 1: LANGUAGE]
        - All content (guideline AND draft) MUST be written strictly in {LANG[current_lang_key]['lang_select']}.

        [CRITICAL RULE 2: FORMAT]
        - Use the exact markdown headers:
          - "### {L['simulation_advice_header']}"
          - "### {L['simulation_draft_header']}"

        [CRITICAL RULE 3: INFORMATION YOU MUST ASK FIRST]
        Before solving the problem, list the essential details the agent must collect from the customer.
        In the guideline, always include a section like "1. 정보 수집 / Information to collect" with bullet points such as:
        - For eSIM / connectivity issues:
          - Device model (e.g. iPhone 12, Galaxy S22)
          - OS version
          - Whether the device supports eSIM
          - Current location / country and whether the customer has already arrived
          - Exact activation steps already tried and at which step it failed
        - For tickets with children:
          - Number of children
          - Each child's date of birth or age range
          - Whether the ticket type changes with age (free / child / youth / adult)
        - Any booking ID, voucher number, or reservation code
        - Customer's preferred contact channel if follow-up is needed.

        [CRITICAL RULE 4: DRAFT STYLE]
        - The draft reply should:
          - Politely thank the customer.
          - Clearly ask for the missing information listed above (but not all in one long sentence).
          - Explain the next troubleshooting steps in simple language.
          - For eSIM cases, mention important checks (airplane mode, roaming settings, APN, profile installation, etc.) if relevant.
          - For child ticket cases, clearly explain how the pricing works by age.

        [CRITICAL RULE 5: ROLEPLAY FOR FUTURE MESSAGES]
        When the Agent subsequently asks for information in later rounds,
        **ROLEPLAY as the customer** who is frustrated but **HIGHLY COOPERATIVE** and
        provide the requested details piece by piece (not all at once).
        The customer MUST NOT argue about why the information is needed.
        
        [CRITICAL RULE 6: ASK FOR ALL REQUIRED DETAILS AT ONCE]
        When composing the draft reply:
        - Do NOT ask one-by-one questions.
        - Instead, request ALL required details in a neatly formatted multi-bullet list.
        - Each bullet point must contain only ONE information category.

        Customer Inquiry:
        {customer_query}
        {contact_info_block}
        """

        if not st.session_state.is_llm_ready or not LLM_API_KEY:
            mock_text = (
                f"### {L['simulation_advice_header']}\n\n"
                f"- (Mock) {customer_type_display} 유형 고객에 대한 응대 가이드라인입니다.\n\n"
                f"### {L['simulation_draft_header']}\n\n"
                f"(Mock) 여기에는 실제 AI 응대 초안이 들어갑니다.\n\n"
            )
            st.session_state.simulator_messages.append({"role": "supervisor", "content": mock_text})
            st.session_state.simulator_memory.chat_memory.add_ai_message(mock_text)
            st.session_state.initial_advice_provided = True
            save_simulation_history_local(
                customer_query,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=False,
            )
            st.warning(L["simulation_no_key_warning"])
            # st.rerun()
        else:
            with st.spinner(L["response_generating"]):
                try:
                    text = st.session_state.simulator_chain.predict(input=initial_prompt)
                    st.session_state.simulator_messages.append({"role": "supervisor", "content": text})
                    st.session_state.initial_advice_provided = True
                    save_simulation_history_local(
                        customer_query,
                        customer_type_display,
                        st.session_state.simulator_messages,
                        is_chat_ended=False,
                    )
                    # st.rerun()
                except Exception as e:
                    st.error(f"AI 조언 생성 중 오류 발생: {e}")

    # 대화 로그 표시
    # 대화 로그 표시
    for msg in st.session_state.simulator_messages:
        role = msg["role"]
        content = msg["content"]

        if role == "customer":
            with st.chat_message("user", avatar="🙋"):
                st.markdown(content)
                render_tts_button(content, st.session_state.language, prefix="customer_")

        elif role == "supervisor":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                render_tts_button(content, st.session_state.language, prefix="supervisor_")

        elif role == "agent_response":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(content)
                render_tts_button(content, st.session_state.language, prefix="agent_")

        elif role in ["customer_rebuttal", "customer_end", "system_end"]:
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(content)
                render_tts_button(content, st.session_state.language, prefix=f"{role}_")

    # 에이전트 응답 / 마이크 입력
    # 에이전트 응답 / 마이크 입력
    if st.session_state.initial_advice_provided and not st.session_state.is_chat_ended:

        last_role = (
            st.session_state.simulator_messages[-1]["role"]
            if st.session_state.simulator_messages else None
        )

        if last_role in ["customer", "supervisor", "customer_rebuttal", "customer_end"]:
            st.markdown(f"### {L['agent_response_header']}")
            col_mic, col_text = st.columns([1, 2])

            # 마이크 녹음
            with col_mic:
                mic_audio = mic_recorder(
                    start_prompt=L["button_mic_input"],
                    stop_prompt="⏹️ 녹음 종료",
                    just_once=False,
                    format="wav",
                    use_container_width=True,
                    key="sim_mic_recorder",
                )

            new_audio_bytes = mic_audio["bytes"] if mic_audio else None

            if new_audio_bytes is not None:
                st.session_state.sim_audio_bytes = new_audio_bytes
                st.info("✅ 녹음 완료! 아래 전사 버튼을 눌러 텍스트로 변환하세요.")

            if st.session_state.sim_audio_bytes:
                st.audio(st.session_state.sim_audio_bytes, format="audio/wav")

            # 전사 버튼
            col_tr, _ = st.columns([1, 2])
            if col_tr.button(L["transcribe_btn"], key="sim_transcribe_btn"):
                if st.session_state.sim_audio_bytes is None:
                    st.warning("먼저 마이크로 녹음을 완료하세요.")
                elif st.session_state.openai_client is None:
                    st.error(L["whisper_client_error"])
                else:
                    # 🔹 여기서 실제 전사 대상 오디오/포맷을 정의
                    audio_bytes_to_transcribe = st.session_state.sim_audio_bytes
                    audio_mime_to_transcribe = "audio/wav"  # mic_recorder(format="wav") 이라서 고정

                    with st.spinner(
                            L.get("whisper_processing", "음성 파일을 텍스트로 변환 중...")
                    ):
                        try:
                            transcribed_text = transcribe_bytes_with_whisper(
                                audio_bytes_to_transcribe,
                                audio_mime_to_transcribe,
                                # 언어키는 세션에서 직접 가져오는 게 더 안전
                                lang_code=st.session_state.language,
                            )

                            if transcribed_text.startswith("❌"):
                                st.error(transcribed_text)
                                st.session_state.last_transcript = ""
                            else:
                                # 마지막 전사 내용과 에이전트 응답창에 동시에 반영
                                st.session_state.last_transcript = transcribed_text
                                st.session_state.agent_response_area_text = transcribed_text.strip()
                                st.session_state.last_transcript = transcribed_text.strip()

                                snippet = transcribed_text[:50].replace("\n", " ") + (
                                    "..." if len(transcribed_text) > 50 else ""
                                )

                                success_msg = L.get(
                                    "whisper_success",
                                    "✅ 음성 전사 완료! 텍스트 창을 확인하세요."
                                ) + f"\n\n**인식 내용:** *{snippet}*"

                                st.success(success_msg)

                        except Exception as e:
                            st.error(f"Whisper Error: {e}")

                            if transcribed_text.startswith("❌"):
                                st.error(transcribed_text)
                                st.session_state.last_transcript = ""
                            else:
                                st.session_state.last_transcript = transcribed_text
                                st.session_state.agent_response_area_text = transcribed_text

                                snippet = (
                                        transcribed_text[:50].replace("\n", " ")
                                        + ("..." if len(transcribed_text) > 50 else "")
                                )

                                success_msg = (
                                        L.get("whisper_success",
                                              "✅ 음성 전사 완료! 텍스트 창을 확인하세요.")
                                        + f"\n\n**인식 내용:** *{snippet}*"
                                )
                        except Exception as e:
                            st.error(f"Whisper Error: {e}")

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # 여기서부터가 문제였던 부분 — 정렬 완전 수정
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

            col_text, col_button = st.columns([4, 1])

            with col_text:
                agent_response = st.text_area(
                    L["agent_response_placeholder"],
                    value=st.session_state.agent_response_area_text,
                    height=150,
                    key="agent_response_text_area"
                )

            with col_button:
                send_clicked = st.button(L["send_response_button"], key="send_response_btn")

            if send_clicked:
                if not agent_response.strip():
                    st.warning(L["empty_response_warning"])
                else:
                    st.session_state.last_transcript = agent_response
                    st.session_state.agent_response_area_text = ""
                    st.session_state.sim_audio_bytes = None

                    st.session_state.simulator_messages.append(
                        {"role": "agent_response", "content": agent_response}
                    )
                    st.session_state.simulator_memory.chat_memory.add_user_message(agent_response)

                    save_simulation_history_local(
                        st.session_state.customer_query_text_area,
                        customer_type_display,
                        st.session_state.simulator_messages,
                        is_chat_ended=False,
                    )

                    # st.rerun()

        # 에이전트 응답 이후: 종료/다음 반응
        last_role = st.session_state.simulator_messages[-1]["role"] if st.session_state.simulator_messages else None

        if last_role == "agent_response":

            st.markdown("### 🤖 고객 반응 생성")

            if st.button(L["customer_generate_response_button"], key="btn_generate_customer"):
                next_prompt = f"""
                You are the CUSTOMER. Respond naturally to the agent's latest message.

                RULES:
                1. If the agent requested information → provide exactly ONE missing detail.
                2. If the agent provided a solution → respond with appreciation.
                3. Appreciation must include a positive phrase like:
                   "{L['customer_positive_response']}"
                4. After appreciation, customer MUST wait for the agent to ask:
                   "{L['customer_closing_confirm']}"
                5. Language must be {LANG[st.session_state.language]['lang_select']}.
                """

                with st.spinner(L["response_generating"]):
                    reaction = st.session_state.simulator_chain.predict(input=next_prompt)

                st.session_state.simulator_messages.append(
                    {"role": "customer", "content": reaction}
                )
                st.session_state.simulator_memory.chat_memory.add_ai_message(reaction)

                st.stop()

        if last_role == "customer":
            customer_text = st.session_state.simulator_messages[-1]["content"].strip().lower()

            appreciation_patterns = ["감사", "thank", "ありがとうございます", "ありがとう", "감사합니다"]
            closing_patterns = ["없습니다", "없어요", "없어", "no more", "nothing else", "結構です", "大丈夫です"]

            # 1) 고객이 감사 인사를 한 경우
            if any(p in customer_text for p in appreciation_patterns):

                st.info("고객이 감사 인사를 했습니다. 에이전트가 추가 문의 여부를 확인해야 합니다.")

                if st.button(L["send_closing_confirm_button"], key="btn_send_closing_confirm"):
                    closing_msg = L["customer_closing_confirm"]

                    st.session_state.simulator_messages.append(
                        {"role": "supervisor", "content": closing_msg}
                    )
                    st.session_state.simulator_memory.chat_memory.add_ai_message(closing_msg)

                    st.success("추가 문의 여부 확인 메시지가 전송되었습니다.")
                    st.stop()

            # 2) 고객이 “추가 문의 없음”을 표현한 경우
            elif any(p in customer_text for p in closing_patterns):

                st.success("고객이 더 이상 문의가 없다고 말했습니다.")

                end_msg = L["prompt_survey"]
                st.session_state.simulator_messages.append({"role": "system_end", "content": end_msg})
                st.session_state.is_chat_ended = True

                save_simulation_history_local(
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.simulator_messages,
                    is_chat_ended=True,
                )

                st.info("📌 상담 종료 단계입니다. 설문조사 메시지를 전송할 수 있습니다.")
                st.stop()

            # 3) 그 외의 경우 → 일반적인 추가 질문
            else:
                pass  # 에이전트 응답 UI 그대로 유지됨

        if last_role == "agent_response":
            col_end, col_next = st.columns([1, 2])

            if col_end.button(L["button_end_chat"], key="sim_end_chat_btn"):
                # 고객에게 "추가 문의 확인" 먼저 보내기
                closing_query = L["customer_closing_confirm"]

                st.session_state.simulator_messages.append(
                    {"role": "supervisor", "content": closing_query}
                )
                st.session_state.simulator_memory.chat_memory.add_ai_message(closing_query)

                # 설문 메시지는 고객이 “없습니다”라고 한 뒤에만 전송
                st.stop()
                # st.rerun()

                if col_next.button(L["request_rebuttal_button"], key="sim_next_rebuttal_btn"):
                    next_prompt = """ ... (LLM에게 customer role 요청) ... """

                    with st.spinner(L["response_generating"]):
                        reaction = st.session_state.simulator_chain.predict(input=next_prompt)

                    st.session_state.simulator_messages.append(
                        {"role": "customer_rebuttal", "content": reaction}
                    )
                    st.session_state.simulator_memory.chat_memory.add_ai_message(reaction)

                    save_simulation_history_local(
                        st.session_state.customer_query_text_area,
                        customer_type_display,
                        st.session_state.simulator_messages,
                        is_chat_ended=False,
                    )

                    st.stop()

                if not st.session_state.is_llm_ready or not LLM_API_KEY:
                    st.warning("API Key가 없어 대화형 시뮬레이션은 불가능합니다.")
                    st.stop()

                # -----------------------------
                # 1) supervisor → customer 역할로 변환 (LLM)
                # -----------------------------
                next_prompt = f"""
            You are now ROLEPLAYING as the CUSTOMER.

            Analyze the dialogue so far and respond naturally.

            RULES:
            1. If the agent requested information → provide EXACTLY ONE missing detail.
            2. If the agent provided a solution → respond with appreciation.
            3. If appreciation is given → ALWAYS respond with:
               "{L['customer_closing_confirm']}"
            4. If the agent already asked:
               "{L['customer_closing_confirm']}"
               AND the customer has no further questions:
               → Respond with "{L['customer_positive_response']}"
               → THEN the chat MUST END.
            5. Language MUST be {LANG[st.session_state.language]['lang_select']}.
                """

                # LLM 실행
                with st.spinner(L["response_generating"]):
                    reaction = st.session_state.simulator_chain.predict(input=next_prompt)

                reaction_lower = reaction.lower()

                # 패턴 정의
                closing_user_signals = [
                    "없습니다", "없어요", "없어",
                    "no more", "nothing else",
                    "結構です", "大丈夫です"
                ]

                appreciation_signals = [
                    "감사", "thank", "ありがとう"
                ]

                # -----------------------------
                # 2) 고객이 "종료 의사" 전달
                # -----------------------------
                if any(k in reaction_lower for k in closing_user_signals):
                    st.session_state.simulator_messages.append(
                        {"role": "customer_end", "content": reaction}
                    )
                    st.session_state.simulator_messages.append(
                        {"role": "system_end", "content": L["prompt_survey"]}
                    )

                    st.session_state.simulator_memory.chat_memory.add_ai_message(reaction)
                    st.session_state.simulator_memory.chat_memory.add_ai_message(L["prompt_survey"])

                    st.session_state.is_chat_ended = True

                    save_simulation_history_local(
                        st.session_state.customer_query_text_area,
                        customer_type_display,
                        st.session_state.simulator_messages,
                        is_chat_ended=True,
                    )
                    st.stop()

                # -----------------------------
                # 3) 고객이 감사 메시지 보내옴 → supervisor가 closing 질문 자동 발송
                # -----------------------------
                # if any(k in reaction_lower for k in appreciation_signals):
                #     # 고객 감사 메시지
                #     st.session_state.simulator_messages.append(
                #         {"role": "customer_rebuttal", "content": reaction}
                #     )
                #     st.session_state.simulator_memory.chat_memory.add_ai_message(reaction)
                #
                #     follow_up = L["customer_closing_confirm"]
                #
                #     # supervisor가 추가 문의 여부 질문
                #     st.session_state.simulator_messages.append(
                #         {"role": "supervisor", "content": follow_up}
                #     )
                #     st.session_state.simulator_memory.chat_memory.add_ai_message(follow_up)
                #
                #     save_simulation_history_local(
                #         st.session_state.customer_query_text_area,
                #         customer_type_display,
                #         st.session_state.simulator_messages,
                #         is_chat_ended=False,
                #     )
                #     st.stop()

                # -----------------------------
                # 4) 기타 일반 반응
                # -----------------------------
                st.session_state.simulator_messages.append(
                    {"role": "customer_rebuttal", "content": reaction}
                )
                st.session_state.simulator_memory.chat_memory.add_ai_message(reaction)

                save_simulation_history_local(
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.simulator_messages,
                    is_chat_ended=False,
                )
                st.stop()

                # 2) 고객이 감사 인사 → 반드시 “추가 문의 여부” 확인 메시지 발송
                # if any(k in reaction_lower for k in appreciation_signals):
                #     follow_up = L["customer_closing_confirm"]
                #
                #     st.session_state.simulator_messages.append(
                #         {"role": "customer_rebuttal", "content": reaction}
                #     )
                #     st.session_state.simulator_messages.append(
                #         {"role": "supervisor", "content": follow_up}
                #     )
                #
                #     st.session_state.simulator_memory.chat_memory.add_ai_message(reaction)
                #     st.session_state.simulator_memory.chat_memory.add_ai_message(follow_up)
                #
                #     save_simulation_history_local(
                #         st.session_state.customer_query_text_area,
                #         customer_type_display,
                #         st.session_state.simulator_messages,
                #         is_chat_ended=False,
                #     )
                #     st.stop()

                # 3) 그 외 일반적 반응
                st.session_state.simulator_messages.append(
                    {"role": "customer_rebuttal", "content": reaction}
                )
                st.session_state.simulator_memory.chat_memory.add_ai_message(reaction)

                save_simulation_history_local(
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.simulator_messages,
                    is_chat_ended=False,
                )

                if is_positive:
                            st.session_state.is_chat_ended = True
                        # st.rerun()

# -------------------- RAG Tab --------------------
elif feature_selection == L["rag_tab"]:
    st.header(L["rag_header"])
    st.markdown(L["rag_desc"])

    if not st.session_state.is_rag_ready or st.session_state.rag_vectorstore is None:
        # 이미 저장된 인덱스가 있으면 로드 시도
        if st.session_state.is_llm_ready:
            vs = load_rag_index(st.session_state.embeddings)
            if vs is not None:
                st.session_state.rag_vectorstore = vs
                st.session_state.is_rag_ready = True
            else:
                st.info(L["warning_rag_not_ready"])
        else:
            st.info(L["warning_rag_not_ready"])

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
            "10 Multiple-Choice Questions": "quiz",
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
                            resp = st.session_state.llm.invoke(system_prompt + "\n\n" + user_msg)
                            raw = resp.content if hasattr(resp, "content") else str(resp)
                            # 단순 출력
                            st.success(f"**{topic}** - {content_display}")
                            st.code(raw, language="json")
                        except Exception as e:
                            st.error(f"Content Generation Error: {e}")
                else:
                    content_prompt = (
                        f"You are a professional AI coach at the {level} level.\n"
                        f"Generate clear and educational content in {target_lang}.\n"
                        f"Content type: {content_display}.\n"
                        f"Topic: {topic}\n"
                    )
                    with st.spinner("콘텐츠 생성 중..."):
                        try:
                            resp = st.session_state.llm.invoke(content_prompt)
                            txt = resp.content if hasattr(resp, "content") else str(resp)
                            st.success(f"**{topic}** - {content_display}")
                            st.markdown(txt)
                        except Exception as e:
                            st.error(f"Content Generation Error: {e}")

# -------------------- LSTM Tab --------------------
elif feature_selection == L["lstm_tab"]:
    st.header(L["lstm_header"])
    st.markdown(L["lstm_desc"])

    if st.button(L["lstm_rerun_button"]):
        st.rerun()

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
