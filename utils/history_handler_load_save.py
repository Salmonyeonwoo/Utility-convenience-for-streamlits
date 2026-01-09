# ========================================
# utils/history_handler_load_save.py
# 이력 관리 - 로드/저장 함수들
# ========================================

import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

from config import SIM_META_FILE, DATA_DIR
from utils import _load_json, _save_json
from utils.history_handler_summaries import generate_call_summary, generate_chat_summary
from utils.history_handler_guides import generate_daily_customer_guide, save_daily_customer_guide

def load_simulation_histories_local(lang_key: str) -> List[Dict[str, Any]]:
    """로컬에 저장된 시뮬레이션 이력을 로드합니다."""
    histories = _load_json(SIM_META_FILE, [])
    return [
        h for h in histories
        if h.get("language_key") == lang_key and (isinstance(h.get("messages"), list) or h.get("summary"))
    ]

def save_simulation_history_local(initial_query: str, customer_type: str, messages: List[Dict[str, Any]],
                                  is_chat_ended: bool, attachment_context: str, is_call: bool = False,
                                  customer_name: str = "", customer_phone: str = "", customer_email: str = "",
                                  customer_id: str = ""):
    """AI 요약 데이터를 중심으로 이력을 저장 (고객 정보 포함)"""
    histories = _load_json(SIM_META_FILE, [])
    doc_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    summary_data = None
    if is_chat_ended or len(messages) > 4 or is_call:
        # 전화 통화의 경우 요약 생성 (문의 내용 + 솔루션 요점만)
        if is_call:
            summary_data = generate_call_summary(messages, initial_query, customer_type, st.session_state.language)
        else:
            # 채팅의 경우 기존 요약 함수 사용
            summary_data = generate_chat_summary(messages, initial_query, customer_type, st.session_state.language)

    if summary_data:
        data = {
            "id": doc_id,
            "initial_query": initial_query,
            "customer_type": customer_type,
            "messages": [],
            "summary": summary_data,
            "language_key": st.session_state.language,
            "timestamp": ts,
            "is_chat_ended": is_chat_ended,
            "attachment_context": attachment_context if attachment_context else "",
            "is_call": is_call,
        }
    else:
        data = {
            "id": doc_id,
            "initial_query": initial_query,
            "customer_type": customer_type,
            "messages": messages[:10] if len(messages) > 10 else messages,
            "summary": None,
            "language_key": st.session_state.language,
            "timestamp": ts,
            "is_chat_ended": is_chat_ended,
            "attachment_context": attachment_context if attachment_context else "",
            "is_call": is_call,
        }

    histories.insert(0, data)
    
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    today_histories = [h for h in histories if h.get("timestamp", "").startswith(today_str)]
    
    if len(today_histories) > 20:
        today_histories_sorted = sorted(today_histories, key=lambda x: x.get("timestamp", ""))
        excess_count = len(today_histories) - 20
        excess_ids = {h.get("id") for h in today_histories_sorted[:excess_count]}
        histories = [h for h in histories if h.get("id") not in excess_ids]
    
    _save_json(SIM_META_FILE, histories[:100])
    
    if summary_data and is_chat_ended:
        try:
            all_histories = _load_json(SIM_META_FILE, [])
            today_str = datetime.now().strftime("%y%m%d")
            guide_filename = f"{today_str}_고객가이드.TXT"
            guide_filepath = os.path.join(DATA_DIR, guide_filename)
            
            guide_content = generate_daily_customer_guide(all_histories, st.session_state.language)
            
            if guide_content:
                saved_path = save_daily_customer_guide(guide_content, st.session_state.language)
                if saved_path:
                    print(f"✅ 고객 가이드가 자동 생성/업데이트되었습니다: {saved_path}")
        except Exception as e:
            print(f"고객 가이드 자동 생성 중 오류 발생 (무시됨): {e}")
    
    return doc_id

def delete_all_history_local():
    """모든 이력을 삭제합니다."""
    _save_json(SIM_META_FILE, [])

