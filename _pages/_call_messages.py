# ========================================
# _pages/_call_messages.py
# 전화 통화 메시지 히스토리 표시 모듈
# ========================================

import streamlit as st
from datetime import datetime
from lang_pack import LANG


def render_call_messages(current_lang, L):
    """통화 메시지 히스토리 렌더링"""
    if not st.session_state.get("call_messages"):
        return
    
    with st.expander(L.get("call_history_label", "💬 통화 기록"), expanded=True):
        _render_message_controls(current_lang, L)
        
        for msg in st.session_state.call_messages:
            role = msg.get("role", "")
            if role == "supervisor" or role == "system_hold":
                st.info(msg.get("content", ""))
            else:
                role_icon = "👤" if role == "agent" else "👥"
                role_label = L.get("agent_label", "에이전트") if role == "agent" else L.get("customer_label", "고객")
                
                with st.chat_message(role):
                    st.write(f"{role_icon} **{role_label}**: {msg.get('content', '')}")
                    
                    if msg.get("audio"):
                        audio_format = "audio/mp3" if role == "customer" else "audio/wav"
                        st.audio(msg["audio"], format=audio_format, autoplay=False)
                    
                    if msg.get("timestamp"):
                        try:
                            ts = datetime.fromisoformat(msg["timestamp"])
                            st.caption(ts.strftime("%H:%M:%S"))
                        except:
                            pass


def _render_message_controls(current_lang, L):
    """메시지 제어 버튼 렌더링"""
    col_clear, col_load, _ = st.columns([1, 1, 3])
    
    with col_clear:
        if st.button(
            L.get("clear_call_history", "🗑️ 기록 초기화"),
            key="clear_call_history",
            help=L.get("clear_call_history_help", "현재 통화 기록을 초기화합니다")
        ):
            st.session_state.call_messages = []
            st.success(L.get("call_history_cleared", "통화 기록이 초기화되었습니다."))
    
    with col_load:
        if st.button(
            L.get("load_call_history", "📥 데이터 가져오기"),
            key="load_call_history",
            help=L.get("load_call_history_help", "고객/전화번호별 이전 기록 불러오기")
        ):
            _load_previous_history(current_lang, L)


def _load_previous_history(current_lang, L):
    """이전 통화 기록 불러오기"""
    phone_number = st.session_state.get("incoming_phone_number", "")
    if not phone_number:
        st.warning("전화번호를 먼저 입력해주세요.")
        return
    
    try:
        from utils.history_handler import load_simulation_histories_local
        all_histories = load_simulation_histories_local(current_lang)
        
        matching_histories = []
        for history in all_histories:
            if history.get("is_call", False):
                initial_query = history.get("initial_query", "")
                summary = history.get("summary", {})
                main_inquiry = summary.get("main_inquiry", "") if isinstance(summary, dict) else ""
                
                if phone_number in initial_query or phone_number in main_inquiry:
                    matching_histories.append(history)
        
        if matching_histories:
            st.info(f"📋 {len(matching_histories)}개의 이전 기록을 찾았습니다.")
            latest_history = matching_histories[0]
            with st.expander("📋 가장 최근 기록", expanded=True):
                if latest_history.get("summary"):
                    summary = latest_history.get("summary", {})
                    if isinstance(summary, dict):
                        st.markdown(f"**초기 문의**: {latest_history.get('initial_query', 'N/A')}")
                        st.markdown(f"**고객 유형**: {latest_history.get('customer_type', 'N/A')}")
                        st.markdown(f"**주요 문의**: {summary.get('main_inquiry', 'N/A')}")
                        st.markdown(f"**고객 감정 점수**: {summary.get('customer_sentiment_score', 'N/A')}")
        else:
            st.info("📋 이전 기록이 없습니다. 새로운 고객입니다.")
    except Exception as e:
        st.warning(f"기록 불러오기 오류: {e}")
