# ========================================
# _pages/_call_ui.py
# 전화 통화 UI 컴포넌트
# ========================================

import streamlit as st
from datetime import datetime
from lang_pack import LANG


def render_call_header(call_number, call_direction, L):
    """통화 헤더 렌더링"""
    if call_number:
        call_duration = 0
        if st.session_state.get("start_time"):
            call_duration = (datetime.now() - st.session_state.start_time).total_seconds()
        else:
            st.session_state.start_time = datetime.now()
            call_duration = 0
        
        minutes = int(call_duration // 60)
        seconds = int(call_duration % 60)
        duration_str = f"{minutes:02d}:{seconds:02d}"
        
        col_info1, col_info2 = st.columns([2, 1])
        with col_info1:
            heading_template = L.get(
                "call_heading_outbound" if call_direction == "outbound" else "call_heading_inbound",
                "📞 전화 통화 중: {number}"
            )
            st.markdown(f"### {heading_template.format(number=call_number)}")
        with col_info2:
            st.metric(L.get("call_duration_label", "통화 시간"), duration_str)


def render_hold_controls(L):
    """Hold/재개 컨트롤 렌더링"""
    hold_elapsed = st.session_state.get("hold_total_seconds", 0)
    if st.session_state.get("is_on_hold") and st.session_state.get("hold_start_time"):
        hold_elapsed += (datetime.now() - st.session_state.hold_start_time).total_seconds()
    hold_minutes = int(hold_elapsed // 60)
    hold_seconds = int(hold_elapsed % 60)
    hold_duration_str = f"{hold_minutes:02d}:{hold_seconds:02d}"
    
    if st.session_state.get("is_on_hold"):
        st.caption(L.get("hold_status", "통화 Hold 중 (누적 Hold 시간: {duration})").format(duration=hold_duration_str))
        if st.button(L.get("button_resume", "▶️ 통화 재개"), use_container_width=True):
            if st.session_state.get("hold_start_time"):
                st.session_state.hold_total_seconds += (datetime.now() - st.session_state.hold_start_time).total_seconds()
            st.session_state.hold_start_time = None
            st.session_state.is_on_hold = False
            st.session_state.provider_call_active = False
            st.success(L.get("call_resumed", "통화를 재개했습니다."))
    else:
        if st.button(L.get("button_hold", "⏸️ Hold (소음 차단)"), use_container_width=True):
            st.session_state.is_on_hold = True
            st.session_state.hold_start_time = datetime.now()
            st.session_state.hold_total_seconds = 0
            st.session_state.provider_call_active = False
            st.session_state.call_messages.append({
                "role": "system_hold",
                "content": L.get("agent_hold_message", "[에이전트: Hold 중입니다. 통화 재개 버튼을 눌러주세요.]"),
                "timestamp": datetime.now().isoformat()
            })


def render_provider_call_button(L):
    """업체 발신 버튼 렌더링"""
    if st.button(
        L.get("button_call_company", "📞 업체에 전화"),
        use_container_width=True,
        disabled=not st.session_state.get("is_on_hold")
    ):
        st.session_state.provider_call_active = True
        st.session_state.is_on_hold = True
        if not st.session_state.get("hold_start_time"):
            st.session_state.hold_start_time = datetime.now()
        st.session_state.call_messages.append({
            "role": "agent",
            "content": L.get("provider_call_message", "업체에 확인해 보겠습니다. 잠시만 기다려 주세요."),
            "timestamp": datetime.now().isoformat()
        })
        st.info(L.get("provider_call_progress", "업체에 확인 중입니다. 잠시만 기다려 주세요."))


def render_hint_button(current_lang, L):
    """응대 힌트 버튼 렌더링"""
    if st.button(
        L.get("button_hint", "💡 응대 힌트"),
        use_container_width=True,
        help=L.get("button_hint_help", "현재 대화 맥락을 기반으로 실시간 응대 힌트를 제공합니다"),
        key="call_hint_button"
    ):
        if st.session_state.is_llm_ready:
            try:
                from simulation_handler import generate_realtime_hint
                session_lang = st.session_state.get("language", current_lang)
                if session_lang not in ["ko", "en", "ja"]:
                    session_lang = current_lang
                
                with st.spinner(L.get("generating_hint", "응대 힌트 생성 중...")):
                    hint = generate_realtime_hint(session_lang, is_call=True)
                    if hint:
                        st.session_state.call_messages.append({
                            "role": "supervisor",
                            "content": f"💡 **{L.get('hint_label', '응대 힌트')}**: {hint}",
                            "timestamp": datetime.now().isoformat()
                        })
                        st.session_state.realtime_hint_text = hint
            except Exception as e:
                st.error(f"응대 힌트 생성 오류: {e}")
        else:
            from llm_client import get_api_key
            has_api_key = any([
                bool(get_api_key("openai")) if get_api_key else False,
                bool(get_api_key("gemini")) if get_api_key else False,
                bool(get_api_key("claude")) if get_api_key else False,
                bool(get_api_key("groq")) if get_api_key else False
            ])
            if not has_api_key:
                st.warning(L.get("simulation_no_key_warning", "LLM이 준비되지 않았습니다."))
            else:
                st.session_state.is_llm_ready = True


def render_end_call_button(L):
    """통화 종료 버튼 렌더링"""
    if st.button(L.get("button_end_call", "📴 종료"), use_container_width=True, type="primary"):
        call_duration = 0
        if st.session_state.get("start_time"):
            call_duration = (datetime.now() - st.session_state.start_time).total_seconds()
            st.session_state.call_duration = call_duration
        
        if st.session_state.get("is_on_hold") and st.session_state.get("hold_start_time"):
            st.session_state.hold_total_seconds += (datetime.now() - st.session_state.hold_start_time).total_seconds()
        
        st.session_state.is_on_hold = False
        st.session_state.hold_start_time = None
        st.session_state.provider_call_active = False
        st.session_state.call_sim_stage = "CALL_ENDED"
        st.session_state.call_active = False
        st.session_state.start_time = None
