# ========================================
# _pages/_call_in_call.py
# 전화 시뮬레이터 - 통화 중 모듈 (메인)
# ========================================

import streamlit as st
from lang_pack import LANG
from datetime import datetime
from _pages._call_ui import render_call_header, render_hold_controls, render_provider_call_button, render_hint_button, render_end_call_button
from _pages._call_audio import process_audio_input
from _pages._call_video import render_video_section
from _pages._call_messages import render_call_messages
from _pages._call_transfer import render_transfer_section

def render_call_in_call():
    """통화 중 UI - 오디오 녹음 + 전사 + 고객 반응 자동 생성"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    st.session_state.setdefault("is_on_hold", False)
    st.session_state.setdefault("hold_start_time", None)
    st.session_state.setdefault("hold_total_seconds", 0)
    st.session_state.setdefault("provider_call_active", False)
    st.session_state.setdefault("call_direction", "inbound")
    
    # ⭐ 수정: 통화 시작 시 call_messages 초기화 확인 (새 통화인 경우)
    if 'call_messages' not in st.session_state:
        st.session_state.call_messages = []
    
    # 통화 헤더 표시
    call_number = st.session_state.get("incoming_phone_number")
    call_direction = st.session_state.get("call_direction", "inbound")
    if call_number:
        render_call_header(call_number, call_direction, L)
    
    st.info(L.get("call_in_progress", "📞 통화 중입니다..."))
    
    # 통화 제어 영역 (5열: Hold/재개, 업체 발신, 응대 힌트, 비디오, 종료)
    col_hold, col_provider, col_hint, col_video, col_end = st.columns([1, 1, 1, 1, 1])
    with col_hold:
        render_hold_controls(L)
    with col_provider:
        render_provider_call_button(L)
    with col_hint:
        render_hint_button(current_lang, L)
    with col_video:
        if 'video_enabled' not in st.session_state:
            st.session_state.video_enabled = False
        st.session_state.video_enabled = st.toggle(
            L.get("button_video_enable", "📹 비디오"),
            value=st.session_state.video_enabled,
            help=L.get("video_enable_help", "비디오 통화를 활성화합니다")
        )
    with col_end:
        render_end_call_button(L)
    
    st.markdown("---")
    
    # 비디오 영역
    render_video_section(L)
    
    # 오디오 녹음 및 전사 섹션
    st.markdown("**🎤 오디오 녹음 및 전사**")
    
    audio_col1, audio_col2 = st.columns([3, 1])
    with audio_col1:
        audio_input = st.audio_input(
            L.get("audio_speak_label", "말씀하세요"),
            key="call_audio_input_in_call",
            help=L.get("audio_input_help", "음성을 녹음하면 자동으로 전사됩니다")
        )
    with audio_col2:
        if st.session_state.get("call_messages"):
            st.caption(L.get("messages_count", "메시지: {count}개").format(count=len(st.session_state.call_messages)))
    
    # 오디오 처리
    process_audio_input(audio_input, current_lang, L)
    
    st.markdown("---")
    
    # 이관 요약 표시
    if st.session_state.get("transfer_summary_text") or (
        st.session_state.get("language_at_transfer_start") and 
        st.session_state.language != st.session_state.get("language_at_transfer_start")
    ):
        with st.expander(f"**{L.get('transfer_summary_header', '이관 요약')}**", expanded=False):
            st.info(L.get("transfer_summary_intro", "다음은 이전 팀에서 전달받은 통화 요약입니다."))
            if st.session_state.get("transfer_summary_text") and st.session_state.get("translation_success", True):
                st.markdown(st.session_state.transfer_summary_text)
            elif st.session_state.get("transfer_summary_text"):
                st.info(st.session_state.transfer_summary_text)
    
    st.markdown("---")
    
    # 통화 메시지 히스토리 표시
    render_call_messages(current_lang, L)
    
    st.markdown("---")
    
    # 언어 팀 이관 기능
    render_transfer_section(current_lang, L)
    
    st.markdown("---")
    
    # 통화 내용 메모
    st.markdown(f"**{L.get('call_content_memo', '📝 통화 내용 메모')}**")
    call_content = st.text_area(
        L.get("memo_input_placeholder", "메모 입력 (선택사항)"),
        value=st.session_state.get("call_content", ""),
        key="call_content_input",
        height=100,
        help=L.get("memo_input_help", "추가 메모를 작성할 수 있습니다")
    )
    
    if call_content:
        st.session_state.call_content = call_content
    
    st.markdown("---")
    
    # 저장 버튼
    col_save, _ = st.columns([1, 3])
    with col_save:
        if st.button("💾 저장", use_container_width=True):
            if call_content.strip() or st.session_state.get("call_messages"):
                st.success("통화 내용이 저장되었습니다.")
            else:
                st.warning("통화 내용을 입력하거나 오디오를 녹음해주세요.")
