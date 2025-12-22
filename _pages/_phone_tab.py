# ========================================
# pages/phone_tab.py
# Phone Call Tab 메인 파일 (참고: app.py 구조)
# ========================================

import streamlit as st
from lang_pack import LANG
from _pages._phone_call_ui import render_waiting_call_ui, render_call_ended_ui
from _pages._phone_call_logic import (
    render_aht_timer, render_call_controls, render_hangup_hold_controls,
    start_inbound_call, start_outbound_call
)


def render_phone_tab():
    """전화 통화 탭 렌더링 함수 (참고: app.py의 show_call_tab 구조)"""
    current_lang = st.session_state.language
    L = LANG[current_lang]

    st.header(L["phone_header"])
    st.markdown(L["simulator_desc"])

    # AHT 타이머 렌더링
    render_aht_timer()

    # 상태별 UI 렌더링
    if st.session_state.call_sim_stage in ["WAITING_CALL", "RINGING"]:
        # 전화 수신 대기 화면
        render_waiting_call_ui()
        
        # 통화 시작 버튼
        col_in, col_out = st.columns(2)
        
        with col_in:
            if st.button(L["button_answer"], key=f"answer_call_btn_{st.session_state.sim_instance_id}"):
                start_inbound_call()
        
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
            
            if call_target_selection == L["call_target_customer"]:
                button_text = L["button_call_outbound_to_customer"]
            else:
                button_text = L["button_call_outbound_to_provider"]
            
            if st.button(button_text, key=f"outbound_call_start_btn_{st.session_state.sim_instance_id}", type="secondary", use_container_width=True):
                start_outbound_call(call_target_selection)
    
    elif st.session_state.call_sim_stage == "IN_CALL":
        # 통화 중 UI
        from _pages._phone_call_audio import render_audio_call_ui
        from _pages._phone_call_video import render_video_call_ui
        
        # 통화 제어
        render_call_controls()
        
        # 통화 중인 경우
        if st.session_state.call_active:
            # 비디오 영역
            if st.session_state.video_enabled:
                render_video_call_ui()
            
            # 실시간 힌트
            from simulation_handler import generate_realtime_hint
            hint_cols = st.columns([4, 1])
            with hint_cols[0]:
                st.info(L["hint_placeholder"] + st.session_state.realtime_hint_text)
            with hint_cols[1]:
                if st.button(L["button_request_hint"], key=f"btn_request_hint_call_{st.session_state.sim_instance_id}"):
                    if "bytes_to_process_call_audio" in st.session_state:
                        st.session_state.bytes_to_process_call_audio = None
                    with st.spinner(L["response_generating"]):
                        hint = generate_realtime_hint(current_lang, is_call=True)
                        st.session_state.realtime_hint_text = hint
            
            # 오디오 통화 영역
            render_audio_call_ui()
            
            # 통화 로그
            call_status = st.session_state.call_handler.get_call_status()
            with st.expander(L.get("call_log_expander", "📋 통화 로그"), expanded=False):
                if call_status:
                    st.json({
                        L.get("call_id_label", "통화 ID"): st.session_state.current_call_id,
                        L.get("call_duration_label", "통화 시간"): f"{int(call_status['duration'] // 60):02d}:{int(call_status['duration'] % 60):02d}",
                        L.get("audio_chunks_label", "오디오 청크"): call_status['chunks_count'],
                        L.get("video_enabled_label", "비디오 활성화"): st.session_state.video_enabled
                    })
        
        st.divider()
        
        # Hangup / Hold 버튼
        render_hangup_hold_controls()
        
        # 요약 및 언어 이관
        from _pages._phone_call_transfer import render_summary_and_transfer
        render_summary_and_transfer()
        
        # Hold 상태 표시
        if st.session_state.is_on_hold:
            st.info(L["call_on_hold_message"])
    
    elif st.session_state.call_sim_stage == "CALL_ENDED":
        # 통화 종료 화면
        render_call_ended_ui()
