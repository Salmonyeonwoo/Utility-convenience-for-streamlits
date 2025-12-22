# -*- coding: utf-8 -*-
"""
전화 시뮬레이터 - 전화 수신 대기 및 문의 입력 모듈
"""
import streamlit as st
from lang_pack import LANG
from datetime import datetime
import uuid

def render_call_waiting():
    """전화 수신 대기 및 문의 입력 UI"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # WAITING_CALL 상태일 때만 전화 번호 입력 표시
    if st.session_state.call_sim_stage == "WAITING_CALL":
        st.subheader("📞 전화 수신")
        caller_phone = st.text_input("발신자 전화번호", placeholder="010-1234-5678", key="call_waiting_phone_input")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("📞 전화 수신", use_container_width=True, type="primary"):
                if caller_phone:
                    st.session_state.incoming_call = {"caller_phone": caller_phone}
                    st.session_state.call_active = True
                    st.session_state.current_call_id = str(uuid.uuid4())
                    st.session_state.incoming_phone_number = caller_phone
                    st.session_state.start_time = datetime.now()
                    st.session_state.call_sim_stage = "RINGING"
                    st.success(f"전화 수신: {caller_phone}")
                else:
                    st.warning("전화번호를 입력해주세요.")
        with col2:
            if st.session_state.get("incoming_call"):
                st.caption(f"수신 중: {st.session_state.incoming_call.get('caller_phone', st.session_state.get('incoming_phone_number', 'N/A'))}")
    # RINGING 상태일 때 문의 입력 섹션 표시
    if st.session_state.call_sim_stage == "RINGING":
        st.markdown("---")
        st.subheader("📝 고객 문의 입력")
        inquiry_text = st.text_area("고객 문의 내용을 입력하세요", value=st.session_state.get("inquiry_text", ""), key="inquiry_text_input", height=100, placeholder="예: 환불 요청, 배송 문의 등...")
        col_start, col_cancel = st.columns([1, 1])
        with col_start:
            if st.button("✅ 통화 시작", use_container_width=True, type="primary"):
                if inquiry_text.strip():
                    st.session_state.inquiry_text = inquiry_text.strip()
                    st.session_state.call_sim_stage = "IN_CALL"
                else:
                    st.warning("문의 내용을 입력해주세요.")
        with col_cancel:
            if st.button("❌ 취소", use_container_width=True):
                st.session_state.call_sim_stage = "WAITING_CALL"
                st.session_state.incoming_call = None
                st.session_state.call_active = False
                st.session_state.start_time = None
