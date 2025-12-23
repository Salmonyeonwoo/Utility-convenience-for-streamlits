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
    
    # ⭐ 추가: 전체 세션 초기화 버튼 (통화 기록 완전 리셋)
    if st.session_state.call_sim_stage == "WAITING_CALL":
        col_reset, _ = st.columns([1, 4])
        with col_reset:
            if st.button("🔄 세션 초기화", key="reset_call_session", help="모든 통화 관련 데이터를 초기화합니다"):
                # 모든 통화 관련 상태 초기화
                st.session_state.call_messages = []
                st.session_state.inquiry_text = ""
                st.session_state.call_content = ""
                st.session_state.incoming_phone_number = None
                st.session_state.incoming_call = None
                st.session_state.call_active = False
                st.session_state.start_time = None
                st.session_state.call_duration = None
                st.session_state.transfer_summary_text = ""
                st.session_state.language_at_transfer_start = None
                st.session_state.current_call_id = None
                st.session_state.call_sim_stage = "WAITING_CALL"
                st.success("✅ 세션이 초기화되었습니다.")
        
        # ⭐ 수정: elif로 변경하여 중복 표시 방지
        st.subheader("📞 전화 수신")
        caller_phone = st.text_input("발신자 전화번호", placeholder="010-1234-5678", key="call_waiting_phone_input")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("📞 전화 수신", use_container_width=True, type="primary"):
                if caller_phone:
                    # ⭐ 수정: 전화번호가 변경되었는지 확인하고, 변경되었으면 기록 초기화
                    previous_phone = st.session_state.get("incoming_phone_number", "")
                    if previous_phone and previous_phone != caller_phone:
                        # 다른 전화번호이므로 이전 기록 초기화
                        st.session_state.call_messages = []
                        st.session_state.inquiry_text = ""
                        st.session_state.call_content = ""
                        st.session_state.transfer_summary_text = ""
                        st.session_state.language_at_transfer_start = None
                        st.session_state.translation_success = True
                    
                    st.session_state.incoming_call = {"caller_phone": caller_phone}
                    st.session_state.call_active = True
                    st.session_state.current_call_id = str(uuid.uuid4())
                    st.session_state.incoming_phone_number = caller_phone
                    # ⭐ 수정: 통화 시간 카운팅은 통화 수신 시작과 동시에 시작
                    st.session_state.start_time = datetime.now()
                    st.session_state.call_sim_stage = "RINGING"
                    st.success(f"전화 수신: {caller_phone}")
                else:
                    st.warning("전화번호를 입력해주세요.")
        with col2:
            if st.session_state.get("incoming_call"):
                st.caption(f"수신 중: {st.session_state.incoming_call.get('caller_phone', st.session_state.get('incoming_phone_number', 'N/A'))}")
    # RINGING 상태일 때 문의 입력 섹션 표시 (elif로 변경하여 중복 방지)
    elif st.session_state.call_sim_stage == "RINGING":
        st.markdown("---")
        st.subheader("📝 고객 문의 입력")
        inquiry_text = st.text_area("고객 문의 내용을 입력하세요", value=st.session_state.get("inquiry_text", ""), key="inquiry_text_input", height=100, placeholder="예: 환불 요청, 배송 문의 등...")
        # ⭐ 추가: 웹 주소 (선택) 필드 별도 추가
        website_url = st.text_input("🌐 웹 주소 (선택)", value=st.session_state.get("call_website_url", ""), key="call_website_url_input", placeholder="https://example.com")
        
        # ⭐ 추가: 고객 아바타 설정 (성별 및 감정 상태)
        st.markdown("---")
        st.subheader("👤 고객 아바타 설정")
        col_gender, col_emotion = st.columns(2)
        with col_gender:
            # 고객 성별 선택
            gender_options = ["남자", "여자"]
            current_gender = st.session_state.customer_avatar.get("gender", "male") if "customer_avatar" in st.session_state else "male"
            gender_display = "남자" if current_gender == "male" else "여자"
            selected_gender_display = st.selectbox("성별", gender_options, index=0 if gender_display == "남자" else 1, key="call_customer_gender")
            selected_gender = "male" if selected_gender_display == "남자" else "female"
        with col_emotion:
            # 고객 감정 상태 선택
            emotion_options = {
                "NEUTRAL": "평상시",
                "HAPPY": "기쁜 고객",
                "ANGRY": "화난 고객",
                "ASKING": "진상 고객",
                "SAD": "슬픈 고객"
            }
            current_emotion = st.session_state.customer_avatar.get("state", "NEUTRAL") if "customer_avatar" in st.session_state else "NEUTRAL"
            emotion_display_options = list(emotion_options.values())
            current_emotion_display = emotion_options.get(current_emotion, "평상시")
            current_emotion_idx = emotion_display_options.index(current_emotion_display) if current_emotion_display in emotion_display_options else 0
            selected_emotion_display = st.selectbox("감정 상태", emotion_display_options, index=current_emotion_idx, key="call_customer_emotion")
            selected_emotion = [k for k, v in emotion_options.items() if v == selected_emotion_display][0]
        
        # customer_avatar 업데이트
        if "customer_avatar" not in st.session_state:
            st.session_state.customer_avatar = {}
        st.session_state.customer_avatar["gender"] = selected_gender
        st.session_state.customer_avatar["state"] = selected_emotion
        
        col_start, col_cancel = st.columns([1, 1])
        with col_start:
            if st.button("✅ 통화 시작", use_container_width=True, type="primary"):
                if inquiry_text.strip():
                    st.session_state.inquiry_text = inquiry_text.strip()
                    # ⭐ 추가: 웹 주소 저장
                    if website_url.strip():
                        st.session_state.call_website_url = website_url.strip()
                    else:
                        st.session_state.call_website_url = ""
                    st.session_state.call_sim_stage = "IN_CALL"
                else:
                    st.warning("문의 내용을 입력해주세요.")
        with col_cancel:
            if st.button("❌ 취소", use_container_width=True):
                # ⭐ 수정: 취소 시 모든 통화 관련 상태 초기화
                st.session_state.call_sim_stage = "WAITING_CALL"
                st.session_state.incoming_call = None
                st.session_state.call_active = False
                st.session_state.start_time = None
                st.session_state.call_messages = []
                st.session_state.inquiry_text = ""
                st.session_state.incoming_phone_number = None
