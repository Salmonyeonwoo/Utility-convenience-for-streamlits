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
    st.session_state.setdefault("is_on_hold", False)
    st.session_state.setdefault("hold_start_time", None)
    st.session_state.setdefault("hold_total_seconds", 0)
    st.session_state.setdefault("provider_call_active", False)
    st.session_state.setdefault("call_direction", "inbound")
    
    # ⭐ 추가: 전체 세션 초기화 버튼 (통화 기록 완전 리셋)
    if st.session_state.call_sim_stage == "WAITING_CALL":
        col_reset, _ = st.columns([1, 4])
        with col_reset:
            if st.button(L.get("button_reset_session", "🔄 세션 초기화"), key="reset_call_session", help=L.get("button_reset_session_help", "모든 통화 관련 데이터를 초기화합니다")):
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
                st.session_state.is_on_hold = False
                st.session_state.hold_start_time = None
                st.session_state.hold_total_seconds = 0
                st.session_state.provider_call_active = False
                st.session_state.call_direction = "inbound"
                st.session_state.call_sim_stage = "WAITING_CALL"
                st.success(L.get("session_reset_success", "✅ 세션이 초기화되었습니다."))
        
        # ⭐ 수정: elif로 변경하여 중복 표시 방지
        st.subheader(L.get("call_receive_header", "📞 전화 수신"))
        
        # 전화번호 입력칸과 다음 버튼 (전화번호 입력칸 길이 축소)
        col_phone, col_next = st.columns([2, 1])
        with col_phone:
            caller_phone = st.text_input(
                L.get("caller_phone_label", "발신자 전화번호"),
                placeholder=L.get("call_number_placeholder", "+82 10-xxxx-xxxx (가상 번호)"),
                key="call_waiting_phone_input",
            )
        with col_next:
            st.write("")  # 공간 확보
            st.write("")  # 공간 확보
            if st.button(L.get("button_next", "다음"), use_container_width=True, type="primary"):
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
                    
                    st.session_state.incoming_phone_number = caller_phone
                    st.session_state.call_sim_stage = "RINGING"
                    st.success(L.get("phone_number_saved", "전화번호가 저장되었습니다: {number}").format(number=caller_phone))
                else:
                    st.warning(L.get("warning_enter_phone", "전화번호를 입력해주세요."))
    # RINGING 상태일 때 문의 입력 섹션 표시
    elif st.session_state.call_sim_stage == "RINGING":
        # sim_perspective 초기화 (없으면 기본값 AGENT)
        if "sim_perspective" not in st.session_state:
            st.session_state.sim_perspective = "AGENT"
        
        st.markdown("---")
        st.subheader(L.get("call_inquiry_header", "📝 고객 문의 입력"))
        inquiry_text = st.text_area(
            L.get("call_inquiry_label", "고객 문의 내용을 입력하세요"),
            value=st.session_state.get("inquiry_text", ""),
            key="inquiry_text_input",
            height=100,
            placeholder=L.get("call_inquiry_placeholder", "예: 환불 요청, 배송 문의 등..."),
        )
        # ⭐ 추가: 웹 주소 (선택) 필드 별도 추가
        website_url = st.text_input(
            L.get("website_url_label", "🌐 웹 주소 (선택)"),
            value=st.session_state.get("call_website_url", ""),
            key="call_website_url_input",
            placeholder=L.get("website_url_placeholder", "https://example.com"),
        )
        
        # ⭐ 추가: 고객 아바타 설정 (성별 및 감정 상태)
        st.markdown("---")
        st.subheader(L.get("customer_avatar_header", "👤 고객 아바타 설정"))
        col_gender, col_emotion, col_spacer1 = st.columns([0.5, 0.5, 2])
        with col_gender:
            # 고객 성별 선택
            gender_options = [
                (L.get("gender_male", "남성"), "male"),
                (L.get("gender_female", "여성"), "female"),
            ]
            current_gender = st.session_state.customer_avatar.get("gender", "male") if "customer_avatar" in st.session_state else "male"
            selected_gender_display = st.selectbox(
                L.get("customer_gender_label", "성별"),
                [label for label, _ in gender_options],
                index=0 if current_gender == "male" else 1,
                key="call_customer_gender",
                label_visibility="visible",
            )
            selected_gender = "male" if selected_gender_display == gender_options[0][0] else "female"
        with col_emotion:
            # 고객 감정 상태 선택
            emotion_options = {
                "NEUTRAL": L.get("emotion_neutral", "평상시"),
                "HAPPY": L.get("emotion_happy", "기쁜 고객"),
                "ANGRY": L.get("emotion_angry", "화난 고객"),
                "ASKING": L.get("emotion_dissatisfied", "진상 고객"),
                "SAD": L.get("emotion_sad", "슬픈 고객")
            }
            current_emotion = st.session_state.customer_avatar.get("state", "NEUTRAL") if "customer_avatar" in st.session_state else "NEUTRAL"
            emotion_display_options = list(emotion_options.values())
            current_emotion_display = emotion_options.get(current_emotion, "평상시")
            current_emotion_idx = emotion_display_options.index(current_emotion_display) if current_emotion_display in emotion_display_options else 0
            selected_emotion_display = st.selectbox(
                L.get("customer_emotion_label", "감정 상태"),
                emotion_display_options,
                index=current_emotion_idx,
                key="call_customer_emotion",
                label_visibility="visible",
            )
            selected_emotion = [k for k, v in emotion_options.items() if v == selected_emotion_display][0]
        
        # customer_avatar 업데이트
        if "customer_avatar" not in st.session_state:
            st.session_state.customer_avatar = {}
        st.session_state.customer_avatar["gender"] = selected_gender
        st.session_state.customer_avatar["state"] = selected_emotion
        
        # ⭐ 수정: 통화 수신/발신 버튼을 감정 선택 옆에 배치하고, 통화 시작 버튼 제거
        col_receive, col_make, col_cancel = st.columns([1, 1, 1])
        with col_receive:
            if st.button(L.get("call_receive_button", "통화 수신"), use_container_width=True, type="primary"):
                if inquiry_text.strip():
                    caller_phone = st.session_state.get("incoming_phone_number", "")
                    if caller_phone:
                        st.session_state.inquiry_text = inquiry_text.strip()
                        # ⭐ 추가: 웹 주소 저장
                        if website_url.strip():
                            st.session_state.call_website_url = website_url.strip()
                        else:
                            st.session_state.call_website_url = ""
                        
                        st.session_state.incoming_call = {"caller_phone": caller_phone}
                        st.session_state.call_active = True
                        st.session_state.current_call_id = str(uuid.uuid4())
                        st.session_state.call_direction = "inbound"
                        st.session_state.start_time = datetime.now()
                        st.session_state.is_on_hold = False
                        st.session_state.hold_start_time = None
                        st.session_state.hold_total_seconds = 0
                        st.session_state.provider_call_active = False
                        st.session_state.call_sim_stage = "IN_CALL"
                        st.success(L.get("inbound_call_started", "전화 수신: {number}").format(number=caller_phone))
                    else:
                        st.warning(L.get("warning_enter_phone", "전화번호를 입력해주세요."))
                else:
                    st.warning(L.get("warning_enter_inquiry", "문의 내용을 입력해주세요."))
        with col_make:
            if st.button(L.get("call_make_button", "통화 발신"), use_container_width=True, type="secondary"):
                if inquiry_text.strip():
                    caller_phone = st.session_state.get("incoming_phone_number", "")
                    if caller_phone:
                        st.session_state.inquiry_text = inquiry_text.strip()
                        # ⭐ 추가: 웹 주소 저장
                        if website_url.strip():
                            st.session_state.call_website_url = website_url.strip()
                        else:
                            st.session_state.call_website_url = ""
                        
                        st.session_state.incoming_call = {"caller_phone": caller_phone, "direction": "outbound"}
                        st.session_state.call_active = True
                        st.session_state.current_call_id = str(uuid.uuid4())
                        st.session_state.call_direction = "outbound"
                        st.session_state.start_time = datetime.now()
                        st.session_state.is_on_hold = False
                        st.session_state.hold_start_time = None
                        st.session_state.hold_total_seconds = 0
                        st.session_state.provider_call_active = False
                        st.session_state.call_sim_stage = "IN_CALL"
                        st.success(L.get("outbound_call_started", "발신을 시작했습니다: {number}").format(number=caller_phone))
                    else:
                        st.warning(L.get("warning_enter_phone", "전화번호를 입력해주세요."))
                else:
                    st.warning(L.get("warning_enter_inquiry", "문의 내용을 입력해주세요."))
        with col_cancel:
            if st.button(L.get("button_cancel", "❌ 취소"), use_container_width=True):
                # ⭐ 수정: 취소 시 모든 통화 관련 상태 초기화
                st.session_state.call_sim_stage = "WAITING_CALL"
                st.session_state.incoming_call = None
                st.session_state.call_active = False
                st.session_state.start_time = None
                st.session_state.call_messages = []
                st.session_state.inquiry_text = ""
                st.session_state.incoming_phone_number = None
                st.session_state.is_on_hold = False
                st.session_state.hold_start_time = None
                st.session_state.hold_total_seconds = 0
                st.session_state.provider_call_active = False
                st.session_state.call_direction = "inbound"
