# -*- coding: utf-8 -*-
"""
전화 시뮬레이터 - 고객 모드: RINGING 상태
문의 입력 및 통화 시작
"""
import streamlit as st
from lang_pack import LANG
from datetime import datetime
import uuid

def render_customer_ringing():
    """RINGING 상태 렌더링 - 문의 입력 및 통화 시작"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    st.markdown("---")
    st.subheader(L.get("call_inquiry_header", "📝 고객 문의 입력"))
    
    inquiry_text = st.text_area(
        L.get("call_inquiry_label", "고객 문의 내용을 입력하세요"),
        value=st.session_state.get("inquiry_text", ""),
        key="inquiry_text_input_customer",
        height=100,
        placeholder=L.get("call_inquiry_placeholder", "예: 환불 요청, 배송 문의 등..."),
    )
    
    # 웹 주소 (선택) 필드
    website_url = st.text_input(
        L.get("website_url_label", "🌐 웹 주소 (선택)"),
        value=st.session_state.get("call_website_url", ""),
        key="call_website_url_input_customer",
        placeholder=L.get("website_url_placeholder", "https://example.com"),
    )
    
    # 고객 아바타 설정 (성별 및 감정 상태)
    st.markdown("---")
    st.subheader(L.get("customer_avatar_header", "👤 고객 아바타 설정"))
    col_gender, col_emotion = st.columns(2)
    with col_gender:
        gender_options = [
            (L.get("gender_male", "남성"), "male"),
            (L.get("gender_female", "여성"), "female"),
        ]
        current_gender = st.session_state.customer_avatar.get("gender", "male") if "customer_avatar" in st.session_state else "male"
        selected_gender_display = st.selectbox(
            L.get("customer_gender_label", "성별"),
            [label for label, _ in gender_options],
            index=0 if current_gender == "male" else 1,
            key="call_customer_gender_customer_mode",
            label_visibility="visible",
        )
        selected_gender = "male" if selected_gender_display == gender_options[0][0] else "female"
    with col_emotion:
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
            key="call_customer_emotion_customer_mode",
            label_visibility="visible",
        )
        selected_emotion = [k for k, v in emotion_options.items() if v == selected_emotion_display][0]
    
    # customer_avatar 업데이트
    if "customer_avatar" not in st.session_state:
        st.session_state.customer_avatar = {}
    st.session_state.customer_avatar["gender"] = selected_gender
    st.session_state.customer_avatar["state"] = selected_emotion
    
    # 통화 시작 버튼 (고객 모드에서는 통화 발신)
    col_start, col_cancel = st.columns([1, 1])
    with col_start:
        if st.button(L.get("call_make_button", "통화 발신"), use_container_width=True, type="primary"):
            if inquiry_text.strip():
                caller_phone = st.session_state.get("incoming_phone_number", "")
                if caller_phone:
                    st.session_state.inquiry_text = inquiry_text.strip()
                    if website_url.strip():
                        st.session_state.call_website_url = website_url.strip()
                    else:
                        st.session_state.call_website_url = ""
                    
                    st.session_state.incoming_call = {"caller_phone": caller_phone}
                    st.session_state.call_active = True
                    st.session_state.current_call_id = str(uuid.uuid4())
                    st.session_state.call_direction = "outbound"
                    st.session_state.start_time = datetime.now()
                    st.session_state.call_sim_stage = "IN_CALL"
                    
                    # AI 상담원 첫 인사말 자동 생성
                    try:
                        from simulation_handler import generate_agent_first_greeting
                        from utils.audio_handler import synthesize_tts
                        
                        recording_notice = L.get("call_recording_notice", "고객님과의 통화 내역이 녹음됩니다.")
                        agent_greeting = generate_agent_first_greeting(
                            lang_key=current_lang,
                            initial_query=inquiry_text
                        )
                        
                        st.session_state.call_messages = [{
                            "role": "system",
                            "content": recording_notice,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }, {
                            "role": "agent",
                            "content": agent_greeting,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }]
                        
                        # TTS 생성
                        try:
                            from utils.audio_handler import synthesize_tts
                            tts_audio, tts_msg = synthesize_tts(
                                text=agent_greeting,
                                lang_key=current_lang,
                                role="agent"
                            )
                            if tts_audio:
                                st.session_state.agent_greeting_audio = tts_audio
                                if st.session_state.call_messages and st.session_state.call_messages[-1].get("role") == "agent":
                                    st.session_state.call_messages[-1]["audio"] = tts_audio
                        except Exception as e:
                            print(f"TTS 생성 오류: {e}")
                        
                        st.success(L.get("call_started_customer_mode", "통화가 시작되었습니다. AI 상담원이 인사말을 했습니다."))
                    except Exception as e:
                        st.error(f"AI 인사말 생성 오류: {str(e)}")
                        st.session_state.call_sim_stage = "IN_CALL"
                else:
                    st.warning(L.get("warning_enter_phone", "전화번호를 입력해주세요."))
            else:
                st.warning(L.get("warning_enter_inquiry", "문의 내용을 입력해주세요."))
    with col_cancel:
        if st.button(L.get("button_cancel", "❌ 취소"), use_container_width=True):
            st.session_state.call_sim_stage = "WAITING_CALL"
            st.session_state.incoming_call = None
            st.session_state.call_active = False
            st.session_state.start_time = None
            st.session_state.call_messages = []
            st.session_state.inquiry_text = ""
            st.session_state.incoming_phone_number = None

