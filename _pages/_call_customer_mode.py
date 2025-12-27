# -*- coding: utf-8 -*-
"""
전화 시뮬레이터 - 고객 모드 (사용자=고객)
고객 입장에서 AI 상담원과 통화하는 모드
"""
import streamlit as st
from lang_pack import LANG
from datetime import datetime
import uuid

def render_call_customer_mode():
    """고객 모드 전화 시뮬레이터 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # sim_perspective 초기화
    if "sim_perspective" not in st.session_state:
        st.session_state.sim_perspective = "CUSTOMER"
    
    # 고객 모드가 아니면 에이전트 모드로 전환
    if st.session_state.sim_perspective != "CUSTOMER":
        st.session_state.sim_perspective = "CUSTOMER"
    
    # 전화 시뮬레이터 세션 상태 초기화
    if "call_sim_stage" not in st.session_state:
        st.session_state.call_sim_stage = "WAITING_CALL"
    if "call_messages" not in st.session_state:
        st.session_state.call_messages = []
    if "incoming_phone_number" not in st.session_state:
        st.session_state.incoming_phone_number = ""
    if "current_call_id" not in st.session_state:
        st.session_state.current_call_id = None
    
    # WAITING_CALL 상태 - 전화 번호 입력 및 통화 시작
    if st.session_state.call_sim_stage == "WAITING_CALL":
        st.subheader(L.get("call_make_header", "📞 전화 발신"))
        
        # ⭐ 새 고객 등록 폼 (사용자=고객 모드에서만 표시)
        st.markdown("---")
        st.subheader(L.get("new_customer_registration", "새 고객 등록"))
        with st.form("customer_registration_form_customer_mode"):
            col_reg1, col_reg2 = st.columns(2)
            with col_reg1:
                reg_customer_name = st.text_input(L.get("customer_name_required", "고객명 *"), key="reg_customer_name_customer_mode")
                reg_phone = st.text_input(L.get("contact_required", "연락처 *"), key="reg_phone_customer_mode")
                reg_email = st.text_input(L.get("email_required", "이메일 *"), key="reg_email_customer_mode")
            with col_reg2:
                reg_personality = st.selectbox(
                    L.get("customer_personality", "고객 성향"), 
                    ["일반", "신중형", "활발형", "가족형", "프리미엄형", "절약형", "자유형"], 
                    key="reg_personality_customer_mode"
                )
                reg_destination = st.text_input(L.get("preferred_destination", "선호 여행지"), key="reg_destination_customer_mode")
            
            col_reg_btn1, col_reg_btn2 = st.columns([1, 1])
            with col_reg_btn1:
                if st.form_submit_button(L.get("customer_registration", "고객 등록"), type="primary", use_container_width=True):
                    if reg_customer_name and reg_phone and reg_email:
                        try:
                            from customer_data_manager import CustomerDataManager
                            manager = CustomerDataManager()
                            customer_data = {
                                'customer_name': reg_customer_name, 
                                'phone': reg_phone, 
                                'email': reg_email,
                                'personality': reg_personality, 
                                'preferred_destination': reg_destination
                            }
                            customer_id = manager.create_customer(customer_data)
                            st.session_state.customer_name = reg_customer_name
                            st.session_state.customer_phone = reg_phone
                            st.session_state.customer_email = reg_email
                            st.success(L.get("customer_registered_success", "고객이 등록되었습니다! 고객 ID: {customer_id}").format(customer_id=customer_id))
                        except Exception as e:
                            st.error(f"고객 등록 오류: {str(e)}")
                    else:
                        st.error(L.get("customer_registration_required_fields", "고객명, 연락처, 이메일은 필수 항목입니다."))
            with col_reg_btn2:
                st.form_submit_button(L.get("button_cancel", "취소"), use_container_width=True)
        
        st.markdown("---")
        
        # 전화번호 입력칸과 다음 버튼
        col_phone, col_next = st.columns([2, 1])
        with col_phone:
            caller_phone = st.text_input(
                L.get("call_center_phone_label", "콜센터 전화번호"),
                placeholder=L.get("call_center_phone_placeholder", "+82 10-xxxx-xxxx (콜센터 번호)"),
                key="call_waiting_phone_input_customer",
            )
        with col_next:
            st.write("")  # 공간 확보
            st.write("")  # 공간 확보
            if st.button(L.get("button_next", "다음"), use_container_width=True, type="primary"):
                if caller_phone:
                    st.session_state.incoming_phone_number = caller_phone
                    st.session_state.call_sim_stage = "RINGING"
                    st.success(L.get("phone_number_saved", "전화번호가 저장되었습니다: {number}").format(number=caller_phone))
                else:
                    st.warning(L.get("warning_enter_phone", "전화번호를 입력해주세요."))
    
    # RINGING 상태 - 문의 입력 및 통화 시작
    elif st.session_state.call_sim_stage == "RINGING":
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
                        st.session_state.call_direction = "outbound"  # 고객 모드에서는 발신
                        st.session_state.start_time = datetime.now()
                        st.session_state.call_sim_stage = "IN_CALL"
                        
                        # ⭐ 고객 모드: 발신 신호 중 녹음 안내 → AI 상담원 첫 인사말 자동 생성
                        try:
                            from simulation_handler import generate_agent_first_greeting
                            from utils.audio_handler import synthesize_tts
                            
                            # 1. 발신 신호 중 녹음 안내 멘트 (다국어 지원)
                            recording_notice = L.get("call_recording_notice", "고객님과의 통화 내역이 녹음됩니다.")
                            
                            # 2. AI 상담원 첫 인사말 생성 (함수 시그니처에 맞게 수정)
                            agent_greeting = generate_agent_first_greeting(
                                lang_key=current_lang,
                                initial_query=inquiry_text
                            )
                            
                            # 3. 메시지에 추가 (녹음 안내 → AI 인사말 순서)
                            st.session_state.call_messages = [{
                                "role": "system",
                                "content": recording_notice,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }, {
                                "role": "agent",
                                "content": agent_greeting,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }]
                            
                            # 4. AI 인사말 TTS 생성 및 메시지에 오디오 저장
                            try:
                                tts_audio = synthesize_tts(
                                    text=agent_greeting,
                                    voice="alloy",  # AI 상담원은 남성 목소리
                                    lang=current_lang
                                )
                                if tts_audio:
                                    st.session_state.agent_greeting_audio = tts_audio
                                    # 첫 인사말 메시지에 오디오 추가
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
    
    # IN_CALL 상태 - 통화 중
    elif st.session_state.call_sim_stage == "IN_CALL":
        # 대화 흐름 상태 초기화
        if "conversation_flow_state" not in st.session_state:
            st.session_state.conversation_flow_state = "GREETING_DONE"  # GREETING_DONE, WAITING_INFO, WAITING_VERIFICATION, PROVIDING_SOLUTION
        if "is_waiting_verification" not in st.session_state:
            st.session_state.is_waiting_verification = False
        if "verification_wait_start_time" not in st.session_state:
            st.session_state.verification_wait_start_time = None
        # 통화 정보 표시
        call_number = st.session_state.get("incoming_phone_number", "")
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
                st.markdown(f"### {L.get('call_heading_outbound', '📞 전화 통화 중: {number}').format(number=call_number)}")
            with col_info2:
                st.metric(L.get("call_duration_label", "통화 시간"), duration_str)
        
        st.info(L.get("call_in_progress", "📞 통화 중입니다..."))
        
        # 대기 중인 경우 처리 (자동으로 진행)
        if st.session_state.get("is_waiting_verification", False):
            wait_start = st.session_state.get("verification_wait_start_time")
            if wait_start:
                wait_seconds = (datetime.now() - wait_start).total_seconds()
                if wait_seconds >= 5:  # 5초 대기 (실제로는 5분이지만 테스트를 위해 5초)
                    # 대기 완료 - 자동으로 다음 응답 생성
                    wait_complete_key = "wait_verification_complete"
                    if wait_complete_key not in st.session_state:
                        st.session_state[wait_complete_key] = True
                        st.session_state.is_waiting_verification = False
                        st.session_state.conversation_flow_state = "PROVIDING_SOLUTION"
                        
                        # 대기 후 자동 응답 생성
                        try:
                            from simulation_handler import generate_agent_response_draft
                            from utils.audio_handler import synthesize_tts
                            from utils.conversation_flow_handler import generate_after_waiting_message
                            
                            # 대기 후 메시지와 함께 솔루션 생성
                            after_waiting_msg = generate_after_waiting_message(current_lang)
                            agent_response = generate_agent_response_draft(current_lang)
                            full_response = f"{after_waiting_msg} {agent_response}"
                            
                            # TTS 생성
                            agent_audio = None
                            try:
                                tts_audio = synthesize_tts(
                                    text=full_response,
                                    voice="alloy",
                                    lang=current_lang
                                )
                                if tts_audio:
                                    agent_audio = tts_audio
                            except Exception as e:
                                print(f"TTS 생성 오류: {e}")
                            
                            # 메시지에 추가
                            st.session_state.call_messages.append({
                                "role": "agent",
                                "content": full_response,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "audio": agent_audio
                            })
                            
                            # 오디오 자동 재생
                            if agent_audio:
                                st.audio(agent_audio, format="audio/mp3", autoplay=True)
                        except Exception as e:
                            st.error(f"응답 생성 오류: {str(e)}")
                else:
                    remaining = int(5 - wait_seconds)
                    st.warning(f"⏳ {L.get('waiting_verification', '확인 중입니다...')} ({remaining}초 남음)")
        
        # 통화 제어 영역 (통화 종료 버튼 포함)
        col_control1, col_control2, col_control3, col_control4 = st.columns([1, 1, 1, 1])
        with col_control4:
            if st.button(L.get("call_end_button", "📴 통화 종료"), use_container_width=True, type="primary"):
                st.session_state.call_sim_stage = "CALL_ENDED"
                if st.session_state.get("start_time"):
                    call_duration = (datetime.now() - st.session_state.start_time).total_seconds()
                    st.session_state.call_duration = call_duration
        
        st.markdown("---")
        
        # 녹음 안내 및 AI 상담원 첫 인사말 표시
        if st.session_state.call_messages:
            for msg in st.session_state.call_messages:
                if msg.get("role") == "system":
                    st.warning(f"**{L.get('system_notice', '시스템 안내')}:** {msg.get('content', '')}")
                elif msg.get("role") == "agent":
                    st.info(f"**AI 상담원:** {msg.get('content', '')}")
                    # AI 인사말 오디오 재생 (있는 경우)
                    if msg.get("audio") or st.session_state.get("agent_greeting_audio"):
                        audio_to_play = msg.get("audio") or st.session_state.get("agent_greeting_audio")
                        if audio_to_play:
                            st.audio(audio_to_play, format="audio/mp3", autoplay=False)
                elif msg.get("role") == "customer":
                    st.success(f"**고객:** {msg.get('content', '')}")
                    # 고객 오디오 재생 (있는 경우)
                    if msg.get("audio"):
                        st.audio(msg.get("audio"), format="audio/wav", autoplay=False)
        
        st.markdown("---")
        
        # ⭐ 실제 전화 통화처럼: 고객 음성 녹음 → 자동 전사 → AI 응답 → 자동 TTS 재생
        st.markdown(f"**🎤 {L.get('customer_voice_recording', '고객 음성 녹음')}**")
        
        # 대기 중이 아닐 때만 음성 입력 받기
        if not st.session_state.get("is_waiting_verification", False):
            customer_audio_input = st.audio_input(
                L.get("customer_audio_input_label", "고객 음성 녹음"),
                key="call_customer_audio_input",
                help=L.get("audio_input_help", "음성을 녹음하면 자동으로 전사되고 AI 상담원이 응답합니다")
            )
        else:
            customer_audio_input = None
            st.info(L.get("please_wait_verification", "확인 중이니 잠시만 기다려 주세요."))
        
        # 고객 음성 전사 결과 처리 (자동으로 진행)
        if customer_audio_input:
            st.audio(customer_audio_input, format="audio/wav", autoplay=False)
            
            # 중복 처리 방지
            audio_key = f"processed_customer_{hash(customer_audio_input.getvalue())}"
            if audio_key not in st.session_state:
                st.session_state[audio_key] = True
                
                try:
                    from utils.audio_handler import transcribe_bytes_with_whisper
                    
                    if not transcribe_bytes_with_whisper:
                        st.warning(L.get("transcription_unavailable", "⚠️ 전사 기능을 사용할 수 없습니다."))
                    else:
                        st.info(L.get("transcribing_audio", "💬 음성이 녹음되었습니다. 전사 처리 중..."))
                        
                        try:
                            # 전사 처리
                            transcript = transcribe_bytes_with_whisper(
                                customer_audio_input.getvalue(),
                                "audio/wav",
                                current_lang
                            )
                            
                            if transcript:
                                st.success(f"💬 {L.get('customer_transcription_result', '고객 전사')}: {transcript}")
                                
                                # 고객 메시지로 추가
                                st.session_state.call_messages.append({
                                    "role": "customer",
                                    "content": transcript,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "audio": customer_audio_input.getvalue()
                                })
                                
                                # AI 상담원 응답 생성 (자연스러운 대화 흐름)
                                try:
                                    from simulation_handler import generate_agent_response_draft
                                    from utils.conversation_flow_handler import (
                                        detect_customer_emotion, generate_empathetic_response,
                                        needs_additional_info, needs_verification,
                                        generate_waiting_message, generate_after_waiting_message
                                    )
                                    
                                    # ⭐ 수정: simulator_messages에 메시지 저장 (함수가 이걸 읽음)
                                    if "simulator_messages" not in st.session_state:
                                        st.session_state.simulator_messages = []
                                    
                                    # call_messages를 simulator_messages 형식으로 변환하여 저장 (중복 방지)
                                    existing_contents = {msg.get("content", "") for msg in st.session_state.simulator_messages}
                                    
                                    for msg in st.session_state.call_messages:
                                        content = msg.get("content", "")
                                        if content and content not in existing_contents:
                                            role = msg.get("role", "")
                                            if role == "customer":
                                                st.session_state.simulator_messages.append({
                                                    "role": "customer",
                                                    "content": content
                                                })
                                                existing_contents.add(content)
                                            elif role == "agent":
                                                st.session_state.simulator_messages.append({
                                                    "role": "agent",
                                                    "content": content
                                                })
                                                existing_contents.add(content)
                                    
                                    # 최신 고객 메시지 추가 (중복 방지)
                                    if transcript and transcript not in existing_contents:
                                        st.session_state.simulator_messages.append({
                                            "role": "customer",
                                            "content": transcript
                                        })
                                    
                                    # 초기 문의 저장 (함수가 읽음)
                                    if "customer_query_text_area" not in st.session_state or not st.session_state.customer_query_text_area:
                                        st.session_state.customer_query_text_area = st.session_state.get("inquiry_text", "")
                                    
                                    # 고객 감정 감지 및 공감 표현 추가
                                    customer_emotion = detect_customer_emotion(transcript, current_lang)
                                    empathetic_response = generate_empathetic_response(customer_emotion, current_lang)
                                    
                                    # AI 응답 생성 (함수 시그니처에 맞게 수정)
                                    agent_response = generate_agent_response_draft(current_lang)
                                    
                                    # 공감 표현이 있으면 응답 앞에 추가
                                    if empathetic_response and customer_emotion != "NEUTRAL":
                                        agent_response = f"{empathetic_response} {agent_response}"
                                    
                                    # AI 응답 TTS 생성 (먼저 생성하여 오디오 저장)
                                    agent_audio = None
                                    try:
                                        from utils.audio_handler import synthesize_tts
                                        tts_audio = synthesize_tts(
                                            text=agent_response,
                                            voice="alloy",
                                            lang=current_lang
                                        )
                                        if tts_audio:
                                            agent_audio = tts_audio
                                    except Exception as e:
                                        print(f"TTS 생성 오류: {e}")
                                    
                                    # 대화 흐름에 따른 처리 (자연스럽게)
                                    # 1. 확인이 필요한 경우 - 자동으로 대기 상태로 전환
                                    if needs_verification(agent_response, current_lang):
                                        st.session_state.conversation_flow_state = "WAITING_VERIFICATION"
                                        st.session_state.is_waiting_verification = True
                                        st.session_state.verification_wait_start_time = datetime.now()
                                        
                                        # 대기 메시지 추가
                                        waiting_msg = generate_waiting_message(current_lang)
                                        agent_response = f"{agent_response}\n\n{waiting_msg}"
                                    
                                    # AI 응답을 메시지에 추가 (오디오 포함)
                                    st.session_state.call_messages.append({
                                        "role": "agent",
                                        "content": agent_response,
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "audio": agent_audio  # TTS 오디오 저장
                                    })
                                    
                                    # AI 응답 오디오 자동 재생 (실제 전화처럼)
                                    if agent_audio:
                                        st.audio(agent_audio, format="audio/mp3", autoplay=True)
                                    
                                except Exception as e:
                                    st.error(f"{L.get('ai_response_error', 'AI 응답 생성 오류')}: {str(e)}")
                        except Exception as e:
                            st.error(f"{L.get('transcription_error', '전사 오류')}: {str(e)}")
                except Exception as e:
                    st.error(f"{L.get('audio_processing_error', '오디오 처리 오류')}: {str(e)}")
        
        st.markdown("---")
        
        # 텍스트 입력도 지원 (오디오와 함께 사용 가능) - 대기 중이 아닐 때만
        if not st.session_state.get("is_waiting_verification", False):
            user_input = st.text_input(
                L.get("customer_inquiry_input_placeholder", "문의 사항을 입력하세요 (고객 입장)..."),
                key="customer_input_call",
                placeholder=L.get("customer_inquiry_input_placeholder", "문의 사항을 입력하세요...")
            )
        else:
            user_input = None
        
        # 텍스트 입력 처리 (오디오와 동일한 흐름)
        if user_input:
            # 사용자 메시지 추가
            st.session_state.call_messages.append({
                "role": "customer",
                "content": user_input,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # AI 상담원 응답 생성 (자연스러운 대화 흐름)
            try:
                from simulation_handler import generate_agent_response_draft
                from utils.conversation_flow_handler import (
                    detect_customer_emotion, generate_empathetic_response,
                    needs_additional_info, needs_verification,
                    generate_waiting_message, generate_after_waiting_message
                )
                
                # ⭐ 수정: simulator_messages에 메시지 저장 (함수가 이걸 읽음)
                if "simulator_messages" not in st.session_state:
                    st.session_state.simulator_messages = []
                
                # call_messages를 simulator_messages 형식으로 변환하여 저장 (중복 방지)
                existing_contents = {msg.get("content", "") for msg in st.session_state.simulator_messages}
                
                for msg in st.session_state.call_messages:
                    content = msg.get("content", "")
                    if content and content not in existing_contents:
                        role = msg.get("role", "")
                        if role == "customer":
                            st.session_state.simulator_messages.append({
                                "role": "customer",
                                "content": content
                            })
                            existing_contents.add(content)
                        elif role == "agent":
                            st.session_state.simulator_messages.append({
                                "role": "agent",
                                "content": content
                            })
                            existing_contents.add(content)
                
                # 최신 고객 메시지 추가 (중복 방지)
                if user_input and user_input not in existing_contents:
                    st.session_state.simulator_messages.append({
                        "role": "customer",
                        "content": user_input
                    })
                
                # 초기 문의 저장 (함수가 읽음)
                if "customer_query_text_area" not in st.session_state or not st.session_state.customer_query_text_area:
                    st.session_state.customer_query_text_area = st.session_state.get("inquiry_text", "")
                
                # 고객 감정 감지 및 공감 표현 추가
                customer_emotion = detect_customer_emotion(user_input, current_lang)
                empathetic_response = generate_empathetic_response(customer_emotion, current_lang)
                
                # AI 응답 생성 (함수 시그니처에 맞게 수정)
                agent_response = generate_agent_response_draft(current_lang)
                
                # 공감 표현이 있으면 응답 앞에 추가
                if empathetic_response and customer_emotion != "NEUTRAL":
                    agent_response = f"{empathetic_response} {agent_response}"
                
                # AI 응답 TTS 생성 (먼저 생성하여 오디오 저장)
                agent_audio = None
                try:
                    from utils.audio_handler import synthesize_tts
                    tts_audio = synthesize_tts(
                        text=agent_response,
                        voice="alloy",
                        lang=current_lang
                    )
                    if tts_audio:
                        agent_audio = tts_audio
                except Exception as e:
                    print(f"TTS 생성 오류: {e}")
                
                # 대화 흐름에 따른 처리 (자연스럽게)
                # 확인이 필요한 경우 - 자동으로 대기 상태로 전환
                if needs_verification(agent_response, current_lang):
                    st.session_state.conversation_flow_state = "WAITING_VERIFICATION"
                    st.session_state.is_waiting_verification = True
                    st.session_state.verification_wait_start_time = datetime.now()
                    
                    # 대기 메시지 추가
                    waiting_msg = generate_waiting_message(current_lang)
                    agent_response = f"{agent_response}\n\n{waiting_msg}"
                
                # AI 응답을 메시지에 추가 (오디오 포함)
                st.session_state.call_messages.append({
                    "role": "agent",
                    "content": agent_response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "audio": agent_audio  # TTS 오디오 저장
                })
                
                # AI 응답 오디오 자동 재생 (실제 전화처럼)
                if agent_audio:
                    st.audio(agent_audio, format="audio/mp3", autoplay=True)
            except Exception as e:
                st.error(f"{L.get('ai_response_error', 'AI 응답 생성 오류')}: {str(e)}")
        
    
    # CALL_ENDED 상태
    elif st.session_state.call_sim_stage == "CALL_ENDED":
        call_duration = st.session_state.get("call_duration", 0)
        minutes = int(call_duration // 60)
        seconds = int(call_duration % 60)
        if minutes > 0:
            duration_msg = L.get("call_ended_with_duration", "통화가 종료되었습니다. (통화 시간: {minutes}분 {seconds}초)").format(minutes=minutes, seconds=seconds)
        else:
            duration_msg = L.get("call_ended_with_seconds", "통화가 종료되었습니다. (통화 시간: {seconds}초)").format(seconds=seconds)
        st.success(duration_msg)
        
        if st.button(L.get("new_call_button", "새 통화 시작"), key="btn_new_call_customer"):
            st.session_state.call_sim_stage = "WAITING_CALL"
            st.session_state.call_messages = []
            st.session_state.inquiry_text = ""
            st.session_state.incoming_phone_number = None
            st.session_state.call_active = False
            st.session_state.start_time = None
            st.session_state.call_duration = None

