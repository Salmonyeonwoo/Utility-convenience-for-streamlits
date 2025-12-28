# -*- coding: utf-8 -*-
"""
전화 시뮬레이터 - 고객 모드: IN_CALL 상태
통화 중 처리 (오디오 및 텍스트 입력)
"""
import streamlit as st
from lang_pack import LANG
from datetime import datetime

def process_audio_input(customer_audio_input, current_lang, L):
    """고객 오디오 입력 처리 및 AI 응답 생성"""
    if not customer_audio_input:
        return
    
    # 오디오 표시
    try:
        st.audio(customer_audio_input, format="audio/wav", autoplay=False)
    except Exception as e:
        st.warning(f"오디오 표시 오류: {str(e)}")
    
    # 중복 처리 방지 - 더 정확한 키 사용
    try:
        audio_bytes = customer_audio_input.getvalue()
        audio_hash = hash(audio_bytes)
        audio_key = f"processed_audio_call_{audio_hash}"
        
        # 이미 처리된 오디오인지 확인 (같은 세션 내에서만)
        if audio_key in st.session_state:
            return
        
        # 처리 시작 표시 (rerun 최소화)
        st.session_state[audio_key] = True
        
        try:
            from utils.audio_handler import transcribe_bytes_with_whisper
            
            if not transcribe_bytes_with_whisper:
                st.warning(L.get("transcription_unavailable", "⚠️ 전사 기능을 사용할 수 없습니다."))
                # 처리 실패 시 키 제거하여 재시도 가능하게
                if audio_key in st.session_state:
                    del st.session_state[audio_key]
                return
            
            # 전사 처리 (즉시 실행, 스피너 없이)
            # 전사는 API 호출이므로 빠르게 처리하되, UI 업데이트는 최소화
            transcript = transcribe_bytes_with_whisper(
                audio_bytes,
                "audio/wav",
                current_lang
            )
            
            if transcript and transcript.strip():
                transcript = transcript.strip()
                
                # 고객 메시지로 즉시 추가 (UI 업데이트 최소화)
                st.session_state.call_messages.append({
                    "role": "customer",
                    "content": transcript,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "audio": audio_bytes
                })
                
                # 즉시 AI 응답 생성 (대화 흐름 자연스럽게 - 지연 없이, 스피너 없이)
                _generate_agent_response(transcript, current_lang, L)
            else:
                st.warning("전사 결과가 비어있습니다.")
                # 처리 실패 시 키 제거하여 재시도 가능하게
                if audio_key in st.session_state:
                    del st.session_state[audio_key]
                    
        except Exception as e:
            st.error(f"{L.get('transcription_error', '전사 오류')}: {str(e)}")
            # 오류 발생 시 키 제거하여 재시도 가능하게
            if audio_key in st.session_state:
                del st.session_state[audio_key]
    except Exception as e:
        st.error(f"{L.get('audio_processing_error', '오디오 처리 오류')}: {str(e)}")

def process_text_input(user_input, current_lang, L):
    """고객 텍스트 입력 처리 및 AI 응답 생성"""
    if not user_input or not user_input.strip():
        return
    
    user_input = user_input.strip()
    
    # 중복 처리 방지 - 최근 처리된 텍스트와 비교 (같은 세션 내에서만)
    last_processed_text_key = "last_processed_text_call"
    last_processed_text = st.session_state.get(last_processed_text_key, "")
    if user_input == last_processed_text:
        return
    
    # 처리 시작 표시
    st.session_state[last_processed_text_key] = user_input
    
    # 사용자 메시지 추가
    st.session_state.call_messages.append({
        "role": "customer",
        "content": user_input,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # 즉시 AI 응답 생성 (대화 흐름 자연스럽게 - 지연 없이)
    _generate_agent_response(user_input, current_lang, L)

def _generate_agent_response(user_text, current_lang, L):
    """AI 에이전트 응답 생성 (공통 함수)"""
    try:
        # conversation.py의 generate_agent_response 사용
        try:
            from conversation import generate_agent_response
            use_conversation_module = True
        except ImportError:
            use_conversation_module = False
        from simulation_handler import generate_agent_response_draft
        from utils.conversation_flow_handler import (
            detect_customer_emotion, generate_empathetic_response,
            needs_verification, generate_waiting_message
        )
        
        # conversation_history 업데이트
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        # 고객 메시지를 conversation_history 형식으로 추가
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation_history.append({
            'role': '고객',
            'text': user_text,
            'time': timestamp
        })
        
        # conversation.py의 generate_agent_response 사용
        if use_conversation_module:
            agent_response = generate_agent_response(
                user_text,
                st.session_state.customer_insight,
                st.session_state.needs_more_info,
                st.session_state.info_requested
            )
            
            # 대화 흐름 상태 관리
            user_text_lower = user_text.lower()
            
            # Closing 로직 (채팅 로직 참고)
            # 솔루션 제공 후 고객이 감사 인사하는 경우에만 추가 문의 확인
            closing_confirm_keywords = L.get("customer_closing_confirm", "다른 문의 사항은 없으십니까?")
            is_closing_question = (
                closing_confirm_keywords in agent_response or
                any(phrase in agent_response for phrase in ["추가 문의", "다른 문의", "다른 도움이 필요"])
            )
            
            # 고객이 솔루션에 만족하는 표현을 했을 때만 추가 문의 확인
            if any(word in user_text_lower for word in ['알겠습니다', '알겠어요', '감사합니다', '고마워요', '고맙습니다', '이해했습니다', '이해했어요', '좋아요', '완벽해요']):
                if st.session_state.conversation_flow_state == "PROVIDING_SOLUTION" and not is_closing_question:
                    # LLM이 이미 추가 문의를 포함하지 않았을 때만 추가
                    if "추가 문의" not in agent_response and "다른 문의" not in agent_response:
                        agent_response += f" {closing_confirm_keywords}"
                    st.session_state.conversation_flow_state = "ASKING_ADDITIONAL"
            
            # 추가 문의 없음 확인 후 끝인사
            no_more_inquiries = L.get("customer_no_more_inquiries", "없습니다. 감사합니다.")
            if any(word in user_text_lower for word in ['없습니다', '없어요', '없음', '없다', '없어']) and any(word in user_text_lower for word in ['감사', '고마워', '고맙']):
                agent_response = "네, 알겠습니다. 감사합니다. 좋은 하루 되세요."
                st.session_state.conversation_flow_state = "ENDING"
            
            # 솔루션 제공 상태로 전환 (구체적인 정보가 제공된 경우)
            if any(word in user_text_lower for word in ['도쿄', '오사카', '역', '구간', 'JR', '패스', 'klook', '여행', '지역']) or len(user_text.strip()) > 30:
                if st.session_state.conversation_flow_state != "ASKING_ADDITIONAL" and st.session_state.conversation_flow_state != "ENDING":
                    st.session_state.conversation_flow_state = "PROVIDING_SOLUTION"
            
            # conversation_history에 에이전트 응답 추가
            st.session_state.conversation_history.append({
                'role': 'AI 에이전트',
                'text': agent_response,
                'time': datetime.now().strftime("%H:%M:%S")
            })
        else:
            # fallback: 기존 로직 사용
            if "simulator_messages" not in st.session_state:
                st.session_state.simulator_messages = []
            
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
            
            if user_text and user_text not in existing_contents:
                st.session_state.simulator_messages.append({
                    "role": "customer",
                    "content": user_text
                })
            
            if "customer_query_text_area" not in st.session_state or not st.session_state.customer_query_text_area:
                st.session_state.customer_query_text_area = st.session_state.get("inquiry_text", "")
            
            customer_emotion = detect_customer_emotion(user_text, current_lang)
            empathetic_response = generate_empathetic_response(customer_emotion, current_lang)
            
            agent_response = generate_agent_response_draft(current_lang)
            
            if empathetic_response and customer_emotion != "NEUTRAL":
                agent_response = f"{empathetic_response} {agent_response}"
            
            if needs_verification(agent_response, current_lang):
                st.session_state.conversation_flow_state = "WAITING_VERIFICATION"
                st.session_state.is_waiting_verification = True
                st.session_state.verification_wait_start_time = datetime.now()
                waiting_msg = generate_waiting_message(current_lang)
                agent_response = f"{agent_response}\n\n{waiting_msg}"
            
            # AI 응답 TTS 생성
            agent_audio = None
            try:
                from utils.audio_handler import synthesize_tts
                tts_audio, tts_msg = synthesize_tts(
                    text=agent_response,
                    lang_key=current_lang,
                    role="agent"
                )
                if tts_audio:
                    agent_audio = tts_audio
            except Exception as e:
                print(f"TTS 생성 오류: {e}")
            
            # AI 응답을 메시지에 추가
            st.session_state.call_messages.append({
                "role": "agent",
                "content": agent_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "audio": agent_audio
            })
            
            if agent_audio:
                try:
                    st.audio(agent_audio, format="audio/mp3", autoplay=True)
                except Exception as e:
                    print(f"오디오 재생 오류: {e}")
            return
        
        # AI 응답 TTS 생성 (conversation.py 사용 시)
        if use_conversation_module:
            agent_audio = None
            try:
                from utils.audio_handler import synthesize_tts
                tts_audio, tts_msg = synthesize_tts(
                    text=agent_response,
                    lang_key=current_lang,
                    role="agent"
                )
                if tts_audio:
                    agent_audio = tts_audio
            except Exception as e:
                print(f"TTS 생성 오류: {e}")
            
            # AI 응답을 메시지에 추가
            st.session_state.call_messages.append({
                "role": "agent",
                "content": agent_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "audio": agent_audio
            })
            
            if agent_audio:
                try:
                    st.audio(agent_audio, format="audio/mp3", autoplay=True)
                except Exception as e:
                    print(f"오디오 재생 오류: {e}")
                
    except Exception as e:
        st.error(f"{L.get('ai_response_error', 'AI 응답 생성 오류')}: {str(e)}")

def render_customer_in_call():
    """IN_CALL 상태 렌더링 - 통화 중"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # 대화 흐름 상태 초기화
    if "conversation_flow_state" not in st.session_state:
        st.session_state.conversation_flow_state = "GREETING_DONE"
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
    
    # 대기 중인 경우 처리
    if st.session_state.get("is_waiting_verification", False):
        wait_start = st.session_state.get("verification_wait_start_time")
        if wait_start:
            wait_seconds = (datetime.now() - wait_start).total_seconds()
            if wait_seconds >= 5:
                wait_complete_key = "wait_verification_complete"
                if wait_complete_key not in st.session_state:
                    st.session_state[wait_complete_key] = True
                    st.session_state.is_waiting_verification = False
                    st.session_state.conversation_flow_state = "PROVIDING_SOLUTION"
                    
                    try:
                        from simulation_handler import generate_agent_response_draft
                        from utils.audio_handler import synthesize_tts
                        from utils.conversation_flow_handler import generate_after_waiting_message
                        
                        after_waiting_msg = generate_after_waiting_message(current_lang)
                        agent_response = generate_agent_response_draft(current_lang)
                        full_response = f"{after_waiting_msg} {agent_response}"
                        
                        agent_audio = None
                        try:
                            from utils.audio_handler import synthesize_tts
                            tts_audio, tts_msg = synthesize_tts(
                                text=full_response,
                                lang_key=current_lang,
                                role="agent"
                            )
                            if tts_audio:
                                agent_audio = tts_audio
                        except Exception as e:
                            print(f"TTS 생성 오류: {e}")
                        
                        st.session_state.call_messages.append({
                            "role": "agent",
                            "content": full_response,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "audio": agent_audio
                        })
                        
                        if agent_audio:
                            st.audio(agent_audio, format="audio/mp3", autoplay=True)
                    except Exception as e:
                        st.error(f"응답 생성 오류: {str(e)}")
            else:
                remaining = int(5 - wait_seconds)
                st.warning(f"⏳ {L.get('waiting_verification', '확인 중입니다...')} ({remaining}초 남음)")
    
    # 통화 제어 영역
    col_control1, col_control2, col_control3, col_control4 = st.columns([1, 1, 1, 1])
    with col_control4:
        if st.button(L.get("call_end_button", "📴 통화 종료"), use_container_width=True, type="primary"):
            st.session_state.call_sim_stage = "CALL_ENDED"
            if st.session_state.get("start_time"):
                call_duration = (datetime.now() - st.session_state.start_time).total_seconds()
                st.session_state.call_duration = call_duration
    
    st.markdown("---")
    
    # 고객 정보 등록 섹션 (대화 기록 대신)
    st.markdown("### 📝 고객 정보 등록")
    st.caption("통화 이력을 저장하기 위해 고객 정보를 입력해주세요.")
    
    if "call_customer_info" not in st.session_state:
        st.session_state.call_customer_info = {
            "name": "",
            "phone": "",
            "email": ""
        }
    
    with st.form("customer_info_form_call", clear_on_submit=False):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            customer_name = st.text_input(
                "고객 이름",
                value=st.session_state.call_customer_info.get("name", ""),
                placeholder="예: 홍길동"
            )
            customer_phone = st.text_input(
                "전화번호",
                value=st.session_state.call_customer_info.get("phone", ""),
                placeholder="예: 010-1234-5678"
            )
        with col_info2:
            customer_email = st.text_input(
                "이메일",
                value=st.session_state.call_customer_info.get("email", ""),
                placeholder="예: customer@example.com"
            )
            customer_memo = st.text_area(
                "메모 (선택사항)",
                value=st.session_state.call_customer_info.get("memo", ""),
                placeholder="추가 메모를 입력하세요",
                height=80
            )
        
        if st.form_submit_button("💾 고객 정보 저장", type="primary", use_container_width=True):
            if customer_name and customer_phone:
                st.session_state.call_customer_info = {
                    "name": customer_name,
                    "phone": customer_phone,
                    "email": customer_email,
                    "memo": customer_memo
                }
                st.success(f"✅ 고객 정보가 저장되었습니다: {customer_name} ({customer_phone})")
                
                # CustomerDataManager에 저장 (선택사항)
                try:
                    if "customer_data_manager" in st.session_state and st.session_state.customer_data_manager:
                        customer_manager = st.session_state.customer_data_manager
                        # 고객 정보를 찾거나 생성
                        customer_id = f"CUST_{customer_phone.replace('-', '').replace(' ', '')}"
                        st.session_state.call_customer_id = customer_id
                        st.info(f"고객 정보가 저장되었습니다. (고객 ID: {customer_id})")
                except Exception as e:
                    st.warning(f"고객 데이터 저장 중 오류 (이력은 저장됩니다): {str(e)}")
            else:
                st.warning("⚠️ 고객 이름과 전화번호는 필수입니다.")
    
    st.markdown("---")
    
    # 고객 음성 녹음
    st.markdown(f"**🎤 {L.get('customer_voice_recording', '고객 음성 녹음')}**")
    
    if not st.session_state.get("is_waiting_verification", False):
        customer_audio_input = st.audio_input(
            L.get("customer_audio_input_label", "고객 음성 녹음"),
            key="call_customer_audio_input",
            help=L.get("audio_input_help", "음성을 녹음하면 자동으로 전사되고 AI 상담원이 응답합니다")
        )
    else:
        customer_audio_input = None
        st.info(L.get("please_wait_verification", "확인 중이니 잠시만 기다려 주세요."))
    
    # 텍스트 입력
    if not st.session_state.get("is_waiting_verification", False):
        user_input = st.text_input(
            L.get("customer_inquiry_input_placeholder", "문의 사항을 입력하세요 (고객 입장)..."),
            key="customer_input_call",
            placeholder=L.get("customer_inquiry_input_placeholder", "문의 사항을 입력하세요...")
        )
    else:
        user_input = None
    
    # 입력 처리 (텍스트 우선, 없으면 오디오)
    if user_input and user_input.strip():
        # 텍스트 입력 처리
        process_text_input(user_input, current_lang, L)
    elif customer_audio_input:
        # 오디오 입력 처리
        process_audio_input(customer_audio_input, current_lang, L)

