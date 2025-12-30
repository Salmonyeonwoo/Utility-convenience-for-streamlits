# -*- coding: utf-8 -*-
"""
전화 시뮬레이터 - 고객 모드: WAITING_CALL 상태
고객이 상담원에게 전화를 거는 화면 (문의 입력 후 바로 AI 인사말 생성)
"""
import streamlit as st
from datetime import datetime
from lang_pack import LANG

def render_customer_waiting():
    """WAITING_CALL 상태 렌더링 - 고객이 상담원에게 전화를 거는 화면"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # 세션 상태 초기화 (available agents 관련)
    if "available_agents" not in st.session_state:
        try:
            from agents import AVAILABLE_AGENTS
            st.session_state.available_agents = AVAILABLE_AGENTS.copy()
        except ImportError:
            # 기본 에이전트 목록
            st.session_state.available_agents = [
                {'name': '김민수', 'skill': '주문/결제 전문가', 'status': 'available', 'rating': 4.8},
                {'name': '이지은', 'skill': '환불/취소 전문가', 'status': 'available', 'rating': 4.9},
                {'name': '박준호', 'skill': '기술 지원 전문가', 'status': 'available', 'rating': 4.7},
                {'name': '최수진', 'skill': '일반 문의 전문가', 'status': 'available', 'rating': 4.6},
                {'name': '정태영', 'skill': 'VIP 고객 전문가', 'status': 'available', 'rating': 5.0},
            ]
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "needs_more_info" not in st.session_state:
        st.session_state.needs_more_info = False
    if "info_requested" not in st.session_state:
        st.session_state.info_requested = []
    if "customer_insight" not in st.session_state:
        st.session_state.customer_insight = None
    if "selected_agent_for_customer" not in st.session_state:
        st.session_state.selected_agent_for_customer = None
    if "current_call" not in st.session_state:
        st.session_state.current_call = None
    if "call_history" not in st.session_state:
        st.session_state.call_history = []
    if "agent_search_in_progress" not in st.session_state:
        st.session_state.agent_search_in_progress = False
    if "agent_search_attempts" not in st.session_state:
        st.session_state.agent_search_attempts = 0
    if "agent_search_start_time" not in st.session_state:
        st.session_state.agent_search_start_time = None
    if "agent_search_max_duration" not in st.session_state:
        st.session_state.agent_search_max_duration = 60  # 최대 60초 (1분)
    
    # 에이전트 검색이 진행 중이면 자동으로 재시도
    if st.session_state.get("agent_search_in_progress", False):
        # 경과 시간 계산
        if st.session_state.agent_search_start_time is None:
            st.session_state.agent_search_start_time = datetime.now()
            elapsed_time = 0
        else:
            elapsed_time = (datetime.now() - st.session_state.agent_search_start_time).total_seconds()
        
        # 최대 대기 시간 초과 확인
        if elapsed_time >= st.session_state.agent_search_max_duration:
            # 최대 대기 시간 초과 - 재시도 중단 (최대 1회만)
            st.session_state.agent_search_in_progress = False
            st.session_state.agent_search_attempts = 0
            st.session_state.agent_search_start_time = None
            st.session_state.outbound_form_submitted = False
            
            # 디버깅 정보 표시
            available_agents_list = st.session_state.get("available_agents", [])
            available_count = len([a for a in available_agents_list if a.get('status') == 'available'])
            agent_skill = st.session_state.get("outbound_agent_skill", "")
            
            st.error(f"❌ {L.get('agent_search_failed', '사용 가능한 에이전트를 찾을 수 없습니다. 잠시 후 다시 시도해주세요.')}")
            st.warning(f"디버깅 정보: 사용 가능한 에이전트 수 = {available_count}, 요청 스킬 = {agent_skill}")
            if available_agents_list:
                st.write("전체 에이전트 목록:")
                for agent in available_agents_list:
                    st.write(f"- {agent.get('name', 'N/A')}: {agent.get('skill', 'N/A')} (상태: {agent.get('status', 'N/A')})")
        else:
            # 재시도 중 - 로딩 화면 표시
            st.session_state.agent_search_attempts += 1
            progress = min(elapsed_time / st.session_state.agent_search_max_duration, 1.0)
            
            # 로딩 화면 표시
            st.markdown("---")
            with st.spinner(f"🔍 {L.get('searching_agents', '사용 가능한 에이전트를 찾는 중...')}"):
                st.progress(progress, text=f"{L.get('searching_agents', '사용 가능한 에이전트를 찾는 중...')} ({int(elapsed_time)}초 / {st.session_state.agent_search_max_duration}초)")
            
            # 저장된 정보로 에이전트 찾기 시도
            agent_skill = st.session_state.get("outbound_agent_skill", "")
            customer_name = st.session_state.get("outbound_customer_name", "")
            customer_phone = st.session_state.get("outbound_customer_phone", "")
            call_reason = st.session_state.get("outbound_call_reason", "")
            
            # 디버깅: available_agents 확인
            available_agents_list = st.session_state.get("available_agents", [])
            available_count = len([a for a in available_agents_list if a.get('status') == 'available'])
            
            # 에이전트 찾기 시도
            import time
            time.sleep(0.5)  # 0.5초 대기
            
            # 다시 에이전트 찾기 시도
            selected_agent_retry = None
            try:
                from agents import find_agent_by_skill
                selected_agent_retry = find_agent_by_skill(agent_skill, st.session_state.available_agents)
            except ImportError:
                # agents 모듈이 없으면 직접 찾기
                auto_assign_text = L.get("agent_skill_auto_assign", "자동 할당")
                if agent_skill == auto_assign_text:
                    available = [a for a in st.session_state.available_agents if a.get('status') == 'available']
                else:
                    # 번역된 텍스트를 원래 한글 skill로 매핑
                    skill_mapping = {
                        L.get("agent_skill_order_payment", "주문/결제 전문가"): "주문/결제",
                        L.get("agent_skill_refund_cancel", "환불/취소 전문가"): "환불/취소",
                        L.get("agent_skill_tech_support", "기술 지원 전문가"): "기술 지원",
                        L.get("agent_skill_general_inquiry", "일반 문의 전문가"): "일반 문의",
                        L.get("agent_skill_vip", "VIP 고객 전문가"): "VIP 고객"
                    }
                    skill_keyword = skill_mapping.get(agent_skill, "")
                    if not skill_keyword:
                        # 매핑이 없으면 텍스트에서 추출 시도
                        skill_keyword = agent_skill.replace(" 전문가", "").replace(" Specialist", "").replace("専門家", "")
                        if "/" in skill_keyword:
                            skill_keyword = skill_keyword.split("/")[0]
                    
                    available = [a for a in st.session_state.available_agents 
                                if a.get('status') == 'available' and skill_keyword in a.get('skill', '')]
                if available:
                    selected_agent_retry = max(available, key=lambda x: x.get('rating', 0))
                else:
                    selected_agent_retry = None
            except Exception as e:
                print(f"에이전트 찾기 오류: {e}")
                selected_agent_retry = None
            
            if selected_agent_retry:
                # 에이전트를 찾았으므로 연결 처리
                call_id = f"call_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state.current_call = {
                    'id': call_id,
                    'customer_name': customer_name or "고객",
                    'customer_phone': customer_phone,
                    'reason': call_reason,
                    'agent': selected_agent_retry['name'],
                    'agent_skill': selected_agent_retry['skill'],
                    'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'connected'
                }
                
                st.session_state.call_history.append({
                    'id': call_id,
                    'type': 'outbound',
                    'customer_name': customer_name or "고객",
                    'customer_phone': customer_phone,
                    'reason': call_reason,
                    'agent': selected_agent_retry['name'],
                    'start_time': st.session_state.current_call['start_time'],
                    'status': 'connected'
                })
                
                st.session_state.conversation_history = []
                st.session_state.needs_more_info = False
                st.session_state.info_requested = []
                st.session_state.selected_agent_for_customer = selected_agent_retry
                st.session_state.incoming_phone_number = customer_phone
                st.session_state.call_active = True
                st.session_state.start_time = datetime.now()
                st.session_state.current_call_id = call_id
                st.session_state.call_direction = "outbound"
                
                # 에이전트 찾기 상태 초기화
                st.session_state.agent_search_in_progress = False
                st.session_state.agent_search_attempts = 0
                st.session_state.agent_search_start_time = None
                
                # 연결 성공 메시지 표시
                st.success(f"✅ {L.get('agent_connected', '에이전트에 연결되었습니다!')} {selected_agent_retry['name']} ({selected_agent_retry['skill']})")
                
                # 통화 시작
                st.session_state.call_sim_stage = "IN_CALL"
                
                # 첫 인사말 생성
                try:
                    from utils.prompt_generator import generate_agent_first_greeting
                    from utils.audio_handler import synthesize_tts
                    
                    greeting = generate_agent_first_greeting(
                        lang_key=st.session_state.get("language", "ko"),
                        initial_query=call_reason,
                        agent_name=selected_agent_retry['name']
                    )
                    
                    st.session_state.call_messages = [{
                        "role": "agent",
                        "content": greeting,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }]
                    
                    try:
                        from utils.audio_handler import synthesize_tts
                        tts_audio, tts_msg = synthesize_tts(
                            text=greeting,
                            lang_key=st.session_state.get("language", "ko"),
                            role="agent"
                        )
                        if tts_audio and st.session_state.call_messages:
                            st.session_state.call_messages[-1]["audio"] = tts_audio
                    except Exception as e:
                        print(f"TTS 생성 오류: {e}")
                except Exception as e:
                    greeting = f"안녕하세요, {customer_name or '고객'}님. {selected_agent_retry['name']}입니다. {call_reason} 관련하여 연락드렸습니다. 감사합니다."
                    st.session_state.call_messages = [{
                        "role": "agent",
                        "content": greeting,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }]
                
                st.session_state.outbound_form_submitted = False
            else:
                # 아직 에이전트를 찾지 못함 - 계속 재시도 (최대 1회만 - 시간 제한 내에서)
                # 재시도 횟수 체크 (최대 1회만 추가 재시도)
                if st.session_state.agent_search_attempts <= 1:
                    pass  # 재시도 로직 제거됨
                else:
                    # 재시도 횟수 초과 - 중단
                    st.session_state.agent_search_in_progress = False
                    st.session_state.agent_search_attempts = 0
                    st.session_state.agent_search_start_time = None
                    st.session_state.outbound_form_submitted = False
                    
                    # 디버깅 정보 표시
                    available_agents_list = st.session_state.get("available_agents", [])
                    available_count = len([a for a in available_agents_list if a.get('status') == 'available'])
                    
                    st.error(f"❌ {L.get('agent_search_failed', '사용 가능한 에이전트를 찾을 수 없습니다. 잠시 후 다시 시도해주세요.')}")
                    st.warning(f"디버깅 정보: 사용 가능한 에이전트 수 = {available_count}, 요청 스킬 = {agent_skill}")
                    if available_agents_list:
                        st.write("전체 에이전트 목록:")
                        for agent in available_agents_list:
                            st.write(f"- {agent.get('name', 'N/A')}: {agent.get('skill', 'N/A')} (상태: {agent.get('status', 'N/A')})")
    
    # 헤더 - 고객 모드: 고객이 상담원에게 전화를 거는 화면
    st.markdown(f"### 📞 {L.get('call_make_header', '전화 발신')}")
    st.caption(L.get("call_make_description", "상담원에게 전화를 걸어 상담을 시작합니다."))
    
    # 문의 입력 및 전화 발신
    st.markdown("---")
    st.subheader(L.get("call_inquiry_header", "📝 고객 문의 입력"))
    
    inquiry_text = st.text_area(
        L.get("call_inquiry_label", "고객 문의 내용을 입력하세요"),
        value=st.session_state.get("inquiry_text", ""),
        key="inquiry_text_input_customer_waiting",
        height=100,
        placeholder=L.get("call_inquiry_placeholder", "예: 환불 요청, 배송 문의 등..."),
    )
    
    # 전화 발신 버튼
    col_start, col_cancel = st.columns([1, 1])
    with col_start:
        call_button = st.button(L.get("call_make_button", "통화 발신"), use_container_width=True, type="primary")
    with col_cancel:
        cancel_button = st.button(L.get("button_cancel", "❌ 취소"), use_container_width=True)
    
    # 전화 발신 처리
    if call_button:
        if inquiry_text.strip():
            # 전화번호 설정 (없으면 기본값)
            caller_phone = st.session_state.get("incoming_phone_number", "")
            if not caller_phone:
                caller_phone = "010-0000-0000"  # 기본 전화번호
                st.session_state.incoming_phone_number = caller_phone
            
            st.session_state.inquiry_text = inquiry_text.strip()
            st.session_state.incoming_call = {"caller_phone": caller_phone}
            st.session_state.call_active = True
            st.session_state.current_call_id = f"call_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state.call_direction = "outbound"
            st.session_state.start_time = datetime.now()
            st.session_state.call_sim_stage = "IN_CALL"
            
            # AI 상담원 첫 인사말 자동 생성
            try:
                from utils.prompt_generator import generate_agent_first_greeting
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
                # 기본 인사말로 폴백
                default_greeting = L.get("agent_first_greeting_ko", "안녕하세요. 고객님 고객 센터에 연락 주셔서 감사드립니다. 저는 상담원이라고 합니다. 무엇을 도와드릴까요?")
                st.session_state.call_messages = [{
                    "role": "agent",
                    "content": default_greeting,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }]
                st.session_state.call_sim_stage = "IN_CALL"
        else:
            st.warning(L.get("warning_enter_inquiry", "문의 내용을 입력해주세요."))
    
    # 취소 버튼 처리
    if cancel_button:
        st.session_state.call_sim_stage = "WAITING_CALL"
        st.session_state.incoming_call = None
        st.session_state.call_active = False
        st.session_state.start_time = None
        st.session_state.call_messages = []
        st.session_state.inquiry_text = ""
        st.session_state.incoming_phone_number = None

