# -*- coding: utf-8 -*-
"""
전화 시뮬레이터 - 고객 모드: WAITING_CALL 상태
아웃바운드 발신 콜 화면
"""
import streamlit as st
from datetime import datetime
from lang_pack import LANG

def render_customer_waiting():
    """WAITING_CALL 상태 렌더링 - 아웃바운드 발신 콜"""
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
    
    # 헤더
    st.markdown(f"### 📞 {L.get('outbound_call_header', '아웃바운드 발신 콜')}")
    st.caption(L.get("outbound_call_description", "고객에게 전화를 걸어 빠르게 연결합니다."))
    
    # 두 개의 컬럼으로 레이아웃 구성 (왼쪽: 고객 정보 입력, 오른쪽: 발신 상태)
    col_out1, col_out2 = st.columns([2, 1])
    
    with col_out1:
        st.subheader(L.get("customer_info_input_header", "고객 정보 입력"))
        
        # form 제출 플래그 초기화
        if 'outbound_form_submitted' not in st.session_state:
            st.session_state.outbound_form_submitted = False
        
        with st.form("outbound_call_form", clear_on_submit=False):
            customer_name = st.text_input(L.get("customer_name_label", "고객 이름"), placeholder="예: 홍길동", value=st.session_state.get('outbound_customer_name', ''))
            customer_phone = st.text_input(L.get("phone_label", "전화번호"), placeholder="예: 010-1234-5678", value=st.session_state.get('outbound_customer_phone', ''))
            call_reason_options = [
                L.get("call_reason_order_confirmation", "주문 확인"),
                L.get("call_reason_delivery_info", "배송 안내"),
                L.get("call_reason_refund", "환불 처리"),
                L.get("call_reason_product_recommendation", "상품 추천"),
                L.get("call_reason_event_info", "이벤트 안내"),
                L.get("call_reason_satisfaction_survey", "고객 만족도 조사"),
                L.get("call_reason_other", "기타")
            ]
            call_reason = st.selectbox(L.get("call_reason_label", "통화 사유"), call_reason_options, index=st.session_state.get('outbound_call_reason_idx', 0))
            
            agent_skill_options = [
                L.get("agent_skill_auto_assign", "자동 할당"),
                L.get("agent_skill_order_payment", "주문/결제 전문가"),
                L.get("agent_skill_refund_cancel", "환불/취소 전문가"),
                L.get("agent_skill_tech_support", "기술 지원 전문가"),
                L.get("agent_skill_general_inquiry", "일반 문의 전문가"),
                L.get("agent_skill_vip", "VIP 고객 전문가")
            ]
            agent_skill = st.selectbox(L.get("required_agent_skill_label", "필요한 에이전트 스킬"), agent_skill_options, index=st.session_state.get('outbound_agent_skill_idx', 0))
            
            # 에이전트 성별 선택 추가
            agent_gender_options = [
                L.get("gender_male_option", "남성"),
                L.get("gender_female_option", "여성")
            ]
            agent_gender = st.selectbox(L.get("agent_gender_label", "에이전트 성별"), agent_gender_options, index=st.session_state.get('outbound_agent_gender_idx', 0))
            
            col_btn_out1, col_btn_out2 = st.columns(2)
            with col_btn_out1:
                call_button = st.form_submit_button(f"📞 {L.get('make_call_button', '전화 걸기')}", type="primary", use_container_width=True)
            with col_btn_out2:
                cancel_button = st.form_submit_button(L.get("cancel", "취소"), use_container_width=True)
        
        # 전화 걸기 처리
        if call_button:
            st.session_state.outbound_form_submitted = True
            st.session_state.outbound_customer_name = customer_name
            st.session_state.outbound_customer_phone = customer_phone
            st.session_state.outbound_call_reason_idx = call_reason_options.index(call_reason)
            st.session_state.outbound_agent_skill_idx = agent_skill_options.index(agent_skill)
            st.session_state.outbound_agent_gender_idx = agent_gender_options.index(agent_gender)
            
            # 에이전트 성별을 session_state에 저장
            st.session_state.selected_agent_gender = agent_gender
            # 번역된 텍스트를 원래 값으로 변환
            male_text = L.get("gender_male_option", "남성")
            st.session_state.agent_gender = "male" if agent_gender == male_text else "female"
            
            if not customer_phone or customer_phone.strip() == "":
                st.error(f"⚠️ {L.get('phone_number_required', '전화번호를 입력해주세요.')}")
                st.session_state.outbound_form_submitted = False
            else:
                # 에이전트 찾기
                try:
                    from agents import find_agent_by_skill
                    selected_agent = find_agent_by_skill(agent_skill, st.session_state.available_agents)
                except ImportError:
                    # agents 모듈이 없으면 직접 찾기
                    auto_assign_text = L.get("agent_skill_auto_assign", "자동 할당")
                    if agent_skill == auto_assign_text:
                        available = [a for a in st.session_state.available_agents if a['status'] == 'available']
                    else:
                        # 번역된 텍스트에서 "전문가" 또는 "Specialist" 등을 제거
                        skill_keyword = agent_skill.replace(L.get("agent_skill_order_payment", "주문/결제 전문가").split("/")[0] if "/" in agent_skill else "", "")
                        skill_keyword = skill_keyword.replace(" 전문가", "").replace(" Specialist", "").replace("専門家", "")
                        available = [a for a in st.session_state.available_agents 
                                    if a['status'] == 'available' and skill_keyword in a['skill']]
                    if available:
                        selected_agent = max(available, key=lambda x: x['rating'])
                    else:
                        selected_agent = None
                
                if selected_agent:
                    # 에이전트 찾기 성공 - 연결 처리
                    call_id = f"call_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    st.session_state.current_call = {
                        'id': call_id,
                        'customer_name': customer_name or "고객",
                        'customer_phone': customer_phone,
                        'reason': call_reason,
                        'agent': selected_agent['name'],
                        'agent_skill': selected_agent['skill'],
                        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'status': 'connected'
                    }
                    
                    st.session_state.call_history.append({
                        'id': call_id,
                        'type': 'outbound',
                        'customer_name': customer_name or "고객",
                        'customer_phone': customer_phone,
                        'reason': call_reason,
                        'agent': selected_agent['name'],
                        'start_time': st.session_state.current_call['start_time'],
                        'status': 'connected'
                    })
                    
                    st.session_state.conversation_history = []
                    st.session_state.needs_more_info = False
                    st.session_state.info_requested = []
                    st.session_state.selected_agent_for_customer = selected_agent
                    st.session_state.incoming_phone_number = customer_phone
                    st.session_state.call_active = True
                    st.session_state.start_time = datetime.now()
                    st.session_state.current_call_id = call_id
                    st.session_state.call_direction = "outbound"
                    
                    # 에이전트 찾기 상태 초기화
                    st.session_state.agent_search_in_progress = False
                    st.session_state.agent_search_attempts = 0
                    
                    # 통화 시작
                    st.session_state.call_sim_stage = "IN_CALL"
                    
                    # 첫 인사말 생성
                    try:
                        from utils.prompt_generator import generate_agent_first_greeting
                        from utils.audio_handler import synthesize_tts
                        
                        greeting = generate_agent_first_greeting(
                            lang_key=st.session_state.get("language", "ko"),
                            initial_query=call_reason,
                            agent_name=selected_agent['name']
                        )
                        
                        # 메시지에 추가
                        st.session_state.call_messages = [{
                            "role": "agent",
                            "content": greeting,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }]
                        
                        # TTS 생성
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
                        
                        st.success(f"✅ {selected_agent['name']} 에이전트에게 연결되었습니다! ({selected_agent['skill']})")
                        st.info(f"📞 {customer_phone}로 전화를 걸고 있습니다...")
                    except Exception as e:
                        # 기본 인사말
                        greeting = f"안녕하세요, {customer_name or '고객'}님. {selected_agent['name']}입니다. {call_reason} 관련하여 연락드렸습니다. 감사합니다."
                        st.session_state.call_messages = [{
                            "role": "agent",
                            "content": greeting,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }]
                        st.success(f"✅ {selected_agent['name']} 에이전트에게 연결되었습니다! ({selected_agent['skill']})")
                        st.info(f"📞 {customer_phone}로 전화를 걸고 있습니다...")
                    
                    st.session_state.outbound_form_submitted = False
                    st.rerun()  # IN_CALL 상태로 전환하기 위해 rerun
                else:
                    # 에이전트를 찾지 못함 - 재시도 로직 시작
                    if not st.session_state.agent_search_in_progress:
                        st.session_state.agent_search_in_progress = True
                        st.session_state.agent_search_attempts = 0
                        st.session_state.agent_search_start_time = datetime.now()
                    
                    # 경과 시간 계산
                    elapsed_time = (datetime.now() - st.session_state.agent_search_start_time).total_seconds()
                    
                    if elapsed_time < st.session_state.agent_search_max_duration:
                        # 재시도 중 로딩 화면 표시
                        st.session_state.agent_search_attempts += 1
                        progress = min(elapsed_time / st.session_state.agent_search_max_duration, 1.0)
                        
                        # 로딩 화면 표시
                        st.markdown("---")
                        with st.spinner(f"🔍 {L.get('searching_agents', '사용 가능한 에이전트를 찾는 중...')}"):
                            st.progress(progress, text=f"{L.get('searching_agents', '사용 가능한 에이전트를 찾는 중...')} ({int(elapsed_time)}초 / {st.session_state.agent_search_max_duration}초)")
                        
                        # 에이전트 찾기 시도
                        import time
                        time.sleep(0.5)  # 0.5초 대기
                        
                        # 다시 에이전트 찾기 시도
                        try:
                            from agents import find_agent_by_skill
                            selected_agent_retry = find_agent_by_skill(agent_skill, st.session_state.available_agents)
                        except ImportError:
                            auto_assign_text = L.get("agent_skill_auto_assign", "자동 할당")
                            if agent_skill == auto_assign_text:
                                available = [a for a in st.session_state.available_agents if a['status'] == 'available']
                            else:
                                skill_keyword = agent_skill.replace(L.get("agent_skill_order_payment", "주문/결제 전문가").split("/")[0] if "/" in agent_skill else "", "")
                                skill_keyword = skill_keyword.replace(" 전문가", "").replace(" Specialist", "").replace("専門家", "")
                                available = [a for a in st.session_state.available_agents 
                                            if a['status'] == 'available' and skill_keyword in a['skill']]
                            if available:
                                selected_agent_retry = max(available, key=lambda x: x['rating'])
                            else:
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
                            time.sleep(1)  # 연결 메시지를 보여주기 위해 1초 대기
                            st.rerun()  # IN_CALL 상태로 전환
                        else:
                            # 아직 에이전트를 찾지 못함 - 계속 재시도
                            st.session_state.outbound_form_submitted = True  # 계속 재시도하기 위해 유지
                            time.sleep(0.5)  # 0.5초 대기 후 재시도
                            st.rerun()  # 재시도를 위해 rerun
                    else:
                        # 최대 대기 시간 초과
                        st.session_state.agent_search_in_progress = False
                        st.session_state.agent_search_attempts = 0
                        st.session_state.agent_search_start_time = None
                        st.error(f"❌ {L.get('agent_search_failed', '사용 가능한 에이전트를 찾을 수 없습니다. 잠시 후 다시 시도해주세요.')}")
                        st.session_state.outbound_form_submitted = False
        
        # 취소 버튼 처리
        if cancel_button:
            st.session_state.outbound_form_submitted = False
            st.session_state.outbound_customer_name = ""
            st.session_state.outbound_customer_phone = ""
    
    with col_out2:
        st.subheader(f"📊 {L.get('call_status_header', '발신 상태')}")
        if st.session_state.current_call:
            call = st.session_state.current_call
            st.info(f"**{L.get('calling_label', '통화 중')}:** {call['customer_name']}")
            st.write(f"**{L.get('phone_label', '전화번호')}:** {call['customer_phone']}")
            st.write(f"**{L.get('agent_label', '에이전트')}:** {call['agent']}")
            st.write(f"**{L.get('skill_label', '스킬')}:** {call['agent_skill']}")
            st.write(f"**{L.get('start_time_label', '시작 시간')}:** {call['start_time']}")
            
            if st.button(f"📞 {L.get('call_end_button', '통화 종료')}", type="secondary", use_container_width=True, key="end_call_outbound"):
                call['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                call['status'] = 'ended'
                for record in st.session_state.call_history:
                    if record['id'] == call['id']:
                        record['end_time'] = call['end_time']
                        record['status'] = 'ended'
                        break
                st.session_state.current_call = None
                st.session_state.conversation_history = []
                st.session_state.call_sim_stage = "WAITING_CALL"
                st.session_state.call_messages = []
                st.success(L.get("call_ended_message", "통화가 종료되었습니다."))
        else:
            st.info(L.get("no_active_call", "현재 진행 중인 통화가 없습니다."))

