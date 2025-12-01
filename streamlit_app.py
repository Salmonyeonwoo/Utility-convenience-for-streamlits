"""
모듈화된 Streamlit 앱 - 메인 파일
모든 기능을 utils 모듈에서 import하여 사용합니다.
"""
import streamlit as st
import os
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========================================
# Streamlit 페이지 설정 (반드시 최상단에 위치)
# ========================================
st.set_page_config(
    page_title="AI Study Coach & Customer Service Simulator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# 모듈 Import
# ========================================
from utils.config import BASE_DIR, DATA_DIR, DEFAULT_LANG
from utils.i18n import LANG
from utils.session_state import init_session_state
from utils.llm_clients import init_llm_clients_lazy, SUPPORTED_APIS, get_api_key
from utils.data_helpers import load_simulation_histories_local
from utils.rag_helpers import build_rag_index, load_rag_index, rag_answer
from utils.prompt_generator import generate_customer_reaction, generate_agent_response_draft
from utils.tts_whisper import synthesize_tts, transcribe_bytes_with_whisper

# ========================================
# Session State 초기화
# ========================================
init_session_state()

# ========================================
# 다국어 설정
# ========================================
L = LANG[st.session_state.language]

# ========================================
# LLM 초기화 (지연 로딩)
# ========================================
init_llm_clients_lazy()

# ========================================
# 사이드바 설정
# ========================================
with st.sidebar:
    st.header(L["sidebar_title"])
    
    # 언어 선택
    lang_options = {"한국어": "ko", "English": "en", "日本語": "ja"}
    selected_lang_display = st.selectbox(
        L["lang_select"],
        options=list(lang_options.keys()),
        index=list(lang_options.values()).index(st.session_state.language)
    )
    if lang_options[selected_lang_display] != st.session_state.language:
        st.session_state.language = lang_options[selected_lang_display]
        st.rerun()
    
    L = LANG[st.session_state.language]
    
    st.markdown("---")
    
    # LLM 선택
    st.subheader("LLM 모델 선택")
    llm_options = {
        "OpenAI GPT-4": "openai_gpt4",
        "OpenAI GPT-3.5": "openai_gpt35",
        "Gemini Pro": "gemini_pro",
        "Gemini Flash": "gemini_flash",
        "Claude": "claude",
        "Groq": "groq"
    }
    selected_llm_display = st.selectbox(
        "모델 선택",
        options=list(llm_options.keys()),
        index=list(llm_options.values()).index(st.session_state.get("selected_llm", "openai_gpt4"))
    )
    st.session_state.selected_llm = llm_options[selected_llm_display]
    
    st.markdown("---")
    
    # API 키 상태
    st.subheader("API 키 상태")
    for api_name in SUPPORTED_APIS.keys():
        key = get_api_key(api_name)
        status = "✅ 설정됨" if key else "❌ 미설정"
        st.write(f"- **{api_name}**: {status}")
    
    st.markdown("---")
    
    # 기능 선택
    if "feature_selection" not in st.session_state:
        st.session_state.feature_selection = L["sim_tab_chat_email"]
    
    feature_options = [
        L["sim_tab_chat_email"],
        L["sim_tab_phone"],
        L["rag_tab"],
        L["content_tab"],
    ]
    
    feature_selection = st.radio(
        "기능 선택",
        options=feature_options,
        index=feature_options.index(st.session_state.get("feature_selection", L["sim_tab_chat_email"]))
    )
    st.session_state.feature_selection = feature_selection

# ========================================
# 메인 콘텐츠
# ========================================
st.title(L["title"])

st.info("🎯 **프로젝트 목표**: CS 센터 직원 교육용 AI 고객 응대 시뮬레이터 - 궁극적으로 CS 업무 시스템 대체재")

# ========================================
# 기능별 페이지
# ========================================

# 1. AI 고객 응대 시뮬레이터 (채팅/이메일)
if feature_selection == L["sim_tab_chat_email"]:
    st.header(L["simulator_header"])
    st.caption(L["simulator_desc"])
    
    # 고객 문의 입력
    customer_query = st.text_area(
        L["customer_query_label"],
        value=st.session_state.get("customer_query_text_area", ""),
        height=100,
        key="customer_query_input"
    )
    st.session_state.customer_query_text_area = customer_query
    
    # 고객 유형 선택
    customer_type = st.selectbox(
        L["customer_type_label"],
        options=L["customer_type_options"],
        index=0,
        key="customer_type_select"
    )
    st.session_state.customer_type_sim_select = customer_type
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(L["button_simulate"], type="primary"):
            if not customer_query:
                st.warning(L["simulation_warning_query"])
            elif not st.session_state.is_llm_ready:
                st.warning(L["simulation_no_key_warning"])
            else:
                with st.spinner(L["response_generating"]):
                    # 에이전트 응답 초안 생성
                    draft = generate_agent_response_draft(st.session_state.language)
                    if draft:
                        st.session_state.agent_response_area_text = draft
                        st.session_state.initial_advice_provided = True
    
    with col2:
        if st.button(L["customer_generate_response_button"]):
            if not st.session_state.is_llm_ready:
                st.warning(L["simulation_no_key_warning"])
            else:
                with st.spinner(L["generating_customer_response"]):
                    reaction = generate_customer_reaction(st.session_state.language)
                    if reaction:
                        st.session_state.simulator_messages.append({
                            "role": "customer_rebuttal",
                            "content": reaction
                        })
                        st.success("고객 반응 생성 완료!")
    
    # 응답 표시
    if st.session_state.agent_response_area_text:
        st.markdown("---")
        st.subheader(L["simulation_draft_header"])
        st.write(st.session_state.agent_response_area_text)
        
        # TTS 버튼
        if st.button(L["button_listen_audio"]):
            audio_bytes, status = synthesize_tts(
                st.session_state.agent_response_area_text,
                st.session_state.language,
                role="agent"
            )
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
                st.success(status)
            else:
                st.error(status)
    
    # 대화 이력 표시
    if st.session_state.simulator_messages:
        st.markdown("---")
        st.subheader("대화 이력")
        for msg in st.session_state.simulator_messages:
            role_icon = "👤" if "customer" in msg.get("role", "") else "🤖"
            st.write(f"{role_icon} **{msg.get('role', 'unknown')}**: {msg.get('content', '')}")

# 2. RAG 지식 챗봇
elif feature_selection == L["rag_tab"]:
    st.header(L["rag_header"])
    st.caption(L["rag_desc"])
    
    # 파일 업로드
    uploaded_files = st.file_uploader(
        L["file_uploader"],
        type=["pdf", "txt", "html"],
        accept_multiple_files=True,
        key="rag_file_uploader"
    )
    
    if uploaded_files:
        if st.button(L["button_start_analysis"]):
            with st.spinner(L["data_analysis_progress"]):
                vectorstore, chunk_count = build_rag_index(uploaded_files)
                if vectorstore:
                    st.session_state.rag_vectorstore = vectorstore
                    st.session_state.is_rag_ready = True
                    st.success(L["embed_success"].format(count=chunk_count))
                else:
                    st.error(L["embed_fail"])
    
    # RAG 인덱스 로드 시도
    if not st.session_state.is_rag_ready:
        vectorstore = load_rag_index()
        if vectorstore:
            st.session_state.rag_vectorstore = vectorstore
            st.session_state.is_rag_ready = True
            st.info(L["firestore_loading"])
    
    # 질문 입력
    if st.session_state.is_rag_ready:
        question = st.text_input(
            L["rag_input_placeholder"],
            key="rag_question_input"
        )
        
        if question and st.button("질문하기"):
            with st.spinner(L["response_generating"]):
                answer = rag_answer(question, st.session_state.rag_vectorstore, st.session_state.language)
                st.write(answer)
    else:
        st.warning(L["warning_rag_not_ready"])

# 3. 맞춤형 학습 콘텐츠 생성
elif feature_selection == L["content_tab"]:
    st.header(L["content_header"])
    st.caption(L["content_desc"])
    
    topic = st.text_input(L["topic_label"])
    level = st.selectbox(L["level_label"], options=L["level_options"])
    content_type = st.selectbox(L["content_type_label"], options=L["content_options"])
    
    if st.button(L["button_generate"]):
        if not topic:
            st.warning(L["warning_topic"])
        elif not st.session_state.is_llm_ready:
            st.warning(L["simulation_no_key_warning"])
        else:
            with st.spinner(L["response_generating"]):
                from utils.llm_clients import run_llm
                prompt = f"""
                주제: {topic}
                난이도: {level}
                콘텐츠 형식: {content_type}
                
                위 조건에 맞는 학습 콘텐츠를 생성해주세요.
                """
                content = run_llm(prompt)
                st.write(content)

# 4. 전화 시뮬레이터
elif feature_selection == L["sim_tab_phone"]:
    st.header(L["phone_header"])
    st.caption(L["simulator_desc"])
    
    from datetime import datetime, timedelta
    import uuid
    from utils.prompt_generator import generate_agent_first_greeting, generate_customer_reaction_for_call, summarize_history_for_call
    from utils.tts_whisper import synthesize_tts
    
    # 전화 상태 확인
    if st.session_state.call_sim_stage == "WAITING_CALL":
        st.subheader(L["call_status_waiting"])
        
        # 초기 문의 입력
        call_query = st.text_area(
            L["call_query_placeholder"],
            value=st.session_state.get("call_initial_query", ""),
            height=100,
            key="call_initial_query_input"
        )
        st.session_state.call_initial_query = call_query
        
        # 전화번호 입력
        phone_number = st.text_input(
            "전화번호",
            value=st.session_state.get("incoming_phone_number", "+82 10-1234-5678"),
            placeholder=L["call_number_placeholder"],
            key="phone_number_input"
        )
        st.session_state.incoming_phone_number = phone_number
        
        # 고객 유형 선택
        customer_type = st.selectbox(
            L["customer_type_label"],
            options=L["customer_type_options"],
            index=0,
            key="phone_customer_type"
        )
        st.session_state.customer_type_sim_select = customer_type
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(L["button_answer"], type="primary"):
                if not call_query.strip():
                    st.warning(L["simulation_warning_query"])
                elif not st.session_state.is_llm_ready:
                    st.warning(L["simulation_no_key_warning"])
                else:
                    # 통화 시작
                    st.session_state.call_sim_stage = "IN_CALL"
                    st.session_state.start_time = datetime.now()
                    st.session_state.simulator_messages = []
                    st.session_state.just_entered_call = True
                    st.session_state.customer_turn_start = False
                    st.session_state.is_on_hold = False
                    st.session_state.total_hold_duration = timedelta(0)
                    st.session_state.sim_instance_id = str(uuid.uuid4())
                    
                    # 고객 첫 문의 TTS 생성
                    with st.spinner(L["tts_status_generating"]):
                        audio_bytes, msg = synthesize_tts(call_query, st.session_state.language, role="customer")
                        if audio_bytes:
                            st.session_state.customer_initial_audio_bytes = audio_bytes
                    
                    st.rerun()
        
        with col2:
            if st.button(L["button_call_outbound"], type="secondary"):
                st.info("전화 발신 기능은 원본 파일을 참고하여 구현해야 합니다.")
    
    # 통화 중 상태
    elif st.session_state.call_sim_stage == "IN_CALL":
        # AHT 타이머
        if st.session_state.start_time:
            elapsed = datetime.now() - st.session_state.start_time
            total_seconds = int(elapsed.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            if total_seconds > 900:
                delta_str = L["timer_info_risk"]
                delta_color = "inverse"
            elif total_seconds > 600:
                delta_str = L["timer_info_warn"]
                delta_color = "off"
            else:
                delta_str = L["timer_info_ok"]
                delta_color = "normal"
            
            st.metric(L["timer_metric"], time_str, delta=delta_str, delta_color=delta_color)
        
        # 에이전트 인사말 생성 (처음 한 번만)
        if st.session_state.get("just_entered_call", False):
            greeting = generate_agent_first_greeting(
                st.session_state.language,
                st.session_state.call_initial_query
            )
            if greeting:
                st.session_state.simulator_messages.append({
                    "role": "agent",
                    "content": greeting
                })
                
                # TTS 재생
                audio_bytes, msg = synthesize_tts(greeting, st.session_state.language, role="agent")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                
                st.session_state.just_entered_call = False
                st.session_state.customer_turn_start = True
                st.rerun()
        
        # 고객 문의 재생 (처음 한 번만)
        elif st.session_state.get("customer_turn_start", False):
            if st.session_state.get("customer_initial_audio_bytes"):
                st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3")
                st.session_state.simulator_messages.append({
                    "role": "customer",
                    "content": st.session_state.call_initial_query
                })
            st.session_state.customer_turn_start = False
        
        st.markdown("---")
        
        # CC 자막 표시
        st.subheader(L["cc_live_transcript"])
        if st.session_state.simulator_messages:
            for msg in st.session_state.simulator_messages:
                role_icon = "👤" if "customer" in msg.get("role", "") else "🤖"
                st.write(f"{role_icon} **{msg.get('role', 'unknown')}**: {msg.get('content', '')}")
        
        st.markdown("---")
        
        # 에이전트 응답 입력
        st.subheader(L["mic_input_status"])
        agent_response = st.text_area(
            L["agent_response_prompt"],
            value=st.session_state.get("agent_response_input_box_widget_call", ""),
            height=100,
            key="agent_response_call_input"
        )
        st.session_state.agent_response_input_box_widget_call = agent_response
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(L["agent_response_stop_and_send"]):
                if agent_response:
                    st.session_state.simulator_messages.append({
                        "role": "agent_response",
                        "content": agent_response
                    })
                    
                    # TTS 재생
                    audio_bytes, msg = synthesize_tts(agent_response, st.session_state.language, role="agent")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    
                    # 고객 반응 생성
                    with st.spinner(L["generating_customer_response"]):
                        customer_reaction = generate_customer_reaction_for_call(
                            st.session_state.language,
                            agent_response
                        )
                        if customer_reaction:
                            st.session_state.simulator_messages.append({
                                "role": "customer_rebuttal",
                                "content": customer_reaction
                            })
                            st.success("고객 반응 생성 완료!")
                    
                    st.session_state.agent_response_input_box_widget_call = ""
                    st.rerun()
        
        with col2:
            if st.button(L["button_hangup"]):
                # 통화 요약 생성
                with st.spinner("통화 요약 생성 중..."):
                    summary = summarize_history_for_call(
                        st.session_state.simulator_messages,
                        st.session_state.call_initial_query,
                        st.session_state.language
                    )
                    st.session_state.call_summary_text = summary
                
                st.session_state.call_sim_stage = "CALL_ENDED"
                st.success(L["call_end_message"])
                st.rerun()
        
        with col3:
            if st.button(L["button_hold"] if not st.session_state.get("is_on_hold", False) else L["button_resume"]):
                st.session_state.is_on_hold = not st.session_state.get("is_on_hold", False)
                if st.session_state.is_on_hold:
                    st.session_state.hold_start_time = datetime.now()
                st.rerun()
        
        if st.session_state.get("is_on_hold", False):
            st.warning(L["hold_status"].format(duration="00:00"))
    
    # 통화 종료 상태
    elif st.session_state.call_sim_stage == "CALL_ENDED":
        st.success(L["call_end_message"])
        
        if st.session_state.get("call_summary_text"):
            st.subheader(L["call_summary_header"])
            st.write(st.session_state.call_summary_text)
        
        if st.button("새 통화 시작"):
            st.session_state.call_sim_stage = "WAITING_CALL"
            st.session_state.call_summary_text = ""
            st.rerun()

# 기본 메시지
else:
    st.info("기능을 선택해주세요.")

# ========================================
# 하단 정보
# ========================================
st.markdown("---")
st.caption("💡 이 앱은 모듈화된 구조로 재구성되었습니다. 각 기능은 utils 모듈에서 관리됩니다.")