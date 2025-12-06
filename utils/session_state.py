"""
Session State 초기화 모듈
"""
import uuid
import streamlit as st
from datetime import timedelta
from langchain.memory import ConversationBufferMemory
from utils.i18n import DEFAULT_LANG, LANG


def init_session_state():
    """Session State를 한 번에 초기화하는 함수 (앱 시작 시 한 번만 실행)"""
    if "session_state_initialized" in st.session_state:
        return  # 이미 초기화됨
    
    # 기본 언어 설정
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANG
    if "is_llm_ready" not in st.session_state:
        st.session_state.is_llm_ready = False
    if "llm_init_error_msg" not in st.session_state:
        st.session_state.llm_init_error_msg = ""
    if "uploaded_files_state" not in st.session_state:
        st.session_state.uploaded_files_state = None
    if "is_rag_ready" not in st.session_state:
        st.session_state.is_rag_ready = False
    if "rag_vectorstore" not in st.session_state:
        st.session_state.rag_vectorstore = None
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "agent_input" not in st.session_state:
        st.session_state.agent_input = ""
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    if "simulator_messages" not in st.session_state:
        st.session_state.simulator_messages = []
    if "simulator_memory" not in st.session_state:
        st.session_state.simulator_memory = ConversationBufferMemory(memory_key="chat_history")
    if "simulator_chain" not in st.session_state:
        st.session_state.simulator_chain = None
    if "initial_advice_provided" not in st.session_state:
        st.session_state.initial_advice_provided = False
    if "is_chat_ended" not in st.session_state:
        st.session_state.is_chat_ended = False
    if "show_delete_confirm" not in st.session_state:
        st.session_state.show_delete_confirm = False
    if "customer_query_text_area" not in st.session_state:
        st.session_state.customer_query_text_area = ""
    if "agent_response_area_text" not in st.session_state:
        st.session_state.agent_response_area_text = ""
    if "last_transcript" not in st.session_state:
        st.session_state.last_transcript = ""
    if "sim_audio_bytes" not in st.session_state:
        st.session_state.sim_audio_bytes = None
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = "idle"
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = None
    if "openai_init_msg" not in st.session_state:
        st.session_state.openai_init_msg = ""
    if "sim_stage" not in st.session_state:
        st.session_state.sim_stage = "WAIT_FIRST_QUERY"
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "is_solution_provided" not in st.session_state:
        st.session_state.is_solution_provided = False
    if "transfer_summary_text" not in st.session_state:
        st.session_state.transfer_summary_text = ""
    if "language_transfer_requested" not in st.session_state:
        st.session_state.language_transfer_requested = False
    if "customer_attachment_file" not in st.session_state:
        st.session_state.customer_attachment_file = None
    if "language_at_transfer" not in st.session_state:
        st.session_state.language_at_transfer = st.session_state.language
    if "language_at_transfer_start" not in st.session_state:
        st.session_state.language_at_transfer_start = st.session_state.language
    if "transfer_retry_count" not in st.session_state:
        st.session_state.transfer_retry_count = 0
    if "customer_type_sim_select" not in st.session_state:
        default_customer_type = "까다로운 고객"
        if st.session_state.language == "en":
            default_customer_type = "Difficult Customer"
        elif st.session_state.language == "ja":
            default_customer_type = "難しい顧客"
        st.session_state.customer_type_sim_select = default_customer_type
    if "customer_email" not in st.session_state:
        st.session_state.customer_email = ""
    if "customer_phone" not in st.session_state:
        st.session_state.customer_phone = ""
    if "agent_response_input_box_widget" not in st.session_state:
        st.session_state.agent_response_input_box_widget = ""
    if "sim_instance_id" not in st.session_state:
        st.session_state.sim_instance_id = str(uuid.uuid4())
    if "sim_attachment_context_for_llm" not in st.session_state:
        st.session_state.sim_attachment_context_for_llm = ""
    if "realtime_hint_text" not in st.session_state:
        st.session_state.realtime_hint_text = ""
    if "sim_call_outbound_summary" not in st.session_state:
        st.session_state.sim_call_outbound_summary = ""
    if "sim_call_outbound_target" not in st.session_state:
        st.session_state.sim_call_outbound_target = None
    if "call_sim_stage" not in st.session_state:
        st.session_state.call_sim_stage = "WAITING_CALL"
    if "call_sim_mode" not in st.session_state:
        st.session_state.call_sim_mode = "INBOUND"
    if "incoming_phone_number" not in st.session_state:
        st.session_state.incoming_phone_number = "+82 10-1234-5678"
    if "is_on_hold" not in st.session_state:
        st.session_state.is_on_hold = False
    if "hold_start_time" not in st.session_state:
        st.session_state.hold_start_time = None
    if "total_hold_duration" not in st.session_state:
        st.session_state.total_hold_duration = timedelta(0)
    if "current_customer_audio_text" not in st.session_state:
        st.session_state.current_customer_audio_text = ""
    if "current_agent_audio_text" not in st.session_state:
        st.session_state.current_agent_audio_text = ""
    if "agent_response_input_box_widget_call" not in st.session_state:
        st.session_state.agent_response_input_box_widget_call = ""
    if "call_initial_query" not in st.session_state:
        st.session_state.call_initial_query = ""
    if "call_summary_text" not in st.session_state:
        st.session_state.call_summary_text = ""
    if "customer_initial_audio_bytes" not in st.session_state:
        st.session_state.customer_initial_audio_bytes = None
    if "supervisor_policy_context" not in st.session_state:
        st.session_state.supervisor_policy_context = ""
    if "agent_policy_attachment_content" not in st.session_state:
        st.session_state.agent_policy_attachment_content = ""
    if "customer_attachment_b64" not in st.session_state:
        st.session_state.customer_attachment_b64 = ""
    if "customer_history_summary" not in st.session_state:
        st.session_state.customer_history_summary = ""
    if "customer_avatar" not in st.session_state:
        st.session_state.customer_avatar = {
            "gender": "male",
            "state": "NEUTRAL",
        }
    if "bytes_to_process" not in st.session_state:
        st.session_state.bytes_to_process = None
    
    # API 키 관련 초기화
    from utils.llm_clients import SUPPORTED_APIS
    for api, cfg in SUPPORTED_APIS.items():
        if cfg["session_key"] not in st.session_state:
            st.session_state[cfg["session_key"]] = ""
    
    if "selected_llm" not in st.session_state:
        st.session_state.selected_llm = "openai_gpt4"
    
    # 초기화 완료 플래그 설정
    st.session_state.session_state_initialized = True















