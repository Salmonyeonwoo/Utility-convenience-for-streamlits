# ========================================
# session_state_manager.py
# 세션 상태 초기화 및 관리 모듈
# ========================================

import streamlit as st

# LangChain Memory import with fallback support
try:
    try:
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        try:
            from langchain_classic.memory import ConversationBufferMemory
        except ImportError:
            from langchain_core.memory import ConversationBufferMemory
except ImportError:
    # Fallback: Create a simple mock class if langchain is not available
    class ConversationBufferMemory:
        def __init__(self, **kwargs):
            self.memory_key = kwargs.get("memory_key", "chat_history")
            self.chat_memory = type('obj', (object,), {'messages': []})()
        
        def save_context(self, inputs, outputs):
            pass
        
        def load_memory_variables(self, inputs):
            return {self.memory_key: []}

from customer_data_manager import CustomerDataManager
from lang_pack import LANG, DEFAULT_LANG


def initialize_session_state():
    """세션 상태 초기화 함수"""
    
    # 기본 언어 설정
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANG
    if "is_llm_ready" not in st.session_state:
        st.session_state.is_llm_ready = False
    if "llm_init_error_msg" not in st.session_state:
        st.session_state.llm_init_error_msg = ""
    
    # 파일 및 RAG 관련
    if "uploaded_files_state" not in st.session_state:
        st.session_state.uploaded_files_state = None
    if "is_rag_ready" not in st.session_state:
        st.session_state.is_rag_ready = False
    if "rag_vectorstore" not in st.session_state:
        st.session_state.rag_vectorstore = None
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    # 에이전트 입력 관련
    if "agent_input" not in st.session_state:
        st.session_state.agent_input = ""
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    
    # 시뮬레이터 관련
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
    
    # 고객 쿼리 및 응답 관련
    if "customer_query_text_area" not in st.session_state:
        st.session_state.customer_query_text_area = ""
    if "agent_response_area_text" not in st.session_state:
        st.session_state.agent_response_area_text = ""
    if "reset_agent_response_area" not in st.session_state:
        st.session_state.reset_agent_response_area = False
    if "last_transcript" not in st.session_state:
        st.session_state.last_transcript = ""
    if "sim_audio_bytes" not in st.session_state:
        st.session_state.sim_audio_bytes = None
    
    # 채팅 상태
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = "idle"
    if "sim_stage" not in st.session_state:
        st.session_state.sim_stage = "WAIT_FIRST_QUERY"
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    
    # LLM 클라이언트 관련
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = None
    if "openai_init_msg" not in st.session_state:
        st.session_state.openai_init_msg = ""
    
    # 솔루션 및 이관 관련
    if "is_solution_provided" not in st.session_state:
        st.session_state.is_solution_provided = False
    if "transfer_summary_text" not in st.session_state:
        st.session_state.transfer_summary_text = ""
    if "translation_success" not in st.session_state:
        st.session_state.translation_success = True
    if "language_transfer_requested" not in st.session_state:
        st.session_state.language_transfer_requested = False
    if "language_at_transfer" not in st.session_state:
        st.session_state.language_at_transfer = st.session_state.language
    if "language_at_transfer_start" not in st.session_state:
        st.session_state.language_at_transfer_start = st.session_state.language
    if "transfer_retry_count" not in st.session_state:
        st.session_state.transfer_retry_count = 0
    
    # 고객 정보 관련
    if "customer_attachment_file" not in st.session_state:
        st.session_state.customer_attachment_file = None
    if "customer_data_manager" not in st.session_state:
        st.session_state.customer_data_manager = CustomerDataManager()
    if "customer_data" not in st.session_state:
        st.session_state.customer_data = None
    if "customer_email" not in st.session_state:
        st.session_state.customer_email = ""
    if "customer_phone" not in st.session_state:
        st.session_state.customer_phone = ""
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""
    if "customer_company" not in st.session_state:
        st.session_state.customer_company = ""
    
    # UI 표시 관련
    if "show_agent_response_ui" not in st.session_state:
        st.session_state.show_agent_response_ui = False
    if "show_customer_data_ui" not in st.session_state:
        st.session_state.show_customer_data_ui = False
    if "show_verification_ui" not in st.session_state:
        st.session_state.show_verification_ui = False
    if "show_draft_ui" not in st.session_state:
        st.session_state.show_draft_ui = False
    if "show_agent_file_uploader" not in st.session_state:
        st.session_state.show_agent_file_uploader = False
    
    # 고객 타입 및 검증 관련
    if "customer_type_sim_select" not in st.session_state:
        current_lang = st.session_state.get("language", "ko")
        default_customer_type = "까다로운 고객"
        if current_lang == "en":
            default_customer_type = "Difficult Customer"
        elif current_lang == "ja":
            default_customer_type = "難しい顧客"
        st.session_state.customer_type_sim_select = default_customer_type
    
    if "is_customer_verified" not in st.session_state:
        st.session_state.is_customer_verified = False
    if "verification_info" not in st.session_state:
        st.session_state.verification_info = {
            "receipt_number": "",
            "card_last4": "",
            "account_number": "",
            "payment_method": "",
            "customer_name": "",
            "customer_email": "",
            "customer_phone": "",
            "verification_attempts": 0
        }
    if "verification_stage" not in st.session_state:
        st.session_state.verification_stage = "WAIT_VERIFICATION"
    
    # 첨부 파일 관련
    if "agent_attachment_file" not in st.session_state:
        st.session_state.agent_attachment_file = []
    if "sim_attachment_context_for_llm" not in st.session_state:
        st.session_state.sim_attachment_context_for_llm = ""
    
    # 응대 초안 관련
    if "realtime_hint_text" not in st.session_state:
        st.session_state.realtime_hint_text = ""
    if "auto_generated_draft" not in st.session_state:
        st.session_state.auto_generated_draft = None
    
    # 전화 관련
    if "call_sim_stage" not in st.session_state:
        st.session_state.call_sim_stage = "IDLE"
    if "is_on_hold" not in st.session_state:
        st.session_state.is_on_hold = False
    if "hold_start_time" not in st.session_state:
        st.session_state.hold_start_time = None
    if "total_hold_duration" not in st.session_state:
        st.session_state.total_hold_duration = 0
    
    # 인스턴스 ID
    if "sim_instance_id" not in st.session_state:
        import uuid
        st.session_state.sim_instance_id = str(uuid.uuid4())[:8]
    
    # 상담 시작 시간
    if "consultation_started_at" not in st.session_state:
        from datetime import datetime
        st.session_state.consultation_started_at = datetime.now().isoformat()
    
    # 메일 끝인사 관련
    if "has_email_closing" not in st.session_state:
        st.session_state.has_email_closing = False
    
    # API 키 해시
    if "api_keys_hash" not in st.session_state:
        st.session_state.api_keys_hash = ""
    
    # 선택된 LLM
    if "selected_llm" not in st.session_state:
        st.session_state.selected_llm = "openai_gpt4"
    
    # 기능 선택
    current_lang = st.session_state.get("language", "ko")
    L = LANG.get(current_lang, LANG["ko"])
    if "feature_selection" not in st.session_state:
        st.session_state.feature_selection = L.get("sim_tab_chat_email", "채팅/이메일 시뮬레이터")
    
    # 언어 변경 감지
    if "language_changed" not in st.session_state:
        st.session_state.language_changed = False
    
    # 전화 관련 추가 상태
    if "call_handler" not in st.session_state:
        try:
            from call_handler import CallHandler
            st.session_state.call_handler = CallHandler()
        except ImportError:
            # fallback: streamlit_app.py에 정의된 CallHandler 사용
            pass  # streamlit_app.py에서 직접 초기화됨
    if "audio_handler" not in st.session_state:
        try:
            from audio_handler import AppAudioHandler
            st.session_state.audio_handler = AppAudioHandler()
        except ImportError:
            # fallback: streamlit_app.py에 정의된 AppAudioHandler 사용
            pass  # streamlit_app.py에서 직접 초기화됨
    if "call_active" not in st.session_state:
        st.session_state.call_active = False
    if "current_call_id" not in st.session_state:
        st.session_state.current_call_id = None
    if "video_enabled" not in st.session_state:
        st.session_state.video_enabled = False
    if "opponent_video_frames" not in st.session_state:
        st.session_state.opponent_video_frames = []
    if "last_camera_frame" not in st.session_state:
        st.session_state.last_camera_frame = None
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
        from datetime import timedelta
        st.session_state.total_hold_duration = timedelta(0)
    if "current_customer_audio_text" not in st.session_state:
        st.session_state.current_customer_audio_text = ""
    if "current_agent_audio_text" not in st.session_state:
        st.session_state.current_agent_audio_text = ""
    if "agent_response_input_box_widget_call" not in st.session_state:
        st.session_state.agent_response_input_box_widget_call = ""
    if "call_initial_query" not in st.session_state:
        st.session_state.call_initial_query = ""
    if "call_website_url" not in st.session_state:
        st.session_state.call_website_url = ""
    if "call_summary_text" not in st.session_state:
        st.session_state.call_summary_text = ""
    if "customer_initial_audio_bytes" not in st.session_state:
        st.session_state.customer_initial_audio_bytes = None
    if "last_recorded_audio_bytes" not in st.session_state:
        st.session_state.last_recorded_audio_bytes = None
    if "last_customer_audio_bytes" not in st.session_state:
        st.session_state.last_customer_audio_bytes = None
    if "keep_customer_audio_display" not in st.session_state:
        st.session_state.keep_customer_audio_display = False
    if "customer_audio_played_once" not in st.session_state:
        st.session_state.customer_audio_played_once = False
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
    if "current_customer_video" not in st.session_state:
        st.session_state.current_customer_video = None
    if "current_customer_video_bytes" not in st.session_state:
        st.session_state.current_customer_video_bytes = None
    if "is_video_sync_enabled" not in st.session_state:
        st.session_state.is_video_sync_enabled = True
    if "video_male_neutral" not in st.session_state:
        st.session_state.video_male_neutral = None
    if "video_male_happy" not in st.session_state:
        st.session_state.video_male_happy = None
    if "video_male_angry" not in st.session_state:
        st.session_state.video_male_angry = None
    if "video_male_asking" not in st.session_state:
        st.session_state.video_male_asking = None
    if "video_male_sad" not in st.session_state:
        st.session_state.video_male_sad = None
    if "video_female_neutral" not in st.session_state:
        st.session_state.video_female_neutral = None
    if "video_female_happy" not in st.session_state:
        st.session_state.video_female_happy = None
    if "video_female_angry" not in st.session_state:
        st.session_state.video_female_angry = None
    if "video_female_asking" not in st.session_state:
        st.session_state.video_female_asking = None
    if "video_female_sad" not in st.session_state:
        st.session_state.video_female_sad = None
    if "bytes_to_process" not in st.session_state:
        st.session_state.bytes_to_process = None
    if "sim_call_outbound_summary" not in st.session_state:
        st.session_state.sim_call_outbound_summary = ""
    if "sim_call_outbound_target" not in st.session_state:
        st.session_state.sim_call_outbound_target = None
    if "agent_response_input_box_widget" not in st.session_state:
        st.session_state.agent_response_input_box_widget = ""
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""
    if "customer_company" not in st.session_state:
        st.session_state.customer_company = ""
    if "show_verification_ui" not in st.session_state:
        st.session_state.show_verification_ui = False
    if "show_draft_ui" not in st.session_state:
        st.session_state.show_draft_ui = False
    if "show_agent_file_uploader" not in st.session_state:
        st.session_state.show_agent_file_uploader = False
    if "agent_attachment_file" not in st.session_state:
        st.session_state.agent_attachment_file = []
    if "sim_attachment_context_for_llm" not in st.session_state:
        st.session_state.sim_attachment_context_for_llm = ""
    if "realtime_hint_text" not in st.session_state:
        st.session_state.realtime_hint_text = ""
    if "auto_generated_draft" not in st.session_state:
        st.session_state.auto_generated_draft = None
    if "has_email_closing" not in st.session_state:
        st.session_state.has_email_closing = False
    if "consultation_started_at" not in st.session_state:
        from datetime import datetime
        st.session_state.consultation_started_at = datetime.now().isoformat()
    
    # API 키 해시 및 선택된 LLM
    if "api_keys_hash" not in st.session_state:
        st.session_state.api_keys_hash = ""
    
    # SUPPORTED_APIS 관련 세션 상태
    try:
        from config import SUPPORTED_APIS
        for api, cfg in SUPPORTED_APIS.items():
            if cfg["session_key"] not in st.session_state:
                st.session_state[cfg["session_key"]] = ""
    except ImportError:
        pass

