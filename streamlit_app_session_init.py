# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
streamlit_app.py의 세션 상태 초기화 로직을 관리하는 모듈
"""

import streamlit as st
import uuid
from config import DEFAULT_LANG

# LangChain Memory import (다양한 버전 지원)
try:
    try:
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        try:
            from langchain_classic.memory import ConversationBufferMemory
        except ImportError:
            try:
                from langchain_core.memory import ConversationBufferMemory
            except ImportError:
                # LangChain이 없으면 None으로 설정 (선택적 기능)
                ConversationBufferMemory = None
except ImportError:
    ConversationBufferMemory = None

def init_all_session_state():
    """모든 세션 상태를 초기화"""
    from streamlit_app_imports import (
        CustomerDataManager, CallHandler, AppAudioHandler, SESSION_INIT_AVAILABLE
    )
    
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
        if ConversationBufferMemory is not None:
            st.session_state.simulator_memory = ConversationBufferMemory(
                memory_key="chat_history")
        else:
            # LangChain이 없으면 빈 딕셔너리로 대체
            st.session_state.simulator_memory = {}
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
    if "reset_agent_response_area" not in st.session_state:
        st.session_state.reset_agent_response_area = False
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
        st.session_state.sim_stage = "WAIT_ROLE_SELECTION"
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "is_solution_provided" not in st.session_state:
        st.session_state.is_solution_provided = False
    if "transfer_summary_text" not in st.session_state:
        st.session_state.transfer_summary_text = ""
    if "translation_success" not in st.session_state:
        st.session_state.translation_success = True
    if "language_transfer_requested" not in st.session_state:
        st.session_state.language_transfer_requested = False
    if "customer_attachment_file" not in st.session_state:
        st.session_state.customer_attachment_file = None
    if "customer_data_manager" not in st.session_state:
        st.session_state.customer_data_manager = CustomerDataManager()
    if "customer_data" not in st.session_state:
        st.session_state.customer_data = None
    if "show_agent_response_ui" not in st.session_state:
        st.session_state.show_agent_response_ui = False
    if "show_customer_data_ui" not in st.session_state:
        st.session_state.show_customer_data_ui = False
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
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""
    if "customer_email" not in st.session_state:
        st.session_state.customer_email = ""
    if "customer_phone" not in st.session_state:
        st.session_state.customer_phone = ""
    if "agent_response_input_box_widget" not in st.session_state:
        st.session_state.agent_response_input_box_widget = ""
    if "is_customer_verified" not in st.session_state:
        st.session_state.is_customer_verified = False
    if "verification_info" not in st.session_state:
        st.session_state.verification_info = {
            "receipt_number": "",
            "card_last4": "",
            "customer_name": "",
            "customer_email": "",
            "customer_phone": "",
            "verification_attempts": 0
        }
    if "verification_stage" not in st.session_state:
        st.session_state.verification_stage = "WAIT_VERIFICATION"
    if "sim_instance_id" not in st.session_state:
        st.session_state.sim_instance_id = str(uuid.uuid4())
    if "sim_perspective" not in st.session_state:
        st.session_state.sim_perspective = "AGENT"
    if "user_role_selected" not in st.session_state:
        st.session_state.user_role_selected = None
    if "is_auto_playing" not in st.session_state:
        st.session_state.is_auto_playing = False
    if "sim_attachment_context_for_llm" not in st.session_state:
        st.session_state.sim_attachment_context_for_llm = ""
    if "realtime_hint_text" not in st.session_state:
        st.session_state.realtime_hint_text = ""
    if "sim_call_outbound_summary" not in st.session_state:
        st.session_state.sim_call_outbound_summary = ""
    if "sim_call_outbound_target" not in st.session_state:
        st.session_state.sim_call_outbound_target = None
    if "call_handler" not in st.session_state:
        st.session_state.call_handler = CallHandler()
    if "audio_handler" not in st.session_state:
        st.session_state.audio_handler = AppAudioHandler()
    if "call_active" not in st.session_state:
        st.session_state.call_active = False
    
    # 세션 상태 초기화 모듈 호출
    if SESSION_INIT_AVAILABLE:
        from _pages._session_init import init_session_state
        init_session_state()

