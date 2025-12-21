# ========================================
# ui/sidebar.py
# 사이드바 UI 컴포넌트 모듈
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

from lang_pack import LANG
from datetime import timedelta


def render_sidebar():
    """사이드바 UI 렌더링"""
    with st.sidebar:
        # 언어 키 안전하게 가져오기
        if "language" not in st.session_state:
            st.session_state.language = "ko"
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        
        # 회사 목록 초기화 (회사 정보 탭에서 사용)
        if "company_language_priority" not in st.session_state:
            st.session_state.company_language_priority = {
                "default": ["ko", "en", "ja"],
                "companies": {}
            }
        
        st.markdown("---")
        
        # 언어 선택
        if "language" not in st.session_state:
            st.session_state.language = "ko"
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        
        lang_priority = st.session_state.company_language_priority["default"]
        
        selected_lang_key = st.selectbox(
            L["lang_select"],
            options=lang_priority,
            index=lang_priority.index(st.session_state.language) if st.session_state.language in lang_priority else 0,
            format_func=lambda x: {"ko": "한국어", "en": "English", "ja": "日本語"}[x],
        )

        # 🔹 언어 변경 감지
        if selected_lang_key != st.session_state.language:
            st.session_state.language = selected_lang_key
            # 채팅/전화 공통 상태 초기화
            st.session_state.simulator_messages = []
            # ⭐ 안전한 메모리 초기화
            try:
                if hasattr(st.session_state, 'simulator_memory') and st.session_state.simulator_memory is not None:
                    st.session_state.simulator_memory.clear()
            except Exception:
                # 메모리 초기화 실패 시 새로 생성
                try:
                    st.session_state.simulator_memory = ConversationBufferMemory(memory_key="chat_history")
                except Exception:
                    pass  # 초기화 실패해도 계속 진행
            st.session_state.initial_advice_provided = False
            st.session_state.is_chat_ended = False
            # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로 플래그 사용
            st.session_state.reset_agent_response_area = True
            st.session_state.customer_query_text_area = ""
            st.session_state.last_transcript = ""
            st.session_state.sim_audio_bytes = None
            st.session_state.sim_stage = "WAIT_FIRST_QUERY"
            st.session_state.customer_attachment_file = []  # 언어 변경 시 첨부 파일 초기화
            st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
            st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
            # 전화 시뮬레이터 상태 초기화
            st.session_state.call_sim_stage = "WAITING_CALL"
            st.session_state.call_sim_mode = "INBOUND"
            st.session_state.is_on_hold = False
            st.session_state.total_hold_duration = timedelta(0)
            st.session_state.hold_start_time = None
            st.session_state.current_customer_audio_text = ""
            st.session_state.current_agent_audio_text = ""
            st.session_state.agent_response_input_box_widget_call = ""
            st.session_state.call_initial_query = ""
            # 전화 발신 관련 상태 초기화
            st.session_state.sim_call_outbound_summary = ""
            st.session_state.sim_call_outbound_target = None
            # ⭐ 언어 변경 시 재실행 - 무한 루프 방지를 위해 플래그 사용
            if "language_changed" not in st.session_state or not st.session_state.language_changed:
                st.session_state.language_changed = True
            else:
                # 이미 한 번 재실행했으면 플래그 초기화
                st.session_state.language_changed = False

        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])

        st.title(L["sidebar_title"])
        st.markdown("---")

        # ⭐ 기능 선택 - 기본값을 AI 챗 시뮬레이터로 설정 (먼저 배치)
        if "feature_selection" not in st.session_state:
            st.session_state.feature_selection = L["sim_tab_chat_email"]

        # ⭐ 핵심 기능과 더보기 기능 분리 (회사 정보 및 FAQ 추가)
        core_features = [L["sim_tab_chat_email"], L["sim_tab_phone"], L["company_info_tab"]]
        other_features = [L["rag_tab"], L["content_tab"], L["lstm_tab"], L["voice_rec_header"]]
        
        # 모든 기능을 하나의 리스트로 통합 (하나만 선택 가능하도록)
        all_features = core_features + other_features
        
        # 현재 선택된 기능
        current_selection = st.session_state.get("feature_selection", L["sim_tab_chat_email"])
        
        # 현재 선택의 인덱스 찾기
        try:
            current_index = all_features.index(current_selection) if current_selection in all_features else 0
        except (ValueError, AttributeError):
            current_index = 0
        
        # ⭐ 기능 선택 섹션
        st.subheader("📋 기능 선택")
        selected_feature = st.radio(
            "기능 선택",
            all_features,
            index=current_index,
            key="unified_feature_selection",
            label_visibility="visible"
        )
        
        # 선택된 기능 업데이트
        if selected_feature != current_selection:
            st.session_state.feature_selection = selected_feature
        
        feature_selection = st.session_state.get("feature_selection", L["sim_tab_chat_email"])
        
        st.markdown("---")
        
        # ⭐ LLM 모델 선택 (API Key 입력 필드는 제외)
        st.subheader("🤖 LLM 모델 선택")
        
        llm_options = {
            "openai_gpt4": "OpenAI GPT-4",
            "openai_gpt35": "OpenAI GPT-3.5",
            "gemini_pro": "Google Gemini Pro",
            "gemini_flash": "Google Gemini Flash",
            "claude": "Anthropic Claude",
            "groq": "Groq",
            "nvidia": "NVIDIA NIM"
        }
        
        current_llm = st.session_state.get("selected_llm", "openai_gpt4")
        selected_llm = st.selectbox(
            "LLM 모델 선택",
            options=list(llm_options.keys()),
            format_func=lambda x: llm_options[x],
            index=list(llm_options.keys()).index(current_llm) if current_llm in llm_options else 0,
            key="sidebar_llm_select"
        )
        if selected_llm != current_llm:
            st.session_state.selected_llm = selected_llm

