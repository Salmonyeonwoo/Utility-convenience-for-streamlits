# ========================================
# _pages/_chat_simulator.py
# 채팅/이메일 시뮬레이터 모듈 (메인 진입점)
# ========================================

import streamlit as st
from lang_pack import LANG
from _pages._chat_simulator_panels import _render_customer_list_panel, _render_customer_info_panel
from _pages._chat_simulator_history import render_history_management_panel

# 하위 모듈 import
from _pages._chat_initial_query import render_initial_query
from _pages._chat_messages import render_chat_messages
from _pages._chat_agent_turn import render_agent_turn
from _pages._chat_customer_turn import render_customer_turn
from _pages._chat_closing import render_closing_stages

# ⭐ 시뮬레이션 입장 모드 관련
try:
    from simulation_perspective_logic import init_perspective_state, render_perspective_toggle
    PERSPECTIVE_LOGIC_AVAILABLE = True
except ImportError:
    PERSPECTIVE_LOGIC_AVAILABLE = False
    def init_perspective_state():
        if "sim_perspective" not in st.session_state:
            st.session_state.sim_perspective = "AGENT"
        if "is_auto_playing" not in st.session_state:
            st.session_state.is_auto_playing = False
    def render_perspective_toggle():
        pass

def render_chat_simulator():
    """채팅/이메일 시뮬레이터 렌더링 함수"""
    if PERSPECTIVE_LOGIC_AVAILABLE:
        init_perspective_state()
    else:
        if "sim_perspective" not in st.session_state:
            st.session_state.sim_perspective = "AGENT"
        if "is_auto_playing" not in st.session_state:
            st.session_state.is_auto_playing = False
    
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    if PERSPECTIVE_LOGIC_AVAILABLE:
        render_perspective_toggle(L)
    
    # AHT 타이머 (화면 최상단)
    if st.session_state.sim_stage not in ["WAIT_FIRST_QUERY", "CLOSING", "idle"]:
        from _pages._chat_history import render_aht_timer
        render_aht_timer(L)

    # LLM 준비 체크 & 채팅 종료 상태
    from llm_client import get_api_key
    has_api_key = any([
        bool(get_api_key("openai")),
        bool(get_api_key("gemini")),
        bool(get_api_key("claude")),
        bool(get_api_key("groq"))
    ])
    
    if has_api_key:
        st.session_state.is_llm_ready = True
    
    if not has_api_key:
        st.warning(L["simulation_no_key_warning"])

    if st.session_state.sim_stage == "CLOSING":
        from _pages._chat_history import render_closing_downloads
        render_closing_downloads(L, current_lang)

    elif st.session_state.sim_stage == "OUTBOUND_CALL_IN_PROGRESS":
        from _pages._chat_history import render_outbound_call
        render_outbound_call(L, current_lang)

    # 역할 선택 (WAIT_ROLE_SELECTION)
    if st.session_state.sim_stage == "WAIT_ROLE_SELECTION":
        from _pages._chat_role_selection import render_role_selection
        render_role_selection(L, current_lang)
    
    # 초기 문의 입력 (WAIT_FIRST_QUERY)
    elif st.session_state.sim_stage == "WAIT_FIRST_QUERY":
        render_initial_query(L, current_lang)

    # 에이전트 입력 단계 (AGENT_TURN)
    if st.session_state.sim_stage == "AGENT_TURN":
        render_agent_turn(L, current_lang)

    # 에스컬레이션 요청 단계 (ESCALATION_REQUIRED)
    elif st.session_state.sim_stage == "ESCALATION_REQUIRED":
        from _pages._chat_closing import render_escalation
        render_escalation(L, current_lang)

    # 고객 반응 생성 단계 (CUSTOMER_TURN)
    elif st.session_state.sim_stage == "CUSTOMER_TURN":
        render_customer_turn(L, current_lang)

    # 3-column 레이아웃 적용
    use_3column = st.session_state.sim_stage not in [
        "WAIT_ROLE_SELECTION", "WAIT_FIRST_QUERY", "CLOSING", 
        "OUTBOUND_CALL_IN_PROGRESS", "idle", "ESCALATION_REQUIRED"
    ]
    
    if use_3column:
        col1, col2, col3 = st.columns([1, 2, 1.5])
        
        # 고객 목록 및 이력 관리 (col1에 배치)
        with col1:
            _render_customer_list_panel(L, current_lang)
            render_history_management_panel(L, current_lang)

        # 대화 로그 표시 (col2에 배치) - 스크린샷 스타일
        with col2:
            # 채팅 제목 표시
            if st.session_state.get("customer_data"):
                customer_name = st.session_state.customer_data.get("basic_info", {}).get("customer_name", L.get("customer_label", "고객"))
            else:
                customer_name = st.session_state.get("customer_name", L.get("customer_label", "고객")) or L.get("customer_label", "고객")
            # 다국어 지원: 고객 이름과 대화 제목
            conversation_title = L.get("conversation_with", "님과의 대화")
            if current_lang == "ko":
                customer_display = f"{customer_name}{conversation_title}"
            elif current_lang == "ja":
                customer_display = f"{customer_name}{conversation_title}"
            else:  # 영어
                customer_display = f"{conversation_title} {customer_name}"
            st.subheader(f"💬 {customer_display}")
            
            # 스크롤 가능한 채팅 영역 (스크린샷 스타일)
            render_chat_messages(L, current_lang)

        # 고객 정보 표시 (col3에 배치)
        with col3:
            _render_customer_info_panel(L, current_lang)

        # 종료 관련 단계들 (col2에 배치)
        with col2:
            render_closing_stages(L, current_lang)
    else:
        # 특정 단계에서는 기존 레이아웃 유지
        render_chat_messages(L, current_lang)
        render_closing_stages(L, current_lang)
