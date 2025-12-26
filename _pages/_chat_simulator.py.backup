# ========================================
# _pages/_chat_simulator.py
# 채팅/이메일 시뮬레이터 모듈 (메인 진입점)
# ========================================

import streamlit as st
from lang_pack import LANG
from utils.history_handler import get_daily_data_statistics
from datetime import datetime

# 하위 모듈 import
from _pages._chat_history import render_chat_history
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
        pass  # 사이드바 토글 비활성화


def render_chat_simulator():
    """채팅/이메일 시뮬레이터 렌더링 함수"""
    # ⭐ 시뮬레이션 입장 상태 초기화
    if PERSPECTIVE_LOGIC_AVAILABLE:
        init_perspective_state()
    else:
        # 기본 초기화
        if "sim_perspective" not in st.session_state:
            st.session_state.sim_perspective = "AGENT"
        if "is_auto_playing" not in st.session_state:
            st.session_state.is_auto_playing = False
    
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # ⭐ 시뮬레이션 모드 선택 UI를 탭 내부 상단에 표시
    if PERSPECTIVE_LOGIC_AVAILABLE:
        render_perspective_toggle(L)
    
    # =========================
    # 0-1. 일일 데이터 수집 통계 표시
    # =========================
    daily_stats = get_daily_data_statistics(st.session_state.language)
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric(L.get("daily_stats_cases_collected", "오늘 수집된 케이스"), daily_stats["total_cases"])
    with col_stat2:
        st.metric(L.get("daily_stats_unique_customers", "고유 고객 수"), daily_stats["unique_customers"],
                  delta=L.get("daily_stats_target_met", "목표: 5인 이상") if daily_stats["target_met"] else L.get("daily_stats_target_not_met", "목표 미달"))
    with col_stat3:
        st.metric(L.get("daily_stats_summary_completed", "요약 완료 케이스"), daily_stats["cases_with_summary"])
    with col_stat4:
        status_icon = "✅" if daily_stats["target_met"] else "⚠️"
        st.metric(L.get("daily_stats_goal_achievement", "목표 달성"), status_icon,
                  delta=L.get("daily_stats_achieved", "달성") if daily_stats["target_met"] else L.get("daily_stats_not_achieved", "미달성"))

    st.markdown("---")

    # =========================
    # 이력 관리 모듈 호출
    # =========================
    render_chat_history(current_lang, L)

    # =========================
    # AHT 타이머 (화면 최상단)
    # =========================
    if st.session_state.sim_stage not in [
            "WAIT_FIRST_QUERY", "CLOSING", "idle"]:
        from _pages._chat_history import render_aht_timer
        render_aht_timer(L)

    # =========================
    # LLM 준비 체크 & 채팅 종료 상태
    # =========================
    # ⭐ API Key가 실제로 있는지 확인 (항상 최신 상태로 확인)
    from llm_client import get_api_key
    has_api_key = any([
        bool(get_api_key("openai")),
        bool(get_api_key("gemini")),
        bool(get_api_key("claude")),
        bool(get_api_key("groq"))
    ])
    
    # ⭐ API Key가 있으면 is_llm_ready를 강제로 True로 설정
    if has_api_key:
        st.session_state.is_llm_ready = True
    
    # ⭐ API Key가 없을 때만 경고 표시
    if not has_api_key:
        st.warning(L["simulation_no_key_warning"])

    if st.session_state.sim_stage == "CLOSING":
        from _pages._chat_history import render_closing_downloads
        render_closing_downloads(L, current_lang)

    # =========================
    # 전화 발신 진행 중 (OUTBOUND_CALL_IN_PROGRESS)
    # =========================
    elif st.session_state.sim_stage == "OUTBOUND_CALL_IN_PROGRESS":
        from _pages._chat_history import render_outbound_call
        render_outbound_call(L, current_lang)

    # =========================
    # ⭐ 핵심 수정: 로직 처리를 메시지 렌더링보다 먼저 수행
    # =========================
    # 0. 역할 선택 (WAIT_ROLE_SELECTION) - 새로운 단계
    # =========================
    if st.session_state.sim_stage == "WAIT_ROLE_SELECTION":
        from _pages._chat_role_selection import render_role_selection
        render_role_selection(L, current_lang)
    
    # =========================
    # 1. 초기 문의 입력 (WAIT_FIRST_QUERY)
    # =========================
    elif st.session_state.sim_stage == "WAIT_FIRST_QUERY":
        render_initial_query(L, current_lang)

    # =========================
    # 2. 에이전트 입력 단계 (AGENT_TURN) - 로직 처리 먼저
    # =========================
    if st.session_state.sim_stage == "AGENT_TURN":
        render_agent_turn(L, current_lang)

    # =========================
    # 3. 에스컬레이션 요청 단계 (ESCALATION_REQUIRED)
    # =========================
    elif st.session_state.sim_stage == "ESCALATION_REQUIRED":
        from _pages._chat_closing import render_escalation
        render_escalation(L, current_lang)

    # =========================
    # 4. 고객 반응 생성 단계 (CUSTOMER_TURN) - 로직 처리 먼저
    # =========================
    elif st.session_state.sim_stage == "CUSTOMER_TURN":
        render_customer_turn(L, current_lang)

    # =========================
    # 5. 대화 로그 표시 (공통) - 로직 처리 후 메시지 렌더링
    # =========================
    # ⭐ 이 순서로 변경: 로직 처리(위) -> 메시지 렌더링(아래)
    # 이렇게 하면 메시지가 추가되어도 바로 화면에 반영됨
    render_chat_messages(L, current_lang)

    # =========================
    # 종료 관련 단계들
    # =========================
    render_closing_stages(L, current_lang)
