import streamlit as st
import time
from datetime import datetime
from simulation_handler import generate_customer_reaction, generate_agent_response_draft

def init_perspective_state():
    """시뮬레이션 입장(상담원 vs 고객) 상태 초기화"""
    if "sim_perspective" not in st.session_state:
        st.session_state.sim_perspective = "AGENT"  # 기본값: 상담원 테스트 모드
    if "is_auto_playing" not in st.session_state:
        st.session_state.is_auto_playing = False

def render_perspective_toggle(L=None):
    """탭 내부에 입장 변경 토글 렌더링 (사이드바 아님)"""
    # 언어 팩 로드
    if L is None:
        from lang_pack import LANG
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
    
    # 모드 옵션 (언어 팩 사용)
    mode_options = {
        "AGENT": L.get("simulation_mode_agent", "🙋‍♂️ 상담원 테스트 (사용자=상담원)"),
        "CUSTOMER": L.get("simulation_mode_customer", "👤 고객 체험 (사용자=고객)")
    }
    
    # 헤더와 라디오 버튼을 컬럼으로 배치하여 깔끔하게 표시
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"**{L.get('simulation_mode_header', '🔄 시뮬레이션 모드 설정')}**")
    with col2:
        selected_mode = st.radio(
            L.get("simulation_mode_select", "테스트 시점 선택"),
            options=list(mode_options.keys()),
            format_func=lambda x: mode_options[x],
            index=0 if st.session_state.sim_perspective == "AGENT" else 1,
            help=L.get("simulation_mode_help", "상담원 입장에서 AI 고객을 응대할지, 고객 입장에서 AI 상담원에게 문의할지 선택합니다."),
            horizontal=True,
            key="perspective_toggle_main"
        )
    
    if selected_mode != st.session_state.sim_perspective:
        st.session_state.sim_perspective = selected_mode
        # 모드 변경 시 대화 초기화 권장 (simulator_messages 사용)
        if "simulator_messages" in st.session_state:
            st.session_state.simulator_messages = []
        # 초기 단계로 리셋
        st.session_state.sim_stage = "WAIT_FIRST_QUERY"
        # st.rerun()  # 주석 처리: radio 버튼 클릭 후 Streamlit이 자동 rerun함
    
    st.markdown("---")  # 구분선 추가

def handle_simulation_flow(L):
    """입장에 따른 통합 흐름 제어 로직 (각 모듈에서 호출)"""
    perspective = st.session_state.get("sim_perspective", "AGENT")
    stage = st.session_state.get("sim_stage", "WAIT_FIRST_QUERY")

    # ---------------------------------------------------------
    # 1. 상담원 테스트 모드 (기존 로직 유지)
    # ---------------------------------------------------------
    if perspective == "AGENT":
        # 기존 로직은 각 모듈(_chat_agent_turn.py, _chat_customer_turn.py)에서 처리
        return False  # 기존 로직 사용

    # ---------------------------------------------------------
    # 2. 고객 체험 모드 (새로운 로직)
    # ---------------------------------------------------------
    else:
        if stage == "AGENT_TURN":
            # [AI 차례] AI가 에이전트로서 응답 생성
            # 이 로직은 _chat_agent_turn.py에서 처리하도록 함
            return True  # 고객 모드임을 알림
        elif stage == "CUSTOMER_TURN":
            # [사용자 차례] 사용자가 고객으로서 직접 입력
            # 이 로직은 _chat_customer_turn.py에서 처리하도록 함
            return True  # 고객 모드임을 알림
    
    return False

# ==========================================
# 기존 파일 수정 시 참고할 가이드라인
# ==========================================
"""
1. _chat_agent_turn.py:
   - if perspective == "CUSTOMER": 
       AI 답변 자동 생성 후 stage = "CUSTOMER_TURN"으로 변경 로직 추가

2. _chat_customer_turn.py:
   - if perspective == "CUSTOMER":
       st.chat_input()을 표시하여 사용자 입력을 받는 로직으로 대체

3. _chat_initial_query.py:
   - 고객 모드일 경우, 첫 문의 입력 후 바로 AI 답변이 생성되도록 
     sim_stage를 "AGENT_TURN"으로 설정
"""