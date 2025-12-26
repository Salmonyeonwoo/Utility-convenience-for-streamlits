# ========================================
# _pages/_chat_role_selection.py
# 채팅 시뮬레이터 - 역할 선택 모듈
# ========================================

import streamlit as st
from lang_pack import LANG


def render_role_selection(L, current_lang):
    """역할 선택 UI 렌더링 (고객 vs 에이전트)"""
    st.markdown("---")
    st.subheader(L.get("role_selection_header", "당신은 어느 분이십니까?"))
    
    # 역할 선택 라디오 버튼
    role_options = {
        "CUSTOMER": L.get("role_selection_customer", "A. 고객"),
        "AGENT": L.get("role_selection_agent", "B. 에이전트")
    }
    
    # 세션 상태에서 역할이 이미 선택되어 있는지 확인
    if "user_role_selected" not in st.session_state:
        st.session_state.user_role_selected = None
    
    selected_role = st.radio(
        L.get("role_selection_help", "고객으로 문의하시거나, 에이전트로 응대하실 역할을 선택해주세요."),
        options=list(role_options.keys()),
        format_func=lambda x: role_options[x],
        index=0 if st.session_state.user_role_selected == "CUSTOMER" else (1 if st.session_state.user_role_selected == "AGENT" else 0),
        help=L.get("role_selection_help", "고객으로 문의하시거나, 에이전트로 응대하실 역할을 선택해주세요."),
        horizontal=True,
        key="role_selection_radio"
    )
    
    # 역할 선택 저장
    if selected_role != st.session_state.user_role_selected:
        st.session_state.user_role_selected = selected_role
        # 역할에 따라 sim_perspective도 설정
        if selected_role == "CUSTOMER":
            st.session_state.sim_perspective = "CUSTOMER"
        else:
            st.session_state.sim_perspective = "AGENT"
    
    st.markdown("---")
    
    # 역할 선택 완료 후 다음 단계로 진행
    if st.button(
        L.get("button_continue_role", "계속하기"),
        key="role_selection_continue",
        use_container_width=True,
        type="primary"
    ):
        if st.session_state.user_role_selected:
            # 역할 선택 완료 후 문의 입력 단계로 이동
            st.session_state.sim_stage = "WAIT_FIRST_QUERY"
            # st.rerun()은 버튼 클릭 후 자동으로 실행됨
        else:
            st.warning(L.get("warning_select_role", "역할을 선택해주세요."))

