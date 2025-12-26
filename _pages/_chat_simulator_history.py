# ========================================
# _pages/_chat_simulator_history.py
# 채팅 시뮬레이터의 이력 관리 모듈
# ========================================

import streamlit as st
from datetime import datetime
from utils.history_handler import delete_all_history_local, load_simulation_histories_local

def render_history_management_panel(L, current_lang):
    """이력 관리 패널 렌더링 (col1 하단)"""
    st.markdown("---")
    st.markdown("**📋 이력 관리**")
    if st.button("🗑️ 모든 이력 삭제", key="trigger_delete_hist_compact", use_container_width=True):
        st.session_state.show_delete_confirm = True
    if st.button("🔄 세션 초기화", key="reset_all_session_compact", use_container_width=True, help="모든 채팅/통화 응대 기록을 초기화합니다"):
        st.session_state.show_reset_confirm = True
    
    # 이전 이력 로드 (expander로 축소)
    histories = load_simulation_histories_local(current_lang)
    if histories:
        with st.expander("📂 이전 상담 이력 로드 (최근 10건)", expanded=False):
            filtered_histories = [h for h in histories[:10] if not h.get("is_call", False)]
            if filtered_histories:
                def _label(h):
                    try:
                        t = datetime.fromisoformat(h["timestamp"])
                        t_str = t.strftime("%m-%d %H:%M")
                    except Exception:
                        t_str = h.get("timestamp", "")
                    summary = h.get("summary")
                    if summary and isinstance(summary, dict):
                        main_inquiry = summary.get("main_inquiry", h["initial_query"][:20])
                        return f"[{t_str}] {h['customer_type']} - {main_inquiry[:20]}..."
                    else:
                        q = h["initial_query"][:20].replace("\n", " ")
                        return f"[{t_str}] {h['customer_type']} - {q}..."
                
                options_map = {_label(h): h for h in filtered_histories}
                sel_key = st.selectbox("이력 선택", options=list(options_map.keys()), key="hist_select_compact")
                if st.button("로드", key="load_hist_btn_compact", use_container_width=True):
                    h = options_map[sel_key]
                    st.session_state.customer_query_text_area = h["initial_query"]
                    st.session_state.simulator_messages = h.get("messages", [])
                    st.session_state.initial_advice_provided = True
                    st.session_state.is_chat_ended = h.get("is_chat_ended", False)
                    st.session_state.sim_attachment_context_for_llm = h.get("attachment_context", "")
                    st.session_state.sim_stage = "AGENT_TURN" if not h.get("is_chat_ended", False) else "CLOSING"
            else:
                st.info("로드할 이력이 없습니다.")
    
    # 삭제 및 초기화 확인 다이얼로그
    if st.session_state.get("show_delete_confirm", False):
        st.warning("⚠️ 모든 이력이 삭제됩니다. 계속하시겠습니까?")
        if st.button("예, 삭제합니다", key="confirm_del_yes_compact", use_container_width=True):
            delete_all_history_local()
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.show_delete_confirm = False
            st.session_state.is_chat_ended = False
            st.session_state.sim_stage = "WAIT_ROLE_SELECTION"
            st.success("✅ 모든 이력이 삭제되었습니다.")
        if st.button("취소", key="confirm_del_no_compact", use_container_width=True):
            st.session_state.show_delete_confirm = False
    
    if st.session_state.get("show_reset_confirm", False):
        st.warning("⚠️ 모든 세션이 초기화됩니다. 계속하시겠습니까?")
        if st.button("예, 초기화합니다", key="confirm_reset_yes_compact", use_container_width=True):
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.initial_advice_provided = False
            st.session_state.is_chat_ended = False
            st.session_state.agent_response_area_text = ""
            st.session_state.customer_query_text_area = ""
            st.session_state.sim_stage = "WAIT_ROLE_SELECTION"
            st.session_state.show_reset_confirm = False
            st.success("✅ 모든 세션이 초기화되었습니다.")
        if st.button("취소", key="confirm_reset_no_compact", use_container_width=True):
            st.session_state.show_reset_confirm = False

