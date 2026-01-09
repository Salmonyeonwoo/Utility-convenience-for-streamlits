# ========================================
# _pages/chat_modules/agent_turn_guidelines.py
# 에이전트 턴 - 가이드라인 및 정보 표시
# ========================================

import streamlit as st
from utils.history_handler import (
    generate_chat_summary, load_simulation_histories_local,
    recommend_guideline_for_customer
)

def render_guidelines_and_info(L):
    """고객 성향 기반 가이드라인 추천 및 정보 표시"""
    # 고객 성향 기반 가이드라인 추천
    if st.session_state.simulator_messages and len(
            st.session_state.simulator_messages) >= 2:
        try:
            temp_summary = generate_chat_summary(
                st.session_state.simulator_messages,
                st.session_state.customer_query_text_area,
                st.session_state.get("customer_type_sim_select", ""),
                st.session_state.language
            )

            if temp_summary and temp_summary.get("customer_sentiment_score"):
                all_histories = load_simulation_histories_local(
                    st.session_state.language)

                recommended_guideline = recommend_guideline_for_customer(
                    temp_summary,
                    all_histories,
                    st.session_state.language
                )

                if recommended_guideline:
                    with st.expander("💡 고객 성향 기반 응대 가이드라인 추천", expanded=False):
                        st.markdown(recommended_guideline)
                        st.caption(
                            "💡 이 가이드는 유사한 과거 고객 사례를 분석하여 자동 생성되었습니다.")
        except Exception:
            pass

    # 언어 이관 요청 강조 표시
    if st.session_state.language_transfer_requested:
        st.error(
            L.get(
                "language_transfer_requested_msg",
                "🚨 고객이 언어 전환(이관)을 요청했습니다. 즉시 응대하거나 이관을 진행하세요。"))

    # 고객 첨부 파일 정보 재표시
    if st.session_state.sim_attachment_context_for_llm:
        st.info(
            f"📎 최초 문의 시 첨부된 파일 정보:\n\n{st.session_state.sim_attachment_context_for_llm.replace('[ATTACHMENT STATUS]', '').strip()}")

