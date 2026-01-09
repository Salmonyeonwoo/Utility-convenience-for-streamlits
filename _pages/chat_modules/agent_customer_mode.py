# ========================================
# chat_modules/agent_customer_mode.py
# 고객 모드 자동 응답 처리 모듈
# ========================================

import streamlit as st


def handle_customer_mode_auto_response(L, current_lang):
    """고객 모드 자동 응답 처리"""
    if st.session_state.is_llm_ready:
        if not st.session_state.get("ai_agent_response_generated", False):
            st.session_state.ai_agent_response_generated = True
            with st.spinner("🤖 AI 상담원이 답변을 작성 중입니다..."):
                try:
                    from simulation_handler import generate_agent_response_draft
                    ai_agent_reply = generate_agent_response_draft(
                        st.session_state.get("language", current_lang))
                    
                    if ai_agent_reply and ai_agent_reply.strip():
                        if "###" in ai_agent_reply:
                            lines = ai_agent_reply.split("\n")
                            ai_agent_reply = "\n".join(
                                [line for line in lines if not line.strip().startswith("###")])
                        ai_agent_reply = ai_agent_reply.strip()
                        
                        new_msg = {"role": "agent_response", "content": ai_agent_reply, "is_auto_response": True}
                        st.session_state.simulator_messages.append(new_msg)
                        # ⭐ 메시지 추가 후 즉시 화면 업데이트
                        st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
                        st.session_state.sim_stage = "CUSTOMER_TURN"
                        st.session_state.ai_agent_response_generated = False
                        return True
                except Exception as e:
                    st.error(f"AI 응답 생성 중 오류: {e}")
                    st.session_state.ai_agent_response_generated = False
    else:
        st.info("👤 고객 입장 모드: AI 상담원이 답변을 생성하려면 LLM API Key가 필요합니다.")
    return False

