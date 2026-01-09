# ========================================
# _pages/chat_modules/agent_turn_customer_mode.py
# 에이전트 턴 - 고객 체험 모드 처리
# ========================================

import streamlit as st

def handle_customer_mode_auto_response(L, current_lang):
    """고객 체험 모드일 때 AI가 자동으로 응답 생성"""
    perspective = st.session_state.get("sim_perspective", "AGENT")
    if perspective == "CUSTOMER" and st.session_state.sim_stage == "AGENT_TURN":
        # AI가 에이전트로서 자동 응답 생성
        if st.session_state.is_llm_ready:
            # 중복 생성 방지
            if not st.session_state.get("ai_agent_response_generated", False):
                st.session_state.ai_agent_response_generated = True
                with st.spinner("🤖 AI 상담원이 답변을 작성 중입니다..."):
                    try:
                        from simulation_handler import generate_agent_response_draft
                        ai_agent_reply = generate_agent_response_draft(st.session_state.get("language", current_lang))
                        
                        if ai_agent_reply and ai_agent_reply.strip():
                            # 마크다운 헤더 제거
                            if "###" in ai_agent_reply:
                                lines = ai_agent_reply.split("\n")
                                ai_agent_reply = "\n".join([line for line in lines if not line.strip().startswith("###")])
                            ai_agent_reply = ai_agent_reply.strip()
                            
                            # 메시지 추가
                            new_msg = {"role": "agent_response", "content": ai_agent_reply, "is_auto_response": True}
                            st.session_state.simulator_messages.append(new_msg)
                            
                            # 다음 단계로 자동 이동 (사용자가 고객으로서 말할 차례)
                            st.session_state.sim_stage = "CUSTOMER_TURN"
                            st.session_state.ai_agent_response_generated = False  # 다음 응답을 위해 리셋

                    except Exception as e:
                        st.error(f"AI 응답 생성 중 오류: {e}")
                        st.session_state.ai_agent_response_generated = False
        else:
            st.info("👤 고객 입장 모드: AI 상담원이 답변을 생성하려면 LLM API Key가 필요합니다.")
        return True  # 고객 모드일 때는 기존 상담원 입력 UI를 표시하지 않음
    return False

