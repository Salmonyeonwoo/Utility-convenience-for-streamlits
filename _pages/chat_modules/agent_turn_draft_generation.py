# ========================================
# _pages/chat_modules/agent_turn_draft_generation.py
# 에이전트 턴 - AI 응답 초안 생성 (상담원 검토 중심, 자동 무단 전송 금지)
# ========================================

import streamlit as st
from llm_client import get_api_key

def handle_auto_draft_generation(L):
    """
    AI 응답 초안 생성 처리:
    - 고객 문의 내용에 맞게 AI 응답 초안을 작성하여 입력창에 제안합니다.
    - 상담원(에이전시)이 내용을 먼저 검토하고 수정 후 직접 전송할 수 있도록 절대 자동 발송(auto-send)하지 않습니다.
    """
    has_api_key = any([
        bool(get_api_key("openai")),
        bool(get_api_key("gemini")),
        bool(get_api_key("claude")),
        bool(get_api_key("groq"))
    ])
    
    if has_api_key:
        st.session_state.is_llm_ready = True
    
    if st.session_state.is_llm_ready and st.session_state.sim_stage == "AGENT_TURN":
        # 마지막 고객 메시지 확인
        last_customer_msg = None
        last_customer_msg_idx = -1
        for idx, msg in enumerate(reversed(st.session_state.simulator_messages)):
            if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]:
                last_customer_msg = msg.get("content", "")
                last_customer_msg_idx = len(st.session_state.simulator_messages) - 1 - idx
                break

        last_draft_for_idx = st.session_state.get("last_draft_for_message_idx", -1)
        draft_already_generated = (last_draft_for_idx == last_customer_msg_idx and last_customer_msg_idx >= 0)

        # 새 고객 메시지가 있고 아직 이번 턴의 초안이 생성되지 않은 경우에만 초안 생성
        if last_customer_msg and not draft_already_generated:
            if not st.session_state.get("draft_generation_in_progress", False):
                st.session_state.draft_generation_in_progress = True
                try:
                    from simulation_handler import generate_agent_response_draft
                    session_lang = st.session_state.get("language", "ko")
                    if session_lang not in ["ko", "en", "ja"]:
                        session_lang = "ko"
                    
                    # AI 응답 초안 생성
                    draft_text = generate_agent_response_draft(session_lang)
                    
                    if draft_text and draft_text.strip():
                        # 마크다운 헤더 정리
                        draft_text_clean = draft_text
                        if "###" in draft_text_clean:
                            lines = draft_text_clean.split("\n")
                            draft_text_clean = "\n".join([line for line in lines if not line.strip().startswith("###")])
                        draft_text_clean = draft_text_clean.strip()
                        
                        if draft_text_clean:
                            # ⭐ 중요: 자동 전송(auto-send)하지 않고, 입력창에 초안으로 채워 상담사가 검토하도록 설정
                            st.session_state.agent_response_area_text = draft_text_clean
                            st.session_state.auto_generated_draft_text = draft_text_clean
                            st.session_state.auto_draft_generated = True
                            st.session_state.last_draft_for_message_idx = last_customer_msg_idx
                            st.session_state.need_auto_response_on_agent_turn = False
                            
                            print(f"✅ AI 응답 초안 생성 완료 (상담원 검토 대기): {draft_text_clean[:40]}...")
                except Exception as e:
                    st.session_state.auto_draft_generated = False
                    print(f"⚠️ AI 응답 초안 생성 오류: {e}")
                finally:
                    st.session_state.draft_generation_in_progress = False

def handle_transcript_auto_send(L):
    """음성 전사 결과 입력창 반영 (상담원 검토 후 전송)"""
    if st.session_state.get("last_transcript") and st.session_state.last_transcript:
        agent_response_auto = st.session_state.last_transcript.strip()
        if agent_response_auto:
            st.session_state.agent_response_area_text = agent_response_auto
            st.session_state.auto_generated_draft_text = agent_response_auto
            st.session_state.last_transcript = ""
