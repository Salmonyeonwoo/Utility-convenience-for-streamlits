# ========================================
# chat_modules/customer_draft_trigger.py
# 고객 메시지 수신 시 응대 초안 생성 트리거 모듈
# ========================================

import streamlit as st
from llm_client import get_api_key


def trigger_draft_generation():
    """응대 초안 생성 트리거"""
    has_api_key = any([
        bool(get_api_key("openai")),
        bool(get_api_key("gemini")),
        bool(get_api_key("claude")),
        bool(get_api_key("groq"))
    ])
    
    if has_api_key:
        st.session_state.is_llm_ready = True
        
        if not st.session_state.get("draft_generation_in_progress", False):
            st.session_state.draft_generation_in_progress = True
            try:
                from simulation_handler import generate_agent_response_draft
                session_lang = st.session_state.get("language", "ko")
                if session_lang not in ["ko", "en", "ja"]:
                    session_lang = "ko"
                
                draft_text = generate_agent_response_draft(session_lang)
                
                if draft_text and draft_text.strip():
                    if "###" in draft_text:
                        lines = draft_text.split("\n")
                        draft_text = "\n".join([line for line in lines if not line.strip().startswith("###")])
                    draft_text = draft_text.strip()
                    
                    if draft_text:
                        st.session_state.agent_response_area_text = draft_text
                        st.session_state.auto_draft_generated = True
                        st.session_state.auto_generated_draft_text = draft_text
                        st.session_state.last_draft_for_message_idx = len(st.session_state.simulator_messages) - 1
                        
                        print(f"✅ 고객 메시지 수신 시 응대 초안 생성 완료")
            except Exception as e:
                print(f"❌ 응대 초안 자동 생성 오류: {e}")
                st.session_state.auto_draft_generated = False
            finally:
                st.session_state.draft_generation_in_progress = False
    else:
        st.session_state.auto_draft_generated = False
        st.session_state.auto_generated_draft_text = ""
        st.session_state.last_draft_for_message_idx = -1

