# ========================================
# chat_modules/language_transfer.py
# 언어 이관 처리 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import summarize_history_with_ai
from utils.history_handler import save_simulation_history_local
from llm_client import get_api_key


def handle_language_transfer(target_lang, current_messages, L, current_lang):
    """언어 이관 처리"""
    current_lang_at_start = st.session_state.language
    
    if not get_api_key("gemini"):
        st.error(L["simulation_no_key_warning"].replace('API Key', 'Gemini API Key'))
        return
    
    st.session_state.start_time = None
    
    with st.spinner(L["transfer_loading"]):
        import time
        import numpy as np
        time.sleep(np.random.uniform(5, 10))
        
        try:
            original_summary = summarize_history_with_ai(current_lang_at_start)
            
            if not original_summary or original_summary.startswith("❌"):
                history_text = ""
                for msg in current_messages:
                    role = "Customer" if msg["role"].startswith("customer") or msg["role"] == "initial_query" else "Agent"
                    if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response", "customer_closing_response"]:
                        history_text += f"{role}: {msg['content']}\n"
                original_summary = history_text
            
            from utils.translation import translate_text_with_llm
            translated_summary, is_success = translate_text_with_llm(
                original_summary, target_lang, current_lang_at_start
            )
            
            if not translated_summary:
                translated_summary = summarize_history_with_ai(target_lang)
                is_success = True if translated_summary and not translated_summary.startswith("❌") else False
            
            translated_messages = []
            for msg in current_messages:
                translated_msg = msg.copy()
                if msg["role"] in ["initial_query", "customer", "customer_rebuttal", 
                                  "agent_response", "customer_closing_response", "supervisor"]:
                    if msg.get("content"):
                        try:
                            translated_content, trans_success = translate_text_with_llm(
                                msg["content"], target_lang, current_lang_at_start
                            )
                            if trans_success:
                                translated_msg["content"] = translated_content
                        except Exception:
                            pass
                translated_messages.append(translated_msg)
            
            st.session_state.simulator_messages = translated_messages
            st.session_state.transfer_summary_text = translated_summary
            st.session_state.translation_success = is_success
            st.session_state.language_at_transfer_start = current_lang_at_start
            
            st.session_state.language = target_lang
            L_target = LANG.get(target_lang, LANG["ko"])
            
            lang_name_target = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(
                target_lang, "Korean")
            
            system_msg = L_target["transfer_system_msg"].format(target_lang=lang_name_target)
            st.session_state.simulator_messages.append(
                {"role": "system_transfer", "content": system_msg}
            )
            
            summary_msg = f"### {L_target['transfer_summary_header']}\n\n{translated_summary}"
            st.session_state.simulator_messages.append(
                {"role": "supervisor", "content": summary_msg}
            )
            
            customer_type_display = st.session_state.get("customer_type_sim_select", "")
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )
            
            st.session_state.sim_stage = "AGENT_TURN"
            # ⭐ 메시지 추가 후 즉시 화면 업데이트
            st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
            # ⭐ rerun 제거: 메시지 업데이트 트리거로 자동 반영됨
            
        except Exception as e:
            error_msg = L.get("transfer_error", "이관 처리 중 오류 발생: {error}").format(error=str(e))
            st.error(error_msg)

