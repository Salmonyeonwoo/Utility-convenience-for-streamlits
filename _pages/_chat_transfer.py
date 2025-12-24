# ========================================
# _pages/_chat_transfer.py
# 채팅 시뮬레이터 - 이관 요약 렌더링
# ========================================

import streamlit as st
from utils.translation import translate_text_with_llm


def render_transfer_summary(L, current_lang):
    """이관 요약 표시"""
    st.markdown("---")
    st.markdown(f"**{L['transfer_summary_header']}**")
    st.info(L["transfer_summary_intro"])

    is_translation_failed = not st.session_state.get("translation_success", True) or not st.session_state.transfer_summary_text

    if st.session_state.transfer_summary_text and st.session_state.get("translation_success", True):
        st.markdown(st.session_state.transfer_summary_text)

    if is_translation_failed:
        if st.session_state.transfer_summary_text:
            st.info(st.session_state.transfer_summary_text)
        if st.button(
                L.get("button_retry_translation", "번역 다시 시도"),
                key=f"btn_retry_translation_{st.session_state.sim_instance_id}"):
            _handle_retry_translation(L)


def _handle_retry_translation(L):
    """번역 재시도 처리"""
    try:
        source_lang = st.session_state.language_at_transfer_start
        target_lang = st.session_state.language

        if not source_lang or not target_lang:
            st.error(L.get("invalid_language_info", "언어 정보가 올바르지 않습니다."))
        else:
            history_text = _build_history_text()

            if not history_text.strip():
                st.warning(L.get("no_content_to_translate", "번역할 대화 내용이 없습니다."))
            else:
                with st.spinner(L.get("transfer_loading", "번역 중...")):
                    translated_summary, is_success = translate_text_with_llm(
                        history_text, target_lang, source_lang)

                    if not translated_summary:
                        st.warning(L.get("translation_empty", "번역 결과가 비어있습니다. 원본 텍스트를 사용합니다."))
                        translated_summary = history_text
                        is_success = False

                    translated_messages = _translate_all_messages(target_lang, source_lang)

                    st.session_state.simulator_messages = translated_messages
                    st.session_state.transfer_summary_text = translated_summary
                    st.session_state.translation_success = is_success
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(L.get("translation_retry_error", "번역 재시도 중 오류 발생: {error}").format(error=str(e)))
        st.code(error_details)
        st.session_state.transfer_summary_text = L.get("translation_error", "번역 오류: {error}").format(error=str(e))
        st.session_state.translation_success = False


def _build_history_text():
    """대화 이력 텍스트 생성"""
    history_text = ""
    for msg in st.session_state.simulator_messages:
        role = "Customer" if msg["role"].startswith("customer") or msg["role"] == "initial_query" else "Agent"
        if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response", "customer_closing_response"]:
            content = msg.get("content", "").strip()
            if content:
                history_text += f"{role}: {content}\n"
    return history_text


def _translate_all_messages(target_lang, source_lang):
    """모든 메시지 번역"""
    translated_messages = []
    for msg in st.session_state.simulator_messages:
        translated_msg = msg.copy()
        if msg["role"] in ["initial_query", "customer", "customer_rebuttal", "agent_response", "customer_closing_response", "supervisor"]:
            if msg.get("content"):
                try:
                    from utils.translation import translate_text_with_llm
                    translated_content, trans_success = translate_text_with_llm(
                        msg["content"], target_lang, source_lang)
                    if trans_success:
                        translated_msg["content"] = translated_content
                except Exception:
                    pass
        translated_messages.append(translated_msg)
    return translated_messages


