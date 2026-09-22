# ========================================
# _pages/chat_modules/agent_turn_language_transfer.py
# 에이전트 턴 - 언어 이관 처리 (영어/일본어/한국어 팀 즉시 전환)
# ========================================

import streamlit as st
from lang_pack import LANG
from utils.translation import translate_text_with_llm
from utils.history_handler import save_simulation_history_local

def render_language_transfer(L, current_lang):
    """언어 이관 버튼 영역 (영어/일본어/한국어 팀 전용)"""
    st.markdown("---")
    st.markdown(f"**🌐 {L.get('transfer_header', '언어 이관 요청 (다른 팀)')}**")

    languages = ["en", "ja", "ko"]
    transfer_cols = st.columns(len(languages))

    def transfer_session(target_lang, current_messages):
        """세션 언어 이관 및 메시지 전체 번역"""
        current_lang_at_start = st.session_state.get("language", "ko")
        if target_lang == current_lang_at_start:
            st.info(f"이미 {target_lang.upper()} 언어로 상담 중입니다.")
            return

        with st.spinner(f"🌐 {target_lang.upper()} 팀으로 이관 및 대화 내역 번역 중..."):
            translated_messages = []
            for msg in current_messages:
                translated_msg = msg.copy()
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # 시스템 및 일반 메시지 번역
                if content and role in ["customer", "initial_query", "agent_response", "customer_rebuttal", "supervisor"]:
                    try:
                        trans_text, success = translate_text_with_llm(content, target_lang, current_lang_at_start)
                        translated_msg["content"] = trans_text
                    except Exception:
                        pass
                translated_messages.append(translated_msg)

            # 언어 및 메시지 상태 전환
            st.session_state.language = target_lang
            L_target = LANG.get(target_lang, LANG["ko"])

            lang_team_names = {
                "en": "US English Support Team",
                "ja": "JP 日本語サポートチーム",
                "ko": "KR 한국어 지원 팀"
            }
            system_notice = {
                "en": "📞 System: Chat session transferred to US English Support Team. All communication is now in English.",
                "ja": "📞 システム: JP日本語サポートチームへ移管されました。これ以降の対応は日本語で行われます。",
                "ko": "📞 시스템: KR 한국어 지원 팀으로 이관되었습니다. 모든 상담이 한국어로 진행됩니다."
            }

            translated_messages.append({
                "role": "system_transfer",
                "content": system_notice.get(target_lang, f"Session transferred to {target_lang} team.")
            })

            st.session_state.simulator_messages = translated_messages
            st.session_state.language_transfer_requested = False
            
            # 저장
            save_simulation_history_local(
                st.session_state.get("customer_query_text_area", ""),
                st.session_state.get("customer_type_sim_select", "") + f" (Transferred to {target_lang})",
                st.session_state.simulator_messages,
                is_chat_ended=False,
                attachment_context=st.session_state.get("sim_attachment_context_for_llm", "")
            )
            
            st.rerun()

    for idx, lang_code in enumerate(languages):
        lang_labels = {
            "en": "🇺🇸 US 영어 팀으로 이관",
            "ja": "🇯🇵 JP 일본어 팀으로 이관",
            "ko": "🇰🇷 KR 한국어 팀으로 이관"
        }
        transfer_label = lang_labels.get(lang_code, f"{lang_code.upper()} 팀으로 이관")

        with transfer_cols[idx]:
            if st.button(
                    transfer_label,
                    key=f"btn_transfer_{lang_code}_{st.session_state.get('sim_instance_id', 'default')}",
                    use_container_width=True):
                transfer_session(lang_code, st.session_state.get("simulator_messages", []))
