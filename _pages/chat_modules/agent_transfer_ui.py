# ========================================
# chat_modules/agent_transfer_ui.py
# 언어 이관 UI 모듈 (드롭다운 방식)
# ========================================

import streamlit as st
from lang_pack import LANG
from chat_modules.language_transfer import handle_language_transfer


def render_transfer_ui(L, current_lang):
    """언어 이관 UI 렌더링 (드롭다운 방식)"""
    st.markdown("---")
    st.markdown(f"**{L['transfer_header']}**")
    
    # 현재 언어에 따라 이관 옵션 및 드롭다운 텍스트 설정
    lang_options_dict = {}
    if current_lang == 'ko':
        lang_options_dict = {'en': '영어 팀원', 'ja': '일본어 팀원'}
    elif current_lang == 'en':
        lang_options_dict = {'ko': 'Korean Team Member', 'ja': 'Japanese Team Member'}
    elif current_lang == 'ja':
        lang_options_dict = {'ko': '韓国語チームメンバー', 'en': '英語チームメンバー'}
    
    # 이관 다이얼로그 표시 여부 확인
    if st.session_state.get('show_transfer_dialog', False):
        with st.expander(f"🔄 {L.get('transfer_header', '고객 이관')}", expanded=True):
            if lang_options_dict:
                selected_team_option = st.selectbox(
                    "이관할 상담원 선택:",
                    list(lang_options_dict.values()),
                    key=f"transfer_team_select_{st.session_state.sim_instance_id}"
                )
                
                selected_lang_code = None
                for lang_code, lang_name in lang_options_dict.items():
                    if lang_name == selected_team_option:
                        selected_lang_code = lang_code
                        break
                
                # 대화 요약 생성 (미리보기)
                if st.session_state.simulator_messages:
                    from simulation_handler import summarize_history_with_ai
                    with st.spinner(L.get("transfer_loading", "대화 요약 생성 중...")):
                        summary = summarize_history_with_ai(current_lang)
                        st.markdown(f"**{L.get('transfer_summary_header', '대화 요약')}:**")
                        st.info(summary[:500] + "..." if len(summary) > 500 else summary)
                
                col_transfer, col_cancel = st.columns([1, 1])
                with col_transfer:
                    if st.button(L.get("transfer_button", "이관하기"), type="primary", 
                               key=f"confirm_transfer_{st.session_state.sim_instance_id}"):
                        if selected_lang_code:
                            handle_language_transfer(selected_lang_code, st.session_state.simulator_messages, L, current_lang)
                            st.session_state.show_transfer_dialog = False
                            st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
                with col_cancel:
                    if st.button(L.get("cancel_button", "취소"), 
                               key=f"cancel_transfer_{st.session_state.sim_instance_id}"):
                        st.session_state.show_transfer_dialog = False
            else:
                st.info(L.get("no_transfer_options", "이관 가능한 팀이 없습니다."))
    else:
        if st.button(f"🔄 {L.get('transfer_header', '이관')}", 
                    key=f"show_transfer_{st.session_state.sim_instance_id}", use_container_width=True):
            st.session_state.show_transfer_dialog = True

