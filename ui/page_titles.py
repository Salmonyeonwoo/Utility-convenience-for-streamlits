# ========================================
# ui/page_titles.py
# 페이지 타이틀 표시 모듈
# ========================================

import streamlit as st
from lang_pack import LANG


def render_page_title():
    """페이지 타이틀과 설명 렌더링"""
    # 언어 키 안전하게 가져오기
    if "language" not in st.session_state:
        st.session_state.language = "ko"
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])

    # ⭐ 타이틀과 설명을 한 줄로 간결하게 표시
    feature_selection = st.session_state.get("feature_selection", L["sim_tab_chat_email"])
    if feature_selection == L["sim_tab_chat_email"]:
        st.markdown(f"### 📧 {L['sim_tab_chat_email']}")
        st.caption(L['sim_tab_chat_email_desc'])
    elif feature_selection == L["sim_tab_phone"]:
        st.markdown(f"### 📞 {L['sim_tab_phone']}")
        st.caption(L['sim_tab_phone_desc'])
    elif feature_selection == L["rag_tab"]:
        st.markdown(f"### 📚 {L['rag_tab']}")
        st.caption(L['rag_tab_desc'])
    elif feature_selection == L["content_tab"]:
        st.markdown(f"### 📝 {L['content_tab']}")
        st.caption(L['content_tab_desc'])
    elif feature_selection == L["lstm_tab"]:
        st.markdown(f"### 📊 {L['lstm_tab']}")
        st.caption(L['lstm_tab_desc'])
    elif feature_selection == L["voice_rec_header"]:
        st.markdown(f"### 🎤 {L['voice_rec_header']}")
        st.caption(L['voice_rec_header_desc'])
    elif feature_selection == L["company_info_tab"]:
        # 공백 축소: 제목과 설명을 한 줄로 간결하게 표시
        st.markdown(f"#### 📋 {L['company_info_tab']}")
        st.caption(L['company_info_tab_desc'])


