"""
app.py의 사이드바 렌더링 로직
"""

import streamlit as st

def render_operator_sidebar():
    """상담원 사이드바 렌더링 (Chatstack 스타일 - 네비게이션만)"""
    with st.sidebar:
        st.markdown("### 💬")
        
        # 네비게이션 아이콘 (간단한 버전)
        if st.button("🏠", key="nav_home_icon", use_container_width=True, help="홈"):
            st.session_state.current_page = 'home'
        
        if st.button("💬", key="nav_chat_icon", use_container_width=True, help="채팅"):
            st.session_state.current_page = 'chat'
        
        if st.button("📞", key="nav_call_icon", use_container_width=True, help="전화"):
            st.session_state.current_page = 'call'
        
        if st.button("📋", key="nav_customer_data_icon", use_container_width=True, help="고객 데이터"):
            st.session_state.current_page = 'customer_data'
        
        st.markdown("---")
        
        # 언어 선택
        selected_language = st.radio(
            "언어",
            ["한국어", "English", "日本語"],
            index=["한국어", "English", "日本語"].index(
                {"ko": "한국어", "en": "English", "ja": "日本語"}.get(st.session_state.language, "한국어")
            ),
            key="language_select"
        )
        lang_map = {"한국어": "ko", "English": "en", "日本語": "ja"}
        if lang_map[selected_language] != st.session_state.language:
            st.session_state.language = lang_map[selected_language]
        
        st.markdown("---")
        
        # 상담원 프로필
        st.markdown("**상담원**")
        st.markdown("🟢 온라인")
        
        if st.button("🔄 모드 변경", use_container_width=True):
            st.session_state.user_type = None
            st.session_state.current_page = None
        
        st.session_state.auto_refresh = st.checkbox("🔄 자동 새로고침", value=st.session_state.auto_refresh)





