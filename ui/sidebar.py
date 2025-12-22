# ========================================
# ui/sidebar.py
# 사이드바 UI 컴포넌트 모듈 (app.py 구조 복원)
# ========================================

import streamlit as st
from lang_pack import LANG

try:
    from admin import AdminManager
    admin_manager = AdminManager()
except ImportError:
    admin_manager = None


def render_sidebar():
    """사이드바 UI 렌더링 (app.py 스타일 레이아웃 참고)"""
    with st.sidebar:
        # 언어 설정 초기화
        if "language" not in st.session_state:
            st.session_state.language = "ko"
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        
        st.title("💬 앱 설정")
        
        # 언어 선택 (app.py 스타일 유지)
        st.subheader("언어 선택")
        lang_options = {
            "한국어": "ko",
            "English": "en",
            "日本語": "ja"
        }
        lang_display_names = list(lang_options.keys())
        current_lang_display = None
        for display_name, lang_code in lang_options.items():
            if lang_code == current_lang:
                current_lang_display = display_name
                break
        if current_lang_display is None:
            current_lang_display = lang_display_names[0]
        
        selected_lang_display = st.selectbox(
            "언어 선택",
            lang_display_names,
            index=lang_display_names.index(current_lang_display),
            key="language_selector",
            label_visibility="collapsed"
        )
        
        selected_lang_code = lang_options[selected_lang_display]
        if selected_lang_code != current_lang:
            st.session_state.language = selected_lang_code
        
        st.divider()
        
        # 기능 선택 (app.py 스타일 - 라디오 버튼)
        st.subheader("기능 선택")
        feature_options = [
            L.get("sim_tab_chat_email", "AI 고객 응대 시뮬레이터 (채팅/이메일)"),
            L.get("sim_tab_phone", "AI 고객 응대 시뮬레이터 (전화)"),
            L.get("company_info_tab", "회사 정보 및 FAQ"),
            L.get("rag_tab", "RAG 지식 챗봇"),
            L.get("content_tab", "맞춤형 학습 콘텐츠 생성"),
            L.get("lstm_tab", "LSTM 성취도 예측 대시보드"),
            L.get("voice_rec_header", "음성 기록 & 관리")
        ]
        
        current_feature = st.session_state.get("feature_selection", feature_options[0])
        feature_index = 0
        for idx, opt in enumerate(feature_options):
            if opt == current_feature:
                feature_index = idx
                break
        
        selected_feature = st.radio(
            "기능 선택",
            feature_options,
            key="feature_selector",
            index=feature_index
        )
        
        if selected_feature != current_feature:
            st.session_state.feature_selection = selected_feature
