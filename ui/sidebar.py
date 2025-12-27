# ========================================
# ui/sidebar.py
# 사이드바 UI 컴포넌트 모듈 (app.py 구조 복원)
# ========================================

import streamlit as st
from lang_pack import LANG

try:
    from llm_client import get_api_key
except ImportError:
    get_api_key = None

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
        
        # 현재 언어에 해당하는 표시 이름 찾기
        current_lang_display = None
        for display_name, lang_code in lang_options.items():
            if lang_code == current_lang:
                current_lang_display = display_name
                break
        
        # 현재 언어가 없으면 기본값으로 설정
        if current_lang_display is None:
            current_lang_display = lang_display_names[0]
            st.session_state.language = lang_options[current_lang_display]
        
        # 현재 언어에 맞는 인덱스 찾기
        try:
            current_index = lang_display_names.index(current_lang_display)
        except ValueError:
            current_index = 0
        
        selected_lang_display = st.selectbox(
            "언어 선택",
            lang_display_names,
            index=current_index,
            key="language_selector",
            label_visibility="collapsed"
        )
        
        selected_lang_code = lang_options[selected_lang_display]
        if selected_lang_code != current_lang:
            st.session_state.language = selected_lang_code
            # ⭐ 수정: rerun 제거 - 언어 변경은 세션 상태에 저장되어 다음 렌더링에서 자동 반영됨
            # st.rerun()  # 언어 변경 시 즉시 반영
        
        st.divider()
        
        # 기능 선택 (app.py 스타일 - 참고용 구조 추가)
        st.subheader("기능 선택")
        feature_options = [
            L.get("home_tab", "홈"),
            L.get("chat_email_tab", "채팅/이메일"),
            L.get("phone_tab", "전화"),
            L.get("customer_data_inquiry_tab", "고객 데이터 조회")
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
            # st.rerun()  # 주석 처리: Streamlit이 자동으로 rerun함
        
        st.divider()
        
        # API Key 상태 표시
        st.subheader("🔑 API Key 상태")
        if get_api_key:
            # 환경변수 직접 확인 (대소문자 변형 포함)
            import os
            openai_key = get_api_key("openai") or os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key") or ""
            gemini_key = get_api_key("gemini") or os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini_api_key") or ""
            claude_key = get_api_key("claude") or os.environ.get("CLAUDE_API_KEY") or os.environ.get("claude_api_key") or ""
            groq_key = get_api_key("groq") or os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key") or ""
            
            api_status = []
            if openai_key:
                api_status.append("✅ OpenAI")
            if gemini_key:
                api_status.append("✅ Gemini")
            if claude_key:
                api_status.append("✅ Claude")
            if groq_key:
                api_status.append("✅ Groq")
            
            if api_status:
                st.success(f"감지된 API Keys: {', '.join([s.replace('✅ ', '') for s in api_status])}")
            else:
                st.error("⚠️ API Key가 감지되지 않았습니다.")
                st.caption("환경변수 또는 .streamlit/secrets.toml에 API Key를 설정하세요.")
                with st.expander("API Key 설정 방법"):
                    st.markdown("""
                    **방법 1: 환경변수 설정**
                    ```bash
                    export OPENAI_API_KEY="your-key"
                    export GEMINI_API_KEY="your-key"
                    ```
                    
                    **방법 2: Streamlit Secrets**
                    `.streamlit/secrets.toml` 파일 생성:
                    ```toml
                    OPENAI_API_KEY = "your-key"
                    GEMINI_API_KEY = "your-key"
                    ```
                    """)
        else:
            st.warning("API Key 확인 모듈을 불러올 수 없습니다.")
