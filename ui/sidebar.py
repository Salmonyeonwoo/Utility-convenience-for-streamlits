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
            st.session_state.language = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        
        st.title("💬 앱 설정")
        
        # 언어 선택 (app.py 스타일 유지)
        st.subheader("언어 선택")
        lang_codes = ["ko", "en", "ja"]
        lang_names = {"ko": "한국어", "en": "English", "ja": "日本語"}

        # 다른 모듈에서 st.session_state.language를 변경해도, 셀렉터가 따라오도록 동기화
        if st.session_state.get("language_selector") != current_lang:
            st.session_state["language_selector"] = current_lang

        def _on_language_selector_change():
            selected = st.session_state.get("language_selector", "ko")
            if selected not in lang_codes:
                selected = "ko"
            if st.session_state.get("language") != selected:
                st.session_state.language = selected

        st.selectbox(
            "언어 선택",
            lang_codes,
            key="language_selector",
            format_func=lambda c: lang_names.get(c, c),
            label_visibility="collapsed",
            on_change=_on_language_selector_change,
        )
        
        st.divider()
        
        # 기능 선택 (app.py 스타일 - 참고용 구조 추가)
        st.subheader("기능 선택")
        feature_ids = ["home", "chat_email", "phone", "customer_data_inquiry"]
        feature_labels = {
            "home": L.get("home_tab", "홈"),
            "chat_email": L.get("chat_email_tab", "채팅/이메일"),
            "phone": L.get("phone_tab", "전화"),
            "customer_data_inquiry": L.get("customer_data_inquiry_tab", "고객 데이터 조회"),
        }

        # 기존 레거시(feature_selection: 표시 문자열) 상태가 있으면 ID로 승격
        if "feature_selection_id" not in st.session_state:
            legacy_label = st.session_state.get("feature_selection")
            label_to_id = {v: k for k, v in feature_labels.items()}
            st.session_state.feature_selection_id = label_to_id.get(legacy_label, "home")

        current_feature_id = st.session_state.get("feature_selection_id", "home")
        if current_feature_id not in feature_ids:
            current_feature_id = "home"
            st.session_state.feature_selection_id = "home"

        # 언어 변경 시에도 레거시 라벨(feature_selection)을 현재 언어 라벨로 동기화
        desired_label = feature_labels.get(current_feature_id, feature_labels["home"])
        if st.session_state.get("feature_selection") != desired_label:
            st.session_state.feature_selection = desired_label

        # 라디오 위젯 키는 별도로 두어(=feature_selector_id) 다른 모듈과 충돌을 피함
        if st.session_state.get("feature_selector_id") != current_feature_id:
            st.session_state["feature_selector_id"] = current_feature_id

        def _on_feature_selector_change():
            selected_id = st.session_state.get("feature_selector_id", "home")
            if selected_id not in feature_ids:
                selected_id = "home"
                st.session_state["feature_selector_id"] = "home"
            st.session_state.feature_selection_id = selected_id
            st.session_state.feature_selection = feature_labels.get(selected_id, feature_labels["home"])

        st.radio(
            "기능 선택",
            feature_ids,
            key="feature_selector_id",
            format_func=lambda fid: feature_labels.get(fid, fid),
            on_change=_on_feature_selector_change,
        )
        
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

        st.divider()

        # ========================================
        # 디버그/성능(LLM 호출 로깅)
        # ========================================
        with st.expander("🧪 디버그/성능", expanded=False):
            st.checkbox("LLM 호출 로깅(턴/리런/소요시간)", key="telemetry_llm_enabled")

            events = st.session_state.get("llm_call_events", [])
            if st.session_state.get("telemetry_llm_enabled"):
                st.caption(
                    f"rerun_seq={st.session_state.get('rerun_seq')} · "
                    f"sim_stage={st.session_state.get('sim_stage')} · "
                    f"events={len(events)}"
                )
                if st.button("LLM 로그 지우기", key="clear_llm_call_events"):
                    st.session_state.llm_call_events = []
                    events = []

                if events:
                    last_turn_key = events[-1].get("turn_key")
                    last_turn_events = [e for e in events if e.get("turn_key") == last_turn_key] if last_turn_key else []
                    if last_turn_events:
                        total = len(last_turn_events)
                        total_ms = sum(int(e.get("dur_ms") or 0) for e in last_turn_events)
                        st.write(f"**최근 턴({last_turn_key})**: {total}회 호출 · 총 {total_ms}ms")

                    st.write("**최근 10개 호출(최신순)**")
                    for e in reversed(events[-10:]):
                        st.caption(
                            f"[rerun={e.get('rerun_seq')}] "
                            f"{e.get('tag') or '-'} · "
                            f"{e.get('provider')}/{e.get('model')} · "
                            f"{e.get('status')} · "
                            f"{e.get('dur_ms')}ms · "
                            f"turn={e.get('turn_key')}"
                        )
                        if e.get("status") == "error" and e.get("error"):
                            st.caption(f"  - err: {e.get('error')}")
            else:
                st.caption("체크를 켠 상태에서 느려지는 동작을 1~2번 재현하면, 여기서 LLM 호출 횟수/지연을 바로 볼 수 있어요.")
