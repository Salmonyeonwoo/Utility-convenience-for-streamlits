# 전화 통화 오디오 UI
# ⚠️ 이 파일은 복원이 필요합니다. 현재는 임시 stub 파일입니다.

import streamlit as st
from lang_pack import LANG

def render_audio_call_ui():
    """오디오 통화 UI 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    st.warning("⚠️ phone_call_audio.py 파일이 삭제되어 복원이 필요합니다. 원본 파일을 복원해주세요.")


