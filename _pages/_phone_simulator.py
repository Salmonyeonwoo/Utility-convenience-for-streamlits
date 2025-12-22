# ========================================
# _pages/_phone_simulator.py
# 전화 시뮬레이터 모듈
# ========================================

import streamlit as st
from lang_pack import LANG

def render_phone_simulator():
    """전화 시뮬레이터 렌더링 함수"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # 전화 시뮬레이터 세션 상태 초기화
    if "call_sim_stage" not in st.session_state:
        st.session_state.call_sim_stage = "WAITING_CALL"
    if "call_sim_mode" not in st.session_state:
        st.session_state.call_sim_mode = "INBOUND"
    if "incoming_phone_number" not in st.session_state:
        st.session_state.incoming_phone_number = ""
    if "current_call_id" not in st.session_state:
        st.session_state.current_call_id = None
    if "video_enabled" not in st.session_state:
        st.session_state.video_enabled = False
    if "is_on_hold" not in st.session_state:
        st.session_state.is_on_hold = False
    
    # 전화 시뮬레이터 렌더링 - _call_simulator.py의 render_call_simulator() 사용
    try:
        from _pages._call_simulator import render_call_simulator
        render_call_simulator()
    except ImportError:
        # _call_simulator가 없으면 _phone_tab.py 사용
        try:
            from _pages._phone_tab import render_phone_tab
            render_phone_tab()
        except ImportError:
            # 둘 다 없으면 기본 UI 표시
            st.error("전화 시뮬레이터 모듈을 찾을 수 없습니다. 필요한 파일이 있는지 확인해주세요.")
            st.info("필요한 파일: _pages/_call_simulator.py 또는 _pages/_phone_tab.py")
    except Exception as e:
        st.error(f"전화 시뮬레이터 로드 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
        # 기본 폴백 UI
        st.markdown("### 📞 전화 시뮬레이터")
        st.info("전화 시뮬레이터 기능을 사용하려면 필요한 모듈이 설치되어 있어야 합니다.")


