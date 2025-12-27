# 전화 시뮬레이터
import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
import numpy as np
# ⭐ 수정: 사용하지 않는 import를 try-except로 감싸서 import 실패해도 계속 진행
try:
    from simulation_handler import *
except ImportError:
    pass
try:
    from visualization import *
except ImportError:
    pass
try:
    from audio_handler import *
except ImportError:
    pass
try:
    from llm_client import get_api_key
except ImportError:
    pass
from typing import List, Dict, Any
import uuid
import time
import os
from PIL import Image
import io

def render_call_simulator():
    """전화 시뮬레이터 렌더링 (전화 수신, 문의 입력 포함)"""
    # ⭐ 시뮬레이션 입장 상태 초기화
    try:
        from simulation_perspective_logic import init_perspective_state, render_perspective_toggle
        init_perspective_state()
        PERSPECTIVE_LOGIC_AVAILABLE = True
    except ImportError:
        PERSPECTIVE_LOGIC_AVAILABLE = False
        if "sim_perspective" not in st.session_state:
            st.session_state.sim_perspective = "AGENT"
        if "is_auto_playing" not in st.session_state:
            st.session_state.is_auto_playing = False
    
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # ⭐ 시뮬레이션 모드 선택 UI를 탭 내부 상단에 표시
    if PERSPECTIVE_LOGIC_AVAILABLE:
        render_perspective_toggle(L)
    
    # ⭐ 고객 모드일 경우 별도 렌더링
    if st.session_state.get("sim_perspective", "AGENT") == "CUSTOMER":
        try:
            from _pages._call_customer_mode import render_call_customer_mode
            render_call_customer_mode()
            return
        except ImportError:
            st.warning("고객 모드 전화 시뮬레이터 모듈을 찾을 수 없습니다. 에이전트 모드로 전환합니다.")
            st.session_state.sim_perspective = "AGENT"
    
    # AHT 타이머는 streamlit_app.py의 우측 상단에서 표시됨 (제거됨)

    # ========================================
    # 화면 구분 (애니메이션 / CC)
    # ========================================
    # ⭐ 수정: 왼쪽 비디오 업로드 섹션 제거, col_cc만 사용
    col_cc = st.columns([1])[0]

    with col_cc:
        # ⭐ 수정: "전화 수신 중" 메시지는 _call_in_call.py에서 표시하므로 여기서는 제거
        # (중복 표시 방지)
        pass

    # ⭐ 왼쪽 비디오 섹션 제거 (비디오 업로드 내용은 상대방 화면 밑으로 이동)

    # ⭐ col_cc는 위에서 이미 처리됨

    # ========================================
    # WAITING / RINGING 상태 - 전화 수신, 문의 입력 포함
    # ========================================
    try:
        # IDLE 상태도 WAITING_CALL로 처리 (초기 상태)
        if st.session_state.call_sim_stage in ["WAITING_CALL", "RINGING", "IDLE", None]:
            # IDLE이나 None이면 WAITING_CALL로 초기화
            if st.session_state.call_sim_stage in ["IDLE", None]:
                st.session_state.call_sim_stage = "WAITING_CALL"
            
            # _call_waiting 모듈 사용 (중복 제거)
            try:
                from _pages._call_waiting import render_call_waiting
                render_call_waiting()
            except Exception as e:
                st.error(f"❌ _call_waiting 로드 오류: {e}")
                import traceback
                st.code(traceback.format_exc())

        # ------------------
        # IN_CALL 상태 (통화 중)
        # ------------------
        elif st.session_state.call_sim_stage == "IN_CALL":
            try:
                from _pages._call_in_call import render_call_in_call
                render_call_in_call()
            except Exception as e:
                st.error(f"❌ _call_in_call 로드 오류: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.info("📞 통화 중입니다...")
                if st.button("📴 통화 종료", use_container_width=True, type="primary"):
                    # ⭐ 수정: 통화 시간 계산 및 저장
                    from datetime import datetime
                    call_duration = 0
                    if st.session_state.get("start_time"):
                        call_duration = (datetime.now() - st.session_state.start_time).total_seconds()
                        st.session_state.call_duration = call_duration  # 통화 시간 저장
                    
                    st.session_state.call_sim_stage = "CALL_ENDED"
                    st.session_state.call_active = False
                    st.session_state.start_time = None
        
        elif st.session_state.call_sim_stage == "CALL_ENDED":
            try:
                from _pages._call_ended import render_call_ended
                render_call_ended()
            except ImportError:
                # call_ended 모듈이 없으면 기본 종료 화면 표시
                # ⭐ 수정: 통화 시간 표시 (몇 분 몇 초 형식)
                call_duration = st.session_state.get("call_duration", 0)
                minutes = int(call_duration // 60)
                seconds = int(call_duration % 60)
                if minutes > 0:
                    duration_msg = L.get("call_ended_with_duration", "통화가 종료되었습니다. (통화 시간: {minutes}분 {seconds}초)").format(minutes=minutes, seconds=seconds)
                else:
                    duration_msg = L.get("call_ended_with_seconds", "통화가 종료되었습니다. (통화 시간: {seconds}초)").format(seconds=seconds)
                st.success(duration_msg)
                if st.button(L.get("new_call_button", "새 통화 시작"), key="btn_new_call"):
                    # ⭐ 수정: 새 통화 시작 시 모든 통화 관련 상태 완전 초기화
                    st.session_state.call_sim_stage = "WAITING_CALL"
                    st.session_state.call_messages = []
                    st.session_state.inquiry_text = ""
                    st.session_state.call_content = ""
                    st.session_state.incoming_phone_number = None
                    st.session_state.incoming_call = None
                    st.session_state.call_active = False
                    st.session_state.start_time = None
                    st.session_state.call_duration = None
                    st.session_state.transfer_summary_text = ""
                    st.session_state.language_at_transfer_start = None
                    st.session_state.is_on_hold = False
                    st.session_state.hold_start_time = None
                    st.session_state.hold_total_seconds = 0
                    st.session_state.provider_call_active = False
                    st.session_state.call_direction = "inbound"
            except Exception as e:
                st.error(f"❌ _call_ended 로드 오류: {e}")
                import traceback
                st.code(traceback.format_exc())
                # ⭐ 수정: 통화 시간 표시 (몇 분 몇 초 형식)
                call_duration = st.session_state.get("call_duration", 0)
                minutes = int(call_duration // 60)
                seconds = int(call_duration % 60)
                if minutes > 0:
                    duration_msg = L.get("call_ended_with_duration", "통화가 종료되었습니다. (통화 시간: {minutes}분 {seconds}초)").format(minutes=minutes, seconds=seconds)
                else:
                    duration_msg = L.get("call_ended_with_seconds", "통화가 종료되었습니다. (통화 시간: {seconds}초)").format(seconds=seconds)
                st.success(duration_msg)
                if st.button(L.get("new_call_button", "새 통화 시작"), key="btn_new_call_fallback"):
                    # ⭐ 수정: 새 통화 시작 시 모든 통화 관련 상태 완전 초기화
                    st.session_state.call_sim_stage = "WAITING_CALL"
                    st.session_state.call_messages = []
                    st.session_state.inquiry_text = ""
                    st.session_state.call_content = ""
                    st.session_state.incoming_phone_number = None
                    st.session_state.incoming_call = None
                    st.session_state.call_active = False
                    st.session_state.start_time = None
                    st.session_state.call_duration = None
                    st.session_state.transfer_summary_text = ""
                    st.session_state.language_at_transfer_start = None
                    st.session_state.is_on_hold = False
                    st.session_state.hold_start_time = None
                    st.session_state.hold_total_seconds = 0
                    st.session_state.provider_call_active = False
                    st.session_state.call_direction = "inbound"
        else:
            # 알 수 없는 상태일 때 WAITING_CALL로 초기화하고 전화 수신 화면 표시
            st.session_state.call_sim_stage = "WAITING_CALL"
            try:
                from _pages._call_waiting import render_call_waiting
                render_call_waiting()
            except Exception as e:
                st.error(f"❌ _call_waiting 로드 오류: {e}")
                import traceback
                st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"❌ 전화 시뮬레이터 렌더링 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
        # 기본 폴백: _call_waiting 모듈 사용
        try:
            from _pages._call_waiting import render_call_waiting
            render_call_waiting()
        except:
            st.info("전화 시뮬레이터를 초기화할 수 없습니다.")

