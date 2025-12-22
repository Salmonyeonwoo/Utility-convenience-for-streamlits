# ========================================
# _pages/_session_init.py
# 세션 상태 초기화 모듈
# ========================================

import streamlit as st
from datetime import datetime, timedelta
import uuid
import hashlib
from lang_pack import LANG, DEFAULT_LANG as LANG_DEFAULT
from config import SUPPORTED_APIS, DEFAULT_LANG
from llm_client import get_api_key, get_llm_client, init_openai_audio_client
from _pages._classes import CallHandler, AppAudioHandler, CustomerDataManager

def init_session_state():
    """세션 상태 초기화 함수"""
    if "current_call_id" not in st.session_state:

            st.session_state.current_call_id = None
    if "video_enabled" not in st.session_state:

            st.session_state.video_enabled = False
    if "opponent_video_frames" not in st.session_state:

            st.session_state.opponent_video_frames = []  # 상대방 비디오 프레임 저장
    if "last_camera_frame" not in st.session_state:

            st.session_state.last_camera_frame = None

    # ⭐ 전화 기능 관련 상태 추가
    if "call_sim_stage" not in st.session_state:
        # WAITING_CALL, RINGING, IN_CALL, CALL_ENDED
        st.session_state.call_sim_stage = "WAITING_CALL"
    if "call_sim_mode" not in st.session_state:

            st.session_state.call_sim_mode = "INBOUND"  # INBOUND or OUTBOUND
    if "incoming_phone_number" not in st.session_state:

            st.session_state.incoming_phone_number = "+82 10-1234-5678"
    if "is_on_hold" not in st.session_state:

            st.session_state.is_on_hold = False
    if "hold_start_time" not in st.session_state:

            st.session_state.hold_start_time = None
    if "total_hold_duration" not in st.session_state:

            st.session_state.total_hold_duration = timedelta(0)
    if "current_customer_audio_text" not in st.session_state:

            st.session_state.current_customer_audio_text = ""
    if "current_agent_audio_text" not in st.session_state:

            st.session_state.current_agent_audio_text = ""
    if "agent_response_input_box_widget_call" not in st.session_state:  # 전화 탭 전용 입력창
        st.session_state.agent_response_input_box_widget_call = ""
    if "call_initial_query" not in st.session_state:  # 전화 탭 전용 초기 문의
        st.session_state.call_initial_query = ""
    if "call_website_url" not in st.session_state:  # 전화 탭 전용 홈페이지 주소
        st.session_state.call_website_url = ""
    # ⭐ 추가: 통화 요약 및 초기 고객 음성 저장소
    if "call_summary_text" not in st.session_state:

            st.session_state.call_summary_text = ""
    # 고객의 첫 음성 (TTS 결과) 저장
    if "customer_initial_audio_bytes" not in st.session_state:

            st.session_state.customer_initial_audio_bytes = None
    if "last_recorded_audio_bytes" not in st.session_state:  # 마지막 녹음된 오디오 (재생용)
        st.session_state.last_recorded_audio_bytes = None
    if "last_customer_audio_bytes" not in st.session_state:  # 마지막 고객 응답 오디오 (재생용)
        st.session_state.last_customer_audio_bytes = None
    if "keep_customer_audio_display" not in st.session_state:  # 고객 오디오 재생 표시 플래그
        st.session_state.keep_customer_audio_display = False
    if "customer_audio_played_once" not in st.session_state:  # 고객 오디오 재생 상태 플래그
        st.session_state.customer_audio_played_once = False
    if "supervisor_policy_context" not in st.session_state:
        # Supervisor가 업로드한 예외 정책 텍스트를 저장합니다.
        st.session_state.supervisor_policy_context = ""
    if "agent_policy_attachment_content" not in st.session_state:
        # 에이전트가 업로드한 정책 파일 객체(또는 내용)를 저장합니다.
        st.session_state.agent_policy_attachment_content = ""
    if "customer_attachment_b64" not in st.session_state:

            st.session_state.customer_attachment_b64 = ""
    if "customer_history_summary" not in st.session_state:

            st.session_state.customer_history_summary = ""
    if "customer_avatar" not in st.session_state:

            st.session_state.customer_avatar = {
            "gender": "male",  # 기본값
            "state": "NEUTRAL",  # 기본 아바타 상태
        }
    # ⭐ 추가: 비디오 동기화 관련 세션 상태
    if "current_customer_video" not in st.session_state:

            st.session_state.current_customer_video = None  # 현재 재생 중인 고객 비디오 경로
    if "current_customer_video_bytes" not in st.session_state:

            st.session_state.current_customer_video_bytes = None  # 현재 재생 중인 고객 비디오 바이트
    if "is_video_sync_enabled" not in st.session_state:

            st.session_state.is_video_sync_enabled = True  # 비디오 동기화 활성화 여부
    if "video_male_neutral" not in st.session_state:

            st.session_state.video_male_neutral = None  # 남자 중립 비디오 경로
    if "video_male_happy" not in st.session_state:

            st.session_state.video_male_happy = None
    if "video_male_angry" not in st.session_state:

            st.session_state.video_male_angry = None
    if "video_male_asking" not in st.session_state:

            st.session_state.video_male_asking = None
    if "video_male_sad" not in st.session_state:

            st.session_state.video_male_sad = None
    if "video_female_neutral" not in st.session_state:

            st.session_state.video_female_neutral = None  # 여자 중립 비디오 경로
    if "video_female_happy" not in st.session_state:

            st.session_state.video_female_happy = None
    if "video_female_angry" not in st.session_state:

            st.session_state.video_female_angry = None
    if "video_female_asking" not in st.session_state:

            st.session_state.video_female_asking = None
    if "video_female_sad" not in st.session_state:

            st.session_state.video_female_sad = None
    # ⭐ 추가: 전사할 오디오 바이트 임시 저장소
    if "bytes_to_process" not in st.session_state:

            st.session_state.bytes_to_process = None

    # 언어 키 안전하게 가져오기
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])

    # ⭐ 2-A. Gemini 키 초기화 (잘못된 키 잔존 방지)
    if "user_gemini_key" in st.session_state and st.session_state["user_gemini_key"].startswith(
        "AIza"):
        pass

    # ========================================
    # 0. 세션 상태 초기화
    # ========================================

    # 세션 초기화 (SUPPORTED_APIS는 config에서 import됨)
    for api, cfg in SUPPORTED_APIS.items():
        if cfg["session_key"] not in st.session_state:
            st.session_state[cfg["session_key"]] = ""

    if "selected_llm" not in st.session_state:
        st.session_state.selected_llm = "openai_gpt4"


    # ========================================
    # 1. Sidebar UI: API Key 입력 제거
    # ========================================
    # API Key 입력 UI는 제거하고, 환경변수와 Streamlit Secrets만 사용하도록 함.


    # ========================================
    # 2. LLM 클라이언트 라우팅 & 실행
    # ========================================
    # ========================================
    # 2-A. Whisper / TTS 용 OpenAI Client 별도로 초기화
    # ========================================

    if "openai_client" not in st.session_state or st.session_state.openai_client is None:
        try:
            st.session_state.openai_client = init_openai_audio_client()
        except Exception as e:
            st.session_state.openai_client = None
            print(f"OpenAI 클라이언트 초기화 중 오류 (무시됨): {e}")

    # LLM 준비 상태 캐싱 (API 키 변경 시에만 재확인)
    # ⭐ 수정: 초기화 시 블로킹 방지를 위해 try-except 추가
    if "is_llm_ready" not in st.session_state or "llm_ready_checked" not in st.session_state:
        try:
            probe_client, _ = get_llm_client()
            st.session_state.is_llm_ready = probe_client is not None
        except Exception as e:
            # 초기화 실패 시에도 앱이 계속 실행되도록 False로 설정
            st.session_state.is_llm_ready = False
            print(f"LLM 초기화 중 오류 (무시됨): {e}")
        st.session_state.llm_ready_checked = True

    # API 키 변경 감지를 위한 해시 체크
    current_api_keys_hash = hashlib.md5(
    f"{get_api_key('openai')}{get_api_key('gemini')}{get_api_key('claude')}{get_api_key('groq')}".encode()
    ).hexdigest()

    if "api_keys_hash" not in st.session_state:
        st.session_state.api_keys_hash = current_api_keys_hash
    elif st.session_state.api_keys_hash != current_api_keys_hash:
        # API 키가 변경된 경우만 재확인
        # ⭐ 수정: 초기화 시 블로킹 방지를 위해 try-except 추가
        try:
            probe_client, _ = get_llm_client()
            st.session_state.is_llm_ready = probe_client is not None
        except Exception as e:
            st.session_state.is_llm_ready = False
            print(f"LLM 재초기화 중 오류 (무시됨): {e}")
        st.session_state.api_keys_hash = current_api_keys_hash
    # OpenAI 클라이언트도 재초기화
        try:
            st.session_state.openai_client = init_openai_audio_client()
        except Exception as e:
            st.session_state.openai_client = None
            print(f"OpenAI 클라이언트 재초기화 중 오류 (무시됨): {e}")

    if st.session_state.openai_client:
        # 키를 찾았고 클라이언트 객체는 생성되었으나, 실제 인증은 API 호출 시 이루어짐 (401 오류는 여기서 발생)
        st.session_state.openai_init_msg = "✅ OpenAI TTS/Whisper 클라이언트 준비 완료 (Key 확인됨)"
    else:
        # 키를 찾지 못한 경우
        st.session_state.openai_init_msg = L["openai_missing"]

    if not st.session_state.is_llm_ready:


            st.session_state.llm_init_error_msg = L["simulation_no_key_warning"]
    else:
        st.session_state.llm_init_error_msg = ""


    # ----------------------------------------
    # LLM 번역 함수는 simulation_handler.py로 이동됨
    # ----------------------------------------

    # ========================================
    # 3. Whisper / TTS Helper는 audio_handler.py로 이동됨
    # ========================================

    # ========================================
    # 비디오 동기화 관련 함수는 video_handler.py로 이동됨
    # 시뮬레이션 관련 함수는 simulation_handler.py로 이동됨
    # ========================================

    # ========================================
    # 8. LLM (ChatOpenAI) for Simulator / Content
    # (RAG와 동일하게 run_llm으로 통합)
    # ========================================

    # ConversationChain 대신 run_llm을 사용하여 메모리 기능을 수동으로 구현
    # st.session_state.simulator_memory는 유지하여 대화 기록을 관리합니다.
    # 함수들은 visualization.py와 simulation_handler.py에서 import됨

    # ========================================
    # 9. 사이드바는 ui/sidebar.py에서 처리됨
    # ========================================
    
    # 회사 목록 초기화 (회사 정보 탭에서 사용)
    if "company_language_priority" not in st.session_state:
        st.session_state.company_language_priority = {
            "default": ["ko", "en", "ja"],
            "companies": {}
        }

