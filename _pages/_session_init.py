# ========================================
# _pages/_session_init.py
# 세션 상태 초기화 모듈
# ========================================

import streamlit as st
from datetime import datetime, timedelta
import uuid
import hashlib
import os  # ⭐ 환경변수 직접 확인을 위해 추가
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
    # ⭐ 수정: 선택된 모델의 키가 없어도 다른 사용 가능한 API Key가 있으면 True로 설정
    if "is_llm_ready" not in st.session_state or "llm_ready_checked" not in st.session_state:
        # 먼저 사용 가능한 API Key 확인 (선택된 모델과 관계없이)
        openai_key = get_api_key("openai")
        gemini_key = get_api_key("gemini")
        claude_key = get_api_key("claude")
        groq_key = get_api_key("groq")
        
        has_any_key = any([
            bool(openai_key),
            bool(gemini_key),
            bool(claude_key),
            bool(groq_key)
        ])
        
        # 디버깅: 어떤 키가 감지되었는지 확인
        if has_any_key:
            detected_keys = []
            if openai_key:
                detected_keys.append("OpenAI")
            if gemini_key:
                detected_keys.append("Gemini")
            if claude_key:
                detected_keys.append("Claude")
            if groq_key:
                detected_keys.append("Groq")
            print(f"✅ 감지된 API Keys: {', '.join(detected_keys)}")
        else:
            print("⚠️ API Key가 감지되지 않았습니다. 환경변수 또는 .streamlit/secrets.toml을 확인하세요.")
        
        try:
            probe_client, _ = get_llm_client()
            # 선택된 모델의 클라이언트가 없어도, 사용 가능한 API Key가 하나라도 있으면 True
            if probe_client is None:
                st.session_state.is_llm_ready = has_any_key
            else:
                st.session_state.is_llm_ready = True
        except Exception as e:
            # 예외 발생 시에도 사용 가능한 API Key 확인
            st.session_state.is_llm_ready = has_any_key
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
        # ⭐ 수정: 선택된 모델의 키가 없어도 다른 사용 가능한 API Key가 있으면 True로 설정
        # 먼저 사용 가능한 API Key 확인 (선택된 모델과 관계없이)
        openai_key = get_api_key("openai")
        gemini_key = get_api_key("gemini")
        claude_key = get_api_key("claude")
        groq_key = get_api_key("groq")
        
        has_any_key = any([
            bool(openai_key),
            bool(gemini_key),
            bool(claude_key),
            bool(groq_key)
        ])
        
        try:
            probe_client, _ = get_llm_client()
            # 선택된 모델의 클라이언트가 없어도, 사용 가능한 API Key가 하나라도 있으면 True
            if probe_client is None:
                st.session_state.is_llm_ready = has_any_key
            else:
                st.session_state.is_llm_ready = True
        except Exception as e:
            # 예외 발생 시에도 사용 가능한 API Key 확인
            st.session_state.is_llm_ready = has_any_key
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

    # ⭐ API Key가 실제로 있는지 다시 확인 (항상 최신 상태로 확인)
    # os.environ을 직접 확인하여 최신 환경변수 반영
    openai_key = get_api_key("openai")
    gemini_key = get_api_key("gemini")
    claude_key = get_api_key("claude")
    groq_key = get_api_key("groq")
    
    # 환경변수를 직접 확인 (대소문자 변형 포함)
    if not openai_key:
        openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key") or ""
    if not gemini_key:
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini_api_key") or ""
    
    has_any_key_final = any([
        bool(openai_key and openai_key.strip()),
        bool(gemini_key and gemini_key.strip()),
        bool(claude_key and claude_key.strip()),
        bool(groq_key and groq_key.strip())
    ])
    
    # ⭐ API Key가 있으면 is_llm_ready를 항상 True로 강제 설정
    if has_any_key_final:
        st.session_state.is_llm_ready = True
        st.session_state.llm_init_error_msg = ""
        # 디버깅 정보
        detected = []
        if openai_key: detected.append("OpenAI")
        if gemini_key: detected.append("Gemini")
        if claude_key: detected.append("Claude")
        if groq_key: detected.append("Groq")
        print(f"✅ API Keys 감지됨: {', '.join(detected)}")
    else:
        # API Key가 없을 때만 에러 메시지 설정
        st.session_state.is_llm_ready = False
        st.session_state.llm_init_error_msg = L["simulation_no_key_warning"]
        print("⚠️ API Key가 감지되지 않았습니다. 환경변수를 확인하세요:")
        print(f"  - OPENAI_API_KEY: {'설정됨' if os.environ.get('OPENAI_API_KEY') else '없음'}")
        print(f"  - GEMINI_API_KEY: {'설정됨' if os.environ.get('GEMINI_API_KEY') else '없음'}")


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

