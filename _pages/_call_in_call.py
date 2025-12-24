# -*- coding: utf-8 -*-
"""
전화 시뮬레이터 - 통화 중 모듈
"""
import streamlit as st
from lang_pack import LANG
from datetime import datetime
import time
from PIL import Image
import io
import numpy as np
from typing import List, Dict, Any

# 필요한 모듈 import
from simulation_handler import (
    generate_customer_reaction, 
    generate_customer_reaction_for_first_greeting,
    generate_customer_reaction_for_call,
    summarize_history_with_ai
)
from utils.audio_handler import (
    transcribe_bytes_with_whisper, synthesize_tts
)
from utils.translation import translate_text_with_llm

try:
    from llm_client import get_api_key
except ImportError:
    get_api_key = None

def render_call_in_call():
    """통화 중 UI - 오디오 녹음 + 전사 + 고객 반응 자동 생성"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    st.session_state.setdefault("is_on_hold", False)
    st.session_state.setdefault("hold_start_time", None)
    st.session_state.setdefault("hold_total_seconds", 0)
    st.session_state.setdefault("provider_call_active", False)
    st.session_state.setdefault("call_direction", "inbound")
    
    # ⭐ 수정: 통화 시작 시 call_messages 초기화 확인 (새 통화인 경우)
    if 'call_messages' not in st.session_state:
        st.session_state.call_messages = []
    
    # ⭐ 수정: 통화 수신 정보와 통화 시간을 깔끔한 UI로 표시
    # ⭐ 중요: start_time이 없으면 통화 수신 시점부터 시작 (RINGING 상태에서 설정됨)
    call_number = st.session_state.get("incoming_phone_number")
    call_direction = st.session_state.get("call_direction", "inbound")
    if call_number:
        # 통화 시간 계산 (start_time이 없으면 통화 수신 시점부터 시작)
        call_duration = 0
        if st.session_state.get("start_time"):
            call_duration = (datetime.now() - st.session_state.start_time).total_seconds()
        else:
            # ⭐ 수정: start_time이 없으면 현재 시점부터 시작 (통화 수신 시작과 동시에 카운팅)
            # RINGING 상태에서 이미 설정되어야 하지만, 혹시 모를 경우를 대비
            st.session_state.start_time = datetime.now()
            call_duration = 0
        
        minutes = int(call_duration // 60)
        seconds = int(call_duration % 60)
        duration_str = f"{minutes:02d}:{seconds:02d}"
        
        # 통화 정보를 깔끔한 UI로 표시
        col_info1, col_info2 = st.columns([2, 1])
        with col_info1:
            heading_template = L.get(
                "call_heading_outbound" if call_direction == "outbound" else "call_heading_inbound",
                "📞 전화 통화 중: {number}"
            )
            st.markdown(f"### {heading_template.format(number=call_number)}")
        with col_info2:
            st.metric(L.get("call_duration_label", "통화 시간"), duration_str)
    
    st.info(L.get("call_in_progress", "📞 통화 중입니다..."))
    
    # Hold 상태 및 누적 시간 계산
    hold_elapsed = st.session_state.get("hold_total_seconds", 0)
    if st.session_state.get("is_on_hold") and st.session_state.get("hold_start_time"):
        hold_elapsed += (datetime.now() - st.session_state.hold_start_time).total_seconds()
    hold_minutes = int(hold_elapsed // 60)
    hold_seconds = int(hold_elapsed % 60)
    hold_duration_str = f"{hold_minutes:02d}:{hold_seconds:02d}"

    # 통화 제어 영역 (5열: Hold/재개, 업체 발신, 응대 힌트, 비디오, 종료)
    col_hold, col_provider, col_hint, col_video, col_end = st.columns([1, 1, 1, 1, 1])
    with col_hold:
        if st.session_state.get("is_on_hold"):
            st.caption(L.get("hold_status", "통화 Hold 중 (누적 Hold 시간: {duration})").format(duration=hold_duration_str))
            if st.button(L.get("button_resume", "▶️ 통화 재개"), use_container_width=True):
                if st.session_state.get("hold_start_time"):
                    st.session_state.hold_total_seconds += (datetime.now() - st.session_state.hold_start_time).total_seconds()
                st.session_state.hold_start_time = None
                st.session_state.is_on_hold = False
                st.session_state.provider_call_active = False
                st.success(L.get("call_resumed", "통화를 재개했습니다."))
        else:
            if st.button(L.get("button_hold", "⏸️ Hold (소음 차단)"), use_container_width=True):
                st.session_state.is_on_hold = True
                st.session_state.hold_start_time = datetime.now()
                st.session_state.hold_total_seconds = 0  # 새 Hold 시작 시 누적 시간 초기화
                st.session_state.provider_call_active = False
                # 통화 기록에 Hold 알림 추가
                st.session_state.call_messages.append({
                    "role": "system_hold",
                    "content": L.get("agent_hold_message", "[에이전트: Hold 중입니다. 통화 재개 버튼을 눌러주세요.]"),
                    "timestamp": datetime.now().isoformat()
                })
    with col_provider:
        if st.button(
            L.get("button_call_company", "📞 업체에 전화"),
            use_container_width=True,
            disabled=not st.session_state.get("is_on_hold")
        ):
            # 업체 확인 안내: Hold 상태에서만 발신 가능
            st.session_state.provider_call_active = True
            st.session_state.is_on_hold = True
            if not st.session_state.get("hold_start_time"):
                st.session_state.hold_start_time = datetime.now()
            st.session_state.call_messages.append({
                "role": "agent",
                "content": L.get("provider_call_message", "업체에 확인해 보겠습니다. 잠시만 기다려 주세요."),
                "timestamp": datetime.now().isoformat()
            })
            st.info(L.get("provider_call_progress", "업체에 확인 중입니다. 잠시만 기다려 주세요."))
    with col_hint:
        # ⭐ 응대 힌트 버튼 추가
        if st.button(
            L.get("button_hint", "💡 응대 힌트"),
            use_container_width=True,
            help=L.get("button_hint_help", "현재 대화 맥락을 기반으로 실시간 응대 힌트를 제공합니다"),
            key="call_hint_button"
        ):
            if st.session_state.is_llm_ready:
                try:
                    from simulation_handler import generate_realtime_hint
                    session_lang = st.session_state.get("language", current_lang)
                    if session_lang not in ["ko", "en", "ja"]:
                        session_lang = current_lang
                    
                    with st.spinner(L.get("generating_hint", "응대 힌트 생성 중...")):
                        hint = generate_realtime_hint(session_lang, is_call=True)
                        if hint:
                            # 힌트를 시스템 메시지로 추가
                            st.session_state.call_messages.append({
                                "role": "supervisor",
                                "content": f"💡 **{L.get('hint_label', '응대 힌트')}**: {hint}",
                                "timestamp": datetime.now().isoformat()
                            })
                            st.session_state.realtime_hint_text = hint
                except Exception as e:
                    st.error(f"응대 힌트 생성 오류: {e}")
            else:
                has_api_key = any([
                    bool(get_api_key("openai")) if get_api_key else False,
                    bool(get_api_key("gemini")) if get_api_key else False,
                    bool(get_api_key("claude")) if get_api_key else False,
                    bool(get_api_key("groq")) if get_api_key else False
                ])
                if not has_api_key:
                    st.warning(L.get("simulation_no_key_warning", "LLM이 준비되지 않았습니다."))
                else:
                    st.session_state.is_llm_ready = True
    with col_video:
        if 'video_enabled' not in st.session_state:
            st.session_state.video_enabled = False
        st.session_state.video_enabled = st.toggle(
            L.get("button_video_enable", "📹 비디오"),
            value=st.session_state.video_enabled,
            help=L.get("video_enable_help", "비디오 통화를 활성화합니다")
        )
    with col_end:
        if st.button(L.get("button_end_call", "📴 종료"), use_container_width=True, type="primary"):
            # ⭐ 수정: 통화 시간 계산 및 저장
            call_duration = 0
            if st.session_state.get("start_time"):
                call_duration = (datetime.now() - st.session_state.start_time).total_seconds()
                st.session_state.call_duration = call_duration  # 통화 시간 저장
            
            # Hold 누적 시간 정리
            if st.session_state.get("is_on_hold") and st.session_state.get("hold_start_time"):
                st.session_state.hold_total_seconds += (datetime.now() - st.session_state.hold_start_time).total_seconds()
            st.session_state.is_on_hold = False
            st.session_state.hold_start_time = None
            st.session_state.provider_call_active = False
            st.session_state.call_sim_stage = "CALL_ENDED"
            st.session_state.call_active = False
            st.session_state.start_time = None
    st.markdown("---")
    
    # 비디오 영역 (비디오 활성화 시에만 표시)
    if st.session_state.video_enabled:
        video_col1, video_col2 = st.columns(2)
        
        with video_col1:
            st.markdown(f"**{L.get('my_screen', '📹 내 화면')}**")
            camera_image = st.camera_input(
                L.get("webcam_label", "웹캠"),
                key="my_camera_call",
                help=L.get("webcam_help", "내 웹캠 영상"),
            )
            if camera_image:
                st.image(camera_image, use_container_width=True)
                if 'opponent_video_frames' not in st.session_state:
                    st.session_state.opponent_video_frames = []
                if 'last_camera_frame' not in st.session_state:
                    st.session_state.last_camera_frame = None
                st.session_state.last_camera_frame = camera_image
                if len(st.session_state.opponent_video_frames) >= 3:
                    st.session_state.opponent_video_frames.pop(0)
                st.session_state.opponent_video_frames.append({
                    'image': camera_image,
                    'timestamp': time.time()
                })
        
        with video_col2:
            st.markdown(f"**{L.get('opponent_screen', '📹 상대방 화면')}**")
            if st.session_state.get("opponent_video_frames"):
                display_frame_idx = max(0, len(st.session_state.opponent_video_frames) - 2)
                if display_frame_idx < len(st.session_state.opponent_video_frames):
                    opponent_frame = st.session_state.opponent_video_frames[display_frame_idx]['image']
                    try:
                        img = Image.open(io.BytesIO(opponent_frame.getvalue()))
                        mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        img_array = np.array(mirrored_img)
                        img_array = (img_array * 0.9).astype(np.uint8)
                        processed_img = Image.fromarray(img_array)
                        st.image(processed_img, use_container_width=True, caption=L.get("opponent_screen_simulation", "상대방 화면 (시뮬레이션)"))
                    except Exception as e:
                        st.image(opponent_frame, use_container_width=True, caption=L.get("opponent_screen_simulation", "상대방 화면 (시뮬레이션)"))
                else:
                    st.info(L.get("opponent_video_preparing", "상대방 비디오를 준비하는 중..."))
            elif st.session_state.get("last_camera_frame"):
                try:
                    img = Image.open(io.BytesIO(st.session_state.last_camera_frame.getvalue()))
                    mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img_array = np.array(mirrored_img)
                    img_array = (img_array * 0.9).astype(np.uint8)
                    processed_img = Image.fromarray(img_array)
                    st.image(processed_img, use_container_width=True, caption=L.get("opponent_screen_simulation", "상대방 화면 (시뮬레이션)"))
                except:
                    st.image(st.session_state.last_camera_frame, use_container_width=True, caption=L.get("opponent_screen_simulation", "상대방 화면 (시뮬레이션)"))
            else:
                st.info(L.get("opponent_video_waiting", "상대방 비디오 스트림을 기다리는 중..."))
        
        st.markdown("---")
    
    # 오디오 녹음 및 전사 섹션
    st.markdown("**🎤 오디오 녹음 및 전사**")
    
    # 오디오 입력 영역 (2열: 오디오 입력, 상태)
    audio_col1, audio_col2 = st.columns([3, 1])
    
    with audio_col1:
        audio_input = st.audio_input(
            "말씀하세요",
            key="call_audio_input_in_call",
            help="음성을 녹음하면 자동으로 전사됩니다"
        )
    
    with audio_col2:
        if st.session_state.get("call_messages"):
            st.caption(L.get("messages_count", "메시지: {count}개").format(count=len(st.session_state.call_messages)))
    
    # 전사 결과 및 고객 반응 생성 (즉각 반응, 로딩 최소화, rerun 없음)
    if audio_input:
        # 오디오 재생 (즉시 표시, 로딩 없음)
        st.audio(audio_input, format="audio/wav", autoplay=False)
        
        # LLM 준비 상태 확인
        is_llm_ready = st.session_state.get("is_llm_ready", False)
        
        # 이미 처리된 오디오인지 확인 (중복 처리 방지)
        audio_key = f"processed_{hash(audio_input.getvalue())}"
        if audio_key not in st.session_state:
            st.session_state[audio_key] = True
            
            # 즉시 피드백 표시 (전사 처리 전)
            st.info("💬 음성이 녹음되었습니다. 전사 처리 중...")
            
            # 전사 처리 (spinner 없이 즉시 처리, 블로킹 최소화)
            if not transcribe_bytes_with_whisper:
                st.warning("⚠️ 전사 기능을 사용할 수 없습니다.")
            elif not is_llm_ready:
                st.warning("⚠️ LLM이 준비되지 않았습니다.")
            else:
                try:
                    # 전사 처리 (최소 지연, 블로킹 최소화)
                    transcript = transcribe_bytes_with_whisper(
                        audio_input.getvalue(),
                        "audio/wav",
                        lang_code=None,
                        auto_detect=True
                    )
                    
                    if transcript and not transcript.startswith("❌"):
                        # 전사 성공 - 즉시 표시 (이전 메시지 대체)
                        st.success(f"💬 전사: {transcript}")
                        
                        # 에이전트 메시지로 저장
                        if 'call_messages' not in st.session_state:
                            st.session_state.call_messages = []
                        
                        st.session_state.call_messages.append({
                            "role": "agent",
                            "content": transcript,
                            "timestamp": datetime.now().isoformat(),
                            "audio": audio_input.getvalue()
                        })
                        
                        # 고객 반응 자동 생성 (즉시 처리, 블로킹 최소화)
                        # ⭐ 수정: 통화 시작 시 첫 메시지인 경우 초기 문의를 고려한 반응 생성
                        if generate_customer_reaction:
                            try:
                                # 통화 시작 시 첫 에이전트 메시지인지 확인
                                is_first_agent_message = len(st.session_state.call_messages) == 1
                                initial_inquiry = st.session_state.get("inquiry_text", "")
                                
                                # ⭐ 수정: customer_avatar 초기화 확인
                                if "customer_avatar" not in st.session_state:
                                    st.session_state.customer_avatar = {"gender": "male", "state": "NEUTRAL"}
                                
                                if is_first_agent_message and initial_inquiry and generate_customer_reaction_for_first_greeting:
                                    # 첫 인사말에 대한 맞춤형 반응 생성 (초기 문의 고려)
                                    # ⭐ 중요: 초기 문의가 비어있지 않은 경우에만 사용
                                    if initial_inquiry.strip():
                                        customer_response = generate_customer_reaction_for_first_greeting(
                                            current_lang,
                                            transcript,  # 에이전트 인사말
                                            initial_inquiry  # 초기 문의
                                        )
                                    else:
                                        # 초기 문의가 없으면 일반 반응 생성
                                        customer_response = generate_customer_reaction(
                                            current_lang,
                                            is_call=True
                                        )
                                else:
                                    # ⭐ 수정: 일반 고객 반응 생성 시 에이전트 응답을 반영하여 적절히 답변
                                    # generate_customer_reaction_for_call을 사용하여 에이전트의 질문에 직접 답변
                                    if generate_customer_reaction_for_call:
                                        customer_response = generate_customer_reaction_for_call(
                                            current_lang,
                                            transcript  # 에이전트의 전사 결과를 전달
                                        )
                                    else:
                                        # 폴백: generate_customer_reaction_for_call이 없으면 일반 함수 사용
                                        customer_response = generate_customer_reaction(
                                            current_lang,
                                            is_call=True
                                        )
                                
                                # 고객 메시지로 저장
                                customer_audio = None
                                
                                # 고객 응답을 TTS로 오디오 생성 (백그라운드 처리, 블로킹 없음)
                                if synthesize_tts:
                                    try:
                                        customer_audio_result = synthesize_tts(
                                            customer_response,
                                            current_lang,
                                            role="customer"
                                        )
                                        if customer_audio_result and isinstance(customer_audio_result, tuple):
                                            customer_audio_bytes, status_msg = customer_audio_result
                                            if customer_audio_bytes:
                                                customer_audio = customer_audio_bytes
                                        elif customer_audio_result:
                                            customer_audio = customer_audio_result
                                    except Exception:
                                        pass  # TTS 실패해도 계속 진행
                                
                                st.session_state.call_messages.append({
                                    "role": "customer",
                                    "content": customer_response,
                                    "timestamp": datetime.now().isoformat(),
                                    "audio": customer_audio
                                })
                                
                                # 고객 응답 즉시 표시
                                st.info(f"💬 고객: {customer_response}")
                                if customer_audio:
                                    st.audio(customer_audio, format="audio/mp3", autoplay=False)
                                
                            except Exception as e:
                                # 고객 반응 생성 실패 시에도 계속 진행
                                pass
                    else:
                        error_msg = transcript if transcript else L.get("transcription_error", "전사 실패")
                        st.error(f"❌ {error_msg}")
                        
                except Exception as e:
                    # 전사 오류 시에도 계속 진행
                    st.error(f"❌ 전사 오류: {str(e)}")
    
    st.markdown("---")
    
    # 이관 요약 표시 (이관 후에만)
    if st.session_state.get("transfer_summary_text") or (
        st.session_state.get("language_at_transfer_start") and 
        st.session_state.language != st.session_state.get("language_at_transfer_start")
    ):
        with st.expander(f"**{L.get('transfer_summary_header', '이관 요약')}**", expanded=False):
            st.info(L.get("transfer_summary_intro", "다음은 이전 팀에서 전달받은 통화 요약입니다."))
            
            is_translation_failed = not st.session_state.get("translation_success", True) or not st.session_state.get("transfer_summary_text")
            
            if st.session_state.get("transfer_summary_text") and st.session_state.get("translation_success", True):
                st.markdown(st.session_state.transfer_summary_text)
            elif st.session_state.get("transfer_summary_text"):
                st.info(st.session_state.transfer_summary_text)
    
    st.markdown("---")
    
    # 통화 메시지 히스토리 표시 (간결하게)
    if st.session_state.get("call_messages"):
        with st.expander(L.get("call_history_label", "💬 통화 기록"), expanded=True):
            # ⭐ 추가: 기록 초기화 버튼 및 데이터 가져오기 버튼
            col_clear, col_load, _ = st.columns([1, 1, 3])
            with col_clear:
                if st.button(
                    L.get("clear_call_history", "🗑️ 기록 초기화"),
                    key="clear_call_history",
                    help=L.get("clear_call_history_help", "현재 통화 기록을 초기화합니다")):
                    st.session_state.call_messages = []
                    st.success(L.get("call_history_cleared", "통화 기록이 초기화되었습니다."))
            with col_load:
                if st.button(
                    L.get("load_call_history", "📥 데이터 가져오기"),
                    key="load_call_history",
                    help=L.get("load_call_history_help", "고객/전화번호별 이전 기록 불러오기")):
                    # ⭐ 추가: 고객/전화번호별 이전 기록 불러오기 기능
                    phone_number = st.session_state.get("incoming_phone_number", "")
                    if phone_number:
                        # 다운로드된 파일에서 해당 전화번호의 이전 기록 검색
                        try:
                            from utils.history_handler import load_simulation_histories_local
                            all_histories = load_simulation_histories_local(current_lang)
                            
                            # 해당 전화번호와 관련된 이전 기록 찾기
                            matching_histories = []
                            for history in all_histories:
                                # 전화 이력이고, 전화번호가 일치하는 경우
                                if history.get("is_call", False):
                                    # 초기 문의나 요약에서 전화번호 검색
                                    initial_query = history.get("initial_query", "")
                                    summary = history.get("summary", {})
                                    if isinstance(summary, dict):
                                        main_inquiry = summary.get("main_inquiry", "")
                                    else:
                                        main_inquiry = ""
                                    
                                    if phone_number in initial_query or phone_number in main_inquiry:
                                        matching_histories.append(history)
                            
                            if matching_histories:
                                st.info(f"📋 {len(matching_histories)}개의 이전 기록을 찾았습니다.")
                                # 가장 최근 기록 표시
                                latest_history = matching_histories[0]
                                with st.expander("📋 가장 최근 기록", expanded=True):
                                    if latest_history.get("summary"):
                                        summary = latest_history.get("summary", {})
                                        if isinstance(summary, dict):
                                            st.markdown(f"**초기 문의**: {latest_history.get('initial_query', 'N/A')}")
                                            st.markdown(f"**고객 유형**: {latest_history.get('customer_type', 'N/A')}")
                                            st.markdown(f"**주요 문의**: {summary.get('main_inquiry', 'N/A')}")
                                            st.markdown(f"**고객 감정 점수**: {summary.get('customer_sentiment_score', 'N/A')}")
                            else:
                                st.info("📋 이전 기록이 없습니다. 새로운 고객입니다.")
                        except Exception as e:
                            st.warning(f"기록 불러오기 오류: {e}")
                    else:
                        st.warning("전화번호를 먼저 입력해주세요.")
            
            for msg in st.session_state.call_messages:
                role = msg.get("role", "")
                # supervisor 메시지는 별도로 표시
                if role == "supervisor" or role == "system_hold":
                    st.info(msg.get("content", ""))
                else:
                    role_icon = "👤" if role == "agent" else "👥"
                    role_label = L.get("agent_label", "에이전트") if role == "agent" else L.get("customer_label", "고객")
                    with st.chat_message(role):
                        st.write(f"{role_icon} **{role_label}**: {msg.get('content', '')}")
                        # 오디오 재생 (고객 메시지에만)
                        if msg.get("audio") and role == "customer":
                            st.audio(msg["audio"], format="audio/mp3", autoplay=False)
                        elif msg.get("audio") and role == "agent":
                            st.audio(msg["audio"], format="audio/wav", autoplay=False)
                        if msg.get("timestamp"):
                            try:
                                ts = datetime.fromisoformat(msg["timestamp"])
                                st.caption(ts.strftime("%H:%M:%S"))
                            except:
                                pass
    
    st.markdown("---")
    
    # 통화 내용 수동 입력 (보조 기능) - 크기 축소
    st.markdown(f"**{L.get('call_content_memo', '📝 통화 내용 메모')}**")
    call_content = st.text_area(
        L.get("memo_input_placeholder", "메모 입력 (선택사항)"),
        value=st.session_state.get("call_content", ""),
        key="call_content_input",
        height=100,
        help=L.get("memo_input_help", "추가 메모를 작성할 수 있습니다")
    )
    
    if call_content:
        st.session_state.call_content = call_content
    
    st.markdown("---")
    
    # 언어 팀 이관 기능 추가
    st.markdown(f"**{L.get('transfer_header', '언어 팀 이관')}**")
    
    languages = list(LANG.keys())
    if current_lang in languages:
        languages.remove(current_lang)
    
    if languages:
        transfer_cols = st.columns(len(languages))
        
        def transfer_call_session(target_lang: str, current_messages: List[Dict[str, Any]]):
            """전화 통화 세션을 다른 언어 팀으로 이관 (로딩 최소화, 즉각 반응)"""
            current_lang_at_start = st.session_state.language
            L_source = LANG.get(current_lang_at_start, LANG["ko"])
            
            # 즉시 피드백 표시 (로딩 없음)
            lang_name_target = {"ko": "한국어", "en": "영어", "ja": "일본어"}.get(target_lang, target_lang)
            st.info(f"🔄 {lang_name_target} 팀으로 이관 처리 중...")
            
            # API 키 체크
            if get_api_key and not get_api_key("gemini"):
                st.error(L_source.get("simulation_no_key_warning", "⚠️ Gemini API Key가 설정되지 않았습니다.").replace('API Key', 'Gemini API Key'))
                return
            
            if not summarize_history_with_ai or not translate_text_with_llm:
                st.error("⚠️ 이관 기능을 사용할 수 없습니다. 필요한 모듈을 확인해주세요.")
                return
            
            try:
                # 원본 언어로 핵심 요약 생성
                original_summary = summarize_history_with_ai(current_lang_at_start)
                
                if not original_summary or original_summary.startswith("❌"):
                    # 요약 생성 실패 시 대화 기록을 번역할 텍스트로 가공
                    history_text = ""
                    for msg in current_messages:
                        role = "Customer" if msg.get("role") == "customer" else "Agent"
                        if msg.get("content"):
                            history_text += f"{role}: {msg['content']}\n"
                    original_summary = history_text
                
                # 핵심 요약을 번역 대상 언어로 번역
                translated_summary, is_success = translate_text_with_llm(
                    original_summary,
                    target_lang,
                    current_lang_at_start
                )
                
                if not translated_summary:
                    # 번역 실패 시 번역 대상 언어로 요약 재생성
                    translated_summary = summarize_history_with_ai(target_lang)
                    is_success = True if translated_summary and not translated_summary.startswith("❌") else False
                
                # 모든 메시지를 이관된 언어로 번역
                translated_messages = []
                messages_to_translate = []
                
                # 번역할 메시지 수집
                for idx, msg in enumerate(current_messages):
                    translated_msg = msg.copy()
                    if msg.get("role") in ["agent", "customer"] and msg.get("content"):
                        messages_to_translate.append((idx, msg))
                    translated_messages.append(translated_msg)
                
                # 배치 번역: 모든 메시지를 하나의 텍스트로 합쳐서 번역
                if messages_to_translate:
                    try:
                        combined_text = "\n\n".join([
                            f"[{msg['role']}]: {msg['content']}" 
                            for _, msg in messages_to_translate
                        ])
                        
                        translated_combined, trans_success = translate_text_with_llm(
                            combined_text,
                            target_lang,
                            current_lang_at_start
                        )
                        
                        if trans_success and translated_combined:
                            translated_lines = translated_combined.split("\n\n")
                            for i, (idx, original_msg) in enumerate(messages_to_translate):
                                if i < len(translated_lines):
                                    translated_line = translated_lines[i]
                                    if "]: " in translated_line:
                                        translated_content = translated_line.split("]: ", 1)[1]
                                    else:
                                        translated_content = translated_line
                                    translated_messages[idx]["content"] = translated_content
                    except Exception:
                        # 배치 번역 실패 시 개별 번역으로 폴백
                        for idx, msg in messages_to_translate:
                            try:
                                translated_content, trans_success = translate_text_with_llm(
                                    msg["content"],
                                    target_lang,
                                    current_lang_at_start
                                )
                                if trans_success:
                                    translated_messages[idx]["content"] = translated_content
                            except Exception:
                                pass
                
                # 번역된 메시지로 업데이트
                st.session_state.call_messages = translated_messages
                
                # 이관 요약 저장
                st.session_state.transfer_summary_text = translated_summary
                st.session_state.translation_success = is_success
                st.session_state.language_at_transfer_start = current_lang_at_start
                
                # 언어 변경
                st.session_state.language = target_lang
                L_target = LANG.get(target_lang, LANG["ko"])
                
                # 언어 이름 가져오기
                lang_name_target = {"ko": "한국어", "en": "영어", "ja": "일본어"}.get(target_lang, "한국어")
                
                # 시스템 메시지 추가
                system_msg = L_target.get("transfer_system_msg", "📌 시스템 메시지: 통화가 {target_lang} 팀으로 이관되었습니다.").format(target_lang=lang_name_target)
                st.session_state.call_messages.append({
                    "role": "system_transfer",
                    "content": system_msg,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 이관 요약을 supervisor 메시지로 추가
                summary_header = L_target.get("transfer_summary_header", "이관 요약")
                summary_msg = f"### {summary_header}\n\n{translated_summary}"
                st.session_state.call_messages.append({
                    "role": "supervisor",
                    "content": summary_msg,
                    "timestamp": datetime.now().isoformat()
                })
                
                st.success(f"✅ {lang_name_target} 팀으로 이관되었습니다.")
                
            except Exception as e:
                error_msg = L_source.get("transfer_error", "이관 처리 중 오류 발생: {error}").format(error=str(e))
                st.error(error_msg)
        
        # 이관 버튼 렌더링
        for idx, lang_code in enumerate(languages):
            lang_name = {"ko": "한국어", "en": "영어", "ja": "일본어"}.get(lang_code, lang_code)
            if lang_code == "en":
                transfer_label = "US 영어 팀으로 이관"
            elif lang_code == "ja":
                transfer_label = "JP 일본어 팀으로 이관"
            else:
                transfer_label = f"{lang_name} 팀으로 이관"
            
            with transfer_cols[idx]:
                if st.button(
                    transfer_label,
                    key=f"btn_call_transfer_{lang_code}_{st.session_state.get('sim_instance_id', 'default')}",
                    type="secondary",
                    use_container_width=True
                ):
                    transfer_call_session(lang_code, st.session_state.get("call_messages", []))
    else:
        st.info("이관할 다른 언어 팀이 없습니다.")
    
    st.markdown("---")
    
    col_save, _ = st.columns([1, 3])
    with col_save:
        if st.button("💾 저장", use_container_width=True):
            if call_content.strip() or st.session_state.get("call_messages"):
                st.success("통화 내용이 저장되었습니다.")
            else:
                st.warning("통화 내용을 입력하거나 오디오를 녹음해주세요.")
