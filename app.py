# ========================================
# ⚠️ 주의: 이 파일은 레거시/백업 파일입니다.
# 메인 앱은 streamlit_app.py를 사용하세요.
# ========================================
# 이 파일은 이전 버전의 단순한 채팅/전화 앱입니다.
# streamlit_app.py는 더 많은 기능(_pages 모듈, 다국어 지원 등)을 포함합니다.
# GitHub commit 시 두 파일 모두 유지하되, streamlit_app.py를 메인으로 사용하세요.
# ========================================

import streamlit as st
import time
import json
import os
from datetime import datetime
from audio_handler import AudioHandler
from admin import AdminManager
from customer_data import CustomerDataManager
from call_handler import CallHandler
import uuid
from PIL import Image
import io
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="실시간 채팅",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False
if 'audio_handler' not in st.session_state:
    st.session_state.audio_handler = AudioHandler()
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
if 'customer_data_manager' not in st.session_state:
    st.session_state.customer_data_manager = CustomerDataManager()
if 'call_handler' not in st.session_state:
    st.session_state.call_handler = CallHandler()
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "채팅"
if 'call_active' not in st.session_state:
    st.session_state.call_active = False
if 'current_call_id' not in st.session_state:
    st.session_state.current_call_id = None
if 'video_enabled' not in st.session_state:
    st.session_state.video_enabled = False
if 'opponent_video_frames' not in st.session_state:
    st.session_state.opponent_video_frames = []
if 'last_camera_frame' not in st.session_state:
    st.session_state.last_camera_frame = None
if 'incoming_call' not in st.session_state:
    st.session_state.incoming_call = None
if 'inquiry_text' not in st.session_state:
    st.session_state.inquiry_text = ""

# 관리자 매니저 초기화
admin_manager = AdminManager()

def save_chat_log(user_id, message, sender, audio_file=None):
    """채팅 로그 저장"""
    log_dir = "chat_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{user_id}.json")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "sender": sender,
        "message": message,
        "audio_file": audio_file
    }
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def show_call_tab():
    """전화 통화 탭"""
    st.title("📞 실시간 전화 통화")
    st.caption("오디오 및 비디오로 실시간 통화를 시뮬레이션합니다")
    
    # 전화 수신 섹션
    st.subheader("📞 전화 수신")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        caller_phone = st.text_input("발신자 전화번호", placeholder="010-1234-5678", key="caller_phone_input")
        if st.button("📞 전화 수신", use_container_width=True, type="primary"):
            if caller_phone:
                result = st.session_state.call_handler.receive_call(
                    st.session_state.user_id,
                    caller_phone
                )
                st.session_state.incoming_call = result
                st.session_state.call_active = True
                st.session_state.current_call_id = result["call_id"]
                st.success(f"전화 수신: {caller_phone}")
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
            else:
                st.warning("전화번호를 입력해주세요.")
    
    with col2:
        if st.session_state.incoming_call:
            st.info(f"수신 중: {st.session_state.incoming_call.get('caller_phone', 'N/A')}")
    
    st.divider()
    
    # 통화 상태 표시
    call_status = st.session_state.call_handler.get_call_status()
    
    # 통화 제어 영역
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if not st.session_state.call_active:
            if st.button("📞 통화 시작", use_container_width=True, type="primary"):
                call_id = st.session_state.call_handler.start_call(st.session_state.user_id)
                st.session_state.call_active = True
                st.session_state.current_call_id = call_id
                st.success("통화가 시작되었습니다!")
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
        else:
            if st.button("📴 통화 종료", use_container_width=True, type="secondary"):
                duration = st.session_state.call_handler.end_call(
                    st.session_state.user_id,
                    st.session_state.current_call_id
                )
                st.session_state.call_active = False
                st.session_state.current_call_id = None
                st.session_state.incoming_call = None
                # ⭐ 수정: 통화 시간 표시 (몇 분 몇 초 형식)
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                if minutes > 0:
                    duration_msg = f"통화가 종료되었습니다. (통화 시간: {minutes}분 {seconds}초)"
                else:
                    duration_msg = f"통화가 종료되었습니다. (통화 시간: {seconds}초)"
                st.success(duration_msg)
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
    
    with col2:
        if st.session_state.call_active:
            st.session_state.video_enabled = st.toggle(
                "📹 비디오 활성화",
                value=st.session_state.video_enabled,
                help="비디오 통화를 활성화합니다"
            )
    
    with col3:
        if call_status:
            minutes = int(call_status['duration'] // 60)
            seconds = int(call_status['duration'] % 60)
            st.metric("통화 시간", f"{minutes:02d}:{seconds:02d}")
    
    st.divider()
    
    # 문의 입력 섹션
    if st.session_state.call_active:
        st.subheader("💬 문의 입력")
        inquiry_text = st.text_area(
            "고객 문의 내용을 입력하세요",
            value=st.session_state.inquiry_text,
            key="inquiry_input",
            height=100
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📝 문의 저장", use_container_width=True):
                if inquiry_text and st.session_state.current_call_id:
                    result = st.session_state.call_handler.process_inquiry(
                        st.session_state.current_call_id,
                        inquiry_text
                    )
                    if result:
                        st.success("문의가 저장되었습니다!")
                        st.session_state.inquiry_text = ""
                        # st.rerun()  # 주석 처리: 과도한 rerun 방지
                    else:
                        st.error("문의 저장에 실패했습니다.")
                else:
                    st.warning("문의 내용을 입력해주세요.")
        
        with col2:
            if st.button("🗑️ 초기화", use_container_width=True):
                st.session_state.inquiry_text = ""
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
    
    st.divider()
    
    # 통화 중인 경우
    if st.session_state.call_active:
        # 비디오 영역
        if st.session_state.video_enabled:
            video_col1, video_col2 = st.columns(2)
            
            with video_col1:
                st.subheader("📹 내 화면")
                camera_image = st.camera_input("웹캠", key="my_camera", help="내 웹캠 영상")
                if camera_image:
                    st.image(camera_image, use_container_width=True)
                    st.session_state.last_camera_frame = camera_image
                    if len(st.session_state.opponent_video_frames) >= 3:
                        st.session_state.opponent_video_frames.pop(0)
                    st.session_state.opponent_video_frames.append({
                        'image': camera_image,
                        'timestamp': time.time()
                    })
            
            with video_col2:
                st.subheader("📹 상대방 화면")
                if st.session_state.opponent_video_frames:
                    display_frame_idx = max(0, len(st.session_state.opponent_video_frames) - 2)
                    if display_frame_idx < len(st.session_state.opponent_video_frames):
                        opponent_frame = st.session_state.opponent_video_frames[display_frame_idx]['image']
                        try:
                            img = Image.open(io.BytesIO(opponent_frame.getvalue()))
                            mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            img_array = np.array(mirrored_img)
                            img_array = (img_array * 0.9).astype(np.uint8)
                            processed_img = Image.fromarray(img_array)
                            st.image(processed_img, use_container_width=True, caption="상대방 화면 (시뮬레이션)")
                        except Exception as e:
                            st.image(opponent_frame, use_container_width=True, caption="상대방 화면 (시뮬레이션)")
                    else:
                        st.info("상대방 비디오를 준비하는 중...")
                elif st.session_state.last_camera_frame:
                    try:
                        img = Image.open(io.BytesIO(st.session_state.last_camera_frame.getvalue()))
                        mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        img_array = np.array(mirrored_img)
                        img_array = (img_array * 0.9).astype(np.uint8)
                        processed_img = Image.fromarray(img_array)
                        st.image(processed_img, use_container_width=True, caption="상대방 화면 (시뮬레이션)")
                    except:
                        st.image(st.session_state.last_camera_frame, use_container_width=True, caption="상대방 화면 (시뮬레이션)")
                else:
                    st.info("상대방 비디오 스트림을 기다리는 중...")
        
        # 오디오 통화 영역
        st.subheader("🎤 오디오 통화")
        audio_col1, audio_col2 = st.columns(2)
        
        with audio_col1:
            st.markdown("**내 음성**")
            my_audio = st.audio_input("말씀하세요", key="call_audio_input", help="통화 중 음성을 녹음합니다")
            
            if my_audio:
                st.session_state.call_handler.add_audio_chunk(my_audio, "user")
                st.audio(my_audio, format="audio/wav", autoplay=False)
                
                with st.spinner("상대방이 응답하는 중..."):
                    response = st.session_state.call_handler.simulate_response(my_audio)
                    st.info(f"💬 상대방: {response['text']}")
                    st.session_state.call_handler.add_audio_chunk(None, "assistant")
                    # st.rerun()  # 주석 처리: 과도한 rerun 방지
        
        with audio_col2:
            st.markdown("**상대방 음성**")
            st.info("상대방의 음성이 여기에 재생됩니다")
            if call_status:
                st.metric("오디오 청크 수", call_status['chunks_count'])
        
        # 통화 로그
        with st.expander("📋 통화 로그", expanded=False):
            if call_status:
                st.json({
                    "통화 ID": st.session_state.current_call_id,
                    "통화 시간": f"{int(call_status['duration'] // 60):02d}:{int(call_status['duration'] % 60):02d}",
                    "오디오 청크": call_status['chunks_count'],
                    "비디오 활성화": st.session_state.video_enabled
                })
        
        # 통화 시간 업데이트
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 5:
            st.session_state.last_refresh = current_time
            # st.rerun()  # 주석 처리: 과도한 rerun 방지 (5초마다 자동 업데이트 비활성화)
    
    else:
        st.info("""
        ### 📞 전화 통화 기능 사용 방법
        
        1. **전화 수신**: 발신자 전화번호를 입력하고 '전화 수신' 버튼을 클릭합니다
        2. **통화 시작**: '통화 시작' 버튼을 클릭하여 통화를 시작합니다
        3. **문의 입력**: 통화 중 고객 문의 내용을 입력하고 저장할 수 있습니다
        4. **비디오 활성화**: 토글을 켜면 비디오 통화가 가능합니다
        5. **통화 종료**: '통화 종료' 버튼을 클릭하여 통화를 종료합니다
        """)

def main():
    # 사이드바
    with st.sidebar:
        st.title("💬 앱 설정")
        
        # 탭 선택
        st.subheader("기능 선택")
        tab_option = st.radio(
            "탭 선택",
            ["채팅", "전화 통화"],
            key="tab_selector",
            index=0 if st.session_state.current_tab == "채팅" else 1
        )
        
        if tab_option != st.session_state.current_tab:
            st.session_state.current_tab = tab_option
            # st.rerun()  # 주석 처리: 과도한 rerun 방지
        
        st.divider()
        st.title("💬 채팅 설정")
        
        # 관리자 모드
        if st.session_state.admin_logged_in:
            st.success("관리자 모드 활성화됨")
            if st.button("일반 모드로 전환", use_container_width=True):
                st.session_state.admin_logged_in = False
                st.session_state.is_admin = False
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
            
            st.divider()
            if st.button("관리자 대시보드", use_container_width=True):
                st.session_state.show_admin = True
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
        else:
            if st.button("관리자 로그인", use_container_width=True):
                st.session_state.show_admin_login = True
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
        
        st.divider()
        
        # 사용자 정보
        st.subheader("사용자 정보")
        st.text(f"사용자 ID: {st.session_state.user_id[:8]}")
        st.text(f"채팅 수: {len(st.session_state.messages)}")
        
        # 채팅 초기화
        if st.button("채팅 초기화", use_container_width=True):
            st.session_state.messages = []
            # st.rerun()  # 주석 처리: 과도한 rerun 방지
        
        st.divider()
        
        # 입력 방식 선택
        st.subheader("입력 방식")
        input_mode = st.radio(
            "입력 방식 선택",
            ["텍스트", "오디오"],
            key="input_mode"
        )

    # 관리자 로그인 화면
    if st.session_state.get('show_admin_login', False):
        admin_manager.show_login()
        return
    
    # 관리자 대시보드
    if st.session_state.get('show_admin', False):
        admin_manager.show_dashboard()
        return
    
    # 탭에 따라 다른 화면 표시
    if st.session_state.current_tab == "전화 통화":
        show_call_tab()
        return

    # 메인 채팅 인터페이스
    st.title("💬 실시간 채팅")
    st.caption("텍스트 또는 오디오로 대화하세요")
    
    # 채팅 메시지 표시
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("audio_file"):
                    st.audio(message["audio_file"], format="audio/wav")
                if message.get("timestamp"):
                    st.caption(message["timestamp"])
                
                if message["role"] == "user":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("💡 힌트", key=f"hint_{len(st.session_state.messages)}_{message.get('timestamp', '')}", use_container_width=True):
                            st.info("응대 힌트 기능은 추후 구현 예정입니다.")
                    with col2:
                        if st.button("📞 업체", key=f"call_company_{len(st.session_state.messages)}_{message.get('timestamp', '')}", use_container_width=True):
                            st.info("업체에 전화 기능은 추후 구현 예정입니다.")
                    with col3:
                        if st.button("📞 고객", key=f"call_customer_{len(st.session_state.messages)}_{message.get('timestamp', '')}", use_container_width=True):
                            st.info("고객에게 전화 기능은 추후 구현 예정입니다.")

    # 입력 영역
    st.divider()
    
    # 고객 데이터 및 AI 답변 요청 버튼 영역
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        if st.button("📋 고객 데이터", use_container_width=True, type="secondary"):
            customer_data = st.session_state.customer_data_manager.load_customer_data(
                st.session_state.user_id
            )
            if customer_data:
                st.session_state.customer_data = customer_data
                st.success("고객 데이터를 불러왔습니다!")
                customer_info = customer_data.get("data", {})
                info_message = f"📋 **고객 정보 불러옴**\n\n"
                info_message += f"이름: {customer_info.get('name', 'N/A')}\n"
                info_message += f"이메일: {customer_info.get('email', 'N/A')}\n"
                info_message += f"전화번호: {customer_info.get('phone', 'N/A')}\n"
                info_message += f"회사: {customer_info.get('company', 'N/A')}\n"
                info_message += f"메모: {customer_info.get('notes', 'N/A')}"
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": info_message,
                    "timestamp": timestamp
                })
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
            else:
                st.session_state.customer_data_manager.create_sample_data(
                    st.session_state.user_id
                )
                st.info("고객 데이터가 없어 샘플 데이터를 생성했습니다. 다시 시도해주세요.")
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
    
    with col3:
        if st.button("🤖 AI 답변", use_container_width=True, type="primary"):
            if st.session_state.messages:
                recent_messages = st.session_state.messages[-5:]
                context = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in recent_messages
                ])
                
                customer_context = ""
                if st.session_state.customer_data:
                    customer_info = st.session_state.customer_data.get("data", {})
                    customer_context = f"\n\n고객 정보:\n이름: {customer_info.get('name', 'N/A')}\n"
                    customer_context += f"이메일: {customer_info.get('email', 'N/A')}\n"
                    customer_context += f"회사: {customer_info.get('company', 'N/A')}\n"
                
                with st.spinner("AI가 답변을 생성하는 중..."):
                    time.sleep(1)
                    ai_response = f"🤖 **AI 분석 결과**\n\n"
                    ai_response += f"최근 대화 맥락을 분석한 결과, 고객님의 문의사항에 대해 다음과 같이 답변드립니다:\n\n"
                    ai_response += f"대화 내용을 바탕으로 관련 정보를 제공해드리겠습니다. "
                    if customer_context:
                        ai_response += f"고객 정보를 참고하여 더 정확한 답변을 드릴 수 있습니다. "
                    ai_response += f"추가로 필요한 정보가 있으시면 언제든지 말씀해주세요."
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": timestamp,
                    "is_ai_response": True
                })
                
                save_chat_log(st.session_state.user_id, ai_response, "assistant")
                # st.rerun()  # 주석 처리: 과도한 rerun 방지
            else:
                st.warning("대화 내용이 없습니다. 먼저 메시지를 입력해주세요.")
    
    # 고객 데이터 표시
    if st.session_state.customer_data:
        with st.expander("📋 현재 고객 정보", expanded=False):
            customer_info = st.session_state.customer_data.get("data", {})
            st.json(customer_info)
    
    if input_mode == "텍스트":
        user_input = st.chat_input("메시지를 입력하세요...")
        
        if user_input:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })
            
            save_chat_log(st.session_state.user_id, user_input, "user")
            
            with st.chat_message("assistant"):
                with st.spinner("응답 생성 중..."):
                    time.sleep(0.5)
                    
                    if st.session_state.customer_data:
                        customer_info = st.session_state.customer_data.get("data", {})
                        customer_name = customer_info.get("name", "고객님")
                        bot_response = f"안녕하세요 {customer_name}님! '{user_input}'라고 말씀하셨네요. 실시간 채팅이 작동하고 있습니다!"
                    else:
                        bot_response = f"안녕하세요! '{user_input}'라고 말씀하셨네요. 실시간 채팅이 작동하고 있습니다!"
                    
                    st.write(bot_response)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.caption(timestamp)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "timestamp": timestamp
            })
            
            save_chat_log(st.session_state.user_id, bot_response, "assistant")
            # st.rerun()  # 주석 처리: 과도한 rerun 방지
    
    else:  # 오디오 입력
        st.subheader("🎤 오디오 입력")
        audio_bytes = st.audio_input("음성을 녹음하세요", key="audio_input")
        
        if audio_bytes:
            audio_file = st.session_state.audio_handler.save_audio(
                audio_bytes, 
                st.session_state.user_id
            )
            
            with st.spinner("음성을 텍스트로 변환 중..."):
                time.sleep(1)
                transcribed_text = "[오디오 메시지] 음성이 녹음되었습니다."
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({
                "role": "user",
                "content": transcribed_text,
                "audio_file": audio_file,
                "timestamp": timestamp
            })
            
            save_chat_log(
                st.session_state.user_id, 
                transcribed_text, 
                "user",
                audio_file
            )
            
            with st.chat_message("assistant"):
                with st.spinner("응답 생성 중..."):
                    time.sleep(0.5)
                    bot_response = "오디오 메시지를 받았습니다! 음성 채팅이 작동하고 있습니다."
                    st.write(bot_response)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.caption(timestamp)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "timestamp": timestamp
            })
            
            save_chat_log(st.session_state.user_id, bot_response, "assistant")
            # st.rerun()  # 주석 처리: 과도한 rerun 방지

if __name__ == "__main__":
    main()


