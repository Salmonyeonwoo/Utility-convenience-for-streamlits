# ========================================
# _pages/_call_audio.py
# 전화 통화 오디오 처리 모듈
# ========================================

import streamlit as st
from datetime import datetime
from utils.audio_handler import transcribe_bytes_with_whisper, synthesize_tts
from simulation_handler import (
    generate_customer_reaction,
    generate_customer_reaction_for_first_greeting,
    generate_customer_reaction_for_call
)


def process_audio_input(audio_input, current_lang, L):
    """오디오 입력 처리 (전사 및 고객 반응 생성)"""
    if not audio_input:
        return
    
    st.audio(audio_input, format="audio/wav", autoplay=False)
    
    is_llm_ready = st.session_state.get("is_llm_ready", False)
    audio_key = f"processed_{hash(audio_input.getvalue())}"
    
    if audio_key not in st.session_state:
        st.session_state[audio_key] = True
        st.info("💬 음성이 녹음되었습니다. 전사 처리 중...")
        
        if not transcribe_bytes_with_whisper:
            st.warning("⚠️ 전사 기능을 사용할 수 없습니다.")
            return
        
        if not is_llm_ready:
            st.warning("⚠️ LLM이 준비되지 않았습니다.")
            return
        
        try:
            transcript = transcribe_bytes_with_whisper(
                audio_input.getvalue(),
                "audio/wav",
                lang_code=None,
                auto_detect=True
            )
            
            if transcript and not transcript.startswith("❌"):
                st.success(f"💬 전사: {transcript}")
                
                if 'call_messages' not in st.session_state:
                    st.session_state.call_messages = []
                
                st.session_state.call_messages.append({
                    "role": "agent",
                    "content": transcript,
                    "timestamp": datetime.now().isoformat(),
                    "audio": audio_input.getvalue()
                })
                
                # 고객 반응 자동 생성
                _generate_customer_response(transcript, current_lang, L)
                
        except Exception as e:
            st.error(f"❌ 전사 오류: {str(e)}")


def _generate_customer_response(transcript, current_lang, L):
    """고객 반응 생성"""
    if not generate_customer_reaction:
        return
    
    try:
        is_first_agent_message = len(st.session_state.call_messages) == 1
        initial_inquiry = st.session_state.get("inquiry_text", "")
        
        if "customer_avatar" not in st.session_state:
            st.session_state.customer_avatar = {"gender": "male", "state": "NEUTRAL"}
        
        if is_first_agent_message and initial_inquiry and generate_customer_reaction_for_first_greeting:
            if initial_inquiry.strip():
                customer_response = generate_customer_reaction_for_first_greeting(
                    current_lang, transcript, initial_inquiry
                )
            else:
                customer_response = generate_customer_reaction(current_lang, is_call=True)
        else:
            if generate_customer_reaction_for_call:
                customer_response = generate_customer_reaction_for_call(current_lang, transcript)
            else:
                customer_response = generate_customer_reaction(current_lang, is_call=True)
        
        customer_audio = None
        
        # TTS로 오디오 생성
        if synthesize_tts:
            try:
                customer_audio_result = synthesize_tts(
                    customer_response, current_lang, role="customer"
                )
                if customer_audio_result and isinstance(customer_audio_result, tuple):
                    customer_audio_bytes, status_msg = customer_audio_result
                    if customer_audio_bytes:
                        customer_audio = customer_audio_bytes
                elif customer_audio_result:
                    customer_audio = customer_audio_result
            except Exception:
                pass
        
        st.session_state.call_messages.append({
            "role": "customer",
            "content": customer_response,
            "timestamp": datetime.now().isoformat(),
            "audio": customer_audio
        })
        
        st.info(f"💬 고객: {customer_response}")
        if customer_audio:
            st.audio(customer_audio, format="audio/mp3", autoplay=False)
            
    except Exception as e:
        pass
