# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
오디오 처리 모듈
Whisper 전사, TTS 합성 등의 오디오 관련 기능을 제공합니다.
"""

import os
import tempfile
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
import streamlit as st

from config import VOICE_META_FILE, AUDIO_DIR
from utils import _load_json, _save_json
from llm_client import get_api_key
from lang_pack import LANG

# Google Generative AI import
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

# TTS 음성 설정
TTS_VOICES = {
    "customer_male": {"gender": "male", "voice": "alloy"},
    "customer_female": {"gender": "female", "voice": "nova"},
    "customer": {"gender": "male", "voice": "alloy"},
    "agent": {"gender": "female", "voice": "shimmer"},
    "supervisor": {"gender": "female", "voice": "nova"}
}


def transcribe_bytes_with_whisper(audio_bytes: bytes, mime_type: str = "audio/webm", 
                                   lang_code: str = None, auto_detect: bool = True) -> str:
    """OpenAI Whisper API 또는 Gemini API를 사용하여 오디오 바이트를 텍스트로 전사합니다."""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    
    # OpenAI Whisper API 시도 (timeout 설정으로 빠른 응답)
    client = st.session_state.openai_client
    if client is not None:
        try:
            from openai import OpenAI
            # timeout 설정으로 빠른 응답
            if not hasattr(client, '_client') or not hasattr(client._client, 'timeout'):
                # timeout이 설정되지 않은 경우 새 클라이언트 생성
                api_key = get_api_key("openai")
                if api_key:
                    client = OpenAI(api_key=api_key, timeout=8.0)  # 8초 timeout
            
            with open(tmp.name, "rb") as f:
                if auto_detect or lang_code is None:
                    res = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="text",
                    )
                else:
                    whisper_lang = {"ko": "ko", "en": "en", "ja": "ja"}.get(lang_code, "en")
                    res = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="text",
                        language=whisper_lang,
                    )
            result = res.text.strip() if hasattr(res, 'text') else str(res).strip()
            if result:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
                return result
        except Exception as e:
            print(f"OpenAI Whisper failed: {e}")
    
    # Gemini API fallback
    gemini_key = get_api_key("gemini")
    if gemini_key and GENAI_AVAILABLE and genai is not None:
        try:
            import base64
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            
            lang_prompt = ""
            if lang_code:
                lang_map = {"ko": "한국어", "en": "English", "ja": "日本語"}
                lang_prompt = f"이 오디오는 {lang_map.get(lang_code, 'English')}로 말하고 있습니다. "
            
            prompt = f"{lang_prompt}이 오디오를 텍스트로 전사해주세요. 오직 전사된 텍스트만 반환하세요."
            
            try:
                audio_file = genai.upload_file(path=tmp.name, mime_type=mime_type)
                time.sleep(1)
                response = model.generate_content([prompt, audio_file])
                result = response.text.strip() if response.text else ""
                
                try:
                    genai.delete_file(audio_file.name)
                except Exception:
                    pass
                
                if result:
                    try:
                        os.remove(tmp.name)
                    except OSError:
                        pass
                    return result
                else:
                    raise Exception("Gemini returned empty result")
            except Exception as upload_err:
                print(f"Gemini file upload failed: {upload_err}")
                raise upload_err
        except Exception as e:
            print(f"Gemini transcription failed: {e}")
            try:
                os.remove(tmp.name)
            except OSError:
                pass
            return f"❌ {L.get('whisper_client_error', '전사 실패')}: OpenAI와 Gemini 모두 실패했습니다. ({str(e)[:100]})"
    else:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        return f"❌ {L.get('openai_missing', 'OpenAI API Key가 필요합니다.')} 또는 Gemini API Key가 필요합니다."


def transcribe_audio(audio_bytes, filename="audio.wav"):
    """오디오를 텍스트로 전사합니다."""
    client = st.session_state.openai_client

    if client:
        try:
            import io
            bio = io.BytesIO(audio_bytes)
            bio.name = filename
            resp = client.audio.transcriptions.create(model="whisper-1", file=bio)
            return resp.text
        except Exception as e:
            print("Whisper OpenAI failed:", e)

    if GENAI_AVAILABLE and genai is not None:
        try:
            genai.configure(api_key=get_api_key("gemini"))
            model = genai.GenerativeModel("gemini-2.5-flash")
            text = model.generate_content("Transcribe this audio:").text
            return text or ""
        except Exception as e:
            print("Gemini STT failed:", e)

    return "❌ STT not available"


def synthesize_tts(text: str, lang_key: str, role: str = "agent"):
    """TTS로 텍스트를 음성으로 변환합니다."""
    if not lang_key or lang_key not in ["ko", "en", "ja"]:
        lang_key = st.session_state.get("language", "ko")
        if lang_key not in ["ko", "en", "ja"]:
            lang_key = "ko"
    
    L = LANG.get(lang_key, LANG["ko"])
    client = st.session_state.openai_client
    if client is None:
        return None, L.get("openai_missing", "OpenAI API Key가 필요합니다.")

    # 고객 역할인 경우 성별에 따라 음성 선택
    if role == "customer":
        customer_gender = st.session_state.customer_avatar.get("gender", "male")
        if customer_gender == "female":
            voice_key = "customer_female"
        else:
            voice_key = "customer_male"
        
        if voice_key in TTS_VOICES:
            voice_name = TTS_VOICES[voice_key]["voice"]
        else:
            voice_name = TTS_VOICES["customer"]["voice"]
    elif role == "agent":
        # 에이전트 성별에 따라 음성 선택
        agent_gender = st.session_state.get("agent_gender", "female")  # 기본값: 여성
        if agent_gender == "male":
            voice_name = "alloy"  # 남성 음성
        else:
            voice_name = "shimmer"  # 여성 음성
    elif role in TTS_VOICES:
        voice_name = TTS_VOICES[role]["voice"]
    else:
        voice_name = TTS_VOICES["agent"]["voice"]

    try:
        resp = client.audio.speech.create(
            model="tts-1",
            voice=voice_name,
            input=text
        )
        return resp.read(), L["tts_status_success"]
    except Exception as e:
        return None, f"{L['tts_status_error']}: {e}"


def render_tts_button(text, lang_key, role="customer", prefix="", index: int = -1):
    """TTS 재생 버튼을 렌더링합니다."""
    if not lang_key or lang_key not in ["ko", "en", "ja"]:
        lang_key = st.session_state.get("language", "ko")
        if lang_key not in ["ko", "en", "ja"]:
            lang_key = "ko"
    
    L = LANG.get(lang_key, LANG["ko"])

    if index == -1:
        import hashlib
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()
        session_id_part = st.session_state.get('sim_instance_id', 'default_session')
        lang_code = st.session_state.get('language', lang_key)
        safe_key = f"{prefix}_SUMMARY_{session_id_part}_{lang_code}_{content_hash}"
    else:
        import hashlib
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()
        safe_key = f"{prefix}_{index}_{content_hash}"

    if st.button(L["button_listen_audio"], key=safe_key):
        if not st.session_state.openai_client:
            st.error(L["openai_missing"])
            return

        with st.spinner(L["tts_status_generating"]):
            try:
                audio_bytes, msg = synthesize_tts(text, lang_key, role=role)
                if audio_bytes:
                    try:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                        st.success(msg)
                        time.sleep(3)
                    except Exception as e:
                        st.warning(f"오디오 재생 중 오류: {e}. 오디오 파일은 생성되었지만 자동 재생에 실패했습니다.")
                        st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                        st.success(msg)
                else:
                    st.error(msg)
                    time.sleep(1)
            except Exception as e:
                st.error(f"❌ TTS 생성 중 치명적인 오류 발생: {e}")
                time.sleep(1)
            return


def load_voice_records() -> List[Dict[str, Any]]:
    """음성 녹음 기록을 로드합니다."""
    return _load_json(VOICE_META_FILE, [])


def save_voice_records(records: List[Dict[str, Any]]):
    """음성 녹음 기록을 저장합니다."""
    _save_json(VOICE_META_FILE, records)


def save_audio_record_local(
        audio_bytes: bytes,
        filename: str,
        transcript_text: str,
        mime_type: str = "audio/webm",
        meta: Dict[str, Any] = None,
) -> str:
    """오디오 녹음을 로컬에 저장합니다."""
    records = load_voice_records()
    rec_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    ext = filename.split(".")[-1] if "." in filename else "webm"
    audio_filename = f"{rec_id}.{ext}"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    rec = {
        "id": rec_id,
        "created_at": ts,
        "filename": filename,
        "audio_filename": audio_filename,
        "size": len(audio_bytes),
        "transcript": transcript_text,
        "mime_type": mime_type,
        "language": st.session_state.language,
        "meta": meta or {},
    }
    records.insert(0, rec)
    save_voice_records(records)
    return rec_id


def delete_audio_record_local(rec_id: str) -> bool:
    """로컬 오디오 녹음을 삭제합니다."""
    records = load_voice_records()
    idx = next((i for i, r in enumerate(records) if r.get("id") == rec_id), None)
    if idx is None:
        return False
    rec = records.pop(idx)
    audio_filename = rec.get("audio_filename")
    if audio_filename:
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        try:
            os.remove(audio_path)
        except FileNotFoundError:
            pass
    save_voice_records(records)
    return True


def get_audio_bytes_local(rec_id: str):
    """로컬 오디오 녹음의 바이트를 가져옵니다."""
    records = load_voice_records()
    rec = next((r for r in records if r.get("id") == rec_id), None)
    if not rec:
        raise FileNotFoundError("record not found")
    audio_filename = rec["audio_filename"]
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    with open(audio_path, "rb") as f:
        b = f.read()
    return b, rec

