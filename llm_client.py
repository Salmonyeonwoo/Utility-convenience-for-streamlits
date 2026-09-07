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
LLM 클라이언트 모듈
다양한 LLM API (Gemini, Claude, Groq 등)를 통합 관리합니다.
"""

import os
import streamlit as st
import time
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

import google.generativeai as genai

from config import SUPPORTED_APIS


def _telemetry_enabled() -> bool:
    return bool(st.session_state.get("telemetry_llm_enabled", False))


def _infer_last_turn_indices():
    """현재 세션 메시지 기준으로 마지막 고객/에이전트 메시지 인덱스를 추정."""
    msgs = st.session_state.get("simulator_messages") or []
    if not isinstance(msgs, list) or not msgs:
        return None, None

    last_customer_idx = None
    last_agent_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        role = (msgs[i] or {}).get("role")
        if last_customer_idx is None and role in ("customer", "customer_rebuttal", "initial_query", "phone_exchange"):
            last_customer_idx = i
        if last_agent_idx is None and role in ("agent_response", "agent"):
            last_agent_idx = i
        if last_customer_idx is not None and last_agent_idx is not None:
            break
    return last_customer_idx, last_agent_idx


def _infer_turn_key(stage: Optional[str], last_customer_idx, last_agent_idx) -> Optional[str]:
    if not stage:
        return None
    if stage == "AGENT_TURN":
        return f"AGENT_TURN:c{last_customer_idx}"
    if stage == "CUSTOMER_TURN":
        return f"CUSTOMER_TURN:a{last_agent_idx}"
    if stage in ("WAIT_FIRST_QUERY", "CLOSING"):
        return stage
    return f"{stage}:c{last_customer_idx}:a{last_agent_idx}"


def _append_llm_event(event: dict) -> None:
    events = st.session_state.get("llm_call_events")
    if not isinstance(events, list):
        events = []
    events.append(event)
    max_events = int(st.session_state.get("llm_call_events_max", 200) or 200)
    if max_events < 20:
        max_events = 20
    if len(events) > max_events:
        events = events[-max_events:]
    st.session_state.llm_call_events = events


def get_api_key(api):
    """API 키를 가져옵니다 (Streamlit Secrets > 환경변수 > 세션 상태 순서)"""
    cfg = SUPPORTED_APIS.get(api, {})
    secret_key = cfg.get("secret_key", f"{api.upper()}_API_KEY")

    # 1. Streamlit Secrets (.streamlit/secrets.toml) - 최우선
    try:
        if hasattr(st, "secrets") and secret_key in st.secrets:
            key = st.secrets[secret_key]
            if key and str(key).strip():
                return str(key).strip()
    except Exception:
        pass

    # 2. Environment Variable (os.environ) - 대소문자 구분 없이 확인
    env_key = os.environ.get(secret_key)
    if not env_key:
        env_key = os.environ.get(secret_key.upper())
    if not env_key:
        env_key = os.environ.get(secret_key.lower())
    if env_key and env_key.strip():
        return env_key.strip()

    # 3. User Input (Session State)
    session_key = cfg.get("session_key", f"user_{api}_key")
    user_key = st.session_state.get(session_key, "")
    if user_key and str(user_key).strip():
        return str(user_key).strip()

    return ""


def get_llm_client():
    """선택된 모델에 맞는 클라이언트 + 모델코드 반환"""
    model_key = st.session_state.get("selected_llm", "gemini_flash")

    # --- Gemini ---
    if model_key.startswith("gemini") or model_key in ("gemini_flash", "gemini_pro", "gemini_2_0"):
        key = get_api_key("gemini")
        if not key: 
            return None, None
        try:
            genai.configure(api_key=key)
            if model_key == "gemini_pro":
                model_name = "gemini-1.5-pro"
            elif model_key in ("gemini_2_0", "gemini_flash_2_0", "gemini-2.0-flash"):
                model_name = "gemini-2.0-flash"
            else:
                model_name = "gemini-1.5-flash"
            return genai, ("gemini", model_name)
        except Exception:
            return None, None

    # --- Claude ---
    if model_key.startswith("claude"):
        if Anthropic is None:
            return None, None
        key = get_api_key("claude")
        if not key: 
            return None, None
        try:
            client = Anthropic(api_key=key)
            model_name = "claude-3-5-sonnet-latest"
            return client, ("claude", model_name)
        except Exception:
            return None, None

    # --- Groq ---
    if model_key.startswith("groq"):
        try:
            from groq import Groq
        except ImportError:
            return None, None
        key = get_api_key("groq")
        if not key: 
            return None, None
        try:
            client = Groq(api_key=key)
            model_name = (
                "llama3-70b-8192"
                if "llama3" in model_key
                else "mixtral-8x7b-32768"
            )
            return client, ("groq", model_name)
        except Exception:
            return None, None

    # --- OpenAI ---
    if model_key.startswith("openai"):
        if OpenAI is None:
            return None, None
        key = get_api_key("openai")
        if not key: 
            return None, None
        try:
            client = OpenAI(api_key=key)
            model_name = "gpt-4o" if model_key == "openai_gpt4" else "gpt-3.5-turbo"
            return client, ("openai", model_name)
        except Exception:
            return None, None

    # Fallback to Gemini if key exists
    gemini_key = get_api_key("gemini")
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            return genai, ("gemini", "gemini-1.5-flash")
        except Exception:
            pass

    return None, None


def run_llm(prompt: str, max_tokens: int = 2000) -> str:
    """
    선택된 LLM으로 프롬프트 실행 (Gemini 최우선 적용)
    
    Args:
        prompt: LLM에 전달할 프롬프트
        max_tokens: 최대 토큰 수 (기본값: 2000, 채팅 응답에 적합)
    """
    client, info = get_llm_client()
    provider, model_name = info if info else (None, None)

    # Fallback 순서 정의 (Gemini 우선)
    llm_attempts = []

    # 1. Gemini
    gemini_key = get_api_key("gemini")
    if gemini_key:
        if model_name and "pro" in str(model_name):
            g_model = "gemini-1.5-pro"
        elif model_name and ("2.0" in str(model_name) or "2_0" in str(model_name)):
            g_model = "gemini-2.0-flash"
        else:
            g_model = "gemini-1.5-flash"
        llm_attempts.append(("gemini", gemini_key, g_model))

    # 2. Claude
    claude_key = get_api_key("claude")
    if claude_key and Anthropic is not None:
        llm_attempts.append(("claude", claude_key, "claude-3-5-sonnet-latest"))

    # 3. Groq
    groq_key = get_api_key("groq")
    if groq_key:
        groq_model = "llama3-70b-8192" if (model_name and "llama3" in str(model_name)) else "mixtral-8x7b-32768"
        llm_attempts.append(("groq", groq_key, groq_model))

    # 4. OpenAI
    openai_key = get_api_key("openai")
    if openai_key and OpenAI is not None:
        llm_attempts.append(("openai", openai_key, "gpt-4o" if (model_name and "4" in str(model_name)) else "gpt-3.5-turbo"))

    # 주력 모델이 Fallback 목록에 있으면 0번 인덱스로 올리기
    if provider and provider in [attempt[0] for attempt in llm_attempts]:
        primary_attempt = next((attempt for attempt in llm_attempts if attempt[0] == provider), None)
        if primary_attempt:
            llm_attempts.remove(primary_attempt)
            llm_attempts.insert(0, primary_attempt)

    if not llm_attempts:
        return "❌ 사용 가능한 LLM API 키가 설정되지 않았습니다. .streamlit/secrets.toml 또는 환경변수에 GEMINI_API_KEY를 설정해주세요."

    last_error_msg = ""
    for provider, key, model in llm_attempts:
        if not key: 
            continue

        try:
            t0 = time.perf_counter()
            if provider == "gemini":
                genai.configure(api_key=key)
                effective_model = model.replace("gemini-2.5", "gemini-1.5")
                gen_model = genai.GenerativeModel(effective_model)
                generation_config = {
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7,
                }
                resp = gen_model.generate_content(prompt, generation_config=generation_config)
                if _telemetry_enabled():
                    stage = st.session_state.get("sim_stage")
                    last_customer_idx, last_agent_idx = _infer_last_turn_indices()
                    _append_llm_event({
                        "ts": time.time(),
                        "dur_ms": int((time.perf_counter() - t0) * 1000),
                        "status": "success",
                        "provider": provider,
                        "model": effective_model,
                        "tag": st.session_state.get("_llm_call_tag"),
                        "stage": stage,
                        "turn_key": _infer_turn_key(stage, last_customer_idx, last_agent_idx),
                        "last_customer_idx": last_customer_idx,
                        "last_agent_idx": last_agent_idx,
                        "prompt_chars": len(prompt or ""),
                        "max_tokens": max_tokens,
                        "rerun_seq": st.session_state.get("rerun_seq"),
                        "feature_id": st.session_state.get("feature_selection_id"),
                    })
                return resp.text

            elif provider == "claude" and Anthropic:
                c_client = Anthropic(api_key=key, timeout=10.0)
                resp = c_client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                if _telemetry_enabled():
                    stage = st.session_state.get("sim_stage")
                    last_customer_idx, last_agent_idx = _infer_last_turn_indices()
                    _append_llm_event({
                        "ts": time.time(),
                        "dur_ms": int((time.perf_counter() - t0) * 1000),
                        "status": "success",
                        "provider": provider,
                        "model": model,
                        "tag": st.session_state.get("_llm_call_tag"),
                        "stage": stage,
                        "turn_key": _infer_turn_key(stage, last_customer_idx, last_agent_idx),
                        "last_customer_idx": last_customer_idx,
                        "last_agent_idx": last_agent_idx,
                        "prompt_chars": len(prompt or ""),
                        "max_tokens": max_tokens,
                        "rerun_seq": st.session_state.get("rerun_seq"),
                        "feature_id": st.session_state.get("feature_selection_id"),
                    })
                return resp.content[0].text

            elif provider == "groq":
                from groq import Groq
                g_client = Groq(api_key=key, timeout=10.0)
                resp = g_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                if _telemetry_enabled():
                    stage = st.session_state.get("sim_stage")
                    last_customer_idx, last_agent_idx = _infer_last_turn_indices()
                    _append_llm_event({
                        "ts": time.time(),
                        "dur_ms": int((time.perf_counter() - t0) * 1000),
                        "status": "success",
                        "provider": provider,
                        "model": model,
                        "tag": st.session_state.get("_llm_call_tag"),
                        "stage": stage,
                        "turn_key": _infer_turn_key(stage, last_customer_idx, last_agent_idx),
                        "last_customer_idx": last_customer_idx,
                        "last_agent_idx": last_agent_idx,
                        "prompt_chars": len(prompt or ""),
                        "max_tokens": max_tokens,
                        "rerun_seq": st.session_state.get("rerun_seq"),
                        "feature_id": st.session_state.get("feature_selection_id"),
                    })
                return resp.choices[0].message.content

            elif provider == "openai" and OpenAI:
                o_client = OpenAI(api_key=key, timeout=10.0)
                resp = o_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return resp.choices[0].message.content

        except Exception as e:
            last_error_msg = str(e)
            if _telemetry_enabled():
                stage = st.session_state.get("sim_stage")
                last_customer_idx, last_agent_idx = _infer_last_turn_indices()
                _append_llm_event({
                    "ts": time.time(),
                    "dur_ms": int((time.perf_counter() - t0) * 1000) if "t0" in locals() else None,
                    "status": "error",
                    "provider": provider,
                    "model": model,
                    "tag": st.session_state.get("_llm_call_tag"),
                    "stage": stage,
                    "turn_key": _infer_turn_key(stage, last_customer_idx, last_agent_idx),
                    "last_customer_idx": last_customer_idx,
                    "last_agent_idx": last_agent_idx,
                    "prompt_chars": len(prompt or ""),
                    "max_tokens": max_tokens,
                    "rerun_seq": st.session_state.get("rerun_seq"),
                    "feature_id": st.session_state.get("feature_selection_id"),
                    "error": str(e)[:300],
                })
            print(f"LLM {provider} ({model}) failed: {e}")
            continue

    if "ResourceExhausted" in last_error_msg or "429" in last_error_msg:
        return "⚠️ Gemini API 무료 할당량(RPM/TPM)이 일시적으로 초과되었습니다. 잠시 후(약 10~30초 뒤) 다시 시도해 주세요."
    return f"❌ 모든 LLM API 호출에 실패했습니다. (오류: {last_error_msg[:100] if last_error_msg else 'API 키를 확인해주세요'})"


def init_openai_audio_client():
    """Whisper / TTS 용 Gemini Client 초기화"""
    key = get_api_key("gemini")
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        return genai
    except Exception:
        return None
