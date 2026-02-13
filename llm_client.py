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
# from openai import OpenAI  # OpenAI API 키 결제 지원 중단으로 인해 비활성화
from anthropic import Anthropic
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
    # 너무 커지지 않도록 최근 N개만 유지
    max_events = int(st.session_state.get("llm_call_events_max", 200) or 200)
    if max_events < 20:
        max_events = 20
    if len(events) > max_events:
        events = events[-max_events:]
    st.session_state.llm_call_events = events


def get_api_key(api):
    """API 키를 가져옵니다 (Streamlit Secrets > 환경변수 > 세션 상태 순서)"""
    cfg = SUPPORTED_APIS[api]

    # ⭐ 1. Streamlit Secrets (.streamlit/secrets.toml) - 최우선
    try:
        if hasattr(st, "secrets") and cfg["secret_key"] in st.secrets:
            key = st.secrets[cfg["secret_key"]]
            if key and key.strip():
                return key.strip()
    except Exception:
        pass

    # 2. Environment Variable (os.environ) - 대소문자 구분 없이 확인
    env_key = os.environ.get(cfg["secret_key"])
    if not env_key:
        # 대소문자 변형도 확인
        env_key = os.environ.get(cfg["secret_key"].upper())
    if not env_key:
        env_key = os.environ.get(cfg["secret_key"].lower())
    if env_key and env_key.strip():
        return env_key.strip()

    # 3. User Input (Session State)
    user_key = st.session_state.get(cfg["session_key"], "")
    if user_key and user_key.strip():
        return user_key.strip()

    return ""


def get_llm_client():
    """선택된 모델에 맞는 클라이언트 + 모델코드 반환"""
    model_key = st.session_state.get("selected_llm", "gemini_pro")

    # --- OpenAI --- (비활성화: API 키 결제 지원 중단)
    # if model_key.startswith("openai"):
    #     key = get_api_key("openai")
    #     if not key: 
    #         return None, None
    #     try:
    #         client = OpenAI(api_key=key)
    #         model_name = "gpt-4o" if model_key == "openai_gpt4" else "gpt-3.5-turbo"
    #         return client, ("openai", model_name)
    #     except Exception:
    #         return None, None

    # --- Gemini ---
    if model_key.startswith("gemini"):
        key = get_api_key("gemini")
        if not key: 
            return None, None
        try:
            genai.configure(api_key=key)
            model_name = "gemini-2.5-pro" if model_key == "gemini_pro" else "gemini-2.5-flash"
            return genai, ("gemini", model_name)
        except Exception:
            return None, None

    # --- Claude ---
    if model_key.startswith("claude"):
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
        from groq import Groq
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

    return None, None


def run_llm(prompt: str, max_tokens: int = 2000) -> str:
    """
    선택된 LLM으로 프롬프트 실행 (Gemini 우선순위 변경 적용)
    
    Args:
        prompt: LLM에 전달할 프롬프트
        max_tokens: 최대 토큰 수 (기본값: 2000, 채팅 응답에 적합)
                    전화 응답 등 짧은 응답이 필요한 경우 200 등으로 조정 가능
    """
    client, info = get_llm_client()

    # Note: info는 사이드바에서 선택된 주력 모델의 정보를 담고 있습니다.
    provider, model_name = info if info else (None, None)

    # Fallback 순서를 정의합니다. (Gemini 우선)
    llm_attempts = []

    # 1. Gemini를 최우선 Fallback으로 시도 (Keys 확인)
    gemini_key = get_api_key("gemini")
    if gemini_key:
        llm_attempts.append(("gemini", gemini_key, "gemini-2.5-pro" if "pro" in str(model_name) else "gemini-2.5-flash"))

    # 2. OpenAI를 2순위 Fallback으로 시도 (Keys 확인) - 비활성화: API 키 결제 지원 중단
    # openai_key = get_api_key("openai")
    # if openai_key:
    #     llm_attempts.append(("openai", openai_key, "gpt-4o" if "4" in str(model_name) else "gpt-3.5-turbo"))

    # 3. Claude를 3순위 Fallback으로 시도 (Keys 확인)
    claude_key = get_api_key("claude")
    if claude_key:
        llm_attempts.append(("claude", claude_key, "claude-3-5-sonnet-latest"))

    # 4. Groq를 4순위 Fallback으로 시도 (Keys 확인)
    groq_key = get_api_key("groq")
    if groq_key:
        groq_model = "llama3-70b-8192" if "llama3" in str(model_name) else "mixtral-8x7b-32768"
        llm_attempts.append(("groq", groq_key, groq_model))

    # ⭐ 순서 조정: 주력 모델(사용자가 사이드바에서 선택한 모델)을 가장 먼저 시도합니다.
    # 만약 주력 모델이 Fallback 리스트에 포함되어 있다면, 그 모델을 첫 순서로 올립니다.
    if provider and provider in [attempt[0] for attempt in llm_attempts]:
            # 주력 모델을 리스트에서 찾아 제거
            primary_attempt = next((attempt for attempt in llm_attempts if attempt[0] == provider), None)
            if primary_attempt:
                llm_attempts.remove(primary_attempt)
                # 주력 모델이 Gemini가 아니라면, Fallback 순서와 관계없이 가장 먼저 시도하도록 삽입
                llm_attempts.insert(0, primary_attempt)

    # LLM 순차 실행
    for provider, key, model in llm_attempts:
        if not key: 
            continue

        try:
            t0 = time.perf_counter()
            if provider == "gemini":
                genai.configure(api_key=key)
                gen_model = genai.GenerativeModel(model)
                # 채팅 응답을 위한 충분한 토큰 수 설정
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
                return resp.text

            # OpenAI provider 처리 제거 (API 키 결제 지원 중단)
            # elif provider == "openai":
            #     o_client = OpenAI(api_key=key, timeout=10.0)  # 10초 timeout
            #     resp = o_client.chat.completions.create(
            #         model=model,
            #         messages=[{"role": "user", "content": prompt}],
            #         max_tokens=max_tokens,  # 채팅 응답을 위한 충분한 토큰 수
            #         temperature=0.7,
            #     )
            #     if _telemetry_enabled():
            #         stage = st.session_state.get("sim_stage")
            #         last_customer_idx, last_agent_idx = _infer_last_turn_indices()
            #         _append_llm_event({
            #             "ts": time.time(),
            #             "dur_ms": int((time.perf_counter() - t0) * 1000),
            #             "status": "success",
            #             "provider": provider,
            #             "model": model,
            #             "tag": st.session_state.get("_llm_call_tag"),
            #             "stage": stage,
            #             "turn_key": _infer_turn_key(stage, last_customer_idx, last_agent_idx),
            #             "last_customer_idx": last_customer_idx,
            #             "last_agent_idx": last_agent_idx,
            #             "prompt_chars": len(prompt or ""),
            #             "max_tokens": max_tokens,
            #             "rerun_seq": st.session_state.get("rerun_seq"),
            #             "feature_id": st.session_state.get("feature_selection_id"),
            #         })
            #     return resp.choices[0].message.content

            elif provider == "claude":
                c_client = Anthropic(api_key=key, timeout=10.0)  # 10초 timeout
                resp = c_client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,  # 채팅 응답을 위한 충분한 토큰 수
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
                g_client = Groq(api_key=key, timeout=10.0)  # 10초 timeout
                resp = g_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,  # 채팅 응답을 위한 충분한 토큰 수
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

        except Exception as e:
            if _telemetry_enabled():
                # 실패도 기록: Fallback 때문에 느려지는 경우 추적 가능
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
            # 해당 API가 실패하면 다음 API로 넘어갑니다.
            print(f"LLM {provider} ({model}) failed: {e}")
            continue

    # 모든 시도가 실패했을 때
    return "❌ 모든 LLM API 키가 작동하지 않거나 할당량이 소진되었습니다."


def init_openai_audio_client():
    """Whisper / TTS 용 Gemini Client 초기화"""
    key = get_api_key("gemini")
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        return genai
    except:
        return None




























