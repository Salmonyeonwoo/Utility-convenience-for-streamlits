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
다양한 LLM API (OpenAI, Gemini, Claude, Groq 등)를 통합 관리합니다.
"""

import os
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

from config import SUPPORTED_APIS


def get_api_key(api):
    """API 키를 가져옵니다 (Streamlit Secrets > 환경변수 > 세션 상태 순서)"""
    cfg = SUPPORTED_APIS[api]

    # ⭐ 1. Streamlit Secrets (.streamlit/secrets.toml) - 최우선
    try:
        if hasattr(st, "secrets") and cfg["secret_key"] in st.secrets:
            return st.secrets[cfg["secret_key"]]
    except Exception:
        pass

    # 2. Environment Variable (os.environ)
    env_key = os.environ.get(cfg["secret_key"])
    if env_key:
        return env_key

    # 3. User Input (Session State)
    user_key = st.session_state.get(cfg["session_key"], "")
    if user_key:
        return user_key

    return ""


def get_llm_client():
    """선택된 모델에 맞는 클라이언트 + 모델코드 반환"""
    model_key = st.session_state.get("selected_llm", "openai_gpt4")

    # --- OpenAI ---
    if model_key.startswith("openai"):
        key = get_api_key("openai")
        if not key: 
            return None, None
        try:
            client = OpenAI(api_key=key)
            model_name = "gpt-4o" if model_key == "openai_gpt4" else "gpt-3.5-turbo"
            return client, ("openai", model_name)
        except Exception:
            return None, None

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


def run_llm(prompt: str) -> str:
    """선택된 LLM으로 프롬프트 실행 (Gemini 우선순위 변경 적용)"""
    client, info = get_llm_client()

    # Note: info는 사이드바에서 선택된 주력 모델의 정보를 담고 있습니다.
    provider, model_name = info if info else (None, None)

    # Fallback 순서를 정의합니다. (Gemini 우선)
    llm_attempts = []

    # 1. Gemini를 최우선 Fallback으로 시도 (Keys 확인)
    gemini_key = get_api_key("gemini")
    if gemini_key:
        llm_attempts.append(("gemini", gemini_key, "gemini-2.5-pro" if "pro" in str(model_name) else "gemini-2.5-flash"))

    # 2. OpenAI를 2순위 Fallback으로 시도 (Keys 확인)
    openai_key = get_api_key("openai")
    if openai_key:
        llm_attempts.append(("openai", openai_key, "gpt-4o" if "4" in str(model_name) else "gpt-3.5-turbo"))

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
            # 주력 모델이 Gemini나 OpenAI가 아니라면, Fallback 순서와 관계없이 가장 먼저 시도하도록 삽입
            llm_attempts.insert(0, primary_attempt)

    # LLM 순차 실행
    for provider, key, model in llm_attempts:
        if not key: 
            continue

        try:
            if provider == "gemini":
                genai.configure(api_key=key)
                gen_model = genai.GenerativeModel(model)
                resp = gen_model.generate_content(prompt)
                return resp.text

            elif provider == "openai":
                o_client = OpenAI(api_key=key)
                resp = o_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

            elif provider == "claude":
                c_client = Anthropic(api_key=key)
                resp = c_client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text

            elif provider == "groq":
                from groq import Groq
                g_client = Groq(api_key=key)
                resp = g_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

        except Exception as e:
            # 해당 API가 실패하면 다음 API로 넘어갑니다.
            print(f"LLM {provider} ({model}) failed: {e}")
            continue

    # 모든 시도가 실패했을 때
    return "❌ 모든 LLM API 키가 작동하지 않거나 할당량이 소진되었습니다."


def init_openai_audio_client():
    """Whisper / TTS 용 OpenAI Client 초기화"""
    key = get_api_key("openai")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except:
        return None




























