# ========================================
# services/agent_api_client.py
# AI 및 외부 백그라운드 에이전트 통신 어댑터
# (Gemini 실시간 추론 연동 & OpenAI Dots 규격 지원)
# ========================================
import os
import time
import requests
from typing import Dict, Any, Optional

try:
    import streamlit as st
    STREAMLIT_ENV = True
except ImportError:
    STREAMLIT_ENV = False


class AgentApiClient:
    """
    LLM(Gemini, OpenAI 등) 및 외부 백그라운드 에이전트(OpenAI Dots, LangGraph 등)와
    통신을 담당하는 독립 클라이언트 클래스입니다.
    기존 .env, st.secrets, llm_client 등록 키를 자동 탐색하여 실시간 추론을 수행합니다.
    """

    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or self._resolve_api_key()

    def _resolve_api_key(self) -> str:
        """Gemini 및 OpenAI API 키를 다양한 소스에서 안전하게 탐색"""
        # 1. os.environ 확인
        for key_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key"]:
            k = os.environ.get(key_name)
            if k:
                return k.strip()

        # 2. Streamlit Secrets 확인
        if STREAMLIT_ENV:
            try:
                for key_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
                    if key_name in st.secrets:
                        return str(st.secrets[key_name]).strip()
            except Exception:
                pass

        # 3. llm_client 함수 시도
        try:
            from llm_client import get_api_key
            k = get_api_key("gemini")
            if k:
                return k.strip()
        except Exception:
            pass

        return ""

    def call_reasoning_llm(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1500) -> str:
        """
        LLM 추론 호출:
        실제 Gemini API 키 존재 시 실시간 호출을 수행하고,
        키 미설정 또는 네트워크 오류 시 빈 문자열을 반환하여 상위 엔진이 지능형 BPO 실무 추론으로 폴백하도록 합니다.
        """
        gemini_key = self.api_key or self._resolve_api_key()

        # 1. google.generativeai 라이브러리 시도
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                
                # 모델명 호환성 (gemini-1.5-flash -> gemini-pro)
                model = None
                for m_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
                    try:
                        model = genai.GenerativeModel(m_name)
                        break
                    except Exception:
                        continue

                if model:
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    resp = model.generate_content(full_prompt)
                    if resp and resp.text:
                        return resp.text.strip()
            except Exception:
                pass

            # 2. REST API 직접 호출 시도 (SDK 미설치 또는 환경 충돌 대비)
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                payload = {
                    "contents": [{"parts": [{"text": full_text}]}]
                }
                res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=20)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
            except Exception:
                pass

        # 3. OpenAI API 폴백 시도
        openai_key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if STREAMLIT_ENV and hasattr(st, "secrets") else None)
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                return completion.choices[0].message.content.strip()
            except Exception:
                pass

        return ""

    def dispatch_remote_task(self, endpoint_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        향후 독립된 백그라운드 AI 에이전트(OpenAI Dots / FastAPI Worker / LangGraph Server)로
        비동기 작업을 전송하기 위한 표준 HTTP/Webhook 디스패치 인터페이스 규격입니다.
        """
        return {
            "status": "DISPATCHED_TO_BACKGROUND_AGENT",
            "task_id": f"TASK-{int(time.time() * 1000)}",
            "target_endpoint": endpoint_url or "https://api.openai.com/v1/agents/tasks",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "payload_summary": {
                "mission": payload.get("task_type"),
                "target_customer": payload.get("target_customer_name"),
                "step_count": len(payload.get("steps", []))
            }
        }
