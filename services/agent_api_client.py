# ========================================
# services/agent_api_client.py
# AI 및 외부 백그라운드 에이전트 통신 어댑터
# ========================================
import os
import time
from typing import Dict, Any, Optional

class AgentApiClient:
    """
    LLM(OpenAI, Gemini, Claude 등) 및 외부 백그라운드 에이전트(OpenAI Dots, LangGraph 등)와
    통신을 담당하는 독립 클라이언트 클래스입니다.
    Streamlit UI 런타임 종속성 없이 순수 파이썬 환경에서 동작합니다.
    """

    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get(f"{self.provider.upper()}_API_KEY", "")

    def call_reasoning_llm(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1500) -> str:
        """
        LLM 추론 호출:
        실제 API 키 존재 시 OpenAI / Gemini를 호출하고,
        키 미설정 또는 네트워크 오류 시 빈 문자열을 반환하여
        상위 서비스의 지능형 BPO 실무 추론 엔진이 폴백 처리할 수 있도록 합니다.
        """
        # 1. Gemini 호출 시도
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini_api_key") or self.api_key
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                resp = model.generate_content(full_prompt)
                if resp and resp.text:
                    return resp.text.strip()
            except Exception:
                pass

        # 2. OpenAI 호출 시도
        openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
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
