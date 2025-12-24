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
번역 관련 모듈
LLM을 사용한 텍스트 번역 기능을 제공합니다.
"""

from typing import Tuple
from llm_client import get_api_key, run_llm

# Google Generative AI import (지연 로딩)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def translate_text_with_llm(text_content: str, target_lang_code: str, source_lang_code: str) -> Tuple[str, bool]:
    """
    주어진 텍스트를 LLM을 사용하여 대상 언어로 번역합니다.
    
    Returns:
        tuple: (translated_text, is_success) - 번역된 텍스트와 성공 여부
    """
    if not text_content or not text_content.strip():
        return text_content, True
    
    target_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(target_lang_code, "English")
    source_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(source_lang_code, "English")

    MAX_CHARS_PER_CHUNK = 200000
    CHUNK_OVERLAP = 1000
    
    if len(text_content) > MAX_CHARS_PER_CHUNK:
        translated_chunks = []
        start_idx = 0
        
        while start_idx < len(text_content):
            end_idx = min(start_idx + MAX_CHARS_PER_CHUNK, len(text_content))
            chunk = text_content[start_idx:end_idx]
            
            translated_chunk, success = translate_text_with_llm_chunk(chunk, target_lang_code, source_lang_code)
            if success:
                translated_chunks.append(translated_chunk)
            else:
                translated_chunks.append(chunk)
            
            start_idx = end_idx - CHUNK_OVERLAP if end_idx < len(text_content) else end_idx
        
        return "".join(translated_chunks), True
    else:
        return translate_text_with_llm_chunk(text_content, target_lang_code, source_lang_code)


def translate_text_with_llm_chunk(text_content: str, target_lang_code: str, source_lang_code: str) -> Tuple[str, bool]:
    """단일 청크를 번역하는 내부 함수"""
    target_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(target_lang_code, "English")
    source_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(source_lang_code, "English")

    system_prompt = (
        f"You are a professional translation AI. Translate the entire following customer support chat history "
        f"from '{source_lang_name}' to '{target_lang_name}'. "
        f"You MUST translate the content to {target_lang_name} ONLY. "
        f"Do not include any mixed languages, the source text, or any introductory/concluding remarks. "
        f"Output ONLY the translated chat history text. "
    )
    prompt = f"Original Chat History:\n\n{text_content}"

    llm_attempts = [
        ("openai", get_api_key("openai"), "gpt-4o"),
        ("gemini", get_api_key("gemini"), "gemini-2.5-flash"),
        ("claude", get_api_key("claude"), "claude-3-5-sonnet-latest"),
    ]

    last_error = ""

    for provider, key, model_name in llm_attempts:
        if not key:
            continue

        try:
            translated_text = ""

            if provider == "openai" and OpenAI:
                o_client = OpenAI(api_key=key)
                resp = o_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    temperature=0.1
                )
                translated_text = resp.choices[0].message.content.strip()

            elif provider == "gemini":
                if not GENAI_AVAILABLE or genai is None:
                    last_error = f"Gemini API not available: google.generativeai module not found"
                    continue
                genai.configure(api_key=key)
                g_model = genai.GenerativeModel(model_name)
                resp = g_model.generate_content(
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.1)
                )
                translated_text = resp.text.strip()

            elif provider == "claude":
                from anthropic import Anthropic
                c_client = Anthropic(api_key=key)
                resp = c_client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    system=system_prompt
                )
                translated_text = resp.content[0].text.strip()

            if translated_text and len(translated_text.strip()) > 0:
                return translated_text, True
            else:
                last_error = f"Translation failed: {provider} returned empty response."
                continue

        except Exception as e:
            last_error = f"Translation API call failed with {provider} ({model_name}): {e}"
            print(last_error)
            continue

    print(f"Translation failed: {last_error or 'No active API key found.'}. Returning original text.")
    return text_content, False




