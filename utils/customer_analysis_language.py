# ========================================
# utils/customer_analysis_language.py
# 고객 분석 - 언어 감지 모듈
# ========================================

import streamlit as st
from llm_client import run_llm


def detect_text_language(text: str) -> str:
    """
    텍스트의 언어를 자동 감지합니다.
    Returns: "ko", "en", "ja" 중 하나 (기본값: "ko")
    """
    if not text or not text.strip():
        return "ko"  # 기본값
    
    try:
        # 간단한 휴리스틱: 일본어 문자(히라가나, 가타카나, 한자)가 많이 포함되어 있으면 일본어
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF')
        if japanese_chars > len(text) * 0.1:  # 10% 이상 일본어 문자
            return "ja"
        
        # 영어 문자 비율이 높으면 영어
        english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
        if english_chars > len(text) * 0.7:  # 70% 이상 영어 문자
            return "en"
        
        # LLM을 사용한 정확한 언어 감지 시도
        if st.session_state.is_llm_ready:
            try:
                detection_prompt = f"""Detect the language of the following text. Respond with ONLY one word: "ko" (Korean), "en" (English), or "ja" (Japanese).

Text: {text[:200]}

Language:"""
                detected = run_llm(detection_prompt).strip().lower()
                if detected and detected not in ["❌", "error", "failed"] and detected in ["ko", "en", "ja"]:
                    return detected
            except Exception as e:
                print(f"Language detection LLM call failed: {e}")
                pass
    except Exception as e:
        print(f"Language detection error: {e}")
        return "ko"
    
    return "ko"
