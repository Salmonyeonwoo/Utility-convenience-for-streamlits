# ========================================
# utils/customer_analysis_profile.py
# 고객 분석 - 프로필 분석 모듈
# ========================================

import json
import re
import streamlit as st
from typing import Dict, Any
from llm_client import run_llm
from utils.customer_analysis_language import detect_text_language


def analyze_customer_profile(customer_query: str, current_lang_key: str = None) -> Dict[str, Any]:
    """신규 고객의 문의사항과 말투를 분석하여 고객성향 점수를 실시간으로 계산"""
    try:
        detected_lang = detect_text_language(customer_query)
    except Exception as e:
        print(f"Language detection failed in analyze_customer_profile: {e}")
        detected_lang = "ko"
    
    lang_key_to_use = current_lang_key if current_lang_key else detected_lang
    if lang_key_to_use not in ["ko", "en", "ja"]:
        lang_key_to_use = "ko"
    
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[lang_key_to_use]

    analysis_prompt = f"""
You are an AI analyst analyzing a customer's inquiry to determine their profile and sentiment.

Analyze the following customer inquiry and provide a structured analysis in JSON format (ONLY JSON, no markdown).

Analyze:
1. Customer gender (male/female/unknown - analyze based on name, language patterns, or cultural hints)
2. Customer sentiment score (0-100, where 0=very negative/angry, 50=neutral, 100=very positive/happy)
3. Communication style (formal/casual, brief/detailed, polite/direct)
4. Urgency level (low/medium/high)
5. Customer type prediction (normal/difficult/very_dissatisfied)
6. Language and cultural hints (if any)
7. Key concerns or pain points

Output format (JSON only):
{{
  "gender": "male",
  "sentiment_score": 45,
  "communication_style": "brief, direct, slightly frustrated",
  "urgency_level": "high",
  "predicted_customer_type": "difficult",
  "cultural_hints": "unknown",
  "key_concerns": ["issue 1", "issue 2"],
  "tone_analysis": "brief description of tone"
}}

Customer Inquiry:
{customer_query}

JSON Output:
"""

    if not st.session_state.is_llm_ready:
        return {
            "gender": "unknown",
            "sentiment_score": 50,
            "communication_style": "unknown",
            "urgency_level": "medium",
            "predicted_customer_type": "normal",
            "cultural_hints": "unknown",
            "key_concerns": [],
            "tone_analysis": "Unable to analyze"
        }

    try:
        analysis_text = run_llm(analysis_prompt).strip()
        
        # JSON 추출
        if "```" in analysis_text:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', analysis_text, re.DOTALL)
            if json_match:
                analysis_text = json_match.group(1)
            else:
                analysis_text = re.sub(r'```(?:json)?\s*', '', analysis_text)
                analysis_text = re.sub(r'\s*```', '', analysis_text)
        
        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
        if json_match:
            analysis_text = json_match.group(0)
        
        analysis_text = analysis_text.strip()
        
        try:
            analysis_data = json.loads(analysis_text)
        except json.JSONDecodeError as json_err:
            print(f"고객 분석 JSON 파싱 오류: {json_err}")
            analysis_data = {
                "gender": "unknown",
                "sentiment_score": 50,
                "communication_style": "unknown",
                "urgency_level": "medium",
                "predicted_customer_type": "normal",
                "cultural_hints": "unknown",
                "key_concerns": [],
                "tone_analysis": f"Analysis error: {str(json_err)}"
            }
        
        return analysis_data
    except Exception as e:
        return {
            "gender": "unknown",
            "sentiment_score": 50,
            "communication_style": "unknown",
            "urgency_level": "medium",
            "predicted_customer_type": "normal",
            "cultural_hints": "unknown",
            "key_concerns": [],
            "tone_analysis": f"Analysis error: {str(e)}"
        }
