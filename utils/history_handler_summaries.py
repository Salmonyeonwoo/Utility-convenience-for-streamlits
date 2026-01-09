# ========================================
# utils/history_handler_summaries.py
# 이력 관리 - 요약 생성 함수들
# ========================================

import json
import re
import ast
from typing import List, Dict, Any
import streamlit as st
from llm_client import get_api_key, run_llm

def generate_call_summary(messages: List[Dict[str, Any]], initial_query: str, customer_type: str,
                          current_lang_key: str) -> Dict[str, Any]:
    """전화 통화 이력 요약 생성 (문의 내용 + 솔루션 요점만, 다국어 지원)"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    
    # 고객 문의 내용 추출
    customer_inquiries = []
    agent_solutions = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["customer", "customer_rebuttal"]:
            customer_inquiries.append(content)
        elif role in ["agent", "agent_response"]:
            agent_solutions.append(content)
    
    # 요약 프롬프트
    summary_prompt = f"""
You are an AI analyst summarizing a phone call customer support conversation. Generate a DETAILED and COMPREHENSIVE summary.

The summary MUST be in {lang_name} language.

Conversation:
Initial Query: {initial_query}

Customer Messages:
{chr(10).join([f"- {inq}" for inq in customer_inquiries])}

Agent Responses:
{chr(10).join([f"- {sol}" for sol in agent_solutions])}

Generate a JSON summary with the following structure (ONLY VALID JSON, no markdown, no code blocks):
{{
    "customer_inquiry": "EXTREMELY DETAILED and COMPREHENSIVE description (at least 6-8 sentences) of what the customer asked about. Include: all specific questions asked in detail, concerns expressed with context, requirements mentioned with specifics, exact locations/products/services referenced with full names, any constraints or limitations mentioned, the context and background of their inquiry, and any related information they shared. Be extremely thorough and include all nuances, details, and context.",
    "key_solutions": [
        "EXTREMELY DETAILED solution point 1 (at least 3-4 sentences): Provide comprehensive explanation including all specific information provided by the agent, detailed step-by-step guidance if applicable, all important details and specifics, any limitations or exceptions mentioned with full context, follow-up actions required, and any additional relevant information shared",
        "EXTREMELY DETAILED solution point 2 (at least 3-4 sentences): Provide comprehensive explanation including all specific information provided by the agent, detailed step-by-step guidance if applicable, all important details and specifics, any limitations or exceptions mentioned with full context, follow-up actions required, and any additional relevant information shared",
        "EXTREMELY DETAILED solution point 3 (at least 3-4 sentences): Provide comprehensive explanation including all specific information provided by the agent, detailed step-by-step guidance if applicable, all important details and specifics, any limitations or exceptions mentioned with full context, follow-up actions required, and any additional relevant information shared",
        "Additional extremely detailed solution points if applicable (each at least 3-4 sentences with full context and specifics)"
    ],
    "summary": "EXTREMELY COMPREHENSIVE overall summary in {lang_name} (at least 12-15 sentences, preferably more). Must be a detailed narrative covering: 1) The customer's main inquiry in full detail with all context, 2) All specific questions asked by the customer with their exact wording and intent, 3) Extremely detailed solutions provided by the agent with all specific information, step-by-step explanations, and complete context, 4) All important clarifications or follow-up information shared with full details, 5) Any limitations, exceptions, or important notes mentioned with complete context, 6) The resolution status and next steps if applicable with specifics, 7) All specific details like exact locations (with full names), product names (with full names), service types (with full descriptions), procedures (with step-by-step details), dates, prices, or any other concrete information mentioned. Write as a comprehensive, flowing narrative that tells the complete, detailed story of the entire conversation with all nuances and specifics."
}}

CRITICAL JSON FORMATTING REQUIREMENTS:
- Output ONLY valid JSON (no markdown, no code blocks, no explanations)
- Escape all special characters in strings (quotes as \\", newlines as \\n, etc.)
- Ensure all strings are properly closed with quotes
- Do NOT include any text before or after the JSON object
- Be EXTREMELY DETAILED and COMPREHENSIVE (not brief at all, provide extensive information)
- Include ALL specific information mentioned (exact locations with full names, product names with full names, service types with full descriptions, procedures with step-by-step details, dates, prices, limitations, exceptions, etc.)
- key_solutions MUST be extremely detailed explanations (each should be 3-4 sentences minimum, provide comprehensive information, not just brief points)
- summary MUST be an extremely comprehensive narrative (12-15 sentences minimum, preferably 15-20 sentences)
- Use {lang_name} language
- Include ALL relevant details, nuances, context, and specifics from the conversation
- Do NOT summarize - provide EXTENSIVE detailed information
- Write as if you are providing a complete, detailed report of the conversation
"""
    
    try:
        if get_api_key("gemini") or get_api_key("openai"):
            summary_json = run_llm(summary_prompt)
            
            # JSON 추출
            if "```" in summary_json:
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', summary_json, re.DOTALL)
                if json_match:
                    summary_json = json_match.group(1)
                else:
                    summary_json = re.sub(r'```(?:json)?\s*', '', summary_json)
                    summary_json = re.sub(r'\s*```', '', summary_json)
            
            json_match = re.search(r'\{.*\}', summary_json, re.DOTALL)
            if json_match:
                summary_json = json_match.group(0)
            
            summary_json = summary_json.strip()
            
            # JSON 파싱 시도
            try:
                summary_data = json.loads(summary_json)
            except json.JSONDecodeError as json_err:
                try:
                    def fix_json_string(match):
                        key = match.group(1)
                        value = match.group(2)
                        value = re.sub(r'(?<!\\)"(?![,\s}])', '\\"', value)
                        return f'"{key}": "{value}"'
                    
                    summary_json_fixed = re.sub(r'"([^"]+)":\s*"([^"]*)"', fix_json_string, summary_json)
                    summary_data = json.loads(summary_json_fixed)
                except (json.JSONDecodeError, Exception):
                    try:
                        summary_json_py = summary_json.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                        summary_data = ast.literal_eval(summary_json_py)
                        if isinstance(summary_data, dict):
                            summary_data = {k: (None if v is None else v) for k, v in summary_data.items()}
                    except Exception:
                        customer_inquiry_match = re.search(r'"customer_inquiry"\s*:\s*"([^"]*)"', summary_json)
                        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', summary_json)
                        key_solutions_matches = re.findall(r'"key_solutions"\s*:\s*\[(.*?)\]', summary_json, re.DOTALL)
                        
                        customer_inquiry = customer_inquiry_match.group(1) if customer_inquiry_match else initial_query
                        summary_text = summary_match.group(1) if summary_match else f"Phone call conversation about {initial_query}"
                        
                        key_solutions = []
                        if key_solutions_matches:
                            solutions_text = key_solutions_matches[0]
                            solution_matches = re.findall(r'"([^"]*)"', solutions_text)
                            key_solutions = solution_matches[:5]
                        
                        summary_data = {
                            "customer_inquiry": customer_inquiry,
                            "key_solutions": key_solutions,
                            "summary": summary_text,
                            "summary_ko": "",
                            "summary_en": "",
                            "summary_ja": ""
                        }
                        
                        print(f"JSON 파싱 오류 (모든 재시도 실패): {json_err}")
                        print(f"원본 JSON (처음 1000자): {summary_json[:1000]}")
            
            # 다국어 번역 추가
            summary_data["summary_ko"] = summary_data.get("summary", "")
            summary_data["summary_en"] = ""
            summary_data["summary_ja"] = ""
            
            # 영어 번역
            if current_lang_key != "en":
                try:
                    from utils.translation import translate_text_with_llm
                    summary_data["summary_en"] = translate_text_with_llm(
                        summary_data.get("summary", ""), "en", current_lang_key
                    ) or ""
                except:
                    pass
            
            # 일본어 번역
            if current_lang_key != "ja":
                try:
                    from utils.translation import translate_text_with_llm
                    summary_data["summary_ja"] = translate_text_with_llm(
                        summary_data.get("summary", ""), "ja", current_lang_key
                    ) or ""
                except:
                    pass
            
            return summary_data
        else:
            return {
                "customer_inquiry": initial_query,
                "key_solutions": [],
                "summary": f"Phone call conversation about {initial_query}",
                "summary_ko": "",
                "summary_en": "",
                "summary_ja": ""
            }
    except Exception as e:
        return {
            "customer_inquiry": initial_query,
            "key_solutions": [],
            "summary": f"Error generating summary: {str(e)}",
            "summary_ko": "",
            "summary_en": "",
            "summary_ja": ""
        }

def generate_chat_summary(messages: List[Dict[str, Any]], initial_query: str, customer_type: str,
                          current_lang_key: str) -> Dict[str, Any]:
    """채팅 내용을 AI로 요약하여 주요 정보와 점수를 추출"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    conversation_text = f"Initial Query: {initial_query}\n\n"
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["customer", "customer_rebuttal", "phone_exchange"]:
            conversation_text += f"Customer: {content}\n"
        elif role == "agent_response" or role == "agent":
            conversation_text += f"Agent: {content}\n"

    summary_prompt = f"""
You are an AI analyst summarizing a customer support conversation. Your task is to extract comprehensive customer profile data and score various aspects numerically.

Analyze the conversation and provide a structured summary in JSON format (ONLY JSON, no markdown).

Extract and score:
1. Main inquiry topic (what the customer asked about)
2. Key responses provided by the agent (list of max 3 core actions/solutions)
3. Customer sentiment score (0-100, where 0=very negative, 50=neutral, 100=very positive)
4. Customer satisfaction score (0-100, based on final response)
5. Customer characteristics with detailed scoring:
   - Language preference (detected language code: ko/en/ja)
   - Cultural background hints (score 0-100, where higher = more cultural context detected)
   - Location/region (general region only, anonymize specific addresses)
   - Communication style (formal/casual, brief/detailed) with scores:
     * Formality score (0-100, 0=casual, 100=very formal)
     * Detail level score (0-100, 0=brief, 100=very detailed)
   - Customer personality traits (score each 0-100):
     * Patience level (0-100)
     * Assertiveness (0-100)
     * Politeness level (0-100)
     * Technical proficiency (0-100, if technical inquiry)
6. Privacy-sensitive information (anonymize: names, emails, phone numbers, specific addresses)
   - Extract patterns only (e.g., "email provided", "phone number provided", "resides in Asia region")
7. Customer behavior patterns:
   - Response time pattern (fast/moderate/slow based on message frequency)
   - Question complexity (simple/moderate/complex)
   - Escalation tendency (0-100, likelihood to escalate)

Output format (JSON only):
{{
  "main_inquiry": "brief description of main issue",
  "key_responses": ["response 1", "response 2"],
  "customer_sentiment_score": 75,
  "customer_satisfaction_score": 80,
  "customer_characteristics": {{
    "language": "ko/en/ja or unknown",
    "cultural_hints": "brief description or unknown",
    "cultural_score": 60,
    "region": "general region or unknown",
    "communication_style": "formal/casual/brief/detailed",
    "formality_score": 70,
    "detail_level_score": 65,
    "personality_traits": {{
      "patience_level": 60,
      "assertiveness": 70,
      "politeness_level": 80,
      "technical_proficiency": 50
    }}
  }},
  "privacy_info": {{
    "has_email": true/false,
    "has_phone": true/false,
    "has_address": true/false,
    "region_hint": "general region or unknown"
  }},
  "behavior_patterns": {{
    "response_time": "fast/moderate/slow",
    "question_complexity": "simple/moderate/complex",
    "escalation_tendency": 30
  }},
  "summary": "overall conversation summary in {lang_name}"
}}

Conversation:
{conversation_text}

JSON Output:
"""

    if not st.session_state.is_llm_ready:
        return {
            "main_inquiry": initial_query[:100],
            "key_responses": [],
            "customer_sentiment_score": 50,
            "customer_satisfaction_score": 50,
            "customer_characteristics": {
                "language": current_lang_key,
                "cultural_hints": "unknown",
                "cultural_score": 0,
                "region": "unknown",
                "communication_style": "unknown",
                "formality_score": 50,
                "detail_level_score": 50,
                "personality_traits": {
                    "patience_level": 50,
                    "assertiveness": 50,
                    "politeness_level": 50,
                    "technical_proficiency": 50
                }
            },
            "privacy_info": {
                "has_email": False,
                "has_phone": False,
                "has_address": False,
                "region_hint": "unknown"
            },
            "behavior_patterns": {
                "response_time": "moderate",
                "question_complexity": "moderate",
                "escalation_tendency": 50
            },
            "summary": f"Customer inquiry about: {initial_query[:100]}"
        }

    try:
        summary_text = run_llm(summary_prompt).strip()
        
        # JSON 추출
        if "```" in summary_text:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', summary_text, re.DOTALL)
            if json_match:
                summary_text = json_match.group(1)
            else:
                summary_text = re.sub(r'```(?:json)?\s*', '', summary_text)
                summary_text = re.sub(r'\s*```', '', summary_text)
        
        json_match = re.search(r'\{.*\}', summary_text, re.DOTALL)
        if json_match:
            summary_text = json_match.group(0)
        
        summary_text = summary_text.strip()
        
        # JSON 파싱 시도
        try:
            summary_data = json.loads(summary_text)
        except json.JSONDecodeError as json_err:
            try:
                summary_text_py = summary_text.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                summary_data = ast.literal_eval(summary_text_py)
                if isinstance(summary_data, dict):
                    summary_data = {k: (None if v is None else v) for k, v in summary_data.items()}
            except Exception:
                main_inquiry_match = re.search(r'"main_inquiry"\s*:\s*"([^"]*)"', summary_text)
                summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', summary_text)
                key_responses_matches = re.findall(r'"key_responses"\s*:\s*\[(.*?)\]', summary_text, re.DOTALL)
                
                main_inquiry = main_inquiry_match.group(1) if main_inquiry_match else initial_query[:100]
                summary_text_val = summary_match.group(1) if summary_match else f"Customer inquiry about: {initial_query[:100]}"
                
                key_responses = []
                if key_responses_matches:
                    responses_text = key_responses_matches[0]
                    response_matches = re.findall(r'"([^"]*)"', responses_text)
                    key_responses = response_matches[:3]
                
                summary_data = {
                    "main_inquiry": main_inquiry,
                    "key_responses": key_responses,
                    "customer_sentiment_score": 50,
                    "customer_satisfaction_score": 50,
                    "customer_characteristics": {
                        "language": current_lang_key,
                        "cultural_hints": "unknown",
                        "cultural_score": 0,
                        "region": "unknown",
                        "communication_style": "unknown",
                        "formality_score": 50,
                        "detail_level_score": 50,
                        "personality_traits": {
                            "patience_level": 50,
                            "assertiveness": 50,
                            "politeness_level": 50,
                            "technical_proficiency": 50
                        }
                    },
                    "privacy_info": {
                        "has_email": False,
                        "has_phone": False,
                        "has_address": False,
                        "region_hint": "unknown"
                    },
                    "behavior_patterns": {
                        "response_time": "moderate",
                        "question_complexity": "moderate",
                        "escalation_tendency": 50
                    },
                    "summary": summary_text_val
                }
                
                print(f"JSON 파싱 오류 (모든 재시도 실패): {json_err}")
                print(f"원본 JSON (처음 1000자): {summary_text[:1000]}")
        
        return summary_data
    except Exception as e:
        st.warning(f"요약 생성 중 오류 발생: {e}")
        return {
            "main_inquiry": initial_query[:100],
            "key_responses": [],
            "customer_sentiment_score": 50,
            "customer_satisfaction_score": 50,
            "customer_characteristics": {
                "language": current_lang_key,
                "cultural_hints": "unknown",
                "cultural_score": 0,
                "region": "unknown",
                "communication_style": "unknown",
                "formality_score": 50,
                "detail_level_score": 50,
                "personality_traits": {
                    "patience_level": 50,
                    "assertiveness": 50,
                    "politeness_level": 50,
                    "technical_proficiency": 50
                }
            },
            "privacy_info": {
                "has_email": False,
                "has_phone": False,
                "has_address": False,
                "region_hint": "unknown"
            },
            "behavior_patterns": {
                "response_time": "moderate",
                "question_complexity": "moderate",
                "escalation_tendency": 50
            },
            "summary": f"Error generating summary: {str(e)}"
        }

