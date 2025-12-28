# -*- coding: utf-8 -*-
"""
대화 흐름 관리 유틸리티
자연스러운 콜센터 대화 흐름을 관리
"""
import streamlit as st
from lang_pack import LANG
import re


def detect_customer_emotion(customer_message: str, current_lang: str) -> str:
    """고객 메시지에서 감정 상태 감지"""
    if not customer_message:
        return "NEUTRAL"
    
    message_lower = customer_message.lower()
    
    # 화남/짜증 키워드
    angry_keywords = {
        "ko": ["화나", "짜증", "불만", "화났", "짜증나", "답답", "열받", "빡치", "미치", "개빡", "왜이래", "이게뭐야"],
        "en": ["angry", "mad", "furious", "annoyed", "frustrated", "upset", "irritated", "pissed"],
        "ja": ["怒", "イライラ", "不満", "腹立", "むかつ"]
    }
    
    # 슬픔 키워드
    sad_keywords = {
        "ko": ["슬프", "우울", "힘들", "어렵", "막막", "답답", "힘들어", "어려워"],
        "en": ["sad", "depressed", "difficult", "hard", "struggling", "tough"],
        "ja": ["悲", "つらい", "難しい", "困", "苦"]
    }
    
    # 불편/불만 키워드
    dissatisfied_keywords = {
        "ko": ["불편", "불만", "문제", "이상", "안되", "안돼", "문제있", "문제가"],
        "en": ["problem", "issue", "wrong", "not working", "broken", "error"],
        "ja": ["不便", "問題", "おかしい", "動かない"]
    }
    
    lang_keywords = angry_keywords.get(current_lang, angry_keywords["ko"])
    if any(keyword in message_lower for keyword in lang_keywords):
        return "ANGRY"
    
    lang_keywords = sad_keywords.get(current_lang, sad_keywords["ko"])
    if any(keyword in message_lower for keyword in lang_keywords):
        return "SAD"
    
    lang_keywords = dissatisfied_keywords.get(current_lang, dissatisfied_keywords["ko"])
    if any(keyword in message_lower for keyword in lang_keywords):
        return "DISSATISFIED"
    
    return "NEUTRAL"


def generate_empathetic_response(emotion: str, current_lang: str) -> str:
    """고객 감정에 따른 공감 표현 생성"""
    L = LANG.get(current_lang, LANG["ko"])
    
    if emotion == "ANGRY":
        responses = {
            "ko": [
                "고객님 정말 답답하시고 화나실 것 같습니다. 깊이 이해합니다.",
                "고객님의 불편하신 마음을 충분히 이해합니다. 정말 죄송합니다.",
                "정말 불편하셨을 것 같습니다. 깊이 사과드립니다."
            ],
            "en": [
                "I completely understand how frustrating this must be for you. I'm truly sorry.",
                "I can only imagine how upset you must be. I sincerely apologize.",
                "I deeply understand your frustration. I'm very sorry about this."
            ],
            "ja": [
                "お客様のお気持ち、大変よく理解いたします。本当に申し訳ございません。",
                "ご不便をおかけして、心よりお詫び申し上げます。",
                "お客様のご不満、深く理解いたします。誠に申し訳ございません。"
            ]
        }
    elif emotion == "SAD":
        responses = {
            "ko": [
                "고객님 정말 얼마나 안타까우실까요? 깊이 이해합니다.",
                "정말 힘드실 것 같습니다. 충분히 이해하고 있습니다.",
                "고객님의 어려운 상황을 깊이 이해합니다. 도와드리겠습니다."
            ],
            "en": [
                "I can only imagine how difficult this must be for you. I deeply understand.",
                "I truly understand how hard this must be. I'm here to help.",
                "I completely understand your situation. Let me help you with this."
            ],
            "ja": [
                "お客様のお気持ち、大変よく理解いたします。",
                "お困りの状況、深く理解いたします。サポートさせていただきます。",
                "お客様のご状況、心より理解いたします。お手伝いさせていただきます。"
            ]
        }
    elif emotion == "DISSATISFIED":
        responses = {
            "ko": [
                "불편을 드려 정말 죄송합니다. 바로 확인해보겠습니다.",
                "문제가 발생해서 죄송합니다. 즉시 확인하겠습니다.",
                "불편을 드려 깊이 사과드립니다. 바로 처리해드리겠습니다."
            ],
            "en": [
                "I'm very sorry for the inconvenience. Let me check this right away.",
                "I apologize for the problem. I'll look into this immediately.",
                "I sincerely apologize for the inconvenience. Let me handle this right now."
            ],
            "ja": [
                "ご不便をおかけして申し訳ございません。すぐに確認いたします。",
                "問題が発生し、申し訳ございません。即座に確認いたします。",
                "ご不便をおかけして心よりお詫び申し上げます。すぐに対応いたします。"
            ]
        }
    else:
        return ""
    
    import random
    lang_responses = responses.get(current_lang, responses["ko"])
    return random.choice(lang_responses)


def needs_additional_info(agent_response: str, current_lang: str) -> bool:
    """에이전트 응답이 추가 정보 요청인지 확인"""
    if not agent_response:
        return False
    
    response_lower = agent_response.lower()
    
    # 추가 정보 요청 키워드
    info_request_keywords = {
        "ko": ["알려주시", "말씀해주시", "확인해주시", "알려주실", "말씀해주실", "확인해주실", "필요", "궁금"],
        "en": ["please tell", "could you", "can you", "need", "require", "information", "details"],
        "ja": ["教えて", "お聞かせ", "確認", "必要", "詳細"]
    }
    
    lang_keywords = info_request_keywords.get(current_lang, info_request_keywords["ko"])
    return any(keyword in response_lower for keyword in lang_keywords)


def needs_verification(agent_response: str, current_lang: str) -> bool:
    """에이전트 응답이 확인이 필요한지 판단"""
    if not agent_response:
        return False
    
    response_lower = agent_response.lower()
    
    # 확인 필요 키워드
    verification_keywords = {
        "ko": ["확인해보겠", "확인하겠", "검토해보겠", "조사해보겠", "점검해보겠", "살펴보겠"],
        "en": ["check", "verify", "review", "investigate", "look into", "examine"],
        "ja": ["確認", "検討", "調査", "確認いたし", "調べ"]
    }
    
    lang_keywords = verification_keywords.get(current_lang, verification_keywords["ko"])
    return any(keyword in response_lower for keyword in lang_keywords)


def generate_waiting_message(current_lang: str) -> str:
    """대기 메시지 생성"""
    messages = {
        "ko": "확인해 보겠습니다. 5분 정도 대기 부탁드립니다.",
        "en": "Let me check on that. Please wait about 5 minutes.",
        "ja": "確認いたします。5分ほどお待ちください。"
    }
    return messages.get(current_lang, messages["ko"])


def generate_after_waiting_message(current_lang: str) -> str:
    """대기 후 메시지 생성"""
    messages = {
        "ko": "기다려 주셔서 감사드립니다.",
        "en": "Thank you for waiting.",
        "ja": "お待たせいたしました。"
    }
    return messages.get(current_lang, messages["ko"])


