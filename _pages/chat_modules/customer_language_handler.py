# ========================================
# chat_modules/customer_language_handler.py
# 고객 언어 전환 처리 모듈
# ========================================

import streamlit as st
from lang_pack import LANG


def detect_and_handle_language_change(customer_response, current_lang):
    """언어 전환 요청 감지 및 처리"""
    language_change_requested = False
    requested_lang = None
    
    # 영어 전환 요청 감지
    english_requests = [
        "can we speak english", "speak english", "english please", 
        "english, please", "in english", "use english",
        "영어로", "영어로 말씀해주세요", "영어로 해주세요", "영어로 부탁합니다"
    ]
    
    # 일본어 전환 요청 감지
    japanese_requests = [
        "日本語で", "日本語でお願いします", "日本語で話してください",
        "speak japanese", "japanese please", "in japanese", "use japanese"
    ]
    
    # 한국어 전환 요청 감지
    korean_requests = [
        "한국어로", "한국어로 말씀해주세요", "한국어로 해주세요",
        "speak korean", "korean please", "in korean", "use korean"
    ]
    
    customer_response_lower = customer_response.lower()
    
    # 언어 전환 요청 확인
    if any(req.lower() in customer_response_lower for req in english_requests):
        requested_lang = "en"
        language_change_requested = True
    elif any(req.lower() in customer_response_lower for req in japanese_requests):
        requested_lang = "ja"
        language_change_requested = True
    elif any(req.lower() in customer_response_lower for req in korean_requests):
        requested_lang = "ko"
        language_change_requested = True
    
    # 언어 전환 처리
    if language_change_requested and requested_lang:
        current_lang_state = st.session_state.get("language", "ko")
        if requested_lang != current_lang_state:
            st.session_state.language = requested_lang
            lang_names = {"ko": "한국어", "en": "English", "ja": "日本語"}
            st.info(f"🌐 고객의 요청에 따라 언어가 {lang_names[requested_lang]}로 자동 변경되었습니다.")
            return requested_lang
    else:
        # 언어 전환 요청이 없으면 메시지 언어 자동 감지
        try:
            from utils.customer_analysis import detect_text_language
            detected_lang = detect_text_language(customer_response)
            if detected_lang in ["ko", "en", "ja"]:
                current_lang_state = st.session_state.get("language", "ko")
                if detected_lang != current_lang_state:
                    st.session_state.language = detected_lang
                    lang_names = {"ko": "한국어", "en": "English", "ja": "日本語"}
                    st.info(f"🌐 입력 언어가 감지되어 언어 설정이 {lang_names[detected_lang]}로 자동 변경되었습니다.")
                    return detected_lang
        except Exception as e:
            print(f"Language detection failed: {e}")
    
    return current_lang

