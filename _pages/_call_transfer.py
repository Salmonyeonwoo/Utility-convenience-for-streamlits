# ========================================
# _pages/_call_transfer.py
# 전화 통화 언어 팀 이관 모듈
# ========================================

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
from lang_pack import LANG
from simulation_handler import summarize_history_with_ai
from utils.translation import translate_text_with_llm
from llm_client import get_api_key


def render_transfer_section(current_lang, L):
    """언어 팀 이관 섹션 렌더링"""
    st.markdown(f"**{L.get('transfer_header', '언어 팀 이관')}**")
    
    languages = list(LANG.keys())
    if current_lang in languages:
        languages.remove(current_lang)
    
    if not languages:
        st.info("이관할 다른 언어 팀이 없습니다.")
        return
    
    transfer_cols = st.columns(len(languages))
    
    for idx, lang_code in enumerate(languages):
        lang_name = {"ko": "한국어", "en": "영어", "ja": "일본어"}.get(lang_code, lang_code)
        transfer_label = _get_transfer_label(lang_code, lang_name)
        
        with transfer_cols[idx]:
            if st.button(
                transfer_label,
                key=f"btn_call_transfer_{lang_code}_{st.session_state.get('sim_instance_id', 'default')}",
                type="secondary",
                use_container_width=True
            ):
                transfer_call_session(lang_code, st.session_state.get("call_messages", []), L)


def _get_transfer_label(lang_code, lang_name):
    """이관 라벨 생성"""
    if lang_code == "en":
        return "US 영어 팀으로 이관"
    elif lang_code == "ja":
        return "JP 일본어 팀으로 이관"
    else:
        return f"{lang_name} 팀으로 이관"


def transfer_call_session(target_lang: str, current_messages: List[Dict[str, Any]], L):
    """전화 통화 세션을 다른 언어 팀으로 이관"""
    current_lang_at_start = st.session_state.language
    L_source = LANG.get(current_lang_at_start, LANG["ko"])
    
    lang_name_target = {"ko": "한국어", "en": "영어", "ja": "일본어"}.get(target_lang, target_lang)
    st.info(f"🔄 {lang_name_target} 팀으로 이관 처리 중...")
    
    if get_api_key and not get_api_key("gemini"):
        st.error(L_source.get("simulation_no_key_warning", "⚠️ Gemini API Key가 설정되지 않았습니다.").replace('API Key', 'Gemini API Key'))
        return
    
    if not summarize_history_with_ai or not translate_text_with_llm:
        st.error("⚠️ 이관 기능을 사용할 수 없습니다. 필요한 모듈을 확인해주세요.")
        return
    
    try:
        # 원본 언어로 핵심 요약 생성
        original_summary = summarize_history_with_ai(current_lang_at_start)
        
        if not original_summary or original_summary.startswith("❌"):
            history_text = ""
            for msg in current_messages:
                role = "Customer" if msg.get("role") == "customer" else "Agent"
                if msg.get("content"):
                    history_text += f"{role}: {msg['content']}\n"
            original_summary = history_text
        
        # 번역
        translated_summary, is_success = translate_text_with_llm(
            original_summary, target_lang, current_lang_at_start
        )
        
        if not translated_summary:
            translated_summary = summarize_history_with_ai(target_lang)
            is_success = True if translated_summary and not translated_summary.startswith("❌") else False
        
        # 메시지 번역
        translated_messages = _translate_messages(current_messages, target_lang, current_lang_at_start)
        
        # 상태 업데이트
        st.session_state.call_messages = translated_messages
        st.session_state.transfer_summary_text = translated_summary
        st.session_state.translation_success = is_success
        st.session_state.language_at_transfer_start = current_lang_at_start
        st.session_state.language = target_lang
        L_target = LANG.get(target_lang, LANG["ko"])
        
        lang_name_target = {"ko": "한국어", "en": "영어", "ja": "일본어"}.get(target_lang, "한국어")
        
        # 시스템 메시지 추가
        system_msg = L_target.get("transfer_system_msg", "📌 시스템 메시지: 통화가 {target_lang} 팀으로 이관되었습니다.").format(target_lang=lang_name_target)
        st.session_state.call_messages.append({
            "role": "system_transfer",
            "content": system_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        summary_header = L_target.get("transfer_summary_header", "이관 요약")
        summary_msg = f"### {summary_header}\n\n{translated_summary}"
        st.session_state.call_messages.append({
            "role": "supervisor",
            "content": summary_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        st.success(f"✅ {lang_name_target} 팀으로 이관되었습니다.")
        
    except Exception as e:
        error_msg = L_source.get("transfer_error", "이관 처리 중 오류 발생: {error}").format(error=str(e))
        st.error(error_msg)


def _translate_messages(messages, target_lang, current_lang_at_start):
    """메시지 번역"""
    translated_messages = []
    messages_to_translate = []
    
    for idx, msg in enumerate(messages):
        translated_msg = msg.copy()
        if msg.get("role") in ["agent", "customer"] and msg.get("content"):
            messages_to_translate.append((idx, msg))
        translated_messages.append(translated_msg)
    
    if messages_to_translate:
        try:
            combined_text = "\n\n".join([
                f"[{msg['role']}]: {msg['content']}" 
                for _, msg in messages_to_translate
            ])
            
            translated_combined, trans_success = translate_text_with_llm(
                combined_text, target_lang, current_lang_at_start
            )
            
            if trans_success and translated_combined:
                translated_lines = translated_combined.split("\n\n")
                for i, (idx, original_msg) in enumerate(messages_to_translate):
                    if i < len(translated_lines):
                        translated_line = translated_lines[i]
                        if "]: " in translated_line:
                            translated_content = translated_line.split("]: ", 1)[1]
                        else:
                            translated_content = translated_line
                        translated_messages[idx]["content"] = translated_content
        except Exception:
            # 개별 번역으로 폴백
            for idx, msg in messages_to_translate:
                try:
                    translated_content, trans_success = translate_text_with_llm(
                        msg["content"], target_lang, current_lang_at_start
                    )
                    if trans_success:
                        translated_messages[idx]["content"] = translated_content
                except Exception:
                    pass
    
    return translated_messages
