# ========================================
# _pages/_content_utils.py
# 콘텐츠 생성 유틸리티 함수
# ========================================

import json
import uuid
import streamlit as st


def get_level_map():
    """난이도 맵핑 반환"""
    return {
        "초급": "Beginner",
        "중급": "Intermediate",
        "고급": "Advanced",
        "Beginner": "Beginner",
        "Intermediate": "Intermediate",
        "Advanced": "Advanced",
        "初級": "Beginner",
        "中級": "Intermediate",
        "上級": "Advanced",
    }


def get_content_map():
    """콘텐츠 유형 맵핑 반환"""
    return {
        "핵심 요약 노트": "summary",
        "객관식 퀴즈 10문항": "quiz",
        "실습 예제 아이디어": "example",
        "Key Summary Note": "summary",
        "10 MCQ Questions": "quiz",
        "Practical Example Idea": "example",
        "核心要約ノート": "summary",
        "選択式クイズ10問": "quiz",
        "実践例のアイデア": "example",
    }


def extract_json_from_text(text):
    """텍스트에서 JSON 객체를 추출하는 함수"""
    if not text:
        return None

    text = text.strip()

    # 1. Markdown 코드 블록 제거
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()

    # 2. 첫 번째 '{' 부터 마지막 '}' 까지 추출
    first_brace = text.find('{')
    if first_brace == -1:
        return None

    # 중괄호 매칭으로 JSON 객체 끝 찾기
    brace_count = 0
    last_brace = -1
    for i in range(first_brace, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                last_brace = i
                break

    if last_brace != -1:
        json_str = text[first_brace:last_brace + 1]
        return json_str.strip()

    return None


def mock_download(file_type: str, file_name: str):
    """모의 다운로드 기능"""
    st.toast(f"📥 {file_type} 파일을 생성하여 다운로드를 시작합니다: {file_name}")
