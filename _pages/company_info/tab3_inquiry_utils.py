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
탭3: 고객 문의 유틸리티 모듈 (파일 처리, 번역 등)
"""

import streamlit as st
import tempfile
import os
import base64
from langchain_community.document_loaders import PyPDFLoader
from llm_client import get_api_key, run_llm
import google.generativeai as genai
from lang_pack import LANG


def process_uploaded_file(uploaded_file, current_lang: str, L: dict) -> tuple:
    """업로드된 파일 처리 및 내용 추출"""
    if uploaded_file is None:
        return None, ""
    
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size = len(uploaded_file.getvalue())
    
    st.success(
        L.get("inquiry_attachment_uploaded", "✅ 첨부 파일이 업로드되었습니다: {filename}").format(
            filename=file_name))

    uploaded_file_info = {
        "name": file_name,
        "type": file_type,
        "size": file_size
    }

    file_content_extracted = ""
    file_content_translated = ""

    # 파일 내용 추출
    if file_name.lower().endswith(('.pdf', '.txt', '.png', '.jpg', '.jpeg')):
        try:
            with st.spinner(L.get("extracting_file_content", "파일 내용 추출 중...")):
                if file_name.lower().endswith('.pdf'):
                    file_content_extracted = _extract_pdf_content(uploaded_file)
                elif file_name.lower().endswith('.txt'):
                    uploaded_file.seek(0)
                    file_content_extracted = uploaded_file.read().decode("utf-8", errors="ignore")
                elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_content_extracted = _extract_image_text(uploaded_file, file_type, current_lang, L)

                # 파일 내용 번역 (필요한 경우)
                if file_content_extracted and current_lang in ["ja", "en"]:
                    file_content_translated = _translate_file_content(
                        file_content_extracted, current_lang, L)
        except Exception as e:
            error_msg = L.get("file_extraction_error", "파일 내용 추출 중 오류가 발생했습니다: {error}")
            st.warning(error_msg.format(error=str(e)))

    return uploaded_file_info, file_content_translated if file_content_translated else file_content_extracted


def _extract_pdf_content(uploaded_file) -> str:
    """PDF 파일 내용 추출"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    try:
        loader = PyPDFLoader(tmp.name)
        file_docs = loader.load()
        return "\n".join([doc.page_content for doc in file_docs])
    finally:
        try:
            os.remove(tmp.name)
        except BaseException:
            pass


def _extract_image_text(uploaded_file, file_type: str, current_lang: str, L: dict) -> str:
    """이미지에서 텍스트 추출 (OCR)"""
    uploaded_file.seek(0)
    image_bytes = uploaded_file.getvalue()

    ocr_prompt = """이 이미지에 있는 모든 텍스트를 정확히 추출해주세요.
이미지에 한국어, 일본어, 영어 등 어떤 언어의 텍스트가 있든 모두 추출하고,
텍스트의 구조와 순서를 유지해주세요.
이미지에 텍스트가 없으면 "텍스트 없음"이라고 답변하세요.

추출된 텍스트:"""

    try:
        gemini_key = get_api_key("gemini")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([
                {"mime_type": file_type, "data": image_bytes},
                ocr_prompt
            ])
            return response.text if response.text else ""
        else:
            st.info(L.get("ocr_requires_manual", 
                         "이미지 OCR을 위해서는 Gemini API 키가 필요합니다. 이미지의 텍스트를 수동으로 입력해주세요."))
            return ""
    except Exception as ocr_error:
        error_msg = L.get("ocr_error", "이미지 텍스트 추출 중 오류: {error}")
        st.warning(error_msg.format(error=str(ocr_error)))
        return ""


def _translate_file_content(file_content: str, current_lang: str, L: dict) -> str:
    """파일 내용 번역"""
    with st.spinner(L.get("detecting_language", "언어 감지 중...")):
        detect_prompts = {
            "ja": f"""次のテキストの言語を検出してください。韓国語、日本語、英語のいずれかで答えてください。

テキスト:
{file_content[:500]}

言語:""",
            "en": f"""Detect the language of the following text. Answer with only one of: Korean, Japanese, or English.

Text:
{file_content[:500]}

Language:""",
            "ko": f"""다음 텍스트의 언어를 감지해주세요. 한국어, 일본어, 영어 중 하나로만 답변하세요.

텍스트:
{file_content[:500]}

언어:"""}
        detect_prompt = detect_prompts.get(current_lang, detect_prompts["ko"])
        detected_lang = run_llm(detect_prompt).strip().lower()

        if "한국어" in detected_lang or "korean" in detected_lang or "ko" in detected_lang:
            with st.spinner(L.get("translating_content", "파일 내용 번역 중...")):
                translate_prompts = {
                    "ja": f"""次の韓国語テキストを日本語に翻訳してください。原文の意味とトーンを正確に維持しながら、自然な日本語で翻訳してください。

韓国語テキスト:
{file_content}

日本語翻訳:""",
                    "en": f"""Please translate the following Korean text into English. Maintain the exact meaning and tone of the original text while translating into natural English.

Korean text:
{file_content}

English translation:"""}
                translate_prompt = translate_prompts.get(current_lang)
                if translate_prompt:
                    translated = run_llm(translate_prompt)
                    if translated and not translated.startswith("❌"):
                        st.info(L.get("file_translated", "✅ 파일 내용이 번역되었습니다."))
                        return translated
    return ""


def create_attachment_info(
    uploaded_file_info: dict, file_content: str, current_lang: str) -> str:
    """첨부 파일 정보 문자열 생성"""
    if not uploaded_file_info:
        return ""
    
    file_name = uploaded_file_info["name"]
    file_type = uploaded_file_info["type"]
    file_size = uploaded_file_info["size"]
    
    content_section = ""
    if file_content:
        content_section = f"\n\n[파일 내용]\n{file_content[:2000]}"
        if len(file_content) > 2000:
            content_section += "\n...(내용이 길어 일부만 표시됨)"

    attachment_info_by_lang = {
        "ko": f"\n\n[고객 첨부 파일 정보]\n- 파일명: {file_name}\n- 파일 타입: {file_type}\n- 파일 크기: {file_size} bytes\n- 참고: 고객이 {file_name} 파일을 첨부했습니다. 이 파일은 비행기 지연, 여권 이슈, 질병 등 불가피한 사유로 인한 취소 불가 여행상품 관련 증빙 자료일 수 있습니다. 파일 내용을 참고하여 응대하세요.{content_section}",
        "en": f"\n\n[Customer Attachment Information]\n- File name: {file_name}\n- File type: {file_type}\n- File size: {file_size} bytes\n- Note: The customer has attached the file {file_name}. This file may be evidence related to non-refundable travel products due to unavoidable reasons such as flight delays, passport issues, illness, etc. Please refer to the file content when responding.{content_section}",
        "ja": f"\n\n[顧客添付ファイル情報]\n- ファイル名: {file_name}\n- ファイルタイプ: {file_type}\n- ファイルサイズ: {file_size} bytes\n- 参考: 顧客が{file_name}ファイルを添付しました。このファイルは、飛行機の遅延、パスポートの問題、病気などやむを得ない理由によるキャンセル不可の旅行商品に関連する証拠資料である可能性があります。ファイルの内容を参照して対応してください。{content_section}"}
    
    return attachment_info_by_lang.get(current_lang, attachment_info_by_lang["ko"])

