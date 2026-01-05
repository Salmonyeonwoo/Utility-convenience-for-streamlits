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
회사 검색 모듈
"""

import streamlit as st
from lang_pack import LANG
from faq.database import load_faq_database, save_faq_database
from faq.llm_generator import generate_company_info_with_llm
from llm_client import get_api_key


def render_company_search(current_lang: str, L: dict) -> tuple:
    """회사 검색 UI 및 로직"""
    # FAQ 데이터베이스 로드
    faq_data = load_faq_database()
    companies = list(faq_data.get("companies", {}).keys())

    # 회사명 검색 입력
    col_search_header, col_search_input, col_search_btn = st.columns([0.5, 1.2, 0.2])
    with col_search_header:
        st.write(f"**{L['search_company']}**")
    with col_search_input:
        company_search_input = st.text_input(
            "",
            placeholder=L["company_search_placeholder"],
            key="company_search_input",
            value=st.session_state.get("searched_company", ""),
            label_visibility="collapsed"
        )
    with col_search_btn:
        search_button = st.button(
            f"🔍 {L['company_search_button']}",
            key="company_search_btn",
            type="primary",
            use_container_width=True)

    # 검색된 회사 정보 저장
    searched_company = st.session_state.get("searched_company", "")
    searched_company_data = st.session_state.get("searched_company_data", None)

    # 검색 버튼 클릭 시 LLM으로 회사 정보 생성
    if search_button and company_search_input:
        openai_key = get_api_key("openai")
        gemini_key = get_api_key("gemini")
        if not openai_key and not gemini_key:
            import os
            openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key") or ""
            gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini_api_key") or ""
        
        if not openai_key and not gemini_key:
            st.error("❌ OpenAI 또는 Gemini API Key가 필요합니다. 환경변수로 설정해주세요.")
            st.info("💡 PowerShell에서 설정:\n`$env:OPENAI_API_KEY=\"sk-...\"`\n또는\n`$env:GEMINI_API_KEY=\"AIza...\"`")
        else:
            with st.spinner(f"{company_search_input} {L['generating_company_info']}"):
                try:
                    generated_data = generate_company_info_with_llm(
                        company_search_input, current_lang)
                    st.session_state.searched_company = company_search_input
                    st.session_state.searched_company_data = generated_data
                    searched_company = company_search_input
                    searched_company_data = generated_data

                    # 생성된 데이터를 데이터베이스에 저장
                    if company_search_input not in faq_data.get("companies", {}):
                        faq_data.setdefault("companies", {})[company_search_input] = {
                            f"info_{current_lang}": generated_data.get("company_info", ""),
                            "info_ko": generated_data.get("company_info", ""),
                            "info_en": "",
                            "info_ja": "",
                            "popular_products": generated_data.get("popular_products", []),
                            "trending_topics": generated_data.get("trending_topics", []),
                            "faqs": generated_data.get("faqs", []),
                            "interview_questions": generated_data.get("interview_questions", []),
                            "ceo_info": generated_data.get("ceo_info", {})
                        }
                        save_faq_database(faq_data)
                    st.success(f"✅ {company_search_input} 회사 정보를 생성했습니다!")
                except Exception as e:
                    st.error(f"❌ 회사 정보 생성 중 오류: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    searched_company = st.session_state.get("searched_company", "")
                    searched_company_data = st.session_state.get("searched_company_data", None)
    else:
        searched_company = st.session_state.get("searched_company", "")
        searched_company_data = st.session_state.get("searched_company_data", None)

    # 검색된 회사가 있으면 해당 데이터 사용, 없으면 기존 회사 선택
    if searched_company and searched_company_data:
        display_company = searched_company
        display_data = searched_company_data
        # 데이터베이스에도 저장되어 있으면 업데이트
        if display_company in faq_data.get("companies", {}):
            faq_data["companies"][display_company].update({
                f"info_{current_lang}": display_data.get("company_info", ""),
                "popular_products": display_data.get("popular_products", []),
                "trending_topics": display_data.get("trending_topics", []),
                "faqs": display_data.get("faqs", []),
                "interview_questions": display_data.get("interview_questions", []),
                "ceo_info": display_data.get("ceo_info", {})
            })
            save_faq_database(faq_data)
    elif companies:
        display_company = st.selectbox(
            L["select_company"],
            options=companies,
            key="company_select_display"
        )
        company_db_data = faq_data["companies"][display_company]
        display_data = {
            "company_info": company_db_data.get(
                f"info_{current_lang}", company_db_data.get(
                    "info_ko", "")), 
            "popular_products": company_db_data.get("popular_products", []), 
            "trending_topics": company_db_data.get("trending_topics", []), 
            "faqs": company_db_data.get("faqs", []), 
            "interview_questions": company_db_data.get("interview_questions", []), 
            "ceo_info": company_db_data.get("ceo_info", {})
        }
    else:
        display_company = None
        display_data = None

    return display_company, display_data, faq_data

