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

# ========================================
# _pages/_company_info.py
# 회사 정보 탭 모듈 (메인)
# ========================================

import streamlit as st
from lang_pack import LANG
from faq.database import load_faq_database

# 하위 모듈 import
from _pages.company_info.search import render_company_search
from _pages.company_info.tab1_info import render_tab1_company_info
from _pages.company_info.tab2_faq import render_tab2_faq
from _pages.company_info.tab3_inquiry import render_tab3_inquiry


def render_company_info():
    """회사 정보 탭 렌더링 함수"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])

    # 회사 검색 및 선택
    display_company, display_data, faq_data = render_company_search(current_lang, L)
    
    # FAQ 데이터베이스에서 companies 목록 가져오기
    companies = list(faq_data.get("companies", {}).keys())

    # 탭 생성
    tab1, tab2, tab3 = st.tabs([
        L["company_info"],
        L["company_faq"],
        L["button_add_company"]
    ])

    # 탭 1: 회사 소개 및 시각화
    with tab1:
        render_tab1_company_info(display_company, display_data, current_lang, L, faq_data)

    # 탭 2: 자주 묻는 질문 (FAQ)
    with tab2:
        render_tab2_faq(display_company, display_data, current_lang, L)

    # 탭 3: 고객 문의 재확인 (에이전트용)
    with tab3:
        render_tab3_inquiry(companies, current_lang, L, faq_data)
