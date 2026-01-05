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
탭1: CEO 정보 표시 모듈
"""

import streamlit as st


def render_ceo_section(ceo_info: dict, current_lang: str, L: dict):
    """CEO/대표이사 정보 렌더링"""
    ceo_name = ceo_info.get(f"name_{current_lang}", ceo_info.get("name_ko", ""))
    ceo_position = ceo_info.get(f"position_{current_lang}", ceo_info.get("position_ko", ""))
    ceo_bio = ceo_info.get(f"bio_{current_lang}", ceo_info.get("bio_ko", ""))
    ceo_tenure = ceo_info.get(f"tenure_{current_lang}", ceo_info.get("tenure_ko", ""))
    ceo_education = ceo_info.get(f"education_{current_lang}", ceo_info.get("education_ko", ""))
    ceo_career = ceo_info.get(f"career_{current_lang}", ceo_info.get("career_ko", ""))

    if ceo_name or ceo_position:
        st.markdown(f"#### 👔 {L.get('ceo_info', 'CEO/대표이사 정보')}")
        st.markdown("---")

        # CEO 정보 카드 형태로 표시
        col_ceo_left, col_ceo_right = st.columns([1, 2])

        with col_ceo_left:
            if ceo_name:
                st.markdown(f"### {ceo_name}")
            if ceo_position:
                st.markdown(f"**{L.get('position', '직책')}:** {ceo_position}")
            if ceo_tenure:
                st.markdown(f"**{L.get('tenure', '재임 기간')}:** {ceo_tenure}")

        with col_ceo_right:
            if ceo_bio:
                st.markdown(f"**{L.get('ceo_bio', '소개')}**")
                st.markdown(ceo_bio)

        # 학력 및 경력 정보
        if ceo_education or ceo_career:
            st.markdown("---")
            col_edu, col_career = st.columns(2)

            with col_edu:
                if ceo_education:
                    st.markdown(f"**{L.get('education', '학력')}**")
                    st.markdown(ceo_education)

            with col_career:
                if ceo_career:
                    st.markdown(f"**{L.get('career', '주요 경력')}**")
                    st.markdown(ceo_career)

        st.markdown("---")

