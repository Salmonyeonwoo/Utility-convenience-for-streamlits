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
탭1: 면접 질문 표시 모듈
"""

import streamlit as st


def render_interview_section(interview_questions: list, current_lang: str, L: dict):
    """면접 질문 목록 렌더링"""
    st.markdown(f"#### 💼 {L.get('interview_questions', '면접 예상 질문')}")
    st.markdown(
        f"*{L.get('interview_questions_desc', '면접에서 나올 만한 핵심 질문들과 상세한 답변입니다. 면접 준비와 회사 이해에 도움이 됩니다.')}*")
    st.markdown("---")

    # 카테고리별로 그룹화
    interview_by_category = {}
    for idx, iq in enumerate(interview_questions):
        question = iq.get(f"question_{current_lang}", iq.get("question_ko", ""))
        answer = iq.get(f"answer_{current_lang}", iq.get("answer_ko", ""))
        category = iq.get(
            f"category_{current_lang}", 
            iq.get("category_ko", L.get("interview_category_other", "기타"))
        )

        if category not in interview_by_category:
            interview_by_category[category] = []
        interview_by_category[category].append({
            "question": question,
            "answer": answer,
            "index": idx + 1
        })

    # 카테고리별로 표시
    for category, questions in interview_by_category.items():
        with st.expander(f"📋 **{category}** ({len(questions)}{L.get('items', '개')})"):
            for item in questions:
                st.markdown(f"**{item['index']}. {item['question']}**")
                st.markdown(item['answer'])
                st.markdown("---")

