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
탭1: 화제의 소식 표시 모듈
"""

import streamlit as st
from llm_client import run_llm
from faq.database import save_faq_database


def render_topics_section(trending_topics: list, display_company: str, current_lang: str, L: dict, faq_data: dict):
    """화제의 소식 목록 렌더링"""
    st.markdown(f"#### {L['trending_topics']}")
    
    for idx, topic in enumerate(trending_topics, 1):
        topic_text = topic.get(f"text_{current_lang}", topic.get("text_ko", ""))
        topic_score = topic.get("score", 0)
        topic_detail = topic.get(f"detail_{current_lang}", topic.get("detail_ko", ""))

        with st.expander(f"{idx}. **{topic_text}** ({L.get('trend_score', '화제도')}: {topic_score})"):
            if topic_detail:
                st.write(topic_detail)
            else:
                # 상세 내용이 없으면 LLM으로 생성
                if display_company:
                    try:
                        detail_prompts = {
                            "ko": f"{display_company}의 '{topic_text}'에 대한 상세 내용을 200자 이상 작성해주세요.",
                            "en": f"Please write detailed content of at least 200 characters about '{topic_text}' from {display_company}.",
                            "ja": f"{display_company}の「{topic_text}」に関する詳細内容を200文字以上で作成してください。"}
                        detail_prompt = detail_prompts.get(current_lang, detail_prompts["ko"])
                        generated_detail = run_llm(detail_prompt)
                        
                        if generated_detail and not generated_detail.startswith("❌"):
                            st.write(generated_detail)
                            # 생성된 상세 내용을 데이터베이스에 저장
                            if display_company in faq_data.get("companies", {}):
                                topic_idx = idx - 1
                                if topic_idx < len(
                                    faq_data["companies"][display_company].get("trending_topics", [])):
                                    faq_data["companies"][display_company]["trending_topics"][
                                        topic_idx][f"detail_{current_lang}"] = generated_detail
                                    save_faq_database(faq_data)
                        else:
                            st.write(L.get("generating_detail", "상세 내용을 생성하는 중입니다..."))
                    except Exception as e:
                        st.write(
                            L.get(
                                "checking_additional_info",
                                "상세 내용: {topic}에 대한 추가 정보를 확인 중입니다.").format(
                                topic=topic_text))
                else:
                    st.write(
                        L.get(
                            "checking_additional_info",
                            "상세 내용: {topic}에 대한 추가 정보를 확인 중입니다.").format(
                            topic=topic_text))

