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
FAQ 검색 모듈
"""

from typing import List, Dict, Any
from faq.database import load_faq_database


def search_faq(faq_data: Dict[str, Any], company: str, query: str, lang: str = "ko") -> List[Dict[str, str]]:
    """FAQ 검색"""
    if not query or not query.strip():
        return []
    
    results = []
    query_lower = query.lower().strip()
    
    # 회사별 FAQ 검색
    if company and company in faq_data.get("companies", {}):
        company_faqs = faq_data["companies"][company].get("faqs", [])
        for faq in company_faqs:
            question = faq.get(f"question_{lang}", faq.get("question_ko", ""))
            answer = faq.get(f"answer_{lang}", faq.get("answer_ko", ""))
            
            if query_lower in question.lower() or query_lower in answer.lower():
                results.append({
                    "question": question,
                    "answer": answer,
                    "company": company
                })
    
    # 기본 FAQ 검색 (회사가 없거나 기본 설정인 경우)
    default_settings_texts = ["기본 설정", "Default Settings", "デフォルト設定"]
    if not company or company in default_settings_texts:
        default_faqs = faq_data.get("default", {}).get("faqs", [])
        for faq in default_faqs:
            question = faq.get(f"question_{lang}", faq.get("question_ko", ""))
            answer = faq.get(f"answer_{lang}", faq.get("answer_ko", ""))
            
            if query_lower in question.lower() or query_lower in answer.lower():
                results.append({
                    "question": question,
                    "answer": answer,
                    "company": "기본"
                })
    
    return results

