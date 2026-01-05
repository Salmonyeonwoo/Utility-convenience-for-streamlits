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
LLM을 사용한 회사 정보 생성 모듈
"""

import json
import re
from typing import Dict, Any
from faq.llm_prompts import get_company_info_prompt
from faq.common_faqs import get_common_product_faqs


def generate_company_info_with_llm(company_name: str, lang: str = "ko") -> Dict[str, Any]:
    """LLM을 사용하여 회사 정보 생성"""
    prompt = get_company_info_prompt(company_name, lang)
    
    try:
        from llm_client import run_llm
        response = run_llm(prompt)
        
        # JSON 파싱 시도
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group()
            try:
                company_data = json.loads(json_str)
                
                # 공동 대표 제품 FAQ 추가
                common_faqs = get_common_product_faqs(company_name, lang)
                if common_faqs:
                    existing_faqs = company_data.get("faqs", [])
                    company_data["faqs"] = common_faqs + existing_faqs
                
                # FAQ가 10개 미만이면 기본 FAQ 추가
                if len(company_data.get("faqs", [])) < 10:
                    default_faqs_by_lang = {
                        "ko": [
                            {"question_ko": "회사 설립일은 언제인가요?", "answer_ko": "회사 설립일에 대한 정보를 확인 중입니다."},
                            {"question_ko": "주요 사업 분야는 무엇인가요?", "answer_ko": "주요 사업 분야에 대한 정보를 확인 중입니다."},
                            {"question_ko": "본사 위치는 어디인가요?", "answer_ko": "본사 위치에 대한 정보를 확인 중입니다."},
                            {"question_ko": "직원 수는 얼마나 되나요?", "answer_ko": "직원 수에 대한 정보를 확인 중입니다."},
                            {"question_ko": "주요 제품/서비스는 무엇인가요?", "answer_ko": "주요 제품/서비스에 대한 정보를 확인 중입니다."},
                        ],
                        "en": [
                            {"question_en": "When was the company founded?", "answer_en": "We are checking information about the company's founding date."},
                            {"question_en": "What are the main business areas?", "answer_en": "We are checking information about the main business areas."},
                            {"question_en": "Where is the headquarters located?", "answer_en": "We are checking information about the headquarters location."},
                            {"question_en": "How many employees does the company have?", "answer_en": "We are checking information about the number of employees."},
                            {"question_en": "What are the main products/services?", "answer_en": "We are checking information about the main products/services."},
                        ],
                        "ja": [
                            {"question_ja": "会社の設立日はいつですか？", "answer_ja": "会社の設立日に関する情報を確認中です。"},
                            {"question_ja": "主要な事業分野は何ですか？", "answer_ja": "主要な事業分野に関する情報を確認中です。"},
                            {"question_ja": "本社の所在地はどこですか？", "answer_ja": "本社の所在地に関する情報を確認中です。"},
                            {"question_ja": "従業員数は何人ですか？", "answer_ja": "従業員数に関する情報を確認中です。"},
                            {"question_ja": "主要な製品・サービスは何ですか？", "answer_ja": "主要な製品・サービスに関する情報を確認中です。"},
                        ]
                    }
                    default_faqs = default_faqs_by_lang.get(lang, default_faqs_by_lang["ko"])
                    existing_faqs = company_data.get("faqs", [])
                    while len(existing_faqs) < 10:
                        idx = len(existing_faqs) % len(default_faqs)
                        existing_faqs.append(default_faqs[idx])
                    company_data["faqs"] = existing_faqs[:10]
                return company_data
            except json.JSONDecodeError:
                return _create_default_response(response)
        else:
            return _create_default_response(response)
    except Exception as e:
        error_messages = {
            "ko": f"회사 정보 생성 중 오류가 발생했습니다: {str(e)}",
            "en": f"An error occurred while generating company information: {str(e)}",
            "ja": f"会社情報の生成中にエラーが発生しました: {str(e)}"
        }
        return {
            "company_info": error_messages.get(lang, error_messages["ko"]),
            "popular_products": [],
            "trending_topics": [],
            "faqs": [],
            "interview_questions": [],
            "ceo_info": {}
        }


def _create_default_response(response: str) -> Dict[str, Any]:
    """기본 응답 구조 생성"""
    return {
        "company_info": response[:1000] if len(response) > 1000 else response,
        "popular_products": [],
        "trending_topics": [],
        "faqs": [],
        "interview_questions": [],
        "ceo_info": {}
    }

