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
FAQ 데이터베이스 관리 모듈
"""

from typing import Dict, Any
from config import FAQ_DB_FILE
from utils import _load_json, _save_json


def load_faq_database() -> Dict[str, Any]:
    """FAQ 데이터베이스 로드"""
    return _load_json(FAQ_DB_FILE, {"companies": {}})


def save_faq_database(faq_data: Dict[str, Any]):
    """FAQ 데이터베이스 저장"""
    _save_json(FAQ_DB_FILE, faq_data)


def get_company_info_faq(company: str, lang: str = "ko") -> Dict[str, Any]:
    """회사 소개 및 FAQ 가져오기"""
    faq_data = load_faq_database()
    if company in faq_data.get("companies", {}):
        company_data = faq_data["companies"][company]
        return {
            "info": company_data.get(f"info_{lang}", company_data.get("info_ko", "")),
            "popular_products": company_data.get("popular_products", []),
            "trending_topics": company_data.get("trending_topics", []),
            "faqs": company_data.get("faqs", []),
            "interview_questions": company_data.get("interview_questions", []),
            "ceo_info": company_data.get("ceo_info", {})
        }
    return {
        "info": "",
        "popular_products": [],
        "trending_topics": [],
        "faqs": [],
        "interview_questions": [],
        "ceo_info": {}
    }

