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
LLM 프롬프트 생성 모듈
"""


def get_company_info_prompt(company_name: str, lang: str = "ko") -> str:
    """회사 정보 생성을 위한 프롬프트 생성"""
    lang_prompts = {
        "ko": f"""다음 회사에 대한 상세 정보를 제공해주세요: {company_name}

⚠️ 중요: 모든 텍스트는 한국어로 작성해주세요. 제품명, 설명 등 모든 내용은 한국어로 제공해주세요.

다음 형식으로 JSON으로 응답해주세요:
{{
    "company_info": "회사 소개 (500자 이상, 한국어로 작성)",
    "popular_products": [
        {{"text_ko": "상품명1", "text_en": "Product Name 1", "text_ja": "商品名1", "score": 85, "image_url": ""}},
        {{"text_ko": "상품명2", "text_en": "Product Name 2", "text_ja": "商品名2", "score": 80, "image_url": ""}},
        {{"text_ko": "상품명3", "text_en": "Product Name 3", "text_ja": "商品名3", "score": 75, "image_url": ""}}
    ],
    "trending_topics": [
        {{"text_ko": "화제 소식1", "text_en": "Trending News 1", "text_ja": "話題のニュース1", "score": 90, "detail_ko": "화제 소식1에 대한 상세 내용입니다. 구체적인 설명과 배경 정보를 포함합니다.", "detail_en": "Detailed content about Trending News 1.", "detail_ja": "話題のニュース1に関する詳細内容です。"}},
        {{"text_ko": "화제 소식2", "text_en": "Trending News 2", "text_ja": "話題のニュース2", "score": 85, "detail_ko": "화제 소식2에 대한 상세 내용입니다. 구체적인 설명과 배경 정보를 포함합니다.", "detail_en": "Detailed content about Trending News 2.", "detail_ja": "話題のニュース2に関する詳細内容です。"}},
        {{"text_ko": "화제 소식3", "text_en": "Trending News 3", "text_ja": "話題のニュース3", "score": 80, "detail_ko": "화제 소식3에 대한 상세 내용입니다. 구체적인 설명과 배경 정보를 포함합니다.", "detail_en": "Detailed content about Trending News 3.", "detail_ja": "話題のニュース3に関する詳細内容です。"}}
    ],
    "faqs": [
        {{"question_ko": "질문1", "question_en": "Question 1", "question_ja": "質問1", "answer_ko": "답변1", "answer_en": "Answer 1", "answer_ja": "回答1"}},
        {{"question_ko": "질문2", "answer_ko": "답변2"}},
        {{"question_ko": "질문3", "answer_ko": "답변3"}},
        {{"question_ko": "질문4", "answer_ko": "답변4"}},
        {{"question_ko": "질문5", "answer_ko": "답변5"}},
        {{"question_ko": "질문6", "answer_ko": "답변6"}},
        {{"question_ko": "질문7", "answer_ko": "답변7"}},
        {{"question_ko": "질문8", "answer_ko": "답변8"}},
        {{"question_ko": "질문9", "answer_ko": "답변9"}},
        {{"question_ko": "질문10", "answer_ko": "답변10"}}
    ],
    "interview_questions": [
        {{"question_ko": "면접 질문1", "answer_ko": "면접 질문1에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "회사 이해"}},
        {{"question_ko": "면접 질문2", "answer_ko": "면접 질문2에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "업무 관련"}},
        {{"question_ko": "면접 질문3", "answer_ko": "면접 질문3에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "직무 이해"}},
        {{"question_ko": "면접 질문4", "answer_ko": "면접 질문4에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "회사 이해"}},
        {{"question_ko": "면접 질문5", "answer_ko": "면접 질문5에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "업무 관련"}}
    ],
    "ceo_info": {{
        "name_ko": "대표이사/CEO 이름",
        "position_ko": "직책 (예: 대표이사, CEO, 공동대표이사 등)",
        "bio_ko": "대표이사/CEO에 대한 상세한 소개입니다. 학력, 경력, 주요 성과, 리더십 스타일 등을 포함하여 300자 이상 작성해주세요.",
        "tenure_ko": "재임 기간 (예: 2020년 ~ 현재)",
        "education_ko": "학력 정보",
        "career_ko": "주요 경력 및 성과"
    }}
}}

FAQ는 10개를 생성해주세요. 실제로 자주 묻는 질문과 답변을 포함해주세요.
면접 질문은 5개 이상 생성해주세요. 실제 면접에서 나올 만한 핵심 질문들을 포함하고, 각 질문에 대한 상세한 답변(200자 이상)과 카테고리(회사 이해, 업무 관련, 직무 이해 등)를 제공해주세요.
CEO/대표이사 정보는 현재 재임 중인 대표이사 또는 CEO의 실제 정보를 포함해주세요. 이름, 직책, 상세 소개(300자 이상), 재임 기간, 학력, 주요 경력을 포함해주세요.""",
        "en": f"""Please provide detailed information about the following company: {company_name}

⚠️ Important: All text should be written in English. Product names, descriptions, and all content should be provided in English.

Respond in JSON format as follows:
{{
    "company_info": "Company introduction (500+ characters, written in English)",
    "popular_products": [
        {{"text_ko": "상품명1", "text_en": "Product Name 1", "text_ja": "商品名1", "score": 85, "image_url": ""}},
        {{"text_ko": "상품명2", "text_en": "Product Name 2", "text_ja": "商品名2", "score": 80, "image_url": ""}},
        {{"text_ko": "상품명3", "text_en": "Product Name 3", "text_ja": "商品名3", "score": 75, "image_url": ""}}
    ],
    "trending_topics": [
        {{"text_ko": "화제 소식1", "text_en": "Trending News 1", "text_ja": "話題のニュース1", "score": 90, "detail_ko": "화제 소식1에 대한 상세 내용입니다.", "detail_en": "Detailed content about Trending News 1, including specific explanations and background information.", "detail_ja": "話題のニュース1に関する詳細内容です。"}},
        {{"text_ko": "화제 소식2", "text_en": "Trending News 2", "text_ja": "話題のニュース2", "score": 85, "detail_ko": "화제 소식2에 대한 상세 내용입니다.", "detail_en": "Detailed content about Trending News 2, including specific explanations and background information.", "detail_ja": "話題のニュース2に関する詳細内容です。"}},
        {{"text_ko": "화제 소식3", "text_en": "Trending News 3", "text_ja": "話題のニュース3", "score": 80, "detail_ko": "화제 소식3에 대한 상세 내용입니다.", "detail_en": "Detailed content about Trending News 3, including specific explanations and background information.", "detail_ja": "話題のニュース3に関する詳細内容です。"}}
    ],
    "faqs": [
        {{"question_ko": "질문1", "question_en": "Question 1", "question_ja": "質問1", "answer_ko": "답변1", "answer_en": "Answer 1", "answer_ja": "回答1"}},
        {{"question_en": "Question2", "answer_en": "Answer2"}},
        {{"question_en": "Question3", "answer_en": "Answer3"}},
        {{"question_en": "Question4", "answer_en": "Answer4"}},
        {{"question_en": "Question5", "answer_en": "Answer5"}},
        {{"question_en": "Question6", "answer_en": "Answer6"}},
        {{"question_en": "Question7", "answer_en": "Answer7"}},
        {{"question_en": "Question8", "answer_en": "Answer8"}},
        {{"question_en": "Question9", "answer_en": "Answer9"}},
        {{"question_en": "Question10", "answer_en": "Answer10"}}
    ],
    "interview_questions": [
        {{"question_en": "Interview Question 1", "answer_en": "Detailed answer for interview question 1. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Company Understanding"}},
        {{"question_en": "Interview Question 2", "answer_en": "Detailed answer for interview question 2. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Work Related"}},
        {{"question_en": "Interview Question 3", "answer_en": "Detailed answer for interview question 3. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Job Understanding"}},
        {{"question_en": "Interview Question 4", "answer_en": "Detailed answer for interview question 4. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Company Understanding"}},
        {{"question_en": "Interview Question 5", "answer_en": "Detailed answer for interview question 5. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Work Related"}}
    ],
    "ceo_info": {{
        "name_en": "CEO/President Name",
        "position_en": "Position (e.g., CEO, President, Co-CEO, etc.)",
        "bio_en": "Detailed introduction of the CEO/President. Include education, career, major achievements, leadership style, etc. (300+ characters)",
        "tenure_en": "Tenure (e.g., 2020 - Present)",
        "education_en": "Education Information",
        "career_en": "Major Career and Achievements"
    }}
}}

Generate 10 FAQs with real frequently asked questions and answers.
Generate at least 5 interview questions that are likely to be asked in actual interviews. Include core questions with detailed answers (200+ characters) and categories (Company Understanding, Work Related, Job Understanding, etc.) for each question.
Include CEO/President information for the current CEO or President. Include name, position, detailed introduction (300+ characters), tenure, education, and major career achievements.""",
        "ja": f"""次の会社に関する詳細情報を提供してください: {company_name}

⚠️ 重要: すべてのテキストは日本語で記述してください。商品名、説明など、すべての内容は日本語で提供してください。

次の形式でJSONで応答してください:
{{
    "company_info": "会社紹介 (500文字以上、日本語で記述)",
    "popular_products": [
        {{"text_ko": "상품명1", "text_en": "Product Name 1", "text_ja": "商品名1", "score": 85, "image_url": ""}},
        {{"text_ko": "상품명2", "text_en": "Product Name 2", "text_ja": "商品名2", "score": 80, "image_url": ""}},
        {{"text_ko": "상품명3", "text_en": "Product Name 3", "text_ja": "商品名3", "score": 75, "image_url": ""}}
    ],
    "trending_topics": [
        {{"text_ko": "화제 소식1", "text_en": "Trending News 1", "text_ja": "話題のニュース1", "score": 90, "detail_ko": "화제 소식1에 대한 상세 내용입니다.", "detail_en": "Detailed content about Trending News 1.", "detail_ja": "話題のニュース1に関する詳細内容です。具体的な説明と背景情報を含みます。"}},
        {{"text_ko": "화제 소식2", "text_en": "Trending News 2", "text_ja": "話題のニュース2", "score": 85, "detail_ko": "화제 소식2에 대한 상세 내용입니다.", "detail_en": "Detailed content about Trending News 2.", "detail_ja": "話題のニュース2に関する詳細内容です。具体的な説明と背景情報を含みます。"}},
        {{"text_ko": "화제 소식3", "text_en": "Trending News 3", "text_ja": "話題のニュース3", "score": 80, "detail_ko": "화제 소식3에 대한 상세 내용입니다.", "detail_en": "Detailed content about Trending News 3.", "detail_ja": "話題のニュース3に関する詳細内容です。具体的な説明と背景情報を含みます。"}}
    ],
    "faqs": [
        {{"question_ko": "질문1", "question_en": "Question 1", "question_ja": "質問1", "answer_ko": "답변1", "answer_en": "Answer 1", "answer_ja": "回答1"}},
        {{"question_ja": "質問2", "answer_ja": "回答2"}},
        {{"question_ja": "質問3", "answer_ja": "回答3"}},
        {{"question_ja": "質問4", "answer_ja": "回答4"}},
        {{"question_ja": "質問5", "answer_ja": "回答5"}},
        {{"question_ja": "質問6", "answer_ja": "回答6"}},
        {{"question_ja": "質問7", "answer_ja": "回答7"}},
        {{"question_ja": "質問8", "answer_ja": "回答8"}},
        {{"question_ja": "質問9", "answer_ja": "回答9"}},
        {{"question_ja": "質問10", "answer_ja": "回答10"}}
    ],
    "interview_questions": [
        {{"question_ja": "面接質問1", "answer_ja": "面接質問1に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "会社理解"}},
        {{"question_ja": "面接質問2", "answer_ja": "面接質問2に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "業務関連"}},
        {{"question_ja": "面接質問3", "answer_ja": "面接質問3に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "職務理解"}},
        {{"question_ja": "面接質問4", "answer_ja": "面接質問4に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "会社理解"}},
        {{"question_ja": "面接質問5", "answer_ja": "面接質問5に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "業務関連"}}
    ],
    "ceo_info": {{
        "name_ja": "代表取締役/CEO名",
        "position_ja": "役職（例：代表取締役、CEO、共同代表取締役など）",
        "bio_ja": "代表取締役/CEOに関する詳細な紹介です。学歴、経歴、主要な成果、リーダーシップスタイルなどを含めて300文字以上で作成してください。",
        "tenure_ja": "在任期間（例：2020年～現在）",
        "education_ja": "学歴情報",
        "career_ja": "主要な経歴および成果"
    }}
}}

FAQは10個生成してください。実際によくある質問と回答を含めてください。
面接質問は5個以上生成してください。実際の面接で出る可能性のある核心的な質問を含め、各質問に対する詳細な回答（200文字以上）とカテゴリー（会社理解、業務関連、職務理解など）を提供してください。
CEO/代表取締役情報は現在在任中の代表取締役またはCEOの実際の情報を含めてください。名前、役職、詳細な紹介（300文字以上）、在任期間、学歴、主要な経歴を含めてください。"""
    }
    
    return lang_prompts.get(lang, lang_prompts["ko"])

