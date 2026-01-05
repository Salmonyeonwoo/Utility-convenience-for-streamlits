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
탭3: 고객 문의 재확인 모듈
"""

import streamlit as st
import html as html_escape
from lang_pack import LANG
from faq.database import load_faq_database, get_company_info_faq
from llm_client import run_llm
from .tab3_inquiry_utils import (
    process_uploaded_file,
    create_attachment_info
)


def render_tab3_inquiry(companies: list, current_lang: str, L: dict, faq_data: dict):
    """탭3: 고객 문의 재확인 렌더링"""
    st.markdown(f"#### {L['customer_inquiry_review']}")
    st.caption(L.get("customer_inquiry_review_desc",
                     "에이전트가 상사들에게 고객 문의 내용을 재확인하고, AI 답안 및 힌트를 생성할 수 있는 기능입니다."))

    # 세션 상태 초기화
    if "generated_ai_answer" not in st.session_state:
        st.session_state.generated_ai_answer = None
    if "generated_hint" not in st.session_state:
        st.session_state.generated_hint = None

    # 회사 선택
    selected_company_for_inquiry = None
    if companies:
        all_option = L.get("all_companies", "전체")
        selected_company_for_inquiry = st.selectbox(
            f"{L['select_company']} ({L.get('optional', '선택사항')})",
            options=[all_option] + companies,
            key="inquiry_company_select"
        )
        if selected_company_for_inquiry == all_option:
            selected_company_for_inquiry = None

    # 고객 문의 내용 입력
    customer_inquiry = st.text_area(
        L["inquiry_question_label"],
        placeholder=L["inquiry_question_placeholder"],
        key="customer_inquiry_input",
        height=150
    )

    # 고객 첨부 파일 업로드
    uploaded_file = st.file_uploader(
        L.get("inquiry_attachment_label", "📎 고객 첨부 파일 업로드 (사진/스크린샷)"),
        type=["png", "jpg", "jpeg", "pdf"],
        key="customer_inquiry_attachment",
        help=L.get("inquiry_attachment_help",
                   "특히 취소 불가 여행상품의 비행기 지연, 여권 이슈 등 불가피한 사유의 경우, 반드시 사진이나 스크린샷을 첨부해주세요."))

    # 파일 처리
    uploaded_file_info = None
    attachment_info = ""
    if uploaded_file is not None:
        uploaded_file_info, file_content = process_uploaded_file(uploaded_file, current_lang, L)
        attachment_info = create_attachment_info(uploaded_file_info, file_content, current_lang)
        
        # 이미지 파일인 경우 미리보기 표시
        if uploaded_file_info and uploaded_file_info["type"].startswith("image/"):
            st.image(uploaded_file, caption=uploaded_file_info["name"], use_container_width=True)

    # AI 답안 및 힌트 생성 버튼
    col_ai_answer, col_hint = st.columns(2)

    with col_ai_answer:
        if st.button(L["button_generate_ai_answer"], key="generate_ai_answer_btn", type="primary"):
            if customer_inquiry:
                with st.spinner(L["generating_ai_answer"]):
                    company_context = _get_company_context(
                        selected_company_for_inquiry, current_lang, L, faq_data)
                    
                    prompt = _create_ai_answer_prompt(
                        customer_inquiry, company_context, attachment_info, current_lang)
                    ai_answer = run_llm(prompt)
                    st.session_state.generated_ai_answer = ai_answer
                    st.success(f"✅ {L.get('ai_answer_generated', 'AI 답안이 생성되었습니다.')}")
            else:
                st.warning(L.get("warning_enter_inquiry", "고객 문의 내용을 입력해주세요."))

    with col_hint:
        if st.button(L["button_generate_hint"], key="generate_hint_btn", type="primary"):
            if customer_inquiry:
                with st.spinner(L["generating_hint"]):
                    company_context = _get_company_context(
                        selected_company_for_inquiry, current_lang, L, faq_data)
                    
                    prompt = _create_hint_prompt(
                        customer_inquiry, company_context, attachment_info, current_lang)
                    hint = run_llm(prompt)
                    st.session_state.generated_hint = hint
                    st.success(f"✅ {L.get('hint_generated', '응대 힌트가 생성되었습니다.')}")
            else:
                st.warning(L.get("warning_enter_inquiry", "고객 문의 내용을 입력해주세요."))

    # 생성된 결과 표시
    if st.session_state.get("generated_ai_answer"):
        _render_generated_answer(st.session_state.generated_ai_answer, L)

    if st.session_state.get("generated_hint"):
        _render_generated_hint(st.session_state.generated_hint, L)


def _get_company_context(selected_company: str, current_lang: str, L: dict, faq_data: dict) -> str:
    """회사 컨텍스트 생성"""
    company_context = ""
    if selected_company and selected_company in faq_data.get("companies", {}):
        company_data = get_company_info_faq(selected_company, current_lang)
        company_info_label = L.get("company_info", "회사 정보")
        company_context = f"\n\n{company_info_label}: {company_data.get('info', '')}"
        
        related_faqs = company_data.get("faqs", [])[:5]
        if related_faqs:
            faq_label = L.get("company_faq", "자주 나오는 질문")
            faq_context = f"\n\n{faq_label}:\n"
            for faq in related_faqs:
                q = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                a = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                faq_context += f"Q: {q}\nA: {a}\n"
            company_context += faq_context
    return company_context


def _create_ai_answer_prompt(customer_inquiry: str, company_context: str, attachment_info: str, current_lang: str) -> str:
    """AI 답안 생성 프롬프트 생성"""
    lang_prompts = {
        "ko": f"""다음 고객 문의에 대한 전문적이고 친절한 답안을 작성해주세요.

고객 문의: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

답안은 다음을 포함해야 합니다:
1. 고객의 문의에 대한 명확한 답변
2. 필요한 경우 추가 정보나 안내
3. 친절하고 전문적인 톤
4. 첨부 파일이 있는 경우, 해당 파일 내용을 참고하여 응대하세요. 특히 취소 불가 여행상품의 비행기 지연, 여권 이슈 등 불가피한 사유의 경우, 첨부된 증빙 자료를 확인하고 적절히 대응하세요.

답안:""",
        "en": f"""Please write a professional and friendly answer to the following customer inquiry.

Customer Inquiry: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

The answer should include:
1. Clear answer to the customer's inquiry
2. Additional information or guidance if needed
3. Friendly and professional tone
4. If there is an attachment, please reference the file content in your response. For non-refundable travel products with unavoidable reasons (flight delays, passport issues, etc.), review the attached evidence and respond appropriately.

Answer:""",
        "ja": f"""次の顧客問い合わせに対する専門的で親切な回答を作成してください。

顧客問い合わせ: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

回答には以下を含める必要があります:
1. 顧客の問い合わせに対する明確な回答
2. 必要に応じて追加情報や案内
3. 親切で専門的なトーン
4. 添付ファイルがある場合は、そのファイルの内容を参照して対応してください。特にキャンセル不可の旅行商品で、飛行機の遅延、パスポートの問題などやむを得ない理由がある場合は、添付された証拠資料を確認し、適切に対応してください。

回答:"""}
    return lang_prompts.get(current_lang, lang_prompts["ko"])


def _create_hint_prompt(customer_inquiry: str, company_context: str, attachment_info: str, current_lang: str) -> str:
    """응대 힌트 생성 프롬프트 생성"""
    lang_prompts = {
        "ko": f"""다음 고객 문의에 대한 응대 힌트를 작성해주세요.

고객 문의: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

응대 힌트는 다음을 포함해야 합니다:
1. 고객 문의의 핵심 포인트
2. 응대 시 주의사항
3. 권장 응대 방식
4. 추가 확인이 필요한 사항 (있는 경우)
5. 첨부 파일이 있는 경우, 해당 파일을 확인하고 증빙 자료로 활용하세요. 특히 취소 불가 여행상품의 경우, 첨부된 사진이나 스크린샷을 통해 불가피한 사유를 확인하고 적절한 조치를 취하세요.

응대 힌트:""",
        "en": f"""Please write response hints for the following customer inquiry.

Customer Inquiry: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

Response hints should include:
1. Key points of the customer inquiry
2. Precautions when responding
3. Recommended response method
4. Items that need additional confirmation (if any)
5. If there is an attachment, review the file and use it as evidence. For non-refundable travel products, verify unavoidable reasons through attached photos or screenshots and take appropriate action.

Response Hints:""",
        "ja": f"""次の顧客問い合わせに対する対応ヒントを作成してください。

顧客問い合わせ: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

対応ヒントには以下を含める必要があります:
1. 顧客問い合わせの核心ポイント
2. 対応時の注意事項
3. 推奨対応方法
4. 追加確認が必要な事項（ある場合）
5. 添付ファイルがある場合は、そのファイルを確認し、証拠資料として活用してください。特にキャンセル不可の旅行商品の場合、添付された写真やスクリーンショットを通じてやむを得ない理由を確認し、適切な措置を取ってください。

対応ヒント:"""}
    return lang_prompts.get(current_lang, lang_prompts["ko"])


def _render_generated_answer(answer_text: str, L: dict):
    """생성된 답안 표시"""
    st.markdown("---")
    st.subheader(L["ai_answer_header"])

    answer_escaped = html_escape.escape(answer_text)
    st.markdown(f"""
    <div style="font-size: 18px; line-height: 1.8; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
    <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Malgun Gothic', '맑은 고딕', 'Noto Sans JP', sans-serif; margin: 0; font-size: 18px; color: #212529;">{answer_escaped}</pre>
    </div>
    """, unsafe_allow_html=True)

    col_copy, col_download = st.columns(2)
    with col_copy:
        st.info(L.get("copy_instruction", "💡 위 텍스트를 선택하고 Ctrl+C (Mac: Cmd+C)로 복사하세요."))
    with col_download:
        st.download_button(
            label=f"📥 {L.get('button_download_answer', '답안 다운로드')}",
            data=answer_text.encode('utf-8'),
            file_name=f"ai_answer_{st.session_state.get('copy_answer_id', 0)}.txt",
            mime="text/plain",
            key="download_answer_btn")


def _render_generated_hint(hint_text: str, L: dict):
    """생성된 힌트 표시"""
    st.markdown("---")
    st.subheader(L["hint_header"])

    hint_escaped = html_escape.escape(hint_text)
    st.markdown(f"""
    <div style="font-size: 18px; line-height: 1.8; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
    <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Malgun Gothic', '맑은 고딕', 'Noto Sans JP', sans-serif; margin: 0; font-size: 18px; color: #212529;">{hint_escaped}</pre>
    </div>
    """, unsafe_allow_html=True)

