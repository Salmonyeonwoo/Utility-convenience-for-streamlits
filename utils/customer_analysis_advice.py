# ========================================
# utils/customer_analysis_advice.py
# 고객 분석 - 초기 조언 생성 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from llm_client import run_llm
from utils.customer_analysis_language import detect_text_language
from utils.customer_analysis_profile import analyze_customer_profile
from utils.customer_analysis_similarity import find_similar_cases
from utils.customer_analysis_guidelines import generate_guideline_from_past_cases


def generate_initial_advice(customer_query, customer_type_display, customer_email, customer_phone, current_lang_key,
                             customer_attachment_file):
    """Supervisor 가이드라인과 초안을 생성하는 함수"""
    if current_lang_key and current_lang_key in ["ko", "en", "ja"]:
        lang_key_to_use = current_lang_key
    else:
        lang_key_to_use = st.session_state.get("language", "ko")
        if lang_key_to_use not in ["ko", "en", "ja"]:
            try:
                detected_lang = detect_text_language(customer_query)
                if detected_lang and detected_lang in ["ko", "en", "ja"]:
                    lang_key_to_use = detected_lang
                else:
                    lang_key_to_use = "ko"
            except Exception:
                lang_key_to_use = "ko"
    
    L = LANG.get(lang_key_to_use, LANG["ko"])

    # 연락처 정보 블록
    contact_info_block = _build_contact_info_block(customer_email, customer_phone, lang_key_to_use, L)
    
    # 첨부 파일 블록
    attachment_block = ""
    if customer_attachment_file:
        file_name = customer_attachment_file.name
        attachment_block = f"\n\n[ATTACHMENT NOTE]: {L['attachment_info_llm'].format(filename=file_name)}"

    # 고객 프로필 분석
    customer_profile = analyze_customer_profile(customer_query, lang_key_to_use)

    # 유사 케이스 찾기
    similar_cases = find_similar_cases(customer_query, customer_profile, lang_key_to_use, limit=5)

    # 과거 케이스 기반 가이드라인 생성
    past_cases_guideline = ""
    if similar_cases:
        past_cases_guideline = generate_guideline_from_past_cases(
            customer_query, customer_profile, similar_cases, lang_key_to_use
        )

    # 프로필 블록
    profile_block = _build_profile_block(customer_profile, lang_key_to_use)
    
    # 과거 케이스 블록
    past_cases_block = _build_past_cases_block(past_cases_guideline, similar_cases, lang_key_to_use)

    # 프롬프트 생성
    initial_prompt = _build_initial_prompt(
        customer_query, customer_type_display, lang_key_to_use, L,
        contact_info_block, attachment_block, profile_block, past_cases_block
    )

    if not st.session_state.is_llm_ready:
        return _generate_mock_text(customer_type_display, lang_key_to_use, L)
    else:
        with st.spinner(L["response_generating"]):
            try:
                return run_llm(initial_prompt)
            except Exception as e:
                error_msg = {
                    "ko": f"AI 조언 생성 중 오류 발생: {e}",
                    "en": f"Error occurred while generating AI advice: {e}",
                    "ja": f"AIアドバイス生成中にエラーが発生しました: {e}"
                }.get(lang_key_to_use, f"Error: {e}")
                st.error(error_msg)
                return f"❌ {error_msg}"


def _build_contact_info_block(customer_email, customer_phone, lang_key_to_use, L):
    """연락처 정보 블록 생성"""
    if not customer_email and not customer_phone:
        return ""
    
    if lang_key_to_use == "ko":
        contact_info_text = "고객 연락처 정보 (참고용, 응대 초안에는 사용하지 마세요!)"
        email_label, phone_label, na_text = "이메일", "전화번호", "없음"
    elif lang_key_to_use == "en":
        contact_info_text = "Customer contact info for reference (DO NOT use these in your reply draft!)"
        email_label, phone_label, na_text = "Email", "Phone", "N/A"
    else:  # ja
        contact_info_text = "顧客連絡先情報（参考用、対応草案には使用しないでください！）"
        email_label, phone_label, na_text = "メール", "電話番号", "なし"
    
    return (
        f"\n\n[{contact_info_text}]\n"
        f"- {email_label}: {customer_email or na_text}\n"
        f"- {phone_label}: {customer_phone or na_text}"
    )


def _build_profile_block(customer_profile, lang_key_to_use):
    """프로필 블록 생성"""
    gender_display = customer_profile.get('gender', 'unknown')
    
    if lang_key_to_use == "ko":
        return f"""
[고객 프로필 분석]
- 성별: {gender_display}
- 감정 점수: {customer_profile.get('sentiment_score', 50)}/100
- 커뮤니케이션 스타일: {customer_profile.get('communication_style', 'unknown')}
- 긴급도: {customer_profile.get('urgency_level', 'medium')}
- 예측 유형: {customer_profile.get('predicted_customer_type', 'normal')}
- 주요 관심사: {', '.join(customer_profile.get('key_concerns', []))}
- 톤: {customer_profile.get('tone_analysis', 'unknown')}
"""
    elif lang_key_to_use == "en":
        return f"""
[Customer Profile Analysis]
- Gender: {gender_display}
- Sentiment Score: {customer_profile.get('sentiment_score', 50)}/100
- Communication Style: {customer_profile.get('communication_style', 'unknown')}
- Urgency Level: {customer_profile.get('urgency_level', 'medium')}
- Predicted Type: {customer_profile.get('predicted_customer_type', 'normal')}
- Key Concerns: {', '.join(customer_profile.get('key_concerns', []))}
- Tone: {customer_profile.get('tone_analysis', 'unknown')}
"""
    else:  # ja
        return f"""
[顧客プロファイル分析]
- 性別: {gender_display}
- 感情スコア: {customer_profile.get('sentiment_score', 50)}/100
- コミュニケーションスタイル: {customer_profile.get('communication_style', 'unknown')}
- 緊急度: {customer_profile.get('urgency_level', 'medium')}
- 予測タイプ: {customer_profile.get('predicted_customer_type', 'normal')}
- 主要な関心事: {', '.join(customer_profile.get('key_concerns', []))}
- トーン: {customer_profile.get('tone_analysis', 'unknown')}
"""


def _build_past_cases_block(past_cases_guideline, similar_cases, lang_key_to_use):
    """과거 케이스 블록 생성"""
    if past_cases_guideline:
        if lang_key_to_use == "ko":
            return f"""
[유사한 과거 {len(similar_cases)}개 사례 기반 가이드라인]
{past_cases_guideline}
"""
        elif lang_key_to_use == "en":
            return f"""
[Guidelines Based on {len(similar_cases)} Similar Past Cases]
{past_cases_guideline}
"""
        else:  # ja
            return f"""
[類似した過去{len(similar_cases)}件の事例に基づくガイドライン]
{past_cases_guideline}
"""
    elif similar_cases:
        if lang_key_to_use == "ko":
            return f"""
[참고: 유사한 과거 사례 {len(similar_cases)}개를 찾았지만 상세 가이드라인을 생성할 수 없습니다.
패턴을 확인하기 위해 과거 사례를 수동으로 검토하는 것을 고려하세요.]
"""
        elif lang_key_to_use == "en":
            return f"""
[Note: Found {len(similar_cases)} similar past cases, but unable to generate detailed guidelines.
Consider reviewing past cases manually for patterns.]
"""
        else:  # ja
            return f"""
[注: 類似した過去の事例{len(similar_cases)}件が見つかりましたが、詳細なガイドラインを生成できませんでした。
パターンを確認するために、過去の事例を手動で確認することを検討してください。]
"""
    return ""


def _build_initial_prompt(customer_query, customer_type_display, lang_key_to_use, L,
                          contact_info_block, attachment_block, profile_block, past_cases_block):
    """초기 프롬프트 생성"""
    if lang_key_to_use == "ko":
        return f"""
🔴 🔴 🔴 극도로 중요 🔴 🔴 🔴
당신의 모든 응답(가이드라인과 초안 포함)은 반드시 100% 한국어로 작성되어야 합니다.
영어나 일본어를 절대 사용하지 마세요. 모든 텍스트는 한국어여야 합니다.
🔴 🔴 🔴 극도로 중요 🔴 🔴 🔴

당신은 AI 고객 지원 슈퍼바이저입니다. 다음 고객 문의를 분석하여 제공하세요:
고객 유형: **{customer_type_display}**

1) 상담원을 위한 상세한 **응대 가이드라인** (단계별, 반드시 한국어로)
2) **전송 가능한 응대 초안** (반드시 한국어로)

[응답 형식 - 반드시 이 형식으로 작성하세요]
### {L['simulation_advice_header']}

(여기에 한국어로 가이드라인 작성)

### {L['simulation_draft_header']}

(여기에 한국어로 초안 작성)

[중요 가이드라인 규칙]
1. **초기 정보 수집 (요청 3):** 가이드라인의 첫 번째 단계는 문제 해결을 시도하기 전에 필수적인 초기 진단 정보(예: 기기 호환성, 현지 상태/위치, 주문 번호)를 요청해야 합니다.
2. **어려운 고객에 대한 공감 (요청 5):** 고객 유형이 '어려운 고객' 또는 '매우 불만족 고객'인 경우, 정책(예: 환불 불가)을 강제해야 하더라도 가이드라인은 극도의 정중함, 공감, 사과를 강조해야 합니다.
3. **24-48시간 후속 조치 (요청 6):** 문제를 즉시 해결할 수 없거나 현지 파트너/슈퍼바이저의 확인이 필요한 경우, 가이드라인은 다음 절차를 명시해야 합니다:
   - 문제를 인정합니다.
   - 고객에게 24시간 또는 48시간 내에 명확한 답변을 받을 것임을 알립니다.
   - 후속 연락을 위해 고객의 이메일 또는 전화번호를 요청합니다. (제공된 연락처 정보가 있으면 사용)
4. **과거 사례 학습:** 과거 사례 가이드라인이 제공된 경우, 해당 사례의 성공적인 전략을 권장사항에 통합하세요.

⚠️ 언어 요구사항: 모든 텍스트(가이드라인, 초안, 설명 등)는 반드시 한국어로만 작성하세요. 영어나 일본어를 사용하면 안 됩니다.

고객 문의:
{customer_query}
{contact_info_block}
{attachment_block}
{profile_block}
{past_cases_block}
"""
    elif lang_key_to_use == "en":
        return f"""
🔴 🔴 🔴 EXTREMELY IMPORTANT 🔴 🔴 🔴
ALL your responses (including guidelines and draft) MUST be written 100% in English.
Do NOT use Korean or Japanese. All text must be in English.
🔴 🔴 🔴 EXTREMELY IMPORTANT 🔴 🔴 🔴

You are an AI Customer Support Supervisor. Your role is to analyze the following customer inquiry
from a **{customer_type_display}** and provide:

1) A detailed **response guideline for the human agent** (step-by-step, must be in English).
2) A **ready-to-send draft reply** (must be in English).

[RESPONSE FORMAT - You MUST write in this format]
### {L['simulation_advice_header']}

(Write guidelines here in English)

### {L['simulation_draft_header']}

(Write draft here in English)

[CRITICAL GUIDELINE RULES]
1. **Initial Information Collection (Req 3):** The first step in the guideline MUST be to request the necessary initial diagnostic information (e.g., device compatibility, local status/location, order number) BEFORE attempting to troubleshoot or solve the problem.
2. **Empathy for Difficult Customers (Req 5):** If the customer type is 'Difficult Customer' or 'Highly Dissatisfied Customer', the guideline MUST emphasize extreme politeness, empathy, and apologies, even if the policy (e.g., no refund) must be enforced.
3. **24-48 Hour Follow-up (Req 6):** If the issue cannot be solved immediately or requires confirmation from a local partner/supervisor, the guideline MUST state the procedure:
   - Acknowledge the issue.
   - Inform the customer they will receive a definite answer within 24 or 48 hours.
   - Request the customer's email or phone number for follow-up contact. (Use provided contact info if available)
4. **Past Cases Learning:** If past cases guidelines are provided, incorporate successful strategies from those cases into your recommendations.

⚠️ LANGUAGE REQUIREMENT: All text (guidelines, draft, descriptions, etc.) MUST be written ONLY in English. Do NOT use Korean or Japanese.

Customer Inquiry:
{customer_query}
{contact_info_block}
{attachment_block}
{profile_block}
{past_cases_block}
"""
    else:  # ja
        return f"""
🔴 🔴 🔴 極めて重要 🔴 🔴 🔴
あなたのすべての応答（ガイドラインと草案を含む）は必ず100%日本語で作成する必要があります。
英語や韓国語を使用しないでください。すべてのテキストは日本語でなければなりません。
🔴 🔴 🔴 極めて重要 🔴 🔴 🔴

あなたはAIカスタマーサポートスーパーバイザーです。以下の顧客問い合わせを分析し、提供してください：
顧客タイプ: **{customer_type_display}**

1) 人間のエージェントのための詳細な**対応ガイドライン**（ステップバイステップ、必ず日本語で）
2) **送信可能な対応草案**（必ず日本語で）

[応答形式 - 必ずこの形式で作成してください]
### {L['simulation_advice_header']}

(ここに日本語でガイドラインを作成)

### {L['simulation_draft_header']}

(ここに日本語で草案を作成)

[重要なガイドライン規則]
1. **初期情報収集（要件3）：** ガイドラインの最初のステップは、問題のトラブルシューティングや解決を試みる前に、必要な初期診断情報（例：デバイスの互換性、現地の状態/場所、注文番号）を要求することです。
2. **困難な顧客への共感（要件5）：** 顧客タイプが「困難な顧客」または「非常に不満足な顧客」の場合、ポリシー（例：返金不可）を強制する必要がある場合でも、ガイドラインは極度の丁寧さ、共感、謝罪を強調する必要があります。
3. **24-48時間のフォローアップ（要件6）：** 問題を即座に解決できない場合、または現地パートナー/スーパーバイザーの確認が必要な場合、ガイドラインは次の手順を記載する必要があります：
   - 問題を認識します。
   - 顧客に24時間または48時間以内に明確な回答を受けることを通知します。
   - フォローアップ連絡のために顧客のメールまたは電話番号を要求します。（提供された連絡先情報がある場合は使用）
4. **過去の事例学習：** 過去の事例ガイドラインが提供されている場合、それらの事例の成功戦略を推奨事項に組み込んでください。

⚠️ 言語要件: すべてのテキスト（ガイドライン、草案、説明など）は必ず日本語でのみ作成してください。英語や韓国語を使用しないでください。

顧客問い合わせ:
{customer_query}
{contact_info_block}
{attachment_block}
{profile_block}
{past_cases_block}
"""


def _generate_mock_text(customer_type_display, lang_key_to_use, L):
    """Mock 텍스트 생성"""
    if lang_key_to_use == "ko":
        return (
            f"### {L['simulation_advice_header']}\n\n"
            f"- (Mock) {customer_type_display} 유형 고객 응대 가이드입니다. (요청 3, 5, 6 반영)\n\n"
            f"### {L['simulation_draft_header']}\n\n"
            f"(Mock) 에이전트 응대 초안이 여기에 들어갑니다.\n\n"
        )
    elif lang_key_to_use == "en":
        return (
            f"### {L['simulation_advice_header']}\n\n"
            f"- (Mock) Response guide for {customer_type_display} type customer. (Reflects Req 3, 5, 6)\n\n"
            f"### {L['simulation_draft_header']}\n\n"
            f"(Mock) Agent response draft will appear here.\n\n"
        )
    else:  # ja
        return (
            f"### {L['simulation_advice_header']}\n\n"
            f"- (Mock) {customer_type_display}タイプの顧客対応ガイドです。（要件3、5、6を反映）\n\n"
            f"### {L['simulation_draft_header']}\n\n"
            f"(Mock) エージェント対応草案がここに表示されます。\n\n"
        )
