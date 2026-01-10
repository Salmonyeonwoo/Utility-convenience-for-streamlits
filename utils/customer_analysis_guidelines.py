# ========================================
# utils/customer_analysis_guidelines.py
# 고객 분석 - 가이드라인 생성 모듈
# ========================================

import streamlit as st
from typing import List, Dict, Any
from llm_client import run_llm


def generate_guideline_from_past_cases(customer_query: str, customer_profile: Dict[str, Any],
                                       similar_cases: List[Dict[str, Any]], current_lang_key: str) -> str:
    """과거 유사 케이스의 성공적인 해결 방법을 바탕으로 가이드라인 생성"""
    if not similar_cases:
        return ""

    past_cases_text = ""
    for idx, similar_case in enumerate(similar_cases, 1):
        summary = similar_case["summary"]
        similarity = similar_case["similarity_score"]

        past_cases_text += f"""
[Case {idx}] (Similarity: {similarity:.1f}%)
- Inquiry: {summary.get('main_inquiry', 'N/A')}
- Customer Sentiment: {summary.get('customer_sentiment_score', 50)}/100
- Customer Satisfaction: {summary.get('customer_satisfaction_score', 50)}/100
- Key Responses: {', '.join(summary.get('key_responses', [])[:3])}
- Summary: {summary.get('summary', 'N/A')[:200]}
"""

    if current_lang_key == "ko":
        guideline_prompt = f"""
당신은 과거 성공 사례를 분석하여 가이드를 제공하는 AI 고객 지원 슈퍼바이저입니다.

다음 유사한 과거 사례와 그들의 성공적인 해결 전략을 바탕으로 현재 고객 문의를 처리하기 위한 실행 가능한 가이드라인을 제공하세요.

현재 고객 문의:
{customer_query}

현재 고객 프로필:
- 성별: {customer_profile.get('gender', 'unknown')}
- 감정 점수: {customer_profile.get('sentiment_score', 50)}/100
- 커뮤니케이션 스타일: {customer_profile.get('communication_style', 'unknown')}
- 긴급도: {customer_profile.get('urgency_level', 'medium')}
- 예측 유형: {customer_profile.get('predicted_customer_type', 'normal')}

유사한 과거 사례 (성공적인 해결):
{past_cases_text}

다음 내용을 포함한 간결한 가이드라인을 한국어로 제공하세요:
1. 유사한 과거 사례에서 잘 작동한 것 식별
2. 성공적인 패턴을 기반으로 한 구체적인 접근 방법 제안
3. 과거 경험을 바탕으로 한 잠재적 함정 경고
4. 높은 고객 만족도로 이어진 응답 전략 권장

가이드라인 (한국어로):
"""
    elif current_lang_key == "en":
        guideline_prompt = f"""
You are an AI Customer Support Supervisor analyzing past successful cases to provide guidance.

Based on the following similar past cases and their successful resolution strategies, provide actionable guidelines for handling the current customer inquiry.

Current Customer Inquiry:
{customer_query}

Current Customer Profile:
- Gender: {customer_profile.get('gender', 'unknown')}
- Sentiment Score: {customer_profile.get('sentiment_score', 50)}/100
- Communication Style: {customer_profile.get('communication_style', 'unknown')}
- Urgency: {customer_profile.get('urgency_level', 'medium')}
- Predicted Type: {customer_profile.get('predicted_customer_type', 'normal')}

Similar Past Cases (Successful Resolutions):
{past_cases_text}

Provide a concise guideline in English that:
1. Identifies what worked well in similar past cases
2. Suggests specific approaches based on successful patterns
3. Warns about potential pitfalls based on past experiences
4. Recommends response strategies that led to high customer satisfaction

Guideline (in English):
"""
    else:  # ja
        guideline_prompt = f"""
あなたは過去の成功事例を分析してガイドを提供するAIカスタマーサポートスーパーバイザーです。

以下の類似した過去の事例とその成功した解決戦略に基づいて、現在の顧客問い合わせを処理するための実行可能なガイドラインを提供してください。

現在の顧客問い合わせ:
{customer_query}

現在の顧客プロファイル:
- 性別: {customer_profile.get('gender', 'unknown')}
- 感情スコア: {customer_profile.get('sentiment_score', 50)}/100
- コミュニケーションスタイル: {customer_profile.get('communication_style', 'unknown')}
- 緊急度: {customer_profile.get('urgency_level', 'medium')}
- 予測タイプ: {customer_profile.get('predicted_customer_type', 'normal')}

類似した過去の事例（成功した解決）:
{past_cases_text}

以下の内容を含む簡潔なガイドラインを日本語で提供してください:
1. 類似した過去の事例でうまくいったことを特定
2. 成功したパターンに基づく具体的なアプローチ方法の提案
3. 過去の経験に基づく潜在的な落とし穴の警告
4. 高い顧客満足度につながった応答戦略の推奨

ガイドライン（日本語で）:
"""

    if not st.session_state.is_llm_ready:
        return ""

    try:
        guideline = run_llm(guideline_prompt).strip()
        return guideline
    except Exception as e:
        error_msg = {
            "ko": f"가이드라인 생성 오류: {str(e)}",
            "en": f"Guideline generation error: {str(e)}",
            "ja": f"ガイドライン生成エラー: {str(e)}"
        }.get(current_lang_key, f"Error: {str(e)}")
        return error_msg
