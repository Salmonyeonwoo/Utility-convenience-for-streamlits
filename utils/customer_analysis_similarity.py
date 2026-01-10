# ========================================
# utils/customer_analysis_similarity.py
# 고객 분석 - 유사 케이스 찾기 모듈
# ========================================

from typing import List, Dict, Any
from utils.history_handler import load_simulation_histories_local


def find_similar_cases(customer_query: str, customer_profile: Dict[str, Any], current_lang_key: str,
                       limit: int = 5) -> List[Dict[str, Any]]:
    """저장된 요약 데이터에서 유사한 케이스를 찾아 반환"""
    histories = load_simulation_histories_local(current_lang_key)

    if not histories:
        return []

    cases_with_summary = [
        h for h in histories
        if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
           and not h.get("is_call", False)
    ]

    if not cases_with_summary:
        return []

    similar_cases = []
    query_lower = customer_query.lower()
    customer_sentiment = customer_profile.get("sentiment_score", 50)

    for case in cases_with_summary:
        summary = case.get("summary", {})
        main_inquiry = summary.get("main_inquiry", "").lower()
        case_sentiment = summary.get("customer_sentiment_score", 50)
        case_satisfaction = summary.get("customer_satisfaction_score", 50)

        similarity_score = 0

        # 문의 내용 유사도
        query_words = set(query_lower.split())
        inquiry_words = set(main_inquiry.split())
        if query_words and inquiry_words:
            word_overlap = len(query_words & inquiry_words) / len(query_words | inquiry_words)
            similarity_score += word_overlap * 40

        # 감정 점수 유사도
        sentiment_diff = abs(customer_sentiment - case_sentiment)
        sentiment_similarity = max(0, 1 - (sentiment_diff / 100)) * 30
        similarity_score += sentiment_similarity

        # 만족도 점수
        satisfaction_bonus = (case_satisfaction / 100) * 30
        similarity_score += satisfaction_bonus

        if similarity_score > 30:
            similar_cases.append({
                "case": case,
                "similarity_score": similarity_score,
                "summary": summary
            })

    similar_cases.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar_cases[:limit]
