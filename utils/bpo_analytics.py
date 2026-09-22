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
BPO 비즈니스 임팩트 & ROI 분석 엔진
- AI 초안 채택률 (Draft Acceptance Rate) & 편집 거리 측정
- AHT (Average Handling Time) 절감률 및 티켓/월간 비용 절감액 계산
- 실시간 고객 감정 전환율 (Sentiment Shift) 및 CSAT 예측 모델
"""

import difflib
from typing import List, Dict, Any, Optional


def calculate_draft_adoption(draft_text: str, sent_text: str) -> Dict[str, Any]:
    """
    AI가 생성한 초안(draft_text)과 상담원이 실제 전송한 메시지(sent_text)를 비교하여
    초안 채택 여부, 편집 유사도(0~100%), 채택 유형(Full Adopt / Partial Adopt / Rejected)을 산출합니다.
    """
    draft = (draft_text or "").strip()
    sent = (sent_text or "").strip()

    if not draft or not sent:
        return {
            "adoption_type": "None",
            "adoption_label": "미사용",
            "similarity_pct": 0.0,
            "draft_char_len": len(draft),
            "sent_char_len": len(sent),
            "is_adopted": False
        }

    # 유사도 계산 (SequenceMatcher)
    matcher = difflib.SequenceMatcher(None, draft, sent)
    similarity = matcher.ratio()
    similarity_pct = round(similarity * 100.0, 1)

    # 채택 유형 판정
    if similarity >= 0.88:
        adoption_type = "Full Adopt"
        adoption_label = "완전 채택 (100%)"
        is_adopted = True
    elif similarity >= 0.40:
        adoption_type = "Partial Adopt"
        adoption_label = "수정 채택 (부분 보완)"
        is_adopted = True
    else:
        adoption_type = "Rejected"
        adoption_label = "미채택 (수기 재작성)"
        is_adopted = False

    return {
        "adoption_type": adoption_type,
        "adoption_label": adoption_label,
        "similarity_pct": similarity_pct,
        "draft_char_len": len(draft),
        "sent_char_len": len(sent),
        "is_adopted": is_adopted
    }


def calculate_aht_roi(
    actual_aht_seconds: float,
    baseline_aht_seconds: float = 180.0,
    hourly_wage: float = 25000.0
) -> Dict[str, Any]:
    """
    상담원 수기 응대 기준 AHT 대비, AI Copilot 활용 시 실제 처리 시간(AHT)의 절감률과
    티켓당/월간 인건비 절감액(ROI)을 계산합니다.
    - baseline_aht_seconds: 업계 수기 기준 AHT (기본 180초 / 3분)
    - hourly_wage: 상담원 시간당 인건비 (기본 25,000원)
    """
    actual_sec = max(1.0, float(actual_aht_seconds or 0.0))
    base_sec = max(actual_sec, float(baseline_aht_seconds or 180.0))

    saved_seconds = max(0.0, base_sec - actual_sec)
    reduction_rate_pct = round((saved_seconds / base_sec) * 100.0, 1)

    cost_per_second = hourly_wage / 3600.0
    saved_cost_per_ticket = int(saved_seconds * cost_per_second)

    monthly_1000_savings = saved_cost_per_ticket * 1000
    monthly_10000_savings = saved_cost_per_ticket * 10000

    return {
        "actual_aht_seconds": round(actual_sec, 1),
        "baseline_aht_seconds": round(base_sec, 1),
        "saved_seconds": round(saved_seconds, 1),
        "reduction_rate_pct": reduction_rate_pct,
        "saved_cost_per_ticket": saved_cost_per_ticket,
        "monthly_1000_savings": monthly_1000_savings,
        "monthly_10000_savings": monthly_10000_savings,
        "hourly_wage": int(hourly_wage)
    }


def analyze_sentiment_shift(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    대화 이력에서 고객 발화의 감정 점수 변화(초기 불만/불안 -> 최종 안심/만족)를 추적하고,
    예상 고객 만족도(CSAT 1~5점) 및 이탈 위험도(Churn Risk %)를 계산합니다.
    """
    if not isinstance(messages, list) or not messages:
        return {
            "initial_sentiment_score": 50,
            "final_sentiment_score": 50,
            "sentiment_shift_delta": 0,
            "shift_label": "변화 없음",
            "predicted_csat": 3.0,
            "churn_risk_pct": 50
        }

    customer_msgs = [
        msg.get("content", "")
        for msg in messages
        if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"] and msg.get("content")
    ]

    if not customer_msgs:
        return {
            "initial_sentiment_score": 50,
            "final_sentiment_score": 50,
            "sentiment_shift_delta": 0,
            "shift_label": "변화 없음",
            "predicted_csat": 3.0,
            "churn_risk_pct": 50
        }

    positive_keywords = [
        "감사", "고맙", "해결", "만족", "좋아요", "좋네요", "안심", "다행", "친절", "수고",
        "thank", "great", "resolved", "good", "helpful", "appreciate", "clear",
        "ありがとう", "助かり", "安心", "満足", "良かったです", "解決"
    ]
    negative_keywords = [
        "안 돼요", "안돼요", "안 됨", "안됨", "오류", "고장", "불만", "환불", "취소", "지연",
        "답답", "화나", "문제", "error", "fail", "broken", "issue", "problem", "delay",
        "困って", "動かない", "エラー", "解約", "キャンセル", "不便"
    ]

    def _score_text(text: str) -> int:
        t_lower = text.lower()
        pos_hits = sum(1 for kw in positive_keywords if kw in t_lower)
        neg_hits = sum(1 for kw in negative_keywords if kw in t_lower)

        base = 50
        score = base + (pos_hits * 18) - (neg_hits * 20)
        return max(10, min(98, score))

    initial_score = _score_text(customer_msgs[0])
    final_score = _score_text(customer_msgs[-1])
    delta = final_score - initial_score

    if delta > 15:
        shift_label = "대폭 개선 (불만 해소)"
    elif delta > 0:
        shift_label = "개선 (안정 전환)"
    elif delta == 0:
        shift_label = "유지"
    else:
        shift_label = "주의 (추가 관리 필요)"

    # CSAT 예측: 1.0 ~ 5.0
    predicted_csat = round(1.0 + (final_score / 100.0) * 4.0, 1)
    churn_risk_pct = max(2, min(95, 100 - final_score))

    return {
        "initial_sentiment_score": initial_score,
        "final_sentiment_score": final_score,
        "sentiment_shift_delta": delta,
        "shift_label": shift_label,
        "predicted_csat": predicted_csat,
        "churn_risk_pct": churn_risk_pct
    }


def generate_bpo_summary(
    actual_aht_seconds: float,
    draft_adoption_history: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    baseline_aht_seconds: float = 180.0,
    hourly_wage: float = 25000.0
) -> Dict[str, Any]:
    """
    세션 전체의 BPO 종합 성과 리포트를 생성합니다.
    """
    aht_roi = calculate_aht_roi(actual_aht_seconds, baseline_aht_seconds, hourly_wage)
    sentiment = analyze_sentiment_shift(messages)

    # 초안 채택률 통계
    total_drafts = len(draft_adoption_history or [])
    adopted_count = sum(1 for d in (draft_adoption_history or []) if d.get("is_adopted", False))
    full_count = sum(1 for d in (draft_adoption_history or []) if d.get("adoption_type") == "Full Adopt")
    partial_count = sum(1 for d in (draft_adoption_history or []) if d.get("adoption_type") == "Partial Adopt")
    rejected_count = total_drafts - adopted_count

    adoption_rate_pct = round((adopted_count / total_drafts * 100.0), 1) if total_drafts > 0 else 0.0

    return {
        "aht_roi": aht_roi,
        "sentiment": sentiment,
        "draft_stats": {
            "total_drafts": total_drafts,
            "adopted_count": adopted_count,
            "full_adopt_count": full_count,
            "partial_adopt_count": partial_count,
            "rejected_count": rejected_count,
            "adoption_rate_pct": adoption_rate_pct
        }
    }
