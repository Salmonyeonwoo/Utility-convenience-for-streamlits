# ========================================
# utils/history_handler_guides.py
# 이력 관리 - 가이드라인 생성 함수들
# ========================================

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from llm_client import get_api_key, run_llm
from config import DATA_DIR

def recommend_guideline_for_customer(new_customer_summary: Dict[str, Any], histories: List[Dict[str, Any]], language: str = "ko") -> Optional[str]:
    """신규 고객의 문의사항과 말투 등을 종합하여 고객 성향 점수를 수치화하고, 저장된 데이터를 바탕으로 최적의 가이드라인을 추천합니다."""
    if not histories or not get_api_key("gemini") and not get_api_key("openai"):
        return None
    
    try:
        similar_customers = []
        new_scores = {
            "sentiment": new_customer_summary.get("customer_sentiment_score", 50),
            "satisfaction": new_customer_summary.get("customer_satisfaction_score", 50),
            "formality": new_customer_summary.get("customer_characteristics", {}).get("formality_score", 50),
            "patience": new_customer_summary.get("customer_characteristics", {}).get("personality_traits", {}).get("patience_level", 50),
            "assertiveness": new_customer_summary.get("customer_characteristics", {}).get("personality_traits", {}).get("assertiveness", 50),
        }
        
        for h in histories:
            if not h.get("summary") or not isinstance(h.get("summary"), dict):
                continue
            
            summary = h["summary"]
            old_scores = {
                "sentiment": summary.get("customer_sentiment_score", 50),
                "satisfaction": summary.get("customer_satisfaction_score", 50),
                "formality": summary.get("customer_characteristics", {}).get("formality_score", 50),
                "patience": summary.get("customer_characteristics", {}).get("personality_traits", {}).get("patience_level", 50),
                "assertiveness": summary.get("customer_characteristics", {}).get("personality_traits", {}).get("assertiveness", 50),
            }
            
            similarity = sum(abs(new_scores[k] - old_scores[k]) for k in new_scores.keys())
            
            if similarity < 100:
                similar_customers.append({
                    "history": h,
                    "similarity": similarity,
                    "scores": old_scores
                })
        
        similar_customers.sort(key=lambda x: x["similarity"])
        
        if similar_customers:
            lang_name = {"ko": "한국어", "en": "English", "ja": "日本語"}.get(language, "한국어")
            
            similar_cases_text = json.dumps([
                {
                    "initial_query": c["history"].get("initial_query", ""),
                    "key_responses": c["history"].get("summary", {}).get("key_responses", []),
                    "scores": c["scores"],
                    "satisfaction": c["history"].get("summary", {}).get("customer_satisfaction_score", 50)
                }
                for c in similar_customers[:5]
            ], ensure_ascii=False, indent=2)
            
            recommendation_prompt = (
                f"당신은 CS 센터 전문가입니다. 신규 고객의 성향 점수를 분석하고, 유사한 과거 고객들의 성공 사례를 바탕으로 최적의 응대 가이드라인을 추천하세요.\n\n"
                f"신규 고객 프로필:\n{json.dumps(new_customer_summary, ensure_ascii=False, indent=2)}\n\n"
                f"유사한 과거 고객 사례 (상위 5개):\n{similar_cases_text}\n\n"
                f"다음 내용을 포함하여 {lang_name}로 가이드라인을 작성하세요:\n"
                f"1. 고객 성향 분석 (점수 기반)\n"
                f"2. 예상되는 고객 반응 패턴\n"
                f"3. 효과적인 응대 전략 (유사 사례 기반)\n"
                f"4. 주의해야 할 사항\n"
                f"5. 권장 응대 톤 및 스타일\n\n"
                f"실용적이고 구체적인 가이드라인을 제공하세요."
            )
            
            recommendation = run_llm(recommendation_prompt)
            return recommendation if recommendation and not recommendation.startswith("❌") else None
        
        return None
        
    except Exception as e:
        print(f"가이드라인 추천 중 오류 발생: {e}")
        return None

def generate_daily_customer_guide(histories: List[Dict[str, Any]], language: str = "ko") -> Optional[str]:
    """일일 고객 가이드 생성 함수"""
    if not histories or not get_api_key("gemini") and not get_api_key("openai"):
        return None
    
    try:
        histories_with_summary = [h for h in histories if h.get("summary") and isinstance(h.get("summary"), dict)]
        
        if not histories_with_summary:
            return None
        
        recent_histories = histories_with_summary[:50]
        
        customer_data_map = {}
        for h in recent_histories:
            customer_id = h.get("id", "")
            customer_type = h.get("customer_type", "")
            summary = h.get("summary", {})
            
            if customer_id not in customer_data_map:
                customer_data_map[customer_id] = {
                    "customer_type": customer_type,
                    "histories": [],
                    "total_interactions": 0
                }
            
            customer_data_map[customer_id]["histories"].append({
                "initial_query": h.get("initial_query", ""),
                "summary": summary,
                "timestamp": h.get("timestamp", ""),
                "language": h.get("language_key", language)
            })
            customer_data_map[customer_id]["total_interactions"] += 1
        
        lang_name = {"ko": "한국어", "en": "English", "ja": "日本語"}.get(language, "한국어")
        
        guide_prompt = (
            f"당신은 CS 센터 교육 전문가입니다. 다음 고객 응대 이력 데이터를 분석하여 종합적인 고객 응대 가이드라인을 작성하세요.\n\n"
            f"분석할 이력 데이터 (고객별 누적 데이터 포함):\n{json.dumps(list(customer_data_map.values())[:20], ensure_ascii=False, indent=2)}\n\n"
            f"다음 내용을 포함하여 가이드라인을 {lang_name}로 작성하세요:\n"
            f"1. 고객 유형별 응대 전략 (일반/까다로운/매우 불만족)\n"
            f"2. 문화권별 응대 가이드 (언어, 문화적 배경 고려)\n"
            f"3. 주요 문의 유형별 해결 방법\n"
            f"4. 고객 감정 점수에 따른 응대 전략\n"
            f"5. 개인정보 처리 가이드\n"
            f"6. 효과적인 소통 스타일 권장사항\n"
            f"7. 동일 고객의 반복 문의에 대한 대응 전략\n"
            f"8. 강성 고객 가이드라인 (까다로운 고객, 매우 불만족 고객)\n\n"
            f"가이드라인을 {lang_name}로 작성하세요. 실제 사례를 바탕으로 구체적이고 실용적인 내용으로 작성해주세요."
        )
        
        guide_content = run_llm(guide_prompt)
        
        if not guide_content or guide_content.startswith("❌"):
            return None
        
        today_str = datetime.now().strftime("%y%m%d")
        formatted_guide = (
            f"고객 응대 가이드라인\n"
            f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"분석 이력 수: {len(recent_histories)}\n"
            f"고객 수: {len(customer_data_map)}\n"
            f"=" * 80 + "\n\n"
            f"{guide_content}\n\n"
            f"=" * 80 + "\n"
            f"이 가이드는 AI 고객 응대 시뮬레이터 데이터를 기반으로 자동 생성되었습니다.\n"
            f"고객 데이터가 추가될 때마다 업데이트됩니다."
        )
        
        return formatted_guide
        
    except Exception as e:
        print(f"고객 가이드 생성 중 오류 발생: {e}")
        return None

def save_daily_customer_guide(guide_content: str, language: str = "ko") -> Optional[str]:
    """일일 고객 가이드를 파일로 저장합니다."""
    try:
        today_str = datetime.now().strftime("%y%m%d")
        guide_filename = f"{today_str}_고객가이드.TXT"
        guide_filepath = os.path.join(DATA_DIR, guide_filename)
        
        if os.path.exists(guide_filepath):
            with open(guide_filepath, "r", encoding="utf-8") as f:
                existing_content = f.read()
            
            updated_content = (
                f"{existing_content}\n\n"
                f"{'=' * 80}\n"
                f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'=' * 80}\n\n"
                f"{guide_content}"
            )
            
            with open(guide_filepath, "w", encoding="utf-8") as f:
                f.write(updated_content)
        else:
            with open(guide_filepath, "w", encoding="utf-8") as f:
                f.write(guide_content)
        
        return guide_filepath
        
    except Exception as e:
        print(f"고객 가이드 저장 중 오류 발생: {e}")
        return None

