# coding: utf-8
# ========================================
# utils/rag_knowledge_engine.py
# Enterprise RAG Knowledge Search & Citation Engine
# ========================================

import os
import json
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional

_KNOWLEDGE_CACHE = None

def get_base_dir() -> str:
    """프로젝트 루트 디렉토리 경로 반환"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_knowledge_database() -> Dict[str, Any]:
    """사내 FAQ 및 규정 데이터베이스 로드 및 캐싱"""
    global _KNOWLEDGE_CACHE
    if _KNOWLEDGE_CACHE is not None:
        return _KNOWLEDGE_CACHE

    base_dir = get_base_dir()
    candidate_paths = [
        os.path.join(base_dir, "local_db", "json", "faq_database.json"),
        os.path.join(base_dir, "local_db", "faq_database.json"),
        os.path.join(base_dir, "data", "faq_database.json"),
    ]

    loaded_data = {"companies": {}}
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if "companies" in data:
                            loaded_data["companies"].update(data["companies"])
                        else:
                            loaded_data["companies"].update(data)
                        break
            except Exception as e:
                print(f"[RAG Engine] Failed to load {path}: {e}")

    _KNOWLEDGE_CACHE = loaded_data
    return _KNOWLEDGE_CACHE

def normalize_stem(w: str) -> str:
    """한국어 조사 및 기본 어미 정규화"""
    suffixes = ["에서", "으로", "하고", "에는", "에", "로", "을", "를", "이", "가", "은", "는", "도", "의", "된", "할", "한"]
    for s in suffixes:
        if w.endswith(s) and len(w) > len(s) + 1:
            return w[:-len(s)]
    return w

def extract_keywords(text: str) -> List[str]:
    """텍스트에서 핵심 검색 키워드 추출 (불용어 제거 및 어근 정규화)"""
    if not text:
        return []
    
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = clean.split()
    
    stopwords = {
        "이", "그", "저", "것", "수", "등", "및", "에", "를", "을", "의", "가", "은", "는", 
        "로", "으로", "에서", "와", "과", "도", "만", "요", "안", "못", "좀", "해", "주세요",
        "합니다", "합니다만", "되나요", "어떻게", "있나요", "없나요", "때문에", "대한",
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or"
    }
    
    raw_keywords = [w for w in words if len(w) >= 2 and w not in stopwords]
    normalized = [normalize_stem(w) for w in raw_keywords]
    return list(set(raw_keywords + normalized))

def retrieve_grounding_knowledge(query: str, lang: str = "ko", top_k: int = 1) -> Optional[Dict[str, Any]]:
    """
    고객 문의 내용을 분석하여 사내 지식 베이스에서 가장 관련성 높은 규정/FAQ를 검색하고
    신뢰도 점수 및 인용(Citation) 객체를 생성합니다.
    """
    if not query or not query.strip():
        return None

    db = load_knowledge_database()
    companies = db.get("companies", {})
    if not companies:
        return None

    query_lower = query.lower()
    query_keywords = extract_keywords(query)
    
    best_match = None
    best_score = 0.0

    lang_suffix = f"_{lang}" if lang in ["ko", "en", "ja"] else "_ko"
    q_field = f"question{lang_suffix}"
    a_field = f"answer{lang_suffix}"

    # 중요 도메인 토큰 정의
    key_entities = ["esim", "이심", "유심", "환불", "취소", "예약", "파리", "프랑스", "태국", "호텔", "투어", "배송", "s25", "갤럭시"]

    for comp_name, comp_info in companies.items():
        if not isinstance(comp_info, dict):
            continue

        comp_name_lower = comp_name.lower()
        comp_boost = 1.3 if (comp_name_lower in query_lower or comp_name in query) else 1.0

        faqs = comp_info.get("faqs", [])
        if isinstance(faqs, list):
            for idx, faq in enumerate(faqs):
                if not isinstance(faq, dict):
                    continue

                q_text = faq.get(q_field, faq.get("question_ko", faq.get("question_en", "")))
                a_text = faq.get(a_field, faq.get("answer_ko", faq.get("answer_en", "")))
                if not q_text and not a_text:
                    continue

                combined_target = f"{q_text} {a_text}".lower()
                target_keywords = extract_keywords(combined_target)
                if not target_keywords:
                    continue

                # 1. 키워드 일치도
                matched_kw = [k for k in query_keywords if any(k in tk or tk in k for tk in target_keywords)]
                kw_ratio = len(matched_kw) / max(1, len(query_keywords))

                # 2. 질문과의 직접 유사도 (SequenceMatcher)
                seq_q = SequenceMatcher(None, query_lower, q_text.lower()).ratio()
                seq_a = SequenceMatcher(None, query_lower, a_text[:120].lower()).ratio()
                seq_max = max(seq_q, seq_a)

                # 3. 핵심 엔티티 일치 검증
                entity_hits = [e for e in key_entities if e in query_lower and (e in q_text.lower() or e in a_text.lower())]
                
                # 핵심 엔티티가 질문(Q)에 직접 포함된 경우 높은 신뢰도 부여
                q_entity_hits = [e for e in key_entities if e in query_lower and e in q_text.lower()]
                
                score_weight = 0.0
                if q_entity_hits:
                    # e.g., 'esim' in query and 'esim' in FAQ title -> Base 80%
                    score_weight = 0.80 + (0.10 * len(q_entity_hits)) + (0.08 * kw_ratio)
                elif entity_hits:
                    score_weight = 0.60 + (0.15 * kw_ratio) + (0.15 * seq_max)
                else:
                    score_weight = (kw_ratio * 0.6 + seq_max * 0.4)

                final_score = min(98.5, round(score_weight * comp_boost * 100, 1))

                if final_score > best_score:
                    best_score = final_score
                    
                    clause_name = q_text
                    if len(clause_name) > 35:
                        clause_name = clause_name[:32] + "..."

                    excerpt_clean = a_text.replace("\n", " ").strip()
                    if len(excerpt_clean) > 160:
                        excerpt_clean = excerpt_clean[:157] + "..."

                    best_match = {
                        "company": comp_name,
                        "source_doc": f"사내 FAQ 및 서비스 가이드 ({comp_name} 부문)",
                        "clause": clause_name,
                        "confidence_score": final_score,
                        "confidence_level": "매우 높음 (Verified)" if final_score >= 80.0 else ("높음 (High)" if final_score >= 55.0 else "보통 (Medium)"),
                        "matched_q": q_text,
                        "excerpt": excerpt_clean,
                        "full_answer": a_text,
                        "grounding_context": f"[사내 공식 규정 / FAQ - {comp_name}]\n질문: {q_text}\n공식 지침: {a_text}"
                    }

    if best_match and best_match["confidence_score"] >= 35.0:
        return best_match
    return None

def format_citation_markdown(citation: Dict[str, Any], lang: str = "ko") -> str:
    """UI 표시용 정규화된 출처 인용 마크다운 카드 생성"""
    if not citation:
        return ""

    score = citation.get("confidence_score", 0.0)
    level = citation.get("confidence_level", "보통")
    source = citation.get("source_doc", "사내 규정 DB")
    clause = citation.get("clause", "서비스 표준 가이드")
    excerpt = citation.get("excerpt", "")

    if lang == "en":
        return f"""
> 🔍 **RAG Knowledge Grounding (Confidence: {score:.1f}% - {level})**
> - **Source Policy**: `{source}` > `{clause}`
> - **Verified Excerpt**: "{excerpt}"
"""
    elif lang == "ja":
        return f"""
> 🔍 **RAGナレッジ根拠検証 (信頼度: {score:.1f}% - {level})**
> - **参照規定**: `{source}` > `{clause}`
> - **公式根拠**: 「{excerpt}」
"""
    else:
        return f"""
> 🔍 **RAG 지식 베이스 출처 검증 (신뢰도: {score:.1f}% - {level})**
> - **참조 규정**: `{source}` > `{clause}`
> - **공식 근거**: "{excerpt}"
"""
