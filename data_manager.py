import json
from typing import Dict, List


def load_customers():
    """고객 데이터 로드"""
    try:
        with open('data/customers.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def load_chats():
    """채팅 데이터 로드"""
    try:
        with open('data/chats.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def load_dashboard_stats():
    """대시보드 통계 데이터 로드"""
    try:
        with open('data/dashboard_stats.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "today_cases": 0,
            "assigned_customers": 0,
            "goal_achievements": 0,
            "daily_goal": 10,
            "completion_rate": 0.0
        }


def save_chats(chats_data):
    """채팅 데이터 저장"""
    with open('data/chats.json', 'w', encoding='utf-8') as f:
        json.dump(chats_data, f, ensure_ascii=False, indent=2)


def save_customers(customers_data):
    """고객 데이터 저장"""
    with open('data/customers.json', 'w', encoding='utf-8') as f:
        json.dump(customers_data, f, ensure_ascii=False, indent=2)


def save_dashboard_stats(stats_data):
    """대시보드 통계 저장"""
    with open('data/dashboard_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)


def load_operators():
    """상담원 데이터 로드"""
    try:
        with open('data/operators.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def load_calls():
    """통화 기록 로드"""
    try:
        with open('data/calls.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_calls(calls_data):
    """통화 기록 저장"""
    with open('data/calls.json', 'w', encoding='utf-8') as f:
        json.dump(calls_data, f, ensure_ascii=False, indent=2)


def load_auto_responses():
    """자동응답 템플릿 로드"""
    try:
        with open('data/auto_responses.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"templates": []}


def load_rag_analysis():
    """RAG 분석 결과 로드"""
    try:
        with open('data/rag_analysis.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_rag_analysis(rag_data):
    """RAG 분석 결과 저장"""
    with open('data/rag_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(rag_data, f, ensure_ascii=False, indent=2)


def load_call_conversations():
    """전화 대화 내역 로드"""
    try:
        with open('data/call_conversations.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_call_conversations(call_conv_data):
    """전화 대화 내역 저장"""
    with open('data/call_conversations.json', 'w', encoding='utf-8') as f:
        json.dump(call_conv_data, f, ensure_ascii=False, indent=2)


def load_company_info():
    """회사 정보 로드"""
    try:
        with open('data/company_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"companies": []}


def search_company(query: str, lang: str = "ko") -> List[dict]:
    """회사 및 FAQ 데이터베이스 검색 (다국어 및 RAG 지원, 스마트 랭킹 및 중복 제거)"""
    import os
    import json
    from faq.company_visual_engine import (
        get_company_visual_and_details,
        classify_company_category,
        COMPANY_ALIAS_MAP,
        CANONICAL_COMPANY_NAMES
    )

    if lang not in ["ko", "en", "ja"]:
        lang = "ko"

    all_companies = []
    seen_names = set()

    # 1. data/company_info.json 로드
    company_data = load_company_info()
    for c in company_data.get('companies', []):
        name = c.get('company_name', '')
        if name and name.lower() not in seen_names:
            seen_names.add(name.lower())
            enriched = get_company_visual_and_details(name, c, lang=lang)
            all_companies.append(enriched)

    # 2. local_db/json/faq_database.json 및 local_db/faq_database.json 로드
    faq_paths = [
        'local_db/json/faq_database.json',
        'local_db/faq_database.json'
    ]
    for fp in faq_paths:
        if os.path.exists(fp):
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                    faq_db = json.load(f)
                    comp_dict = faq_db.get('companies', {})
                    for cname, cdata in comp_dict.items():
                        if not isinstance(cdata, dict) or cname.lower() in seen_names:
                            continue
                        seen_names.add(cname.lower())
                        enriched = get_company_visual_and_details(cname, cdata, lang=lang)
                        all_companies.append(enriched)
            except Exception as e:
                print(f"Error loading {fp}: {e}")

    if not query or not query.strip():
        return all_companies

    query_lower = query.strip().lower()
    query_canon = COMPANY_ALIAS_MAP.get(query_lower)
    q_cat = classify_company_category(query_lower)

    scored_results = []
    for comp in all_companies:
        cname = str(comp.get('company_name', '')).lower()
        c_canon = COMPANY_ALIAS_MAP.get(cname) or COMPANY_ALIAS_MAP.get(comp.get('company_id', ''))
        ind = str(comp.get('industry', '')).lower()
        desc = str(comp.get('description', '')).lower()
        comp_cat = comp.get('category', classify_company_category(cname))
        
        services_text = []
        for s in comp.get('services', []):
            if isinstance(s, str):
                services_text.append(s.lower())
            elif isinstance(s, dict):
                services_text.append(str(s.get('name', s.get('title', ''))).lower())

        for p in comp.get('popular_products', []):
            if isinstance(p, str):
                services_text.append(p.lower())
            elif isinstance(p, dict):
                services_text.append(str(p.get('name', p.get('title', ''))).lower())
        
        score = 0
        # 1. 정규화 ID 일치 (최우선)
        if query_canon and c_canon and query_canon == c_canon:
            score += 100
        # 2. 회사명 직접 매칭
        elif query_lower in cname or cname in query_lower:
            score += 80
        # 3. 제품 / 업종 / 본문 키워드 매칭
        elif any(query_lower in st for st in services_text) or query_lower in ind or query_lower in desc:
            score += 50
        # 4. 동일 카테고리 매칭
        elif q_cat != "general" and comp_cat == q_cat:
            score += 20

        if score > 0:
            scored_results.append((score, comp))

    # 점수 높은 순으로 정렬
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # 중복 제거 (동일 정규화 ID 또는 동일 카테고리의 유사 기업 필터링)
    deduped_results = []
    seen_canon = set()
    for score, comp in scored_results:
        cname = str(comp.get('company_name', '')).lower()
        c_canon = COMPANY_ALIAS_MAP.get(cname) or comp.get('category')
        if c_canon not in seen_canon:
            seen_canon.add(c_canon)
            deduped_results.append(comp)

    # 일치하는 회사가 없을 경우, 사용자가 검색한 회사명으로 실시간 프로필 생성
    if not deduped_results and query and query.strip():
        generated_comp = get_company_visual_and_details(query.strip(), lang=lang)
        deduped_results.append(generated_comp)
            
    return deduped_results

# ========== 고객 데이터 관리 V2 함수들 ==========

def load_consultations():
    """상담 이력 로드"""
    try:
        with open('data/consultations.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('consultations', [])
    except FileNotFoundError:
        return []


def save_consultations(consultations_data):
    """상담 이력 저장"""
    import os
    os.makedirs('data', exist_ok=True)
    data = {"consultations": consultations_data}
    with open('data/consultations.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_consultation_surveys():
    """상담 설문 로드"""
    try:
        with open('data/consultation_surveys.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('surveys', [])
    except FileNotFoundError:
        return []


def save_consultation_surveys(surveys_data):
    """상담 설문 저장"""
    import os
    os.makedirs('data', exist_ok=True)
    data = {"surveys": surveys_data}
    with open('data/consultation_surveys.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_customer_sentiments():
    """고객 감정 분석 로드"""
    try:
        with open('data/customer_sentiment.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('sentiments', [])
    except FileNotFoundError:
        return []


def save_customer_sentiments(sentiments_data):
    """고객 감정 분석 저장"""
    import os
    os.makedirs('data', exist_ok=True)
    data = {"sentiments": sentiments_data}
    with open('data/customer_sentiment.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_customer_evaluations():
    """고객 평가 데이터 로드"""
    try:
        with open('data/customer_evaluation_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('evaluations', [])
    except FileNotFoundError:
        return []


def save_customer_evaluations(evaluations_data):
    """고객 평가 데이터 저장"""
    import os
    os.makedirs('data', exist_ok=True)
    data = {"evaluations": evaluations_data}
    with open('data/customer_evaluation_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)