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


def search_company(query: str) -> List[dict]:
    """회사 정보 검색"""
    company_data = load_company_info()
    companies = company_data.get('companies', [])
    
    if not query:
        return companies
    
    query_lower = query.lower()
    results = []
    for company in companies:
        if (query_lower in company.get('company_name', '').lower() or
            query_lower in company.get('industry', '').lower() or
            query_lower in company.get('description', '').lower() or
            any(query_lower in service.lower() for service in company.get('services', []))):
            results.append(company)
    return results


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