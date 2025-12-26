"""
상담 데이터 저장 유틸리티
채팅/이메일 및 전화 응대 종료 시 고객 데이터 저장
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from data_manager import (
    load_customers, save_customers,
    load_consultations, save_consultations,
    load_consultation_surveys, save_consultation_surveys,
    load_customer_sentiments, save_customer_sentiments,
    load_customer_evaluations, save_customer_evaluations
)


def find_customer_by_contact(email: Optional[str] = None, phone: Optional[str] = None) -> Optional[Dict]:
    """연락처로 고객 찾기 (동일 고객 확인)"""
    customers = load_customers()
    
    # 리스트 형식인 경우와 딕셔너리 형식인 경우 처리
    if isinstance(customers, dict):
        customers = customers.get('customers', []) if 'customers' in customers else []
    elif not isinstance(customers, list):
        customers = []
    
    for customer in customers:
        # email 우선순위 1
        if email and customer.get('email') == email:
            return customer
        # phone 우선순위 2
        if phone and customer.get('phone') == phone:
            return customer
    
    return None


def create_or_update_customer(customer_data: Dict) -> str:
    """고객 생성 또는 업데이트 (동일 고객 확인)"""
    customers = load_customers()
    
    # 리스트 형식인 경우와 딕셔너리 형식인 경우 처리
    if isinstance(customers, dict):
        customers = customers.get('customers', [])
    elif not isinstance(customers, list):
        customers = []
    
    # 동일 고객 확인
    existing_customer = find_customer_by_contact(
        email=customer_data.get('email'),
        phone=customer_data.get('phone')
    )
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if existing_customer:
        # 업데이트
        customer_id = existing_customer.get('customer_id')
        if not customer_id:
            customer_id = f"CUST{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
            existing_customer['customer_id'] = customer_id
        
        existing_customer.update({
            'customer_name': customer_data.get('customer_name', existing_customer.get('customer_name', '')),
            'phone': customer_data.get('phone', existing_customer.get('phone', '')),
            'email': customer_data.get('email', existing_customer.get('email', '')),
            'last_login_date': now,
            'personality_type': customer_data.get('personality_type', existing_customer.get('personality_type', '일반')),
            'preferred_destination': customer_data.get('preferred_destination', existing_customer.get('preferred_destination', '')),
            'updated_at': now
        })
        
        # customers 리스트에서 해당 고객 업데이트
        for i, c in enumerate(customers):
            if c.get('customer_id') == customer_id:
                customers[i] = existing_customer
                break
    else:
        # 새로 생성
        customer_id = f"CUST{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
        new_customer = {
            'customer_id': customer_id,
            'customer_name': customer_data.get('customer_name', ''),
            'phone': customer_data.get('phone', ''),
            'email': customer_data.get('email', ''),
            'account_created_date': now,
            'last_login_date': now,
            'last_consultation_date': None,
            'personality_type': customer_data.get('personality_type', '일반'),
            'preferred_destination': customer_data.get('preferred_destination', ''),
            'created_at': now,
            'updated_at': now
        }
        customers.append(new_customer)
    
    save_customers(customers)
    return customer_id


def save_consultation_data(
    customer_id: str,
    consultation_type: str,  # 'chat', 'email', 'phone'
    consultation_content: str,
    consultation_summary: str = '',
    operator_id: str = 'OPER001',
    duration_minutes: int = 0,
    status: str = 'completed'
) -> str:
    """
    상담 데이터 저장
    
    Args:
        customer_id: 고객 ID
        consultation_type: 상담 유형 ('chat', 'email', 'phone')
        consultation_content: 상담 내용 (전체 대화/전화 내용)
        consultation_summary: 상담 요약 (선택사항)
        operator_id: 상담원 ID
        duration_minutes: 상담 시간 (분)
        status: 상담 상태 ('completed', 'in_progress', 'cancelled')
    
    Returns:
        consultation_id: 생성된 상담 ID
    """
    consultations = load_consultations()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    consultation_id = f"CONSULT{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
    
    new_consultation = {
        'consultation_id': consultation_id,
        'customer_id': customer_id,
        'consultation_type': consultation_type,
        'consultation_date': now,
        'consultation_content': consultation_content,
        'consultation_summary': consultation_summary,
        'operator_id': operator_id,
        'duration_minutes': duration_minutes,
        'status': status,
        'created_at': now,
        'updated_at': now
    }
    
    consultations.append(new_consultation)
    save_consultations(consultations)
    
    # 고객의 last_consultation_date 업데이트
    customers = load_customers()
    if isinstance(customers, dict):
        customers = customers.get('customers', [])
    elif not isinstance(customers, list):
        customers = []
    
    for customer in customers:
        if customer.get('customer_id') == customer_id:
            customer['last_consultation_date'] = now
            customer['updated_at'] = now
            break
    
    save_customers(customers)
    
    return consultation_id


def save_survey_data(
    consultation_id: str,
    customer_id: str,
    satisfaction_score: float = 0.0,
    response_time_score: float = 0.0,
    problem_resolution_score: float = 0.0,
    overall_rating: float = 0.0,
    survey_comments: str = ''
) -> str:
    """상담 설문 데이터 저장"""
    surveys = load_consultation_surveys()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    survey_id = f"SURVEY{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
    
    new_survey = {
        'survey_id': survey_id,
        'consultation_id': consultation_id,
        'customer_id': customer_id,
        'satisfaction_score': satisfaction_score,
        'response_time_score': response_time_score,
        'problem_resolution_score': problem_resolution_score,
        'overall_rating': overall_rating,
        'survey_comments': survey_comments,
        'survey_date': now,
        'created_at': now
    }
    
    surveys.append(new_survey)
    save_consultation_surveys(surveys)
    return survey_id


def save_sentiment_data(
    customer_id: str,
    consultation_id: str,
    sentiment_type: str = 'neutral',
    sentiment_score: float = 0.0,
    emotion_keywords: List[str] = None
) -> str:
    """고객 감정 분석 데이터 저장"""
    sentiments = load_customer_sentiments()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    sentiment_id = f"SENT{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
    
    new_sentiment = {
        'sentiment_id': sentiment_id,
        'customer_id': customer_id,
        'consultation_id': consultation_id,
        'sentiment_type': sentiment_type,
        'sentiment_score': sentiment_score,
        'emotion_keywords': emotion_keywords or [],
        'analysis_date': now,
        'created_at': now
    }
    
    sentiments.append(new_sentiment)
    save_customer_sentiments(sentiments)
    return sentiment_id


def save_evaluation_data(
    customer_id: str,
    consultation_id: str,
    evaluation_metrics: Dict
) -> str:
    """고객 평가 데이터 저장"""
    evaluations = load_customer_evaluations()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    evaluation_id = f"EVAL{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
    
    new_evaluation = {
        'evaluation_id': evaluation_id,
        'customer_id': customer_id,
        'consultation_id': consultation_id,
        'evaluation_metrics': evaluation_metrics,
        'evaluation_date': now,
        'created_at': now
    }
    
    evaluations.append(new_evaluation)
    save_customer_evaluations(evaluations)
    return evaluation_id


def save_chat_consultation(
    customer_data: Dict,
    chat_messages: List[Dict],
    consultation_summary: str = '',
    duration_minutes: int = 0
) -> str:
    """
    채팅 상담 종료 시 데이터 저장
    
    Args:
        customer_data: 고객 정보 딕셔너리
        chat_messages: 채팅 메시지 리스트
        consultation_summary: 상담 요약
        duration_minutes: 상담 시간 (분)
    
    Returns:
        consultation_id: 생성된 상담 ID
    """
    # 고객 생성 또는 업데이트
    customer_id = create_or_update_customer(customer_data)
    
    # 채팅 내용 문자열로 변환
    consultation_content = "\n".join([
        f"[{msg.get('timestamp', '')}] {msg.get('sender_name', 'Unknown')}: {msg.get('message', '')}"
        for msg in chat_messages
    ])
    
    # 상담 데이터 저장
    consultation_id = save_consultation_data(
        customer_id=customer_id,
        consultation_type='chat',
        consultation_content=consultation_content,
        consultation_summary=consultation_summary,
        duration_minutes=duration_minutes,
        status='completed'
    )
    
    return consultation_id


def save_phone_consultation(
    customer_data: Dict,
    call_transcript: str,
    consultation_summary: str = '',
    duration_minutes: int = 0
) -> str:
    """
    전화 상담 종료 시 데이터 저장
    
    Args:
        customer_data: 고객 정보 딕셔너리
        call_transcript: 전화 전사 내용
        consultation_summary: 상담 요약
        duration_minutes: 상담 시간 (분)
    
    Returns:
        consultation_id: 생성된 상담 ID
    """
    # 고객 생성 또는 업데이트
    customer_id = create_or_update_customer(customer_data)
    
    # 상담 데이터 저장
    consultation_id = save_consultation_data(
        customer_id=customer_id,
        consultation_type='phone',
        consultation_content=call_transcript,
        consultation_summary=consultation_summary,
        duration_minutes=duration_minutes,
        status='completed'
    )
    
    return consultation_id

