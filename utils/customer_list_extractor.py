# ========================================
# utils/customer_list_extractor.py
# 고객 목록 추출 모듈 (정제 및 유효성 검증 포함)
# ========================================

import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

INVALID_NAMES = {
    "고객", "customer", "user", "activity", "none", "null", "unknown", 
    "test", "가작동", "が作動", "理由に当該", "inquiry", "system", "agent",
    "상담사", "상담원", "오퍼레이터", "operator", "메시지", "message",
    "please", "hello", "hotel", "travel", "flight", "booking"
}

def is_valid_customer_name(name: str) -> bool:
    """유효한 고객명인지 검증 (오인 추출 방지)"""
    if not name or not isinstance(name, str):
        return False
    name_clean = name.strip()
    if len(name_clean) < 2 or len(name_clean) > 20:
        return False
    if name_clean.lower() in INVALID_NAMES:
        return False
    # 일본어 조사나 불완전 구문 필터링
    if any(name_clean.startswith(p) for p in ["が", "に", "を", "は", "で", "の"]):
        return False
    if any(name_clean.endswith(p) for p in ["に", "で", "を", "が", "作動", "当該"]):
        return False
    # 영문의 경우 너무 일반적인 단어 제외
    if re.match(r'^[A-Za-z]+$', name_clean) and name_clean.lower() in INVALID_NAMES:
        return False
    return True

def extract_customers_from_data_directories(data_dirs: List[str]) -> List[Dict[str, Any]]:
    """
    데이터 디렉토리에서 고객 정보를 추출하여 고객 목록 생성
    """
    customers_dict = {}
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)
                    
                    customer_info = _extract_customer_info(data, file_path)
                    if customer_info:
                        customer_name = customer_info.get('customer_name', '')
                        if customer_name and is_valid_customer_name(customer_name):
                            if customer_name in customers_dict:
                                customers_dict[customer_name]['consultation_count'] += 1
                                if customer_info.get('last_consultation_date'):
                                    existing_date = customers_dict[customer_name].get('last_consultation_date', '')
                                    if customer_info['last_consultation_date'] > existing_date:
                                        customers_dict[customer_name]['last_consultation_date'] = customer_info['last_consultation_date']
                                        customers_dict[customer_name]['customer_id'] = customer_info.get('customer_id', '')
                                        customers_dict[customer_name]['customer_data'] = customer_info.get('customer_data', {})
                            else:
                                customers_dict[customer_name] = {
                                    'customer_name': customer_name,
                                    'customer_id': customer_info.get('customer_id', ''),
                                    'phone': customer_info.get('phone', ''),
                                    'email': customer_info.get('email', ''),
                                    'consultation_count': 1,
                                    'last_consultation_date': customer_info.get('last_consultation_date', ''),
                                    'customer_data': customer_info.get('customer_data', {}),
                                    'source_file': file_path
                                }
                except Exception:
                    continue
    
    customers_list = list(customers_dict.values())
    customers_list.sort(key=lambda x: x.get('last_consultation_date', ''), reverse=True)
    return customers_list


def _extract_customer_info(data: Any, file_path: str) -> Optional[Dict[str, Any]]:
    """데이터에서 고객 정보 추출"""
    try:
        # 형식 1: basic_info 구조
        if isinstance(data, dict) and 'basic_info' in data:
            basic_info = data.get('basic_info', {})
            customer_data = data.get('data', {})
            customer_name = basic_info.get('customer_name', '')
            customer_id = basic_info.get('customer_id', '')
            phone = basic_info.get('phone') or customer_data.get('phone', '')
            email = basic_info.get('email') or customer_data.get('email', '')
            
            if not customer_name or not is_valid_customer_name(customer_name):
                return None
            
            consultation_history = customer_data.get('consultation_history', [])
            last_date = ''
            if consultation_history:
                latest = consultation_history[-1]
                last_date = latest.get('date', latest.get('timestamp', ''))
            
            return {
                'customer_name': customer_name,
                'customer_id': customer_id,
                'phone': phone,
                'email': email,
                'last_consultation_date': last_date,
                'customer_data': data
            }
        
        # 형식 2: flat customer dict (CustomerDataManager 포맷)
        elif isinstance(data, dict) and ('customer_name' in data or 'name' in data):
            customer_name = data.get('customer_name') or data.get('name', '')
            if not customer_name or not is_valid_customer_name(customer_name):
                return None
            return {
                'customer_name': customer_name,
                'customer_id': data.get('customer_id', ''),
                'phone': data.get('phone', ''),
                'email': data.get('email', ''),
                'last_consultation_date': data.get('last_consultation', data.get('last_login', '')),
                'customer_data': data
            }
        
        # 형식 3: 시뮬레이션 이력 포맷
        elif isinstance(data, dict) and ('initial_query' in data or 'messages' in data):
            customer_name = None
            summary = data.get('summary', {})
            if isinstance(summary, dict):
                candidate = summary.get('customer_name', '')
                if is_valid_customer_name(candidate):
                    customer_name = candidate
            
            # 파일명이 CUST002.json 형태인 경우
            file_name = os.path.basename(file_path)
            customer_id = ''
            if file_name.startswith('CUST'):
                customer_id = file_name.replace('.json', '')
            
            if customer_name and is_valid_customer_name(customer_name):
                return {
                    'customer_name': customer_name,
                    'customer_id': customer_id,
                    'phone': data.get('phone', ''),
                    'email': data.get('email', ''),
                    'last_consultation_date': data.get('timestamp', ''),
                    'customer_data': data
                }
        
        # 형식 4: 리스트 형태
        elif isinstance(data, list) and len(data) > 0:
            return _extract_customer_info(data[0], file_path)
            
        return None
    except Exception:
        return None


def extract_customers_from_histories(histories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """시뮬레이션 이력에서 고객 목록 추출"""
    customers_dict = defaultdict(lambda: {
        'consultation_count': 0,
        'last_consultation_date': '',
        'customer_data': None,
        'phone': '',
        'email': ''
    })
    
    for history in histories:
        customer_name = None
        summary = history.get('summary', {})
        if isinstance(summary, dict):
            c = summary.get('customer_name', '')
            if is_valid_customer_name(c):
                customer_name = c
        
        if not customer_name:
            c = history.get('customer_name', '')
            if is_valid_customer_name(c):
                customer_name = c
        
        if customer_name and is_valid_customer_name(customer_name):
            customers_dict[customer_name]['customer_name'] = customer_name
            customers_dict[customer_name]['consultation_count'] += 1
            customers_dict[customer_name]['phone'] = history.get('customer_phone', '')
            customers_dict[customer_name]['email'] = history.get('customer_email', '')
            timestamp = history.get('timestamp', '')
            if timestamp > customers_dict[customer_name]['last_consultation_date']:
                customers_dict[customer_name]['last_consultation_date'] = timestamp
                customers_dict[customer_name]['customer_data'] = history
    
    customers_list = list(customers_dict.values())
    customers_list.sort(key=lambda x: x.get('last_consultation_date', ''), reverse=True)
    return customers_list
