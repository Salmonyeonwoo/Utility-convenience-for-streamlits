# ========================================
# utils/customer_list_extractor.py
# 고객 목록 추출 모듈
# ========================================

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

def extract_customers_from_data_directories(data_dirs: List[str]) -> List[Dict[str, Any]]:
    """
    데이터 디렉토리에서 고객 정보를 추출하여 고객 목록 생성
    
    Args:
        data_dirs: 스캔할 데이터 디렉토리 경로 리스트
    
    Returns:
        고객 정보 리스트 (이름, 상담 횟수 등)
    """
    customers_dict = {}  # customer_name을 키로 하는 딕셔너리
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
        
        # 디렉토리 내 모든 JSON 파일 스캔
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 고객 정보 추출
                    customer_info = _extract_customer_info(data, file_path)
                    if customer_info:
                        customer_name = customer_info.get('customer_name', '')
                        if customer_name:
                            # 이미 존재하는 고객이면 상담 횟수 증가
                            if customer_name in customers_dict:
                                customers_dict[customer_name]['consultation_count'] += 1
                                # 최근 상담 정보 업데이트
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
                                    'consultation_count': 1,
                                    'last_consultation_date': customer_info.get('last_consultation_date', ''),
                                    'customer_data': customer_info.get('customer_data', {}),
                                    'source_file': file_path
                                }
                except Exception as e:
                    # 파일 읽기 오류는 무시
                    continue
    
    # 리스트로 변환하고 정렬 (최근 상담일 기준)
    customers_list = list(customers_dict.values())
    customers_list.sort(key=lambda x: x.get('last_consultation_date', ''), reverse=True)
    
    return customers_list


def _extract_customer_info(data: Dict[str, Any], file_path: str) -> Optional[Dict[str, Any]]:
    """
    데이터에서 고객 정보 추출
    
    Args:
        data: JSON 데이터
        file_path: 파일 경로
    
    Returns:
        고객 정보 딕셔너리 또는 None
    """
    try:
        # 형식 1: basic_info가 있는 경우 (고객 데이터 형식)
        if isinstance(data, dict) and 'basic_info' in data:
            basic_info = data.get('basic_info', {})
            customer_data = data.get('data', {})
            
            customer_name = basic_info.get('customer_name', '')
            customer_id = basic_info.get('customer_id', '')
            
            if not customer_name:
                return None
            
            # 상담 이력에서 최근 상담일 추출
            consultation_history = customer_data.get('consultation_history', [])
            last_consultation_date = ''
            if consultation_history:
                latest = consultation_history[-1]
                last_consultation_date = latest.get('date', latest.get('timestamp', ''))
            
            return {
                'customer_name': customer_name,
                'customer_id': customer_id,
                'last_consultation_date': last_consultation_date,
                'customer_data': data
            }
        
        # 형식 2: 시뮬레이션 이력 형식
        elif isinstance(data, dict) and ('initial_query' in data or 'messages' in data):
            # customer_name을 찾기
            customer_name = None
            
            # summary에서 고객 정보 추출 시도
            summary = data.get('summary', {})
            if isinstance(summary, dict):
                customer_name = summary.get('customer_name', '')
            
            # messages에서 고객 이름 추출 시도
            if not customer_name:
                messages = data.get('messages', [])
                for msg in messages:
                    if msg.get('role') == 'customer':
                        content = msg.get('content', '')
                        # 간단한 이름 추출 시도 (예: "김민수님의 문의" -> "김민수")
                        import re
                        # "XXX님" 패턴 찾기
                        match = re.search(r'([가-힣]{2,4})님', content)
                        if match:
                            customer_name = match.group(1)
                            break
            
            # 파일명에서 고객 ID 추출 시도 (CUST-xxxxx.json)
            file_name = os.path.basename(file_path)
            customer_id = ''
            if file_name.startswith('CUST-'):
                customer_id = file_name.replace('.json', '').replace('CUST-', '')
            
            # customer_name이 없으면 파일명에서 추출 시도
            if not customer_name and customer_id:
                # customer_id를 기반으로 고객 이름 생성 (임시)
                customer_name = f"고객-{customer_id[:8]}"
            
            if customer_name:
                return {
                    'customer_name': customer_name,
                    'customer_id': customer_id,
                    'last_consultation_date': data.get('timestamp', ''),
                    'customer_data': data
                }
        
        # 형식 3: 리스트인 경우 (여러 이력)
        elif isinstance(data, list) and len(data) > 0:
            # 첫 번째 항목에서 고객 정보 추출
            first_item = data[0]
            return _extract_customer_info(first_item, file_path)
        
        # 형식 4: 직접 고객 정보가 있는 경우
        elif isinstance(data, dict):
            customer_name = data.get('customer_name') or data.get('name', '')
            if customer_name:
                return {
                    'customer_name': customer_name,
                    'customer_id': data.get('customer_id', ''),
                    'last_consultation_date': data.get('timestamp', data.get('date', '')),
                    'customer_data': data
                }
        
        return None
        
    except Exception:
        return None


def extract_customers_from_histories(histories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    시뮬레이션 이력에서 고객 목록 추출
    
    Args:
        histories: 시뮬레이션 이력 리스트
    
    Returns:
        고객 정보 리스트
    """
    customers_dict = defaultdict(lambda: {
        'consultation_count': 0,
        'last_consultation_date': '',
        'customer_data': None
    })
    
    for history in histories:
        # 고객 이름 추출
        customer_name = None
        
        # summary에서 추출
        summary = history.get('summary', {})
        if isinstance(summary, dict):
            customer_name = summary.get('customer_name', '')
        
        # initial_query에서 추출 시도
        if not customer_name:
            initial_query = history.get('initial_query', '')
            # 간단한 패턴 매칭 (실제로는 더 정교한 추출 필요)
            pass
        
        # messages에서 추출 시도
        if not customer_name:
            messages = history.get('messages', [])
            for msg in messages:
                if msg.get('role') == 'customer':
                    # 메시지에서 이름 추출 시도
                    pass
        
        if customer_name:
            customers_dict[customer_name]['customer_name'] = customer_name
            customers_dict[customer_name]['consultation_count'] += 1
            
            # 최근 상담일 업데이트
            timestamp = history.get('timestamp', '')
            if timestamp > customers_dict[customer_name]['last_consultation_date']:
                customers_dict[customer_name]['last_consultation_date'] = timestamp
                customers_dict[customer_name]['customer_data'] = history
    
    # 리스트로 변환하고 정렬
    customers_list = list(customers_dict.values())
    customers_list.sort(key=lambda x: x.get('last_consultation_date', ''), reverse=True)
    
    return customers_list

