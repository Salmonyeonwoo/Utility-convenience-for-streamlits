# ========================================
# 고객 데이터 관리 모듈
# 고객 정보를 JSON/CSV 형식으로 저장 및 관리
# ========================================

import json
import csv
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import hashlib

# 고객 데이터 저장 경로
CUSTOMER_DATA_DIR = "data/customers"
CUSTOMER_DB_FILE = "data/customers_database.json"
CUSTOMER_CSV_FILE = "data/customers_database.csv"

# 디렉토리 생성
os.makedirs(CUSTOMER_DATA_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


class CustomerDataManager:
    """고객 데이터 관리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.data_dir = CUSTOMER_DATA_DIR
        self.db_file = CUSTOMER_DB_FILE
        self.csv_file = CUSTOMER_CSV_FILE
        os.makedirs(self.data_dir, exist_ok=True)
    
    def generate_customer_id(self) -> str:
        """고객 ID 생성"""
        return f"CUST{uuid.uuid4().hex[:8].upper()}"
    
    def generate_identity_hash(self, phone: str, email: str) -> str:
        """동일 고객 여부 판별용 해시 생성"""
        raw_str = f"{phone.strip()}|{email.strip().lower()}"
        return hashlib.md5(raw_str.encode()).hexdigest()
    
    def load_all_customers(self) -> List[Dict]:
        """모든 고객 데이터 로드"""
        try:
            with open(self.db_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('customers', [])
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"고객 데이터 로드 오류: {e}")
            return []
    
    def save_all_customers(self, customers: List[Dict]):
        """모든 고객 데이터 저장"""
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump({'customers': customers}, f, ensure_ascii=False, indent=2)
            
            # CSV로도 저장
            if customers:
                df = pd.DataFrame(customers)
                df.to_csv(self.csv_file, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"고객 데이터 저장 오류: {e}")
    
    def get_customer_by_id(self, customer_id: str) -> Optional[Dict]:
        """고객 ID로 고객 정보 가져오기"""
        customers = self.load_all_customers()
        for customer in customers:
            if customer.get('customer_id') == customer_id:
                return customer
        return None
    
    def find_customer_by_contact(self, phone: str = None, email: str = None) -> Optional[Dict]:
        """연락처나 이메일로 고객 찾기"""
        customers = self.load_all_customers()
        for customer in customers:
            if phone and customer.get('phone') == phone:
                return customer
            if email and customer.get('email') == email:
                return customer
        return None
    
    def create_customer(self, customer_data: Dict) -> str:
        """새 고객 생성"""
        customers = self.load_all_customers()
        
        # 동일 고객 확인
        if customer_data.get('phone') or customer_data.get('email'):
            existing = self.find_customer_by_contact(
                customer_data.get('phone'),
                customer_data.get('email')
            )
            if existing:
                return existing['customer_id']
        
        # 새 고객 생성
        customer_id = self.generate_customer_id()
        identity_hash = self.generate_identity_hash(
            customer_data.get('phone', ''),
            customer_data.get('email', '')
        )
        
        new_customer = {
            'customer_id': customer_id,
            'customer_name': customer_data.get('customer_name', ''),
            'phone': customer_data.get('phone', ''),
            'email': customer_data.get('email', ''),
            'identity_hash': identity_hash,
            'account_created': datetime.now().strftime("%Y-%m-%d"),
            'last_login': datetime.now().strftime("%Y-%m-%d"),
            'last_consultation': datetime.now().strftime("%Y-%m-%d"),
            'consultation_history': [],
            'personality': customer_data.get('personality', '일반'),
            'personality_summary': customer_data.get('personality_summary', ''),
            'preferred_destination': customer_data.get('preferred_destination', ''),
            'travel_budget': customer_data.get('travel_budget', ''),
            'survey_score': customer_data.get('survey_score', 0.0),
            'service_rating': customer_data.get('service_rating', 0.0),
            'evaluation_data': [],
            'sentiment_analysis': []
        }
        
        customers.append(new_customer)
        self.save_all_customers(customers)
        return customer_id
    
    def update_customer(self, customer_id: str, update_data: Dict):
        """고객 정보 업데이트"""
        customers = self.load_all_customers()
        for i, customer in enumerate(customers):
            if customer.get('customer_id') == customer_id:
                customers[i].update(update_data)
                customers[i]['last_consultation'] = datetime.now().strftime("%Y-%m-%d")
                self.save_all_customers(customers)
                return True
        return False
    
    def add_consultation(self, customer_id: str, consultation_data: Dict):
        """상담 이력 추가"""
        customers = self.load_all_customers()
        for i, customer in enumerate(customers):
            if customer.get('customer_id') == customer_id:
                if 'consultation_history' not in customers[i]:
                    customers[i]['consultation_history'] = []
                
                consultation_record = {
                    'consultation_id': f"CONSULT{uuid.uuid4().hex[:6].upper()}",
                    'consultation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'consultation_type': consultation_data.get('type', '채팅'),
                    'consultation_content': consultation_data.get('content', ''),
                    'consultation_summary': consultation_data.get('summary', ''),
                    'survey_score': consultation_data.get('survey_score', 0.0),
                    'service_rating': consultation_data.get('service_rating', 0.0),
                    'sentiment': consultation_data.get('sentiment', 'neutral'),
                    'emotion': consultation_data.get('emotion', 'Normal')
                }
                
                customers[i]['consultation_history'].append(consultation_record)
                customers[i]['last_consultation'] = datetime.now().strftime("%Y-%m-%d")
                
                # 평균 점수 업데이트
                if customers[i]['consultation_history']:
                    scores = [c.get('survey_score', 0) for c in customers[i]['consultation_history'] if c.get('survey_score', 0) > 0]
                    if scores:
                        customers[i]['survey_score'] = sum(scores) / len(scores)
                    
                    ratings = [c.get('service_rating', 0) for c in customers[i]['consultation_history'] if c.get('service_rating', 0) > 0]
                    if ratings:
                        customers[i]['service_rating'] = sum(ratings) / len(ratings)
                
                self.save_all_customers(customers)
                return True
        return False
    
    def add_sentiment_analysis(self, customer_id: str, sentiment_data: Dict):
        """감정 분석 결과 추가"""
        customers = self.load_all_customers()
        for i, customer in enumerate(customers):
            if customer.get('customer_id') == customer_id:
                if 'sentiment_analysis' not in customers[i]:
                    customers[i]['sentiment_analysis'] = []
                
                sentiment_record = {
                    'analysis_id': f"SENT{uuid.uuid4().hex[:6].upper()}",
                    'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'sentiment': sentiment_data.get('sentiment', 'neutral'),
                    'emotion': sentiment_data.get('emotion', 'Normal'),
                    'confidence': sentiment_data.get('confidence', 0.0),
                    'keywords': sentiment_data.get('keywords', [])
                }
                
                customers[i]['sentiment_analysis'].append(sentiment_record)
                self.save_all_customers(customers)
                return True
        return False
    
    def delete_customer(self, customer_id: str) -> bool:
        """고객 삭제"""
        customers = self.load_all_customers()
        customers = [c for c in customers if c.get('customer_id') != customer_id]
        self.save_all_customers(customers)
        return True
    
    def export_to_csv(self, filepath: str = None):
        """CSV로 내보내기"""
        if filepath is None:
            filepath = self.csv_file
        
        customers = self.load_all_customers()
        if customers:
            df = pd.DataFrame(customers)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            return True
        return False
    
    def import_from_csv(self, filepath: str):
        """CSV에서 가져오기"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            customers = df.to_dict('records')
            self.save_all_customers(customers)
            return True
        except Exception as e:
            print(f"CSV 가져오기 오류: {e}")
            return False
    
    # 기존 코드 호환성을 위한 메서드들
    def load_customer_data(self, customer_id: str):
        """기존 코드 호환성: 고객 데이터 불러오기 (이메일 기반)"""
        # customer_id가 이메일인 경우 이메일로 검색
        customer = self.find_customer_by_contact(email=customer_id)
        if customer:
            return {
                "customer_id": customer.get('customer_id'),
                "updated_at": customer.get('last_login', datetime.now().strftime("%Y-%m-%d")),
                "data": {
                    "name": customer.get('customer_name', ''),
                    "email": customer.get('email', ''),
                    "phone": customer.get('phone', ''),
                    "company": "",
                    "notes": customer.get('personality_summary', '')
                }
            }
        return None
    
    def create_sample_data(self, customer_id: str):
        """기존 코드 호환성: 샘플 고객 데이터 생성"""
        # 이미 존재하는지 확인
        existing = self.load_customer_data(customer_id)
        if existing:
            return existing
        
        # 샘플 데이터 생성
        sample_data = {
            'customer_name': customer_id.split('@')[0] if '@' in customer_id else customer_id,
            'email': customer_id if '@' in customer_id else f"{customer_id}@example.com",
            'phone': '',
            'personality': '일반',
            'personality_summary': '샘플 고객 데이터',
            'preferred_destination': '',
            'travel_budget': ''
        }
        new_customer_id = self.create_customer(sample_data)
        return self.load_customer_data(customer_id)
