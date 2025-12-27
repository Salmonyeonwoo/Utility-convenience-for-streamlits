"""
고객 데이터 관리자 V2
새로운 스키마에 맞춘 고객 데이터 저장/로드 기능
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import uuid


class CustomerDataManagerV2:
    """고객 데이터 관리자 (새 스키마)"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 파일 경로
        self.customers_file = os.path.join(data_dir, "customers.json")
        self.consultations_file = os.path.join(data_dir, "consultations.json")
        self.consultation_surveys_file = os.path.join(data_dir, "consultation_surveys.json")
        self.customer_sentiment_file = os.path.join(data_dir, "customer_sentiment.json")
        self.customer_evaluation_file = os.path.join(data_dir, "customer_evaluation_data.json")
    
    # ========== Customers 테이블 ==========
    
    def load_customers(self) -> List[Dict]:
        """고객 목록 로드"""
        try:
            with open(self.customers_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('customers', [])
        except FileNotFoundError:
            return []
    
    def save_customers(self, customers: List[Dict]):
        """고객 목록 저장"""
        data = {"customers": customers}
        with open(self.customers_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def find_customer_by_contact(self, email: Optional[str] = None, phone: Optional[str] = None) -> Optional[Dict]:
        """연락처로 고객 찾기 (동일 고객 확인)"""
        customers = self.load_customers()
        
        for customer in customers:
            # email 우선순위 1
            if email and customer.get('email') == email:
                return customer
            # phone 우선순위 2
            if phone and customer.get('phone') == phone:
                return customer
        
        return None
    
    def get_customer_by_id(self, customer_id: str) -> Optional[Dict]:
        """고객 ID로 고객 조회"""
        customers = self.load_customers()
        return next((c for c in customers if c.get('customer_id') == customer_id), None)
    
    def create_or_update_customer(self, customer_data: Dict) -> str:
        """고객 생성 또는 업데이트 (동일 고객 확인)"""
        customers = self.load_customers()
        
        # 동일 고객 확인
        existing_customer = self.find_customer_by_contact(
            email=customer_data.get('email'),
            phone=customer_data.get('phone')
        )
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if existing_customer:
            # 업데이트
            existing_customer.update({
                'customer_name': customer_data.get('customer_name', existing_customer.get('customer_name')),
                'phone': customer_data.get('phone', existing_customer.get('phone')),
                'email': customer_data.get('email', existing_customer.get('email')),
                'last_login_date': now,
                'personality_type': customer_data.get('personality_type', existing_customer.get('personality_type', '일반')),
                'preferred_destination': customer_data.get('preferred_destination', existing_customer.get('preferred_destination')),
                'updated_at': now
            })
            customer_id = existing_customer['customer_id']
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
        
        self.save_customers(customers)
        return customer_id
    
    # ========== Consultations 테이블 ==========
    
    def load_consultations(self) -> List[Dict]:
        """상담 이력 로드"""
        try:
            with open(self.consultations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('consultations', [])
        except FileNotFoundError:
            return []
    
    def save_consultations(self, consultations: List[Dict]):
        """상담 이력 저장"""
        data = {"consultations": consultations}
        with open(self.consultations_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create_consultation(self, consultation_data: Dict) -> str:
        """상담 이력 생성"""
        consultations = self.load_consultations()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        consultation_id = f"CONSULT{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
        
        new_consultation = {
            'consultation_id': consultation_id,
            'customer_id': consultation_data.get('customer_id'),
            'consultation_type': consultation_data.get('consultation_type', 'chat'),  # chat/email/phone
            'consultation_date': consultation_data.get('consultation_date', now),
            'consultation_content': consultation_data.get('consultation_content', ''),
            'consultation_summary': consultation_data.get('consultation_summary', ''),
            'operator_id': consultation_data.get('operator_id', 'OPER001'),
            'duration_minutes': consultation_data.get('duration_minutes', 0),
            'status': consultation_data.get('status', 'completed'),
            'created_at': now,
            'updated_at': now
        }
        
        consultations.append(new_consultation)
        self.save_consultations(consultations)
        
        # 고객의 last_consultation_date 업데이트
        customer_id = consultation_data.get('customer_id')
        if customer_id:
            customers = self.load_customers()
            for customer in customers:
                if customer.get('customer_id') == customer_id:
                    customer['last_consultation_date'] = now
                    customer['updated_at'] = now
                    break
            self.save_customers(customers)
        
        return consultation_id
    
    # ========== Consultation Surveys 테이블 ==========
    
    def load_consultation_surveys(self) -> List[Dict]:
        """상담 설문 로드"""
        try:
            with open(self.consultation_surveys_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('surveys', [])
        except FileNotFoundError:
            return []
    
    def save_consultation_surveys(self, surveys: List[Dict]):
        """상담 설문 저장"""
        data = {"surveys": surveys}
        with open(self.consultation_surveys_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create_survey(self, survey_data: Dict) -> str:
        """상담 설문 생성"""
        surveys = self.load_consultation_surveys()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        survey_id = f"SURVEY{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
        
        new_survey = {
            'survey_id': survey_id,
            'consultation_id': survey_data.get('consultation_id'),
            'customer_id': survey_data.get('customer_id'),
            'satisfaction_score': survey_data.get('satisfaction_score', 0.0),
            'response_time_score': survey_data.get('response_time_score', 0.0),
            'problem_resolution_score': survey_data.get('problem_resolution_score', 0.0),
            'overall_rating': survey_data.get('overall_rating', 0.0),
            'survey_comments': survey_data.get('survey_comments', ''),
            'survey_date': survey_data.get('survey_date', now),
            'created_at': now
        }
        
        surveys.append(new_survey)
        self.save_consultation_surveys(surveys)
        return survey_id
    
    # ========== Customer Sentiment 테이블 ==========
    
    def load_customer_sentiments(self) -> List[Dict]:
        """고객 감정 분석 로드"""
        try:
            with open(self.customer_sentiment_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('sentiments', [])
        except FileNotFoundError:
            return []
    
    def save_customer_sentiments(self, sentiments: List[Dict]):
        """고객 감정 분석 저장"""
        data = {"sentiments": sentiments}
        with open(self.customer_sentiment_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create_sentiment(self, sentiment_data: Dict) -> str:
        """고객 감정 분석 생성"""
        sentiments = self.load_customer_sentiments()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sentiment_id = f"SENT{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
        
        new_sentiment = {
            'sentiment_id': sentiment_id,
            'customer_id': sentiment_data.get('customer_id'),
            'consultation_id': sentiment_data.get('consultation_id'),
            'sentiment_type': sentiment_data.get('sentiment_type', 'neutral'),
            'sentiment_score': sentiment_data.get('sentiment_score', 0.0),
            'emotion_keywords': sentiment_data.get('emotion_keywords', []),
            'analysis_date': sentiment_data.get('analysis_date', now),
            'created_at': now
        }
        
        sentiments.append(new_sentiment)
        self.save_customer_sentiments(sentiments)
        return sentiment_id
    
    # ========== Customer Evaluation Data 테이블 ==========
    
    def load_customer_evaluations(self) -> List[Dict]:
        """고객 평가 데이터 로드"""
        try:
            with open(self.customer_evaluation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('evaluations', [])
        except FileNotFoundError:
            return []
    
    def save_customer_evaluations(self, evaluations: List[Dict]):
        """고객 평가 데이터 저장"""
        data = {"evaluations": evaluations}
        with open(self.customer_evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create_evaluation(self, evaluation_data: Dict) -> str:
        """고객 평가 데이터 생성"""
        evaluations = self.load_customer_evaluations()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        evaluation_id = f"EVAL{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
        
        new_evaluation = {
            'evaluation_id': evaluation_id,
            'customer_id': evaluation_data.get('customer_id'),
            'consultation_id': evaluation_data.get('consultation_id'),
            'evaluation_metrics': evaluation_data.get('evaluation_metrics', {}),
            'evaluation_date': evaluation_data.get('evaluation_date', now),
            'created_at': now
        }
        
        evaluations.append(new_evaluation)
        self.save_customer_evaluations(evaluations)
        return evaluation_id


