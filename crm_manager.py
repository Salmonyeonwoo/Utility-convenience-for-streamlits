"""
CRM 데이터베이스 관리자 모듈
고객 및 티켓 데이터의 저장, 조회, 업데이트를 담당
"""
import json
import os
import uuid
import hashlib
from datetime import datetime


class TicketCRMManager:
    """고객 상담 티켓 및 KPI 관리 클래스"""
    
    def __init__(self, db_path="data/crm_db.json"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """데이터베이스 초기화 (로컬 모드)"""
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(self.db_path):
            initial_data = {"customers": {}, "tickets": []}
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f)

    def _load_data(self):
        """데이터베이스에서 데이터 로드"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"customers": {}, "tickets": []}

    def _save_data(self, data):
        """데이터베이스에 데이터 저장"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def generate_identity_hash(self, phone, email):
        """동일 고객 식별을 위한 해시 (연락처+이메일 조합)"""
        phone_str = phone.strip() if phone else ""
        email_str = email.strip().lower() if email else ""
        raw = f"{phone_str}|{email_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def save_ticket(self, customer_info, ticket_info):
        """티켓 저장 및 고객 마스터 업데이트 (KPI 데이터 누적)"""
        db = self._load_data()
        
        # 1. 고객 식별 및 생성/업데이트
        identity_hash = self.generate_identity_hash(customer_info['phone'], customer_info['email'])
        
        cust_id = None
        for cid, cinfo in db['customers'].items():
            if cinfo.get('identity_hash') == identity_hash:
                cust_id = cid
                break
        
        if not cust_id:
            cust_id = f"CUST-{uuid.uuid4().hex[:6].upper()}"
            db['customers'][cust_id] = {
                "name": customer_info['name'],
                "phone": customer_info['phone'],
                "email": customer_info['email'],
                "identity_hash": identity_hash,
                "trait": customer_info['trait'],
                "created_at": datetime.now().isoformat(),
                "total_solved": 0,
                "csat_avg": 0.0
            }

        # 2. 티켓 추가 (상담 유형 포함)
        ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
        new_ticket = {
            "ticket_id": ticket_id,
            "cust_id": cust_id,
            "date": datetime.now().isoformat(),
            "consult_type": ticket_info['consult_type'],
            "status": ticket_info['status'],  # 'Solved' or 'Pending'
            "content": ticket_info['content'],
            "summary": ticket_info['summary'],
            "analysis": ticket_info['analysis'],  # sentiment, score
        }
        db['tickets'].append(new_ticket)

        # 3. Solved 카운팅 및 만족도 업데이트
        if ticket_info['status'] == "Solved":
            db['customers'][cust_id]['total_solved'] += 1
        
        # 고객별 평균 CSAT 계산
        cust_tickets = [t for t in db['tickets'] if t['cust_id'] == cust_id]
        if cust_tickets:
            scores = [t['analysis']['score'] for t in cust_tickets]
            db['customers'][cust_id]['csat_avg'] = round(sum(scores) / len(scores), 2)
        
        db['customers'][cust_id]['last_consult_date'] = datetime.now().isoformat()
        
        self._save_data(db)
        return ticket_id

    def get_scanned_files_tracking_path(self):
        """스캔된 파일 추적 파일 경로 반환"""
        return "data/scanned_files.json"

    def load_scanned_files(self):
        """스캔된 파일 목록 로드"""
        tracking_path = self.get_scanned_files_tracking_path()
        try:
            if os.path.exists(tracking_path):
                with open(tracking_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_scanned_files(self, scanned_files):
        """스캔된 파일 목록 저장"""
        tracking_path = self.get_scanned_files_tracking_path()
        with open(tracking_path, 'w', encoding='utf-8') as f:
            json.dump(scanned_files, f, ensure_ascii=False, indent=4)

    def is_file_scanned(self, file_path):
        """파일이 이미 스캔되었는지 확인"""
        scanned_files = self.load_scanned_files()
        file_key = os.path.abspath(file_path)
        if file_key in scanned_files:
            # 파일 수정 시간 확인
            try:
                file_mtime = os.path.getmtime(file_path)
                saved_mtime = scanned_files[file_key].get('mtime', 0)
                if file_mtime == saved_mtime:
                    return True
            except OSError:
                pass
        return False

    def mark_file_as_scanned(self, file_path, imported_count):
        """파일을 스캔 완료로 표시"""
        scanned_files = self.load_scanned_files()
        file_key = os.path.abspath(file_path)
        try:
            file_mtime = os.path.getmtime(file_path)
            scanned_files[file_key] = {
                'mtime': file_mtime,
                'imported_count': imported_count,
                'scanned_at': datetime.now().isoformat()
            }
            self.save_scanned_files(scanned_files)
        except OSError:
            pass






