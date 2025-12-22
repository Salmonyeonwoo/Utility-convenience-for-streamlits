import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

class CustomerDataManager:
    """고객 데이터 관리 클래스"""
    
    def __init__(self):
        self.data_dir = "customer_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_customer_data(self, customer_id: str, data: Dict[str, Any], format: str = "json") -> str:
        """고객 데이터 저장 (JSON 또는 CSV 형식)
        
        Args:
            customer_id: 고객 ID
            data: 저장할 데이터
            format: 저장 형식 ("json" 또는 "csv")
        
        Returns:
            저장된 파일 경로
        """
        customer_info = {
            "customer_id": customer_id,
            "updated_at": datetime.now().isoformat(),
            "data": data
        }
        
        if format.lower() == "csv":
            filepath = os.path.join(self.data_dir, f"{customer_id}.csv")
            self._save_to_csv(filepath, customer_info)
        else:
            filepath = os.path.join(self.data_dir, f"{customer_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(customer_info, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def _save_to_csv(self, filepath: str, customer_info: Dict[str, Any]):
        """CSV 형식으로 저장"""
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 헤더
            writer.writerow(["Key", "Value"])
            # 기본 정보
            writer.writerow(["customer_id", customer_info.get("customer_id", "")])
            writer.writerow(["updated_at", customer_info.get("updated_at", "")])
            
            # 데이터 섹션
            data = customer_info.get("data", {})
            if isinstance(data, dict):
                writer.writerow(["", ""])  # 빈 줄
                writer.writerow(["Data Section", ""])
                for key, value in data.items():
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value, ensure_ascii=False)
                    writer.writerow([key, value])
    
    def load_customer_data(self, customer_id: str, format: str = "json") -> Optional[Dict[str, Any]]:
        """고객 데이터 불러오기
        
        Args:
            customer_id: 고객 ID
            format: 파일 형식 ("json" 또는 "csv")
        
        Returns:
            고객 데이터 또는 None
        """
        if format.lower() == "csv":
            filepath = os.path.join(self.data_dir, f"{customer_id}.csv")
            if not os.path.exists(filepath):
                # CSV가 없으면 JSON 시도
                filepath = os.path.join(self.data_dir, f"{customer_id}.json")
                format = "json"
        else:
            filepath = os.path.join(self.data_dir, f"{customer_id}.json")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            if format.lower() == "csv":
                return self._load_from_csv(filepath)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            return None
    
    def _load_from_csv(self, filepath: str) -> Dict[str, Any]:
        """CSV 형식에서 로드"""
        result = {"customer_id": "", "updated_at": "", "data": {}}
        data_section = False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                key, value = row[0], row[1]
                
                if key == "customer_id":
                    result["customer_id"] = value
                elif key == "updated_at":
                    result["updated_at"] = value
                elif key == "Data Section":
                    data_section = True
                elif data_section and key:
                    # JSON 문자열인지 확인
                    try:
                        value = json.loads(value)
                    except:
                        pass
                    result["data"][key] = value
        
        return result
    
    def check_customer_exists(self, customer_id: str) -> bool:
        """고객 데이터 존재 여부 확인"""
        json_path = os.path.join(self.data_dir, f"{customer_id}.json")
        csv_path = os.path.join(self.data_dir, f"{customer_id}.csv")
        return os.path.exists(json_path) or os.path.exists(csv_path)
    
    def list_all_customers(self) -> List[Dict[str, Any]]:
        """모든 고객 목록 가져오기"""
        if not os.path.exists(self.data_dir):
            return []
        
        customers = []
        seen_ids = set()
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json') or filename.endswith('.csv'):
                customer_id = filename.rsplit('.', 1)[0]
                if customer_id in seen_ids:
                    continue
                seen_ids.add(customer_id)
                
                data = self.load_customer_data(customer_id)
                if data:
                    customers.append({
                        "customer_id": customer_id,
                        "updated_at": data.get("updated_at", ""),
                        "has_data": True
                    })
        
        return sorted(customers, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    def export_to_csv(self, customer_id: str) -> Optional[str]:
        """고객 데이터를 CSV로 내보내기"""
        data = self.load_customer_data(customer_id, format="json")
        if not data:
            return None
        
        filepath = self.save_customer_data(customer_id, data.get("data", {}), format="csv")
        return filepath
    
    def export_to_json(self, customer_id: str) -> Optional[str]:
        """고객 데이터를 JSON으로 내보내기"""
        data = self.load_customer_data(customer_id)
        if not data:
            return None
        
        filepath = self.save_customer_data(customer_id, data.get("data", {}), format="json")
        return filepath
    
    def create_sample_data(self, customer_id: str):
        """샘플 고객 데이터 생성 (테스트용)"""
        sample_data = {
            "name": "홍길동",
            "email": "hong@example.com",
            "phone": "010-1234-5678",
            "address": "서울시 강남구",
            "company": "테스트 회사",
            "notes": "VIP 고객",
            "purchase_history": [
                {"date": "2024-01-15", "item": "상품 A", "amount": 100000},
                {"date": "2024-02-20", "item": "상품 B", "amount": 200000}
            ],
            "preferences": {
                "language": "한국어",
                "contact_method": "이메일",
                "timezone": "Asia/Seoul"
            }
        }
        
        return self.save_customer_data(customer_id, sample_data)




