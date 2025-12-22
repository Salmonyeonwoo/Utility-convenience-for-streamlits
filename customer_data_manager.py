"""
고객 데이터 관리자 모듈
고객 데이터의 로드, 저장, 검증을 담당하는 메인 클래스
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from customer_data_storage import CustomerDataStorage
from customer_data_validator import CustomerDataValidator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomerDataManager:
    """고객 데이터 관리 클래스"""
    
    def __init__(self, data_dir: str = "customer_data"):
        """
        초기화
        
        Args:
            data_dir: 고객 데이터 저장 디렉토리 경로
        """
        self.data_dir = data_dir
        self.storage = CustomerDataStorage(data_dir)
        self.validator = CustomerDataValidator()
        
        # 디렉토리 생성
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"고객 데이터 디렉토리 초기화 완료: {self.data_dir}")
        except Exception as e:
            logger.error(f"고객 데이터 디렉토리 생성 실패: {e}")
            raise
    
    def load_customer_data(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        고객 데이터 로드
        
        Args:
            customer_id: 고객 ID (이메일 또는 전화번호)
            
        Returns:
            고객 데이터 딕셔너리 또는 None
        """
        try:
            logger.info(f"고객 데이터 로드 시도: customer_id={customer_id}")
            
            # 고객 ID 검증
            if not customer_id or not isinstance(customer_id, str):
                logger.warning(f"유효하지 않은 고객 ID: {customer_id}")
                return None
            
            # 저장소에서 로드
            data = self.storage.load(customer_id)
            
            if data:
                logger.info(f"고객 데이터 로드 성공: customer_id={customer_id}")
                return data
            else:
                logger.info(f"고객 데이터 없음: customer_id={customer_id}")
                return None
                
        except Exception as e:
            logger.error(f"고객 데이터 로드 실패: customer_id={customer_id}, error={e}", exc_info=True)
            return None
    
    def save_customer_data(
        self,
        customer_id: str,
        customer_data: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        고객 데이터 저장
        
        Args:
            customer_id: 고객 ID (이메일 또는 전화번호)
            customer_data: 저장할 고객 데이터
            merge: 기존 데이터와 병합할지 여부 (기본값: True)
            
        Returns:
            저장 성공 여부
        """
        try:
            logger.info(f"고객 데이터 저장 시도: customer_id={customer_id}, merge={merge}")
            
            # 고객 ID 검증
            if not customer_id or not isinstance(customer_id, str):
                logger.error(f"유효하지 않은 고객 ID: {customer_id}")
                return False
            
            # 데이터 검증
            validation_result = self.validator.validate(customer_data)
            if not validation_result["valid"]:
                logger.error(f"고객 데이터 검증 실패: {validation_result['errors']}")
                return False
            
            # 기존 데이터 로드 (병합 모드인 경우)
            if merge:
                existing_data = self.storage.load(customer_id)
                if existing_data:
                    logger.info(f"기존 데이터 발견, 병합 모드: customer_id={customer_id}")
                    # 병합 로직
                    merged_data = self._merge_customer_data(existing_data, customer_data)
                    customer_data = merged_data
            
            # 업데이트 시간 추가
            customer_data["updated_at"] = datetime.now().isoformat()
            if "created_at" not in customer_data:
                customer_data["created_at"] = datetime.now().isoformat()
            
            # 저장
            success = self.storage.save(customer_id, customer_data)
            
            if success:
                logger.info(f"고객 데이터 저장 성공: customer_id={customer_id}")
            else:
                logger.error(f"고객 데이터 저장 실패: customer_id={customer_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"고객 데이터 저장 중 오류: customer_id={customer_id}, error={e}", exc_info=True)
            return False
    
    def _merge_customer_data(
        self,
        existing_data: Dict[str, Any],
        new_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        기존 고객 데이터와 새 데이터 병합
        
        Args:
            existing_data: 기존 고객 데이터
            new_data: 새로운 고객 데이터
            
        Returns:
            병합된 고객 데이터
        """
        try:
            merged = existing_data.copy()
            
            # 기본 정보 업데이트 (새 값이 있으면 덮어쓰기)
            if "data" in new_data:
                if "data" not in merged:
                    merged["data"] = {}
                
                # 기본 정보 병합
                for key in ["name", "email", "phone", "company"]:
                    if key in new_data["data"]:
                        merged["data"][key] = new_data["data"][key]
                
                # 구매 이력 병합 (중복 제거)
                if "purchase_history" in new_data["data"]:
                    if "purchase_history" not in merged["data"]:
                        merged["data"]["purchase_history"] = []
                    
                    existing_purchases = {
                        (p.get("date"), p.get("item")) 
                        for p in merged["data"]["purchase_history"]
                    }
                    
                    for purchase in new_data["data"]["purchase_history"]:
                        purchase_key = (purchase.get("date"), purchase.get("item"))
                        if purchase_key not in existing_purchases:
                            merged["data"]["purchase_history"].append(purchase)
                    
                    # 날짜순 정렬
                    merged["data"]["purchase_history"].sort(
                        key=lambda x: x.get("date", ""),
                        reverse=True
                    )
                
                # 메모 병합
                if "notes" in new_data["data"]:
                    existing_notes = merged["data"].get("notes", "")
                    new_notes = new_data["data"]["notes"]
                    if new_notes:
                        if existing_notes:
                            merged["data"]["notes"] = f"{existing_notes}\n\n{new_notes}"
                        else:
                            merged["data"]["notes"] = new_notes
            
            # 대화 이력 병합
            if "conversations" in new_data:
                if "conversations" not in merged:
                    merged["conversations"] = []
                merged["conversations"].extend(new_data["conversations"])
            
            logger.info("고객 데이터 병합 완료")
            return merged
            
        except Exception as e:
            logger.error(f"고객 데이터 병합 중 오류: {e}", exc_info=True)
            return existing_data
    
    def list_all_customers(self) -> List[Dict[str, Any]]:
        """
        모든 고객 목록 조회
        
        Returns:
            고객 목록 (최신순 정렬)
        """
        try:
            logger.info("모든 고객 목록 조회 시도")
            
            customers = []
            if not os.path.exists(self.data_dir):
                logger.warning(f"고객 데이터 디렉토리 없음: {self.data_dir}")
                return []
            
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    customer_id = filename.replace('.json', '')
                    data = self.load_customer_data(customer_id)
                    if data:
                        customers.append({
                            "customer_id": customer_id,
                            "updated_at": data.get("updated_at", ""),
                            "created_at": data.get("created_at", ""),
                            "has_data": True,
                            "name": data.get("data", {}).get("name", "N/A"),
                            "email": data.get("data", {}).get("email", "N/A"),
                            "phone": data.get("data", {}).get("phone", "N/A")
                        })
            
            # 업데이트 시간 기준 최신순 정렬
            sorted_customers = sorted(
                customers,
                key=lambda x: x.get("updated_at", ""),
                reverse=True
            )
            
            logger.info(f"고객 목록 조회 완료: {len(sorted_customers)}건")
            return sorted_customers
            
        except Exception as e:
            logger.error(f"고객 목록 조회 중 오류: {e}", exc_info=True)
            return []
    
    def delete_customer_data(self, customer_id: str) -> bool:
        """
        고객 데이터 삭제
        
        Args:
            customer_id: 고객 ID
            
        Returns:
            삭제 성공 여부
        """
        try:
            logger.info(f"고객 데이터 삭제 시도: customer_id={customer_id}")
            success = self.storage.delete(customer_id)
            
            if success:
                logger.info(f"고객 데이터 삭제 성공: customer_id={customer_id}")
            else:
                logger.warning(f"고객 데이터 삭제 실패: customer_id={customer_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"고객 데이터 삭제 중 오류: {e}", exc_info=True)
            return False
    
    def get_customer_count(self) -> int:
        """
        전체 고객 수 조회
        
        Returns:
            고객 수
        """
        try:
            customers = self.list_all_customers()
            return len(customers)
        except Exception as e:
            logger.error(f"고객 수 조회 중 오류: {e}", exc_info=True)
            return 0





