"""
고객 데이터 검증 모듈
고객 데이터의 유효성을 검증하는 클래스
"""
import re
import logging
from typing import Dict, Any, List

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomerDataValidator:
    """고객 데이터 검증 클래스"""
    
    def __init__(self):
        """초기화"""
        # 이메일 정규식
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        # 전화번호 정규식 (한국, 국제 형식 지원)
        self.phone_pattern = re.compile(
            r'^(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}$'
        )
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        고객 데이터 검증
        
        Args:
            data: 검증할 고객 데이터
            
        Returns:
            검증 결과 딕셔너리
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        errors = []
        warnings = []
        
        try:
            # 기본 구조 검증
            if not isinstance(data, dict):
                errors.append("데이터가 딕셔너리 형식이 아닙니다.")
                return {"valid": False, "errors": errors, "warnings": warnings}
            
            # data 필드 검증
            if "data" not in data:
                warnings.append("'data' 필드가 없습니다. 기본 구조를 생성합니다.")
                data["data"] = {}
            
            customer_info = data.get("data", {})
            
            # 이메일 검증
            if "email" in customer_info and customer_info["email"]:
                email = customer_info["email"]
                if not self._validate_email(email):
                    errors.append(f"유효하지 않은 이메일 형식: {email}")
            
            # 전화번호 검증
            if "phone" in customer_info and customer_info["phone"]:
                phone = customer_info["phone"]
                if not self._validate_phone(phone):
                    warnings.append(f"전화번호 형식이 표준이 아닐 수 있습니다: {phone}")
            
            # 구매 이력 검증
            if "purchase_history" in customer_info:
                purchase_history = customer_info["purchase_history"]
                if not isinstance(purchase_history, list):
                    errors.append("구매 이력이 리스트 형식이 아닙니다.")
                else:
                    for idx, purchase in enumerate(purchase_history):
                        if not isinstance(purchase, dict):
                            errors.append(f"구매 이력 {idx+1}번 항목이 딕셔너리 형식이 아닙니다.")
                        else:
                            # 필수 필드 확인
                            if "date" not in purchase:
                                warnings.append(f"구매 이력 {idx+1}번 항목에 날짜가 없습니다.")
                            if "item" not in purchase:
                                warnings.append(f"구매 이력 {idx+1}번 항목에 상품명이 없습니다.")
            
            # 대화 이력 검증
            if "conversations" in data:
                conversations = data["conversations"]
                if not isinstance(conversations, list):
                    errors.append("대화 이력이 리스트 형식이 아닙니다.")
                else:
                    for idx, conv in enumerate(conversations):
                        if not isinstance(conv, dict):
                            errors.append(f"대화 이력 {idx+1}번 항목이 딕셔너리 형식이 아닙니다.")
                        else:
                            # 필수 필드 확인
                            if "role" not in conv:
                                errors.append(f"대화 이력 {idx+1}번 항목에 역할(role)이 없습니다.")
                            if "content" not in conv:
                                errors.append(f"대화 이력 {idx+1}번 항목에 내용(content)이 없습니다.")
            
            valid = len(errors) == 0
            
            if valid:
                logger.info("고객 데이터 검증 성공")
            else:
                logger.warning(f"고객 데이터 검증 실패: {errors}")
            
            return {
                "valid": valid,
                "errors": errors,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"고객 데이터 검증 중 오류: {e}", exc_info=True)
            return {
                "valid": False,
                "errors": [f"검증 중 오류 발생: {str(e)}"],
                "warnings": warnings
            }
    
    def _validate_email(self, email: str) -> bool:
        """
        이메일 형식 검증
        
        Args:
            email: 검증할 이메일 주소
            
        Returns:
            유효성 여부
        """
        if not email or not isinstance(email, str):
            return False
        return bool(self.email_pattern.match(email.strip()))
    
    def _validate_phone(self, phone: str) -> bool:
        """
        전화번호 형식 검증
        
        Args:
            phone: 검증할 전화번호
            
        Returns:
            유효성 여부
        """
        if not phone or not isinstance(phone, str):
            return False
        
        # 숫자와 특수문자만 남기기
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # 최소 길이 확인 (국제 형식 포함)
        if len(cleaned) < 7:
            return False
        
        return bool(self.phone_pattern.match(phone.strip()))
    
    def sanitize_customer_id(self, customer_id: str) -> str:
        """
        고객 ID 정제 (파일명에 사용할 수 있도록)
        
        Args:
            customer_id: 원본 고객 ID
            
        Returns:
            정제된 고객 ID
        """
        if not customer_id:
            return ""
        
        # 파일명에 사용할 수 없는 문자 제거
        sanitized = customer_id.replace("/", "_").replace("\\", "_")
        sanitized = sanitized.replace(":", "_").replace("*", "_")
        sanitized = sanitized.replace("?", "_").replace('"', "_")
        sanitized = sanitized.replace("<", "_").replace(">", "_")
        sanitized = sanitized.replace("|", "_")
        
        return sanitized.strip()





