"""
고객 데이터 저장소 모듈
고객 데이터의 파일 시스템 저장/로드를 담당
"""
import os
import json
import logging
from typing import Dict, Any, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomerDataStorage:
    """고객 데이터 저장소 클래스"""
    
    def __init__(self, data_dir: str = "customer_data"):
        """
        초기화
        
        Args:
            data_dir: 고객 데이터 저장 디렉토리 경로
        """
        self.data_dir = data_dir
        
        # 디렉토리 생성
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"저장소 디렉토리 초기화: {self.data_dir}")
        except Exception as e:
            logger.error(f"저장소 디렉토리 생성 실패: {e}")
            raise
    
    def _get_filepath(self, customer_id: str) -> str:
        """
        고객 ID에 해당하는 파일 경로 반환
        
        Args:
            customer_id: 고객 ID
            
        Returns:
            파일 경로
        """
        # 파일명에 사용할 수 없는 문자 제거
        safe_id = customer_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return os.path.join(self.data_dir, f"{safe_id}.json")
    
    def load(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        고객 데이터 로드
        
        Args:
            customer_id: 고객 ID
            
        Returns:
            고객 데이터 딕셔너리 또는 None
        """
        try:
            filepath = self._get_filepath(customer_id)
            
            if not os.path.exists(filepath):
                logger.debug(f"파일 없음: {filepath}")
                return None
            
            logger.debug(f"파일 로드 시도: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"파일 로드 성공: {filepath}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {filepath}, error={e}")
            return None
        except Exception as e:
            logger.error(f"파일 로드 실패: {filepath}, error={e}", exc_info=True)
            return None
    
    def save(self, customer_id: str, data: Dict[str, Any]) -> bool:
        """
        고객 데이터 저장
        
        Args:
            customer_id: 고객 ID
            data: 저장할 데이터
            
        Returns:
            저장 성공 여부
        """
        try:
            filepath = self._get_filepath(customer_id)
            
            # 임시 파일로 먼저 저장 (원자성 보장)
            temp_filepath = f"{filepath}.tmp"
            
            logger.debug(f"파일 저장 시도: {filepath}")
            
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 임시 파일을 실제 파일로 이동 (원자성 보장)
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_filepath, filepath)
            
            logger.info(f"파일 저장 성공: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"파일 저장 실패: customer_id={customer_id}, error={e}", exc_info=True)
            
            # 임시 파일 정리
            temp_filepath = f"{self._get_filepath(customer_id)}.tmp"
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            
            return False
    
    def delete(self, customer_id: str) -> bool:
        """
        고객 데이터 삭제
        
        Args:
            customer_id: 고객 ID
            
        Returns:
            삭제 성공 여부
        """
        try:
            filepath = self._get_filepath(customer_id)
            
            if not os.path.exists(filepath):
                logger.warning(f"삭제할 파일 없음: {filepath}")
                return False
            
            os.remove(filepath)
            logger.info(f"파일 삭제 성공: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"파일 삭제 실패: customer_id={customer_id}, error={e}", exc_info=True)
            return False
    
    def exists(self, customer_id: str) -> bool:
        """
        고객 데이터 존재 여부 확인
        
        Args:
            customer_id: 고객 ID
            
        Returns:
            존재 여부
        """
        filepath = self._get_filepath(customer_id)
        return os.path.exists(filepath)


