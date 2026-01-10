# ========================================
# video_handler_database.py
# 비디오 처리 - 데이터베이스 관리 모듈
# ========================================

import os
import json
from datetime import datetime
from typing import List, Dict, Any
from config import VIDEO_MAPPING_DB_FILE


def load_video_mapping_database() -> Dict[str, Any]:
    """비디오 매핑 데이터베이스를 로드합니다."""
    if os.path.exists(VIDEO_MAPPING_DB_FILE):
        try:
            with open(VIDEO_MAPPING_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"비디오 매핑 데이터베이스 로드 오류: {e}")
            return {"mappings": [], "feedback_history": []}
    return {"mappings": [], "feedback_history": []}


def save_video_mapping_database(db_data: Dict[str, Any]):
    """비디오 매핑 데이터베이스를 저장합니다."""
    try:
        with open(VIDEO_MAPPING_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"비디오 매핑 데이터베이스 저장 오류: {e}")


def add_video_mapping_feedback(
    customer_text: str,
    selected_video_path: str,
    emotion: str,
    gesture: str,
    context_keywords: List[str],
    user_rating: int,
    user_comment: str = ""
) -> None:
    """사용자 피드백을 비디오 매핑 데이터베이스에 추가합니다."""
    db_data = load_video_mapping_database()
    
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "customer_text": customer_text[:200],
        "selected_video": os.path.basename(selected_video_path) if selected_video_path else None,
        "video_path": selected_video_path,
        "emotion": emotion,
        "gesture": gesture,
        "context_keywords": context_keywords,
        "user_rating": user_rating,
        "user_comment": user_comment[:500] if user_comment else "",
        "is_natural_match": user_rating >= 4
    }
    
    db_data["feedback_history"].append(feedback_entry)
    
    # 매핑 규칙 업데이트 (평가가 높은 경우)
    if user_rating >= 4:
        mapping_key = f"{emotion}_{gesture}_{'_'.join(context_keywords) if context_keywords else 'none'}"
        
        existing_mapping = None
        for mapping in db_data["mappings"]:
            if mapping.get("key") == mapping_key:
                existing_mapping = mapping
                break
        
        if existing_mapping:
            total_rating = existing_mapping.get("total_rating", 0) + user_rating
            count = existing_mapping.get("count", 0) + 1
            existing_mapping["total_rating"] = total_rating
            existing_mapping["count"] = count
            existing_mapping["avg_rating"] = total_rating / count
            existing_mapping["last_updated"] = datetime.now().isoformat()
        else:
            db_data["mappings"].append({
                "key": mapping_key,
                "emotion": emotion,
                "gesture": gesture,
                "context_keywords": context_keywords,
                "recommended_video": os.path.basename(selected_video_path) if selected_video_path else None,
                "video_path": selected_video_path,
                "total_rating": user_rating,
                "count": 1,
                "avg_rating": float(user_rating),
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            })
    
    save_video_mapping_database(db_data)


def get_recommended_video_from_database(
    emotion: str,
    gesture: str,
    context_keywords: List[str]
) -> str:
    """데이터베이스에서 추천 비디오 경로를 가져옵니다."""
    db_data = load_video_mapping_database()
    
    mapping_key = f"{emotion}_{gesture}_{'_'.join(context_keywords) if context_keywords else 'none'}"
    
    # 정확한 매칭 찾기
    for mapping in db_data["mappings"]:
        if mapping.get("key") == mapping_key and mapping.get("avg_rating", 0) >= 4.0:
            video_path = mapping.get("video_path")
            if video_path and os.path.exists(video_path):
                return video_path
    
    # 부분 매칭 시도 (감정과 제스처만)
    partial_key = f"{emotion}_{gesture}_none"
    for mapping in db_data["mappings"]:
        if mapping.get("key") == partial_key and mapping.get("avg_rating", 0) >= 4.0:
            video_path = mapping.get("video_path")
            if video_path and os.path.exists(video_path):
                return video_path
    
    return None
