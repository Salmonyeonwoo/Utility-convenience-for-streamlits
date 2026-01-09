# ========================================
# utils/history_handler_stats.py
# 이력 관리 - 통계 함수들
# ========================================

import hashlib
from datetime import datetime
from typing import Dict, Any
from config import SIM_META_FILE
from utils import _load_json

def get_daily_data_statistics(language: str = "ko") -> Dict[str, Any]:
    """일일 데이터 수집 통계를 반환합니다."""
    histories = _load_json(SIM_META_FILE, [])
    today = datetime.now().date()
    
    today_histories = []
    for h in histories:
        try:
            ts = datetime.fromisoformat(h.get("timestamp", "")).date()
            if ts == today and h.get("summary") and isinstance(h.get("summary"), dict):
                today_histories.append(h)
        except:
            continue
    
    unique_customers = set()
    for h in today_histories:
        customer_id = h.get("id", "")
        initial_query = h.get("initial_query", "")
        customer_hash = hashlib.md5(f"{customer_id}_{initial_query[:50]}".encode()).hexdigest()
        unique_customers.add(customer_hash)
    
    return {
        "date": today.isoformat(),
        "total_cases": len(today_histories),
        "unique_customers": len(unique_customers),
        "target_met": len(unique_customers) >= 5,
        "cases_with_summary": len([h for h in today_histories if h.get("summary")])
    }

