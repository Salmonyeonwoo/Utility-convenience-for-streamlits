"""
데이터 입출력 헬퍼 함수 모듈
JSON 파일 읽기/쓰기, 시뮬레이션 이력 관리, 오디오 기록 관리 등을 포함합니다.
"""
import os
import json
from typing import List, Dict, Any
from utils.config import (
    VOICE_META_FILE, SIM_META_FILE, AUDIO_DIR, DATA_DIR
)


def _load_json(path: str, default: Any):
    """JSON 파일을 안전하게 로드"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: str, data: Any):
    """JSON 파일을 안전하게 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_voice_records() -> List[Dict[str, Any]]:
    """음성 기록 목록 로드"""
    return _load_json(VOICE_META_FILE, [])


def save_voice_records(records: List[Dict[str, Any]]):
    """음성 기록 목록 저장"""
    _save_json(VOICE_META_FILE, records)


def load_simulation_histories_local(lang_key: str) -> List[Dict[str, Any]]:
    """시뮬레이션 이력 로드 (언어별)"""
    all_histories = _load_json(SIM_META_FILE, [])
    # 언어 필터링은 필요시 추가
    return all_histories


def save_simulation_history_local(
    initial_query: str,
    customer_type: str,
    messages: List[Dict[str, Any]],
    is_chat_ended: bool,
    attachment_context: str,
    is_call: bool = False
):
    """시뮬레이션 이력 저장"""
    all_histories = _load_json(SIM_META_FILE, [])
    
    new_history = {
        "id": len(all_histories) + 1,
        "initial_query": initial_query,
        "customer_type": customer_type,
        "messages": messages,
        "is_chat_ended": is_chat_ended,
        "attachment_context": attachment_context,
        "is_call": is_call,
        "timestamp": str(os.path.getmtime(SIM_META_FILE)) if os.path.exists(SIM_META_FILE) else ""
    }
    
    all_histories.append(new_history)
    _save_json(SIM_META_FILE, all_histories)


def delete_all_history_local():
    """모든 이력 삭제"""
    _save_json(SIM_META_FILE, [])


def export_history_to_json(histories: List[Dict[str, Any]], filename: str = None) -> str:
    """이력을 JSON 파일로 내보내기"""
    if filename is None:
        from datetime import datetime
        filename = f"simulation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    filepath = os.path.join(DATA_DIR, filename)
    _save_json(filepath, histories)
    return filepath


def export_history_to_text(histories: List[Dict[str, Any]], filename: str = None) -> str:
    """이력을 텍스트 파일로 내보내기"""
    if filename is None:
        from datetime import datetime
        filename = f"simulation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for hist in histories:
            f.write(f"=== 시뮬레이션 ID: {hist.get('id', 'N/A')} ===\n")
            f.write(f"초기 문의: {hist.get('initial_query', 'N/A')}\n")
            f.write(f"고객 유형: {hist.get('customer_type', 'N/A')}\n")
            f.write("\n--- 대화 이력 ---\n")
            for msg in hist.get('messages', []):
                f.write(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}\n")
            f.write("\n" + "="*50 + "\n\n")
    
    return filepath


def export_history_to_excel(histories: List[Dict[str, Any]], filename: str = None) -> str:
    """이력을 Excel 파일로 내보내기"""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas와 openpyxl이 필요합니다: pip install pandas openpyxl")
    
    if filename is None:
        from datetime import datetime
        filename = f"simulation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    filepath = os.path.join(DATA_DIR, filename)
    
    # 데이터를 DataFrame으로 변환
    rows = []
    for hist in histories:
        for msg in hist.get('messages', []):
            rows.append({
                '시뮬레이션 ID': hist.get('id', 'N/A'),
                '초기 문의': hist.get('initial_query', 'N/A'),
                '고객 유형': hist.get('customer_type', 'N/A'),
                '역할': msg.get('role', 'unknown'),
                '내용': msg.get('content', ''),
                '타임스탬프': hist.get('timestamp', 'N/A')
            })
    
    df = pd.DataFrame(rows)
    df.to_excel(filepath, index=False, engine='openpyxl')
    return filepath
