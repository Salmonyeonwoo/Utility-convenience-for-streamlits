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


def summarize_and_score_conversation(
    messages: List[Dict[str, Any]],
    initial_query: str,
    customer_type: str,
    llm_summarize_func
) -> Dict[str, Any]:
    """
    채팅 내용을 요약하고 점수화하여 저장
    개인정보(거주지, 문화권, 언어 등)는 제외하고 주요 문의 내용과 답변, 고객성향 등을 수치로 점수화
    """
    # 대화 내용을 텍스트로 변환 (개인정보 제외)
    conversation_text = f"초기 문의: {initial_query}\n고객 유형: {customer_type}\n\n"
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        # 개인정보 패턴 제거 (이름, 전화번호, 주소 등)
        import re
        content = re.sub(r'\b\d{3}-\d{4}-\d{4}\b', '[전화번호]', content)  # 전화번호
        content = re.sub(r'\b\d{2,3}-\d{3,4}-\d{4}\b', '[전화번호]', content)
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[이메일]', content)  # 이메일
        conversation_text += f"{role}: {content}\n"
    
    # LLM을 사용하여 요약 및 점수화
    prompt = f"""
    다음 고객 응대 대화를 분석하여 요약하고 점수화해주세요.
    
    {conversation_text}
    
    다음 형식으로 JSON으로 응답해주세요:
    {{
        "summary": "주요 문의 내용과 답변 요약 (개인정보 제외)",
        "customer_tendency_score": {{
            "불만도": 0-100,
            "긴급도": 0-100,
            "복잡도": 0-100,
            "만족도": 0-100
        }},
        "inquiry_category": "문의 유형",
        "solution_status": "해결 상태",
        "key_points": ["주요 포인트1", "주요 포인트2"]
    }}
    """
    
    try:
        summary_result = llm_summarize_func(prompt)
        # JSON 파싱 시도 (강화된 오류 처리)
        import json
        import re
        
        summary_text = summary_result.strip()
        
        # JSON 추출 (더 강력한 방법)
        if "```" in summary_text:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', summary_text, re.DOTALL)
            if json_match:
                summary_text = json_match.group(1)
            else:
                summary_text = re.sub(r'```(?:json)?\s*', '', summary_text)
                summary_text = re.sub(r'\s*```', '', summary_text)
        
        # JSON 객체 찾기
        json_match = re.search(r'\{.*\}', summary_text, re.DOTALL)
        if json_match:
            summary_text = json_match.group(0)
        
        summary_text = summary_text.strip()
        
        if summary_text.startswith('{'):
            try:
                summary_data = json.loads(summary_text)
            except json.JSONDecodeError as json_err:
                # 파싱 실패 시 기본 구조 생성
                print(f"요약 JSON 파싱 오류: {json_err}")
                summary_data = {
                    "summary": summary_result[:500],  # 처음 500자만
                    "customer_tendency_score": {
                        "불만도": 50,
                        "긴급도": 50,
                        "복잡도": 50,
                        "만족도": 50
                    },
                    "inquiry_category": customer_type,
                    "solution_status": "진행중",
                    "key_points": []
                }
        else:
            # JSON이 아닌 경우 기본 구조 생성
            summary_data = {
                "summary": summary_result[:500],  # 처음 500자만
                "customer_tendency_score": {
                    "불만도": 50,
                    "긴급도": 50,
                    "복잡도": 50,
                    "만족도": 50
                },
                "inquiry_category": customer_type,
                "solution_status": "진행중",
                "key_points": []
            }
        
        summary_data["timestamp"] = str(os.path.getmtime(SIM_META_FILE)) if os.path.exists(SIM_META_FILE) else ""
        summary_data["initial_query"] = initial_query
        summary_data["customer_type"] = customer_type
        
        return summary_data
    except Exception as e:
        # 오류 발생 시 기본 구조 반환
        return {
            "summary": conversation_text[:500],
            "customer_tendency_score": {
                "불만도": 50,
                "긴급도": 50,
                "복잡도": 50,
                "만족도": 50
            },
            "inquiry_category": customer_type,
            "solution_status": "오류",
            "key_points": [],
            "error": str(e)
        }


def save_daily_customer_guide(
    summaries: List[Dict[str, Any]],
    date_str: str = None
) -> str:
    """
    일일 고객 가이드 파일 생성 (예: 251130_고객가이드.TXT)
    고객별 응답 내용과 문화권, 강성 고객 가이드라인 등을 문서화
    """
    from datetime import datetime
    
    if date_str is None:
        date_str = datetime.now().strftime("%y%m%d")
    
    filename = f"{date_str}_고객가이드.TXT"
    filepath = os.path.join(DATA_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"=== {date_str} 고객 응대 가이드라인 ===\n\n")
        f.write("이 문서는 AI 고객 응대 시뮬레이터에서 수집된 데이터를 기반으로 생성되었습니다.\n")
        f.write("고객별 응답 내용, 문화권, 강성 고객 가이드라인 등을 포함합니다.\n\n")
        f.write("="*60 + "\n\n")
        
        for idx, summary in enumerate(summaries, 1):
            f.write(f"--- 케이스 {idx} ---\n")
            f.write(f"초기 문의: {summary.get('initial_query', 'N/A')}\n")
            f.write(f"고객 유형: {summary.get('customer_type', 'N/A')}\n")
            f.write(f"문의 유형: {summary.get('inquiry_category', 'N/A')}\n")
            f.write(f"요약: {summary.get('summary', 'N/A')}\n\n")
            
            scores = summary.get('customer_tendency_score', {})
            f.write("고객 성향 점수:\n")
            f.write(f"  - 불만도: {scores.get('불만도', 0)}/100\n")
            f.write(f"  - 긴급도: {scores.get('긴급도', 0)}/100\n")
            f.write(f"  - 복잡도: {scores.get('복잡도', 0)}/100\n")
            f.write(f"  - 만족도: {scores.get('만족도', 0)}/100\n\n")
            
            key_points = summary.get('key_points', [])
            if key_points:
                f.write("주요 포인트:\n")
                for point in key_points:
                    f.write(f"  - {point}\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    return filepath
