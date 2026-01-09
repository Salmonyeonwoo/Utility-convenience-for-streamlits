# ========================================
# chat_modules/customer_response_analyzer.py
# 고객 응답 분석 유틸리티 모듈
# ========================================


def detect_complaint(customer_response):
    """고객 불만/직접 응대 요청 감지"""
    complaint_keywords = [
        "불만", "불만족", "해결 안 됨", "도와주세요", "에이전트", "상담원", "직접",
        "상담원 연결", "직접 상담", "사람과", "complaint", "dissatisfied",
        "not resolved", "help me", "agent", "representative", "directly",
        "speak to", "talk to", "connect me"
    ]
    return any(keyword in customer_response.lower() for keyword in complaint_keywords)


def detect_cancellation_request(customer_response):
    """취소/환불 요청 감지"""
    cancellation_keywords = [
        "취소", "환불", "환불해주세요", "취소해주세요", "cancel", "refund",
        "cancel please", "refund please", "キャンセル", "返金"
    ]
    return any(keyword in customer_response.lower() for keyword in cancellation_keywords)


def detect_exception_reason(customer_response):
    """예외 사유 키워드 감지"""
    exception_keywords = [
        # 여행/숙박 관련
        "비행기 결항", "비행기 지연", "항공편 결항", "항공편 지연", "항공사", "airline",
        "flight cancelled", "flight delayed", "cancelled flight", "delayed flight",
        "날씨", "태풍", "폭설", "weather", "typhoon", "snowstorm",
        # 건강/긴급 상황
        "병가", "병원", "입원", "수술", "응급", "긴급", "sick", "hospital", "emergency",
        "medical", "surgery", "urgent", "critical",
        # 제품/배송 관련
        "기기 결함", "제품 결함", "불량품", "오작동", "고장", "작동 안 함", "안 됨", "안돼",
        "defect", "malfunction", "faulty", "broken", "not working", "doesn't work",
        "배송 지연", "배송 오류", "배송 누락", "배송 안 됨", "배송 못 받음", "배송 안 옴",
        "delivery delay", "delivery error", "delivery missing", "late delivery", 
        "wrong delivery", "not delivered", "missing delivery",
        # 제품 품질 문제
        "품질 문제", "품질 불량", "불량", "quality issue", "quality problem", "poor quality",
        # 포장/파손 문제
        "포장 파손", "박스 파손", "상품 파손", "포장 뜯김", "damaged", "broken package",
        # 교환/반품 관련
        "교환", "반품", "exchange", "return", "교환 요청", "반품 요청",
        # 일반 예외 사유
        "불가피", "예외", "특별한 사정", "특수한 경우", "unavoidable", "exceptional",
        "special circumstances", "unforeseen", "unexpected",
        # 법적/정책적 사유
        "법적", "정책", "규정", "legal", "policy", "regulation"
    ]
    return any(keyword in customer_response.lower() for keyword in exception_keywords)


def detect_satisfaction(customer_response):
    """만족/해결 키워드 감지"""
    satisfaction_keywords = [
        "감사합니다", "감사해요", "해결됐어요", "해결되었습니다", "알겠습니다", "좋아요",
        "thank you", "thanks", "resolved", "solved", "ok", "okay", "good",
        "ありがとうございます", "解決しました", "了解しました"
    ]
    return any(keyword in customer_response.lower() for keyword in satisfaction_keywords)

