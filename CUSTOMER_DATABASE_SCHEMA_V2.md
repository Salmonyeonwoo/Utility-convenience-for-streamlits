# 고객 데이터베이스 스키마 V2

## 테이블 구조

### 1. customers (고객 기본 정보 테이블)
```json
{
  "customer_id": "string (PK)",
  "customer_name": "string",
  "phone": "string",
  "email": "string",
  "account_created_date": "datetime",
  "last_login_date": "datetime",
  "last_consultation_date": "datetime",
  "personality_type": "string (일반/신중형/활발형/가족형/프리미엄형/절약형/자유형)",
  "preferred_destination": "string",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### 2. consultations (상담 이력 테이블)
```json
{
  "consultation_id": "string (PK)",
  "customer_id": "string (FK -> customers.customer_id)",
  "consultation_type": "string (chat/email/phone)",
  "consultation_date": "datetime",
  "consultation_content": "text",
  "consultation_summary": "text",
  "operator_id": "string",
  "duration_minutes": "integer",
  "status": "string (completed/in_progress/cancelled)",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### 3. consultation_surveys (상담별 설문 및 응대평가 테이블)
```json
{
  "survey_id": "string (PK)",
  "consultation_id": "string (FK -> consultations.consultation_id)",
  "customer_id": "string (FK -> customers.customer_id)",
  "satisfaction_score": "float (1-5)",
  "response_time_score": "float (1-5)",
  "problem_resolution_score": "float (1-5)",
  "overall_rating": "float (1-5)",
  "survey_comments": "text",
  "survey_date": "datetime",
  "created_at": "datetime"
}
```

### 4. customer_sentiment (고객 감정 분석 테이블)
```json
{
  "sentiment_id": "string (PK)",
  "customer_id": "string (FK -> customers.customer_id)",
  "consultation_id": "string (FK -> consultations.consultation_id)",
  "sentiment_type": "string (positive/neutral/negative)",
  "sentiment_score": "float (-1.0 to 1.0)",
  "emotion_keywords": "array of strings",
  "analysis_date": "datetime",
  "created_at": "datetime"
}
```

### 5. customer_evaluation_data (고객 평가 데이터 테이블)
```json
{
  "evaluation_id": "string (PK)",
  "customer_id": "string (FK -> customers.customer_id)",
  "consultation_id": "string (FK -> consultations.consultation_id)",
  "evaluation_metrics": {
    "response_quality": "float",
    "communication_style": "string",
    "problem_resolution": "boolean",
    "customer_satisfaction": "float"
  },
  "evaluation_date": "datetime",
  "created_at": "datetime"
}
```

## JSON 파일 구조 (현재 파일 기반 저장 방식)

### data/customers.json
```json
{
  "customers": [
    {
      "customer_id": "CUST001",
      "customer_name": "홍길동",
      "phone": "010-1234-5678",
      "email": "hong@example.com",
      "account_created_date": "2024-01-01 00:00:00",
      "last_login_date": "2024-12-25 10:30:00",
      "last_consultation_date": "2024-12-25 10:30:00",
      "personality_type": "일반",
      "preferred_destination": "제주도",
      "created_at": "2024-01-01 00:00:00",
      "updated_at": "2024-12-25 10:30:00"
    }
  ]
}
```

### data/consultations.json
```json
{
  "consultations": [
    {
      "consultation_id": "CONSULT001",
      "customer_id": "CUST001",
      "consultation_type": "chat",
      "consultation_date": "2024-12-25 10:30:00",
      "consultation_content": "전체 상담 내용 텍스트",
      "consultation_summary": "상담 요약",
      "operator_id": "OPER001",
      "duration_minutes": 15,
      "status": "completed",
      "created_at": "2024-12-25 10:30:00",
      "updated_at": "2024-12-25 10:45:00"
    }
  ]
}
```

### data/consultation_surveys.json
```json
{
  "surveys": [
    {
      "survey_id": "SURVEY001",
      "consultation_id": "CONSULT001",
      "customer_id": "CUST001",
      "satisfaction_score": 4.5,
      "response_time_score": 4.0,
      "problem_resolution_score": 5.0,
      "overall_rating": 4.5,
      "survey_comments": "친절하게 응대해주셔서 감사합니다.",
      "survey_date": "2024-12-25 10:45:00",
      "created_at": "2024-12-25 10:45:00"
    }
  ]
}
```

### data/customer_sentiment.json
```json
{
  "sentiments": [
    {
      "sentiment_id": "SENT001",
      "customer_id": "CUST001",
      "consultation_id": "CONSULT001",
      "sentiment_type": "positive",
      "sentiment_score": 0.75,
      "emotion_keywords": ["만족", "친절", "빠른응답"],
      "analysis_date": "2024-12-25 10:45:00",
      "created_at": "2024-12-25 10:45:00"
    }
  ]
}
```

### data/customer_evaluation_data.json
```json
{
  "evaluations": [
    {
      "evaluation_id": "EVAL001",
      "customer_id": "CUST001",
      "consultation_id": "CONSULT001",
      "evaluation_metrics": {
        "response_quality": 4.5,
        "communication_style": "친절함",
        "problem_resolution": true,
        "customer_satisfaction": 4.5
      },
      "evaluation_date": "2024-12-25 10:45:00",
      "created_at": "2024-12-25 10:45:00"
    }
  ]
}
```

## 동일 고객 확인 로직

동일 고객 여부는 다음 필드들을 기반으로 확인:
1. email (우선순위 1)
2. phone (우선순위 2)
3. customer_name + phone (우선순위 3)
4. customer_name + email (우선순위 4)

## 데이터 저장 시점

1. 채팅/이메일 종료 시:
   - consultations 테이블에 상담 이력 저장
   - consultation_content에 전체 대화 내용 저장
   - consultation_summary에 AI 요약 저장
   - customers 테이블의 last_consultation_date 업데이트

2. 전화 종료 시:
   - consultations 테이블에 상담 이력 저장
   - consultation_content에 전사된 전화 내용 저장
   - consultation_summary에 AI 요약 저장
   - customers 테이블의 last_consultation_date 업데이트

3. 설문 완료 시:
   - consultation_surveys 테이블에 설문 데이터 저장

4. 감정 분석 수행 시:
   - customer_sentiment 테이블에 감정 분석 결과 저장

