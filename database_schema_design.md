# 고객 데이터베이스 스키마 설계

## 개요
고객 상담 시뮬레이터를 위한 데이터베이스 스키마 설계 문서입니다.

## 테이블 구조

### 1. customers (고객 기본 정보 테이블)
고객의 기본 정보를 저장하는 마스터 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| customer_id | VARCHAR(50) | PRIMARY KEY | 고객 고유 ID (이메일 또는 전화번호 기반) |
| customer_name | VARCHAR(100) | | 고객명 |
| email | VARCHAR(255) | UNIQUE, INDEX | 이메일 주소 |
| phone | VARCHAR(20) | INDEX | 전화번호 |
| account_created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 계정 생성일 |
| last_access_at | DATETIME | | 마지막 접속일 |
| last_consultation_at | DATETIME | | 마지막 상담일자 |
| customer_type | VARCHAR(50) | | 고객 유형 (일반, VIP, 기업 등) |
| customer_tendency | TEXT | | 고객 성향 (JSON 형식으로 저장) |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 레코드 생성일시 |
| updated_at | DATETIME | DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 레코드 수정일시 |

**고객 성향 (customer_tendency) JSON 구조:**
```json
{
  "sentiment_score": 50,
  "urgency_level": "medium",
  "predicted_customer_type": "normal",
  "communication_style": "formal",
  "preferred_language": "ko"
}
```

### 2. consultation_sessions (상담 세션 테이블)
각 상담 세션의 기본 정보를 저장합니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| session_id | VARCHAR(100) | PRIMARY KEY | 상담 세션 고유 ID (UUID) |
| customer_id | VARCHAR(50) | FOREIGN KEY → customers.customer_id | 고객 ID |
| initial_query | TEXT | | 초기 문의 내용 |
| customer_type | VARCHAR(50) | | 해당 상담의 고객 유형 |
| consultation_date | DATETIME | DEFAULT CURRENT_TIMESTAMP | 상담 시작일시 |
| consultation_end_date | DATETIME | | 상담 종료일시 |
| consultation_duration | INT | | 상담 소요 시간 (초) |
| consultation_summary | TEXT | | 상담 요약 (AI 생성) |
| consultation_language | VARCHAR(10) | | 상담 언어 (ko, en, ja) |
| transfer_occurred | BOOLEAN | DEFAULT FALSE | 이관 발생 여부 |
| transfer_summary | TEXT | | 이관 요약 (번역 포함) |
| is_chat_ended | BOOLEAN | DEFAULT FALSE | 상담 종료 여부 |
| attachment_context | TEXT | | 첨부 파일 컨텍스트 (JSON) |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 레코드 생성일시 |
| updated_at | DATETIME | DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 레코드 수정일시 |

**인덱스:**
- `idx_customer_id` ON (customer_id)
- `idx_consultation_date` ON (consultation_date)

### 3. consultation_messages (상담 메시지 테이블)
상담 중 주고받은 모든 메시지를 저장합니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| message_id | BIGINT | PRIMARY KEY, AUTO_INCREMENT | 메시지 고유 ID |
| session_id | VARCHAR(100) | FOREIGN KEY → consultation_sessions.session_id | 상담 세션 ID |
| message_order | INT | | 메시지 순서 |
| role | VARCHAR(50) | | 메시지 역할 (customer, agent_response, supervisor 등) |
| content | TEXT | | 메시지 내용 |
| feedback | VARCHAR(20) | | 피드백 (thumbs_up, thumbs_down, null) |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 메시지 생성일시 |

**인덱스:**
- `idx_session_id` ON (session_id)
- `idx_session_order` ON (session_id, message_order)

### 4. customer_sentiment_analysis (고객 감정 분석 테이블)
각 상담 세션별 고객 감정 분석 결과를 저장합니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| analysis_id | BIGINT | PRIMARY KEY, AUTO_INCREMENT | 분석 고유 ID |
| session_id | VARCHAR(100) | FOREIGN KEY → consultation_sessions.session_id | 상담 세션 ID |
| customer_id | VARCHAR(50) | FOREIGN KEY → customers.customer_id | 고객 ID |
| sentiment_score | INT | | 감정 점수 (0-100) |
| sentiment_label | VARCHAR(50) | | 감정 레이블 (positive, neutral, negative) |
| urgency_level | VARCHAR(20) | | 긴급도 (low, medium, high) |
| emotion_keywords | TEXT | | 감정 키워드 (JSON 배열) |
| analysis_timestamp | DATETIME | DEFAULT CURRENT_TIMESTAMP | 분석 시점 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 레코드 생성일시 |

**인덱스:**
- `idx_session_id` ON (session_id)
- `idx_customer_id` ON (customer_id)
- `idx_sentiment_score` ON (sentiment_score)

### 5. customer_surveys (고객 설문 조사 테이블)
상담 종료 후 고객이 작성한 설문 조사 결과를 저장합니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| survey_id | BIGINT | PRIMARY KEY, AUTO_INCREMENT | 설문 고유 ID |
| session_id | VARCHAR(100) | FOREIGN KEY → consultation_sessions.session_id | 상담 세션 ID |
| customer_id | VARCHAR(50) | FOREIGN KEY → customers.customer_id | 고객 ID |
| satisfaction_score | INT | | 만족도 점수 (1-5 또는 1-10) |
| response_speed_score | INT | | 응답 속도 점수 |
| solution_quality_score | INT | | 해결 품질 점수 |
| agent_friendliness_score | INT | | 상담원 친절도 점수 |
| overall_rating | INT | | 전체 평점 |
| survey_comments | TEXT | | 설문 의견/코멘트 |
| survey_language | VARCHAR(10) | | 설문 작성 언어 |
| submitted_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 설문 제출일시 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 레코드 생성일시 |

**인덱스:**
- `idx_session_id` ON (session_id)
- `idx_customer_id` ON (customer_id)
- `idx_satisfaction_score` ON (satisfaction_score)

### 6. agent_evaluations (응대 평가 테이블)
에이전트의 응대에 대한 평가 데이터를 저장합니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| evaluation_id | BIGINT | PRIMARY KEY, AUTO_INCREMENT | 평가 고유 ID |
| session_id | VARCHAR(100) | FOREIGN KEY → consultation_sessions.session_id | 상담 세션 ID |
| message_id | BIGINT | FOREIGN KEY → consultation_messages.message_id | 평가 대상 메시지 ID |
| agent_id | VARCHAR(50) | | 상담원 ID |
| feedback_type | VARCHAR(20) | | 피드백 유형 (thumbs_up, thumbs_down) |
| evaluation_timestamp | DATETIME | DEFAULT CURRENT_TIMESTAMP | 평가 시점 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 레코드 생성일시 |

**인덱스:**
- `idx_session_id` ON (session_id)
- `idx_message_id` ON (message_id)
- `idx_agent_id` ON (agent_id)

### 7. customer_verification_logs (고객 검증 로그 테이블)
고객 검증 시도 및 결과를 저장합니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| verification_id | BIGINT | PRIMARY KEY, AUTO_INCREMENT | 검증 고유 ID |
| session_id | VARCHAR(100) | FOREIGN KEY → consultation_sessions.session_id | 상담 세션 ID |
| customer_id | VARCHAR(50) | FOREIGN KEY → customers.customer_id | 고객 ID |
| verification_type | VARCHAR(50) | | 검증 유형 (receipt, card, name, email, phone 등) |
| verification_data | TEXT | | 검증 데이터 (마스킹된 정보) |
| verification_result | BOOLEAN | | 검증 결과 (성공/실패) |
| verification_attempts | INT | DEFAULT 1 | 검증 시도 횟수 |
| ocr_result | TEXT | | OCR 결과 (첨부 파일 검증 시) |
| verified_at | DATETIME | | 검증 완료일시 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 레코드 생성일시 |

**인덱스:**
- `idx_session_id` ON (session_id)
- `idx_customer_id` ON (customer_id)
- `idx_verification_result` ON (verification_result)

### 8. consultation_transfers (상담 이관 테이블)
언어 이관 및 상담 이관 정보를 저장합니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| transfer_id | BIGINT | PRIMARY KEY, AUTO_INCREMENT | 이관 고유 ID |
| session_id | VARCHAR(100) | FOREIGN KEY → consultation_sessions.session_id | 상담 세션 ID |
| transfer_type | VARCHAR(50) | | 이관 유형 (language_transfer, escalation 등) |
| source_language | VARCHAR(10) | | 원본 언어 |
| target_language | VARCHAR(10) | | 대상 언어 |
| transfer_summary_original | TEXT | | 원본 이관 요약 |
| transfer_summary_translated | TEXT | | 번역된 이관 요약 |
| translation_success | BOOLEAN | | 번역 성공 여부 |
| transferred_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 이관 일시 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 레코드 생성일시 |

**인덱스:**
- `idx_session_id` ON (session_id)
- `idx_transfer_type` ON (transfer_type)

## 동일 고객 식별 로직

### 고객 ID 생성 규칙
1. **이메일 기반**: 이메일이 제공된 경우, 이메일을 customer_id로 사용
2. **전화번호 기반**: 이메일이 없고 전화번호가 있는 경우, 전화번호를 customer_id로 사용
3. **임시 ID**: 둘 다 없는 경우, `customer_{session_id}` 형식으로 임시 ID 생성

### 동일 고객 확인 방법
1. **이메일 매칭**: 정확한 이메일 주소 일치
2. **전화번호 매칭**: 정규화된 전화번호 일치 (하이픈, 공백 제거 후 비교)
3. **감정 분석 패턴**: 감정 분석 결과와 문의 패턴 비교
4. **이름 매칭**: 고객명과 검증 정보 비교

## 데이터베이스 선택 권장사항

### 옵션 1: SQLite (개발/소규모)
- 파일 기반, 설정 간단
- Python 내장 지원
- 소규모 프로젝트에 적합

### 옵션 2: PostgreSQL (프로덕션)
- 관계형 데이터베이스
- JSON 컬럼 지원 (고객 성향, 감정 키워드 등)
- 확장성과 성능 우수

### 옵션 3: MySQL/MariaDB
- 널리 사용되는 관계형 데이터베이스
- JSON 컬럼 지원 (MySQL 5.7+, MariaDB 10.2+)

## 마이그레이션 전략

1. **초기 버전**: SQLite로 시작
2. **확장 시**: PostgreSQL로 마이그레이션
3. **백업**: 정기적인 데이터 백업 필수

## 보안 고려사항

1. **개인정보 보호**: 
   - 민감 정보는 마스킹 처리
   - GDPR/개인정보보호법 준수
   
2. **데이터 암호화**:
   - 전송 중 암호화 (TLS/SSL)
   - 저장 시 암호화 (선택적)

3. **접근 제어**:
   - 데이터베이스 접근 권한 관리
   - 로그 기록 및 감사











