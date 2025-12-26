# 고객 데이터 저장 폼 설계

## 개요
고객 데이터를 입력하고 관리하는 폼의 구조 및 기능 설계입니다.

## 폼 구조

### 1. 고객 기본 정보 입력 폼
**파일 위치**: `_pages/_customer_data_form.py`

#### 입력 필드
- **고객 ID** (자동 생성 또는 수동 입력)
  - 이메일 또는 전화번호 기반 자동 생성
  - 수동 입력 옵션 제공
  
- **고객명** (필수)
  - 텍스트 입력
  - 최대 100자
  
- **이메일** (필수, 유효성 검사)
  - 이메일 형식 검증
  - 중복 확인
  
- **전화번호** (필수, 유효성 검사)
  - 전화번호 형식 정규화
  - 하이픈 자동 추가/제거
  
- **고객 유형** (선택)
  - 드롭다운: 일반, VIP, 기업, 기타
  
- **계정 생성일** (자동)
  - 현재 날짜/시간 자동 설정
  
- **고객 성향** (자동 분석 또는 수동 입력)
  - JSON 형식으로 저장
  - 자동 분석 결과 표시 및 수정 가능

### 2. 상담 이력 관리 폼
**파일 위치**: `_pages/_consultation_history_form.py`

#### 입력/표시 필드
- **상담 세션 목록**
  - 테이블 형태로 표시
  - 날짜, 시간, 요약, 상태 표시
  
- **상담 상세 정보**
  - 초기 문의 내용
  - 상담 메시지 전체
  - 상담 요약 (AI 생성)
  - 상담 소요 시간
  
- **이관 정보** (해당 시)
  - 이관 유형
  - 원본 언어 → 대상 언어
  - 이관 요약

### 3. 고객 감정 분석 결과 폼
**파일 위치**: `_pages/_customer_sentiment_form.py`

#### 표시 필드
- **감정 점수** (0-100)
  - 시각화 (게이지 또는 차트)
  
- **감정 레이블**
  - Positive, Neutral, Negative
  
- **긴급도**
  - Low, Medium, High
  
- **감정 키워드**
  - 태그 형태로 표시
  
- **감정 변화 추이**
  - 시계열 차트로 표시

### 4. 설문 조사 결과 폼
**파일 위치**: `_pages/_customer_survey_form.py`

#### 입력 필드
- **만족도 점수** (1-5 또는 1-10)
  - 별점 또는 슬라이더
  
- **응답 속도 점수**
- **해결 품질 점수**
- **상담원 친절도 점수**
- **전체 평점**
- **의견/코멘트** (텍스트 영역)

### 5. 고객 검증 로그 폼
**파일 위치**: `_pages/_customer_verification_form.py`

#### 표시 필드
- **검증 유형**
- **검증 데이터** (마스킹된 정보)
- **검증 결과** (성공/실패)
- **검증 시도 횟수**
- **OCR 결과** (해당 시)
- **검증 완료일시**

## 기능별 페이지 분리 계획

### 현재 구조
```
_pages/
├── _chat_simulator.py (메인)
├── _chat_history.py
├── _chat_initial_query.py
├── _chat_messages.py
├── _chat_agent_turn.py
├── _chat_customer_turn.py
└── _chat_closing.py
```

### 제안하는 새로운 구조
```
_pages/
├── chat_simulator/
│   ├── __init__.py
│   ├── main.py (메인 진입점)
│   ├── history.py (이력 관리)
│   ├── initial_query.py (초기 문의)
│   ├── messages.py (메시지 표시)
│   ├── agent_turn.py (에이전트 입력)
│   ├── customer_turn.py (고객 반응)
│   └── closing.py (종료 처리)
│
├── customer_data/
│   ├── __init__.py
│   ├── form.py (고객 데이터 입력 폼)
│   ├── list.py (고객 목록)
│   ├── detail.py (고객 상세 정보)
│   ├── sentiment.py (감정 분석)
│   ├── survey.py (설문 조사)
│   └── verification.py (검증 로그)
│
├── database/
│   ├── __init__.py
│   ├── models.py (데이터베이스 모델)
│   ├── connection.py (DB 연결)
│   ├── customers.py (고객 관련 쿼리)
│   ├── consultations.py (상담 관련 쿼리)
│   └── analytics.py (분석 관련 쿼리)
│
└── utils/
    ├── __init__.py
    ├── customer_id_generator.py (고객 ID 생성)
    ├── customer_matcher.py (동일 고객 확인)
    └── data_validator.py (데이터 검증)
```

## 데이터베이스 연동 모듈

### database/models.py
```python
# SQLAlchemy 또는 다른 ORM을 사용한 모델 정의
class Customer:
    customer_id
    customer_name
    email
    phone
    # ... 등등
```

### database/customers.py
```python
def create_customer(customer_data):
    """고객 생성"""
    pass

def get_customer(customer_id):
    """고객 정보 조회"""
    pass

def update_customer(customer_id, update_data):
    """고객 정보 업데이트"""
    pass

def find_similar_customers(email, phone):
    """동일 고객 찾기"""
    pass
```

## 고객 데이터 저장 시점

1. **초기 문의 시**
   - 고객 기본 정보 저장
   - 상담 세션 생성
   
2. **상담 진행 중**
   - 메시지 저장
   - 감정 분석 결과 저장
   - 검증 로그 저장
   
3. **상담 종료 시**
   - 상담 요약 저장
   - 고객 정보 업데이트 (마지막 상담일자 등)
   
4. **설문 조사 완료 시**
   - 설문 결과 저장
   - 고객 만족도 업데이트

## 동일 고객 확인 로직

### customer_matcher.py
```python
def identify_customer(email, phone, name=None):
    """
    동일 고객 확인
    1. 이메일 매칭
    2. 전화번호 매칭
    3. 이름 + 이메일/전화번호 조합 매칭
    4. 감정 분석 패턴 매칭 (선택적)
    """
    pass
```

## 보안 및 개인정보 보호

1. **데이터 마스킹**
   - 이메일: `user***@example.com`
   - 전화번호: `010-****-5678`
   
2. **접근 제어**
   - 권한별 데이터 접근 제한
   
3. **로그 기록**
   - 데이터 접근 로그
   - 수정 이력 추적








