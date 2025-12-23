# 파일 구조 재구성 계획

## 현재 구조 분석

### 기존 파일 구조
```
_pages/
├── _chat_simulator.py (메인 진입점)
├── _chat_history.py (이력 관리)
├── _chat_initial_query.py (초기 문의)
├── _chat_messages.py (메시지 표시)
├── _chat_agent_turn.py (에이전트 입력)
├── _chat_customer_turn.py (고객 반응)
└── _chat_closing.py (종료 처리)
```

## 제안하는 새로운 구조

### 1. 채팅 시뮬레이터 모듈
```
_pages/chat_simulator/
├── __init__.py
├── main.py                    # 메인 진입점 (render_chat_simulator)
├── history.py                 # 이력 관리 (render_chat_history)
├── initial_query.py           # 초기 문의 (render_initial_query)
├── messages.py                # 메시지 표시 (render_chat_messages)
├── agent_turn.py              # 에이전트 입력 (render_agent_turn)
├── customer_turn.py           # 고객 반응 (render_customer_turn)
├── closing.py                 # 종료 처리 (render_closing_stages)
└── components/
    ├── __init__.py
    ├── aht_timer.py           # AHT 타이머
    ├── outbound_call.py       # 발신 전화
    └── downloads.py           # 다운로드 기능
```

### 2. 고객 데이터 관리 모듈
```
_pages/customer_data/
├── __init__.py
├── form.py                    # 고객 데이터 입력/수정 폼
├── list.py                    # 고객 목록 조회
├── detail.py                  # 고객 상세 정보
├── sentiment.py               # 감정 분석 결과
├── survey.py                  # 설문 조사 결과
└── verification.py            # 검증 로그
```

### 3. 데이터베이스 모듈
```
database/
├── __init__.py
├── connection.py              # DB 연결 관리
├── models.py                  # SQLAlchemy 모델 정의
├── customers.py               # 고객 관련 쿼리
├── consultations.py           # 상담 관련 쿼리
├── messages.py                # 메시지 관련 쿼리
├── sentiment.py              # 감정 분석 관련 쿼리
├── surveys.py                # 설문 조사 관련 쿼리
└── analytics.py              # 분석 관련 쿼리
```

### 4. 유틸리티 모듈
```
utils/
├── __init__.py
├── customer_id_generator.py   # 고객 ID 생성
├── customer_matcher.py       # 동일 고객 확인
├── data_validator.py         # 데이터 검증
└── formatters.py             # 데이터 포맷팅
```

## 마이그레이션 계획

### 단계 1: 채팅 시뮬레이터 모듈화
1. `_chat_simulator.py` → `chat_simulator/main.py`
2. 각 하위 모듈을 `chat_simulator/` 디렉토리로 이동
3. import 경로 수정

### 단계 2: 고객 데이터 모듈 생성
1. `customer_data/` 디렉토리 생성
2. 고객 데이터 관련 기능 구현
3. 데이터베이스 연동

### 단계 3: 데이터베이스 모듈 생성
1. `database/` 디렉토리 생성
2. 모델 정의 및 쿼리 함수 구현
3. 기존 파일 시스템 기반 저장소와 연동

### 단계 4: 유틸리티 모듈 생성
1. 공통 유틸리티 함수 분리
2. 재사용 가능한 모듈로 구성

## 파일별 역할 정의

### chat_simulator/main.py
```python
"""
채팅 시뮬레이터 메인 진입점
- 일일 통계 표시
- 각 단계별 모듈 호출
"""
def render_chat_simulator():
    # 통계 표시
    # 각 단계별 모듈 호출
    pass
```

### customer_data/form.py
```python
"""
고객 데이터 입력/수정 폼
- 고객 기본 정보 입력
- 데이터 검증
- 데이터베이스 저장
"""
def render_customer_form():
    # 폼 렌더링
    # 데이터 저장
    pass
```

### database/customers.py
```python
"""
고객 관련 데이터베이스 쿼리
- 고객 생성/조회/수정/삭제
- 동일 고객 찾기
"""
def create_customer(data):
    pass

def get_customer(customer_id):
    pass

def find_similar_customers(email, phone):
    pass
```

## Import 경로 변경 예시

### 변경 전
```python
from _pages._chat_history import render_chat_history
from _pages._chat_initial_query import render_initial_query
```

### 변경 후
```python
from _pages.chat_simulator.history import render_chat_history
from _pages.chat_simulator.initial_query import render_initial_query
```

## 장점

1. **모듈화**: 기능별로 명확히 분리
2. **재사용성**: 공통 기능을 모듈로 분리하여 재사용
3. **유지보수성**: 각 기능을 독립적으로 수정 가능
4. **확장성**: 새로운 기능 추가가 용이
5. **테스트 용이성**: 각 모듈을 독립적으로 테스트 가능

## 주의사항

1. **순환 참조 방지**: 모듈 간 의존성 관리
2. **상태 관리**: Streamlit session_state 공유
3. **에러 처리**: 각 모듈의 에러 처리 일관성 유지
4. **성능**: 불필요한 import 최소화


