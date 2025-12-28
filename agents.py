"""에이전트 관리 모듈"""

# 사용 가능한 에이전트 목록
AVAILABLE_AGENTS = [
    {'name': '김민수', 'skill': '주문/결제 전문가', 'status': 'available', 'rating': 4.8, 'calls_today': 12},
    {'name': '이지은', 'skill': '환불/취소 전문가', 'status': 'available', 'rating': 4.9, 'calls_today': 15},
    {'name': '박준호', 'skill': '기술 지원 전문가', 'status': 'available', 'rating': 4.7, 'calls_today': 8},
    {'name': '최수진', 'skill': '일반 문의 전문가', 'status': 'available', 'rating': 4.6, 'calls_today': 20},
    {'name': '정태영', 'skill': 'VIP 고객 전문가', 'status': 'available', 'rating': 5.0, 'calls_today': 5},
]

def find_best_agent(customer_insight, available_agents):
    """고급 라우팅: 고객 인사이트를 기반으로 최적의 에이전트 찾기"""
    available = [a for a in available_agents if a['status'] == 'available']
    
    if not available:
        return None
    
    # 의도와 스킬 매칭
    intent = customer_insight.get('intent', '일반 문의')
    skill_mapping = {
        '주문': '주문/결제 전문가',
        '환불/취소': '환불/취소 전문가',
        '배송 문의': '일반 문의 전문가',
        '문의': '일반 문의 전문가',
        '예약': '일반 문의 전문가',
    }
    
    required_skill = skill_mapping.get(intent, '일반 문의 전문가')
    
    # 스킬이 일치하는 에이전트 찾기
    matching_agents = [a for a in available if required_skill in a['skill']]
    
    if matching_agents:
        # 평점이 높은 순으로 정렬
        matching_agents.sort(key=lambda x: x['rating'], reverse=True)
        return matching_agents[0]
    else:
        # 매칭되는 에이전트가 없으면 가장 평점이 높은 에이전트 선택
        available.sort(key=lambda x: x['rating'], reverse=True)
        return available[0]

def find_agent_by_skill(agent_skill, available_agents):
    """스킬 기반 에이전트 찾기 (아웃바운드용)"""
    if agent_skill == "자동 할당":
        available = [a for a in available_agents if a['status'] == 'available']
    else:
        skill_keyword = agent_skill.replace(" 전문가", "")
        available = [a for a in available_agents 
                    if a['status'] == 'available' and skill_keyword in a['skill']]
    
    if available:
        # 가장 높은 평점의 에이전트 선택
        return max(available, key=lambda x: x['rating'])
    return None

