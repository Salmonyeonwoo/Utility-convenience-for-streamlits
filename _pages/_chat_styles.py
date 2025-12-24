# ========================================
# _pages/_chat_styles.py
# 채팅 시뮬레이터 - CSS 스타일 정의
# ========================================

def get_chat_styles():
    """채팅 UI를 위한 CSS 스타일 반환"""
    return """
    <style>
    /* 채팅 컨테이너 배경 */
    .main .block-container {
        background-color: #F5F5F5;
        padding-top: 1rem;
    }
    
    /* 메시지 말풍선 기본 스타일 */
    .message-bubble {
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.4;
        font-size: 15px;
        position: relative;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* 고객 메시지 (오른쪽, 노란색) - 카카오톡 스타일 */
    .message-customer {
        background: linear-gradient(135deg, #FEE500 0%, #FFD700 100%);
        margin-left: auto;
        margin-right: 0;
        text-align: right;
        box-shadow: 0 2px 8px rgba(254, 229, 0, 0.3);
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    .message-customer::after {
        content: '';
        position: absolute;
        right: -8px;
        bottom: 12px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 8px 0 8px 8px;
        border-color: transparent transparent transparent #FEE500;
    }
    
    
    /* 에이전트 메시지 (왼쪽, 흰색) - 카카오톡 스타일 */
    .message-agent {
        background: #FFFFFF;
        margin-right: auto;
        margin-left: 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .message-agent::before {
        content: '';
        position: absolute;
        left: -8px;
        bottom: 12px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 8px 8px 8px 0;
        border-color: transparent #FFFFFF transparent transparent;
    }
    
    /* Supervisor 메시지 (중앙, 연한 초록색) */
    .message-supervisor {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        margin: 12px auto;
        max-width: 90%;
        font-size: 0.9em;
        border: 1px solid rgba(76, 175, 80, 0.2);
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.1);
    }
    
    /* 아이콘 버튼 스타일 */
    .icon-button {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 8px;
        font-size: 1.2em;
        cursor: pointer;
        padding: 6px 8px;
        margin: 0 2px;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .icon-button:hover {
        background: rgba(255, 255, 255, 1);
        transform: scale(1.05);
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .customer-message-wrapper {
        animation: slideInRight 0.4s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    </style>
    """

