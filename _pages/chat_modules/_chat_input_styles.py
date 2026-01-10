# ========================================
# _pages/chat_modules/_chat_input_styles.py
# 채팅 입력 UI 스타일 (CSS)
# ========================================

def get_chat_input_styles():
    """채팅 입력 UI 스타일 반환"""
    return """
    <style>
    .kakao-chat-input {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 24px;
        padding: 12px 20px;
        font-size: 15px;
        min-height: 50px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .kakao-chat-input:focus {
        outline: none;
        border-color: #FEE500;
        box-shadow: 0 2px 8px rgba(254, 229, 0, 0.3);
    }
    .stChatInputContainer,
    div[data-testid="stChatInputContainer"],
    div[data-baseweb="input"] {
        position: relative !important;
    }
    
    .chat-input-attach-btn {
        position: absolute !important;
        left: 10px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-size: 22px !important;
        font-weight: bold !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.2s ease !important;
        z-index: 1000 !important;
        line-height: 1 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .chat-input-attach-btn:hover {
        transform: translateY(-50%) scale(1.1) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
    }
    
    .chat-input-attach-btn:active {
        transform: translateY(-50%) scale(0.95) !important;
    }
    
    textarea[data-testid="stChatInputTextArea"],
    textarea[data-baseweb="textarea"],
    textarea.stChatInputTextArea {
        padding-left: 48px !important;
    }
    
    div[data-testid="stChatInputContainer"],
    div[data-baseweb="input"] {
        position: relative !important;
    }
    
    div[data-baseweb="input"] > div {
        position: relative !important;
    }
    
    button[data-testid*="btn_add_attachment_unified_hidden"] {
        display: none !important;
    }
    
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes fadeOut {
        to {
            opacity: 0;
            transform: translateY(-10px);
            height: 0;
            margin: 0;
            padding: 0;
        }
    }
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    </style>
    """
