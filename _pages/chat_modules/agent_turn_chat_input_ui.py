# ========================================
# _pages/chat_modules/agent_turn_chat_input_ui.py
# 에이전트 턴 - 채팅 입력 UI (카카오톡 스타일)
# ========================================

import streamlit as st
import json

def render_chat_input_ui(L, current_lang, perspective):
    """채팅 입력 UI 렌더링 (카카오톡 스타일)"""
    # 응대 초안이 있으면 자동으로 입력창에 표시
    initial_value = ""
    if st.session_state.get("auto_generated_draft_text") and st.session_state.auto_generated_draft_text:
        initial_value = st.session_state.auto_generated_draft_text
    elif st.session_state.get("agent_response_area_text") and st.session_state.agent_response_area_text:
        initial_value = st.session_state.agent_response_area_text
    
    placeholder_text = L.get("agent_response_placeholder", "고객에게 응답하세요...")
    
    # ⭐ 상담원 모드일 때만 카카오톡 스타일 채팅 입력창 및 파일 첨부 버튼 표시
    if perspective == "AGENT":
        st.markdown("""
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
        </style>
        """, unsafe_allow_html=True)
    
    # Streamlit의 chat_input 사용 (자동 업데이트 지원)
    agent_response_input = None
    if perspective == "AGENT":
        agent_response_input = st.chat_input(placeholder_text, key="agent_chat_input_main")
    
    # ⭐ 응대 초안이 있으면 입력창에 자동으로 채우기
    if perspective == "AGENT" and agent_response_input is not None and st.session_state.get("auto_generated_draft_text") and st.session_state.auto_generated_draft_text:
        draft_text = st.session_state.auto_generated_draft_text
        draft_text_json = json.dumps(draft_text)
        
        st.markdown(f"""
        <script>
        (function() {{
            var draftText = {draft_text_json};
            var filled = false;
            var fillAttempts = 0;
            var maxAttempts = 30;
            
            function fillChatInput() {{
                fillAttempts++;
                
                var selectors = [
                    'textarea[data-testid="stChatInputTextArea"]',
                    'textarea[aria-label*="고객"]',
                    'textarea[placeholder*="고객"]',
                    'textarea.stChatInputTextArea',
                    'textarea[placeholder*="응답"]',
                    'textarea'
                ];
                
                var chatInput = null;
                for (var i = 0; i < selectors.length; i++) {{
                    var elements = document.querySelectorAll(selectors[i]);
                    for (var j = 0; j < elements.length; j++) {{
                        if (elements[j] && elements[j].offsetParent !== null) {{
                            chatInput = elements[j];
                            break;
                        }}
                    }}
                    if (chatInput) break;
                }}
                
                if (chatInput && !filled) {{
                    var currentValue = chatInput.value || '';
                    if (!currentValue.trim() || currentValue.trim() !== draftText.trim()) {{
                        chatInput.value = draftText;
                        chatInput.focus();
                        
                        var events = ['input', 'change', 'keyup', 'keydown'];
                        events.forEach(function(eventType) {{
                            var event = new Event(eventType, {{ bubbles: true, cancelable: true }});
                            chatInput.dispatchEvent(event);
                        }});
                        
                        if (chatInput._valueTracker) {{
                            chatInput._valueTracker.setValue('');
                            chatInput._valueTracker.setValue(draftText);
                        }}
                        
                        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                        nativeInputValueSetter.call(chatInput, draftText);
                        var inputEvent = new Event('input', {{ bubbles: true }});
                        chatInput.dispatchEvent(inputEvent);
                        
                        filled = true;
                        console.log('✅ 응대 초안이 입력창에 자동으로 채워졌습니다.');
                        showDraftNotification();
                    }}
                }} else if (!filled && fillAttempts < maxAttempts) {{
                    setTimeout(fillChatInput, 100);
                }}
            }}
            
            function showDraftNotification() {{
                var notification = document.getElementById('draft-notification');
                if (notification) {{
                    notification.style.display = 'block';
                    notification.style.animation = 'slideInDown 0.3s ease-out';
                    setTimeout(function() {{
                        if (notification) {{
                            notification.style.animation = 'fadeOut 0.3s ease-in forwards';
                            setTimeout(function() {{
                                if (notification) notification.style.display = 'none';
                            }}, 300);
                        }}
                    }}, 5000);
                }}
            }}
            
            fillChatInput();
            
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', fillChatInput);
            }}
            
            var observer = new MutationObserver(function(mutations) {{
                if (!filled) {{
                    fillChatInput();
                }}
            }});
            
            observer.observe(document.body, {{
                childList: true,
                subtree: true
            }});
            
            var intervals = [50, 100, 150, 200, 300, 500, 800, 1200, 2000, 3000];
            intervals.forEach(function(delay) {{
                setTimeout(function() {{
                    if (!filled) fillChatInput();
                }}, delay);
            }});
        }})();
        </script>
        
        <div id="draft-notification" style="display: none; background: rgba(33, 150, 243, 0.08); 
                    padding: 8px 12px; 
                    border-radius: 8px; 
                    margin-bottom: 8px;
                    border-left: 3px solid #2196F3;
                    font-size: 0.85em;
                    color: #1976D2;">
            <span style="display: inline-flex; align-items: center; gap: 6px;">
                <span style="font-size: 1.1em;">✨</span>
                <span id="draft-notification-text"></span>
            </span>
        </div>
        <script>
        (function() {{
            var lang = '{current_lang}';
            var notificationText = '';
            if (lang === 'ko') {{
                notificationText = '응대 초안이 자동 생성되어 입력창에 채워졌습니다';
            }} else if (lang === 'en') {{
                notificationText = 'Response draft has been automatically generated and filled in the input field';
            }} else if (lang === 'ja') {{
                notificationText = '対応草案が自動生成され、入力欄に記入されました';
            }} else {{
                notificationText = '응대 초안이 자동 생성되어 입력창에 채워졌습니다';
            }}
            var notificationElement = document.getElementById('draft-notification-text');
            if (notificationElement) {{
                notificationElement.textContent = notificationText;
            }}
        }})();
        </script>
        <style>
        @keyframes slideInDown {{
            from {{
                opacity: 0;
                transform: translateY(-10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        @keyframes fadeOut {{
            to {{
                opacity: 0;
                transform: translateY(-10px);
                height: 0;
                margin: 0;
                padding: 0;
            }}
        }}
        </style>
        """, unsafe_allow_html=True)

    # ⭐ 파일 첨부 버튼을 입력창 안쪽에 배치 (카카오톡 스타일)
    if perspective == "AGENT":
        st.markdown("""
        <style>
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
        </style>
        <script>
        (function() {{
            function addAttachButton() {{
                var existingBtn = document.getElementById('chat-attach-btn');
                if (existingBtn) {{
                    existingBtn.remove();
                }}
                
                var chatInput = document.querySelector('textarea[data-testid="stChatInputTextArea"]')
                    || document.querySelector('textarea[data-baseweb="textarea"]')
                    || document.querySelector('textarea.stChatInputTextArea');
                
                if (chatInput) {{
                    var container = chatInput.closest('[data-testid="stChatInputContainer"]') 
                        || chatInput.closest('[data-baseweb="input"]')
                        || chatInput.closest('.stChatInputContainer')
                        || chatInput.parentElement.parentElement
                        || chatInput.parentElement;
                    
                    var attachBtn = document.createElement('button');
                    attachBtn.id = 'chat-attach-btn';
                    attachBtn.className = 'chat-input-attach-btn';
                    attachBtn.innerHTML = '+';
                    attachBtn.title = '파일 첨부';
                    attachBtn.type = 'button';
                    attachBtn.setAttribute('aria-label', '파일 첨부');
                    
                    attachBtn.addEventListener('click', function(e) {{
                        e.preventDefault();
                        e.stopPropagation();
                        
                        var hiddenBtn = document.querySelector('button[data-testid*="btn_add_attachment_unified_hidden"]')
                            || document.querySelector('button[data-baseweb="button"][aria-label*="파일"]')
                            || Array.from(document.querySelectorAll('button')).find(function(btn) {{
                                return btn.textContent.includes('➕') || btn.textContent.includes('파일');
                            }});
                        
                        if (hiddenBtn) {{
                            hiddenBtn.click();
                            var clickEvent = new MouseEvent('click', {{
                                bubbles: true,
                                cancelable: true,
                                view: window
                            }});
                            hiddenBtn.dispatchEvent(clickEvent);
                        }}
                    }});
                    
                    if (container) {{
                        if (container.style) {{
                            container.style.position = 'relative';
                        }}
                        var oldBtn = container.querySelector('#chat-attach-btn');
                        if (oldBtn) {{
                            oldBtn.remove();
                        }}
                        container.appendChild(attachBtn);
                    }} else {{
                        var parent = chatInput.parentElement;
                        if (parent) {{
                            parent.style.position = 'relative';
                            var oldBtn = parent.querySelector('#chat-attach-btn');
                            if (oldBtn) {{
                                oldBtn.remove();
                            }}
                            parent.appendChild(attachBtn);
                        }}
                    }}
                }}
            }}
            
            addAttachButton();
            
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', addAttachButton);
            }}
            
            var observer = new MutationObserver(function(mutations) {{
                addAttachButton();
            }});
            
            observer.observe(document.body, {{
                childList: true,
                subtree: true
            }});
            
            var intervals = [50, 100, 200, 300, 500, 800, 1200];
            intervals.forEach(function(delay) {{
                setTimeout(addAttachButton, delay);
            }});
        }})();
        </script>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <style>
        button[data-testid*="btn_add_attachment_unified_hidden"] {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button(
                "➕",
                key="btn_add_attachment_unified_hidden",
                help=L.get("button_add_attachment", "➕ 파일 첨부"),
                use_container_width=False,
                type="secondary"):
            st.session_state.show_agent_file_uploader = True
    
    return agent_response_input

