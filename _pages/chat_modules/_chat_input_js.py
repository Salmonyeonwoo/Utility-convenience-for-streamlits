# ========================================
# _pages/chat_modules/_chat_input_js.py
# 채팅 입력 UI JavaScript 핸들러
# ========================================

import streamlit as st
import json


def render_draft_fill_script(draft_text, current_lang):
    """응대 초안 자동 채우기 스크립트 렌더링"""
    draft_text_json = json.dumps(draft_text)
    
    notification_texts = {
        'ko': '응대 초안이 자동 생성되어 입력창에 채워졌습니다',
        'en': 'Response draft has been automatically generated and filled in the input field',
        'ja': '対応草案が自動生成され、入力欄に記入されました'
    }
    notification_text = notification_texts.get(current_lang, notification_texts['ko'])
    
    return f"""
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
            <span id="draft-notification-text">{notification_text}</span>
        </span>
    </div>
    """


def render_attach_button_script():
    """파일 첨부 버튼 스크립트 렌더링"""
    return """
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
    """
