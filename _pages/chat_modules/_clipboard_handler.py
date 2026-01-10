# ========================================
# _pages/chat_modules/_clipboard_handler.py
# 클립보드 붙여넣기 핸들러
# ========================================

import streamlit as st
import base64
import io
from datetime import datetime


def render_clipboard_paste_handler(L, current_lang):
    """Ctrl+V로 이미지/동영상 붙여넣기 핸들러 렌더링"""
    # 클립보드 이미지/동영상 처리 상태 초기화
    if "clipboard_paste_processing" not in st.session_state:
        st.session_state.clipboard_paste_processing = False
    if "clipboard_paste_data" not in st.session_state:
        st.session_state.clipboard_paste_data = []
    if "pending_clipboard_files" not in st.session_state:
        st.session_state.pending_clipboard_files = []
    
    return """
    <script>
    (function() {
        var pasteHandlerAdded = false;
        var clipboardDataQueue = [];
        
        function addPasteHandler() {
            if (pasteHandlerAdded) return;
            
            // Document 레벨에서 paste 이벤트 리스너 추가 (모든 곳에서 감지)
            document.addEventListener('paste', handlePaste, true);
            pasteHandlerAdded = true;
            console.log('✅ Ctrl+V 붙여넣기 핸들러가 추가되었습니다 (document 레벨).');
        }
        
        function handlePaste(e) {
            var items = e.clipboardData.items;
            if (!items) return;
            
            var imageItem = null;
            var videoItem = null;
            
            // 클립보드에서 이미지 또는 동영상 찾기
            for (var i = 0; i < items.length; i++) {
                var item = items[i];
                if (item.type.indexOf('image') !== -1) {
                    imageItem = item;
                    break;
                } else if (item.type.indexOf('video') !== -1) {
                    videoItem = item;
                    break;
                }
            }
            
            if (imageItem || videoItem) {
                e.preventDefault();
                e.stopPropagation();
                
                var item = imageItem || videoItem;
                var file = item.getAsFile();
                
                if (file) {
                    var reader = new FileReader();
                    reader.onload = function(event) {
                        var base64Data = event.target.result;
                        var mimeType = file.type;
                        
                        // 파일 확장자 결정
                        var fileExtension = 'png';
                        if (mimeType.indexOf('jpeg') !== -1 || mimeType.indexOf('jpg') !== -1) {
                            fileExtension = 'jpg';
                        } else if (mimeType.indexOf('gif') !== -1) {
                            fileExtension = 'gif';
                        } else if (mimeType.indexOf('webp') !== -1) {
                            fileExtension = 'webp';
                        } else if (mimeType.indexOf('mp4') !== -1) {
                            fileExtension = 'mp4';
                        } else if (mimeType.indexOf('webm') !== -1) {
                            fileExtension = 'webm';
                        } else if (mimeType.indexOf('mov') !== -1) {
                            fileExtension = 'mov';
                        }
                        
                        var fileName = 'pasted_' + Date.now() + '.' + fileExtension;
                        
                        // 알림 표시
                        showPasteNotification(fileName, mimeType);
                        
                        // 미리보기 표시
                        displayClipboardPreview(base64Data, fileName, mimeType);
                        
                        // LocalStorage에 데이터 저장 (Streamlit rerun 사이에 유지)
                        saveToLocalStorage(base64Data.split(',')[1], mimeType, fileName, file.size);
                        
                        // 파일 업로더 자동 열기
                        setTimeout(function() {
                            openFileUploader();
                        }, 300);
                    };
                    
                    reader.readAsDataURL(file);
                }
            }
        }
        
        function saveToLocalStorage(base64Data, mimeType, fileName, fileSize) {
            // LocalStorage에 저장 (Streamlit rerun 사이에 유지)
            var data = {
                data: base64Data,
                mimeType: mimeType,
                fileName: fileName,
                fileSize: fileSize,
                timestamp: new Date().toISOString()
            };
            
            if (!localStorage.getItem('clipboardPasteQueue')) {
                localStorage.setItem('clipboardPasteQueue', '[]');
            }
            
            var queue = JSON.parse(localStorage.getItem('clipboardPasteQueue'));
            queue.push(data);
            localStorage.setItem('clipboardPasteQueue', JSON.stringify(queue));
            
            console.log('📎 클립보드 데이터를 LocalStorage에 저장:', fileName);
            
            // 파일 업로더가 열려 있으면 바로 처리
            setTimeout(function() {
                processClipboardQueue();
            }, 500);
        }
        
        function processClipboardQueue() {
            // LocalStorage에서 데이터 읽어서 파일 업로더에 추가
            var queue = localStorage.getItem('clipboardPasteQueue');
            if (queue) {
                try {
                    var items = JSON.parse(queue);
                    if (items.length > 0) {
                        console.log('📎 클립보드 파일 처리 중:', items.length, '개');
                        
                        // 파일 업로더 찾기
                        var fileUploader = document.querySelector('input[type="file"][data-testid*="stFileUploader"]')
                            || document.querySelector('input[type="file"]');
                        
                        if (fileUploader) {
                            var dataTransfer = new DataTransfer();
                            
                            items.forEach(function(item) {
                                // base64를 Blob으로 변환
                                var byteCharacters = atob(item.data);
                                var byteNumbers = new Array(byteCharacters.length);
                                for (var i = 0; i < byteCharacters.length; i++) {
                                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                                }
                                var byteArray = new Uint8Array(byteNumbers);
                                var blob = new Blob([byteArray], {type: item.mimeType});
                                
                                // File 객체 생성
                                var file = new File([blob], item.fileName, {type: item.mimeType});
                                dataTransfer.items.add(file);
                                
                                console.log('✅ 파일 추가:', item.fileName);
                            });
                            
                            // 파일 업로더에 파일 설정
                            fileUploader.files = dataTransfer.files;
                            
                            // Change 이벤트 트리거하여 Streamlit이 파일을 인식하도록
                            var changeEvent = new Event('change', { bubbles: true });
                            fileUploader.dispatchEvent(changeEvent);
                            
                            // Input 이벤트도 트리거
                            var inputEvent = new Event('input', { bubbles: true });
                            fileUploader.dispatchEvent(inputEvent);
                            
                            console.log('✅ 파일이 업로더에 추가되었습니다:', items.length, '개');
                            
                            // LocalStorage 초기화
                            localStorage.removeItem('clipboardPasteQueue');
                        } else {
                            console.log('⚠️ 파일 업로더를 찾을 수 없습니다.');
                        }
                    }
                } catch (e) {
                    console.error('클립보드 데이터 처리 오류:', e);
                }
            }
        }
        
        function openFileUploader() {
            // 파일 업로더 버튼 찾기 및 클릭
            var uploadButton = document.querySelector('button[data-testid*="btn_add_attachment_unified_hidden"]')
                || document.querySelector('button[aria-label*="파일"]')
                || document.querySelector('button[aria-label*="첨부"]');
            
            if (uploadButton) {
                uploadButton.click();
                console.log('📎 파일 업로더가 열렸습니다.');
                
                // 업로더가 열린 후 파일 처리
                setTimeout(function() {
                    processClipboardQueue();
                }, 500);
            } else {
                console.log('⚠️ 파일 업로더 버튼을 찾을 수 없습니다.');
                // 버튼이 없어도 파일 처리 시도
                setTimeout(function() {
                    processClipboardQueue();
                }, 1000);
            }
        }
        
        function showPasteNotification(fileName, mimeType) {
            // 기존 알림 제거
            var existing = document.getElementById('paste-notification');
            if (existing) {
                existing.remove();
            }
            
            var notification = document.createElement('div');
            notification.id = 'paste-notification';
            notification.style.cssText = 'position: fixed; top: 20px; right: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 10000; font-size: 14px; font-family: system-ui, -apple-system, sans-serif;';
            
            var icon = mimeType.indexOf('image') !== -1 ? '🖼️' : '🎥';
            notification.innerHTML = '<span style="margin-right: 8px;">' + icon + '</span><strong>' + fileName + '</strong> 가 붙여넣어졌습니다.';
            
            document.body.appendChild(notification);
            
            setTimeout(function() {
                notification.style.transition = 'opacity 0.3s ease-out';
                notification.style.opacity = '0';
                setTimeout(function() {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }, 3000);
        }
        
        function displayClipboardPreview(base64Data, fileName, mimeType) {
            var existingPreview = document.getElementById('clipboard-paste-preview');
            if (existingPreview) {
                existingPreview.remove();
            }
            
            var chatContainer = document.querySelector('[data-testid="stVerticalBlock"]')
                || document.querySelector('.main') || document.body;
            
            var previewContainer = document.createElement('div');
            previewContainer.id = 'clipboard-paste-preview';
            previewContainer.style.cssText = 'margin: 10px 0; padding: 12px; background: #f0f2f6; border-radius: 8px; border: 2px dashed #667eea; position: relative; max-width: 100%;';
            
            var previewContent = '';
            if (mimeType.indexOf('image') !== -1) {
                previewContent = '<div style="margin-bottom: 8px;"><strong style="color: #667eea;">📎 붙여넣은 이미지:</strong></div>' +
                    '<img src="' + base64Data + '" style="max-width: 100%; max-height: 300px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />' +
                    '<div style="margin-top: 8px; font-size: 12px; color: #666;">파일명: ' + fileName + '</div>';
            } else if (mimeType.indexOf('video') !== -1) {
                previewContent = '<div style="margin-bottom: 8px;"><strong style="color: #667eea;">📎 붙여넣은 동영상:</strong></div>' +
                    '<video src="' + base64Data + '" controls style="max-width: 100%; max-height: 300px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></video>' +
                    '<div style="margin-top: 8px; font-size: 12px; color: #666;">파일명: ' + fileName + '</div>';
            }
            
            var closeButton = document.createElement('button');
            closeButton.innerHTML = '✕';
            closeButton.style.cssText = 'position: absolute; top: 8px; right: 8px; background: #ff4444; color: white; border: none; border-radius: 50%; width: 24px; height: 24px; cursor: pointer; font-size: 14px; line-height: 1; display: flex; align-items: center; justify-content: center;';
            closeButton.onclick = function() {
                previewContainer.remove();
            };
            
            previewContainer.innerHTML = previewContent;
            previewContainer.appendChild(closeButton);
            
            var chatInput = document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (chatInput && chatInput.parentElement) {
                chatInput.parentElement.insertBefore(previewContainer, chatInput);
            } else {
                chatContainer.insertBefore(previewContainer, chatContainer.firstChild);
            }
        }
        
        // 페이지 로드 시 핸들러 추가
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(addPasteHandler, 500);
            });
        } else {
            setTimeout(addPasteHandler, 500);
        }
        
        // 동적으로 추가되는 요소를 감지하여 핸들러 추가
        var observer = new MutationObserver(function(mutations) {
            addPasteHandler();
        });
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // 여러 번 시도 (Streamlit이 동적으로 요소를 추가할 수 있음)
        var intervals = [100, 300, 500, 1000, 2000];
        intervals.forEach(function(delay) {
            setTimeout(addPasteHandler, delay);
        });
        
        // processClipboardQueue 함수를 전역으로 노출 (다른 스크립트에서 호출 가능)
        window.processClipboardQueue = processClipboardQueue;
    })();
    </script>
    """


def handle_clipboard_processing(L):
    """클립보드 데이터 처리"""
    # JavaScript에서 LocalStorage를 통해 자동으로 처리되므로,
    # 여기서는 파일 업로더가 열려 있을 때만 처리 확인 (rerun 최소화)
    # JavaScript에서 자동으로 처리하므로 Python 레벨에서는 불필요한 처리가 없음
    pass


def process_pending_clipboard_data():
    """JavaScript에서 LocalStorage에 저장한 클립보드 데이터 처리"""
    # JavaScript에서 LocalStorage에 저장한 데이터를 읽어서 처리
    # processClipboardQueue 함수가 JavaScript에서 자동으로 실행됨
    # 여기서는 추가 처리가 필요 없음
    pass
