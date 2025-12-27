"""
모던 채팅 페이지 (이미지 참고 UI)
3단 레이아웃: 왼쪽 고객 리스트, 가운데 채팅 영역, 오른쪽 고객 정보
"""
import streamlit as st
import json
from datetime import datetime
from data_manager import load_customers, load_chats, save_chats, save_dashboard_stats, load_dashboard_stats
from ai_services import get_ai_response
from config import get_api_key


def render_modern_chat_page():
    """모던 3단 레이아웃 채팅 페이지 렌더링"""
    customers = load_customers()
    chats = load_chats()
    
    # CSS 스타일 적용
    st.markdown("""
    <style>
        /* 고객 리스트 스타일 */
        .customer-list-item {
            padding: 12px;
            margin: 4px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .customer-list-item:hover {
            background-color: #f0f2f6;
        }
        .customer-list-item.selected {
            background-color: #1f77b4;
            color: white;
        }
        .unread-badge {
            background-color: #ff4444;
            color: white;
            border-radius: 50%;
            padding: 2px 8px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 8px;
        }
        
        /* 채팅 메시지 스타일 */
        .chat-message-operator {
            background-color: #e3f2fd;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            margin: 8px 0;
            margin-left: auto;
            max-width: 70%;
            text-align: right;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .chat-message-customer {
            background-color: #f5f5f5;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 0;
            margin-right: auto;
            max-width: 70%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .chat-message-header {
            font-weight: bold;
            margin-bottom: 4px;
            font-size: 14px;
        }
        .chat-message-time {
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }
        
        /* 고객 정보 패널 스타일 */
        .customer-info-section {
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        .customer-info-label {
            font-weight: bold;
            color: #666;
            font-size: 12px;
            margin-bottom: 4px;
        }
        .customer-info-value {
            color: #333;
            font-size: 14px;
        }
        
        /* 채팅 입력 영역 */
        .chat-input-container {
            display: flex;
            gap: 8px;
            padding: 12px;
            background-color: #f9f9f9;
            border-top: 1px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 3단 레이아웃
    col1, col2, col3 = st.columns([0.8, 2, 1.2], gap="small")
    
    # 왼쪽: 고객 리스트 사이드바
    with col1:
        st.markdown("### 💬 CUSTOMERS")
        st.markdown("---")
        
        # 고객 읽지 않은 메시지 수 계산
        unread_counts = {}
        for customer in customers:
            customer_id = customer['customer_id']
            if customer_id in chats:
                customer_messages = [msg for msg in chats[customer_id] if msg['sender'] == 'customer']
                # 마지막 읽은 메시지 이후의 메시지만 카운트
                unread_count = len(customer_messages)
                unread_counts[customer_id] = unread_count
        
        # CUSTOMERS 섹션
        for customer in customers[:10]:  # 최대 10명만 표시
            customer_id = customer['customer_id']
            is_selected = st.session_state.selected_customer_id == customer_id
            
            # 고객 버튼
            button_text = f"👤 {customer['customer_name']}"
            if st.button(
                button_text,
                key=f"customer_btn_{customer_id}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_customer_id = customer_id
            
            # 읽지 않은 메시지 배지
            if customer_id in unread_counts and unread_counts[customer_id] > 0:
                st.markdown(f"<span class='unread-badge'>{unread_counts[customer_id]}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### OTHER CUSTOMERS")
        
        # 기타 고객들
        for customer in customers[10:]:
            customer_id = customer['customer_id']
            is_selected = st.session_state.selected_customer_id == customer_id
            
            if st.button(
                f"👤 {customer['customer_name']}",
                key=f"customer_btn_{customer_id}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_customer_id = customer_id
            
            if customer_id in unread_counts and unread_counts[customer_id] > 0:
                st.markdown(f"<span class='unread-badge'>{unread_counts[customer_id]}</span>", unsafe_allow_html=True)
    
    # 가운데: 채팅 영역
    with col2:
        if st.session_state.selected_customer_id:
            selected_customer = next(
                (c for c in customers if c['customer_id'] == st.session_state.selected_customer_id),
                None
            )
            
            if selected_customer:
                # 채팅 헤더
                st.markdown(f"### 💬 {selected_customer['customer_name']}님과의 대화")
                
                customer_id = selected_customer['customer_id']
                if customer_id not in chats:
                    chats[customer_id] = []
                
                current_chats = chats[customer_id]
                last_msg_id = st.session_state.last_message_id.get(customer_id, "")
                
                # AI 응답 생성
                if current_chats:
                    last_msg = current_chats[-1]
                    current_last_id = last_msg.get('message_id', '')
                    api_key_auto = get_api_key("openai") or get_api_key("gemini")
                    
                    if (last_msg['sender'] == 'customer' and 
                        current_last_id != last_msg_id and 
                        api_key_auto and
                        f'ai_processing_{customer_id}' not in st.session_state):
                        st.session_state[f'ai_processing_{customer_id}'] = True
                        try:
                            ai_response = get_ai_response(last_msg['message'], selected_customer, current_chats)
                            st.session_state.ai_suggestion = {
                                'customer_id': customer_id,
                                'message': ai_response,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        except Exception as e:
                            st.session_state.ai_suggestion = {
                                'customer_id': customer_id,
                                'message': f"오류: {str(e)}",
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        finally:
                            st.session_state[f'ai_processing_{customer_id}'] = False
                        st.session_state.last_message_id[customer_id] = current_last_id
                
                # 채팅 메시지 컨테이너
                chat_container = st.container(height=500)
                with chat_container:
                    for msg in current_chats:
                        sender_class = "chat-message-operator" if msg['sender'] == 'operator' else "chat-message-customer"
                        sender_name = msg.get('sender_name', '알 수 없음')
                        message_time = msg.get('timestamp', '')
                        
                        st.markdown(f"""
                        <div class="{sender_class}">
                            <div class="chat-message-header">{sender_name}</div>
                            <div>{msg['message']}</div>
                            <div class="chat-message-time">{message_time}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # AI 제안 표시
                    if (current_chats and 
                        current_chats[-1]['sender'] == 'customer' and
                        st.session_state.get('ai_suggestion', {}).get('customer_id') == customer_id):
                        ai_suggestion = st.session_state.ai_suggestion
                        st.markdown(f"""
                        <div class="chat-message-operator" style="background-color: #fff3cd; border-left: 4px solid #ffc107;">
                            <div class="chat-message-header">🤖 AI 제안 응답</div>
                            <div>{ai_suggestion['message']}</div>
                            <div class="chat-message-time">{ai_suggestion['timestamp']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("✅ 사용", key=f"use_ai_{customer_id}", use_container_width=True):
                            new_message = {
                                "message_id": f"MSG{len(current_chats) + 1:03d}",
                                "sender": "operator",
                                "sender_name": "상담원",
                                "message": ai_suggestion['message'],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            chats[customer_id].append(new_message)
                            save_chats(chats)
                            st.session_state.ai_suggestion = {}
                    
                    if st.session_state.get(f'ai_processing_{customer_id}', False):
                        st.info("🤖 AI가 응답을 생성하는 중...")
                
                st.markdown("---")
                
                # 채팅 입력 영역
                input_col1, input_col2 = st.columns([5, 1])
                with input_col1:
                    chat_input = st.text_input(
                        "메시지 입력",
                        key=f"chat_input_{customer_id}",
                        placeholder="메시지를 입력하세요...",
                        label_visibility="collapsed"
                    )
                with input_col2:
                    if st.button("전송", type="primary", use_container_width=True, key=f"send_{customer_id}"):
                        if chat_input:
                            new_message = {
                                "message_id": f"MSG{len(chats[customer_id]) + 1:03d}",
                                "sender": "operator",
                                "sender_name": "상담원",
                                "message": chat_input,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            chats[customer_id].append(new_message)
                            save_chats(chats)
                            stats = load_dashboard_stats()
                            stats['today_cases'] += 1
                            save_dashboard_stats(stats)
                            
                            # 입력 필드 초기화를 위해 세션 상태 업데이트
                            st.session_state[f'chat_input_{customer_id}'] = ""
            else:
                st.info("고객을 선택해주세요.")
        else:
            st.info("왼쪽에서 고객을 선택하여 채팅을 시작하세요.")
    
    # 오른쪽: 고객 정보 패널
    with col3:
        if st.session_state.selected_customer_id:
            selected_customer = next(
                (c for c in customers if c['customer_id'] == st.session_state.selected_customer_id),
                None
            )
            if selected_customer:
                st.markdown("### 👤 고객 정보")
                
                # 프로필 이미지 영역 (플레이스홀더)
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #e0e0e0; margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 40px;">
                        👤
                    </div>
                    <h3 style="margin-top: 10px;">{selected_customer['customer_name']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 고객 상세 정보
                info_items = [
                    ("EMAIL", selected_customer.get('email', 'N/A'), "✉️"),
                    ("연락처", selected_customer.get('phone', 'N/A'), "📞"),
                    ("고객 ID", selected_customer.get('customer_id', 'N/A'), "🆔"),
                    ("고객 성향", selected_customer.get('personality', 'N/A'), "🎭"),
                    ("선호 여행지", selected_customer.get('preferred_destination', 'N/A'), "✈️"),
                    ("평균 만족도", f"{selected_customer.get('survey_score', 0.0):.1f} / 5.0", "⭐"),
                    ("마지막 상담", selected_customer.get('last_consultation', 'N/A'), "📅"),
                ]
                
                for label, value, icon in info_items:
                    st.markdown(f"""
                    <div class="customer-info-section">
                        <div class="customer-info-label">{icon} {label}</div>
                        <div class="customer-info-value">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 추가 정보 (스키마 기반)
                if 'basic_info' in selected_customer or 'crm_profile' in selected_customer:
                    st.markdown("#### 상세 정보")
                    if 'crm_profile' in selected_customer:
                        profile = selected_customer['crm_profile']
                        if profile.get('personality_summary'):
                            st.markdown(f"**성향 요약:** {profile['personality_summary']}")
                        if profile.get('travel_budget'):
                            st.markdown(f"**여행 예산:** {profile['travel_budget']}")
                
                # 채팅 상태 및 상담원 정보
                st.markdown("---")
                st.markdown("#### 채팅 상태")
                st.markdown("**상태:** 💬 Chatting")
                st.markdown("**상담원:** 상담원")
                st.markdown("**부서:** Sales")
            else:
                st.info("고객 정보를 불러올 수 없습니다.")
        else:
            st.info("고객을 선택하면 상세 정보가 표시됩니다.")


