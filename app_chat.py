"""
app.py의 채팅 페이지 렌더링 로직
"""

import streamlit as st
from datetime import datetime
from data_manager import load_customers, load_chats, save_chats, load_dashboard_stats, save_dashboard_stats
from ai_services import get_ai_response
from config import get_api_key

def render_chat_page():
    """채팅 페이지 렌더링 (Chatstack 스타일 레이아웃)"""
    customers = load_customers()
    chats = load_chats()
    
    # 참고용 app.py 스타일 레이아웃: 왼쪽 고객 목록 + 가운데 채팅 + 오른쪽 고객 정보
    col1, col2, col3 = st.columns([1, 2, 1.5])
    
    # 고객 리스트 (참고용 app.py 스타일)
    with col1:
        st.subheader("고객 목록")
        unread_counts = {}
        for customer in customers:
            customer_id = customer['customer_id']
            if customer_id in chats:
                customer_messages = [msg for msg in chats[customer_id] if msg['sender'] == 'customer']
                unread_counts[customer_id] = len(customer_messages)
        
        for customer in customers:
            customer_id = customer['customer_id']
            is_selected = st.session_state.selected_customer_id == customer_id
            
            if st.button(f"👤 {customer['customer_name']}", key=f"customer_{customer_id}", 
                        use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.selected_customer_id = customer_id
            
            if customer_id in unread_counts and unread_counts[customer_id] > 0:
                st.caption(f"🔴 {unread_counts[customer_id]}개")
    
    # 채팅 화면
    with col2:
        if st.session_state.selected_customer_id:
            selected_customer = next((c for c in customers if c['customer_id'] == st.session_state.selected_customer_id), None)
            
            if selected_customer:
                st.subheader(f"💬 {selected_customer['customer_name']}님과의 대화")
                
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
                    if (last_msg['sender'] == 'customer' and current_last_id != last_msg_id and api_key_auto):
                        if f'ai_processing_{customer_id}' not in st.session_state:
                            st.session_state[f'ai_processing_{customer_id}'] = True
                            try:
                                ai_response = get_ai_response(last_msg['message'], selected_customer, current_chats)
                                st.session_state.ai_suggestion = {
                                    'customer_id': customer_id, 'message': ai_response,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                            except Exception as e:
                                st.session_state.ai_suggestion = {
                                    'customer_id': customer_id, 'message': f"오류: {str(e)}",
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                            finally:
                                st.session_state[f'ai_processing_{customer_id}'] = False
                        st.session_state.last_message_id[customer_id] = current_last_id
                
                # 메시지 표시
                chat_container = st.container(height=400)
                with chat_container:
                    for msg in current_chats:
                        sender_class = "message-operator" if msg['sender'] == 'operator' else "message-customer"
                        st.markdown(f"""
                        <div class="{sender_class}">
                            <strong>{msg['sender_name']}</strong><br>
                            {msg['message']}<br>
                            <small style="color: #666;">{msg['timestamp']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # AI 제안 표시
                    if (current_chats and current_chats[-1]['sender'] == 'customer' and
                        st.session_state.get('ai_suggestion', {}).get('customer_id') == customer_id):
                        ai_suggestion = st.session_state.ai_suggestion
                        st.markdown(f"""
                        <div class="message-ai-suggestion">
                            <strong>🤖 AI 제안 응답</strong><br>
                            {ai_suggestion['message']}<br>
                            <small style="color: #666;">{ai_suggestion['timestamp']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("✅ 사용", key=f"use_ai_{customer_id}", use_container_width=True):
                            new_message = {
                                "message_id": f"MSG{len(current_chats) + 1:03d}",
                                "sender": "operator", "sender_name": "상담원",
                                "message": ai_suggestion['message'],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            chats[customer_id].append(new_message)
                            save_chats(chats)
                            st.session_state.ai_suggestion = {}
                    
                    if st.session_state.get(f'ai_processing_{customer_id}', False):
                        st.info("🤖 AI가 응답을 생성하는 중...")
                
                st.divider()
                
                # 입력 영역
                chat_input = st.text_input("메시지 입력", key=f"chat_input_{customer_id}", 
                                          placeholder="메시지를 입력하세요...", label_visibility="collapsed")
                
                # 아이콘 버튼들 + 전송 버튼
                col_icon1, col_icon2, col_send = st.columns([1, 1, 4])
                with col_icon1:
                    if st.button("👤", key=f"btn_customer_info_{customer_id}", use_container_width=True, help="고객 정보 업데이트"):
                        st.info("고객 정보 업데이트 기능 (구현 예정)")
                with col_icon2:
                    if st.button("🤖", key=f"btn_ai_guide_{customer_id}", use_container_width=True, help="AI 응대 가이드"):
                        try:
                            if current_chats:
                                last_customer_msg = next((msg for msg in reversed(current_chats) if msg['sender'] == 'customer'), None)
                                if last_customer_msg:
                                    ai_guide = get_ai_response(last_customer_msg['message'], selected_customer, current_chats)
                                    st.info(f"🤖 AI 응대 가이드:\n\n{ai_guide}")
                                else:
                                    st.info("고객 메시지가 없어 AI 가이드를 생성할 수 없습니다.")
                            else:
                                st.info("대화 기록이 없어 AI 가이드를 생성할 수 없습니다.")
                        except Exception as e:
                            st.error(f"AI 가이드 생성 오류: {str(e)}")
                with col_send:
                    if st.button("전송", type="primary", use_container_width=True, key=f"send_{customer_id}"):
                        if chat_input:
                            new_message = {
                                "message_id": f"MSG{len(chats[customer_id]) + 1:03d}",
                                "sender": "operator", "sender_name": "상담원",
                                "message": chat_input, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            chats[customer_id].append(new_message)
                            save_chats(chats)
                            stats = load_dashboard_stats()
                            stats['today_cases'] += 1
                            save_dashboard_stats(stats)
            else:
                st.info("고객을 선택해주세요.")
        else:
            st.info("왼쪽에서 고객을 선택하여 채팅을 시작하세요.")
    
    # 고객 정보
    with col3:
        if st.session_state.selected_customer_id:
            selected_customer = next((c for c in customers if c['customer_id'] == st.session_state.selected_customer_id), None)
            if selected_customer:
                st.subheader("고객 정보")
                st.markdown(f"### 👤 {selected_customer['customer_name']}")
                
                st.markdown(f"**고객 ID:** {selected_customer.get('customer_id', 'N/A')}")
                st.markdown(f"**연락처:** {selected_customer.get('phone', 'N/A')}")
                st.markdown(f"**이메일:** {selected_customer.get('email', 'N/A')}")
                
                if selected_customer.get('account_created'):
                    st.markdown(f"**계정 생성일:** {selected_customer.get('account_created', 'N/A')}")
                if selected_customer.get('last_login'):
                    st.markdown(f"**마지막 접속일:** {selected_customer.get('last_login', 'N/A')}")
                if selected_customer.get('last_consultation'):
                    st.markdown(f"**마지막 상담일자:** {selected_customer.get('last_consultation', 'N/A')}")
                
                st.markdown(f"**성향:** {selected_customer.get('personality', 'N/A')}")
                
                if selected_customer.get('personality_summary'):
                    st.markdown("**고객 성향 요약:**")
                    st.info(selected_customer.get('personality_summary', 'N/A'))
                
                st.metric("설문 점수", f"{selected_customer.get('survey_score', 0):.1f} / 5.0")
                
                if selected_customer.get('service_rating'):
                    st.metric("응대 평가 점수", f"{selected_customer.get('service_rating', 0):.1f} / 5.0")
            else:
                st.info("고객 정보를 불러올 수 없습니다.")
        else:
            st.info("고객을 선택하면 상세 정보가 표시됩니다.")


