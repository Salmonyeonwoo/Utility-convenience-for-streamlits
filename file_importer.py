"""
파일 임포트 모듈
JSON, CSV 파일 파싱 및 파일 임포트/스캔 로직
"""
import os
import json
import pandas as pd
from pathlib import Path
from file_parser import extract_data_from_text


def parse_json(file_path, manager):
    """JSON 파일에서 데이터 추출"""
    try:
        import streamlit as st
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON 구조에 따라 데이터 추출
        if isinstance(data, list):
            # 티켓 리스트인 경우
            result = []
            for item in data:
                if isinstance(item, dict):
                    # 티켓 형식인 경우 고객 정보와 티켓 정보 분리
                    if 'cust_id' in item or 'ticket_id' in item:
                        # 기존 티켓 형식에서 고객 정보 추출
                        cust_id = item.get('cust_id', '')
                        db = manager._load_data()
                        if cust_id and cust_id in db.get('customers', {}):
                            cust_info = db['customers'][cust_id]
                            result.append({
                                "customer_info": {
                                    "name": cust_info.get("name", ""),
                                    "phone": cust_info.get("phone", ""),
                                    "email": cust_info.get("email", ""),
                                    "trait": cust_info.get("trait", "일반")
                                },
                                "ticket_info": {
                                    "consult_type": item.get("consult_type", "기타"),
                                    "status": item.get("status", "Pending"),
                                    "content": item.get("content", ""),
                                    "summary": item.get("summary", ""),
                                    "analysis": item.get("analysis", {"sentiment": "보통", "score": 5})
                                }
                            })
                        else:
                            # 고객 정보가 없는 경우 텍스트에서 추출 시도
                            text = item.get("content", "") + " " + item.get("summary", "")
                            parsed = extract_data_from_text(text)
                            if parsed:
                                result.extend(parsed)
                    else:
                        result.append(item)
            return result if result else None
        elif isinstance(data, dict):
            if 'tickets' in data:
                # CRM DB 형식인 경우
                tickets = data['tickets']
                customers = data.get('customers', {})
                result = []
                for ticket in tickets:
                    cust_id = ticket.get('cust_id', '')
                    if cust_id and cust_id in customers:
                        cust_info = customers[cust_id]
                        result.append({
                            "customer_info": {
                                "name": cust_info.get("name", ""),
                                "phone": cust_info.get("phone", ""),
                                "email": cust_info.get("email", ""),
                                "trait": cust_info.get("trait", "일반")
                            },
                            "ticket_info": {
                                "consult_type": ticket.get("consult_type", "기타"),
                                "status": ticket.get("status", "Pending"),
                                "content": ticket.get("content", ""),
                                "summary": ticket.get("summary", ""),
                                "analysis": ticket.get("analysis", {"sentiment": "보통", "score": 5})
                            }
                        })
                return result if result else None
            else:
                return [data]
        return None
    except Exception as e:
        import streamlit as st
        st.error(f"JSON 파싱 오류: {str(e)}")
        return None


def parse_csv(file_path):
    """CSV 파일에서 데이터 추출"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        try:
            df = pd.read_csv(file_path, encoding='cp949')
        except Exception as e2:
            import streamlit as st
            st.error(f"CSV 파싱 오류: {str(e2)}")
            return None
    
    # CSV 컬럼명을 표준화
    records = df.to_dict('records')
    result = []
    for record in records:
        # 컬럼명 매핑 (다양한 형식 지원)
        name = record.get('name') or record.get('고객명') or record.get('이름') or record.get('Name') or ""
        phone = record.get('phone') or record.get('연락처') or record.get('전화') or record.get('Phone') or ""
        email = record.get('email') or record.get('이메일') or record.get('Email') or ""
        trait = record.get('trait') or record.get('고객성향') or record.get('성향') or "일반"
        consult_type = record.get('consult_type') or record.get('상담유형') or record.get('유형') or "기타"
        status = record.get('status') or record.get('상태') or record.get('Status') or "Pending"
        content = str(record.get('content', '')) or str(record.get('상담내용', '')) or str(record.get('내용', '')) or ""
        summary = str(record.get('summary', '')) or str(record.get('요약', '')) or ""
        score = record.get('score') or record.get('CSAT') or record.get('만족도') or record.get('점수') or 5
        sentiment = record.get('sentiment') or record.get('감정') or "보통"
        
        if name or phone:
            result.append({
                "name": str(name),
                "phone": str(phone),
                "email": str(email) if email else "",
                "trait": str(trait),
                "consult_type": str(consult_type),
                "status": str(status),
                "content": str(content),
                "summary": str(summary),
                "analysis": {
                    "sentiment": str(sentiment),
                    "score": int(score) if isinstance(score, (int, float)) else 5
                }
            })
    
    return result if result else None


def import_from_file(file_path, manager, debug=False):
    """파일에서 데이터를 읽어서 DB에 저장"""
    from file_parser import parse_pdf, parse_docx, parse_pptx
    
    file_ext = Path(file_path).suffix.lower()
    parsed_data = []
    
    try:
        if file_ext == '.pdf':
            parsed_data = parse_pdf(file_path)
        elif file_ext in ['.doc', '.docx']:
            parsed_data = parse_docx(file_path)
        elif file_ext == '.pptx':
            parsed_data = parse_pptx(file_path)
        elif file_ext == '.json':
            parsed_data = parse_json(file_path, manager)
        elif file_ext == '.csv':
            parsed_data = parse_csv(file_path)
        else:
            if debug:
                import streamlit as st
                st.write(f"⚠️ 지원하지 않는 파일 형식: {file_ext}")
            return 0
    except Exception as e:
        if debug:
            import streamlit as st
            st.error(f"❌ 파일 파싱 오류 ({os.path.basename(file_path)}): {str(e)}")
        return 0
    
    if not parsed_data:
        if debug:
            import streamlit as st
            st.write(f"⚠️ 파싱된 데이터 없음: {os.path.basename(file_path)} (파일 내용에 고객 정보가 없을 수 있습니다)")
        return 0
    
    imported_count = 0
    db = manager._load_data()
    existing_tickets = {t.get('ticket_id') for t in db.get('tickets', [])}
    
    for item in parsed_data:
        try:
            # JSON이나 CSV에서 직접 티켓 형식으로 온 경우
            if 'customer_info' in item and 'ticket_info' in item:
                cust_info = item['customer_info']
                tkt_info = item['ticket_info']
            elif 'cust_id' in item or 'ticket_id' in item:
                # 기존 티켓 형식인 경우 - 중복 체크
                ticket_id = item.get('ticket_id')
                if ticket_id and ticket_id in existing_tickets:
                    continue  # 이미 존재하는 티켓은 스킵
                # 고객 정보 추출 시도
                cust_id = item.get('cust_id', '')
                if cust_id and cust_id in db.get('customers', {}):
                    cust_info = {
                        "name": db['customers'][cust_id].get("name", ""),
                        "phone": db['customers'][cust_id].get("phone", ""),
                        "email": db['customers'][cust_id].get("email", ""),
                        "trait": db['customers'][cust_id].get("trait", "일반")
                    }
                else:
                    # 텍스트에서 추출
                    text = item.get("content", "") + " " + item.get("summary", "")
                    parsed = extract_data_from_text(text)
                    if parsed and len(parsed) > 0:
                        cust_info = {
                            "name": parsed[0].get("name", ""),
                            "phone": parsed[0].get("phone", ""),
                            "email": parsed[0].get("email", ""),
                            "trait": parsed[0].get("trait", "일반")
                        }
                    else:
                        continue
                tkt_info = {
                    "consult_type": item.get("consult_type", "기타"),
                    "status": item.get("status", "Pending"),
                    "content": item.get("content", ""),
                    "summary": item.get("summary", ""),
                    "analysis": item.get("analysis", {"sentiment": "보통", "score": 5})
                }
            else:
                # 파싱된 데이터를 티켓 형식으로 변환
                cust_info = {
                    "name": item.get("name", ""),
                    "phone": item.get("phone", ""),
                    "email": item.get("email", ""),
                    "trait": item.get("trait", "일반")
                }
                tkt_info = {
                    "consult_type": item.get("consult_type", "기타"),
                    "status": item.get("status", "Pending"),
                    "content": item.get("content", ""),
                    "summary": item.get("summary", ""),
                    "analysis": item.get("analysis", {"sentiment": "보통", "score": 5})
                }
            
            if cust_info.get("name") or cust_info.get("phone"):
                ticket_id = manager.save_ticket(cust_info, tkt_info)
                existing_tickets.add(ticket_id)  # 중복 방지를 위해 추가
                imported_count += 1
        except Exception as e:
            import streamlit as st
            st.warning(f"데이터 임포트 중 오류: {str(e)}")
            continue
    
    return imported_count


def scan_folder(folder_path, manager, skip_scanned=True, debug=False):
    """폴더 내 모든 지원 파일을 스캔하여 임포트"""
    if not os.path.exists(folder_path):
        if debug:
            import streamlit as st
            st.warning(f"폴더가 존재하지 않습니다: {folder_path}")
        return 0
    
    supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.json', '.csv']
    total_imported = 0
    total_files = 0
    skipped_files = 0
    failed_files = 0
    
    import streamlit as st
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in supported_extensions:
                total_files += 1
                
                # 이미 스캔된 파일인지 확인
                if skip_scanned and manager.is_file_scanned(file_path):
                    skipped_files += 1
                    if debug:
                        st.write(f"⏭️ 건너뛰기: {os.path.basename(file_path)}")
                    continue
                
                try:
                    imported = import_from_file(file_path, manager, debug=debug)
                    if imported > 0:
                        total_imported += imported
                        if debug:
                            st.write(f"✅ 임포트 성공: {os.path.basename(file_path)} ({imported}건)")
                    else:
                        failed_files += 1
                        if debug:
                            st.write(f"⚠️ 데이터 없음: {os.path.basename(file_path)}")
                    
                    # 스캔 완료 표시 (임포트 성공 여부와 관계없이)
                    if skip_scanned:
                        manager.mark_file_as_scanned(file_path, imported)
                except Exception as e:
                    failed_files += 1
                    if debug:
                        st.error(f"❌ 오류: {os.path.basename(file_path)} - {str(e)}")
    
    if debug:
        st.info(f"📊 스캔 결과: 총 {total_files}개 파일 중 {total_imported}건 임포트, {skipped_files}개 건너뛰기, {failed_files}개 실패")
    
    return total_imported

