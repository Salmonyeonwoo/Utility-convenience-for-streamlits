"""
고객 상담 및 Solved 티켓 KPI 관리 시스템
메인 UI 및 애플리케이션 진입점
"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path

# 로컬 모듈 import
from crm_manager import TicketCRMManager
from file_importer import scan_folder, import_from_file


def auto_scan_on_startup(manager, folder_paths):
    """앱 시작 시 자동으로 여러 폴더 스캔 및 카운팅"""
    # 세션 상태로 자동 스캔 여부 확인
    if 'auto_scan_completed' not in st.session_state:
        st.session_state.auto_scan_completed = False
    
    if not st.session_state.auto_scan_completed:
        total_imported = 0
        scanned_folders = []
        
        # 첫 실행 시 강제 스캔 (skip_scanned=False) 또는 기존 스캔 기록이 없으면
        scanned_files = manager.load_scanned_files()
        db_data = manager._load_data()
        ticket_count = len(db_data.get('tickets', []))
        # 스캔 기록이 없거나 티켓이 5개 미만이면 강제 스캔
        force_scan = len(scanned_files) == 0 or ticket_count < 5
        
        with st.spinner("🔄 자동 카운팅 중... (여러 폴더 스캔)"):
            for folder_path in folder_paths:
                if os.path.exists(folder_path):
                    # 첫 실행이거나 강제 스캔이면 skip_scanned=False
                    skip_flag = not force_scan
                    imported_count = scan_folder(folder_path, manager, skip_scanned=skip_flag, debug=False)
                    if imported_count > 0:
                        total_imported += imported_count
                        scanned_folders.append((folder_path, imported_count))
                else:
                    # 폴더가 없어도 계속 진행 (다른 폴더는 있을 수 있음)
                    pass
        
        if total_imported > 0 or force_scan:
            st.session_state.auto_scan_completed = True
            st.session_state.last_auto_scan_count = total_imported
            st.session_state.scanned_folders = scanned_folders
            return total_imported
        else:
            st.session_state.auto_scan_completed = True
            return 0
    
    return 0


def render_crm_app():
    """CRM 앱 메인 렌더링 함수"""
    st.set_page_config(page_title="KPI 기반 고객 관리 시스템", layout="wide")
    manager = TicketCRMManager()
    
    # 기본 폴더 경로들 (여러 폴더 자동 스캔)
    # GitHub 배포 시에는 환경 변수나 설정 파일을 사용하도록 수정 필요
    default_folders = []
    
    # 로컬 환경에서만 폴더 경로 추가 (절대 경로는 로컬 전용)
    if os.name == 'nt':  # Windows 환경
        local_folders = [
            r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\customer data histories via streamlits",
            r"C:\Users\Admin\OneDrive\ドキュメント\Yeonwoo_streamlit_app_test\customer data histories via streamlits (practicing)"
        ]
        # 폴더가 존재하는 경우에만 추가
        for folder in local_folders:
            if os.path.exists(folder):
                default_folders.append(folder)
    
    # 환경 변수에서 폴더 경로 읽기 (GitHub 배포 시 사용)
    env_folders = os.getenv('CRM_DATA_FOLDERS', '')
    if not env_folders:
        # Streamlit secrets에서도 읽기 시도
        try:
            env_folders = st.secrets.get('CRM_DATA_FOLDERS', '')
        except:
            pass
    
    if env_folders:
        for folder in env_folders.split(';'):
            folder = folder.strip()
            if folder and os.path.exists(folder):
                default_folders.append(folder)
    
    # 앱 시작 시 자동 스캔 (여러 폴더 모두 스캔)
    auto_imported = auto_scan_on_startup(manager, default_folders)
    
    db_data = manager._load_data()

    st.title("📂 고객 상담 및 Solved 티켓 KPI 관리")
    
    # 자동 스캔 결과 표시
    if auto_imported > 0 and 'last_auto_scan_count' in st.session_state:
        if 'scanned_folders' in st.session_state and st.session_state.scanned_folders:
            folder_info = " | ".join([f"{os.path.basename(f[0])}: {f[1]}건" for f in st.session_state.scanned_folders])
            st.success(f"✅ 자동 카운팅 완료: 총 {st.session_state.last_auto_scan_count}건의 데이터가 추가되었습니다! ({folder_info})")
        else:
            st.success(f"✅ 자동 카운팅 완료: {st.session_state.last_auto_scan_count}건의 데이터가 추가되었습니다!")
    
    # --- 상단 KPI 대시보드 (실무 핵심 지표) ---
    all_tickets = db_data['tickets']
    solved_count = sum(1 for t in all_tickets if t['status'] == "Solved")
    pending_count = sum(1 for t in all_tickets if t['status'] == "Pending")
    
    # 평균 만족도 계산
    total_avg_csat = 0.0
    if all_tickets:
        total_avg_csat = sum(t['analysis']['score'] for t in all_tickets) / len(all_tickets)

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("총 해결(Solved)", f"{solved_count} 건")
    col_kpi2.metric("처리 대기(Pending)", f"{pending_count} 건", delta=f"{pending_count}건 남음", delta_color="inverse")
    col_kpi3.metric("전체 평균 CSAT", f"{total_avg_csat:.2f} / 5.0")
    col_kpi4.metric("누적 고객 수", f"{len(db_data['customers'])} 명")

    tab1, tab2, tab3, tab4 = st.tabs(["📝 상담 입력", "🔍 고객별 통계", "📊 유형별 분석", "📁 파일 임포트"])

    with tab1:
        st.subheader("신규 상담 티켓 생성")
        with st.form("ticket_form", clear_on_submit=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                name = st.text_input("고객명")
                phone = st.text_input("연락처")
            with c2:
                # 실무형 상담 유형 드롭다운
                consult_type = st.selectbox("상담 유형", [
                    "배송문의/지연", "환불/반품요청", "결제/오류문의", 
                    "상품정보/재고", "계정/로그인", "강성/컴플레인", "기타"
                ])
                status = st.radio("최종 상태", ["Solved", "Pending"], horizontal=True)
            with c3:
                trait = st.selectbox("고객 성향", ["일반", "부드러움", "합리적", "급함", "까다로움", "진상/강성"])
                email = st.text_input("이메일")
            
            st.divider()
            content = st.text_area("상담 상세 내용 (진상 대응 사례 등 구체적 입력)")
            summary = st.text_input("핵심 요약 (한 줄)")
            
            c4, c5 = st.columns(2)
            with c4:
                sentiment = st.select_slider("AI 감정 분석", options=["매우나쁨", "나쁨", "보통", "좋음", "매우좋음"], value="보통")
            with c5:
                score = st.slider("고객 응대 평가 (CSAT)", 1, 5, 5)

            if st.form_submit_button("상담 데이터 확정 저장"):
                if name and phone:
                    cust_info = {"name": name, "phone": phone, "email": email, "trait": trait}
                    tkt_info = {
                        "consult_type": consult_type,
                        "status": status, 
                        "content": content, 
                        "summary": summary,
                        "analysis": {"sentiment": sentiment, "score": score}
                    }
                    tid = manager.save_ticket(cust_info, tkt_info)
                    st.success(f"티켓 {tid} 저장 완료! 대시보드 수치가 갱신되었습니다.")
                    st.rerun()
                else:
                    st.warning("고객명과 연락처는 필수입니다.")
        
        # 파일 업로더 섹션
        st.divider()
        st.subheader("📤 파일에서 데이터 임포트")
        uploaded_file = st.file_uploader(
            "고객 데이터 파일 업로드 (PDF, Word, PPTX, JSON, CSV)",
            type=['pdf', 'docx', 'doc', 'pptx', 'json', 'csv'],
            help="다운로드한 이력 파일을 업로드하면 자동으로 데이터가 카운팅됩니다."
        )
        
        if uploaded_file is not None:
            # 임시 파일로 저장
            temp_dir = "temp_uploads"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("파일 임포트 실행"):
                with st.spinner("파일을 분석하고 데이터를 임포트하는 중..."):
                    imported_count = import_from_file(temp_path, manager)
                    if imported_count > 0:
                        st.success(f"✅ {imported_count}건의 데이터가 성공적으로 임포트되었습니다!")
                        st.rerun()
                    else:
                        st.warning("임포트된 데이터가 없습니다. 파일 형식을 확인해주세요.")
                
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    with tab2:
        st.subheader("고객 마스터 데이터베이스")
        if db_data['customers']:
            df_cust = pd.DataFrame.from_dict(db_data['customers'], orient='index')
            # 정렬 및 컬럼 선택
            df_display = df_cust[["name", "trait", "total_solved", "csat_avg", "last_consult_date"]]
            st.dataframe(df_display.sort_values("total_solved", ascending=False), use_container_width=True)
        else:
            st.info("데이터가 없습니다.")

    with tab3:
        st.subheader("업무 효율 및 유형 분석")
        if all_tickets:
            df_tickets = pd.DataFrame(all_tickets)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.write("**[유형별 티켓 분포]**")
                type_counts = df_tickets["consult_type"].value_counts()
                st.bar_chart(type_counts)
            
            with col_chart2:
                st.write("**[상태별 해결 비율]**")
                status_counts = df_tickets["status"].value_counts()
                st.write(status_counts)
                
            st.divider()
            st.write("**[상담 유형별 평균 만족도(KPI)]**")
            # 유형별 평균 점수 계산
            type_csat = df_tickets.groupby("consult_type").apply(lambda x: x["analysis"].apply(lambda y: y["score"]).mean())
            st.line_chart(type_csat)
        else:
            st.info("분석할 데이터가 부족합니다.")
    
    with tab4:
        st.subheader("📁 폴더에서 자동 임포트")
        st.write("**폴더의 모든 파일을 자동으로 스캔하여 데이터를 임포트합니다.**")
        st.info("💡 **팁**: 앱 실행 시 자동으로 여러 폴더가 스캔되어 카운팅됩니다. 수동으로 다시 스캔하려면 아래 버튼을 사용하세요.")
        
        # 여러 폴더 스캔 옵션
        scan_mode = st.radio(
            "스캔 모드",
            ["기본 폴더들 모두 스캔", "개별 폴더 스캔"],
            horizontal=True,
            help="기본 폴더들을 모두 스캔하거나 개별 폴더를 선택하여 스캔할 수 있습니다."
        )
        
        if scan_mode == "기본 폴더들 모두 스캔":
            st.write("**기본 폴더 목록:**")
            for i, folder in enumerate(default_folders, 1):
                exists = "✅" if os.path.exists(folder) else "❌"
                st.write(f"{i}. {exists} `{folder}`")
            
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            with col_opt1:
                skip_scanned = st.checkbox("이미 스캔한 파일 건너뛰기", value=False, 
                                          help="체크하면 이미 스캔한 파일은 다시 스캔하지 않습니다.")
            with col_opt2:
                force_rescan = st.checkbox("강제 재스캔 (모든 파일)", value=True,
                                          help="체크하면 모든 파일을 다시 스캔합니다.")
            with col_opt3:
                debug_mode = st.checkbox("디버그 모드 (상세 로그)", value=False,
                                        help="체크하면 각 파일의 스캔 상태를 상세히 표시합니다.")
            
            scan_all_button = st.button("🔍 모든 기본 폴더 스캔 및 임포트", type="primary")
            
            if scan_all_button:
                total_imported = 0
                scanned_folders = []
                
                with st.spinner("여러 폴더를 스캔하고 데이터를 임포트하는 중..."):
                    skip_flag = skip_scanned and not force_rescan
                    for folder_path in default_folders:
                        if os.path.exists(folder_path):
                            if debug_mode:
                                st.write(f"📁 폴더 스캔 중: {folder_path}")
                            imported_count = scan_folder(folder_path, manager, skip_scanned=skip_flag, debug=debug_mode)
                            if imported_count > 0:
                                total_imported += imported_count
                                scanned_folders.append((folder_path, imported_count))
                        else:
                            if debug_mode:
                                st.warning(f"⚠️ 폴더가 존재하지 않습니다: {folder_path}")
                
                if total_imported > 0:
                    folder_info = " | ".join([f"{os.path.basename(f[0])}: {f[1]}건" for f in scanned_folders])
                    st.success(f"✅ 총 {total_imported}건의 데이터가 성공적으로 임포트되었습니다! ({folder_info})")
                    st.balloons()
                    st.rerun()
                else:
                    st.warning("⚠️ 임포트된 데이터가 없습니다. 다음을 확인해주세요:")
                    st.write("1. 폴더 내에 지원되는 파일 형식(PDF, Word, PPTX, JSON, CSV)이 있는지 확인")
                    st.write("2. 파일 내용에 고객명 또는 연락처 정보가 있는지 확인")
                    st.write("3. 디버그 모드를 활성화하여 상세 로그 확인")
                    
                    # 스캔된 파일 통계 표시
                    scanned_files = manager.load_scanned_files()
                    if scanned_files:
                        st.info(f"📊 현재 {len(scanned_files)}개 파일이 스캔 기록에 있습니다.")
                        
                        # 스캔 기록 초기화 버튼
                        if st.button("🗑️ 스캔 기록 초기화 (모든 파일 다시 스캔)", type="secondary"):
                            manager.save_scanned_files({})
                            st.success("✅ 스캔 기록이 초기화되었습니다. 다음 스캔 시 모든 파일이 다시 스캔됩니다.")
                            st.rerun()
        
        else:  # 개별 폴더 스캔
            col_folder1, col_folder2 = st.columns([3, 1])
            with col_folder1:
                folder_path = st.text_input(
                    "폴더 경로",
                    value=default_folders[0] if default_folders else "",
                    help="스캔할 폴더의 전체 경로를 입력하세요"
                )
            
            with col_folder2:
                st.write("")  # 공간 맞추기
                st.write("")  # 공간 맞추기
                scan_button = st.button("🔍 폴더 스캔 및 임포트", type="primary")
            
            # 수동 스캔 옵션
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                skip_scanned = st.checkbox("이미 스캔한 파일 건너뛰기", value=True, 
                                          help="체크하면 이미 스캔한 파일은 다시 스캔하지 않습니다.")
            with col_opt2:
                force_rescan = st.checkbox("강제 재스캔 (모든 파일)", value=False,
                                          help="체크하면 모든 파일을 다시 스캔합니다.")
            
            if scan_button:
                if os.path.exists(folder_path):
                    with st.spinner("폴더를 스캔하고 데이터를 임포트하는 중..."):
                        skip_flag = skip_scanned and not force_rescan
                        imported_count = scan_folder(folder_path, manager, skip_scanned=skip_flag)
                        if imported_count > 0:
                            st.success(f"✅ 총 {imported_count}건의 데이터가 성공적으로 임포트되었습니다!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.info("임포트된 데이터가 없습니다. 폴더 내에 지원되는 파일 형식(PDF, Word, PPTX, JSON, CSV)이 있는지 확인해주세요.")
                else:
                    st.error(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        
        st.divider()
        st.subheader("📋 지원 파일 형식")
        st.write("""
        - **PDF** (.pdf): PDF 문서에서 텍스트 추출
        - **Word** (.docx, .doc): Word 문서에서 텍스트 추출
        - **PowerPoint** (.pptx): PowerPoint 프레젠테이션에서 텍스트 추출
        - **JSON** (.json): JSON 형식의 구조화된 데이터
        - **CSV** (.csv): CSV 형식의 표 데이터
        """)
        
        st.divider()
        st.subheader("ℹ️ 데이터 추출 규칙")
        st.write("""
        파일에서 다음 정보를 자동으로 추출합니다:
        - **고객명**: "고객명", "이름", "성함" 등의 키워드로 추출
        - **연락처**: "연락처", "전화", "Phone" 등의 키워드로 추출
        - **이메일**: 이메일 형식 자동 인식
        - **상담 유형**: 파일 내용에서 상담 유형 키워드 검색
        - **상태**: "Solved", "해결", "완료" 키워드로 Solved 판단
        - **CSAT 점수**: "CSAT", "만족도", "점수" 키워드로 추출 (1-5점)
        - **감정 분석**: "매우나쁨", "나쁨", "좋음" 등의 키워드로 추출
        """)
        
        st.divider()
        st.subheader("🔄 자동 카운팅 기능")
        st.write("""
        **자동 카운팅이 활성화되어 있습니다!**
        
        - 앱 실행 시 여러 기본 폴더가 자동으로 스캔됩니다
        - 로컬 환경: Windows 환경에서 자동으로 로컬 폴더를 감지합니다
        - GitHub 배포: 환경 변수 `CRM_DATA_FOLDERS`에 폴더 경로를 세미콜론(;)으로 구분하여 설정하세요
        - 새로운 파일이 추가되면 다음 앱 실행 시 자동으로 카운팅됩니다
        - 이미 스캔한 파일은 수정 시간을 확인하여 변경된 경우에만 다시 스캔합니다
        - 수동으로 재스캔하려면 위의 "모든 기본 폴더 스캔 및 임포트" 버튼을 사용하세요
        """)
        
        # 현재 활성화된 폴더 표시
        if default_folders:
            st.write("**현재 활성화된 폴더:**")
            for folder in default_folders:
                st.write(f"- `{folder}`")
        else:
            st.warning("⚠️ 활성화된 폴더가 없습니다. 로컬 환경이거나 환경 변수를 설정해주세요.")


if __name__ == "__main__":
    render_crm_app()
