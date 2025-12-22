# ========================================
# pages/rag_page.py
# RAG Tab 모듈
# ========================================

import os
import streamlit as st
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import DATA_DIR, RAG_INDEX_DIR
from lang_pack import LANG
from rag_handler import (
    build_rag_index, get_embedding_function, split_documents,
    rag_answer
)
from simulation_handler import (
    load_simulation_histories_local,
    generate_daily_customer_guide, save_daily_customer_guide
)


def render_rag_page():
    """RAG Tab 렌더링 함수"""
    # 현재 언어 확인 및 L 변수 정의
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    uploaded_files = st.file_uploader(
        L["file_uploader"],
        type=["pdf", "txt", "html"],
        key="rag_file_uploader", # RAG 전용 키
        accept_multiple_files=True,
        help="RAG에 사용할 학습 자료를 업로드하세요. PDF, TXT, HTML 파일을 지원합니다."
    )

    if uploaded_files:
        if uploaded_files != st.session_state.get("uploaded_files_state"):
            # 파일이 변경되면 RAG 상태 초기화
            st.session_state.is_rag_ready = False
            st.session_state.rag_vectorstore = None
            st.session_state.uploaded_files_state = uploaded_files

        if not st.session_state.get("is_rag_ready", False):
            if st.button(L["button_start_analysis"]):
                if not st.session_state.get("is_llm_ready", False):
                    st.error(L["simulation_no_key_warning"])
                else:
                    with st.spinner(L["data_analysis_progress"]):
                        vectorstore, count = build_rag_index(uploaded_files)

                    if vectorstore:
                        st.session_state.rag_vectorstore = vectorstore
                        st.session_state.is_rag_ready = True
                        st.success(L["embed_success"].format(count=count))
                        st.session_state.rag_messages = [
                            {"role": "assistant", "content": f"✅ {len(uploaded_files)}개 파일 분석 완료. 질문해 주세요."}
                        ]
                    else:
                        st.error(L["embed_fail"])
                        st.session_state.is_rag_ready = False
    else:
        st.info(L["warning_no_files"])
        st.session_state.is_rag_ready = False
        st.session_state.rag_vectorstore = None
        st.session_state.rag_messages = []

    st.markdown("---")

    # ⭐ RAG 데이터 학습 기능 추가 - AI 고객 응대 시뮬레이터 데이터를 일일 파일로 학습
    st.subheader("📚 고객 가이드 자동 생성 및 관리 (일일 학습)")
    
    # 오늘 날짜의 가이드 파일 확인
    today_str = datetime.now().strftime("%y%m%d")
    guide_filename = f"{today_str}_고객가이드.TXT"
    guide_filepath = os.path.join(DATA_DIR, guide_filename)
    
    # 기존 가이드 파일 표시
    if os.path.exists(guide_filepath):
        st.info(f"✅ 오늘의 고객 가이드가 이미 생성되어 있습니다: {guide_filename}")
        with st.expander("📄 생성된 가이드 미리보기"):
            try:
                with open(guide_filepath, "r", encoding="utf-8") as f:
                    guide_preview = f.read()
                st.text_area("가이드 내용", guide_preview[:2000] + "..." if len(guide_preview) > 2000 else guide_preview, height=300, disabled=True)
            except Exception as e:
                st.error(f"가이드 파일 읽기 오류: {e}")
    else:
        st.info("💡 고객 응대 시뮬레이션을 실행하면 자동으로 가이드가 생성됩니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(L.get("button_generate_daily_guide", "🔄 오늘 날짜 고객 가이드 수동 생성/업데이트"), key="generate_daily_guide", use_container_width=True):
            # 최근 이력 로드
            all_histories = load_simulation_histories_local(st.session_state.language)
            
            if all_histories:
                if st.session_state.get("is_llm_ready", False):
                    # simulation_handler의 함수 사용
                    with st.spinner(L.get("generating_customer_guide", "고객 가이드 생성 중...")):
                        guide_content = generate_daily_customer_guide(all_histories, st.session_state.language)
                        
                        if guide_content:
                            saved_path = save_daily_customer_guide(guide_content, st.session_state.language)
                            
                            if saved_path:
                                st.success(L.get("guide_generated", "✅ 고객 가이드가 생성/업데이트되었습니다: {filename}").format(filename=guide_filename))
                                st.info(L.get("guide_file_location", "파일 위치: {path}").format(path=saved_path))
                            else:
                                st.error(L.get("guide_save_failed", "가이드 저장에 실패했습니다."))
                        else:
                            st.warning(L.get("guide_generation_failed", "가이드 생성에 실패했습니다. LLM API Key를 확인해주세요."))
                else:
                    st.error(L.get("llm_not_ready", "LLM이 준비되지 않았습니다. API Key를 설정해주세요."))
            else:
                st.warning(L.get("no_history_for_analysis", "분석할 이력이 없습니다. 먼저 고객 응대 시뮬레이션을 실행하세요."))
    
    with col2:
        # 생성된 가이드를 RAG에 자동 추가하는 기능
        if os.path.exists(guide_filepath):
            if st.button(L.get("button_add_guide_to_rag", "📚 생성된 가이드를 RAG 인덱스에 추가"), key="add_guide_to_rag", use_container_width=True):
                if not st.session_state.get("is_llm_ready", False):
                    st.error(L.get("llm_not_ready", "LLM이 준비되지 않았습니다. API Key를 설정해주세요."))
                else:
                    try:
                        # 가이드 파일을 RAG 인덱스에 추가
                        with st.spinner("RAG 인덱스 업데이트 중..."):
                            # 가이드 파일 읽기
                            with open(guide_filepath, "r", encoding="utf-8") as f:
                                guide_text = f.read()
                            
                            # 문서 생성
                            new_doc = Document(
                                page_content=guide_text,
                                metadata={"source": guide_filepath, "type": "customer_guide", "date": today_str}
                            )
                            
                            # 기존 RAG 인덱스가 있으면 로드하여 병합
                            if st.session_state.get("rag_vectorstore"):
                                # 임베딩 함수 가져오기
                                embedding_func = get_embedding_function()
                                
                                if embedding_func:
                                    # 문서를 청크로 분할
                                    chunks = split_documents([new_doc])
                                    
                                    # 기존 벡터스토어에 추가
                                    st.session_state.rag_vectorstore.add_documents(chunks)
                                    
                                    # 인덱스 저장
                                    st.session_state.rag_vectorstore.save_local(RAG_INDEX_DIR)
                                    
                                    st.success(f"✅ 고객 가이드가 RAG 인덱스에 추가되었습니다! (추가된 청크 수: {len(chunks)})")
                                else:
                                    st.error("임베딩 함수를 초기화할 수 없습니다.")
                            else:
                                # 새 인덱스 생성
                                try:
                                    vectorstore, count = build_rag_index([guide_filepath])
                                    
                                    if vectorstore:
                                        st.session_state.rag_vectorstore = vectorstore
                                        st.session_state.is_rag_ready = True
                                        st.success(f"✅ RAG 인덱스가 생성되었습니다. (문서 수: {count})")
                                    else:
                                        st.error("RAG 인덱스 생성에 실패했습니다.")
                                except Exception as e:
                                    st.error(f"RAG 인덱스 생성 중 오류: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                                
                    except Exception as e:
                        st.error(f"RAG 인덱스 업데이트 중 오류 발생: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("먼저 고객 가이드를 생성해주세요.")
    
    st.markdown("---")

    # --- 챗봇 섹션 (app.py 스타일로 간소화) ---
    if st.session_state.get("is_rag_ready", False) and st.session_state.get("rag_vectorstore"):
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = [{"role": "assistant", "content": "분석된 자료에 대해 질문해 주세요."}]

        # 메시지 표시 (app.py 스타일)
        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # 입력 (app.py 스타일: st.chat_input 사용)
        if prompt := st.chat_input(L.get("rag_input_placeholder", "질문을 입력하세요...")):
            # 사용자 메시지 추가 및 표시
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # AI 응답 생성 및 표시
            with st.chat_message("assistant"):
                with st.spinner(L.get("response_generating", "답변 생성 중...")):
                    response = rag_answer(
                        prompt,
                        st.session_state.rag_vectorstore,
                        st.session_state.language
                    )
                    st.write(response)

            # 응답을 메시지에 추가
            st.session_state.rag_messages.append({"role": "assistant", "content": response})
    else:
        st.warning(L.get("warning_rag_not_ready", "RAG가 준비되지 않았습니다. 파일을 업로드하고 분석을 시작하세요."))
