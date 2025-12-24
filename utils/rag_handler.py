"""
RAG (Retrieval Augmented Generation) 처리 모듈
문서 로드, 임베딩, 인덱스 구축, RAG 답변 생성 기능을 제공합니다.
"""

import os
import tempfile
from typing import List, Optional
import streamlit as st

from config import RAG_INDEX_DIR
from llm_client import get_api_key, run_llm, get_llm_client
from lang_pack import LANG

# Langchain imports for RAG functionality
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# 임베딩 모델 사용 가능 여부 확인
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    IS_GEMINI_EMBEDDING_AVAILABLE = True
except ImportError:
    IS_GEMINI_EMBEDDING_AVAILABLE = False

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    IS_NVIDIA_EMBEDDING_AVAILABLE = True
except ImportError:
    IS_NVIDIA_EMBEDDING_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None


def load_documents(files) -> List[Document]:
    """문서 파일들을 로드합니다."""
    docs: List[Document] = []
    for f in files:
        name = f.name
        lower = name.lower()
        if lower.endswith(".pdf"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(f.read())
            tmp.flush()
            tmp.close()
            loader = PyPDFLoader(tmp.name)
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source"] = name
            docs.extend(file_docs)
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        elif lower.endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
        elif lower.endswith(".html") or lower.endswith(".htm"):
            text = f.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """문서를 청크로 분할합니다."""
    if RecursiveCharacterTextSplitter is None:
        return docs
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embedding_model():
    """임베딩 모델을 반환합니다 (구버전 호환용)."""
    if get_api_key("openai"):
        try:
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except:
            pass
    if get_api_key("gemini"):
        try:
            return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        except:
            pass
    return None


def get_embedding_function():
    """
    RAG 임베딩에 사용할 임베딩 모델을 결정합니다.
    API 키 유효성 순서: OpenAI (사용자 설정 시) -> Gemini -> NVIDIA -> HuggingFace (fallback)
    API 인증 오류 발생 시 다음 모델로 이동하도록 처리합니다.
    """
    # 1. OpenAI 임베딩 시도 (사용자가 유효한 키를 설정했을 경우)
    openai_key = get_api_key("openai")
    if openai_key:
        try:
            st.info("🔹 RAG: OpenAI Embedding 사용 중")
            return OpenAIEmbeddings(openai_api_key=openai_key)
        except Exception as e:
            st.warning(f"OpenAI 임베딩 실패 → Gemini로 Fallback: {e}")

    # 2. Gemini 임베딩 시도
    gemini_key = get_api_key("gemini")
    if IS_GEMINI_EMBEDDING_AVAILABLE and gemini_key:
        try:
            st.info("🔹 RAG: Gemini Embedding 사용 중")
            # ⭐ 수정: 모델 이름 형식을 'models/model-name'으로 수정
            return GoogleGenerativeAIEmbeddings(google_api_key=gemini_key, model="models/text-embedding-004")
        except Exception as e:
            st.warning(f"Gemini 임베딩 실패 → NVIDIA로 Fallback: {e}")

    # 3. NVIDIA 임베딩 시도
    nvidia_key = get_api_key("nvidia")
    if IS_NVIDIA_EMBEDDING_AVAILABLE and nvidia_key:
        try:
            st.info("🔹 RAG: NVIDIA Embedding 사용 중")
            # NIM 모델 사용 (실제 키가 유효해야 함)
            return NVIDIAEmbeddings(api_key=nvidia_key, model="ai-embed-qa-4")
        except Exception as e:
            st.warning(f"NVIDIA 임베딩 실패 → HuggingFace Fallback: {e}")

    # 4. HuggingFace Embeddings (Local Fallback)
    try:
        st.info("🔹 RAG: Local HuggingFace Embedding 사용 중")
        # 경량 모델 사용
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"최종 Fallback 임베딩 실패: {e}")

    st.error("❌ RAG 임베딩 실패: 사용 가능한 API Key가 없습니다.")
    return None


def build_rag_index(files):
    """RAG 인덱스를 구축합니다."""
    # 언어 키 안전하게 가져오기
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    if not files: 
        return None, 0

    # 임베딩 함수를 시도하는 과정에서 에러 메시지가 발생할 수 있으므로 try-except로 감쌉니다.
    try:
        embeddings = get_embedding_function()
    except Exception as e:
        st.error(f"RAG 임베딩 함수 초기화 중 치명적인 오류 발생: {e}")
        return None, 0

    if embeddings is None:
        # 어떤 임베딩 모델도 초기화할 수 없음을 알림
        error_msg = L["rag_embed_error_none"]

        # 상세 오류 정보 구성 (실제 사용 가능한 임베딩 모델이 없는 경우)
        if not get_api_key("openai"):
            error_msg += f"\n- {L['rag_embed_error_openai']}"
        if not get_api_key("gemini"):
            error_msg += f"\n- {L['rag_embed_error_gemini']}"
        if not get_api_key("nvidia"):
            error_msg += f"\n- {L['rag_embed_error_nvidia']}"

        st.error(error_msg)
        return None, 0

    # 임베딩 객체 초기화 성공 후, 데이터 로드 및 분할
    docs = load_documents(files)
    if not docs: 
        return None, 0

    chunks = split_documents(docs)
    if not chunks: 
        return None, 0

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # 저장
        vectorstore.save_local(RAG_INDEX_DIR)
    except Exception as e:
        # API 인증 실패 등 실제 API 호출 오류 처리
        st.error(f"RAG 인덱스 생성 중 오류: {e}")
        return None, 0

    return vectorstore, len(chunks)


def load_rag_index():
    """RAG 인덱스를 로드합니다."""
    # RAG 인덱스 로드 시에도 유효한 임베딩 함수가 필요합니다.
    try:
        embeddings = get_embedding_function()
    except Exception:
        # get_embedding_function 내에서 에러 메시지를 처리하거나 스킵하므로 여기서는 조용히 처리
        return None

    if embeddings is None:
        return None

    try:
        # allow_dangerous_deserialization=True는 필수
        vs = FAISS.load_local(RAG_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception:
        return None


def rag_answer(question: str, vectorstore: FAISS, lang_key: str) -> str:
    """RAG를 사용하여 질문에 답변합니다."""
    # RAG Answer는 LLM 클라이언트 라우팅을 사용하도록 수정
    llm_client, info = get_llm_client()
    if llm_client is None:
        # 언어 키 검증
        if lang_key not in ["ko", "en", "ja"]:
            lang_key = "ko"
        return LANG.get(lang_key, LANG["ko"]).get("simulation_no_key_warning", "API Key가 필요합니다.")

    # Langchain ChatOpenAI 대신 run_llm을 사용하기 위해 prompt를 직접 구성
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # ⭐ 수정: LangChain 버전 호환성 - get_relevant_documents 대신 invoke 사용
    try:
        # 최신 LangChain 버전 (invoke 사용)
        docs = retriever.invoke(question)
    except AttributeError:
        # 구버전 LangChain (get_relevant_documents 사용)
        try:
            docs = retriever.get_relevant_documents(question)
        except AttributeError:
            # 대체 방법: vectorstore에서 직접 검색
            docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content[:1500] for d in docs)

    # ⭐ RAG 다국어 인식 오류 해결: 답변 생성 모델에게 질문 언어로 일관되게 답하도록 강력히 지시
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(lang_key, "English")

    prompt = (
        f"You are a helpful AI tutor. Answer the question using ONLY the provided context.\n"
        f"The answer MUST be STRICTLY in {lang_name}, which is the language of the question.\n"
        f"If you cannot find the answer in the context, say you don't know in {lang_name}.\n"
        f"Note: The context may be in a different language, but you must still answer in {lang_name}.\n\n"
        "Question:\n" + question + "\n\n"
        "Context:\n" + context + "\n\n"
        f"Answer (in {lang_name}):"
    )
    return run_llm(prompt)


def load_or_train_lstm():
    """LSTM 모델을 로드하거나 학습합니다 (Mock 구현)."""
    try:
        import numpy as np
        # 실제 LSTM 대신 랜덤 + sin 파형 기반 Mock
        np.random.seed(42)
        n_points = 50
        ts = 60 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_points)) + np.random.normal(0, 5, n_points)
        ts = np.clip(ts, 50, 100).astype(np.float32)
        return ts
    except ImportError:
        return None




