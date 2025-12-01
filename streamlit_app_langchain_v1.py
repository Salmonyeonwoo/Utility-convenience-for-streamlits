# ========================================
# streamlit_app_rebuilt_langchain_v1.py
# ========================================
# 🔥 최신 LangChain 1.x / LCEL 기반 + 로컬 JSON DB 버전
# - Firebase 제거
# - ConversationalRetrievalChain / ConversationChain 제거
# - 최신 Runnable 체인 구조 적용
# - Whisper / Audio / Simulator 완전 작동
# ========================================

import os
import json
import uuid
import streamlit as st
import tempfile
import time
from datetime import datetime
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

# LangChain 최신 1.x 구조
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# ========================================
# 1. 환경 준비
# ========================================
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_KEY)

# JSON DB 디렉토리
BASE_DIR = ".venv/local_db"
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
SIM_HISTORY_JSON = os.path.join(BASE_DIR, "simulation_histories.json")
VOICE_JSON = os.path.join(BASE_DIR, "voice_records.json")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# JSON DB utils

def json_load(path):
    if not os.path.exists(path): return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def json_save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ========================================
# 2. 최신 LangChain 기반 Conversational Agent
# ========================================
# 메모리 구조 (수동 관리)
if "sim_history" not in st.session_state:
    st.session_state.sim_history = []  # {"user":..., "assistant":...}

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_KEY)

# Prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI support agent. Maintain professional tone."),
    ("placeholder", "{history}"),
    ("human", "{input}")
])

# Runnable Chain
agent_chain = (
    {
        "input": RunnablePassthrough(),
        "history": lambda _: st.session_state.sim_history
    }
    | chat_prompt
    | llm
)

# ========================================
# 3. Whisper 음성 전사
# ========================================

def transcribe(audio_bytes, mime_type):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_bytes)
    tmp.close()
    try:
        with open(tmp.name, "rb") as f:
            result = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return result
    except Exception as e:
        return f"Whisper Error: {e}"
    finally:
        os.remove(tmp.name)

# ========================================
# 4. JSON DB 저장 함수
# ========================================

def save_simulation_json(initial_query, messages):
    db = json_load(SIM_HISTORY_JSON)
    db.append({
        "id": str(uuid.uuid4()),
        "initial_query": initial_query,
        "messages": messages,
        "created": datetime.now().isoformat()
    })
    json_save(SIM_HISTORY_JSON, db)


def save_voice_json(filename, audio_bytes, transcript, mime_type):
    db = json_load(VOICE_JSON)
    audio_path = os.path.join(AUDIO_DIR, filename)

    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    db.append({
        "id": str(uuid.uuid4()),
        "filename": filename,
        "audio_path": audio_path,
        "mime_type": mime_type,
        "transcript": transcript,
        "created": datetime.now().isoformat()
    })

    json_save(VOICE_JSON, db)

# ========================================
# 5. Streamlit UI 시작
# ========================================
st.title("AI 상담 시뮬레이터 (LangChain v1 + JSON DB)")

# ========================================
# A. 음성 업로드 + Whisper
# ========================================
st.header("🎙 음성 전사")
audio_file = st.file_uploader("Upload audio", type=["wav","mp3","webm","m4a"])

if audio_file:
    audio_bytes = audio_file.getvalue()
    st.audio(audio_bytes, format=audio_file.type)

    if st.button("전사 실행"):
        text = transcribe(audio_bytes, audio_file.type)
        st.session_state.last_transcript = text
        st.text_area("전사 결과", value=text)

    if st.button("저장하기"):
        filename = f"voice_{int(time.time())}.wav"
        save_voice_json(filename, audio_bytes, st.session_state.get("last_transcript", ""), audio_file.type)
        st.success("음성 저장 완료!")

# ========================================
# B. 상담 시뮬레이터 (최신 LCEL 기반)
# ========================================
st.header("🧑‍💼 상담 시뮬레이터")

initial_input = st.text_input("고객 메시지 입력")

if st.button("응답 생성"):
    if not initial_input:
        st.warning("입력 필요")
    else:
        # 유저 메시지 기록
        st.session_state.sim_history.append({"role": "human", "content": initial_input})

        # AI 응답 생성
        response = agent_chain.invoke(initial_input)

        st.session_state.sim_history.append(
            {"role": "assistant", "content": response.content}
        )

        # 저장
        save_simulation_json(initial_input, st.session_state.sim_history)

        st.rerun()

# 메시지 렌더링
for msg in st.session_state.sim_history:
    sender = "user" if msg["role"] == "human" else "assistant"
    st.chat_message(sender).markdown(msg["content"])

# ========================================
# C. 상담 이력
# ========================================
st.header("📜 상담 이력")
histories = json_load(SIM_HISTORY_JSON)

if not histories:
    st.write("이력이 없습니다.")
else:
    for h in histories[-10:][::-1]:
        st.subheader(h["initial_query"])
        for msg in h["messages"]:
            role = msg["role"]
            st.write(f"**{role}**: {msg['content']}")
