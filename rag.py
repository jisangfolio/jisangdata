import streamlit as st
import pandas as pd
import os
import time
from typing import List

# LangChain & Gemini Imports
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# 1. Configuration & API Setup
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")

# 2. 해당 경로의 .env 파일을 강제로 로드합니다.
load_dotenv(dotenv_path=env_path)

# (디버깅용) 화면에 경로가 제대로 잡히는지 확인해보세요. 해결되면 지우셔도 됩니다.
st.write(f"검색 경로: {env_path}")
st.write(f"파일 존재 여부: {os.path.exists(env_path)}")

# 3. API 키 확인
if not os.getenv("GOOGLE_API_KEY"):
    st.error(f"API Key를 찾을 수 없습니다. 경로를 확인해주세요: {env_path}")
    st.stop()

GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"

st.set_page_config(page_title="AnyData Chatbot", page_icon="📂")
st.title("📂 내 파일과 대화하기 (AnyData Chatbot)")

# 2. File Upload Logic
# =========================================================
with st.sidebar:
    st.header("파일 업로드")
    uploaded_file = st.file_uploader("CSV 또는 Excel 파일을 업로드하세요", type=["csv", "xlsx"])

@st.cache_resource(show_spinner="AI가 문서를 읽고 있습니다... (데이터가 많으면 시간이 걸릴 수 있습니다)")
def process_uploaded_file(file):
    if file is None:
        return None, None

    # 1. 파일 읽기
    try:
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding='cp949')
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None, None

    # 2. 텍스트 변환 (Document 생성)
    documents = []
    for idx, row in df.iterrows():
        content_parts = []
        for col in df.columns:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                content_parts.append(f"{col}: {val}")
        
        page_content = "\n".join(content_parts)
        title_col = df.columns[0]
        row_title = str(row[title_col])[:50] 

        doc = Document(
            page_content=page_content,
            metadata={
                "row": idx,
                "source": file.name,
                "summary_title": row_title
            }
        )
        documents.append(doc)

    # 3. 청크 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(documents)

    # 4. 임베딩 및 벡터 저장 (배치 처리 + 속도 조절)
    embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # 진행률 표시바 생성
    progress_text = "벡터 변환 중입니다. 잠시만 기다려주세요..."
    my_bar = st.progress(0, text=progress_text)
    
    batch_size = 20  # 한 번에 처리할 문서 수 (너무 크면 429 에러 발생)
    total_splits = len(splits)
    vectorstore = None
    
    for i in range(0, total_splits, batch_size):
        batch = splits[i : i + batch_size]
        
        # 첫 번째 배치로 VectorStore 생성, 그 이후는 추가(add)
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embedding=embedding)
        else:
            vectorstore.add_documents(batch)
            
        # 진행률 업데이트
        percent_complete = min((i + batch_size) / total_splits, 1.0)
        my_bar.progress(percent_complete, text=f"벡터 변환 중... ({int(percent_complete*100)}%)")
        
        # API 제한을 피하기 위해 1초 대기 (데이터가 많으면 2~3초로 늘리세요)
        time.sleep(1)

    my_bar.empty() # 완료되면 진행바 삭제
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return df, retriever

# 3. Main Logic
# =========================================================

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="안녕하세요! CSV나 Excel 파일을 업로드해주시면 내용을 분석해 드릴게요.")
    ]

# 파일 처리
if uploaded_file:
    df, retriever = process_uploaded_file(uploaded_file)
    if retriever:
        st.success(f"✅ '{uploaded_file.name}' 분석 완료! ({len(df)}개의 데이터)")
else:
    # 파일이 없으면 안내 메시지 표시하고 중단
    st.info("👈 왼쪽 사이드바에서 파일을 업로드해주세요.")
    df, retriever = None, None

# 대화 히스토리 표시
for msg in st.session_state["messages"]:
    st.chat_message(msg.role).write(msg.content)

# LLM 초기화
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

# 사용자 입력 처리
user_input = st.chat_input("이 데이터에 대해 궁금한 점을 물어보세요")

if user_input and retriever:
    # 사용자 메시지 표시
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    # 검색 (RAG)
    retrieved_docs = retriever.invoke(user_input)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 프롬프트 (범용)
    prompt = ChatPromptTemplate.from_template(
        """당신은 업로드된 데이터를 기반으로 답변하는 AI 데이터 분석가입니다.
        아래의 [데이터 문맥]을 바탕으로 사용자의 질문에 답변하세요.
        
        규칙:
        1. 문맥에 없는 내용은 지어내지 말고 "데이터에서 찾을 수 없습니다"라고 답하세요.
        2. 답변은 친절하고 전문적으로 작성하세요.
        3. 출처(데이터의 내용)를 근거로 답변하세요.

        [데이터 문맥]:
        {context}

        질문:
        {question}

        답변:"""
    )

    chain = prompt | llm

    # 답변 생성 및 스트리밍
    with st.chat_message("assistant"):
        with st.spinner("데이터 분석 중..."):
            response_container = st.empty()
            full_response = ""
            
            for chunk in chain.stream({
                "question": user_input,
                "context": context_text
            }):
                full_response += chunk.content
                response_container.markdown(full_response)
            
            # 출처 표시 (선택 사항)
            source_titles = set([doc.metadata['summary_title'] for doc in retrieved_docs])
            if source_titles:
                st.caption(f"참고 데이터: {', '.join(list(source_titles)[:3])} 등")

            st.session_state["messages"].append(ChatMessage(role="assistant", content=full_response))

elif user_input and not retriever:
    st.warning("먼저 파일을 업로드해주세요.")