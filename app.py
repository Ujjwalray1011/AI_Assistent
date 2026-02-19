import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# ===== LangChain Core =====
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ===== Memory & Vector DB =====
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Splitter import (Streamlit Cloud safe)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===== RAG Chains =====
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

# ===== Google Search Agent =====
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import Tool

load_dotenv()

# ===== Streamlit Config =====
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# ===== Minimal UI Styling =====
st.markdown(
    """
    <style>
    .stApp { background-color: #212121; color: #ececec; }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== Session State =====
defaults = {
    "messages": [],
    "vectorstore": None,
    "rag_ready": False,
    "chat_store": {},
    "search_enabled": False,
    "temperature": 0.7,
    "max_tokens": 700,
    "input_key": 0,
    "file_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===== Embeddings =====
@st.cache_resource(show_spinner="Loading embeddings...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ===== Google Search Tool =====
@st.cache_resource(show_spinner=False)
def get_google_tool():
    # Needs GOOGLE_API_KEY + GOOGLE_CSE_ID in Streamlit secrets/env
    search = GoogleSearchAPIWrapper()
    return Tool(
        name="Google Search",
        description="Search Google for real-time information, current events, latest updates",
        func=search.run
    )

# ===== File loaders =====
def extract_text_from_pdf(file_bytes):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        docs = []
        for i, p in enumerate(reader.pages):
            text = p.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": "pdf", "page": i + 1}))
        return docs if docs else [Document(page_content="PDF contained no extractable text.")]
    except Exception as e:
        return [Document(page_content=f"PDF error: {e}")]

def extract_text_from_txt(file_bytes):
    return [Document(page_content=file_bytes.decode("utf-8", errors="ignore"))]

def extract_text_from_csv(file_bytes):
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        return [Document(page_content=df.to_string(index=False), metadata={"source": "csv"})]
    except Exception as e:
        return [Document(page_content=f"CSV error: {e}")]

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # ✅ Chroma: Streamlit Cloud friendly (FAISS often fails)
    return Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        collection_name="rag_docs"
    )

# ===== Prompts =====
plain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer clearly and helpfully."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and latest question, rewrite it as a standalone question if needed. Keep it short."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY using the context below. If not in context, say you don't know.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ===== LLM =====
def get_llm():
    groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("Missing GROQ_API_KEY. Add it in Streamlit Secrets.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        api_key=groq_key
    )

def get_history(session_id="default") -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

# ===== Response Generator =====
def generate_response(question: str) -> str:
    llm = get_llm()
    history = get_history()

    # 🔍 Google Search path
    if st.session_state.search_enabled:
        # Guard: Google creds required
        gkey = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        gcse = st.secrets.get("GOOGLE_CSE_ID") or os.getenv("GOOGLE_CSE_ID")
        if not gkey or not gcse:
            return "Google Search is ON, but GOOGLE_API_KEY / GOOGLE_CSE_ID is missing in Secrets."

        tool = get_google_tool()
        agent = initialize_agent(
            tools=[tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )
        answer = agent.run(question)
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer

    # 📄 RAG path
    if st.session_state.rag_ready and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        har = create_history_aware_retriever(llm, retriever, contextualize_prompt)
        doc_chain = create_stuff_documents_chain(llm, rag_prompt)
        rag_chain = create_retrieval_chain(har, doc_chain)

        chain = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        result = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "default"}}
        )
        return result.get("answer", "No answer returned.")

    # 💬 Normal chat
    answer = (plain_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "chat_history": history.messages
    })
    history.add_user_message(question)
    history.add_ai_message(answer)
    return answer

# ===== UI =====
st.title("🤖 AI Chat Assistant (RAG + Google Search)")

top1, top2, top3 = st.columns([3, 1, 1])
with top2:
    st.session_state.temperature = st.slider("Temp", 0.0, 1.0, st.session_state.temperature, 0.1)
with top3:
    st.session_state.max_tokens = st.slider("Tokens", 200, 2000, st.session_state.max_tokens, 100)

toggle_col1, toggle_col2 = st.columns([4, 1])
with toggle_col2:
    if st.button("🔍 Google ON" if st.session_state.search_enabled else "🔍 Google OFF"):
        st.session_state.search_enabled = not st.session_state.search_enabled
        st.rerun()

uploaded_file = st.file_uploader("Upload PDF / TXT / CSV (for RAG)", type=["pdf", "txt", "csv"])
if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    data = uploaded_file.read()

    loaders = {
        "pdf": extract_text_from_pdf,
        "txt": extract_text_from_txt,
        "csv": extract_text_from_csv,
    }

    with st.spinner("Building RAG index..."):
        docs = loaders[ext](data)
        st.session_state.vectorstore = build_vectorstore(docs)
        st.session_state.rag_ready = True
        st.session_state.file_name = uploaded_file.name

    st.success(f"RAG ready: {uploaded_file.name}")

if st.session_state.rag_ready and st.session_state.file_name:
    st.caption(f"📄 RAG Active: {st.session_state.file_name}")

st.divider()

# chat history
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        try:
            answer = generate_response(user_input)
        except Exception as e:
            answer = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
