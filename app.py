import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Splitter: cloud-safe
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # fallback if text-splitters isn't present for some reason
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_community import GoogleSearchAPIWrapper

load_dotenv()

# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="wide")

# =========================
# Session state
# =========================
defaults = {
    "messages": [],
    "chat_store": {},
    "vectorstore": None,
    "rag_ready": False,
    "file_name": None,
    "search_enabled": False,
    "temperature": 0.7,
    "max_tokens": 700,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# Keys / LLM
# =========================
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

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

# =========================
# Embeddings / Vectorstore
# =========================
@st.cache_resource(show_spinner="Loading embeddings...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    return Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        collection_name="rag_docs"
    )

# =========================
# File loaders
# =========================
def extract_text_from_pdf(file_bytes):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        docs = []
        for i, p in enumerate(reader.pages):
            text = p.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": "pdf", "page": i + 1}))
        return docs if docs else [Document(page_content="PDF had no extractable text.")]
    except Exception as e:
        return [Document(page_content=f"PDF error: {e}")]

def extract_text_from_txt(file_bytes):
    return [Document(page_content=file_bytes.decode("utf-8", errors="ignore"), metadata={"source": "txt"})]

def extract_text_from_csv(file_bytes):
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        return [Document(page_content=df.to_string(index=False), metadata={"source": "csv"})]
    except Exception as e:
        return [Document(page_content=f"CSV error: {e}")]

# =========================
# Prompts
# =========================
plain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. If you are unsure, say you don't know."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# Rewrite follow-up question to standalone (history-aware)
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest question as a standalone question using chat history if needed. "
               "If it is already standalone, return it unchanged. Return ONLY the rewritten question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# RAG answering prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context below. If the answer isn't in the context, say you don't know.\n\n"
               "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# Google answering prompt
google_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the Google search results below to answer. "
               "If results don't contain the answer, say you don't know.\n\n"
               "Google Results:\n{search_results}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# =========================
# Google Search Wrapper
# =========================
@st.cache_resource(show_spinner=False)
def get_google_search():
    return GoogleSearchAPIWrapper()

# =========================
# Core logic
# =========================
def format_docs(docs):
    # Keep it compact to avoid token blowups
    parts = []
    for d in docs[:6]:
        meta = d.metadata or {}
        page = meta.get("page", None)
        tag = f"(page {page})" if page else ""
        parts.append(f"{tag}\n{d.page_content}".strip())
    return "\n\n---\n\n".join(parts)

def answer_with_rag(question: str, session_id="default") -> str:
    llm = get_llm()

    # 1) rewrite question (history aware)
    history = get_session_history(session_id)
    standalone_q = (rewrite_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "chat_history": history.messages
    })

    # 2) retrieve
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(standalone_q)

    # 3) answer from context
    context = format_docs(docs)
    ans = (rag_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "context": context,
        "chat_history": history.messages
    })

    # update history
    history.add_user_message(question)
    history.add_ai_message(ans)
    return ans

def answer_with_google(question: str, session_id="default") -> str:
    # requires google keys
    gkey = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gcse = st.secrets.get("GOOGLE_CSE_ID") or os.getenv("GOOGLE_CSE_ID")
    if not gkey or not gcse:
        return "Google Search is ON, but GOOGLE_API_KEY / GOOGLE_CSE_ID is missing in Secrets."

    llm = get_llm()
    history = get_session_history(session_id)

    search = get_google_search()
    results = search.run(question)  # text blob

    ans = (google_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "search_results": results,
        "chat_history": history.messages
    })

    history.add_user_message(question)
    history.add_ai_message(ans)
    return ans

def answer_plain(question: str, session_id="default") -> str:
    llm = get_llm()
    history = get_session_history(session_id)
    ans = (plain_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "chat_history": history.messages
    })
    history.add_user_message(question)
    history.add_ai_message(ans)
    return ans

def generate_response(question: str) -> str:
    if st.session_state.search_enabled:
        return answer_with_google(question)
    if st.session_state.rag_ready and st.session_state.vectorstore:
        return answer_with_rag(question)
    return answer_plain(question)

# =========================
# UI
# =========================
st.title("🤖 AI Chat Assistant (RAG + Google Search)")

c1, c2, c3 = st.columns([2.5, 1, 1])
with c2:
    st.session_state.temperature = st.slider("Temp", 0.0, 1.0, st.session_state.temperature, 0.1)
with c3:
    st.session_state.max_tokens = st.slider("Tokens", 200, 2000, st.session_state.max_tokens, 100)

toggle_col1, toggle_col2 = st.columns([4, 1])
with toggle_col2:
    if st.button("🔍 Google ON" if st.session_state.search_enabled else "🔍 Google OFF"):
        st.session_state.search_enabled = not st.session_state.search_enabled
        st.rerun()

uploaded = st.file_uploader("Upload PDF / TXT / CSV (for RAG)", type=["pdf", "txt", "csv"])
if uploaded:
    ext = uploaded.name.split(".")[-1].lower()
    data = uploaded.read()
    loaders = {"pdf": extract_text_from_pdf, "txt": extract_text_from_txt, "csv": extract_text_from_csv}

    with st.spinner("Building RAG index..."):
        docs = loaders[ext](data)
        st.session_state.vectorstore = build_vectorstore(docs)
        st.session_state.rag_ready = True
        st.session_state.file_name = uploaded.name

    st.success(f"RAG ready: {uploaded.name}")

if st.session_state.rag_ready and st.session_state.file_name:
    st.caption(f"📄 RAG Active: {st.session_state.file_name}")

st.divider()

# chat history render
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            resp = generate_response(prompt)
        except Exception as e:
            resp = f"Error: {e}"
    st.session_state.messages.append({"role": "assistant", "content": resp})
    st.rerun()
