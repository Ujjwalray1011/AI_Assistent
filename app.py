import os
import io
import base64
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime

load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "AI Chat Assistant (RAG)"

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
.stApp { background-color: #212121 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2a2a2a !important;
    min-width: 250px !important;
}

/* Collapse button — visible circular pill on the right edge */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
    background: #2a2a2a !important;
    border: 1px solid #444 !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    color: #aaa !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover {
    background: #3a3a3a !important;
    border-color: #666 !important;
    color: #fff !important;
    box-shadow: 0 3px 14px rgba(0,0,0,0.6) !important;
}
/* Make the expand button (when sidebar is hidden) also visible */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Top navbar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: #171717;
    border-bottom: 1px solid #2f2f2f;
    position: sticky;
    top: 0;
    z-index: 999;
}
.topbar-title {
    font-size: 1em;
    font-weight: 600;
    color: #ececec;
    letter-spacing: -0.2px;
}
.topbar-sub {
    font-size: 0.7em;
    color: #555;
    margin-top: 1px;
}

/* ── Settings panel ── */
.settings-panel {
    background: #1a1a1a;
    border: 1px solid #2f2f2f;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 12px 0 16px 0;
}
.settings-label {
    font-size: 0.7em;
    color: #666;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: block;
}

/* ── Chat bubbles ── */
.user-message {
    background: #2f2f2f;
    color: #ececec;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 10px 0 10px auto;
    max-width: 75%;
    font-size: 0.93em;
    line-height: 1.65;
}
.assistant-message {
    background: transparent;
    color: #ececec;
    padding: 4px 0;
    margin: 10px 0;
    font-size: 0.93em;
    line-height: 1.75;
}
.msg-label {
    font-size: 0.7em;
    color: #555;
    margin-bottom: 4px;
}
.timestamp { font-size: 0.67em; color: #555; margin-top: 4px; }

/* ── Input ── */
.stTextInput > div > div > input {
    background: #2f2f2f !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 14px !important;
    color: #ececec !important;
    padding: 13px 18px !important;
    font-size: 0.93em !important;
    transition: border 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #555 !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input::placeholder { color: #555 !important; }

/* ── Buttons ── */
.stButton > button {
    background: #2f2f2f !important;
    color: #ececec !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    font-size: 0.86em !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #3a3a3a !important;
    border-color: #555 !important;
}

/* ── File badge ── */
.file-badge {
    background: #1e2a1e;
    border: 1px solid #2d4a2d;
    border-radius: 10px;
    padding: 8px 14px;
    color: #4ade80;
    font-size: 0.83em;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Info box ── */
.info-box {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 14px;
    padding: 16px 20px;
    color: #aaa;
    font-size: 0.9em;
    line-height: 1.7;
    margin: 16px 0;
}

/* ── RAG source ── */
.rag-source {
    background: #252525;
    border-left: 2px solid #444;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    color: #888;
    font-size: 0.8em;
    margin: 4px 0;
    line-height: 1.5;
}

/* ── Divider ── */
hr { border-color: #2f2f2f !important; }

/* ── Expander ── */
.streamlit-expanderHeader { color: #666 !important; font-size: 0.82em !important; }
</style>
""", unsafe_allow_html=True)

# ─── AUTO OPEN SIDEBAR ───────────────────────────────────────────────────────
st.markdown("""
    <script>
        // Wait for Streamlit to render, then open sidebar if collapsed
        function openSidebar() {
            try {
                const doc = window.parent.document;
                const sidebar = doc.querySelector('[data-testid="stSidebar"]');
                if (!sidebar) return;
                const isCollapsed = sidebar.getAttribute('aria-expanded') === 'false'
                    || getComputedStyle(sidebar).transform.includes('matrix')
                    || sidebar.getBoundingClientRect().width < 50;
                if (isCollapsed) {
                    const btn = doc.querySelector('[data-testid="collapsedControl"] button')
                             || doc.querySelector('button[aria-label="Open sidebar"]');
                    if (btn) btn.click();
                }
            } catch(e) {}
        }
        // Try multiple times as Streamlit loads async
        setTimeout(openSidebar, 300);
        setTimeout(openSidebar, 800);
        setTimeout(openSidebar, 1500);
    </script>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
defaults = {
    'messages': [],
    'message_count': 0,
    'max_tokens': 500,
    'temperature': 0.7,
    'last_question': None,
    'prev_max_tokens': 500,
    'trigger_regenerate': False,
    'input_key': 0,
    'file_name': None,
    'file_type': None,
    'show_uploader': False,
    'show_settings': False,
    'vectorstore': None,
    'rag_ready': False,
    'plain_context': None,
    'chat_store': {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── EMBEDDINGS ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── FILE HELPERS ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return [Document(page_content=p.extract_text() or "", metadata={"source": "pdf", "page": i+1})
                for i, p in enumerate(reader.pages) if p.extract_text()]
    except Exception as e:
        return [Document(page_content=f"PDF error: {e}")]

def extract_text_from_txt(file_bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    return [Document(page_content=text)]

def extract_text_from_csv(file_bytes):
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        docs = [Document(page_content=f"CSV: {len(df)} rows, columns: {', '.join(df.columns)}")]
        for i in range(0, len(df), 20):
            docs.append(Document(page_content=df.iloc[i:i+20].to_string(index=False),
                                 metadata={"rows": f"{i}-{i+20}"}))
        return docs
    except Exception as e:
        return [Document(page_content=f"CSV error: {e}")]

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return FAISS.from_documents(splitter.split_documents(docs), get_embeddings())

def image_to_base64(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")

# ─── PROMPTS ──────────────────────────────────────────────────────────────────
plain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Do not guess or invent facts."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and latest question, reformulate as standalone question. Return as-is if not needed."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

rag_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context below. Be detailed and accurate.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

image_prompt = ChatPromptTemplate.from_messages([
    ("system", "The user shared an image named '{filename}'. Answer their question based on the filename and context."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# ─── LLM & RESPONSE ───────────────────────────────────────────────────────────
def get_llm(model_name, temperature, max_tokens):
    return ChatGroq(
        model=model_name, temperature=temperature, max_tokens=max_tokens,
        api_key=st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    )

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

def generate_response(question, model_name, temperature, max_tokens, session_id="default"):
    llm     = get_llm(model_name, temperature, max_tokens)
    parser  = StrOutputParser()
    history = get_session_history(session_id)

    if st.session_state.rag_ready and st.session_state.vectorstore:
        retriever  = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        har        = create_history_aware_retriever(llm, retriever, contextualize_prompt)
        doc_chain  = create_stuff_documents_chain(llm, rag_answer_prompt)
        rag_chain  = create_retrieval_chain(har, doc_chain)
        conv_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        result = conv_chain.invoke({"input": question},
                                   config={"configurable": {"session_id": session_id}})
        return result.get("answer", ""), result.get("context", [])

    elif st.session_state.file_type == "image":
        answer = (image_prompt | llm | parser).invoke({
            "question": question,
            "filename": st.session_state.file_name or "image",
            "chat_history": history.messages
        })
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, []

    else:
        answer = (plain_prompt | llm | parser).invoke({
            "question": question, "chat_history": history.messages
        })
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, []

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding:20px 4px 16px 4px;">
            <div style="font-size:1.05em;font-weight:600;color:#ececec;letter-spacing:-0.3px;">
                AI Assistant
            </div>
            <div style="font-size:0.7em;color:#555;margin-top:3px;">
                Groq · LangChain · RAG
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Model
    st.markdown('<p style="font-size:0.7em;color:#666;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Model</p>', unsafe_allow_html=True)
    model_name = st.selectbox("Model",
        ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama2-70b-4096"],
        label_visibility="collapsed")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Temperature
    st.markdown('<p style="font-size:0.7em;color:#666;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Temperature</p>', unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.0,
        value=st.session_state.temperature, step=0.1,
        label_visibility="collapsed", key="temp_slider")
    st.session_state.temperature = temperature

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Max Tokens
    st.markdown('<p style="font-size:0.7em;color:#666;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Max Tokens</p>', unsafe_allow_html=True)
    max_tokens = st.slider("Max Tokens", 100, 2500,
        value=st.session_state.max_tokens, step=100,
        label_visibility="collapsed", key="token_slider")
    st.session_state.max_tokens = max_tokens
    st.markdown(f'<p style="font-size:0.7em;color:#555;margin-top:2px;">{max_tokens} tokens · ~{max_tokens//4} words</p>', unsafe_allow_html=True)

    if max_tokens != st.session_state.prev_max_tokens and st.session_state.last_question:
        st.session_state.prev_max_tokens = max_tokens
        st.session_state.trigger_regenerate = True

    st.divider()

    # Active file
    if st.session_state.file_name:
        st.markdown('<p style="font-size:0.7em;color:#666;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">Active File</p>', unsafe_allow_html=True)
        color = "#4ade80" if st.session_state.rag_ready else "#60a5fa"
        tag   = "RAG Active" if st.session_state.rag_ready else "Image"
        ftype = (st.session_state.file_type or "").upper()
        st.markdown(f"""
            <div style="background:#1e1e1e;border:1px solid #2a2a2a;border-radius:10px;
                        padding:10px 12px;margin-bottom:10px;">
                <div style="font-size:0.8em;color:{color};font-weight:500;">{ftype} · {tag}</div>
                <div style="font-size:0.75em;color:#666;margin-top:3px;word-break:break-all;">
                    {st.session_state.file_name}
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Remove File", use_container_width=True):
            st.session_state.vectorstore   = None
            st.session_state.rag_ready     = False
            st.session_state.file_name     = None
            st.session_state.file_type     = None
            st.session_state.plain_context = None
            st.session_state.chat_store    = {}
            st.rerun()
        st.divider()

    # Session stats
    st.markdown('<p style="font-size:0.7em;color:#666;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">Session</p>', unsafe_allow_html=True)
    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    ai_msgs   = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    sc1, sc2  = st.columns(2)
    with sc1:
        st.metric("You", user_msgs)
    with sc2:
        st.metric("AI", ai_msgs)

    st.divider()

    if st.button("New Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.message_count = 0
        st.session_state.last_question = None
        st.session_state.chat_store    = {}
        st.session_state.input_key    += 1
        st.rerun()

# ─── CHAT AREA ────────────────────────────────────────────────────────────────

# Image preview
if st.session_state.file_type == "image" and st.session_state.plain_context:
    st.markdown(f"""
        <div style="text-align:center;margin:10px 0;">
            <img src="data:image/png;base64,{st.session_state.plain_context}"
                 style="max-height:200px;border-radius:12px;border:1px solid #3a3a3a;"/>
            <div style="color:#666;font-size:0.75em;margin-top:4px;">{st.session_state.file_name}</div>
        </div>
    """, unsafe_allow_html=True)

# Messages
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-message">
                    {message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="msg-label">Assistant</div>', unsafe_allow_html=True)
            st.markdown(message["content"])
            st.markdown(f'<div class="timestamp">{message["timestamp"]}</div>', unsafe_allow_html=True)
            if message.get("sources"):
                with st.expander(f"Sources — {len(message['sources'])} chunks"):
                    for i, src in enumerate(message["sources"]):
                        meta  = src.metadata
                        label = f"Chunk {i+1}"
                        if "page" in meta: label += f" · Page {meta['page']}"
                        elif "rows" in meta: label += f" · Rows {meta['rows']}"
                        st.caption(label)
                        st.markdown(f"> {src.page_content[:300]}...")
                        if i < len(message["sources"]) - 1:
                            st.divider()
else:
    if st.session_state.rag_ready:
        st.markdown(f"""
            <div class="info-box">
                RAG Active — <strong>{st.session_state.file_name}</strong><br>
                Try: "Summarize this" · "What does it say about X?"
            </div>
        """, unsafe_allow_html=True)
    else:
        if not st.session_state.messages:
            st.markdown("""
                <div style="text-align:center;padding:50px 20px 30px;">
                    <div style="font-size:2.2em;font-weight:600;color:#ececec;letter-spacing:-1px;margin-bottom:8px;">
                        What can I help with?
                    </div>
                    <div style="font-size:0.88em;color:#555;">Powered by Groq · LangChain · RAG</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("""
                <div class="info-box">
                    💡 <strong>Tips:</strong><br>
                    📎 Upload PDF, TXT or CSV — AI uses Conversational RAG for accurate answers<br>
                    🖼️ Upload an image and ask questions about it<br>
                    🔁 Ask follow-up questions — AI remembers the full conversation
                </div>
            """, unsafe_allow_html=True)

# ─── FILE BADGE ───────────────────────────────────────────────────────────────
if st.session_state.file_name and not st.session_state.show_settings:
    color = "#4ade80" if st.session_state.rag_ready else "#60a5fa"
    tag   = "RAG" if st.session_state.rag_ready else "Image"
    st.markdown(f"""
        <div class="file-badge">
            <span style="color:{color};font-weight:600;">{tag}</span>
            <span style="color:#888;">{st.session_state.file_name}</span>
        </div>
    """, unsafe_allow_html=True)

# ─── INPUT BAR ────────────────────────────────────────────────────────────────
placeholder = f"Ask about {st.session_state.file_name}..." if st.session_state.file_name else "Type your message here..."

col1, col2, col3 = st.columns([5.5, 1.6, 1.2])
with col1:
    user_input = st.text_input("Message", placeholder=placeholder,
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_key}")
with col2:
    if st.button("📎 Upload", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
with col3:
    send_button = st.button("Send", use_container_width=True)

# Upload panel
if st.session_state.show_uploader:
    uploaded_file = st.file_uploader(
        "Choose a file — PDF, TXT, CSV or Image",
        type=["pdf", "txt", "csv", "png", "jpg", "jpeg"],
        key=f"file_upload_{st.session_state.input_key}"
    )
    if uploaded_file and uploaded_file.name != st.session_state.file_name:
        file_bytes = uploaded_file.read()
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext in ("pdf", "txt", "csv"):
            with st.spinner("Building RAG index..."):
                docs_map = {"pdf": extract_text_from_pdf,
                            "txt": extract_text_from_txt,
                            "csv": extract_text_from_csv}
                docs = docs_map[ext](file_bytes)
                st.session_state.vectorstore  = build_vectorstore(docs)
                st.session_state.rag_ready    = True
                st.session_state.file_name    = uploaded_file.name
                st.session_state.file_type    = ext
                st.session_state.plain_context = None
                st.session_state.show_uploader = False
            st.rerun()
        else:
            st.session_state.plain_context  = image_to_base64(file_bytes)
            st.session_state.file_name      = uploaded_file.name
            st.session_state.file_type      = "image"
            st.session_state.vectorstore    = None
            st.session_state.rag_ready      = False
            st.session_state.show_uploader  = False
            st.rerun()

# ─── HANDLE SEND ──────────────────────────────────────────────────────────────
if user_input and send_button:
    st.session_state.last_question = user_input
    timestamp = datetime.now().strftime("%H:%M")
    display = user_input
    if st.session_state.file_name:
        display = f"<span style='font-size:0.78em;color:#666;'>[{st.session_state.file_name}]</span><br>{user_input}"
    st.session_state.messages.append({"role":"user","content":display,"timestamp":timestamp})
    st.session_state.message_count += 1

    with st.spinner("Thinking..."):
        try:
            answer, sources = generate_response(
                user_input, model_name, temperature, max_tokens, session_id="default")
            st.session_state.messages.append({
                "role":"assistant","content":answer,
                "timestamp":datetime.now().strftime("%H:%M"),"sources":sources})
            st.session_state.message_count += 1
            st.session_state.input_key += 1
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ─── AUTO-REGENERATE ──────────────────────────────────────────────────────────
if st.session_state.get('trigger_regenerate') and st.session_state.last_question:
    st.session_state.trigger_regenerate = False
    if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant':
        st.session_state.messages.pop()
        st.session_state.message_count -= 1
    with st.spinner("Regenerating..."):
        try:
            answer, sources = generate_response(
                st.session_state.last_question, model_name,
                st.session_state.temperature, st.session_state.max_tokens)
            st.session_state.messages.append({
                "role":"assistant","content":answer,
                "timestamp":datetime.now().strftime("%H:%M"),"sources":sources})
            st.session_state.message_count += 1
            st.rerun()
        except Exception as e:
            st.error(f"Regeneration failed: {str(e)}")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
    <div style="text-align:center;color:#444;font-size:0.72em;padding:16px;margin-top:20px;">
        AI can make mistakes · Streamlit · Groq · LangChain RAG
    </div>
""", unsafe_allow_html=True)
