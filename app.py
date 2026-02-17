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

# -------- LangSmith (optional) --------
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "AI Chat Assistant (RAG)"

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* ── Base ── */
    .stApp { background-color: #212121 !important; }
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #2f2f2f !important;
    }
    * { font-family: 'Inter', sans-serif !important; }

    /* ── User bubble ── */
    .user-message {
        background-color: #2f2f2f;
        color: #ececec;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 12px 0 12px auto;
        max-width: 75%;
        font-size: 0.94em;
        line-height: 1.65;
    }
    /* ── AI bubble ── */
    .assistant-message {
        background-color: #212121;
        color: #ececec;
        padding: 16px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 12px auto 12px 0;
        max-width: 85%;
        font-size: 0.94em;
        line-height: 1.75;
    }
    .timestamp { font-size: 0.68em; color: #666; margin-top: 6px; }

    /* ── Input ── */
    .stTextInput > div > div > input {
        background: #2f2f2f !important;
        border: 1px solid #3f3f3f !important;
        border-radius: 16px !important;
        color: #ececec !important;
        padding: 14px 20px !important;
        font-size: 0.95em !important;
        transition: border 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #555 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: #666 !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #2f2f2f !important;
        color: #ececec !important;
        border: 1px solid #3f3f3f !important;
        border-radius: 12px !important;
        padding: 10px 18px !important;
        font-size: 0.88em !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #3f3f3f !important;
        border-color: #555 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Boxes ── */
    .info-box {
        background: #2f2f2f;
        border: 1px solid #3f3f3f;
        border-radius: 16px;
        padding: 18px 22px;
        color: #bbb;
        font-size: 0.91em;
        line-height: 1.75;
        margin: 16px 0;
    }
    .file-badge {
        background: #2f2f2f;
        border: 1px solid #4ade80;
        border-radius: 10px;
        padding: 10px 16px;
        color: #4ade80;
        font-size: 0.86em;
        font-weight: 500;
        margin: 8px 0;
    }
    .rag-source {
        background: #2a2a2a;
        border-left: 3px solid #555;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        color: #888;
        font-size: 0.82em;
        margin: 5px 0;
        line-height: 1.55;
    }

    /* ── Sidebar text ── */
    .stSidebar, .stSidebar * { color: #ccc !important; }
    label, .stSelectbox label, .stSlider label { color: #aaa !important; }

    /* Hide branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Expander fix */
    .streamlit-expanderHeader { font-size: 0.85em !important; color: #888 !important; }
    .streamlit-expanderContent { background: #2a2a2a !important; border-radius: 0 0 10px 10px !important; }

    /* ── Force sidebar always visible on ALL screen sizes ── */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: none !important;
        min-width: 240px !important;
        max-width: 280px !important;
        width: 260px !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        min-width: 240px !important;
        margin-left: 0 !important;
        transform: none !important;
    }

    /* Hide all collapse/expand arrow buttons */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="baseButton-header"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebarExpandButton"] { display: none !important; }
    button[aria-label="Close sidebar"] { display: none !important; }
    button[aria-label="Open sidebar"] { display: none !important; }
    section[data-testid="stSidebar"] > div:first-child > div:first-child > button { display: none !important; }
    .st-emotion-cache-1dp5vir { display: none !important; }
    .st-emotion-cache-1cypcdb { display: none !important; }
    .st-emotion-cache-czk5ss { display: none !important; }
    span[data-testid="stIconMaterial"] { display: none !important; }

    /* ── Responsive: make layout work on small screens ── */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            min-width: 200px !important;
            max-width: 220px !important;
            width: 210px !important;
            font-size: 0.85em !important;
        }
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────────────────
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
    'vectorstore': None,
    'rag_ready': False,
    'plain_context': None,
    'chat_store': {},         # stores ChatMessageHistory per session
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── EMBEDDINGS (cached so it only loads once) ───────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── FILE → DOCUMENTS ───────────────────────────────────────────────────────
def file_to_documents(file_bytes, ext, filename):
    """Convert uploaded file bytes into LangChain Documents for RAG."""
    if ext == "pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": i + 1}
                    ))
            return docs
        except Exception as e:
            return [Document(page_content=f"PDF error: {e}", metadata={"source": filename})]

    elif ext == "txt":
        text = file_bytes.decode("utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": filename})]

    elif ext == "csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            # Convert each row into a document for granular retrieval
            docs = []
            # Also add a summary doc
            summary = f"CSV File: {filename}\nRows: {len(df)}, Columns: {len(df.columns)}\nColumn names: {', '.join(df.columns.tolist())}\n\n"
            docs.append(Document(page_content=summary, metadata={"source": filename, "type": "summary"}))
            # Chunk rows into groups of 20
            chunk_size = 20
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                docs.append(Document(
                    page_content=chunk.to_string(index=False),
                    metadata={"source": filename, "rows": f"{i}-{i+chunk_size}"}
                ))
            return docs
        except Exception as e:
            return [Document(page_content=f"CSV error: {e}", metadata={"source": filename})]
    return []

def build_vectorstore(docs):
    """Split docs and build FAISS vectorstore."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = get_embeddings()
    return FAISS.from_documents(split_docs, embeddings)

def image_to_base64(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")

# ─── PROMPTS ────────────────────────────────────────────────────────────────

# Plain chat prompt
plain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Do NOT guess or invent facts. "
               "Remember the conversation history and refer back to it when relevant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# Contextualize question using chat history (makes follow-ups work)
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question which might reference "
     "the chat history, formulate a standalone question that can be understood "
     "without the history. Do NOT answer — just reformulate if needed, else return as-is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# RAG answer prompt (history-aware)
rag_answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert assistant for question-answering tasks. "
     "Use the retrieved context below to answer accurately and in detail. "
     "If the answer is not in the context, say so clearly. "
     "Remember prior conversation turns when relevant.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

image_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. The user has shared an image named '{filename}'. "
               "Answer their question based on the filename and context."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# ─── LLM & RESPONSE ─────────────────────────────────────────────────────────
def get_llm(model_name, temperature, max_tokens):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    )

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (or create) a ChatMessageHistory for this session."""
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

def generate_response(question, model_name, temperature, max_tokens, session_id="default"):
    """Route to conversational RAG, image, or plain chat."""
    llm    = get_llm(model_name, temperature, max_tokens)
    parser = StrOutputParser()
    history = get_session_history(session_id)

    # ── Conversational RAG (PDF / TXT / CSV) ────────────────────────────────
    if st.session_state.rag_ready and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

        # Makes retriever aware of chat history to handle follow-up questions
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )
        doc_chain = create_stuff_documents_chain(llm, rag_answer_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

        # Wrap with message history so it auto-reads & writes history
        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        result = conversational_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        return result.get("answer", ""), result.get("context", [])

    # ── Image Q&A ────────────────────────────────────────────────────────────
    elif st.session_state.file_type == "image":
        chain = image_prompt | llm | parser
        answer = chain.invoke({
            "question": question,
            "filename": st.session_state.file_name or "image",
            "chat_history": history.messages
        })
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, []

    # ── Plain conversational chat ────────────────────────────────────────────
    else:
        chain = plain_prompt | llm | parser
        answer = chain.invoke({
            "question": question,
            "chat_history": history.messages
        })
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, []

# ─── FORCE SIDEBAR OPEN (JS) ────────────────────────────────────────────────
st.markdown("""
    <script>
    // Keep sidebar always expanded
    function keepSidebarOpen() {
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            const btn = window.parent.document.querySelector('[data-testid="stSidebarExpandButton"]')
                     || window.parent.document.querySelector('button[aria-label="Open sidebar"]')
                     || window.parent.document.querySelector('[data-testid="collapsedControl"] button');
            if (btn) btn.click();
        }
    }
    // Run on load and watch for changes
    window.addEventListener('load', keepSidebarOpen);
    setInterval(keepSidebarOpen, 500);
    </script>
""", unsafe_allow_html=True)

# ─── HEADER ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
        <div style="text-align:center; padding: 60px 20px 30px 20px;">
            <h1 style="font-family:'Inter',sans-serif; font-size:2.6em; font-weight:600;
                       color:#ececec; letter-spacing:-1px; margin-bottom:8px;">
                What can I help with?
            </h1>
            <p style="color:#666; font-size:0.95em; font-weight:400;">
                Powered by Groq · LangChain · RAG
            </p>
        </div>
    """, unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── App Title ─────────────────────────────────────────────────────────
    st.markdown("""
        <div style="padding: 24px 4px 16px 4px;">
            <div style="font-size:1.1em; font-weight:600; color:#ececec; letter-spacing:-0.3px;">
                AI Assistant
            </div>
            <div style="font-size:0.72em; color:#555; margin-top:3px;">
                Groq · LangChain · RAG
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Model ─────────────────────────────────────────────────────────────
    st.markdown('<p style="font-size:0.75em;color:#888;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:6px;">Model</p>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Model", ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama2-70b-4096"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Temperature ───────────────────────────────────────────────────────
    st.markdown('<p style="font-size:0.75em;color:#888;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:6px;">Temperature</p>', unsafe_allow_html=True)
    temperature = st.slider(
        "Temperature", 0.0, 1.0,
        value=st.session_state.temperature, step=0.1,
        label_visibility="collapsed", key="temp_slider"
    )
    st.session_state.temperature = temperature

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Max Tokens ────────────────────────────────────────────────────────
    st.markdown('<p style="font-size:0.75em;color:#888;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:6px;">Max Tokens</p>', unsafe_allow_html=True)
    max_tokens = st.slider(
        "Max Tokens", 100, 2500,
        value=st.session_state.max_tokens, step=100,
        label_visibility="collapsed", key="token_slider"
    )
    st.session_state.max_tokens = max_tokens
    st.markdown(f'<p style="font-size:0.72em;color:#555;margin-top:2px;">{max_tokens} tokens · ~{max_tokens//4} words</p>', unsafe_allow_html=True)

    if max_tokens != st.session_state.prev_max_tokens and st.session_state.last_question:
        st.session_state.prev_max_tokens = max_tokens
        st.session_state.trigger_regenerate = True

    st.divider()

    # ── Active File ───────────────────────────────────────────────────────
    if st.session_state.rag_ready and st.session_state.file_name:
        st.markdown('<p style="font-size:0.75em;color:#888;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:8px;">Active File</p>', unsafe_allow_html=True)
        fname = st.session_state.file_name
        ftype = st.session_state.file_type.upper() if st.session_state.file_type else ""
        st.markdown(f"""
            <div style="background:#2a2a2a;border:1px solid #3a3a3a;border-radius:10px;
                        padding:10px 14px;margin-bottom:10px;">
                <div style="font-size:0.82em;color:#4ade80;font-weight:500;">{ftype} · RAG Active</div>
                <div style="font-size:0.78em;color:#888;margin-top:2px;word-break:break-all;">{fname}</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Remove File", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.rag_ready   = False
            st.session_state.file_name   = None
            st.session_state.file_type   = None
            st.session_state.plain_context = None
            st.session_state.chat_store  = {}
            st.rerun()
        st.divider()

    elif st.session_state.file_type == "image" and st.session_state.file_name:
        st.markdown('<p style="font-size:0.75em;color:#888;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:8px;">Active File</p>', unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background:#2a2a2a;border:1px solid #3a3a3a;border-radius:10px;
                        padding:10px 14px;margin-bottom:10px;">
                <div style="font-size:0.82em;color:#60a5fa;font-weight:500;">IMAGE</div>
                <div style="font-size:0.78em;color:#888;margin-top:2px;">{st.session_state.file_name}</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Remove Image", use_container_width=True):
            st.session_state.file_name     = None
            st.session_state.file_type     = None
            st.session_state.plain_context = None
            st.rerun()
        st.divider()

    # ── Session Stats ─────────────────────────────────────────────────────
    st.markdown('<p style="font-size:0.75em;color:#888;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:12px;">Session</p>', unsafe_allow_html=True)

    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    ai_msgs   = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("You", user_msgs)
    with c2:
        st.metric("AI", ai_msgs)

    st.divider()

    # ── Actions ───────────────────────────────────────────────────────────
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.message_count = 0
        st.session_state.last_input    = ''
        st.session_state.last_question = None
        st.session_state.chat_store    = {}
        st.session_state.input_key    += 1
        st.rerun()

# ─── MAIN CHAT AREA ──────────────────────────────────────────────────────────

# Image preview
if st.session_state.file_type == "image" and st.session_state.plain_context:
    b64 = st.session_state.plain_context
    st.markdown(f"""
        <div style="text-align:center; margin-bottom:10px;">
            <img src="data:image/png;base64,{b64}"
                 style="max-height:220px; border-radius:12px; border:2px solid #404040;"/>
            <div style="color:#aaa; font-size:0.8em; margin-top:5px;">
                🖼️ {st.session_state.file_name}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Chat messages
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-message">
                    <strong>You</strong><br>{message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Render header separately, then content natively (avoids HTML injection bug)
            st.markdown("""
                <div style="font-size:0.75em;color:#666;margin:18px 0 4px 0;">Assistant</div>
            """, unsafe_allow_html=True)
            st.markdown(message["content"])
            st.markdown(f"""
                <div class="timestamp" style="margin-bottom:8px;">{message["timestamp"]}</div>
            """, unsafe_allow_html=True)
            # Show RAG sources if available
            if message.get("sources"):
                with st.expander(f"Sources — {len(message['sources'])} chunks used"):
                    for i, src in enumerate(message["sources"]):
                        meta = src.metadata
                        label = f"Chunk {i+1}"
                        if "page" in meta:
                            label += f"  ·  Page {meta['page']}"
                        elif "rows" in meta:
                            label += f"  ·  Rows {meta['rows']}"
                        st.caption(label)
                        st.markdown(f"> {src.page_content[:300]}...")
                        if i < len(message["sources"]) - 1:
                            st.divider()
else:
    if st.session_state.rag_ready:
        st.markdown(f"""
            <div class="info-box">
                🧠 <strong>RAG Active</strong> — <em>{st.session_state.file_name}</em><br>
                Try: "Summarize this" · "What does it say about X?" · "Give me key points"
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="info-box">
                <strong>💡 Tips to get started:</strong><br><br>
                📎 Upload a PDF, TXT or CSV — AI uses <strong>Conversational RAG</strong> for accurate answers<br>
                🖼️ Upload an image and ask questions about it<br>
                🔁 Ask follow-up questions naturally — AI remembers context
            </div>
        """, unsafe_allow_html=True)

# ─── INPUT BAR ───────────────────────────────────────────────────────────────
placeholder = (
    f"Ask about {st.session_state.file_name}..."
    if st.session_state.file_name else "Type your message here..."
)

# Active file badge
if st.session_state.file_name:
    icons = {"pdf": "📄", "txt": "📝", "csv": "📊", "image": "🖼️"}
    icon  = icons.get(st.session_state.file_type, "📎")
    rag_tag = " 🧠 RAG" if st.session_state.rag_ready else ""
    st.markdown(f"""
        <div class="file-badge">
            {icon} <strong>{st.session_state.file_name}</strong>{rag_tag}
            &nbsp;<span style="font-size:0.8em;color:#aaa;">active — questions will use this file</span>
        </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([5.5, 1.8, 1.2])
with col1:
    user_input = st.text_input(
        "Message", placeholder=placeholder,
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_key}"
    )
with col2:
    if st.button("📎 Upload", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
with col3:
    send_button = st.button("Send 📤", use_container_width=True)

# Upload panel
if st.session_state.show_uploader:
    uploaded_file = st.file_uploader(
        "Choose a file — PDF, TXT, CSV or Image",
        type=["pdf", "txt", "csv", "png", "jpg", "jpeg"],
        key=f"file_upload_{st.session_state.input_key}"
    )
    if uploaded_file:
        file_bytes = uploaded_file.read()
        ext = uploaded_file.name.split(".")[-1].lower()

        if uploaded_file.name != st.session_state.file_name:
            if ext in ("pdf", "txt", "csv"):
                with st.spinner("🧠 Building RAG index... this may take a moment"):
                    docs = file_to_documents(file_bytes, ext, uploaded_file.name)
                    vs   = build_vectorstore(docs)
                    st.session_state.vectorstore = vs
                    st.session_state.rag_ready   = True
                    st.session_state.file_name   = uploaded_file.name
                    st.session_state.file_type   = ext
                    st.session_state.plain_context = None
                    st.session_state.show_uploader = False
                st.rerun()
            else:
                # Image
                st.session_state.plain_context  = image_to_base64(file_bytes)
                st.session_state.file_name      = uploaded_file.name
                st.session_state.file_type      = "image"
                st.session_state.vectorstore    = None
                st.session_state.rag_ready      = False
                st.session_state.show_uploader  = False
                st.rerun()

# ─── HANDLE SEND ─────────────────────────────────────────────────────────────
if user_input and send_button:
    st.session_state.last_input    = user_input
    st.session_state.last_question = user_input
    timestamp = datetime.now().strftime("%H:%M")

    display_content = user_input
    if st.session_state.file_name:
        icons = {"pdf": "📄", "txt": "📝", "csv": "📊", "image": "🖼️"}
        icon  = icons.get(st.session_state.file_type, "📎")
        display_content = f"{icon} <em style='font-size:0.8em;opacity:0.7;'>[{st.session_state.file_name}]</em><br>{user_input}"

    st.session_state.messages.append({
        "role": "user", "content": display_content, "timestamp": timestamp
    })
    st.session_state.message_count += 1

    with st.spinner("🤔 Thinking..."):
        try:
            answer, sources = generate_response(
                user_input, model_name, temperature, max_tokens,
                session_id="default"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().strftime("%H:%M"),
                "sources": sources
            })
            st.session_state.message_count += 1
            st.session_state.input_key += 1  # clears the text input box
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ─── AUTO-REGENERATE ON TOKEN CHANGE ─────────────────────────────────────────
if st.session_state.get('trigger_regenerate') and st.session_state.last_question:
    st.session_state.trigger_regenerate = False
    if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant':
        st.session_state.messages.pop()
        st.session_state.message_count -= 1
    with st.spinner("🔄 Regenerating..."):
        try:
            answer, sources = generate_response(
                st.session_state.last_question, model_name,
                st.session_state.temperature, st.session_state.max_tokens,
                session_id="default"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().strftime("%H:%M"),
                "sources": sources
            })
            st.session_state.message_count += 1
            st.rerun()
        except Exception as e:
            st.error(f"❌ Regeneration failed: {str(e)}")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <div style="text-align:center; color:#555; font-size:0.78em; padding:16px; letter-spacing:0.3px;">
        AI can make mistakes · Streamlit · Groq · LangChain RAG
    </div>
""", unsafe_allow_html=True)
