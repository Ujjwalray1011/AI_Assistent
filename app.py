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
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Sora', sans-serif !important;
        background-color: #f0ede8 !important;
    }
    .main { background-color: #f0ede8 !important; }
    section[data-testid="stSidebar"] {
        background-color: #e8e4de !important;
        border-right: 1px solid #d4cfc8 !important;
    }

    /* ── User bubble ── */
    .user-message {
        background-color: #1a1a1a;
        color: #f5f5f5;
        padding: 14px 20px;
        border-radius: 20px 20px 4px 20px;
        margin: 16px 0 16px auto;
        max-width: 72%;
        font-size: 0.95em;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    /* ── AI bubble ── */
    .assistant-message {
        background-color: #ffffff;
        color: #1a1a1a;
        padding: 16px 22px;
        border-radius: 20px 20px 20px 4px;
        margin: 16px auto 16px 0;
        max-width: 82%;
        font-size: 0.95em;
        line-height: 1.7;
        border: 1px solid #e0dbd4;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .timestamp {
        font-size: 0.68em;
        color: #b0a89e;
        margin-top: 6px;
        letter-spacing: 0.3px;
    }

    /* ── Input box ── */
    .stTextInput > div > div > input {
        border-radius: 16px !important;
        border: 1.5px solid #d4cfc8 !important;
        background: #ffffff !important;
        padding: 14px 20px !important;
        font-size: 0.95em !important;
        font-family: 'Sora', sans-serif !important;
        color: #1a1a1a !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1a1a1a !important;
        box-shadow: 0 0 0 3px rgba(26,26,26,0.08) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #b0a89e !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #1a1a1a !important;
        color: #f5f5f5 !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 10px 20px !important;
        font-size: 0.88em !important;
        font-weight: 500 !important;
        font-family: 'Sora', sans-serif !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.2px !important;
    }
    .stButton > button:hover {
        background: #333333 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    /* ── Sidebar elements ── */
    .stSelectbox > div > div,
    .stSlider > div {
        background: transparent !important;
    }

    /* ── Info box ── */
    .info-box {
        background-color: #ffffff;
        border-left: 3px solid #1a1a1a;
        padding: 18px 20px;
        border-radius: 0 14px 14px 0;
        margin: 20px 0;
        color: #3a3a3a;
        font-size: 0.92em;
        line-height: 1.7;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .file-badge {
        background-color: #f8f6f2;
        border: 1.5px solid #1a1a1a;
        border-radius: 12px;
        padding: 10px 16px;
        margin: 10px 0;
        color: #1a1a1a;
        font-size: 0.87em;
        font-weight: 500;
    }
    .rag-source {
        background-color: #f8f6f2;
        border-left: 3px solid #1a1a1a;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        color: #5a5a5a;
        font-size: 0.83em;
        margin: 6px 0;
        line-height: 1.5;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
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

# ─── HEADER ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
        <div style="text-align:center; padding: 60px 20px 30px 20px;">
            <h1 style="font-family:'Sora',sans-serif; font-size:2.6em; font-weight:700;
                       color:#1a1a1a; letter-spacing:-1px; margin-bottom:8px;">
                What can I help with?
            </h1>
            <p style="color:#9a9288; font-size:1em; font-weight:400;">
                Powered by Groq · LangChain · RAG
            </p>
        </div>
    """, unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    model_name = st.selectbox(
        "🔧 Select Model",
        ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama2-70b-4096"],
        help="Choose the AI model"
    )

    st.markdown("---")

    temperature = st.slider(
        "🌡️ Temperature", 0.0, 1.0,
        value=st.session_state.temperature, step=0.1,
        help="Lower = focused, Higher = creative",
        key="temp_slider"
    )
    st.session_state.temperature = temperature

    max_tokens = st.slider(
        "📏 Max Tokens", 100, 2500,
        value=st.session_state.max_tokens, step=100,
        help="500 = short · 1500 = detailed · 2500 = very long",
        key="token_slider"
    )
    st.session_state.max_tokens = max_tokens
    st.caption(f"ℹ️ **{max_tokens}** tokens (~{max_tokens // 4} words)")

    if max_tokens != st.session_state.prev_max_tokens and st.session_state.last_question:
        st.session_state.prev_max_tokens = max_tokens
        st.session_state.trigger_regenerate = True

    st.markdown("---")

    # RAG status indicator in sidebar
    if st.session_state.rag_ready:
        st.success(f"🧠 RAG Active\n\n📄 {st.session_state.file_name}")
        st.markdown("")
        if st.button("🗑️ Remove File", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.rag_ready = False
            st.session_state.file_name = None
            st.session_state.file_type = None
            st.session_state.plain_context = None
            st.session_state.chat_store = {}
            st.rerun()
    elif st.session_state.file_type == "image":
        st.info(f"🖼️ Image Loaded\n\n{st.session_state.file_name}")
        st.markdown("")
        if st.button("🗑️ Remove Image", use_container_width=True):
            st.session_state.file_name = None
            st.session_state.file_type = None
            st.session_state.plain_context = None
            st.rerun()

    st.markdown("---")

    # ── Premium Session Stats ──────────────────────────────────────────────
    user_msgs   = sum(1 for m in st.session_state.messages if m["role"] == "user")
    ai_msgs     = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    model_short = model_name.split('-')[0].upper()

    st.markdown("⚡ **SESSION STATS**")

    # Message counters
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="💬 You", value=user_msgs)
    with c2:
        st.metric(label="🤖 AI", value=ai_msgs)

    # Info rows
    st.markdown(f"**🧠 Model:** `{model_short}`")
    st.markdown(f"**🌡️ Temp:** `{temperature}`")
    if st.session_state.file_name:
        st.markdown(f"**📁 File:** 🟢 `{st.session_state.file_name[:20]}`")
    else:
        st.markdown("**📁 File:** ⚪ None")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.session_state.last_input = ''
        st.session_state.last_question = None
        st.session_state.chat_store = {}   # clear conversation memory
        st.session_state.input_key += 1
        st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.markdown("""
        **AI Chat Assistant** — Groq + LangChain + RAG

        **Features:**
        - 💬 Chat with AI
        - 🧠 RAG for PDF, TXT, CSV (semantic search)
        - 🖼️ Image Q&A
        - 📏 Token & temperature control
        - 🔄 Auto-regenerate on token change
        """)

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
            st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant</strong><br>{message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)
            # Show RAG sources if available
            if message.get("sources"):
                with st.expander(f"📚 View {len(message['sources'])} source chunks used"):
                    for i, src in enumerate(message["sources"]):
                        meta = src.metadata
                        label = f"Chunk {i+1}"
                        if "page" in meta:
                            label += f" — Page {meta['page']}"
                        elif "rows" in meta:
                            label += f" — Rows {meta['rows']}"
                        st.markdown(f"""
                            <div class="rag-source">
                                <strong>{label}</strong><br>
                                {src.page_content[:300]}...
                            </div>
                        """, unsafe_allow_html=True)
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
    <div style="text-align:center; color:#b0a89e; font-size:0.78em; padding:16px;
                font-family:'Sora',sans-serif; letter-spacing:0.3px;">
        AI can make mistakes · Built with Streamlit · Groq · LangChain RAG
    </div>
""", unsafe_allow_html=True)
