import os
import io
import base64
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
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
    .main { background-color: #0e1117; }
    .user-message {
        background-color: #007bff; color: white;
        padding: 15px 20px; border-radius: 18px;
        margin: 10px 0; max-width: 80%; margin-left: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: #2d2d2d; color: #e8e8e8;
        padding: 15px 20px; border-radius: 18px;
        margin: 10px 0; max-width: 80%;
        border: 1px solid #404040;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px; border-radius: 15px; margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title { color: white; font-size: 2.5em; font-weight: 700; margin: 0; text-align: center; }
    .header-subtitle { color: rgba(255,255,255,0.9); font-size: 1.1em; text-align: center; margin-top: 10px; }
    .stTextInput > div > div > input {
        border-radius: 25px; border: 2px solid #e1e8ed;
        padding: 12px 20px; font-size: 16px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 25px;
        padding: 12px 30px; font-size: 16px; font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.3);
    }
    .timestamp { font-size: 0.75em; color: #a0a0a0; margin-top: 5px; }
    .info-box {
        background-color: #2a3f5f; border-left: 4px solid #3498db;
        padding: 15px; border-radius: 5px; margin: 20px 0; color: #e8e8e8;
    }
    .metric-card {
        background-color: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin: 10px 0;
    }
    .file-badge {
        background-color: #1a2f1a; border: 1px solid #27ae60;
        border-radius: 10px; padding: 10px 14px; margin: 8px 0;
        color: #2ecc71; font-size: 0.9em;
    }
    .rag-source {
        background-color: #1e1e2e; border-left: 3px solid #667eea;
        padding: 10px 14px; border-radius: 5px;
        color: #a0a0c0; font-size: 0.85em; margin: 5px 0;
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
    'vectorstore': None,      # FAISS vectorstore for RAG
    'rag_ready': False,       # whether RAG is ready
    'plain_context': None,    # fallback plain text for images
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
plain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Do NOT guess or invent facts."),
    ("user", "Question: {question}")
])

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context below.
Be detailed and accurate. If the answer is not in the context, say so clearly.

<context>
{context}
</context>

Question: {input}
""")

image_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. The user has shared an image named '{filename}'. "
               "Answer their question as best you can based on the filename and context."),
    ("user", "Question: {question}")
])

# ─── LLM & RESPONSE ─────────────────────────────────────────────────────────
def get_llm(model_name, temperature, max_tokens):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    )

def generate_response(question, model_name, temperature, max_tokens):
    """Route to RAG, image, or plain response based on session state."""
    llm = get_llm(model_name, temperature, max_tokens)
    parser = StrOutputParser()

    # ── RAG path (PDF / TXT / CSV) ──────────────────────────────────────────
    if st.session_state.rag_ready and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        doc_chain   = create_stuff_documents_chain(llm, rag_prompt)
        rag_chain   = create_retrieval_chain(retriever, doc_chain)
        result      = rag_chain.invoke({"input": question})
        answer      = result.get("answer", "")
        source_docs = result.get("context", [])
        return answer, source_docs

    # ── Image path ───────────────────────────────────────────────────────────
    elif st.session_state.file_type == "image":
        chain = image_prompt | llm | parser
        answer = chain.invoke({
            "question": question,
            "filename": st.session_state.file_name or "image"
        })
        return answer, []

    # ── Plain chat ───────────────────────────────────────────────────────────
    else:
        chain = plain_prompt | llm | parser
        answer = chain.invoke({"question": question})
        return answer, []

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Chat Assistant</h1>
        <p class="header-subtitle">Powered by Groq · LangChain · RAG</p>
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
        st.markdown(f"""
            <div style="background:#1a2f1a;border:1px solid #27ae60;border-radius:8px;
                        padding:10px;color:#2ecc71;font-size:0.85em;">
                🧠 <strong>RAG Active</strong><br>
                <span style="color:#aaa;">📄 {st.session_state.file_name}</span><br>
                <span style="color:#aaa;font-size:0.8em;">Semantic search enabled</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("🗑️ Remove File", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.rag_ready = False
            st.session_state.file_name = None
            st.session_state.file_type = None
            st.session_state.plain_context = None
            st.rerun()
    elif st.session_state.file_type == "image":
        st.markdown(f"""
            <div style="background:#1a1a2f;border:1px solid #667eea;border-radius:8px;
                        padding:10px;color:#a0a0ff;font-size:0.85em;">
                🖼️ <strong>Image Loaded</strong><br>
                <span style="color:#aaa;">{st.session_state.file_name}</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("🗑️ Remove Image", use_container_width=True):
            st.session_state.file_name = None
            st.session_state.file_type = None
            st.session_state.plain_context = None
            st.rerun()

    st.markdown("---")

    st.markdown("### 📊 Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.9em;color:#7f8c8d;">Messages</div>
                <div style="font-size:1.8em;font-weight:bold;color:#667eea;">{st.session_state.message_count}</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.9em;color:#7f8c8d;">Model</div>
                <div style="font-size:1em;font-weight:bold;color:#764ba2;">{model_name.split('-')[0].upper()}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.session_state.last_input = ''
        st.session_state.last_question = None
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
                <strong>🧠 RAG Ready!</strong> Semantic search is active for
                <strong>{st.session_state.file_name}</strong>.<br><br>
                💡 Try: <em>"Summarize this document"</em> or <em>"What does it say about X?"</em>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="info-box">
                <strong>👋 Welcome!</strong> Ask me anything or upload a file.
                <br><br>💡 <strong>Tips:</strong>
                <ul>
                    <li>📎 Upload PDF, TXT, or CSV → AI uses <strong>RAG</strong> for accurate answers</li>
                    <li>🖼️ Upload an image → ask questions about it</li>
                    <li>Adjust temperature &amp; tokens for better results</li>
                </ul>
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
                user_input, model_name, temperature, max_tokens
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
                st.session_state.temperature, st.session_state.max_tokens
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
    <div style="text-align:center; color:#95a5a6; font-size:0.9em; padding:20px;">
        Built with ❤️ using Streamlit · LangChain · Groq · FAISS RAG
    </div>
""", unsafe_allow_html=True)
