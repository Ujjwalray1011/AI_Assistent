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
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler

load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "AI Chat Assistant (RAG)"

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
.stApp { background-color: #212121 !important; }

/* Hide sidebar completely — settings in main area */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }
#MainMenu, footer, header         { visibility: hidden; }

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
    'search_enabled': False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── EMBEDDINGS ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── SEARCH TOOLS ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_search_tools():
    arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=400))
    wiki_tool  = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=400))
    ddg_tool   = DuckDuckGoSearchRun(name='WebSearch')
    return [ddg_tool, wiki_tool, arxiv_tool]

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

    elif st.session_state.get("search_enabled") and not st.session_state.rag_ready:
        tools  = get_search_tools()
        agent  = initialize_agent(
            tools, llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=False
        )
        st_cb  = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        answer = agent.run(question, callbacks=[st_cb])
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, []

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

# ─── TOP BAR ─────────────────────────────────────────────────────────────────
t1, t2, t3 = st.columns([4, 3, 3])
with t1:
    st.markdown("""
        <div style="padding:6px 0 2px 0;">
            <span style="font-size:1em;font-weight:600;color:#ececec;">AI Assistant</span>
            <span style="font-size:0.7em;color:#555;margin-left:8px;">Groq · LangChain · RAG</span>
        </div>
    """, unsafe_allow_html=True)
with t2:
    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    ai_msgs   = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    st.markdown(f"""
        <div style="text-align:center;padding:6px 0;">
            <span style="font-size:0.75em;color:#555;">You <b style="color:#aaa;">{user_msgs}</b>
            &nbsp;·&nbsp; AI <b style="color:#aaa;">{ai_msgs}</b></span>
        </div>
    """, unsafe_allow_html=True)
with t3:
    tb1, tb2 = st.columns(2)
    with tb1:
        if st.button("⚙ Settings", use_container_width=True, key="settings_btn"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)
            st.rerun()
    with tb2:
        if st.button("＋ New Chat", use_container_width=True, key="newchat_btn"):
            st.session_state.messages      = []
            st.session_state.message_count = 0
            st.session_state.last_question = None
            st.session_state.chat_store      = {}
            st.session_state.search_enabled  = False
            st.session_state.input_key      += 1
            st.rerun()

st.markdown("<hr style='border-color:#2a2a2a;margin:6px 0 10px 0;'>", unsafe_allow_html=True)

# ─── SETTINGS PANEL ──────────────────────────────────────────────────────────
if st.session_state.get("show_settings", False):
    with st.container():
        st.markdown("""
            <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:14px;
                        padding:18px 20px 10px 20px;margin-bottom:12px;">
        """, unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        with p1:
            st.caption("MODEL")
            model_name = st.selectbox("Model",
                ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama2-70b-4096"],
                label_visibility="collapsed", key="model_sel")
        with p2:
            st.caption("TEMPERATURE")
            temperature = st.slider("Temp", 0.0, 1.0,
                value=st.session_state.temperature, step=0.1,
                label_visibility="collapsed", key="temp_slider")
            st.session_state.temperature = temperature
        with p3:
            st.caption("MAX TOKENS")
            max_tokens = st.slider("Tokens", 100, 2500,
                value=st.session_state.max_tokens, step=100,
                label_visibility="collapsed", key="token_slider")
            st.session_state.max_tokens = max_tokens
            st.caption(f"{max_tokens} tokens · ~{max_tokens//4} words")

        if max_tokens != st.session_state.prev_max_tokens and st.session_state.last_question:
            st.session_state.prev_max_tokens = max_tokens
            st.session_state.trigger_regenerate = True

        if st.session_state.file_name:
            st.markdown("<hr style='border-color:#2a2a2a;margin:8px 0;'>", unsafe_allow_html=True)
            fa, fb = st.columns([5,1])
            with fa:
                color = "#4ade80" if st.session_state.rag_ready else "#60a5fa"
                tag   = "RAG" if st.session_state.rag_ready else "Image"
                st.markdown(f"""
                    <span style="font-size:0.8em;color:{color};font-weight:500;">{tag}</span>
                    <span style="font-size:0.8em;color:#666;margin-left:6px;">{st.session_state.file_name}</span>
                """, unsafe_allow_html=True)
            with fb:
                if st.button("✕", use_container_width=True, key="remove_file"):
                    st.session_state.vectorstore   = None
                    st.session_state.rag_ready     = False
                    st.session_state.file_name     = None
                    st.session_state.file_type     = None
                    st.session_state.plain_context = None
                    st.session_state.chat_store    = {}
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
else:
    model_name  = st.session_state.get("model_sel", "llama-3.1-8b-instant")
    temperature = st.session_state.temperature
    max_tokens  = st.session_state.max_tokens

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
if st.session_state.file_name:
    placeholder = f"Ask about {st.session_state.file_name}..."
elif st.session_state.search_enabled:
    placeholder = "Search the web, Wikipedia or Arxiv..."
else:
    placeholder = "Type your message here..."

if st.session_state.search_enabled:
    st.markdown(
        '<div style="background:#1a2a1a;border:1px solid #2d4a2d;border-radius:8px;'
        'padding:6px 14px;margin-bottom:6px;font-size:0.8em;color:#4ade80;">'
        '🔍 Web Search Active — DuckDuckGo · Wikipedia · Arxiv</div>',
        unsafe_allow_html=True
    )

col1, col2, col3, col4 = st.columns([4.2, 1.6, 1.6, 1.2])
with col1:
    user_input = st.text_input("Message", placeholder=placeholder,
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_key}")
with col2:
    s_label = "🔍 ON" if st.session_state.search_enabled else "🔍 Search"
    s_style = "background:#1a2a1a!important;border-color:#2d4a2d!important;color:#4ade80!important;" if st.session_state.search_enabled else ""
    if st.button(s_label, use_container_width=True, key="search_toggle"):
        st.session_state.search_enabled = not st.session_state.search_enabled
        st.rerun()
with col3:
    if st.button("📎 Upload", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
with col4:
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
