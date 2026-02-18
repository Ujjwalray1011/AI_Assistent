import os
import io
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

# ============================================================
# ALL FUNCTIONS DEFINED BEFORE st.markdown / CSS BLOCKS
# This prevents inspect.getsource tokenize.TokenError on
# Streamlit Cloud when @st.cache_resource hashes the function.
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def extract_text_from_pdf(file_bytes):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return [
            Document(
                page_content=p.extract_text() or "",
                metadata={"source": "pdf", "page": i + 1},
            )
            for i, p in enumerate(reader.pages)
            if p.extract_text()
        ]
    except Exception as exc:
        return [Document(page_content="PDF error: " + str(exc))]


def extract_text_from_txt(file_bytes):
    return [Document(page_content=file_bytes.decode("utf-8", errors="ignore"))]


def extract_text_from_csv(file_bytes):
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        docs = [Document(page_content="CSV: " + str(len(df)) + " rows, columns: " + ", ".join(df.columns))]
        for i in range(0, len(df), 20):
            docs.append(
                Document(
                    page_content=df.iloc[i : i + 20].to_string(index=False),
                    metadata={"rows": str(i) + "-" + str(i + 20)},
                )
            )
        return docs
    except Exception as exc:
        return [Document(page_content="CSV error: " + str(exc))]


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return FAISS.from_documents(splitter.split_documents(docs), get_embeddings())


plain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Do not guess or invent facts."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given chat history and latest question, reformulate as standalone question. "
        "Return as-is if not needed.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

rag_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context below. Be detailed and accurate.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def get_llm(model_name, temperature, max_tokens):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY"),
    )


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]


def generate_response(question, model_name, temperature, max_tokens, session_id="default"):
    llm = get_llm(model_name, temperature, max_tokens)
    parser = StrOutputParser()
    history = get_session_history(session_id)

    if st.session_state.rag_ready and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        har = create_history_aware_retriever(llm, retriever, contextualize_prompt)
        doc_chain = create_stuff_documents_chain(llm, rag_answer_prompt)
        rag_chain = create_retrieval_chain(har, doc_chain)
        conv_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        result = conv_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return result.get("answer", ""), result.get("context", [])
    else:
        answer = (plain_prompt | llm | parser).invoke(
            {"question": question, "chat_history": history.messages}
        )
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, []


# ============================================================
# PAGE CONFIG  (must come before any st.* rendering calls)
# ============================================================
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS  — built as a plain string (no triple-quotes) to avoid
#         tokenizer issues with inspect.getsource
# ============================================================
_css_parts = [
    "<style>",
    "@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');",
    "* { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }",
    ".stApp { background-color: #f0f0f8 !important; }",
    "section[data-testid='stSidebar'] { display: none !important; }",
    "[data-testid='collapsedControl'] { display: none !important; }",
    "#MainMenu, footer, header { visibility: hidden; }",
    ".block-container {",
    "    max-width: 540px !important; margin: 0 auto !important; padding: 0 !important;",
    "    background: #ffffff; min-height: 100vh; box-shadow: 0 0 40px rgba(0,0,0,0.12);",
    "}",
    ".chat-header {",
    "    background: linear-gradient(135deg, #5b21b6 0%, #4c1d95 100%);",
    "    padding: 16px 20px; display: flex; align-items: center; gap: 12px;",
    "    position: sticky; top: 0; z-index: 100;",
    "}",
    ".header-avatar {",
    "    width: 42px; height: 42px; background: rgba(255,255,255,0.2);",
    "    border-radius: 50%; display: flex; align-items: center; justify-content: center;",
    "    color: #fff; font-size: 1.1em; font-weight: 700;",
    "}",
    ".header-name { font-size: 1em; font-weight: 600; color: #fff; }",
    ".header-status { font-size: 0.72em; color: #a78bfa; margin-top: 1px;",
    "    display: flex; align-items: center; gap: 5px; }",
    ".status-dot { width: 7px; height: 7px; border-radius: 50%;",
    "    background: #34d399; display: inline-block; }",
    ".messages-area { padding: 20px 16px; display: flex; flex-direction: column; gap: 4px; }",
    ".bot-bubble-wrap { display: flex; align-items: flex-end; gap: 8px; margin: 6px 0; }",
    ".bot-avatar {",
    "    width: 30px; height: 30px; border-radius: 50%;",
    "    background: linear-gradient(135deg, #5b21b6, #7c3aed);",
    "    flex-shrink: 0; display: flex; align-items: center; justify-content: center;",
    "    color: #fff; font-size: 0.65em; font-weight: 700;",
    "}",
    ".bot-bubble {",
    "    background: #f3f4f6; color: #111827; padding: 11px 15px;",
    "    border-radius: 18px 18px 18px 4px; font-size: 0.88em;",
    "    line-height: 1.6; max-width: 78%;",
    "}",
    ".user-bubble-wrap {",
    "    display: flex; align-items: flex-end; gap: 8px;",
    "    margin: 6px 0; flex-direction: row-reverse;",
    "}",
    ".user-avatar {",
    "    width: 30px; height: 30px; border-radius: 50%;",
    "    background: #d1d5db; flex-shrink: 0;",
    "    display: flex; align-items: center; justify-content: center;",
    "    color: #374151; font-size: 0.65em; font-weight: 700;",
    "}",
    ".user-bubble {",
    "    background: #f3f4f6; color: #111827; padding: 11px 15px;",
    "    border-radius: 18px 18px 4px 18px; font-size: 0.88em;",
    "    line-height: 1.6; max-width: 78%;",
    "}",
    ".bubble-meta { font-size: 0.66em; color: #9ca3af; margin: 2px 0 6px 38px; }",
    ".bubble-meta.user-meta { text-align: right; margin: 2px 38px 6px 0; }",
    ".read-tick { color: #6d28d9; font-weight: 600; }",
    ".source-note { font-size: 0.68em; color: #7c3aed; margin: 0 0 10px 38px; font-style: italic; }",
    ".chips-row { display: flex; gap: 8px; flex-wrap: wrap; padding: 8px 16px 12px; }",
    ".chip {",
    "    background: #f3f4f6; border: 1px solid #e5e7eb;",
    "    border-radius: 20px; padding: 6px 14px;",
    "    font-size: 0.78em; color: #374151; cursor: pointer; white-space: nowrap;",
    "}",
    ".stTextInput > div > div > input {",
    "    background: #f9fafb !important; border: 1px solid #e5e7eb !important;",
    "    border-radius: 24px !important; color: #111827 !important;",
    "    padding: 11px 18px !important; font-size: 0.88em !important;",
    "}",
    ".stTextInput > div > div > input:focus {",
    "    border-color: #7c3aed !important;",
    "    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;",
    "}",
    ".stTextInput > div > div > input::placeholder { color: #9ca3af !important; }",
    ".stTextInput > label { display: none !important; }",
    ".stButton > button {",
    "    background: #5b21b6 !important; color: #fff !important;",
    "    border: none !important; border-radius: 24px !important;",
    "    padding: 10px 20px !important; font-size: 0.85em !important;",
    "    font-weight: 500 !important; transition: all 0.18s !important;",
    "}",
    ".stButton > button:hover { background: #4c1d95 !important; }",
    ".btn-ghost .stButton > button {",
    "    background: #f3f4f6 !important; color: #374151 !important;",
    "    border: 1px solid #e5e7eb !important; border-radius: 24px !important;",
    "    padding: 9px 16px !important; font-size: 0.82em !important; font-weight: 400 !important;",
    "}",
    ".btn-ghost .stButton > button:hover { background: #e5e7eb !important; }",
    ".settings-panel { background: #faf9ff; border-bottom: 1px solid #ede9fe; padding: 14px 18px; }",
    ".file-badge {",
    "    margin: 8px 16px; background: #ede9fe; border: 1px solid #c4b5fd;",
    "    border-radius: 8px; padding: 7px 12px; color: #5b21b6; font-size: 0.8em;",
    "}",
    ".welcome-hero { text-align: center; padding: 40px 24px 24px; }",
    ".welcome-hero h2 { font-size: 1.3em; font-weight: 600; color: #111827; margin-bottom: 6px; }",
    ".welcome-hero p { font-size: 0.84em; color: #6b7280; }",
    ".info-card {",
    "    background: #faf9ff; border: 1px solid #ede9fe;",
    "    border-radius: 12px; padding: 14px 16px;",
    "    margin: 0 16px 12px; font-size: 0.83em; color: #6b7280; line-height: 1.7;",
    "}",
    ".chat-footer {",
    "    text-align: center; color: #9ca3af; font-size: 0.68em;",
    "    padding: 12px; border-top: 1px solid #f3f4f6;",
    "}",
    "div[data-testid='column'] { padding: 0 4px !important; }",
    "</style>",
]
st.markdown("\n".join(_css_parts), unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
_defaults = {
    "messages": [],
    "message_count": 0,
    "max_tokens": 500,
    "temperature": 0.7,
    "last_question": None,
    "prev_max_tokens": 500,
    "trigger_regenerate": False,
    "input_key": 0,
    "file_name": None,
    "file_type": None,
    "show_uploader": False,
    "show_settings": False,
    "vectorstore": None,
    "rag_ready": False,
    "plain_context": None,
    "chat_store": {},
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="chat-header">'
    '<div class="header-avatar">AI</div>'
    '<div>'
    '<div class="header-name">AI Assistant</div>'
    '<div class="header-status">'
    '<span class="status-dot"></span>'
    '<span>Online</span>'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# SETTINGS PANEL
# ============================================================
if st.session_state.get("show_settings", False):
    st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.caption("MODEL")
        model_name = st.selectbox(
            "Model",
            ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama2-70b-4096"],
            label_visibility="collapsed",
            key="model_sel",
        )
    with p2:
        st.caption("TEMPERATURE")
        temperature = st.slider(
            "Temp",
            0.0,
            1.0,
            value=st.session_state.temperature,
            step=0.1,
            label_visibility="collapsed",
            key="temp_slider",
        )
        st.session_state.temperature = temperature
    with p3:
        st.caption("MAX TOKENS")
        max_tokens = st.slider(
            "Tokens",
            100,
            2500,
            value=st.session_state.max_tokens,
            step=100,
            label_visibility="collapsed",
            key="token_slider",
        )
        st.session_state.max_tokens = max_tokens

    if max_tokens != st.session_state.prev_max_tokens and st.session_state.last_question:
        st.session_state.prev_max_tokens = max_tokens
        st.session_state.trigger_regenerate = True

    if st.session_state.file_name:
        fa, fb = st.columns([5, 1])
        with fa:
            st.markdown(
                '<span style="font-size:0.8em;color:#5b21b6;font-weight:500;">RAG Active</span>'
                '<span style="font-size:0.8em;color:#6b7280;margin-left:6px;">'
                + st.session_state.file_name
                + "</span>",
                unsafe_allow_html=True,
            )
        with fb:
            if st.button("Remove", use_container_width=True, key="remove_file"):
                st.session_state.vectorstore = None
                st.session_state.rag_ready = False
                st.session_state.file_name = None
                st.session_state.file_type = None
                st.session_state.plain_context = None
                st.session_state.chat_store = {}
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
else:
    model_name = st.session_state.get("model_sel", "llama-3.1-8b-instant")
    temperature = st.session_state.temperature
    max_tokens = st.session_state.max_tokens

# ============================================================
# FILE BADGE
# ============================================================
if st.session_state.rag_ready and not st.session_state.get("show_settings"):
    st.markdown(
        '<div class="file-badge"><strong>RAG Active</strong> — '
        + st.session_state.file_name
        + "</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# TOP CONTROLS ROW
# ============================================================
_user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
_ai_msgs = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

tc1, tc2, tc3 = st.columns([3, 1, 1])
with tc1:
    st.markdown(
        '<div style="padding:8px 4px;font-size:0.75em;color:#9ca3af;">'
        "Messages — You: "
        + str(_user_msgs)
        + " &middot; AI: "
        + str(_ai_msgs)
        + "</div>",
        unsafe_allow_html=True,
    )
with tc2:
    st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
    if st.button("Settings", use_container_width=True, key="settings_btn"):
        st.session_state.show_settings = not st.session_state.get("show_settings", False)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with tc3:
    st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
    if st.button("New Chat", use_container_width=True, key="newchat_btn"):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.session_state.last_question = None
        st.session_state.chat_store = {}
        st.session_state.input_key += 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# CHAT MESSAGES
# ============================================================
st.markdown('<div class="messages-area">', unsafe_allow_html=True)

if st.session_state.messages:
    for _msg in st.session_state.messages:
        _ts = _msg.get("timestamp", "")
        if _msg["role"] == "user":
            st.markdown(
                '<div class="user-bubble-wrap">'
                '<div class="user-avatar">You</div>'
                '<div class="user-bubble">' + _msg["content"] + "</div>"
                "</div>"
                '<div class="bubble-meta user-meta">'
                + _ts
                + ' <span class="read-tick">Read</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="bot-bubble-wrap">'
                '<div class="bot-avatar">AI</div>'
                '<div class="bot-bubble">' + _msg["content"] + "</div>"
                "</div>"
                '<div class="bubble-meta">' + _ts + "</div>",
                unsafe_allow_html=True,
            )
            if _msg.get("sources"):
                _n = len(_msg["sources"])
                st.markdown(
                    '<div class="source-note">Answered from '
                    + str(_n)
                    + " section"
                    + ("s" if _n > 1 else "")
                    + " of the document</div>",
                    unsafe_allow_html=True,
                )
else:
    st.markdown(
        '<div class="welcome-hero">'
        "<h2>What can I help with?</h2>"
        "<p>Powered by Groq &middot; LangChain &middot; RAG</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-card">'
        "<strong>Getting started</strong><br>"
        "Upload a PDF, TXT or CSV file to activate conversational RAG for accurate document answers.<br>"
        "Ask follow-up questions — the AI remembers the full conversation."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# QUICK CHIPS (welcome only)
# ============================================================
if not st.session_state.messages:
    st.markdown(
        '<div class="chips-row">'
        '<span class="chip">What is this app?</span>'
        '<span class="chip">Pricing</span>'
        '<span class="chip">Help</span>'
        "</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# INPUT BAR
# ============================================================
_ph = (
    "Ask about " + st.session_state.file_name + "..."
    if st.session_state.file_name
    else "Type your message here..."
)

col1, col2, col3 = st.columns([5, 1.4, 1.2])
with col1:
    user_input = st.text_input(
        "Message",
        placeholder=_ph,
        label_visibility="collapsed",
        key="user_input_" + str(st.session_state.input_key),
    )
with col2:
    if st.button("Upload", use_container_width=True, key="upload_btn"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
with col3:
    send_button = st.button("Send", use_container_width=True)

# ============================================================
# UPLOAD PANEL
# ============================================================
if st.session_state.show_uploader:
    uploaded_file = st.file_uploader(
        "Choose a file — PDF, TXT, or CSV",
        type=["pdf", "txt", "csv"],
        key="file_upload_" + str(st.session_state.input_key),
    )
    if uploaded_file and uploaded_file.name != st.session_state.file_name:
        _file_bytes = uploaded_file.read()
        _ext = uploaded_file.name.split(".")[-1].lower()
        with st.spinner("Building RAG index..."):
            _docs_map = {
                "pdf": extract_text_from_pdf,
                "txt": extract_text_from_txt,
                "csv": extract_text_from_csv,
            }
            _docs = _docs_map[_ext](_file_bytes)
            st.session_state.vectorstore = build_vectorstore(_docs)
            st.session_state.rag_ready = True
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_type = _ext
            st.session_state.plain_context = None
            st.session_state.show_uploader = False
        st.rerun()

# ============================================================
# HANDLE SEND
# ============================================================
if user_input and send_button:
    st.session_state.last_question = user_input
    _ts_now = datetime.now().strftime("%H:%M")
    _display = user_input
    if st.session_state.file_name:
        _display = (
            '<span style="font-size:0.75em;color:#7c3aed;">['
            + st.session_state.file_name
            + "]</span><br>"
            + user_input
        )
    st.session_state.messages.append(
        {"role": "user", "content": _display, "timestamp": _ts_now}
    )
    st.session_state.message_count += 1

    with st.spinner("Thinking..."):
        try:
            _answer, _sources = generate_response(
                user_input, model_name, temperature, max_tokens, session_id="default"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": _answer,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "sources": _sources,
                }
            )
            st.session_state.message_count += 1
            st.session_state.input_key += 1
            st.rerun()
        except Exception as _e:
            st.error("Error: " + str(_e))

# ============================================================
# AUTO-REGENERATE
# ============================================================
if st.session_state.get("trigger_regenerate") and st.session_state.last_question:
    st.session_state.trigger_regenerate = False
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages.pop()
        st.session_state.message_count -= 1
    with st.spinner("Regenerating..."):
        try:
            _answer, _sources = generate_response(
                st.session_state.last_question,
                model_name,
                st.session_state.temperature,
                st.session_state.max_tokens,
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": _answer,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "sources": _sources,
                }
            )
            st.session_state.message_count += 1
            st.rerun()
        except Exception as _e:
            st.error("Regeneration failed: " + str(_e))

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="chat-footer">'
    "AI can make mistakes &middot; Streamlit &middot; Groq &middot; LangChain RAG"
    "</div>",
    unsafe_allow_html=True,
)
