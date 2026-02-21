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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime

load_dotenv()

st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #212121 !important; }
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
.user-message { background: #2f2f2f; color: #ececec; padding: 12px 16px; border-radius: 18px 18px 4px 18px; margin: 10px 0 10px auto; max-width: 75%; font-size: 0.93em; }
.msg-label { font-size: 0.7em; color: #555; margin-bottom: 4px; }
.timestamp { font-size: 0.67em; color: #555; margin-top: 4px; }
.stTextInput > div > div > input { background: #2f2f2f !important; border: 1px solid #3a3a3a !important; border-radius: 14px !important; color: #ececec !important; padding: 13px 18px !important; font-size: 0.93em !important; }
.stTextInput > div > div > input:focus { border-color: #555 !important; }
.stTextInput > div > div > input::placeholder { color: #555 !important; }
.stButton > button { background: #2f2f2f !important; color: #ececec !important; border: 1px solid #3a3a3a !important; border-radius: 12px !important; padding: 10px 16px !important; font-size: 0.86em !important; width: 100%; }
.stButton > button:hover { background: #3a3a3a !important; }
.file-badge { background: #1e2a1e; border: 1px solid #2d4a2d; border-radius: 10px; padding: 8px 14px; color: #4ade80; font-size: 0.83em; margin: 8px 0; }
.settings-box { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 14px; padding: 18px 20px 10px 20px; margin-bottom: 12px; }
hr { border-color: #2f2f2f !important; }
</style>
""", unsafe_allow_html=True)

defaults = {'messages': [], 'message_count': 0, 'max_tokens': 500, 'temperature': 0.7, 'last_question': None, 
           'prev_max_tokens': 500, 'trigger_regenerate': False, 'input_key': 0, 'file_name': None, 'file_type': None,
           'show_uploader': False, 'show_settings': False, 'vectorstore': None, 'rag_ready': False, 'chat_store': {}}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def extract_text_from_pdf(file_bytes):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(page_content=text, metadata={"source": "pdf", "page": i+1}))
        return docs if docs else [Document(page_content="Error: No text found in PDF")]
    except Exception as e:
        return [Document(page_content=f"PDF error: {e}")]

def extract_text_from_txt(file_bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    return [Document(page_content=text)] if text.strip() else [Document(page_content="Error: Empty file")]

def extract_text_from_csv(file_bytes):
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        docs = [Document(page_content=f"CSV: {len(df)} rows, columns: {', '.join(df.columns)}")]
        for i in range(0, len(df), 20):
            docs.append(Document(page_content=df.iloc[i:i+20].to_string(index=False), metadata={"rows": f"{i}-{i+20}"}))
        return docs
    except Exception as e:
        return [Document(page_content=f"CSV error: {e}")]

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    if not split_docs:
        raise ValueError("No content to index")
    return Chroma.from_documents(split_docs, get_embeddings())

plain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's date is {current_date}. Always use this date for context."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformulate question as standalone. Return as-is if standalone already."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
rag_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def get_llm(model_name, temperature, max_tokens):
    return ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens, 
                   api_key=st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY"))

def get_session_history(session_id: str):
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

def generate_response(question, model_name, temperature, max_tokens, session_id="default"):
    llm = get_llm(model_name, temperature, max_tokens)
    parser = StrOutputParser()
    history = get_session_history(session_id)
    
    if st.session_state.rag_ready and st.session_state.vectorstore:
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain, create_history_aware_retriever
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        har = create_history_aware_retriever(llm, retriever, contextualize_prompt)
        doc_chain = create_stuff_documents_chain(llm, rag_answer_prompt)
        rag_chain = create_retrieval_chain(har, doc_chain)
        conv_chain = RunnableWithMessageHistory(rag_chain, get_session_history,
                                               input_messages_key="input", history_messages_key="chat_history",
                                               output_messages_key="answer")
        result = conv_chain.invoke({"input": question}, config={"configurable": {"session_id": session_id}})
        return result.get("answer", ""), result.get("context", [])
    else:
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        answer = (plain_prompt | llm | parser).invoke({"question": question, "chat_history": history.messages, "current_date": current_date})
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, []

user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
ai_msgs = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

t1, t2, t3 = st.columns([4, 3, 3])
with t1:
    st.markdown('<div style="padding:6px 0 2px 0;"><span style="font-size:1em;font-weight:600;color:#ececec;">AI Assistant</span>'
               '<span style="font-size:0.7em;color:#555;margin-left:8px;">Groq · LangChain · RAG</span></div>', unsafe_allow_html=True)
with t2:
    st.markdown(f'<div style="text-align:center;padding:6px 0;"><span style="font-size:0.75em;color:#555;">'
               f'You <b style="color:#aaa;">{user_msgs}</b> &middot; AI <b style="color:#aaa;">{ai_msgs}</b></span></div>', unsafe_allow_html=True)
with t3:
    tb1, tb2 = st.columns(2)
    with tb1:
        if st.button("Settings", use_container_width=True, key="settings_btn"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)
            st.rerun()
    with tb2:
        if st.button("New Chat", use_container_width=True, key="newchat_btn"):
            st.session_state.update({'messages': [], 'message_count': 0, 'last_question': None, 'chat_store': {}, 'input_key': st.session_state.input_key + 1})
            st.rerun()

st.markdown("<hr style='border-color:#2a2a2a;margin:6px 0 10px 0;'>", unsafe_allow_html=True)

if st.session_state.get("show_settings", False):
    st.markdown('<div class="settings-box">', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.caption("MODEL")
        model_name = st.selectbox("Model", ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama2-70b-4096"], label_visibility="collapsed", key="model_sel")
    with p2:
        st.caption("TEMPERATURE")
        temperature = st.slider("Temp", 0.0, 1.0, value=st.session_state.temperature, step=0.1, label_visibility="collapsed", key="temp_slider")
        st.session_state.temperature = temperature
    with p3:
        st.caption("MAX TOKENS")
        max_tokens = st.slider("Tokens", 100, 2500, value=st.session_state.max_tokens, step=100, label_visibility="collapsed", key="token_slider")
        st.session_state.max_tokens = max_tokens
        st.caption(f"{max_tokens} tokens · ~{max_tokens//4} words")
    if max_tokens != st.session_state.prev_max_tokens and st.session_state.last_question:
        st.session_state.prev_max_tokens = max_tokens
        st.session_state.trigger_regenerate = True
    if st.session_state.file_name:
        st.markdown("<hr style='border-color:#2a2a2a;margin:8px 0;'>", unsafe_allow_html=True)
        fa, fb = st.columns([5, 1])
        with fa:
            st.markdown(f'<span style="font-size:0.8em;color:#4ade80;font-weight:500;">RAG</span>'
                       f'<span style="font-size:0.8em;color:#666;margin-left:6px;">{st.session_state.file_name}</span>', unsafe_allow_html=True)
        with fb:
            if st.button("Remove", use_container_width=True, key="remove_file"):
                st.session_state.update({'vectorstore': None, 'rag_ready': False, 'file_name': None, 'file_type': None, 'chat_store': {}})
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    model_name = st.session_state.get("model_sel", "llama-3.1-8b-instant")
    temperature = st.session_state.temperature
    max_tokens = st.session_state.max_tokens

if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}<div class="timestamp">{message["timestamp"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="msg-label">Assistant</div>', unsafe_allow_html=True)
            st.markdown(message["content"])
            st.markdown(f'<div class="timestamp">{message["timestamp"]}</div>', unsafe_allow_html=True)
            if message.get("sources"):
                n = len(message["sources"])
                st.markdown(f'<div style="font-size:0.7em;color:#444;margin-top:2px;">Answered from {n} section{"s" if n > 1 else ""}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align:center;padding:50px 20px 30px;"><div style="font-size:2.2em;font-weight:600;color:#ececec;">What can I help with?</div>'
               '<div style="font-size:0.88em;color:#555;">Groq · LangChain · RAG</div></div>', unsafe_allow_html=True)

if st.session_state.rag_ready and not st.session_state.get("show_settings"):
    st.markdown(f'<div class="file-badge"><span style="color:#4ade80;font-weight:600;">RAG</span>'
               f'<span style="color:#888;margin-left:8px;">{st.session_state.file_name}</span></div>', unsafe_allow_html=True)

placeholder = f"Ask about {st.session_state.file_name}..." if st.session_state.file_name else "Type your message here..."
col1, col2, col3 = st.columns([5.5, 1.6, 1.2])
with col1:
    user_input = st.text_input("Message", placeholder=placeholder, label_visibility="collapsed", key=f"user_input_{st.session_state.input_key}")
with col2:
    if st.button("📎 Upload", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
with col3:
    send_button = st.button("Send", use_container_width=True)

if st.session_state.show_uploader:
    uploaded_file = st.file_uploader("Choose a file — PDF, TXT, or CSV", type=["pdf", "txt", "csv"], key=f"file_upload_{st.session_state.input_key}")
    if uploaded_file and uploaded_file.name != st.session_state.file_name:
        file_bytes = uploaded_file.read()
        ext = uploaded_file.name.split(".")[-1].lower()
        with st.spinner("Building RAG index..."):
            docs_map = {"pdf": extract_text_from_pdf, "txt": extract_text_from_txt, "csv": extract_text_from_csv}
            docs = docs_map[ext](file_bytes)
            st.session_state.vectorstore = build_vectorstore(docs)
            st.session_state.rag_ready = True
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_type = ext
            st.session_state.show_uploader = False
        st.rerun()

if user_input and send_button:
    st.session_state.last_question = user_input
    timestamp = datetime.now().strftime("%H:%M")
    display = user_input
    if st.session_state.file_name:
        display = f'<span style="font-size:0.78em;color:#666;">[{st.session_state.file_name}]</span><br>{user_input}'
    st.session_state.messages.append({"role": "user", "content": display, "timestamp": timestamp})
    st.session_state.message_count += 1
    with st.spinner("Thinking..."):
        try:
            answer, sources = generate_response(user_input, model_name, temperature, max_tokens)
            st.session_state.messages.append({"role": "assistant", "content": answer, "timestamp": datetime.now().strftime("%H:%M"), "sources": sources})
            st.session_state.message_count += 1
            st.session_state.input_key += 1
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

if st.session_state.get("trigger_regenerate") and st.session_state.last_question:
    st.session_state.trigger_regenerate = False
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages.pop()
        st.session_state.message_count -= 1
    with st.spinner("Regenerating..."):
        try:
            answer, sources = generate_response(st.session_state.last_question, model_name, st.session_state.temperature, st.session_state.max_tokens)
            st.session_state.messages.append({"role": "assistant", "content": answer, "timestamp": datetime.now().strftime("%H:%M"), "sources": sources})
            st.session_state.message_count += 1
            st.rerun()
        except Exception as e:
            st.error(f"Regeneration failed: {str(e)}")

st.markdown('<div style="text-align:center;color:#444;font-size:0.72em;padding:16px;margin-top:20px;">AI can make mistakes · Streamlit · Groq · LangChain RAG</div>', unsafe_allow_html=True)
