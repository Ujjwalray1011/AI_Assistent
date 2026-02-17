import os
import io
import base64
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

load_dotenv()

# -------- LangSmith (optional) --------
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot (Cloud)"

# Page configuration
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
        background-color: #007bff;
        color: white;
        padding: 15px 20px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: #2d2d2d;
        color: #e8e8e8;
        padding: 15px 20px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 80%;
        border: 1px solid #404040;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title {
        color: white; font-size: 2.5em; font-weight: 700;
        margin: 0; text-align: center;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9); font-size: 1.1em;
        text-align: center; margin-top: 10px;
    }
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
        background-color: #1e3a5f; border: 1px solid #3498db;
        border-radius: 10px; padding: 10px 14px; margin: 8px 0;
        color: #7ec8e3; font-size: 0.9em;
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
    'file_context': None,       # extracted text from uploaded file
    'file_name': None,          # name of uploaded file
    'file_type': None,          # type: pdf / txt / csv / image
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── FILE EXTRACTION HELPERS ────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"[PDF read error: {e}]"

def extract_text_from_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore").strip()

def extract_text_from_csv(file_bytes):
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        summary = f"CSV has {len(df)} rows and {len(df.columns)} columns.\n"
        summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        summary += df.head(50).to_string(index=False)
        return summary
    except Exception as e:
        return f"[CSV read error: {e}]"

def image_to_base64(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")

# ─── PROMPT TEMPLATES ───────────────────────────────────────────────────────
plain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Do NOT guess or invent facts."),
    ("user", "Question: {question}")
])

file_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. The user has uploaded a file.
Use ONLY the content below to answer questions. Do not guess or invent facts.

--- FILE CONTENT START ---
{file_context}
--- FILE CONTENT END ---
"""),
    ("user", "Question: {question}")
])

# ─── RESPONSE GENERATOR ─────────────────────────────────────────────────────
def generate_response(question, model_name, temperature, max_tokens, file_context=None):
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    )
    parser = StrOutputParser()
    if file_context:
        chain = file_prompt | llm | parser
        return chain.invoke({"question": question, "file_context": file_context[:12000]})
    else:
        chain = plain_prompt | llm | parser
        return chain.invoke({"question": question})

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Chat Assistant</h1>
        <p class="header-subtitle">Powered by Advanced Language Models</p>
    </div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    model_name = st.selectbox(
        "🔧 Select Model",
        ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama2-70b-4096"],
        help="Choose the AI model for generating responses"
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

    st.markdown("---")

    # ── SESSION STATS ────────────────────────────────────────────────────────
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.9em;color:#7f8c8d;">Messages</div>
                <div style="font-size:1.8em;font-weight:bold;color:#667eea;">{st.session_state.message_count}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
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
        **AI Chat Assistant** — powered by Groq + LangChain.

        **Features:**
        - 💬 Chat with AI
        - 📎 Upload & ask about files
        - 🎨 Professional dark UI
        - 📝 Chat history with timestamps
        - ⚙️ Customizable model settings
        """)

# ─── MAIN CHAT AREA ──────────────────────────────────────────────────────────

# Show image preview if an image is uploaded
if st.session_state.file_type == "image" and st.session_state.file_context:
    b64 = st.session_state.file_context.replace("[IMAGE:", "").replace("]", "")
    st.markdown(f"""
        <div style="text-align:center; margin-bottom:10px;">
            <img src="data:image/png;base64,{b64}"
                 style="max-height:220px; border-radius:12px; border:2px solid #404040;"/>
            <div style="color:#aaa; font-size:0.8em; margin-top:5px;">
                🖼️ {st.session_state.file_name} — Ask me anything about this image
            </div>
        </div>
    """, unsafe_allow_html=True)

# Display messages
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
else:
    if st.session_state.file_context and st.session_state.file_type != "image":
        st.markdown(f"""
            <div class="info-box">
                <strong>📎 File ready!</strong> You can now ask questions about
                <strong>{st.session_state.file_name}</strong>.<br><br>
                💡 Try: <em>"Summarize this file"</em> or <em>"What are the key points?"</em>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="info-box">
                <strong>👋 Welcome!</strong> Ask me anything or upload a file to get started.
                <br><br>
                💡 <strong>Tips:</strong>
                <ul>
                    <li>Upload a PDF, TXT, CSV or Image from the sidebar</li>
                    <li>Then ask questions about its content</li>
                    <li>Adjust temperature &amp; tokens for better results</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ─── INPUT BAR ───────────────────────────────────────────────────────────────
placeholder = (
    f"Ask about {st.session_state.file_name}..."
    if st.session_state.file_name
    else "Type your message here..."
)

# Show active file badge above input bar if a file is loaded
if st.session_state.file_name:
    icons = {"pdf": "📄", "txt": "📝", "csv": "📊", "image": "🖼️"}
    icon = icons.get(st.session_state.file_type, "📎")
    fcol1, fcol2 = st.columns([9, 1])
    with fcol1:
        st.markdown(f"""
            <div class="file-badge">
                {icon} <strong>{st.session_state.file_name}</strong>
                &nbsp;<span style="font-size:0.8em;color:#aaa;">active — questions will use this file</span>
            </div>
        """, unsafe_allow_html=True)
    with fcol2:
        if st.button("✖", help="Remove file", use_container_width=True):
            st.session_state.file_context = None
            st.session_state.file_name = None
            st.session_state.file_type = None
            st.rerun()

col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    user_input = st.text_input(
        "Message", placeholder=placeholder,
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_key}"
    )
with col2:
    uploaded_file = st.file_uploader(
        "📎", type=["pdf", "txt", "csv", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key=f"file_upload_{st.session_state.input_key}"
    )
    if uploaded_file:
        file_bytes = uploaded_file.read()
        ext = uploaded_file.name.split(".")[-1].lower()
        if uploaded_file.name != st.session_state.file_name:
            with st.spinner("📖 Reading..."):
                if ext == "pdf":
                    ctx = extract_text_from_pdf(file_bytes)
                    ftype = "pdf"
                elif ext == "txt":
                    ctx = extract_text_from_txt(file_bytes)
                    ftype = "txt"
                elif ext == "csv":
                    ctx = extract_text_from_csv(file_bytes)
                    ftype = "csv"
                else:
                    ctx = f"[IMAGE:{image_to_base64(file_bytes)}]"
                    ftype = "image"
                st.session_state.file_context = ctx
                st.session_state.file_name = uploaded_file.name
                st.session_state.file_type = ftype
            st.rerun()
    else:
        if st.session_state.file_context and not st.session_state.file_name:
            st.session_state.file_context = None
            st.session_state.file_type = None
with col3:
    send_button = st.button("Send 📤", use_container_width=True)

# ─── HANDLE SEND ─────────────────────────────────────────────────────────────
if (user_input and send_button) or (user_input and user_input != st.session_state.get('last_input', '')):
    st.session_state.last_input = user_input
    st.session_state.last_question = user_input

    timestamp = datetime.now().strftime("%H:%M")

    # Show file badge in chat if file is active
    display_content = user_input
    if st.session_state.file_name:
        display_content = f"📎 <em style='font-size:0.8em;opacity:0.7;'>[{st.session_state.file_name}]</em><br>{user_input}"

    st.session_state.messages.append({
        "role": "user",
        "content": display_content,
        "timestamp": timestamp
    })
    st.session_state.message_count += 1

    with st.spinner("🤔 Thinking..."):
        try:
            # Pass image description context for images
            ctx = st.session_state.file_context
            if ctx and ctx.startswith("[IMAGE:"):
                # For images, describe that an image was shared
                ctx = "The user has shared an image. Describe what you can infer from context and answer their question as best you can based on the file name and any context provided."

            response = generate_response(
                user_input, model_name, temperature, max_tokens,
                file_context=ctx
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M")
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
    with st.spinner("🔄 Regenerating with new token limit..."):
        try:
            ctx = st.session_state.file_context
            if ctx and ctx.startswith("[IMAGE:"):
                ctx = "The user has shared an image."
            response = generate_response(
                st.session_state.last_question, model_name,
                st.session_state.temperature, st.session_state.max_tokens,
                file_context=ctx
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.message_count += 1
            st.rerun()
        except Exception as e:
            st.error(f"❌ Regeneration failed: {str(e)}")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <div style="text-align:center; color:#95a5a6; font-size:0.9em; padding:20px;">
        Built with ❤️ using Streamlit and LangChain | Powered by Groq
    </div>
""", unsafe_allow_html=True)
