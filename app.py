import os
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

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Chat message styling */
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
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1em;
        text-align: center;
        margin-top: 10px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e1e8ed;
        padding: 12px 20px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Chat container */
    .chat-container {
        background-color: #1e1e1e;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Timestamp styling */
    .timestamp {
        font-size: 0.75em;
        color: #a0a0a0;
        margin-top: 5px;
    }
    
    /* Clear chat button */
    .clear-button {
        background-color: #e74c3c;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 20px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .clear-button:hover {
        background-color: #c0392b;
    }
    
    /* Info box */
    .info-box {
        background-color: #2a3f5f;
        border-left: 4px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
        color: #e8e8e8;
    }
    
    /* Metrics styling */
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'message_count' not in st.session_state:
    st.session_state.message_count = 0

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries. Do NOT guess or invent facts."),
    ("user", "Question: {question}")
])

def generate_response(question, model_name, temperature, max_tokens):
    """Generate response from the LLM"""
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"question": question})

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Chat Assistant</h1>
        <p class="header-subtitle">Powered by Advanced Language Models</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model selection
    model_name = st.selectbox(
        "🔧 Select Model",
        ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama2-70b-4096"],
        help="Choose the AI model for generating responses"
    )
    
    st.markdown("---")
    
    # Temperature slider
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness: Lower = more focused, Higher = more creative"
    )
    
    # Max tokens slider
    max_tokens = st.slider(
        "📏 Max Tokens",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum length of the response"
    )
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9em; color: #7f8c8d;">Messages</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #667eea;">{st.session_state.message_count}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9em; color: #7f8c8d;">Model</div>
                <div style="font-size: 1em; font-weight: bold; color: #764ba2;">{model_name.split('-')[0].upper()}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.rerun()
    
    st.markdown("---")
    
    # Info section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **AI Chat Assistant** uses state-of-the-art language models to provide 
        intelligent responses to your questions.
        
        **Features:**
        - 💬 Real-time chat interface
        - 🎨 Professional UI design
        - 📝 Chat history tracking
        - ⚙️ Customizable parameters
        """)

# Main chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-message">
                    <strong>You</strong><br>
                    {message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant</strong><br>
                    {message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="info-box">
            <strong>👋 Welcome!</strong> Ask me anything and I'll do my best to help you.
            <br><br>
            💡 <strong>Tips:</strong>
            <ul>
                <li>Be specific with your questions</li>
                <li>Adjust temperature for more creative or focused responses</li>
                <li>Use the sidebar to customize model parameters</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input area
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Type your message here...",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send_button = st.button("Send 📤", use_container_width=True)

# Handle user input
if (user_input and send_button) or (user_input and user_input != st.session_state.get('last_input', '')):
    st.session_state.last_input = user_input
    
    # Add user message to history
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    st.session_state.message_count += 1
    
    # Generate and add assistant response
    with st.spinner("🤔 Thinking..."):
        try:
            response = generate_response(user_input, model_name, temperature, max_tokens)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.message_count += 1
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.markdown("""
                <div class="info-box" style="border-left-color: #e74c3c; background-color: #3d2a2a;">
                    <strong>⚠️ Troubleshooting:</strong>
                    <ul>
                        <li>Check your API key configuration</li>
                        <li>Verify your internet connection</li>
                        <li>Try a different model</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #95a5a6; font-size: 0.9em; padding: 20px;">
        Built with ❤️ using Streamlit and LangChain | Powered by Groq
    </div>
""", unsafe_allow_html=True)
