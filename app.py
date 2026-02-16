import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -------- LangSmith (optional) --------
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot (Cloud)"
# -------------------------------------

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries. Do NOT guess or invent facts. "),
    ("user", "Question: {question}")
])

def generate_response(question, model_name, temperature, max_tokens):
    # Groq LLM (Cloud friendly)
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=st.secrets["GROQ_API_KEY"]  # Streamlit secrets
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"question": question})

# Title
st.title("Enhanced Q&A Chatbot (Cloud Ready)")

# Sidebar model selection (Groq models)
model_name = st.sidebar.selectbox(
    "Select Model",
    ["llama-3.1-8b-instant"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 500)

st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    try:
        response = generate_response(user_input, model_name, temperature, max_tokens)
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.write("Please provide the user input")




