# 🤖 AI Chat Assistant

A powerful conversational AI chatbot with **Retrieval-Augmented Generation (RAG)** capabilities, built with Streamlit, LangChain, and Groq.

## 🚀 Live Demo

**[Try it now →](https://assistent-ai-ujjwal.streamlit.app)**

## ✨ Features

- 📄 **Document Q&A** - Upload PDF, TXT, or CSV files and ask questions
- 🧠 **Conversational Memory** - AI remembers the full conversation context
- 🎨 **Clean Dark UI** - ChatGPT-inspired modern interface
- ⚙️ **Customizable Settings** - Adjust model, temperature, and token limits
- 🔄 **Auto-Regenerate** - Responses update when you change settings
- 💬 **Source Attribution** - See which document sections were used

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq (llama-3.1-8b-instant)
- **Framework:** LangChain
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)

## 📦 Installation

### Prerequisites

- Python 3.8+
- Groq API Key ([Get it here](https://console.groq.com/keys))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-chat-assistant.git
   cd ai-chat-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file or add to `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage

### Chat Mode
Simply type your message and get AI-powered responses.

### Document Mode (RAG)
1. Click **📎 Upload**
2. Select a PDF, TXT, or CSV file
3. Ask questions about the document
4. AI answers using only the document content

### Settings
Click **Settings** to customize:
- **Model:** Choose between llama-3.1, mixtral, or llama2
- **Temperature:** Control response creativity (0.0 - 1.0)
- **Max Tokens:** Set response length (100 - 2500)

## 📁 Project Structure

```
ai-chat-assistant/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .env               # Environment variables (not tracked)
```

## 🔧 Configuration

### Supported Models
- `llama-3.1-8b-instant` (Default - Fast & Efficient)

### File Support
- **PDF** - Extracts text from all pages
- **TXT** - Plain text files
- **CSV** - Tabular data with automatic chunking

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in app settings:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Deploy!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Developer

**Developed by:** [Ujjwal Kumar Ray](https://github.com/ujjwal-ray)

---
