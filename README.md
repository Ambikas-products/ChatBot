# 💬 MediBot - AI Q&A Chatbot with LangChain + Streamlit

MediBot is a Streamlit-based chatbot that answers user queries based on a custom knowledge base powered by FAISS vector search and the HuggingFace Mistral LLM.

---

## 🚀 Features

- 💡 Contextual Q&A using LangChain's RetrievalQA
- 🔍 FAISS vector search over custom documents
- 🤖 Powered by HuggingFace `mistralai/Mistral-7B-Instruct-v0.3`
- ⚡ Streamlit UI for real-time interaction
- ✅ HuggingFace token-based API access
- ❌ No hallucination: Bot responds only from the provided context

---

## 🛠️ Installation

1. **Clone this repo or navigate to your working directory:**
   ```bash
   cd pythonlab
Install required Python packages (inside Jupyter or terminal):

python
Copy
Edit
!pip install -q streamlit langchain langchain-community huggingface_hub python-dotenv faiss-cpu
Create .env file with your HuggingFace token (already handled in script):

ini
Copy
Edit
HF_TOKEN=your_huggingface_api_key
🧠 Setup Vector Store
Make sure your vectorstore/db_faiss folder contains precomputed FAISS index based on your documents. If not, you'll need to generate it using LangChain's vector store creation script.

▶️ Run the App
In your command line:

bash
Copy
Edit
streamlit run medibot.py
It will open a browser window where you can chat with MediBot.

🔐 Environment Variables
Variable	Purpose
HF_TOKEN	HuggingFace API token (required)

📦 Folder Structure
bash
Copy
Edit
.
├── medibot.py                # Main app
├── vectorstore/
│   └── db_faiss              # FAISS index directory
├── .env                      # HuggingFace token (auto-generated)
└── README.md                 # You are here
🧩 Dependencies
Python 3.8+

Streamlit

LangChain

FAISS

HuggingFace Hub

✅ Notes
Make sure your HuggingFace token has access to inference endpoints.

The app won't generate responses outside of the vector database context.

To add new docs, re-embed and save to db_faiss.

🙌 Credits
Built using LangChain

Hosted locally with Streamlit

Model: Mistral 7B Instruct
