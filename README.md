# RAG PDF Search API 🔍📄

This project is a **FastAPI-based API** for performing retrieval-augmented generation (RAG) over uploaded PDFs. It allows you to search and query documents intelligently using LLMs and vector search.

## 🚀 Features

- 🔐 Secure endpoints using Bearer token
- 📄 Load and split PDFs using `PyMuPDF`
- 🔎 Embedding + Vector store powered by `Pinecone`
- 🧠 Query processing using LLMs
- 🌐 Deployable to Render.com with `render.yaml`

## 📦 Requirements

- Python 3.10+
- A `.env` file with secrets (example below)
- Groq API key
- OpenAI API key

## 🛠️ Setup

```bash
# Clone the repo
git clone https://github.com/Akshat030307/RAG.git
cd RAG

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload
