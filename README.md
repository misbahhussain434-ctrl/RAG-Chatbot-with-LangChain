# QuickRAG

A lightweight **Retrieval-Augmented Generation (RAG)** service. Upload your documents (PDF / text), and ask questions — the system retrieves the most relevant chunks from your data and generates grounded, context-aware answers using a Groq LLM.

---

## ✨ Features

- 📤 **Document Upload** — ingest PDF, `.txt`, `.md`, and `.log` files
- ✂️ **Smart Chunking** — recursive text splitting with overlap for better context
- 🔎 **Semantic Search** — vector similarity search powered by ChromaDB
- 🧠 **Query Refinement** — the raw query is refined before retrieval for better results
- 💬 **Grounded Answers** — responses are based strictly on your uploaded content
- ⚡ **Fast LLM** — uses Groq (default model: `llama-3.3-70b-versatile`)
- 🌐 **Interactive Docs** — built-in Swagger UI at `/docs`

---

## 🧱 Tech Stack

| Layer        | Technology                                   |
|--------------|----------------------------------------------|
| API          | FastAPI + Uvicorn                            |
| LLM          | Groq (`langchain-groq`)                      |
| Vector Store | ChromaDB (`langchain-chroma`)               |
| Embeddings   | Sentence Transformers (`all-MiniLM-L6-v2`)  |
| Parsing      | LangChain document loaders (PyPDF, Text)    |

---

## 🚀 Quick Start

### Option 1 — One Click (Windows)
Just **double-click `start.bat`**. It will:
1. Create a virtual environment
2. Install all dependencies
3. Start the server

### Option 2 — Manual (PowerShell)

```powershell
# 1. Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
---

## 🔑 Configuration

Create a `.env` file in the project root (a template `.env` is included):

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
CHROMA_DIR=./chroma_db
CHROMA_COLLECTION=rag_collection
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```
Get a free Groq API key from: https://console.groq.com/keys

---

## 📡 API Endpoints

| Method | Endpoint   | Description                              |
|--------|-----------|------------------------------------------|
| `GET`  | `/health` | Health check                             |
| `POST` | `/upload` | Upload & ingest a document into the DB   |
| `POST` | `/query`  | Ask a question and get a grounded answer |

