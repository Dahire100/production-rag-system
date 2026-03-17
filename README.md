# 💼 Enterprise AI Assistant — RAG System

> Production-grade Retrieval-Augmented Generation (RAG) application with Hybrid Search, Multi-Query Expansion, Cohere Reranking, and a full enterprise feature set.

---

## 📁 Project Structure

```
RAG/
│
├── app.py                      # 🚀 Main Streamlit application (entry point)
├── pyproject.toml              # 📦 Project metadata & dependencies
├── .env                        # 🔑 API keys & credentials (never commit)
├── .gitignore                  # 🚫 Git ignore rules
│
├── src/                        # 🧠 Core application modules
│   ├── __init__.py
│   ├── config.py               # ⚙️  Centralized configuration (all settings here)
│   ├── auth.py                 # 🔐 Authentication (login, session, SHA-256 hashing)
│   ├── ingest.py               # 📥 Document ingestion (PDF, DOCX, CSV, TXT, MD)
│   ├── generate.py             # 🤖 LLM generation, retrieval, streaming, summarization
│   ├── retrieve.py             # 🔍 Retrieval pipeline (BM25 + Chroma + Cohere rerank)
│   ├── conversations.py        # 💬 Persistent conversation storage (per-user JSON)
│   └── audit.py                # 📋 Audit & feedback logging (CSV)
│
├── data/                       # 📂 Uploaded source documents
│   └── (your PDFs, DOCX, CSVs, TXT files go here)
│
├── db/                         # 🗄️  ChromaDB vector index (auto-generated)
│   └── (auto-generated, do not edit)
│
├── conversations/              # 💾 Saved chat history (per user)
│   └── admin/
│       └── <conv_id>.json
│
├── eval/                       # 🧪 Evaluation pipeline
│   ├── test_rag.py             # Ragas evaluation tests
│   └── eval_data.json          # Golden Q&A dataset
│
├── .github/
│   └── workflows/
│       └── eval.yml            # ♻️  CI/CD evaluation pipeline (GitHub Actions)
│
├── audit.csv                   # 📊 Query audit log (auto-generated)
├── feedback.csv                # 👍 User feedback log (auto-generated)
├── chunks.pkl                  # 🗃️  Serialized document chunks (auto-generated)
├── bm25_index.pkl              # 🗃️  BM25 sparse index (auto-generated)
└── ingested_hashes.pkl         # 🔑 Content hashes for dedup (auto-generated)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -e .
```

### 2. Configure environment
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
APP_USERS=admin=admin123,user=user123
```

### 3. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔑 Default Login Credentials

| Role  | Username | Password   |
|-------|----------|------------|
| 👑 Admin | `admin`  | `admin123` |
| 👤 User  | `user`   | `user123`  |

> Add more users by extending `APP_USERS` in `.env`:
> `APP_USERS=admin=admin123,user=user123,alice=mypass`

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Multi-Query Expansion (LLM generates 3 variations)
    │
    ├──► BM25 Sparse Retrieval  (keyword matching)
    │
    └──► ChromaDB Dense Retrieval (semantic similarity)
              │
              ▼
         Ensemble Fusion (60% dense / 40% sparse)
              │
              ▼
         Cohere Rerank (rerank-v3.5) → Top 5 chunks
              │
              ▼
         LLM (Groq · llama-3.1-8b-instant)
              │
              ▼
         Streamed Citation-backed Answer
```

---

## 📦 Supported File Formats

| Format | Extension | Notes                        |
|--------|-----------|------------------------------|
| PDF    | `.pdf`    | Text + image-heavy PDFs (PyMuPDF) |
| Word   | `.docx` `.doc` | Microsoft Word documents |
| CSV    | `.csv`    | Each row becomes a chunk     |
| Text   | `.txt`    | Plain text files             |
| Markdown | `.md`  | Markdown documents           |

---

## ✨ Features

| Feature                | Description                                      |
|------------------------|--------------------------------------------------|
| 🔐 Authentication       | Login with SHA-256 hashed credentials            |
| 💬 Persistent Chats     | Conversations saved per user across restarts     |
| 🌊 Streaming Responses  | Tokens stream live to the screen                 |
| 🔍 Multi-Query Retrieval| LLM rewrites query 3×, retrieves + deduplicates  |
| 📚 Citation Enforcement | Every claim cited to source doc + page number    |
| 🧠 Document Summarizer  | One-click AI summary per indexed document        |
| 🔎 Segment Explorer     | Browse raw chunks, filter by page & keyword      |
| 💡 Follow-up Suggestions| 3 AI-generated follow-up chips after each answer |
| 👍👎 Feedback System    | Rate answers; logged to `feedback.csv`           |
| 📊 Admin Panel          | Audit log, feedback analytics, KPI metrics       |
| 🧑‍💼 Custom Persona      | Add custom instructions to shape AI behavior     |
| ⬇️ Export Chat          | Download conversation as Markdown                |
| ⚙️ Config Module        | All settings in `src/config.py`, env-overridable |

---

## ⚙️ Configuration Reference

All settings live in `src/config.py` and can be overridden via environment variables:

| Variable          | Default                  | Description                    |
|-------------------|--------------------------|--------------------------------|
| `GROQ_MODEL`      | `llama-3.1-8b-instant`   | LLM model name                 |
| `EMBED_MODEL`     | `all-MiniLM-L6-v2`       | HuggingFace embedding model    |
| `RERANK_MODEL`    | `rerank-v3.5`            | Cohere reranker model          |
| `CHUNK_SIZE`      | `800`                    | Tokens per chunk               |
| `CHUNK_OVERLAP`   | `150`                    | Overlap between chunks         |
| `RERANK_TOP_N`    | `5`                      | Chunks kept after reranking    |
| `MULTI_QUERY_N`   | `3`                      | Alternative query expansions   |
| `CHROMA_K`        | `10`                     | Initial vector search results  |
| `BM25_K`          | `10`                     | Initial BM25 results           |
| `APP_USERS`       | *(see .env)*             | `user=pass,user2=pass2` format |

---

## 🧪 Evaluation

Run the Ragas evaluation pipeline:
```bash
pytest eval/test_rag.py -v
```

CI runs automatically on every push via `.github/workflows/eval.yml`.

---

## 🛡️ Production Checklist

- [ ] Change default passwords in `.env`
- [ ] Add `.env` to `.gitignore` ✅ (already done)
- [ ] Set `GROQ_API_KEY` and `COHERE_API_KEY`
- [ ] Mount `data/`, `conversations/`, `db/` as persistent volumes in Docker
- [ ] Set up HTTPS via a reverse proxy (nginx/caddy) for production
