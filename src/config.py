"""
Centralized configuration for the Enterprise AI Assistant.
All runtime settings are defined here. Override via environment variables.
"""
import os

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_MODEL     = os.getenv("GROQ_MODEL",      "llama-3.1-8b-instant")
GROQ_TEMP      = float(os.getenv("GROQ_TEMP", "0"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBED_MODEL    = os.getenv("EMBED_MODEL",     "all-MiniLM-L6-v2")

# ── Retrieval ─────────────────────────────────────────────────────────────────
BM25_K         = int(os.getenv("BM25_K",      "10"))
CHROMA_K       = int(os.getenv("CHROMA_K",    "10"))
RERANK_TOP_N   = int(os.getenv("RERANK_TOP_N","5"))
RERANK_MODEL   = os.getenv("RERANK_MODEL",    "rerank-v3.5")
ENSEMBLE_DENSE = float(os.getenv("ENSEMBLE_DENSE", "0.6"))  # weight for vector search
MULTI_QUERY_N  = int(os.getenv("MULTI_QUERY_N", "3"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE",  "800"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP","150"))

# ── Storage paths ─────────────────────────────────────────────────────────────
DB_DIR              = os.getenv("DB_DIR",            "./db")
CHUNKS_PKL          = os.getenv("CHUNKS_PKL",        "./chunks.pkl")
BM25_PKL            = os.getenv("BM25_PKL",          "./bm25_index.pkl")
HASH_PKL            = os.getenv("HASH_PKL",          "./ingested_hashes.pkl")
DATA_DIR            = os.getenv("DATA_DIR",          "./data")
CONVERSATIONS_DIR   = os.getenv("CONVERSATIONS_DIR", "./conversations")
FEEDBACK_CSV        = os.getenv("FEEDBACK_CSV",      "./feedback.csv")
AUDIT_CSV           = os.getenv("AUDIT_CSV",         "./audit.csv")

# ── Auth ──────────────────────────────────────────────────────────────────────
# Format:  USER1=password1,USER2=password2  (set in .env)
USERS_ENV      = os.getenv("APP_USERS", "admin=admin123,user=user123")

def get_users() -> dict:
    """Return {username: password} from APP_USERS env var."""
    users = {}
    for pair in USERS_ENV.split(","):
        if "=" in pair:
            u, p = pair.strip().split("=", 1)
            users[u.strip()] = p.strip()
    return users
