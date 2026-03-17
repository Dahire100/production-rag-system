import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import pickle
import warnings
from typing import List, Dict, Tuple, Generator
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
from src.config import (
    GROQ_MODEL, GROQ_TEMP, EMBED_MODEL,
    BM25_K, CHROMA_K, RERANK_TOP_N, RERANK_MODEL,
    ENSEMBLE_DENSE, MULTI_QUERY_N,
    DB_DIR, CHUNKS_PKL, BM25_PKL,
)

warnings.filterwarnings("ignore")
load_dotenv()

_embeddings = None
_llm        = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model_name=GROQ_MODEL, temperature=GROQ_TEMP)
    return _llm


def reset_chain():
    global _embeddings, _llm
    _embeddings = None
    _llm        = None


# ── Retriever ─────────────────────────────────────────────────────────────────

def get_base_retriever():
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=_get_embeddings())
    chroma_ret  = vectorstore.as_retriever(search_kwargs={"k": CHROMA_K})

    with open(CHUNKS_PKL, "rb") as f:
        chunks = pickle.load(f)
    bm25_ret = BM25Retriever.from_documents(chunks)
    bm25_ret.k = BM25_K

    ensemble = EnsembleRetriever(
        retrievers=[bm25_ret, chroma_ret],
        weights=[1 - ENSEMBLE_DENSE, ENSEMBLE_DENSE],
    )
    reranker = CohereRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)
    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=ensemble
    )


def _expand_queries(question: str) -> List[str]:
    """Generate alternative phrasings to improve recall."""
    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_template(
            "Generate {n} alternative phrasings of this question to improve document retrieval.\n"
            "Output ONLY the questions, one per line.\n\nQuestion: {question}\n\nPhrasings:"
        )
        result = (prompt | llm | StrOutputParser()).invoke(
            {"question": question, "n": MULTI_QUERY_N}
        )
        alts = [q.strip() for q in result.strip().split("\n") if q.strip()]
        return [question] + alts[:MULTI_QUERY_N]
    except Exception:
        return [question]


def multi_query_retrieve(question: str) -> List[Document]:
    """Retrieve using multiple query expansions, deduplicated."""
    retriever = get_base_retriever()
    queries   = _expand_queries(question)
    seen, docs = set(), []
    for q in queries:
        try:
            for doc in retriever.invoke(q):
                did = doc.metadata.get("doc_id", doc.page_content[:40])
                if did not in seen:
                    seen.add(did)
                    docs.append(doc)
        except Exception:
            pass
    return docs


# ── Formatting ────────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        did    = doc.metadata.get("doc_id",  "?")
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page   = doc.metadata.get("page",   "?")
        parts.append(f"[{did}] (file: {source}, page: {page}):\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_history_str(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for m in history[-4:]:
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
        elif m["role"] == "assistant" and not m.get("contexts"):
            lines.append(f"Assistant: {m['content']}")
    return "\n".join(lines) + "\n\n" if len(lines) > 1 else ""


# ── Prompts ───────────────────────────────────────────────────────────────────

RAG_TEMPLATE = """{history}You are a precise enterprise document assistant. Answer only from the provided context.

{extra}Instructions:
- Use **markdown formatting** (bold, bullet lists, headers where appropriate).
- Cite EVERY claim with the source [doc_id], e.g. [doc_0]. Multiple citations: [doc_0][doc_2].
- If context lacks information, say so clearly — never fabricate.
- Be thorough but concise.

Context:
{context}

Question: {question}

Answer:"""

FOLLOWUP_TEMPLATE = """Suggest 3 concise follow-up questions based on this Q&A.
Output one question per line, no numbers or prefixes.

Question: {question}
Answer: {answer}

Follow-ups:"""

SUMMARY_TEMPLATE = """You are a document analyst. Write a structured summary of this document.

Use this format:
## Overview
(1-2 sentences on the document's purpose)

## Key Topics
- (bullet point list of main subjects)

## Important Details
(key facts, figures, or conclusions)

Document: {filename}
Content:
{context}

Summary:"""


# ── Public generation API ─────────────────────────────────────────────────────

# Store last docs used (allows app.py to retrieve them post-stream)
_last_docs: List[Document] = []


def stream_answer(
    question: str,
    history: List[Dict] = None,
    extra_instructions: str = "",
) -> Generator:
    global _last_docs

    if not os.path.exists(CHUNKS_PKL):
        yield "⚠️ No documents indexed yet. Please upload a file in the sidebar."
        return

    docs = multi_query_retrieve(question)
    _last_docs = docs

    if not docs:
        yield "I couldn't find relevant content for that question in the indexed documents."
        return

    extra_block = (
        f"Additional user instructions: {extra_instructions}\n\n"
        if extra_instructions.strip() else ""
    )
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain  = prompt | _get_llm() | StrOutputParser()

    yield from chain.stream({
        "context":  format_docs(docs),
        "question": question,
        "history":  _build_history_str(history or []),
        "extra":    extra_block,
    })


def get_last_docs() -> List[Document]:
    return _last_docs


def get_followup_suggestions(question: str, answer: str) -> List[str]:
    try:
        prompt = ChatPromptTemplate.from_template(FOLLOWUP_TEMPLATE)
        result = (prompt | _get_llm() | StrOutputParser()).invoke(
            {"question": question, "answer": answer}
        )
        return [q.strip() for q in result.strip().split("\n") if q.strip()][:3]
    except Exception:
        return []


def summarize_document(filename: str, chunks: List) -> str:
    step    = max(1, len(chunks) // 30)
    sampled = chunks[::step][:30]
    context = "\n\n---\n\n".join(
        f"[Page {c.metadata.get('page','?')}]: {c.page_content}" for c in sampled
    )
    prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
    return (prompt | _get_llm() | StrOutputParser()).invoke(
        {"filename": filename, "context": context}
    )
