"""
Generation module — instrumented with observability tracing.
Every public function now records latency spans, estimated token usage,
and cost per request into the MetricsStore (SQLite).
"""

from __future__ import annotations

import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import pickle
import warnings
from typing import List, Dict, Generator, Optional

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
    ENSEMBLE_DENSE, MULTI_QUERY_N, DB_DIR, CHUNKS_PKL,
)
from src.observability import (
    trace_context, record_latency, record_cost, record_quality,
    get_store, current_trace_id,
)

warnings.filterwarnings("ignore")
load_dotenv()

_embeddings  = None
_llm         = None
_last_docs: List[Document] = []

# ── Rough token estimator (≈ 4 chars per token, no tiktoken needed) ─────────

def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── Lazy singletons ──────────────────────────────────────────────────────────

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model_name=GROQ_MODEL, temperature=GROQ_TEMP)
    return _llm


def reset_chain():
    global _embeddings, _llm, _last_docs
    _embeddings = None
    _llm        = None
    _last_docs  = []


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _get_base_retriever():
    vs     = Chroma(persist_directory=DB_DIR, embedding_function=_get_embeddings())
    ch_ret = vs.as_retriever(search_kwargs={"k": CHROMA_K})
    with open(CHUNKS_PKL, "rb") as f:
        chunks = pickle.load(f)
    bm25      = BM25Retriever.from_documents(chunks)
    bm25.k    = BM25_K
    ens        = EnsembleRetriever(retrievers=[bm25, ch_ret],
                                   weights=[1 - ENSEMBLE_DENSE, ENSEMBLE_DENSE])
    rerank     = CohereRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)
    return ContextualCompressionRetriever(base_compressor=rerank, base_retriever=ens)


def _expand_queries(question: str) -> List[str]:
    try:
        prompt = ChatPromptTemplate.from_template(
            "Generate {n} alternative phrasings of this question for better document retrieval.\n"
            "Output ONLY the questions, one per line.\n\nQuestion: {question}\n\nPhrasings:"
        )
        result = (prompt | _get_llm() | StrOutputParser()).invoke(
            {"question": question, "n": MULTI_QUERY_N}
        )
        alts = [q.strip() for q in result.strip().split("\n") if q.strip()]
        return [question] + alts[:MULTI_QUERY_N]
    except Exception:
        return [question]


def multi_query_retrieve(question: str, trace_id: Optional[str] = None) -> List[Document]:
    t0 = time.perf_counter()
    with trace_context("multi_query_retrieve", {"question": question[:100]}):
        ret    = _get_base_retriever()
        seen, docs = set(), []
        for q in _expand_queries(question):
            try:
                for d in ret.invoke(q):
                    did = d.metadata.get("doc_id", d.page_content[:40])
                    if did not in seen:
                        seen.add(did)
                        docs.append(d)
            except Exception:
                pass
    latency = time.perf_counter() - t0
    record_latency("retrieval", latency)
    return docs


# ── Formatting ────────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        did    = d.metadata.get("doc_id", "?")
        source = os.path.basename(d.metadata.get("source", "unknown"))
        page   = d.metadata.get("page", "?")
        parts.append(f"[{did}] (file: {source}, page: {page}):\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _history_str(history: List[Dict]) -> str:
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

RAG_TEMPLATE = """{history}You are a precise enterprise document assistant.

{extra}Instructions:
- Use **markdown** (bold key terms, bullet lists, code blocks where relevant).
- Cite EVERY claim with [doc_id]. Multiple citations: [doc_0][doc_2].
- If context lacks the answer, say so clearly. Never fabricate.
- Be thorough yet concise.

Context:
{context}

Question: {question}

Answer:"""


# ── Public API ────────────────────────────────────────────────────────────────

def stream_answer(
    question: str,
    history: List[Dict] = None,
    extra_instructions: str = "",
    user: Optional[str] = None,
) -> Generator:
    """
    Main RAG pipeline — instruments every stage with tracing + cost recording.
    Yields answer tokens for st.write_stream compatibility.
    """
    global _last_docs

    if not os.path.exists(CHUNKS_PKL):
        yield "⚠️ No documents indexed yet. Upload a file in the sidebar."
        return

    pipeline_start = time.perf_counter()

    # ── Retrieval span ──────────────────────────────────────────────────
    with trace_context("rag_pipeline", {"user": user or "unknown", "question": question[:100]}) as root_span:
        tid = current_trace_id()

        with trace_context("retrieval"):
            t0   = time.perf_counter()
            docs = multi_query_retrieve(question)
            _last_docs = docs
            retrieval_lat = time.perf_counter() - t0

        if not docs:
            root_span.set_metadata(status="no_docs")
            yield "I couldn't find relevant content for that question."
            return

        # ── LLM generation span ─────────────────────────────────────────
        extra = f"Additional instructions: {extra_instructions}\n\n" if extra_instructions.strip() else ""
        prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chain  = prompt | _get_llm() | StrOutputParser()

        context_str = format_docs(docs)
        input_payload = {
            "context":  context_str,
            "question": question,
            "history":  _history_str(history or []),
            "extra":    extra,
        }

        # Estimate input tokens from prompt payload
        full_prompt_text = context_str + question + extra
        input_tok = _est_tokens(full_prompt_text)

        gen_start = time.perf_counter()
        with trace_context("llm_generate", {"model": GROQ_MODEL, "num_docs": len(docs)}):
            answer_chunks: list[str] = []
            for chunk in chain.stream(input_payload):
                answer_chunks.append(chunk)
                yield chunk

        # ── Post-generation metrics ─────────────────────────────────────
        full_answer   = "".join(answer_chunks)
        output_tok    = _est_tokens(full_answer)
        gen_latency   = time.perf_counter() - gen_start
        total_latency = time.perf_counter() - pipeline_start

        record_latency("llm_generate",  gen_latency)
        record_latency("rag_e2e",       total_latency)
        record_cost(
            model         = GROQ_MODEL,
            input_tokens  = input_tok,
            output_tokens = output_tok,
            user          = user,
            trace_id      = tid,
        )

        root_span.set_metadata(
            num_docs      = len(docs),
            input_tokens  = input_tok,
            output_tokens = output_tok,
            retrieval_lat = round(retrieval_lat, 3),
            gen_latency   = round(gen_latency,   3),
            total_latency = round(total_latency,  3),
        )


def get_last_docs() -> List[Document]:
    return _last_docs


def get_followup_suggestions(question: str, answer: str) -> List[str]:
    try:
        prompt = ChatPromptTemplate.from_template(
            "Suggest 3 concise follow-up questions based on this Q&A.\n"
            "Output one per line, no numbers.\n\nQ: {question}\nA: {answer}\n\nFollow-ups:"
        )
        result = (prompt | _get_llm() | StrOutputParser()).invoke(
            {"question": question, "answer": answer}
        )
        return [q.strip() for q in result.strip().split("\n") if q.strip()][:3]
    except Exception:
        return []


def summarize_document(filename: str, chunks: List) -> str:
    t0      = time.perf_counter()
    step    = max(1, len(chunks) // 30)
    sampled = chunks[::step][:30]
    context = "\n\n---\n\n".join(
        f"[Page {c.metadata.get('page','?')}]: {c.page_content}" for c in sampled
    )
    prompt = ChatPromptTemplate.from_template(
        "Write a structured summary of this document.\n"
        "## Overview\n(purpose)\n## Key Topics\n- (bullets)\n## Important Details\n(key facts)\n\n"
        "Document: {filename}\nContent:\n{context}\n\nSummary:"
    )
    result = (prompt | _get_llm() | StrOutputParser()).invoke(
        {"filename": filename, "context": context}
    )
    record_latency("summarize", time.perf_counter() - t0)
    return result
