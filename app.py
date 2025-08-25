#!/usr/bin/env python3
"""
app.py — Intelligent Customer Support Chatbot (Single-file implementation)

Covers the 4 required parts end‑to‑end:

Part 1: Intent Classification
  • Multi-intent detection (billing, technical, account, complaints, general)
  • Entity extraction (order_id, email, phone, dates, product) via regex + optional spaCy
  • Confidence scoring and routing signals

Part 2: Knowledge Base Integration (RAG)
  • Load documents (PDF/Markdown/TXT/CSV)
  • Chunking with metadata
  • Hybrid Retrieval: FAISS (semantic) + BM25 (keyword)
  • Optional Cross-Encoder reranker (HuggingFace transformers)
  • Context packing with token budget and citations
  • Answer extraction / generation (Gemini or OpenAI or GROQ) via unified LLM interface

Part 3: Conversation Management
  • Memory: buffer + summarization (token-aware) for long dialogs
  • Clarifying questions when ambiguity/low-confidence
  • Graceful fallbacks (no-answer, handoff)

Part 4: Analytics & Improvement
  • Request/response timing, retrieval metrics, hit/miss analytics
  • Satisfaction signal ingestion (thumbs up/down)
  • Minimal continuous learning hooks (intent keyword expansion)

FASTAPI API
  • POST /chat  — main chat endpoint
  • POST /feedback — record satisfaction feedback
  • GET  /health — liveness
  • GET  /metrics — minimal metrics snapshot

CLI
  • python app.py --rebuild-index --data ./data

Notes
  • Keep API keys in environment variables.
  • External deps are optional and guarded (spaCy, transformers, google-langextract).
  • This file is intentionally verbose with comments for clarity.
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import fnmatch
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------
# Configuration & Logging
# ------------------------------

APP_NAME = "rag_support_bot"
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_DIR = os.getenv("INDEX_DIR", "./index_store")
LOG_DIR = os.getenv("LOG_DIR", "./logs")

# Model config (env-driven)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cohere")  # cohere | openai | sentence-transformers
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-multilingual-v3.0")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini | openai | groq
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

# Optional reranker
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-multilingual-v3.0")

# Memory / tokens
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "5"))
HYBRID_WEIGHTS = tuple(float(x) for x in os.getenv("HYBRID_WEIGHTS", "0.6,0.4").split(","))  # (semantic, keyword)

# Clarification thresholds
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.3"))
ANSWER_MIN_CONFIDENCE = float(os.getenv("ANSWER_MIN_CONFIDENCE", "0.35"))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create directories
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(LOG_DIR) / f"{APP_NAME}.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(APP_NAME)

# ------------------------------
# Utility helpers
# ------------------------------

ISO_8601 = "%Y-%m-%dT%H:%M:%SZ"

def utcnow_str() -> str:
    try:
        # Python 3.11+
        return datetime.now(datetime.UTC).strftime(ISO_8601)
    except AttributeError:
        # Python 3.8-3.10
        return datetime.utcnow().strftime(ISO_8601)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# ------------------------------
# Data structures
# ------------------------------

@dataclass
class DocChunk:
    doc_id: str
    chunk_id: str
    text: str
    source: str
    page: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: DocChunk
    score: float
    rank: int
    method: str  # "semantic" or "bm25" or "rerank"


@dataclass
class ChatTurn:
    role: str  # user|assistant|system
    content: str
    timestamp: str


@dataclass
class ChatState:
    history: List[ChatTurn] = field(default_factory=list)
    summary: str = ""
    token_estimate: int = 0
    slots: Dict[str, Any] = field(default_factory=lambda: {
        "order_id": None,
        "purchase_date": None,
        "product_name": None,
        "issue_type": None,
        "issue_description": None,
        "additional_info": None,
        "email": None,
        "amount": None
    })
    ticket_created: bool = False
    conversation_stage: str = "greeting"  # greeting, collecting_info, resolved


# ------------------------------
# Lightweight token estimator (character heuristic)
# ------------------------------

def estimate_tokens(text: str) -> int:
    # Approximation: 1 token ~= 4 chars for English text
    return max(1, math.ceil(len(text) / 4))


# ------------------------------
# Document loading & chunking
# ------------------------------

SUPPORTED_EXTS = ["*.txt", "*.md", "*.markdown", "*.csv", "*.pdf"]


def discover_files(data_dir: str) -> List[Path]:
    root = Path(data_dir)
    files: List[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        for pat in SUPPORTED_EXTS:
            if fnmatch.fnmatch(p.name.lower(), pat):
                files.append(p)
                break
    return files


try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def load_pdf(path: Path) -> str:
    if fitz is None:
        logger.warning("PyMuPDF not installed; cannot parse PDF: %s", path)
        return ""
    try:
        doc = fitz.open(str(path))
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts)
    except Exception as e:
        logger.exception("PDF parse failed for %s: %s", path, e)
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


def build_corpus(data_dir: str) -> List[DocChunk]:
    files = discover_files(data_dir)
    corpus: List[DocChunk] = []
    for fp in files:
        raw = ""
        if fp.suffix.lower() == ".pdf":
            raw = load_pdf(fp)
        else:
            raw = read_text(fp)
        if not raw.strip():
            continue
        doc_id = sha1(str(fp.resolve()))
        chunks = chunk_text(raw)
        for idx, ch in enumerate(chunks):
            corpus.append(
                DocChunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-{idx}",
                    text=ch,
                    source=str(fp),
                    page=None,
                    meta={"filename": fp.name},
                )
            )
    logger.info("Built corpus: %d chunks from %d files", len(corpus), len(files))
    return corpus


# ------------------------------
# Embeddings + FAISS (semantic)
# ------------------------------

class EmbeddingBackend:
    def __init__(self, provider: str = EMBEDDING_PROVIDER, model: str = EMBEDDING_MODEL):
        self.provider = provider
        self.model = model
        self._setup()

    def _setup(self):
        self.client = None
        if self.provider == "cohere":
            try:
                import cohere
                self.client = cohere.Client(COHERE_API_KEY)
            except Exception:
                logger.exception("Cohere client init failed")
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            except Exception:
                logger.exception("OpenAI client init failed")
        elif self.provider == "sentence-transformers":
            try:
                # Fallback to Cohere if sentence-transformers not available
                import cohere
                self.client = cohere.Client(COHERE_API_KEY)
                self.provider = "cohere"  # Switch to Cohere
                self.model = "embed-multilingual-v3.0"  # Use Cohere model
            except Exception:
                logger.exception("SentenceTransformer fallback failed")
        else:
            logger.warning("Unknown embedding provider: %s", self.provider)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "cohere" and self.client:
            resp = self.client.embed(texts=texts, model=self.model, input_type="search_query")
            return [v for v in resp.embeddings]
        elif self.provider == "openai" and self.client:
            # Uses text-embedding-3-large/small style
            out = []
            for t in texts:
                emb = self.client.embeddings.create(model=self.model, input=t).data[0].embedding
                out.append(emb)
            return out
        elif self.provider == "sentence-transformers" and self.client:
            # This should not happen now as we fallback to Cohere
            return self.client.encode(texts, normalize_embeddings=True).tolist()
        else:
            # Fallback: random vectors (for dev only)
            logger.warning("Embedding fallback to random vectors; set proper provider & key.")
            dim = 384
            return [[random.random() for _ in range(dim)] for _ in texts]


class FAISSIndex:
    def __init__(self, dim: int = 384, index_dir: str = INDEX_DIR):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.id2chunk: Dict[int, DocChunk] = {}
        self.faiss = None

    def build(self, chunks: List[DocChunk], embedder: EmbeddingBackend):
        try:
            import faiss
        except Exception as e:
            logger.error("FAISS import failed: %s", e)
            raise
        texts = [c.text for c in chunks]
        vecs = embedder.embed(texts)
        if not vecs:
            raise RuntimeError("No embeddings produced.")
        self.dim = len(vecs[0])
        self.faiss = faiss.IndexFlatIP(self.dim)
        import numpy as np
        mat = np.array(vecs, dtype="float32")
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(mat)
        self.faiss.add(mat)
        self.id2chunk = {i: chunks[i] for i in range(len(chunks))}
        self._save()
        logger.info("FAISS built with %d vectors (dim=%d)", len(chunks), self.dim)

    def _paths(self) -> Tuple[Path, Path]:
        return (self.index_dir / "faiss.index", self.index_dir / "faiss_meta.json")

    def _save(self):
        if self.faiss is None:
            return
        import faiss
        idx_path, meta_path = self._paths()
        faiss.write_index(self.faiss, str(idx_path))
        meta = {
            "dim": self.dim,
            "id2chunk": {str(i): dataclasses.asdict(c) for i, c in self.id2chunk.items()},
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def load(self) -> bool:
        try:
            import faiss
            idx_path, meta_path = self._paths()
            if not idx_path.exists() or not meta_path.exists():
                return False
            self.faiss = faiss.read_index(str(idx_path))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.dim = meta["dim"]
            self.id2chunk = {int(i): DocChunk(**c) for i, c in meta["id2chunk"].items()}
            logger.info("Loaded FAISS index with %d vectors", len(self.id2chunk))
            return True
        except Exception as e:
            logger.exception("Failed to load FAISS: %s", e)
            return False

    def search(self, query: str, embedder: EmbeddingBackend, k: int = TOP_K) -> List[RetrievalResult]:
        if self.faiss is None:
            return []
        import numpy as np
        vec = embedder.embed([query])[0]
        vec = np.array([vec], dtype="float32")
        # Normalize
        import faiss as _fa
        _fa.normalize_L2(vec)
        scores, idxs = self.faiss.search(vec, k)
        results: List[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
            if idx == -1:
                continue
            results.append(
                RetrievalResult(
                    chunk=self.id2chunk.get(int(idx)),
                    score=float(score),
                    rank=rank + 1,
                    method="semantic",
                )
            )
        return results


# ------------------------------
# Simple BM25 (keyword) retriever
# ------------------------------

class BM25:
    def __init__(self, docs: List[DocChunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = 1.0
        self.doc_freq: Dict[str, int] = collections.defaultdict(int)
        self.doc_terms: List[List[str]] = []
        self.doc_len: List[int] = []
        self._build()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _build(self):
        for d in self.docs:
            terms = self._tokenize(d.text)
            self.doc_terms.append(terms)
            self.doc_len.append(len(terms))
            for t in set(terms):
                self.doc_freq[t] += 1
        self.avgdl = sum(self.doc_len) / max(1, self.N)

    def score(self, qterms: List[str], doc_idx: int) -> float:
        score = 0.0
        terms = self.doc_terms[doc_idx]
        tf = collections.Counter(terms)
        dl = self.doc_len[doc_idx]
        for t in qterms:
            if t not in self.doc_freq:
                continue
            n = self.doc_freq[t]
            idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            freq = tf.get(t, 0)
            score += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def search(self, query: str, k: int = TOP_K) -> List[RetrievalResult]:
        qterms = self._tokenize(query)
        scored: List[Tuple[int, float]] = []
        for i, _ in enumerate(self.docs):
            s = self.score(qterms, i)
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[RetrievalResult] = []
        for rank, (idx, s) in enumerate(scored[:k]):
            results.append(
                RetrievalResult(chunk=self.docs[idx], score=float(s), rank=rank + 1, method="bm25")
            )
        return results


# ------------------------------
# Optional: Cross-encoder reranker
# ------------------------------

class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self.enabled = RERANK_ENABLED
        self.client = None
        if self.enabled:
            try:
                import cohere
                self.client = cohere.Client(COHERE_API_KEY)
            except Exception:
                logger.exception("Cohere reranker init failed; continuing without rerank.")
                self.enabled = False

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = TOP_K) -> List[RetrievalResult]:
        if not self.enabled or not candidates:
            return candidates[:top_k]
        
        try:
            # Use Cohere rerank API
            documents = [c.chunk.text for c in candidates]
            response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=len(documents)
            )
            
            # Create a mapping of document text to RetrievalResult
            doc_to_result = {c.chunk.text: c for c in candidates}
            
            # Reorder based on Cohere rerank results
            out: List[RetrievalResult] = []
            for rank, result in enumerate(response.results[:top_k]):
                # Check if result.document exists and has text
                if not result.document or not hasattr(result.document, 'text') or not result.document.text:
                    continue
                    
                doc_text = result.document.text
                original_result = doc_to_result.get(doc_text)
                if original_result:
                    out.append(
                        RetrievalResult(
                            chunk=original_result.chunk, 
                            score=float(result.relevance_score), 
                            rank=rank + 1, 
                            method="rerank"
                        )
                    )
            
            # If Cohere rerank fails, return original candidates
            if not out:
                return candidates[:top_k]
            
            return out
            
        except Exception as e:
            logger.exception("Cohere rerank failed: %s", e)
            return candidates[:top_k]


# ------------------------------
# Hybrid retrieval orchestration
# ------------------------------

class HybridRetriever:
    def __init__(self, faiss: FAISSIndex, bm25: BM25, embedder: EmbeddingBackend, weights: Tuple[float, float] = HYBRID_WEIGHTS):
        self.faiss = faiss
        self.bm25 = bm25
        self.embedder = embedder
        self.w_sem, self.w_kw = weights

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[RetrievalResult]:
        sem = self.faiss.search(query, self.embedder, k=top_k * 3)  # over-fetch
        kw = self.bm25.search(query, k=top_k * 3)
        pool: Dict[str, RetrievalResult] = {}
        # Merge with weighted score; unique by chunk_id
        for r in sem:
            cid = r.chunk.chunk_id
            s = self.w_sem * r.score
            if cid not in pool or s > pool[cid].score:
                pool[cid] = RetrievalResult(chunk=r.chunk, score=s, rank=r.rank, method=r.method)
        for r in kw:
            cid = r.chunk.chunk_id
            s = (pool.get(cid).score if cid in pool else 0.0) + self.w_kw * r.score
            pool[cid] = RetrievalResult(chunk=r.chunk, score=s, rank=r.rank, method=r.method)
        merged = sorted(pool.values(), key=lambda x: x.score, reverse=True)
        return merged[: top_k * 2]


# ------------------------------
# LangExtract (structured extraction)
# ------------------------------

class StructuredExtractor:
    def __init__(self):
        self.available = False
        self.Schema = None
        self.GeminiLLM = None
        self.OpenAILLM = None
        
        try:
            # Try to import LangExtract components
            import importlib.util
            
            # Check if langextract is available
            if importlib.util.find_spec("langextract"):
                # Use string-based imports to avoid Pylance errors
                import importlib
                
                # Import Schema
                langextract_module = importlib.import_module("langextract")
                self.Schema = getattr(langextract_module, "Schema", None)
                
                # Try to get LLM classes from different possible locations
                self.GeminiLLM = None
                self.OpenAILLM = None
                
                # Try newer integration paths
                try:
                    google_module = importlib.import_module("langextract.integrations.google")
                    self.GeminiLLM = getattr(google_module, "Gemini", None)
                except ImportError:
                    pass
                
                try:
                    openai_module = importlib.import_module("langextract.integrations.openai")
                    self.OpenAILLM = getattr(openai_module, "OpenAI", None)
                except ImportError:
                    pass
                
                # Try older model_providers path
                if not self.GeminiLLM or not self.OpenAILLM:
                    try:
                        providers_module = importlib.import_module("langextract.model_providers")
                        if not self.GeminiLLM:
                            self.GeminiLLM = getattr(providers_module, "GeminiLLM", None)
                        if not self.OpenAILLM:
                            self.OpenAILLM = getattr(providers_module, "OpenAILLM", None)
                    except ImportError:
                        pass
                
                if self.Schema:
                    self.available = True
                    logger.info("LangExtract loaded successfully for structured extraction.")
                else:
                    self.available = False
                    logger.warning("LangExtract Schema not found")
            else:
                raise ImportError("langextract module not found")
                
        except ImportError:
            logger.warning("LangExtract not available. Install google-langextract to enable structured extraction.")
            self.available = False
        except Exception as e:
            logger.warning("LangExtract import failed: %s", e)
            self.available = False
        if self.available and self.Schema:
            try:
                self.schema = self.Schema.from_dict(
                    {
                        "order_id": "string: The customer order number if present",
                        "issue_type": "string: refund_request | shipping_delay | damaged_item | billing_issue | account_issue | general",
                        "product": "string: Product or SKU if present",
                        "email": "string: Email if present",
                        "amount": "string: Monetary amount if present",
                        "date": "string: Date if present (ISO or natural language)",
                    }
                )
                logger.info("LangExtract schema created successfully")
            except Exception as e:
                logger.warning("Failed to create LangExtract schema: %s", e)
                self.available = False
                self.schema = None

    def extract(self, text: str) -> Dict[str, Any]:
        # Always fallback to regex extraction if LangExtract is not available
        if not self.available or not hasattr(self, 'schema') or not self.schema:
            return naive_entity_extract(text)
        
        try:
            # Check if run_extraction is available
            import importlib.util
            if not importlib.util.find_spec("langextract"):
                return naive_entity_extract(text)
                
            # Use string-based import to avoid Pylance errors
            import importlib
            langextract_module = importlib.import_module("langextract")
            run_extraction = getattr(langextract_module, "run_extraction", None)
            
            if not run_extraction:
                return naive_entity_extract(text)
            
            llm = None
            
            # Try to create LLM instance if available
            if (LLM_PROVIDER == "gemini" and GEMINI_API_KEY and 
                hasattr(self, 'GeminiLLM') and self.GeminiLLM):
                try:
                    # Try different initialization patterns
                    try:
                        llm = self.GeminiLLM(model=LLM_MODEL)
                    except TypeError:
                        # Some versions expect different parameters
                        llm = self.GeminiLLM()
                except Exception:
                    pass
            elif (LLM_PROVIDER == "openai" and OPENAI_API_KEY and 
                  hasattr(self, 'OpenAILLM') and self.OpenAILLM):
                try:
                    try:
                        llm = self.OpenAILLM(model=LLM_MODEL)
                    except TypeError:
                        llm = self.OpenAILLM()
                except Exception:
                    pass
            
            if llm:
                result = run_extraction(self.schema, text, llm)
                return result or {}
            else:
                return naive_entity_extract(text)
                
        except ImportError:
            logger.warning("LangExtract not available; using regex fallback.")
            return naive_entity_extract(text)
        except Exception as e:
            logger.exception("LangExtract extraction failed; fallback to regex: %s", e)
            return naive_entity_extract(text)


# ------------------------------
# Naive regex-based entity extraction (fallback)
# ------------------------------

ORDER_ID_RE = re.compile(r"(?:order|ord|#)\s*[:#-]?\s*([A-Z0-9-]{4,})", re.I)
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s-]{7,}\d")
AMOUNT_RE = re.compile(r"(?:rs|inr|usd|\$)\s*([0-9]+(?:\.[0-9]{1,2})?)", re.I)
DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{1,2} \w+ \d{4})\b")

PRODUCT_HINTS = ["shirt", "phone", "laptop", "charger", "headphones", "jeans", "dress", "earbuds"]


def naive_entity_extract(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = ORDER_ID_RE.search(text)
    if m:
        out["order_id"] = m.group(1)
    m = EMAIL_RE.search(text)
    if m:
        out["email"] = m.group(0)
    m = PHONE_RE.search(text)
    if m:
        out["phone"] = m.group(0)
    m = AMOUNT_RE.search(text)
    if m:
        out["amount"] = m.group(1)
    m = DATE_RE.search(text)
    if m:
        out["date"] = m.group(0)
    # crude product guess
    for p in PRODUCT_HINTS:
        if re.search(rf"\b{re.escape(p)}\b", text, re.I):
            out.setdefault("product", p)
            break
    return out


# ------------------------------
# Intent classification (multi-intent + confidence)
# ------------------------------

INTENTS = ["billing", "technical", "account", "complaints", "general"]

INTENT_KEYWORDS = {
    "billing": ["invoice", "payment", "charge", "refund", "bill", "amount"],
    "technical": ["bug", "error", "crash", "not working", "technical", "issue", "broken"],
    "account": ["login", "password", "account", "email", "sign in", "profile"],
    "complaints": ["complaint", "angry", "bad", "late", "broken", "damaged", "wrong size"],
    "general": ["query", "question", "info", "information", "help"],
}


def softmax(xs: List[float]) -> List[float]:
    m = max(xs) if xs else 0
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1
    return [e / s for e in exps]


def score_intents(text: str) -> Dict[str, float]:
    # Keyword hit-rate + simple heuristics; optionally blend with LLM zero-shot
    scores: List[float] = []
    for intent in INTENTS:
        hits = sum(1 for kw in INTENT_KEYWORDS[intent] if kw in text.lower())
        scores.append(float(hits))
    probs = softmax(scores)
    return {INTENTS[i]: probs[i] for i in range(len(INTENTS))}


def multi_intent(text: str, threshold: float = 0.15) -> List[Tuple[str, float]]:
    probs = score_intents(text)
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return [(k, v) for k, v in ranked if v >= threshold]


# ------------------------------
# Embeddings-based intent classifier (Cohere)
# ------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-8
    nb = math.sqrt(sum(y * y for y in b)) or 1e-8
    return dot / (na * nb)


INTENT_SEEDS: Dict[str, List[str]] = {
    "billing": [
        "refund request",
        "return request",
        "exchange request",
        "return an order",
        "charged incorrectly",
        "payment failed",
        "billing issue",
    ],
    "technical": [
        "app not working",
        "website error",
        "bug in product",
        "can't login due to error",
        "feature broken",
    ],
    "account": [
        "reset password",
        "update email",
        "account locked",
        "change profile details",
        "login help",
    ],
    "complaints": [
        "package arrived damaged",
        "late delivery",
        "poor quality",
        "wrong size received",
        "file a complaint",
    ],
    "general": [
        "hi",
        "hello",
        "i need help",
        "have a question",
        "support",
    ],
}


class IntentClassifier:
    def __init__(self, embedder: "EmbeddingBackend"):
        self.embedder = embedder
        self.centroids: Dict[str, List[float]] = {}
        try:
            # Build centroids by averaging seed embeddings per intent
            all_texts: List[str] = []
            index_map: List[Tuple[str, int]] = []  # (intent, local_index)
            for intent, seeds in INTENT_SEEDS.items():
                for s in seeds:
                    index_map.append((intent, len(all_texts)))
                    all_texts.append(s)
            if all_texts:
                vecs = self.embedder.embed(all_texts)
                accum: Dict[str, List[float]] = {k: [0.0] * len(vecs[0]) for k in INTENT_SEEDS.keys()}
                counts: Dict[str, int] = {k: 0 for k in INTENT_SEEDS.keys()}
                for (intent, idx), v in zip(index_map, vecs):
                    accum[intent] = [a + b for a, b in zip(accum[intent], v)]
                    counts[intent] += 1
                for intent in accum.keys():
                    c = counts[intent] or 1
                    self.centroids[intent] = [a / c for a in accum[intent]]
        except Exception:
            logger.exception("Failed to initialize embeddings-based intent centroids; will fallback to keywords.")
            self.centroids = {}

    def classify(self, text: str, threshold: float = 0.05) -> List[Tuple[str, float]]:
        try:
            if not self.centroids:
                raise RuntimeError("No centroids")
            qv = self.embedder.embed([text])[0]
            sims: Dict[str, float] = {}
            for intent, cv in self.centroids.items():
                sims[intent] = _cosine_similarity(qv, cv)
            # Convert similarities to probabilities for stability
            intents = list(sims.keys())
            probs = softmax(list(sims.values()))
            ranked = sorted(zip(intents, probs), key=lambda x: x[1], reverse=True)
            # Prefer non-general over general when close
            if ranked and ranked[0][0] == "general" and len(ranked) > 1:
                second = ranked[1]
                if second[1] >= ranked[0][1] * 0.9:  # within 10%
                    ranked[0], ranked[1] = ranked[1], ranked[0]
            return [(k, float(v)) for k, v in ranked if v >= threshold]
        except Exception:
            # Fallback to keyword model on any failure
            return multi_intent(text)


# ------------------------------
# LLM interface (Gemini/OpenAI/GROQ)
# ------------------------------

class LLM:
    def __init__(self, provider: str = LLM_PROVIDER, model: str = LLM_MODEL):
        self.provider = provider
        self.model = model
        self._setup()

    def _setup(self):
        self.client = None
        try:
            if self.provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self.client = genai.GenerativeModel(self.model)
            elif self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            elif self.provider == "groq":
                import groq
                self.client = groq.Groq(api_key=GROQ_API_KEY)
        except Exception:
            logger.exception("LLM client init failed — will fallback to template responses.")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        if self.provider == "gemini" and self.client:
            try:
                resp = self.client.generate_content(prompt)
                return getattr(resp, "text", "") or ""
            except Exception:
                logger.exception("Gemini generation failed")
                return ""
        elif self.provider == "openai" and self.client:
            try:
                chat = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return chat.choices[0].message.content
            except Exception:
                logger.exception("OpenAI chat failed")
                return ""
        elif self.provider == "groq" and self.client:
            try:
                chat = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return chat.choices[0].message.content
            except Exception:
                logger.exception("GROQ chat failed")
                return ""
        # Fallback deterministic template
        return "I'm unable to generate a response right now. Please try again later."


# ------------------------------
# Context packing & answer synthesis
# ------------------------------

ANSWER_SYS_PROMPT = (
    "You are a helpful, terse customer support assistant. "
    "Use ONLY the provided context. If missing, say you don't have enough info and suggest next steps. "
    "Cite sources as [filename] at the end where relevant."
)


def build_prompt(user_query: str, context_chunks: List[DocChunk], chat_state: ChatState, extracted: Dict[str, Any]) -> str:
    # Build citations map
    ctx_parts = []
    token_budget = MAX_CONTEXT_TOKENS
    for ch in context_chunks:
        t = ch.text.strip()
        if not t:
            continue
        if token_budget <= 0:
            break
        tk = estimate_tokens(t)
        if tk > token_budget:
            t = t[: token_budget * 4]
        ctx_parts.append(f"[Source: {Path(ch.source).name}]\n{t}")
        token_budget -= tk

    history_snippets = []
    # In production, use a summarizer; here we keep last few turns
    for turn in chat_state.history[-6:]:
        history_snippets.append(f"{turn.role}: {turn.content}")
    history_text = "\n".join(history_snippets)

    extracted_text = json.dumps(extracted, ensure_ascii=False)

    prompt = f"""
{ANSWER_SYS_PROMPT}

Conversation so far:
{history_text}

User question:
{user_query}

Extracted structured info (may be partial):
{extracted_text}

Knowledge context:
{os.linesep.join(ctx_parts)}

Write a concise, actionable answer for the user. When appropriate, ask exactly one clarifying question. Include citations using [filename] if you quoted or used a specific source.
""".strip()
    return prompt


# ------------------------------
# Clarification & fallback logic
# ------------------------------

def should_ask_clarification(retrievals: List[RetrievalResult], min_score: float = RETRIEVAL_MIN_SCORE) -> bool:
    if not retrievals:
        return True
    top = retrievals[0].score
    return top < min_score


def build_clarification_question(intents: List[Tuple[str, float]], entities: Dict[str, Any]) -> str:
    # Choose clarifying based on missing core fields
    if "order_id" not in entities:
        return "Could you share your order ID so I can check the details?"
    if not intents:
        return "Could you tell me whether this is about billing, a technical issue, your account, or a complaint?"
    return "Could you clarify a bit more so I can help precisely?"


# ------------------------------
# Analytics store (in-memory + log file persistence)
# ------------------------------

class Analytics:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.interactions: List[Dict[str, Any]] = []

    def record(self, payload: Dict[str, Any]):
        self.interactions.append(payload)
        try:
            p = Path(LOG_DIR) / "interactions.jsonl"
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to persist interaction log")

    def snapshot(self) -> Dict[str, Any]:
        total = len(self.interactions)
        sats = [x.get("feedback", {}).get("satisfaction") for x in self.interactions if x.get("feedback")]
        sat_rate = sum(1 for s in sats if s == "up") / max(1, len(sats)) if sats else None
        avg_latency = None
        latencies = [x.get("timings", {}).get("total_ms", 0) for x in self.interactions if x.get("timings")]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
        return {
            "total_interactions": total,
            "satisfaction_rate": sat_rate,
            "avg_latency_ms": avg_latency,
        }


ANALYTICS = Analytics()


# ------------------------------
# Core Chat Orchestrator
# ------------------------------

class ChatOrchestrator:
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        # Load or build index
        self.embedder = EmbeddingBackend()
        self.faiss = FAISSIndex()
        ok = self.faiss.load()
        if not ok:
            logger.info("No FAISS index found. Building now from %s", data_dir)
            corpus = build_corpus(data_dir)
            if not corpus:
                logger.warning("Empty corpus; hybrid retrieval will be degraded.")
            self.faiss.build(corpus, self.embedder)
        # BM25 over same corpus
        corpus = [c for _, c in self.faiss.id2chunk.items()] if self.faiss.id2chunk else build_corpus(data_dir)
        self.bm25 = BM25(corpus)
        self.hybrid = HybridRetriever(self.faiss, self.bm25, self.embedder, HYBRID_WEIGHTS)
        self.reranker = Reranker()
        self.llm = LLM()
        self.extractor = StructuredExtractor()
        # Embeddings-based intent classifier (falls back automatically)
        self.intent_classifier = IntentClassifier(self.embedder)
        self.sessions: Dict[str, ChatState] = {}

    # --- Session helpers ---
    def _get_state(self, session_id: str) -> ChatState:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatState()
        return self.sessions[session_id]

    def _append_history(self, state: ChatState, role: str, content: str):
        state.history.append(ChatTurn(role=role, content=content, timestamp=utcnow_str()))
        # naive token estimate, rotate if too long
        tot = sum(estimate_tokens(t.content) for t in state.history)
        state.token_estimate = tot
        while tot > MAX_CONTEXT_TOKENS * 1.5 and len(state.history) > 4:
            # drop oldest non-system turn
            state.history.pop(0)
            tot = sum(estimate_tokens(t.content) for t in state.history)
            state.token_estimate = tot

    def _handle_smalltalk(self, user_text: str) -> Optional[str]:
        """Handle basic greetings and smalltalk without going through RAG."""
        user_text = user_text.lower().strip()
        # If the message contains clear issue keywords or error codes, do not smalltalk
        negative_keywords = [
            "error", "exception", "crash", "crashing", "failed", "failure",
            "cannot", "can't", "unable", "bug", "issue", "damaged", "return", "refund",
            "password", "login", "account"
        ]
        if any(k in user_text for k in negative_keywords) or re.search(r"\b\d{3,}\b", user_text):
            return None
        
        # Handle very short responses (whole-word match only)
        if len(user_text.split()) <= 3:
            smalltalk_responses = {
                "hi": "Hello! 👋 How can I assist you today?",
                "hello": "Hi there! 👋 What can I help you with?",
                "hey": "Hey! 👋 How can I support you today?",
                "thank you": "You're welcome! 😊 Glad I could help.",
                "thanks": "You're welcome! 😊 Happy to assist.",
                "yes": "Got it! 👍",
                "no": "Alright, I won't proceed with that then.",
                "ok": "Perfect! 👍",
                "okay": "Great! 👍"
            }
            tokens = set(re.findall(r"[a-zA-Z]+", user_text))
            for key, reply in smalltalk_responses.items():
                if key in tokens:
                    return reply
        
        # Handle longer but still simple greetings (whole-word and short length)
        tokens = re.findall(r"[a-zA-Z]+", user_text)
        token_set = set(tokens)
        greeting_tokens = {"hi", "hello", "hey"}
        phrase_greetings = ["good morning", "good afternoon", "good evening"]
        if any(p in user_text for p in phrase_greetings) and len(tokens) <= 5:
            return "Hello! 👋 How can I assist you today?"
        if (token_set & greeting_tokens) and len(tokens) <= 4:
            return "Hello! 👋 How can I assist you today?"
        
        # Don't treat date-related words as smalltalk
        date_words = ["yesterday", "today", "tomorrow", "last week", "last month", "this week", "this month"]
        if any(word in user_text for word in date_words):
            return None
        
        return None

    def _fill_slots(self, state: ChatState, entities: Dict[str, Any], user_text: str) -> None:
        """Fill conversation slots with extracted entities and user input."""
        # Update slots with extracted entities
        for key, value in entities.items():
            if key in state.slots and value and not state.slots[key]:
                state.slots[key] = value
        
        # Also try to extract from user text directly
        user_lower = user_text.lower().strip()

        # Issue description: capture meaningful description but avoid auto-filling on short first messages
        if not state.slots["issue_description"]:
            trivial = {"hi", "hello", "hey", "ok", "okay", "thanks", "thank you", "yes", "no"}
            meaningful_keywords = ["error", "code", "crash", "cannot", "can't", "unable", "exception", "stack", "trace", "failed", "failure"]
            token_count = len(re.findall(r"[\w']+", user_lower))
            question_words = ["how may i help you", "how can i help", "what can i do"]
            is_meaningful = (
                token_count >= 6 or any(k in user_lower for k in meaningful_keywords)
            )
            if user_lower not in trivial and not any(q in user_lower for q in question_words):
                # Only set description if we're in collecting phase or the message is meaningful enough
                if state.conversation_stage == "collecting_info" or is_meaningful:
                    state.slots["issue_description"] = user_text.strip()
        
        # Handle order ID extraction - more flexible patterns, require digits
        if not state.slots["order_id"]:
            # Try multiple patterns for order ID
            order_patterns = [
                r"(?:order|ord|#)\s*[:#-]?\s*([A-Z]*\d[A-Z0-9-]{2,})",  # must contain at least one digit
                r"(?:id|number)\s*(?:is|:)\s*([A-Z]*\d[A-Z0-9-]{2,})",
                r"\b(\d{3,})\b",                                        # Any 3+ digit sequence (numbers only)
                r"\b([A-Z]{2,}\d{2,})\b",                              # "AB12345" or "ORD12345"
                r"\b(\d{2,}[A-Z]{2,})\b",                              # "12345AB"
            ]
            
            for pattern in order_patterns:
                order_match = re.search(pattern, user_text, re.I)
                if order_match:
                    state.slots["order_id"] = order_match.group(1)
                    break
            
            # If no pattern match, try to extract any reasonable order ID from user input
            if not state.slots["order_id"]:
                # Look for common order ID indicators
                order_indicators = ["order", "id", "number", "tracking", "reference", "confirmation"]
                for indicator in order_indicators:
                    if re.search(rf"\b{re.escape(indicator)}\b", user_lower):
                        # Extract the word after the indicator
                        words = user_text.split()
                        for i, word in enumerate(words):
                            if re.fullmatch(rf"{re.escape(indicator)}\b.*", word.lower()) or word.lower() == indicator.lower():
                                # Try to get the next word as order ID
                                if i + 1 < len(words):
                                    potential_id = words[i + 1].strip(".,!?")
                                    if len(potential_id) >= 3 and any(ch.isdigit() for ch in potential_id):
                                        state.slots["order_id"] = potential_id
                                        break
                        break
        else:
            # If we already have an order_id, allow explicit overwrite if user provides a new clear ID
            explicit = re.search(r"\b(?:order\s*id|order|id|number)\b.*?([A-Z]*\d[A-Z0-9-]{2,}|\d{3,})", user_text, re.I)
            if explicit:
                new_id = explicit.group(1)
                if new_id and new_id != state.slots["order_id"]:
                    state.slots["order_id"] = new_id

        # If the current order_id is bogus (no digits), clear it so clarifiers behave
        if state.slots.get("order_id") and not any(ch.isdigit() for ch in str(state.slots["order_id"])):
            state.slots["order_id"] = None
        
        # Handle email extraction
        if not state.slots["email"]:
            email_match = re.search(r"[a-zA-Z0-9_.+%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_text)
            if email_match:
                state.slots["email"] = email_match.group(0)
        
        # Handle "no order ID" responses
        if "no order" in user_lower or "don't have" in user_lower or "n/a" in user_lower:
            if "order" in user_lower:
                state.slots["order_id"] = "N/A"
        
        # Handle product names - more flexible and accepting
        if not state.slots["product_name"]:
            # First, try to extract from entities if available
            if "product" in entities and entities["product"]:
                state.slots["product_name"] = entities["product"]
            else:
                # Look for product keywords as hints, but be more flexible
                product_keywords = [
                    # Clothing & Fashion
                    "shirt", "t-shirt", "tshirt", "pants", "jeans", "dress", "skirt", "jacket", "coat", "sweater", "hoodie", "sweatshirt",
                    "shoes", "boots", "sandals", "sneakers", "trainers", "heels", "flats", "loafers", "oxfords",
                    "bag", "purse", "handbag", "backpack", "wallet", "belt", "scarf", "hat", "cap", "sunglasses",
                    
                    # Electronics
                    "phone", "smartphone", "mobile", "laptop", "computer", "pc", "tablet", "ipad", "watch", "smartwatch",
                    "headphones", "earbuds", "earphones", "speaker", "camera", "tv", "television", "monitor", "keyboard", "mouse",
                    "charger", "cable", "adapter", "powerbank", "battery", "case", "cover", "screen protector",
                    
                    # Personal Care & Beauty
                    "shampoo", "conditioner", "soap", "body wash", "lotion", "cream", "moisturizer", "sunscreen", "sunscreen lotion",
                    "makeup", "cosmetics", "lipstick", "mascara", "foundation", "concealer", "eyeshadow", "blush", "powder",
                    "perfume", "cologne", "deodorant", "toothpaste", "toothbrush", "razor", "shaving cream", "hair brush", "comb",
                    
                    # Home & Kitchen
                    "book", "magazine", "newspaper", "furniture", "chair", "table", "bed", "sofa", "couch", "lamp", "mirror",
                    "kitchen", "utensils", "pots", "pans", "dishes", "plates", "bowls", "cups", "glasses", "mugs",
                    "appliances", "refrigerator", "fridge", "microwave", "oven", "stove", "dishwasher", "washing machine", "dryer",
                    
                    # Sports & Outdoor
                    "bicycle", "bike", "tent", "sleeping bag", "backpack", "hiking", "fishing", "golf", "tennis", "basketball",
                    "football", "soccer", "baseball", "volleyball", "swimming", "yoga", "fitness", "exercise", "gym equipment",
                    
                    # Toys & Games
                    "toy", "game", "puzzle", "board game", "video game", "console", "controller", "doll", "action figure", "stuffed animal",
                    
                    # Food & Beverages
                    "food", "snack", "beverage", "drink", "coffee", "tea", "juice", "water", "soda", "pop", "chocolate", "candy",
                    
                    # Health & Wellness
                    "vitamins", "supplements", "medicine", "medication", "bandage", "bandaid", "thermometer", "pillow", "blanket", "towel"
                ]
                
                # Check if any product keyword is mentioned
                for product in product_keywords:
                    if product in user_lower:
                        state.slots["product_name"] = product
                        break
                
                # If no keyword match, try to extract any reasonable product name from user input
                if not state.slots["product_name"]:
                    # Look for common product indicators
                    product_indicators = ["product", "item", "thing", "order", "purchase", "bought", "received"]
                    for indicator in product_indicators:
                        if indicator in user_lower:
                            # Extract the word after the indicator or before it
                            words = user_text.split()
                            for i, word in enumerate(words):
                                if indicator.lower() in word.lower():
                                    # Try to get the next word as product name
                                    if i + 1 < len(words) and len(words[i + 1]) > 2:
                                        potential_product = words[i + 1].strip(".,!?")
                                        if not potential_product.isdigit():  # Avoid numbers
                                            state.slots["product_name"] = potential_product
                                            break
                                    # Or try the previous word
                                    elif i > 0 and len(words[i - 1]) > 2:
                                        potential_product = words[i - 1].strip(".,!?")
                                        if not potential_product.isdigit():  # Avoid numbers
                                            state.slots["product_name"] = potential_product
                                            break
                            break
                    
                    # Additional safety check: don't fill product slot with question phrases
                    if not state.slots["product_name"]:
                        question_phrases = ["how may i help you", "how can i help you", "what can i do", "what should i do", "what", "how", "why", "when", "where", "who"]
                        if any(phrase in user_lower for phrase in question_phrases):
                            # Don't fill product slot with questions
                            pass
                    
                    # Only accept user input as product name if it's clearly a product
                    # Don't auto-fill with random user input to avoid confusion
                    pass
        
        # Handle dates - more comprehensive and flexible
        if not state.slots["purchase_date"]:
            # Look for various date patterns
            date_patterns = [
                r"\b(?:\d{4}-\d{2}-\d{2})\b",                    # 2023-01-26, 26-06-2003
                r"\b(?:\d{2}/\d{2}/\d{4})\b",                    # 26/01/2023, 01/26/2023
                r"\b(?:\d{1,2}\s+\w+\s+\d{4})\b",               # 26 january 2023, 26 jan 2023
                r"\b(?:\d{1,2}\s+\w+)\b",                        # 26 january, 26 jan (current year assumed)
                r"\b(?:\w+\s+\d{1,2}\s*,?\s*\d{4})\b",          # january 26, 2023, jan 26 2023
                r"\b(?:\d{1,2}-\d{2}-\d{4})\b",                 # 26-01-2023, 26-06-2003
                r"\b(?:\d{1,2}\.\d{2}\.\d{4})\b",               # 26.01.2023, 26.06.2003
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, user_text, re.I)
                if date_match:
                    state.slots["purchase_date"] = date_match.group(0)
                    break
            
            # If no pattern match, check for relative dates
            if not state.slots["purchase_date"]:
                relative_dates = [
                    "yesterday", "today", "tomorrow", "last week", "last month", 
                    "this week", "this month", "few days ago", "couple of days ago",
                    "recently", "earlier this week", "earlier this month"
                ]
                for relative_date in relative_dates:
                    if relative_date in user_lower:
                        state.slots["purchase_date"] = relative_date
                        break
            
            # Also try to extract from entities if available
            if not state.slots["purchase_date"] and "date" in entities and entities["date"]:
                state.slots["purchase_date"] = entities["date"]

    def _get_next_clarification(self, state: ChatState, intents: List[Tuple[str, float]]) -> str:
        """Get the next clarifying question based on missing slots."""
        if not intents:
            return "Could you tell me whether this is about billing, a technical issue, your account, or a complaint?"
        
        # Set issue type if we have intents
        if intents and not state.slots["issue_type"]:
            state.slots["issue_type"] = intents[0][0]
        
        # Determine required slots by intent
        intent = state.slots["issue_type"] or (intents[0][0] if intents else "general")
        if intent == "billing":
            required = ["order_id", "product_name", "purchase_date", "issue_description"]
        elif intent == "technical":
            required = ["issue_description"]
        elif intent == "account":
            required = ["email", "issue_description"]
        elif intent == "complaints":
            required = ["order_id", "issue_description"]
        else:
            required = ["issue_description"]

        # Ask missing slots with intent-aware prompts
        for slot in required:
            if not state.slots.get(slot):
                if slot == "order_id":
                    return "What is your order ID? (Examples: 12345, AB12345, or just say 'no order ID' if you don't have one)"
                if slot == "product_name":
                    if state.slots["order_id"] and state.slots["order_id"] != "N/A":
                        return f"Got it, order ID {state.slots['order_id']}. What product are you having issues with? (Examples: shampoo, phone, laptop, or just describe the item)"
                    return "What product are you having issues with? (Examples: shampoo, phone, laptop, or just describe the item)"
                if slot == "purchase_date":
                    if state.slots["product_name"]:
                        prefix = f"For {state.slots['product_name']}"
                        if state.slots["order_id"] and state.slots["order_id"] != "N/A":
                            prefix = f"For order {state.slots['order_id']} and {state.slots['product_name']}"
                        return f"Thanks! {prefix}, when did you purchase this? (Examples: today, yesterday, 26 January, 26-01-2023, or any date format)"
                    return "When did you purchase this? (Examples: today, yesterday, 26 January, 26-01-2023, or any date format)"
                if slot == "email":
                    return "What is the email on your account? (e.g., name@example.com)"
                if slot == "issue_description":
                    # Intent-specific hints
                    hints = {
                        "billing": "e.g., refund due to wrong item, duplicate charge, payment failed",
                        "technical": "e.g., app crashes on checkout, cannot add to cart",
                        "account": "e.g., forgot password, cannot access account, change email",
                        "complaints": "e.g., package damaged, delivery late, wrong size",
                        "general": "e.g., need help with product details or availability",
                    }
                    hint = hints.get(intent, "")
                    return f"Could you describe the issue briefly? {hint}"

        # All required slots filled, create ticket
        if not state.ticket_created:
            state.ticket_created = True
            state.conversation_stage = "resolved"
            ticket_id = f"TKT-{int(time.time())}"
            
            if state.slots["order_id"] and state.slots["order_id"] != "N/A":
                prod = f" ({state.slots['product_name']})" if state.slots.get("product_name") else ""
                return f"Perfect! 🎫 I've created a support ticket (ID: {ticket_id}) for your {state.slots['issue_type']} issue with order {state.slots['order_id']}{prod}. Our team will get back to you within 2 hours."
            else:
                target = state.slots.get("product_name") or "your request"
                return f"Perfect! 🎫 I've created a support ticket (ID: {ticket_id}) for your {state.slots['issue_type']} issue regarding {target}. Our team will get back to you within 2 hours."
        
        return "I've already created a ticket for you. Our team will follow up shortly."

    def _debug_slots(self, state: ChatState) -> str:
        """Debug method to show current slot state."""
        filled = [f"{k}: {v}" for k, v in state.slots.items() if v]
        empty = [k for k, v in state.slots.items() if not v]
        return f"Filled: {filled}, Empty: {empty}, Ticket: {state.ticket_created}"

    def _reset_conversation(self, state: ChatState) -> None:
        """Reset conversation state for a new conversation."""
        state.slots = {
            "order_id": None,
            "purchase_date": None,
            "product_name": None,
            "issue_type": None,
            "issue_description": None,
            "additional_info": None,
            "email": None,
            "amount": None
        }
        state.ticket_created = False
        state.conversation_stage = "greeting"
        # Keep only the last few turns for context
        if len(state.history) > 4:
            state.history = state.history[-4:]

    # --- Main chat ---
    def chat(self, user_text: str, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        session_id = session_id or str(uuid.uuid4())
        state = self._get_state(session_id)
        self._append_history(state, "user", user_text)

        # Handle smalltalk and basic responses first
        smalltalk_response = self._handle_smalltalk(user_text)
        if smalltalk_response:
            self._append_history(state, "assistant", smalltalk_response)
            return {
                "session_id": session_id,
                "answer": smalltalk_response,
                "confidence": 1.0,
                "intents": [("general", 1.0)],
                "entities": {},
                "clarification": False,
                "sources": [],
                "latency_ms": int((time.time() - t0) * 1000),
            }
        
        # Check if user wants to start over
        if any(phrase in user_text.lower() for phrase in ["new issue", "start over", "reset", "another problem", "different issue"]):
            self._reset_conversation(state)
            self._append_history(state, "assistant", "Alright, let's start fresh! What can I help you with today?")
            return {
                "session_id": session_id,
                "answer": "Alright, let's start fresh! What can I help you with today?",
                "confidence": 1.0,
                "intents": [("general", 1.0)],
                "entities": {},
                "clarification": False,
                "sources": [],
                "latency_ms": int((time.time() - t0) * 1000),
            }

        # Part 1: intents + entities
        # Prefer embeddings-based classifier, fallback handled internally
        intents = self.intent_classifier.classify(user_text)
        entities = self.extractor.extract(user_text) if self.extractor else naive_entity_extract(user_text)
        
        # Fill conversation slots with extracted information
        self._fill_slots(state, entities, user_text)
        
        # Debug: Log what slots were filled (optional)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Slots after filling: %s", self._debug_slots(state))
        
        # Force debug output to see what's happening
        print(f"DEBUG: Slots after filling: {self._debug_slots(state)}")
        
        # Check if we need to ask for clarification or if we have enough info
        if not state.ticket_created:
            # Use intent-specific required slots
            intent = (state.slots.get("issue_type") or (intents[0][0] if intents else "general"))
            if intent == "billing":
                required = ["order_id", "product_name", "purchase_date", "issue_description"]
            elif intent == "technical":
                required = ["issue_description"]
            elif intent == "account":
                required = ["email", "issue_description"]
            elif intent == "complaints":
                required = ["order_id", "issue_description"]
            else:
                required = ["issue_description"]

            missing = [s for s in required if not state.slots.get(s)]
            if missing:
                # Enter collecting info stage
                state.conversation_stage = "collecting_info"
                answer_text = self._get_next_clarification(state, intents)
                confidence = 0.4
                ask_clarify = True
            else:
                # All required slots filled → create ticket
                answer_text = self._get_next_clarification(state, intents)
                confidence = 0.9
                ask_clarify = False
        else:
            # Ticket already created, provide helpful response
            answer_text = "I've already created a support ticket for you. Our team will get back to you within 2 hours. Is there anything else I can help you with?"
            confidence = 0.9
            ask_clarify = False
        
        # Only do RAG retrieval if we're not in clarification mode
        if not ask_clarify and not state.ticket_created:
            # Part 2: retrieval for detailed answers
            retrieved = self.hybrid.retrieve(user_text, top_k=TOP_K)
            # Optional rerank
            if self.reranker.enabled:
                retrieved = self.reranker.rerank(user_text, retrieved, top_k=TOP_K)
            
            # Build context chunks (cap to token budget)
            context_chunks: List[DocChunk] = []
            used = set()
            budget = MAX_CONTEXT_TOKENS
            for r in retrieved:
                if r.chunk.chunk_id in used:
                    continue
                tk = estimate_tokens(r.chunk.text)
                if tk > budget:
                    continue
                context_chunks.append(r.chunk)
                used.add(r.chunk.chunk_id)
                budget -= tk
                if budget <= 0:
                    break
            
            # Generate detailed response using RAG
            if context_chunks:
                prompt = build_prompt(user_text, context_chunks, state, entities)
                raw = self.llm.generate(prompt, max_tokens=512)
                if raw.strip():
                    answer_text = raw.strip()
                    confidence = min(0.95, max(ANSWER_MIN_CONFIDENCE, (retrieved[0].score if retrieved else 0.2)))
        else:
            # In clarification mode, no retrieval needed
            retrieved = []
            context_chunks = []

        self._append_history(state, "assistant", answer_text)

        t1 = time.time()
        duration_ms = int((t1 - t0) * 1000)

        # Analytics
        ANALYTICS.record(
            {
                "timestamp": utcnow_str(),
                "session_id": session_id,
                "user": user_text,
                "assistant": answer_text,
                "intents": intents,
                "entities": entities,
                "retrieval": [dataclasses.asdict(r) for r in retrieved],
                "timings": {"total_ms": duration_ms},
                "metadata": metadata or {},
            }
        )

        return {
            "session_id": session_id,
            "answer": answer_text,
            "confidence": round(confidence, 3),
            "intents": intents,
            "entities": entities,
            "clarification": ask_clarify,
            "sources": [
                {"filename": Path(r.chunk.source).name, "chunk_id": r.chunk.chunk_id, "score": round(r.score, 3)}
                for r in retrieved[:TOP_K]
            ],
            "latency_ms": duration_ms,
        }

    def feedback(self, session_id: str, satisfaction: str, comment: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "timestamp": utcnow_str(),
            "session_id": session_id,
            "feedback": {"satisfaction": satisfaction, "comment": comment},
        }
        ANALYTICS.record(payload)
        return {"ok": True}

    # Rebuild index CLI
    def rebuild_index(self, data_dir: Optional[str] = None):
        data_dir = data_dir or self.data_dir
        corpus = build_corpus(data_dir)
        if not corpus:
            logger.warning("No documents found to index in %s", data_dir)
        self.faiss = FAISSIndex()
        self.faiss.build(corpus, self.embedder)
        self.bm25 = BM25(corpus)
        self.hybrid = HybridRetriever(self.faiss, self.bm25, self.embedder, HYBRID_WEIGHTS)
        logger.info("Index rebuilt.")


# ------------------------------
# FastAPI server
# ------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.responses import HTMLResponse
    from fastapi.requests import Request
    from pydantic import BaseModel
    from uvicorn import run as uvicorn_run
except Exception:
    FastAPI = None  # type: ignore


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    metadata: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    session_id: str
    satisfaction: str  # "up" | "down"
    comment: Optional[str] = None


def create_app(orchestrator: ChatOrchestrator) -> Any:
    if FastAPI is None:
        raise RuntimeError("FastAPI not installed. Install fastapi and uvicorn.")

    app = FastAPI(title="RAG Support Bot", version="1.0")
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Setup templates
    templates = Jinja2Templates(directory="templates")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/health")
    def health():
        return {"status": "ok", "time": utcnow_str()}

    @app.get("/metrics")
    def metrics():
        return ANALYTICS.snapshot()

    @app.get("/api/analytics")
    async def api_analytics():
        try:
            interactions = ANALYTICS.interactions
            total = len(interactions)
            # Average response time in seconds
            times = [it.get("timings", {}).get("total_ms", 0) for it in interactions if it.get("timings")]
            avg_response_time = (sum(times) / len(times) / 1000.0) if times else 0.0
            # Total tickets (heuristic: assistant text mentions created ticket)
            def _is_ticket(it: dict) -> bool:
                a = (it.get("assistant") or "").lower()
                return ("support ticket created" in a) or ("i've created a support ticket" in a)
            total_tickets = sum(1 for it in interactions if _is_ticket(it))
            # Entity extraction rate
            entity_hits = sum(1 for it in interactions if it.get("entities") and len(it.get("entities")) > 0)
            entity_extraction_rate = (entity_hits / total) if total else 0.0
            # Intent distribution (top intent only)
            from collections import Counter
            c = Counter()
            for it in interactions:
                intents = it.get("intents") or []
                if intents and isinstance(intents, list) and intents[0] and isinstance(intents[0], (list, tuple)):
                    c[intents[0][0]] += 1
            return {
                "total_queries": total,
                "total_tickets": total_tickets,
                "avg_response_time": avg_response_time,
                "entity_extraction_rate": entity_extraction_rate,
                "intent_distribution": dict(c),
            }
        except Exception as e:
            logger.exception("/api/analytics failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/clear-chat")
    async def api_clear_chat(request: Request):
        try:
            body = await request.json()
            sid = body.get("conversation_id")
            if sid:
                # Remove the session entirely; next message starts fresh
                orchestrator.sessions.pop(sid, None)
            return {"ok": True}
        except Exception as e:
            logger.exception("/api/clear-chat failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/chat")
    def chat(req: ChatRequest):
        try:
            return orchestrator.chat(req.query, session_id=req.session_id, metadata=req.metadata)
        except Exception as e:
            logger.exception("/chat failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat")
    async def api_chat(request: Request):
        try:
            body = await request.json()
            message = body.get("message", "")
            conversation_id = body.get("conversation_id", "")
            
            # Call the orchestrator
            result = orchestrator.chat(message, session_id=conversation_id)
            
            # Format response to match frontend expectations
            return {
                "response": result["answer"],
                "metadata": {
                    "intent": result["intents"][0][0] if result["intents"] else "general",
                    "confidence": result["confidence"],
                    "entities": result["entities"],
                    "response_time": result["latency_ms"] / 1000.0,
                    "ticket_id": conversation_id,
                    "relevant_chunks": len(result["sources"])
                }
            }
        except Exception as e:
            logger.exception("/api/chat failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        try:
            return orchestrator.feedback(req.session_id, req.satisfaction, req.comment)
        except Exception as e:
            logger.exception("/feedback failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Expose ASGI app for Gunicorn/Render
orch = ChatOrchestrator(data_dir=DEFAULT_DATA_DIR)
app = create_app(orch)

# ------------------------------
# CLI Entrypoint
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG Customer Support Chatbot")
    parser.add_argument("--serve", action="store_true", help="Run FastAPI server")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS/BM25 index")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="Data directory for documents")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    orch = ChatOrchestrator(data_dir=args.data)

    if args.rebuild_index:
        orch.rebuild_index(args.data)

    if args.serve:
        if FastAPI is None:
            print("Install fastapi & uvicorn: pip install fastapi uvicorn pydantic")
            sys.exit(1)
        app = create_app(orch)
        uvicorn_run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
