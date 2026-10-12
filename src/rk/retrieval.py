"""Indexes and retrieval configurations B1-B4 (spec: Retrieval configurations); B5 is in agent.py.

Documents are indexed as "title. abstract". BM25 (bm25s) gets the project's own
tokenisation (content_tokens + stem from text.py). Dense retrieval uses an
encoder behind the Encoder protocol (sentence-transformers here, FakeEncoder
in tests) with normalised float32 vectors in a FAISS IndexFlatIP. Every
ranking breaks ties by doc id, so results don't depend on library internals.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import bm25s
import faiss
import numpy as np

from rk.config import Config, RetrievalConfig
from rk.corpus import CORPUS_FILE, load_frozen_corpus
from rk.llm import ChatModel, Prompt, Usage, parse_json
from rk.targets import Record, id_number
from rk.text import content_tokens, stem

TOP_K = 100
RRF_K = 60

INDEX_MANIFEST = "manifest.json"
BM25_DIR = "bm25"
DENSE_FILE = "dense.faiss"
DOC_IDS_FILE = "doc_ids.json"


class RetrievalError(RuntimeError):
    """An index can't be built, loaded or used as asked."""


def analyze(text: str) -> list[str]:
    """BM25 tokens: stems of content tokens, the same tokenisation as the leak filter."""
    return [stem(t) for t in content_tokens(text)]


def doc_text(record: Record) -> str:
    return f"{record['title']}. {record['abstract']}"


# --- encoders ----------------------------------------------------------------


class Encoder(Protocol):
    """Maps text to L2-normalised float32 vectors, one row per text."""

    name: str
    revision: str  # model version (a Hugging Face commit hash)
    device: str

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """A sentence-transformers model at a pinned revision, on the GPU if torch sees CUDA."""

    def __init__(
        self, model_name: str, *, revision: str, query_prefix: str = "", batch_size: int = 64
    ):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # the dense extra is optional
            raise RetrievalError(
                'dense retrieval needs the dense extra: pip install -e ".[dense]"'
            ) from exc
        cuda = torch.cuda.is_available()
        self.name = model_name
        self.revision = revision
        self.device = f"cuda ({torch.cuda.get_device_name(0)})" if cuda else "cpu"
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            model_name, revision=revision, device="cuda" if cuda else "cpu"
        )

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts))

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def make_encoder(cfg: RetrievalConfig) -> Encoder:
    return SentenceTransformerEncoder(
        cfg.encoder, revision=cfg.encoder_revision, query_prefix=cfg.query_prefix
    )


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """float32 copy with every non-zero row scaled to length 1 (inner product = cosine)."""
    vectors = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors


# --- indexes -----------------------------------------------------------------


@dataclass
class Index:
    doc_ids: list[str]  # position i in both indexes is doc_ids[i], in id order
    titles: dict[str, str]
    bm25: bm25s.BM25
    dense: faiss.Index
    manifest: dict[str, Any]


def build_index(cfg: Config, encoder: Encoder) -> dict[str, Any]:
    """Build BM25 and dense indexes for the frozen corpus and save them to retrieval.index_dir."""
    out_dir = cfg.retrieval.index_dir
    if (out_dir / INDEX_MANIFEST).exists():
        raise RetrievalError(
            f"{out_dir / INDEX_MANIFEST} already exists: refusing to overwrite an index "
            "(delete the folder to rebuild it)"
        )
    corpus, _, corpus_manifest = load_frozen_corpus(cfg.corpus.out_dir)
    records = sorted(corpus, key=lambda r: id_number(r["id"]))
    texts = [doc_text(r) for r in records]

    start = time.perf_counter()
    bm25 = bm25s.BM25()
    bm25.index([analyze(t) for t in texts], show_progress=False)
    vectors = _unit_rows(encoder.encode_docs(texts))
    if vectors.shape[0] != len(texts):
        raise RetrievalError(f"encoder returned {vectors.shape[0]} vectors for {len(texts)} docs")
    dense = faiss.IndexFlatIP(vectors.shape[1])
    dense.add(vectors)
    seconds = time.perf_counter() - start

    out_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(str(out_dir / BM25_DIR), show_progress=False)
    faiss.write_index(dense, str(out_dir / DENSE_FILE))
    doc_ids = [r["id"] for r in records]
    (out_dir / DOC_IDS_FILE).write_text(json.dumps(doc_ids), encoding="utf-8")
    manifest = {
        "corpus_sha256": corpus_manifest["files"][CORPUS_FILE]["sha256"],
        "encoder": encoder.name,
        "encoder_revision": encoder.revision,
        "device": encoder.device,
        "dim": int(vectors.shape[1]),
        "n_docs": len(doc_ids),
        "build_seconds": round(seconds, 3),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "doc_text": "title. abstract",
        "tokenizer": "rk.text content_tokens + stem",
        "bm25": {"method": bm25.method, "k1": bm25.k1, "b": bm25.b},
    }
    with (out_dir / INDEX_MANIFEST).open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


def load_index(cfg: Config) -> Index:
    """Load the saved indexes; refuse ones built for another corpus than the frozen one."""
    out_dir = cfg.retrieval.index_dir
    manifest_path = out_dir / INDEX_MANIFEST
    if not manifest_path.exists():
        raise RetrievalError(f"no index in {out_dir}: run `python -m rk build-index` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus, _, corpus_manifest = load_frozen_corpus(cfg.corpus.out_dir)
    corpus_sha = corpus_manifest["files"][CORPUS_FILE]["sha256"]
    if manifest["corpus_sha256"] != corpus_sha:
        raise RetrievalError(
            f"the index in {out_dir} was built for corpus sha256 {manifest['corpus_sha256']}, "
            f"but the frozen corpus is {corpus_sha}: rebuild the index"
        )
    doc_ids = json.loads((out_dir / DOC_IDS_FILE).read_text(encoding="utf-8"))
    return Index(
        doc_ids=doc_ids,
        titles={r["id"]: r["title"] for r in corpus},
        bm25=bm25s.BM25.load(str(out_dir / BM25_DIR), show_progress=False),
        dense=faiss.read_index(str(out_dir / DENSE_FILE)),
        manifest=manifest,
    )


# --- rankings and fusion -----------------------------------------------------


def bm25_ranking(index: Index, query: str, k: int = TOP_K) -> list[str]:
    """Docs with a positive BM25 score, best first, ties by doc id."""
    tokens = analyze(query)
    if not tokens:  # bm25s can't score an empty query
        return []
    scores = index.bm25.get_scores(tokens)
    order = np.argsort(-scores, kind="stable")[:k]  # stable: ties keep id order
    return [index.doc_ids[i] for i in order if scores[i] > 0]


def dense_ranking(index: Index, encoder: Encoder, query: str, k: int = TOP_K) -> list[str]:
    """Nearest docs by inner product of unit vectors, ties by doc id."""
    vector = _unit_rows(encoder.encode_queries([query]))
    scores, positions = index.dense.search(vector, min(k, index.dense.ntotal))
    hits = [(float(s), int(p)) for s, p in zip(scores[0], positions[0]) if p >= 0]
    hits.sort(key=lambda h: (-h[0], h[1]))  # positions are in id order
    return [index.doc_ids[p] for _, p in hits]


def rrf(rankings: Sequence[Sequence[str]], *, k: int = RRF_K, top_k: int = TOP_K) -> list[str]:
    """Reciprocal rank fusion: score = sum of 1 / (k + rank), rank from 1; ties by doc id.

    Scores are summed with math.fsum, which is exact up to one final rounding,
    so equal sums compare equal whatever order the lists come in.
    """
    terms: dict[str, list[float]] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            terms.setdefault(doc_id, []).append(1.0 / (k + rank))
    scores = {doc_id: math.fsum(t) for doc_id, t in terms.items()}
    return sorted(scores, key=lambda d: (-scores[d], id_number(d)))[:top_k]


def fuse_runs(runs: Sequence[Sequence[str]]) -> list[str]:
    """The final ranking of B4, B5 and B5-U: RRF over the B3 list of every query run."""
    return rrf(runs)


def normalize_query(query: str) -> str:
    """Key for spotting repeated queries: whitespace collapsed, case folded."""
    return " ".join(query.split()).casefold()


# --- retrievers --------------------------------------------------------------


@dataclass
class Trace:
    """What one retrieval did and cost."""

    queries: list[str] = field(default_factory=list)  # queries actually run, in order
    budget_used: int = 0  # queries counted against max_queries (repeats count)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)  # agent decisions (B5)
    stop_reason: str | None = None  # why an agent loop ended (B5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "budget_used": self.budget_used,
            "llm_calls": self.usage.calls,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "method_cost": self.usage.method_cost,
            "money_spent": self.usage.money_spent,
            "latency_s": self.latency_s,
            "errors": self.errors,
            "prompt_hashes": self.prompt_hashes,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class Result:
    doc_ids: list[str]  # ranked, at most TOP_K
    trace: Trace


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str) -> Result: ...


class _SingleQuery:
    """B1-B3: one query in, one ranking out, no LLM."""

    name = ""

    def rank(self, query: str) -> list[str]:
        raise NotImplementedError

    def retrieve(self, query: str) -> Result:
        start = time.perf_counter()
        ranking = self.rank(query)
        trace = Trace(queries=[query], budget_used=1, latency_s=time.perf_counter() - start)
        return Result(ranking, trace)


class BM25Retriever(_SingleQuery):
    name = "B1"

    def __init__(self, index: Index) -> None:
        self.index = index

    def rank(self, query: str) -> list[str]:
        return bm25_ranking(self.index, query)


class DenseRetriever(_SingleQuery):
    name = "B2"

    def __init__(self, index: Index, encoder: Encoder) -> None:
        built = (index.manifest["encoder"], index.manifest.get("encoder_revision"))
        if (encoder.name, encoder.revision) != built:
            raise RetrievalError(
                f"the index was built with encoder {built[0]!r} at revision {built[1]!r}, "
                f"not {encoder.name!r} at {encoder.revision!r}: rebuild it or change "
                "retrieval.encoder / encoder_revision"
            )
        self.index = index
        self.encoder = encoder

    def rank(self, query: str) -> list[str]:
        return dense_ranking(self.index, self.encoder, query)


class HybridRetriever(_SingleQuery):
    """B3: RRF (k=60) over the B1 and B2 top-100."""

    name = "B3"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever) -> None:
        self.bm25 = bm25
        self.dense = dense

    def rank(self, query: str) -> list[str]:
        return rrf([self.bm25.rank(query), self.dense.rank(query)])


def parse_rewrites(text: str, *, original: str, limit: int) -> list[str]:
    """Up to `limit` distinct rewrites from {"queries": [...]}.

    Raises ValueError if there is no JSON, TypeError if it has the wrong shape.
    """
    data = parse_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise TypeError('expected {"queries": [...]}')
    seen = {normalize_query(original)}
    rewrites = []
    for item in data["queries"]:
        if not isinstance(item, str):
            continue
        query = " ".join(item.split())
        if query and normalize_query(query) not in seen:
            seen.add(normalize_query(query))
            rewrites.append(query)
    return rewrites[:limit]


class RewriteRetriever:
    """B4: one agent call proposes up to max_queries - 1 rewrites; B3 on each; fused."""

    name = "B4"

    def __init__(
        self, hybrid: HybridRetriever, agent: ChatModel, *, max_queries: int, prompt: Prompt
    ) -> None:
        self.hybrid = hybrid
        self.agent = agent
        self.max_queries = max_queries
        self.prompt = prompt

    def retrieve(self, query: str) -> Result:
        start = time.perf_counter()
        trace = Trace(prompt_hashes={self.prompt.name: self.prompt.sha256})
        rewrites: list[str] = []
        n_rewrites = self.max_queries - 1
        if n_rewrites > 0:
            text = self.prompt.render(proposal=query, n_queries=str(n_rewrites))
            reply = self.agent.complete([{"role": "user", "content": text}])
            trace.usage += reply.usage
            try:
                rewrites = parse_rewrites(reply.text, original=query, limit=n_rewrites)
            except (ValueError, TypeError) as exc:  # fall back to the original query alone
                trace.errors.append(f"rewrite_parse_error: {exc}; reply: {reply.text[:200]!r}")
        trace.queries = [query, *rewrites]
        trace.budget_used = len(trace.queries)
        ranking = fuse_runs([self.hybrid.rank(q) for q in trace.queries])
        trace.latency_s = time.perf_counter() - start
        return Result(ranking, trace)
