"""The candidate pool: live search results, deduplicated and reranked locally.

Records from every search go into one pool. Versions of the same work are
merged with Phase 1's corpus dedupe (shared id, shared DOI, or the same long
title at most two years apart; the earliest version keeps its id and the others
become aliases). The pool is then ranked like Phase 1's B3 over the frozen
corpus, with the pool as the corpus: BM25 (rk.retrieval tokenisation) and the
dense encoder each rank the pool's "title. abstract" texts, and RRF (k=60)
fuses the two. Document vectors are computed once per document and kept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import bm25s
import faiss
import numpy as np

from rk.corpus import ORIGINS, dedupe
from rk.live.openalex_search import ORIGIN
from rk.retrieval import (
    BM25Retriever,
    DenseRetriever,
    Encoder,
    HybridRetriever,
    Index,
    _unit_rows,
    analyze,
    doc_text,
    encoder_settings,
    fuse_runs,
)
from rk.targets import Record


@dataclass(frozen=True)
class Hit:
    """Where a search found a work: the query's position, the search kind, the rank (from 1)."""

    query_index: int
    kind: str
    rank: int


class CandidatePool:
    """Deduplicated search results with a hybrid ranker over them.

    At most `max_size` distinct works are kept, in order of first arrival;
    later ones are counted in `dropped`. `titles` maps every kept work's id to
    its title and is updated in place, so an agent holding it sees new results.
    """

    def __init__(self, encoder: Encoder, *, max_size: int) -> None:
        self.encoder = encoder
        self.max_size = max_size
        self.titles: dict[str, str] = {}
        self.dropped = 0
        self._raw: list[Record] = []  # every usable result, in arrival order
        self._hits: dict[str, list[Hit]] = {}  # OpenAlex id -> where it was found
        self._records: list[Record] = []  # kept works, merged, in id order
        self._vectors: dict[str, np.ndarray] = {}  # doc_text -> unit vector
        self._ranker: HybridRetriever | None = None

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[Record]:
        return list(self._records)

    def add(self, records: Sequence[Record], *, query_index: int, kind: str) -> None:
        """Add one search's records (in its result order) and merge versions."""
        for rank, record in enumerate(records, start=1):
            self._hits.setdefault(record["id"], []).append(Hit(query_index, kind, rank))
            self._raw.append(record)
        # rk.corpus.dedupe ranks the corpus build's origins when versions merge. Live
        # records all share one origin, so any fixed corpus origin merges them the same.
        stand_in = [{**r, "origin": ORIGINS[0]} for r in self._raw]
        merged = [{**r, "origin": ORIGIN} for r in dedupe(stand_in)]  # in id order
        arrival: dict[str, int] = {}  # first position in _raw of every OpenAlex id
        for position, record in enumerate(self._raw):
            arrival.setdefault(record["id"], position)

        def first_seen(record: Record) -> int:
            return min(arrival[w] for w in (record["id"], *record["aliases"]))

        kept = {r["id"] for r in sorted(merged, key=first_seen)[: self.max_size]}
        self._records = [r for r in merged if r["id"] in kept]
        self.dropped = len(merged) - len(self._records)
        self.titles.clear()
        self.titles.update({r["id"]: r["title"] for r in self._records})
        self._ranker = None

    def hits(self, record: Record) -> list[Hit]:
        """Every search hit of a work, over all its merged versions."""
        found = [h for w in (record["id"], *record["aliases"]) for h in self._hits.get(w, [])]
        return sorted(found, key=lambda h: (h.query_index, h.kind, h.rank))

    def rank(self, query: str) -> list[str]:
        """The pool ranked for `query` as B3 ranks the frozen corpus (top 100)."""
        if not self._records:
            return []
        if self._ranker is None:
            index = self._index()
            bm25, dense = BM25Retriever(index), DenseRetriever(index, self.encoder)
            self._ranker = HybridRetriever(bm25, dense)
        return self._ranker.rank(query)

    def fused(self, queries: Sequence[str]) -> list[str]:
        """B5's final ranking over the pool: RRF over every query's hybrid ranking."""
        return fuse_runs([self.rank(q) for q in queries])

    def _index(self) -> Index:
        texts = [doc_text(r) for r in self._records]
        new = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if new:
            vectors = self.encoder.encode_docs(new)
            for text, vector in zip(new, _unit_rows(vectors)):
                self._vectors[text] = vector
        vectors = np.stack([self._vectors[t] for t in texts])
        dense = faiss.IndexFlatIP(vectors.shape[1])
        dense.add(vectors)
        bm25 = bm25s.BM25()
        bm25.index([analyze(t) for t in texts], show_progress=False)
        return Index(
            doc_ids=[r["id"] for r in self._records],
            titles=dict(self.titles),
            bm25=bm25,
            dense=dense,
            manifest=encoder_settings(self.encoder),  # DenseRetriever checks these
        )
