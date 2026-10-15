"""The candidate pool: live search results, deduplicated and reranked locally.

Records from every search go into one pool. Versions of the same work are
merged with Phase 1's corpus dedupe (shared id, shared DOI, or the same long
title at most two years apart; the earliest version keeps its id and the others
become aliases). Near-duplicates with different titles are merged too, but
only if all three hold: title Jaccard >= 0.8 (on content stems, the
tokenisation BM25 and the leak filter use), at least one shared author, and
publication years at most 1 apart. Authors are the same if they share an
OpenAlex author id, or if every token of the shorter name (2 tokens or more)
is in the other name, since an author id can be missing. A check's final list
(rk.live.pipeline.merge_same_text) also merges versions by their text alone
(same_text): similar titles and abstracts that share most sentences.

The pool is then ranked like Phase 1's B3 over the frozen corpus, with the pool
as the corpus: BM25 (rk.retrieval tokenisation) and the dense encoder each rank
the pool's "title. abstract" texts, and RRF (k=60) fuses the two. Document
vectors are computed once per document and kept.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import bm25s
import faiss
import numpy as np

from rk.corpus import ORIGINS, dedupe, normalize_doi
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
from rk.targets import Record, id_number
from rk.text import content_stems, normalize_title, sentences, tokenize

NEAR_DUPLICATE_MIN_JACCARD = 0.8
NEAR_DUPLICATE_MAX_YEAR_GAP = 1
# Versions by their text alone (same_text), for a check's final list: chosen on the
# Step 5 dev checks, where a preprint and its journal version (A0053) rewrote every
# sentence a little, so no sentence was identical.
SAME_TEXT_MIN_TITLE_JACCARD = 0.7
SAME_TEXT_MIN_SENTENCE_JACCARD = 0.5
Author = Mapping[str, Any]  # {"id": OpenAlex author id or None, "name": display name}


def _jaccard(x: set[str], y: set[str]) -> float:
    return len(x & y) / len(x | y) if x | y else 0.0


def shared_sentences(a: str, b: str) -> float:
    """The share of the shorter abstract's sentences that the other has, reworded or not.

    A sentence is shared if a sentence of the other abstract has a word-set Jaccard of
    at least SAME_TEXT_MIN_SENTENCE_JACCARD with it (words: rk.text.tokenize).
    """
    first = [s for s in (set(tokenize(t)) for t in sentences(a)) if s]
    second = [s for s in (set(tokenize(t)) for t in sentences(b)) if s]
    if not first or not second:
        return 0.0
    short, other = (first, second) if len(first) <= len(second) else (second, first)
    found = sum(
        any(_jaccard(s, o) >= SAME_TEXT_MIN_SENTENCE_JACCARD for o in other) for s in short
    )
    return found / len(short)


def same_text(a: Record, b: Record) -> bool:
    """Whether two works are versions of one paper by their text, whoever the authors are.

    Their titles' content stems have a Jaccard of at least SAME_TEXT_MIN_TITLE_JACCARD,
    and their abstracts share most sentences: more than half of the shorter one's
    (shared_sentences).
    """
    return (
        title_jaccard(a["title"], b["title"]) >= SAME_TEXT_MIN_TITLE_JACCARD
        and shared_sentences(a["abstract"], b["abstract"]) > 0.5
    )


def title_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of two titles' content stems."""
    return _jaccard(set(content_stems(a)), set(content_stems(b)))


def name_tokens(name: str) -> frozenset[str]:
    """An author name's lower-case word tokens, accents removed, in any order."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return frozenset(re.findall(r"[a-z0-9]+", plain.casefold()))


def same_author(a: Author, b: Author) -> bool:
    """The same OpenAlex author id, or the shorter name's 2+ tokens all in the other name."""
    if a.get("id") and a.get("id") == b.get("id"):
        return True
    shorter, longer = sorted(
        (name_tokens(a.get("name") or ""), name_tokens(b.get("name") or "")), key=len
    )
    return len(shorter) >= 2 and shorter <= longer


def share_author(a: Iterable[Author], b: Iterable[Author]) -> bool:
    return any(same_author(x, y) for x, y in itertools.product(a, list(b)))


def _year(record: Record) -> int:
    return int(record["publication_date"][:4])


def near_duplicates(
    records: Sequence[Record], authors_of: Callable[[Record], list[Author]]
) -> list[tuple[str, str]]:
    """Pairs of record ids (earlier first) that are near-duplicates by all three rules."""
    stems = [set(content_stems(r["title"])) for r in records]
    years = [_year(r) for r in records]
    people = [authors_of(r) for r in records]
    pairs = []
    for i, j in itertools.combinations(range(len(records)), 2):
        if abs(years[i] - years[j]) > NEAR_DUPLICATE_MAX_YEAR_GAP:
            continue
        if _jaccard(stems[i], stems[j]) < NEAR_DUPLICATE_MIN_JACCARD:
            continue
        if share_author(people[i], people[j]):
            pairs.append((records[i]["id"], records[j]["id"]))
    return pairs


def merge_near_duplicates(
    records: Sequence[Record], authors_of: Callable[[Record], list[Author]]
) -> list[Record]:
    """Merge near-duplicate records the way rk.corpus.dedupe merges versions.

    Each pair is linked by giving one record the other's id as an alias, so
    dedupe merges them (chains included): the earliest version keeps its id.
    """
    links: dict[str, set[str]] = {}
    for a, b in near_duplicates(records, authors_of):
        links.setdefault(b, set()).add(a)
    if not links:
        return list(records)
    linked = [
        {**r, "aliases": sorted({*r["aliases"], *links.get(r["id"], ())}, key=id_number)}
        for r in records
    ]
    return dedupe(linked)


def match_original(versions: Iterable[Record], original: Record) -> str | None:
    """How a work is the original paper, checked over all its versions, or None.

    "id" (the OpenAlex id), "alias" (any id or alias of either side), "doi",
    or "title" (the exact normalised title).
    """
    versions = list(versions)
    original_ids = {original["id"], *original.get("aliases", [])}
    if any(v["id"] == original["id"] for v in versions):
        return "id"
    if any({v["id"], *v.get("aliases", [])} & original_ids for v in versions):
        return "alias"
    doi = normalize_doi(original.get("doi"))
    if doi and any(v.get("doi") == doi for v in versions):
        return "doi"
    title = normalize_title(original["title"])
    if title and any(normalize_title(v["title"]) == title for v in versions):
        return "title"
    return None


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
        self._by_id: dict[str, Record] = {}  # OpenAlex id -> its first raw record
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
            self._by_id.setdefault(record["id"], record)
            self._raw.append(record)
        # rk.corpus.dedupe ranks the corpus build's origins when versions merge. Live
        # records all share one origin, so any fixed corpus origin merges them the same.
        stand_in = [{**r, "origin": ORIGINS[0]} for r in self._raw]
        merged = merge_near_duplicates(dedupe(stand_in), self.authors)
        merged = [{**r, "origin": ORIGIN} for r in merged]  # in id order
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

    def versions(self, record: Record) -> list[Record]:
        """The search results merged into a work, one per OpenAlex id."""
        ids = (record["id"], *record["aliases"])
        return [self._by_id[w] for w in ids if w in self._by_id]

    def authors(self, record: Record) -> list[Author]:
        """The authors of every version of a work."""
        return [a for v in self.versions(record) for a in v.get("authors", [])]

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
