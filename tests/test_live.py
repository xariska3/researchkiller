"""Phase 2 live retrieval: OpenAlex search client, candidate pool, B5 on the live pool.

OpenAlex is FakeLiveOpenAlex behind httpx.MockTransport; only the test marked
`network` talks to the real API (run it with -m network).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from bench_data import ABSTRACTS, GOOD, T1, T2, T3, TITLES, corpus_records, target_rows
from conftest import FakeEncoder, FakeLLM, corpus_config_dict
from live_fakes import SEARCH_MAX_CHARS, WORKS, FakeLiveOpenAlex, oa_work

from rk.cli import load_env_file
from rk.config import Config, ConfigError, LiveConfig, config_from_dict, load_config
from rk.corpus import OpenAlexBudgetExhausted, freeze
from rk.live.openalex_search import (
    KEYWORD,
    SELECT,
    SEMANTIC,
    Budget,
    LiveSearchClient,
    OpenAlexCache,
    live_record,
    search_text,
)
from rk.live.pipeline import CheckCostCapExceeded, CheckLedger, LiveRanker, live_search
from rk.live.pool import CandidatePool, Hit, match_original, same_author, title_jaccard
from rk.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    build_index,
    fuse_runs,
    load_index,
)
from rk.text import sentences

LIVE = {
    "cache_path": "openalex.sqlite",
    "results_per_query": 50,
    "semantic_search": True,
    "semantic_results_per_query": 50,
    "keyword_max_words": 32,
    "max_pool_size": 500,
    "top_k": 5,
    "explain_top_k": 3,
    "cost_cap_per_check": 0.10,
    "openalex_timeout_s": 30,
}


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    data = corpus_config_dict(tmp_path / "corpus")
    data["retrieval"]["index_dir"] = str(tmp_path / "index")
    data["live"] = {**LIVE, "cache_path": str(tmp_path / "openalex.sqlite")}
    return config_from_dict(data)


def _client(
    cfg: Config, fake: FakeLiveOpenAlex, sleeps: list[float] | None = None
) -> LiveSearchClient:
    assert cfg.live is not None
    sleeps = [] if sleeps is None else sleeps
    return LiveSearchClient(
        cfg.live,
        OpenAlexCache(cfg.live.cache_path),
        transport=fake.transport,
        sleep=sleeps.append,
        clock=lambda: 0.0,
    )


def _search(query: str) -> str:
    return json.dumps({"action": "search", "query": query})


STOP = json.dumps({"action": "stop", "reason": "covered"})
TYPES = (
    "type:article|preprint|conference-paper|book-chapter|dissertation|review|book|report"
    "|software-paper"
)


# --- config ------------------------------------------------------------------


def test_the_project_config_has_a_live_section() -> None:
    live = load_config(Path("configs/phase1.yaml")).live
    assert live is not None
    assert (live.results_per_query, live.semantic_results_per_query, live.top_k) == (50, 50, 20)
    assert (live.explain_top_k, live.cost_cap_per_check, live.keyword_max_words) == (10, 0.10, 32)
    assert live.work_types == (
        "article",
        "preprint",
        "conference-paper",
        "book-chapter",
        "dissertation",
        "review",
        "book",
        "report",
        "software-paper",
    )
    assert live.work_types == LiveConfig(cache_path=Path("x")).work_types  # the default
    assert live.cache_ttl_days == 7
    assert live.max_search_chars == LiveConfig(cache_path=Path("x")).max_search_chars == 1500


def test_the_live_section_is_optional(tmp_path: Path) -> None:
    assert config_from_dict(corpus_config_dict(tmp_path)).live is None


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("results_per_query", 101, "results_per_query"),
        ("semantic_results_per_query", 51, "semantic_results_per_query"),
        ("explain_top_k", 6, "explain_top_k"),
        ("cost_cap_per_check", 0, "cost_cap_per_check"),
        ("keyword_max_words", 0, "keyword_max_words"),
        ("max_search_chars", 0, "max_search_chars"),
        ("work_types", [], "work_types"),
        ("work_types", ["article", " dataset"], "work_types"),
        ("cache_ttl_days", 0, "cache_ttl_days"),
    ],
)
def test_live_settings_are_checked(tmp_path: Path, key: str, value: Any, message: str) -> None:
    data = corpus_config_dict(tmp_path)
    data["live"] = {**LIVE, key: value}
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


# --- OpenAlex search client --------------------------------------------------


def test_keyword_search_sends_the_documented_parameters(cfg: Config) -> None:
    fake = FakeLiveOpenAlex()
    response = _client(cfg, fake).search("summarisation clusters memory", KEYWORD)
    assert fake.requests == [
        {
            "search": "summarisation clusters memory",
            "filter": f"has_abstract:true,{TYPES}",
            "per_page": "50",
            "select": SELECT,
        }
    ]
    assert "authorships" in SELECT.split(",") and "type" in SELECT.split(",")
    assert [r["id"] for r in response.records] == [T1]
    record = response.records[0]
    assert (record["title"], record["abstract"], record["origin"]) == (
        TITLES[T1],
        ABSTRACTS[T1],
        "live",
    )
    assert (response.kind, response.n_results, response.cost_usd, response.cached) == (
        KEYWORD,
        1,
        0.001,
        False,
    )


def test_no_search_sends_more_than_max_search_chars(cfg: Config) -> None:
    assert cfg.live is not None and cfg.live.max_search_chars == SEARCH_MAX_CHARS
    fake = FakeLiveOpenAlex()  # rejects a longer search with HTTP 400, as OpenAlex does
    text = "memory " * 600  # 4,200 characters, no sentence end
    client = _client(cfg, fake)
    response = client.search(text, SEMANTIC)
    client.search(text, KEYWORD)
    semantic, keyword = fake.requests
    assert semantic["search.semantic"] == keyword["search"] == search_text(text, SEARCH_MAX_CHARS)
    assert len(semantic["search.semantic"]) == 1497  # 214 whole words, cut at a space
    assert (semantic["per_page"], semantic["filter"]) == ("50", f"has_abstract:true,{TYPES}")
    assert response.query == text  # the result keeps the full query
    assert response.records  # memory appears in several abstracts
    client.search(text, SEMANTIC)  # the cut query is the cache key
    assert len(fake.requests) == 2


def test_a_long_query_is_cut_at_its_last_sentence_that_fits() -> None:
    text = "We extend BERT (Devlin et al. 2019) to long inputs. It works. A third one runs on"
    first, second = sentences(text)[:2]
    assert search_text(text, len(text)) == text  # it fits: unchanged
    assert search_text(text, len(first) + len(second)) == first  # the second doesn't fit
    assert search_text(text, len(first) + 1 + len(second)) == f"{first} {second}"
    assert search_text(text, len(text) - 1) == f"{first} {second}"
    # No sentence fits: the last word boundary that does ("et al." ends no sentence), or
    # mid-word for one long word.
    assert search_text(text, 35) == "We extend BERT (Devlin et al. 2019)"  # a space at 35
    assert search_text(text, 33) == "We extend BERT (Devlin et al."  # inside "2019)"
    assert search_text("x" * 20, 5) == "xxxxx"


def test_published_before_filters_on_the_day_before(cfg: Config) -> None:
    old = oa_work(corpus_records()[0], published="2024-12-31")
    new = oa_work(corpus_records()[1], published="2025-01-01")
    fake = FakeLiveOpenAlex([old, new])
    response = _client(cfg, fake).search("memory", SEMANTIC, before=date(2025, 1, 1))
    assert fake.requests[0]["filter"] == (
        f"has_abstract:true,{TYPES},to_publication_date:2024-12-31"
    )
    assert [r["id"] for r in response.records] == [T1]


def test_every_response_is_cached_on_disk(cfg: Config) -> None:
    fake = FakeLiveOpenAlex()
    first = _client(cfg, fake).search("memory", SEMANTIC)
    again = _client(cfg, fake).search("memory", SEMANTIC)  # a new client on the same file
    assert len(fake.requests) == 1
    assert again.records == first.records
    assert (again.cached, again.cost_usd, again.money_spent) == (True, 0.001, 0.0)
    assert (first.cached, first.money_spent) == (False, 0.001)
    _client(cfg, fake).search("memory", KEYWORD)  # another kind is another request
    assert len(fake.requests) == 2


def test_semantic_search_is_throttled_to_one_call_per_second(cfg: Config) -> None:
    sleeps: list[float] = []
    client = _client(cfg, FakeLiveOpenAlex(), sleeps)
    client.search("memory", SEMANTIC)
    client.search("accuracy", SEMANTIC)
    client.search("memory", KEYWORD)
    client.search("accuracy", KEYWORD)
    assert sleeps == [1.0, pytest.approx(0.1)]


def test_rate_limited_calls_are_retried(cfg: Config) -> None:
    sleeps: list[float] = []
    fake = FakeLiveOpenAlex(fail_with=[429])
    response = _client(cfg, fake, sleeps).search("memory", SEMANTIC)
    assert len(fake.requests) == 2 and response.records
    assert sleeps == [1.0, 1.0]  # the backoff, then the one-call-per-second throttle


def test_a_429_the_budget_cant_pay_for_stops_at_once(cfg: Config) -> None:
    fake = FakeLiveOpenAlex(budget_credits=5)  # a search costs 10 credits
    with pytest.raises(OpenAlexBudgetExhausted, match="5 credits left") as caught:
        _client(cfg, fake).search("memory", SEMANTIC)
    assert caught.value.resets_in_s == 3600
    assert len(fake.requests) == 1  # not retried


def test_a_429_with_budget_left_is_a_per_second_limit_and_is_retried(cfg: Config) -> None:
    assert cfg.live is not None
    fake = FakeLiveOpenAlex(budget_credits=1000)
    limited: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if not limited:
            limited.append(request)
            return httpx.Response(429, headers={"x-ratelimit-remaining": "1000"})
        return fake.handle(request)

    client = LiveSearchClient(
        cfg.live,
        OpenAlexCache(cfg.live.cache_path),
        transport=httpx.MockTransport(handle),
        sleep=lambda seconds: None,
        clock=lambda: 0.0,
    )
    assert client.search("memory", SEMANTIC).records
    assert len(limited) == 1 and len(fake.requests) == 1


def test_the_budget_comes_from_the_free_rate_limit_endpoint(cfg: Config) -> None:
    fake = FakeLiveOpenAlex(budget_credits=250)
    assert _client(cfg, fake).budget() == Budget(0.025, 1.0, 3600.0)
    assert (fake.rate_limit_calls, fake.requests) == (1, [])


def test_cost_falls_back_to_the_documented_price(cfg: Config) -> None:
    response = _client(cfg, FakeLiveOpenAlex(cost=None)).search("memory", SEMANTIC)
    assert response.cost_usd == 0.001


def test_results_without_an_abstract_or_a_date_are_dropped() -> None:
    work = oa_work(corpus_records()[0])
    assert live_record(work) is not None
    assert live_record({**work, "abstract_inverted_index": None}) is None
    assert live_record({**work, "publication_date": None}) is None


def _work(
    num: str,
    title: str,
    published: str,
    *,
    work_type: str = "article",
    doi: str | None = None,
    authors: Sequence[tuple[str | None, str]] = (),
) -> dict[str, Any]:
    """A real OpenAlex work's metadata (as looked up on 2026-10-13), with a stand-in abstract."""
    return oa_work(
        {"id": num, "title": title, "abstract": ABSTRACTS[T2]},
        published=published,
        type=work_type,
        doi=f"https://doi.org/{doi}" if doi else None,
        authorships=[
            {"author": {"id": f"https://openalex.org/{a}" if a else None, "display_name": n}}
            for a, n in authors
        ],
    )


# The A0002 smoke-test pair: a conference paper and its author's dissertation. The
# dissertation's only author has no OpenAlex id and is written "Last, First".
BB25_PAPER = _work(
    "W4388676224",
    "BB25HLegalSum: Leveraging BM25 and BERT-based clustering for the summarization of legal "
    "documents",
    "2023-01-01",
    work_type="conference-paper",
    doi="10.26615/978-954-452-092-2_029",
    authors=[("A5093262736", "Leonardo Bonalume"), ("A5074093456", "Karin E. Becker")],
)
BB25_THESIS = _work(
    "W7045572899",
    "BB25HLegalSum : a method for legal document summarization that leverages BM25 and "
    "BERT-based clustering",
    "2024-01-01",
    work_type="dissertation",
    authors=[(None, "Andrade, Leonardo Bonalume de")],
)
# The A0004 smoke-test pair: the paper, and a Zenodo dataset of its replication data.
TEXTUAL_OVERLAP = (
    "Textual overlap rather than domain alignment: A comparative study of fine-tuning "
    "strategies for specialised machine translation with large language models"
)
A0004_PAPER = _work(
    "W7169768705",
    TEXTUAL_OVERLAP,
    "2026-07-20",
    doi="10.1371/journal.pone.0352256",
    authors=[(None, "Lixue Yang"), ("A5101518774", "Jiaxin Zhu"), ("A5049423308", "Z Zhang")],
)
REPLICATION_DATASET = _work(
    "W7165207049",
    f'Minimal replication dataset for "{TEXTUAL_OVERLAP}"',
    "2026-06-18",
    work_type="dataset",
    doi="10.5281/zenodo.20743017",
    authors=[("A5102360992", "Lixue Yang"), ("A5101518774", "Jiaxin Zhu")],
)


def test_only_papers_are_searched_and_kept(cfg: Config) -> None:
    assert cfg.live is not None
    types = cfg.live.work_types
    assert live_record(REPLICATION_DATASET, types) is None
    assert live_record(REPLICATION_DATASET) is not None  # without a type list, kept
    for work in (A0004_PAPER, BB25_PAPER, BB25_THESIS):
        record = live_record(work, types)
        assert record is not None and record["type"] == work["type"]
    # Literature reviews, books, reports and software papers are prior art too.
    for kept in ("review", "book", "report", "software-paper"):
        record = live_record({**A0004_PAPER, "type": kept}, types)
        assert record is not None and record["type"] == kept
    for dropped in ("erratum", "peer-review", "paratext", "letter", "editorial", "other"):
        assert live_record({**A0004_PAPER, "type": dropped}, types) is None
    # The type: filter asks OpenAlex for papers only; anything else that comes back
    # is dropped as well, but still counted among the results.
    fake = FakeLiveOpenAlex([A0004_PAPER, REPLICATION_DATASET], honour_types=False)
    response = _client(cfg, fake).search("textual overlap domain alignment", SEMANTIC)
    assert TYPES in fake.requests[0]["filter"]
    assert ([r["id"] for r in response.records], response.n_results) == (["W7169768705"], 2)


def test_records_keep_their_authors() -> None:
    record = live_record(BB25_THESIS)
    assert record is not None
    assert record["authors"] == [{"id": None, "name": "Andrade, Leonardo Bonalume de"}]
    paper = live_record(BB25_PAPER)
    assert paper is not None and paper["authors"][0] == {
        "id": "A5093262736",
        "name": "Leonardo Bonalume",
    }


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def test_live_cache_expires_after_its_ttl_but_evaluation_never_expires(cfg: Config) -> None:
    assert cfg.live is not None
    clock = _Clock()
    cache = OpenAlexCache(cfg.live.cache_path, now=clock)
    same_text = [oa_work({**corpus_records()[0], "id": f"W{n}"}) for n in (901, 902, 903)]
    fake = FakeLiveOpenAlex(same_text[:2])
    ttl = timedelta(days=7)

    def search(max_age: timedelta | None) -> Any:
        client = LiveSearchClient(
            cfg.live, cache, max_age=max_age, transport=fake.transport, sleep=lambda s: None
        )
        return client.search("memory", SEMANTIC)

    first = search(ttl)
    clock.now += timedelta(days=6)
    assert search(ttl).cached and len(fake.requests) == 1  # 6 days old: still fresh
    fake.works = same_text  # OpenAlex has changed since
    clock.now += timedelta(days=2)
    fresh = search(ttl)  # 8 days old: fetched again, and both responses are kept
    assert (fresh.cached, len(fake.requests), len(cache)) == (False, 2, 2)
    assert fresh.fetched_at == clock.now
    assert len(fresh.records) == 3 and len(first.records) == 2
    replay = search(None)  # an evaluation run reads the first response ever cached
    assert replay.cached and replay.records == first.records
    assert replay.fetched_at == first.fetched_at
    assert len(fake.requests) == 2
    clock.now += timedelta(days=365)
    assert search(None).records == first.records  # and it never expires


def test_from_config_sets_the_ttl_for_the_live_tool_only(cfg: Config) -> None:
    assert cfg.live is not None
    cache = OpenAlexCache(cfg.live.cache_path)
    assert LiveSearchClient.from_config(cfg, cache).max_age == timedelta(days=7)
    assert LiveSearchClient.from_config(cfg, cache, evaluation=True).max_age is None


# --- candidate pool ----------------------------------------------------------


def test_pool_merges_versions_and_keeps_every_hit() -> None:
    preprint = live_record(oa_work(corpus_records()[0], published="2024-05-01", doi="10.1/x"))
    published = live_record(
        oa_work({**corpus_records()[0], "id": "W900"}, published="2025-02-01", doi="10.1/X")
    )
    other = live_record(oa_work(corpus_records()[1]))
    assert preprint and published and other
    pool = CandidatePool(FakeEncoder(), max_size=10)
    pool.add([preprint, other], query_index=0, kind=SEMANTIC)
    pool.add([published], query_index=1, kind=KEYWORD)
    assert len(pool) == 2
    merged = next(r for r in pool.records if r["id"] == T1)  # the earliest version keeps its id
    assert merged["aliases"] == ["W900"]
    assert pool.hits(merged) == [Hit(0, SEMANTIC, 1), Hit(1, KEYWORD, 1)]
    assert pool.titles == {T1: TITLES[T1], T2: TITLES[T2]}


def _pool_of(*works: dict[str, Any]) -> CandidatePool:
    pool = CandidatePool(FakeEncoder(), max_size=50)
    records = [r for w in works if (r := live_record(w)) is not None]
    assert len(records) == len(works)
    pool.add(records, query_index=0, kind=SEMANTIC)
    return pool


def test_near_duplicates_with_different_titles_merge_under_all_three_rules() -> None:
    # BB25HLegalSum: title Jaccard 0.9 on content stems, one year apart, and the
    # same author, matched by name since the dissertation's author has no id.
    assert title_jaccard(BB25_PAPER["title"], BB25_THESIS["title"]) == pytest.approx(0.9)
    pool = _pool_of(BB25_PAPER, BB25_THESIS)
    (merged,) = pool.records
    assert (merged["id"], merged["aliases"]) == ("W4388676224", ["W7045572899"])
    assert merged["type"] == "conference-paper"  # the earlier version's fields
    assert [a["name"] for a in pool.authors(merged)] == [
        "Leonardo Bonalume",
        "Karin E. Becker",
        "Andrade, Leonardo Bonalume de",
    ]


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"publication_date": "2025-01-01"}, "two years apart"),
        ({"authorships": [{"author": {"id": None, "display_name": "Maria Rossi"}}]}, "no author"),
        ({"title": "BB25HLegalSum: legal summarization with BM25"}, "Jaccard 0.44"),
    ],
)
def test_near_duplicates_need_all_three_rules(change: dict[str, Any], why: str) -> None:
    pool = _pool_of(BB25_PAPER, {**BB25_THESIS, **change})
    assert len(pool) == 2, why


def test_same_author_by_id_or_by_every_token_of_the_shorter_name() -> None:
    def author(name: str, author_id: str | None = None) -> dict[str, str | None]:
        return {"id": author_id, "name": name}

    assert same_author(author("L. Bonalume", "A1"), author("Someone else", "A1"))
    assert same_author(author("Leonardo Bonalume"), author("Andrade, Leonardo Bonalume de"))
    assert same_author(author("José García"), author("Garcia, Jose"))
    assert same_author(author("Z Zhang"), author("Z H Zhang"))
    assert not same_author(author("Zhang"), author("Z Zhang"))  # one token isn't enough
    assert not same_author(author("Leonardo Bonalume"), author("Leonardo Rossi"))
    assert not same_author(author("A", None), author("B", None))


def test_the_replication_dataset_is_dropped_before_it_could_merge() -> None:
    # Its title is the paper's plus three stems (Jaccard 0.84), and it shares an
    # author and a year with the paper: only its type keeps it out of the pool.
    assert title_jaccard(A0004_PAPER["title"], REPLICATION_DATASET["title"]) == pytest.approx(
        0.842, abs=0.001
    )
    types = LiveConfig(cache_path=Path("x")).work_types
    assert live_record(REPLICATION_DATASET, types) is None
    pool = CandidatePool(FakeEncoder(), max_size=50)
    pool.add(
        [r for w in (A0004_PAPER, REPLICATION_DATASET) if (r := live_record(w, types))],
        query_index=0,
        kind=SEMANTIC,
    )
    assert [(r["id"], r["aliases"]) for r in pool.records] == [("W7169768705", [])]


def test_versions_with_the_same_title_still_merge_as_in_phase1() -> None:
    preprint = _work("W7128640141", "From Search to GenAI Queries", "2026-02-10",
                     work_type="preprint", authors=[("A5011311687", "Noah D. Finkelstein")])
    article = _work("W7166562055", "From search to GenAI queries", "2026-06-29",
                    authors=[("A5139544687", "Anonymous")])
    assert len(_pool_of(preprint, article)) == 2  # 3 content words: too short for Phase 1
    long_title = "From search to GenAI queries: Global trends in physics information-seeking"
    pool = _pool_of({**preprint, "title": long_title}, {**article, "title": long_title})
    assert [(r["id"], r["aliases"]) for r in pool.records] == [("W7128640141", ["W7166562055"])]


def test_the_original_paper_matches_by_id_alias_doi_or_exact_normalised_title() -> None:
    original = {
        "id": "W7166562055",
        "aliases": ["W1"],
        "doi": "10.1103/g1vh-fl9s",
        "title": "From search to GenAI queries: Global trends",
    }
    live = live_record(_work("W7166562055", "Another title", "2026-06-29"))
    assert live is not None
    assert match_original([live], original) == "id"
    assert match_original([{**live, "id": "W2", "aliases": ["W7166562055"]}], original) == "alias"
    assert match_original([{**live, "id": "W1"}], original) == "alias"  # the original's alias
    assert match_original([{**live, "id": "W2", "doi": "10.1103/g1vh-fl9s"}], original) == "doi"
    retitled = {**live, "id": "W2", "title": "From Search to GenAI Queries - Global Trends"}
    assert match_original([retitled], original) == "title"
    other = {**live, "id": "W2", "title": "From search to GenAI queries"}
    assert match_original([other], original) is None
    assert match_original([other, retitled], original) == "title"  # any version counts


def test_pool_keeps_the_first_arrivals_up_to_its_size() -> None:
    records = [live_record(w) for w in WORKS[:3]]
    pool = CandidatePool(FakeEncoder(), max_size=2)
    pool.add([records[2], records[0], records[1]], query_index=0, kind=SEMANTIC)
    assert ([r["id"] for r in pool.records], pool.dropped) == ([T1, T3], 1)


class _CountingEncoder(FakeEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.encoded: list[str] = []

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return super().encode_docs(texts)  # FakeEncoder encodes queries as documents

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        self.encoded.extend(texts)
        return super().encode_docs(texts)


def test_pool_encodes_each_document_once() -> None:
    encoder = _CountingEncoder()
    records = [live_record(w) for w in WORKS]
    pool = CandidatePool(encoder, max_size=50)
    pool.add(records[:5], query_index=0, kind=SEMANTIC)
    pool.rank("memory")
    pool.add(records[3:], query_index=1, kind=KEYWORD)
    pool.rank("memory")
    pool.rank("accuracy")
    assert len(encoder.encoded) == len(WORKS) == len(set(encoder.encoded))


@pytest.fixture
def phase1_hybrid(tmp_path: Path) -> HybridRetriever:
    """Phase 1's B3 over a frozen corpus of the same documents the pool will hold."""
    data = corpus_config_dict(tmp_path / "corpus")
    data["retrieval"]["index_dir"] = str(tmp_path / "index")
    config = config_from_dict(data)
    freeze(config.corpus.out_dir, corpus_records(), target_rows(), {})
    build_index(config, FakeEncoder())
    index = load_index(config)
    return HybridRetriever(BM25Retriever(index), DenseRetriever(index, FakeEncoder()))


def test_pool_ranks_exactly_as_phase1_b3_and_b5_on_the_same_documents(
    phase1_hybrid: HybridRetriever,
) -> None:
    pool = CandidatePool(FakeEncoder(), max_size=50)
    pool.add([live_record(w) for w in WORKS], query_index=0, kind=SEMANTIC)
    queries = [GOOD[T1], "learned clusters attention memory", "reward calibration"]
    for query in queries:
        assert pool.rank(query) == phase1_hybrid.rank(query)
    assert pool.fused(queries) == fuse_runs([phase1_hybrid.rank(q) for q in queries])
    assert CandidatePool(FakeEncoder(), max_size=5).rank("memory") == []  # an empty pool


# --- B5 on the live pool -----------------------------------------------------


def _live_search(
    cfg: Config, fake: FakeLiveOpenAlex, replies: Sequence[str], text: str = GOOD[T1]
) -> tuple[Any, FakeLLM]:
    llm = FakeLLM(replies)
    result = live_search(
        text, cfg, search=_client(cfg, fake), encoder=FakeEncoder(), agent=llm
    )
    return result, llm


def test_b5_runs_on_live_search_and_reranks_the_final_pool(cfg: Config) -> None:
    fake = FakeLiveOpenAlex()
    q1, q2 = "learned clusters attention memory", "summarisation memory"
    result, llm = _live_search(cfg, fake, [_search(q1), _search(q2), STOP])
    assert result.trace.queries == [GOOD[T1], q1, q2]
    assert result.trace.stop_reason == "agent_stop"
    # The user's 60-word text gets semantic search only; the agent's queries get both.
    assert [next(k for k in ("search.semantic", "search") if k in r) for r in fake.requests] == [
        "search.semantic",
        "search",
        "search.semantic",
        "search",
        "search.semantic",
    ]
    assert [(s.query, s.kind) for s in result.searches][:2] == [
        (GOOD[T1], SEMANTIC),
        (q1, KEYWORD),
    ]
    # The agent saw titles from the live pool.
    assert f"- {TITLES[T1]}" in llm.calls[0][0]["content"]
    assert result.candidates[0].record["id"] == T1
    assert [c.rank for c in result.candidates] == list(range(1, 6))  # top_k = 5
    first_hit = result.candidates[0].hits[0]
    assert (first_hit.query_index, first_hit.kind) == (0, SEMANTIC)
    # The final ranking is B5's: RRF over every query's ranking of the final pool.
    pool = CandidatePool(FakeEncoder(), max_size=500)
    for s in result.searches:
        pool.add(s.records, query_index=0, kind=s.kind)
    expected = pool.fused(result.trace.queries)[:5]
    assert [c.record["id"] for c in result.candidates] == expected
    assert result.pool_size == len(pool) and result.pool_dropped == 0
    assert (result.encoder, result.encoder_device) == ("fake-encoder", "cpu")


def test_a_check_reports_its_cost_and_replays_its_searches_for_free(cfg: Config) -> None:
    fake = FakeLiveOpenAlex()
    replies = [_search("summarisation memory"), STOP]
    first, llm = _live_search(cfg, fake, replies)
    ledger = first.ledger
    assert ledger.openalex_cost == pytest.approx(0.003)  # semantic, then keyword + semantic
    assert ledger.openalex_spent == pytest.approx(0.003)
    assert ledger.llm.calls == 2 == len(llm.calls)
    assert ledger.method_cost == pytest.approx(0.003 + ledger.llm.method_cost)
    assert ledger.billed_cost == pytest.approx(0.003 + ledger.llm.billed_cost)
    again, _ = _live_search(cfg, fake, replies)
    assert len(fake.requests) == 3  # every search came from the cache
    assert again.ledger.openalex_cost == pytest.approx(0.003)
    assert again.ledger.openalex_spent == 0.0
    assert [c.record["id"] for c in again.candidates] == [
        c.record["id"] for c in first.candidates
    ]
    summary = first.to_dict()
    assert summary["queries"] == [GOOD[T1], "summarisation memory"]
    assert summary["cost"]["openalex_cost"] == pytest.approx(0.003)
    assert sorted(summary["candidates"][0]["found_by"][0]) == ["kind", "query_index", "rank"]


def test_the_agent_keeps_the_b5_caps(cfg: Config) -> None:
    fake = FakeLiveOpenAlex()
    replies = [_search(f"memory query {n}") for n in range(10)]
    result, llm = _live_search(cfg, fake, replies)
    assert len(result.trace.queries) == cfg.retrieval.max_queries == 5
    assert len(llm.calls) == 4 and result.trace.stop_reason == "max_queries"


def test_an_agent_call_that_could_cross_the_cost_cap_fails_the_check(tmp_path: Path) -> None:
    data = corpus_config_dict(tmp_path / "corpus")
    data["live"] = {
        **LIVE,
        "cache_path": str(tmp_path / "openalex.sqlite"),
        "cost_cap_per_check": 0.0015,
    }
    capped = config_from_dict(data)
    fake = FakeLiveOpenAlex()
    llm = FakeLLM([_search("memory"), STOP])
    search = _client(capped, fake)
    with pytest.raises(CheckCostCapExceeded, match="an agent call") as caught:
        live_search(GOOD[T1], capped, search=search, encoder=FakeEncoder(), agent=llm)
    assert len(fake.requests) == 1  # the first search fit; the agent call never ran
    assert llm.calls == []
    ledger = caught.value.ledger  # what the check cost when it stopped
    assert ledger is not None
    assert (ledger.searches, ledger.searches_sent, ledger.openalex_cost) == (1, 1, 0.001)


def test_the_ledger_refuses_a_search_over_the_cap(cfg: Config) -> None:
    assert cfg.live is not None
    fake = FakeLiveOpenAlex()
    pool = CandidatePool(FakeEncoder(), max_size=10)
    ranker = LiveRanker(_client(cfg, fake), pool, CheckLedger(cap=0.0025), cfg.live)
    ranker.rank("memory")  # keyword + semantic: $0.002
    assert len(fake.requests) == 2
    with pytest.raises(CheckCostCapExceeded, match="a keyword search"):
        ranker.rank("accuracy")
    assert len(fake.requests) == 2


def test_long_queries_skip_keyword_search(cfg: Config) -> None:
    assert cfg.live is not None
    ranker = LiveRanker(
        _client(cfg, FakeLiveOpenAlex()),
        CandidatePool(FakeEncoder(), max_size=10),
        CheckLedger(cap=1.0),
        cfg.live,
    )
    assert ranker.kinds("word " * 32) == [KEYWORD, SEMANTIC]
    assert ranker.kinds("word " * 33) == [SEMANTIC]
    no_semantic = LiveConfig(cache_path=cfg.live.cache_path, semantic_search=False)
    assert LiveRanker(ranker.search, ranker.pool, ranker.ledger, no_semantic).kinds("a b") == [
        KEYWORD
    ]


# --- the real API --------------------------------------------------------------


@pytest.mark.network
def test_live_openalex_keyword_and_semantic_search(tmp_path: Path) -> None:
    """One keyword and one semantic search against api.openalex.org ($0.002)."""
    load_env_file()
    live = LiveConfig(cache_path=tmp_path / "openalex.sqlite")
    client = LiveSearchClient(
        live, OpenAlexCache(live.cache_path), api_key=os.environ.get("OPENALEX_API_KEY") or None
    )
    query = "terminology shift in prior art retrieval for research proposals"
    for kind in (KEYWORD, SEMANTIC):
        response = client.search(query, kind)
        assert 1 <= response.n_results <= 50
        assert response.records and all(r["abstract"] and r["title"] for r in response.records)
        assert response.cost_usd > 0 and not response.cached
    assert client.search(query, SEMANTIC).cached
    client.close()
