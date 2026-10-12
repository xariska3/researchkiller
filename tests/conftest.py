"""Shared test doubles: FakeLLM, FakeEncoder, and a fake OpenAlex with a fixture corpus.

Tests never touch the network: real httpx transports are blocked for every
test not marked `network`, LLM traffic goes through FakeLLM, and OpenAlex
traffic through FakeOpenAlex. Import the doubles with
``from conftest import FakeLLM, FakeOpenAlex, make_model_config, ...``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from rk.config import Config, ModelConfig, config_from_dict
from rk.corpus import OpenAlexClient
from rk.llm import Completion, Ledger, LLMCache, Message, Usage, token_cost
from rk.text import content_stems

TEST_API_KEY_ENV = "RK_TEST_API_KEY"


def make_model_config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "name": "fake-model",
        "family": "fake",
        "base_url": "http://llm.test/v1",
        "api_key_env": TEST_API_KEY_ENV,
        "price_in_per_m": 1.0,
        "price_out_per_m": 2.0,
        "temperature": 0.0,
        "max_tokens": 64,
        "seed": 7,
        "training_cutoff": date(2024, 1, 1),
    }
    values.update(overrides)
    return ModelConfig(**values)


class FakeLLM:
    """Scripted stand-in for an OpenAI-compatible chat model.

    `replies` is a list of reply strings, served in order, or a function from the
    messages to a reply. Token usage is whitespace word counts, so tests can
    predict it. Use it directly as a ChatModel (`complete`) or under a real
    LLMClient via `transport`, which answers like a /chat/completions endpoint.
    Every prompt received is recorded in `calls`; over the transport, every
    request body and header set in `requests` and `headers`. With `provider`,
    replies name it as the serving provider, as OpenRouter's router metadata does;
    with `billed_cost`, they report it as OpenRouter's usage.cost.
    """

    def __init__(
        self,
        replies: Sequence[str] | Callable[[list[dict[str, str]]], str],
        *,
        model: ModelConfig | None = None,
        provider: str | None = None,
        billed_cost: Any = None,  # any JSON value, to test parsing; None leaves it out
    ) -> None:
        self.model = model or make_model_config()
        self.provider = provider
        self.billed_cost = billed_cost
        self._replies = replies if callable(replies) else list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.seeds: list[int | None] = []  # the seed= of each direct complete() call
        self.response_formats: list[Mapping[str, Any] | None] = []  # and its response_format=
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    def _reply(self, messages: Sequence[Message]) -> tuple[str, int, int]:
        msgs = [dict(m) for m in messages]
        self.calls.append(msgs)
        if callable(self._replies):
            text = self._replies(msgs)
        elif self._replies:
            text = self._replies.pop(0)
        else:
            raise AssertionError(f"FakeLLM script exhausted after {len(self.calls) - 1} replies")
        prompt_tokens = sum(self.count_tokens(m["content"]) for m in msgs)
        return text, prompt_tokens, self.count_tokens(text)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> Completion:
        self.seeds.append(seed)
        self.response_formats.append(response_format)
        text, prompt_tokens, completion_tokens = self._reply(messages)
        cost = token_cost(self.model, prompt_tokens, completion_tokens)
        usage = Usage(
            calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            method_cost=cost,
            money_spent=cost,
        )
        return Completion(
            text=text,
            usage=usage,
            cached=False,
            model=self.model.name,
            finish_reason="stop",
            provider=self.provider,
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        """httpx.MockTransport handler speaking the /chat/completions protocol."""
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(request.headers)
        text, prompt_tokens, completion_tokens = self._reply(body["messages"])
        reply: dict[str, Any] = {
            "id": f"fake-{len(self.calls)}",
            "object": "chat.completion",
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if self.billed_cost is not None:
            reply["usage"]["cost"] = self.billed_cost
        if self.provider is not None:  # the shape of a real OpenRouter reply (2026-10-12)
            reply["openrouter_metadata"] = {
                "requested": body["model"],
                "attempt": 1,
                "endpoints": {
                    "total": 2,
                    "available": [
                        {"provider": "Elsewhere", "model": body["model"], "selected": False},
                        {"provider": self.provider, "model": body["model"], "selected": True},
                    ],
                },
            }
        return httpx.Response(200, json=reply)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def close(self) -> None:
        """Nothing to release; lets FakeLLM stand in where an LLMClient gets closed."""


class FakeEncoder:
    """Deterministic bag-of-stems encoder (rk.retrieval.Encoder): feature-hashed, unit float32.

    Texts that share content stems get higher cosine similarity, so retrieval
    tests behave sensibly. Uses blake2b, not hash(), so vectors are identical
    across processes. Empty texts encode to the zero vector. Queries and
    documents are encoded the same way.
    """

    name = "fake-encoder"
    revision = "fa4e000000000000000000000000000000000000"  # a valid-looking commit hash
    device = "cpu"
    dtype = "float32"
    max_seq_length: int | None = None

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_docs(texts)

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for s in content_stems(text):
                digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
                vectors[row, int.from_bytes(digest, "little") % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.divide(vectors, norms, out=vectors, where=norms > 0)
        return vectors


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("network"):
        return  # live tests opt in explicitly and are deselected by default

    def refuse(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"tests must not touch the network (tried {request.url})")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(TEST_API_KEY_ENV, "sk-test")
    return "sk-test"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """One SQLite file for the cache and the spend table, as in a real run."""
    return tmp_path / "llm.sqlite"


@pytest.fixture
def llm_cache(db_path: Path) -> Iterator[LLMCache]:
    with LLMCache(db_path) as cache:
        yield cache


@pytest.fixture
def make_ledger(db_path: Path) -> Iterator[Callable[..., Ledger]]:
    """Factory for Ledgers on `db_path`: make_ledger(cap=1.0, command="test")."""
    opened: list[Ledger] = []

    def make(cap: float = 20.0, command: str = "test") -> Ledger:
        ledger = Ledger(db_path, cap, command=command)
        opened.append(ledger)
        return ledger

    yield make
    for ledger in opened:
        ledger.close()


# --- OpenAlex fixture corpus -------------------------------------------------

FIXTURE_BUILD_DATE = date(2026, 1, 15)
FIXTURE_TARGET_FROM = date(2025, 6, 1)
SWEEP_SUBFIELD, OTHER_SUBFIELD = 1702, 1703
SWEEP_FILTER = f"primary_topic.subfield.id:{SWEEP_SUBFIELD},from_publication_date:2023-01-01"

_VOCAB = (
    "signal", "estimate", "sparse", "kernel", "latent", "policy", "token", "region", "prior",
    "sample", "domain", "query", "label", "noise", "layer", "budget", "cluster", "vector",
    "agent", "reward", "index", "window", "filter", "margin", "weight", "bound", "scheme",
)


def _abstract(title: str, n_words: int) -> str:
    words = title.lower().split()
    i = sum(map(ord, title))
    while len(words) < n_words:
        words.append(_VOCAB[i % len(_VOCAB)])
        i += 7
    return " ".join(words[:n_words])


def oa_work(
    num: int,
    title: str,
    published: str,
    *,
    words: int = 30,
    refs: Sequence[int] = (),
    doi: str | None = None,
    subfield: int = SWEEP_SUBFIELD,
    language: str = "en",
    has_abstract: bool = True,
) -> dict[str, Any]:
    """A work as the OpenAlex API returns it (abstract as an inverted index)."""
    inverted: dict[str, list[int]] = {}
    for position, word in enumerate(_abstract(title, words).split()):
        inverted.setdefault(word, []).append(position)
    return {
        "id": f"https://openalex.org/W{num}",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "title": title,
        "publication_date": published,
        "abstract_inverted_index": inverted if has_abstract else None,
        "referenced_works": [f"https://openalex.org/W{r}" for r in refs],
        "primary_topic": {
            "id": "https://openalex.org/T10001",
            "display_name": "Fixture topic",
            "subfield": {
                "id": f"https://openalex.org/subfields/{subfield}",
                "display_name": "Artificial Intelligence" if subfield == SWEEP_SUBFIELD else "Other",
            },
        },
        "cited_by_count": 0,
        "language": language,
    }


_REFERENCE_TOPICS = (
    "kernel density estimation", "beam search decoding", "contrastive divergence",
    "sparse coding", "graph kernels", "dropout regularisation", "attention mechanisms",
    "policy gradients", "variational inference", "word embeddings", "label smoothing",
    "early stopping",
)
_SAME_TITLE = "Learning to rank scientific claims with weak supervision signals"
_T2_TITLE = "Contrastive pretraining of molecular graph encoders at scale"

FIXTURE_WORKS: list[dict[str, Any]] = [
    # Targets: post-cutoff sweep works, 120-word abstracts, 10+ references.
    oa_work(1001, "Sparse attention routing for long document summarisation", "2025-09-10",
            words=120, refs=[*range(2001, 2015), 3001, 2999]),  # W2999 doesn't exist
    oa_work(1002, _T2_TITLE, "2025-08-20", words=120, refs=range(2001, 2011), doi="10.1000/t2"),
    oa_work(1003, "Uncertainty calibration for reinforcement learning reward models",
            "2025-11-05", words=120, refs=range(2003, 2013)),
    # Post-cutoff but not candidates: too few references; abstract too short.
    oa_work(1004, "Hierarchical memory for streaming video question answering", "2025-10-01",
            words=120, refs=range(2001, 2006)),
    oa_work(1005, "Tabular foundation models for clinical risk prediction", "2025-07-15",
            words=40, refs=range(2001, 2013)),
    # Preprint and published version: different titles, same DOI (case differs).
    oa_work(1006, "Graph networks for traffic forecasting under sensor failures", "2025-07-02",
            refs=range(2001, 2004), doi="10.1000/pair"),
    oa_work(1007, "Robust traffic forecasting with graph networks under sensor failure",
            "2025-12-01", refs=range(2004, 2012), doi="10.1000/PAIR"),
    # Two unrelated works titled "Editorial": too few content words to merge on.
    oa_work(1008, "Editorial", "2025-06-15"),
    oa_work(1009, "Editorial", "2024-03-01"),
    # Same long title a year apart, different DOIs: merged by the title rule.
    oa_work(1010, _SAME_TITLE, "2024-02-10", doi="10.1000/a"),
    oa_work(1011, _SAME_TITLE, "2025-01-20", doi="10.1000/b"),
    # Dropped: wrong language; no abstract; bogus future date.
    oa_work(1012, "Neuronale Netze für die Wettervorhersage im Hochgebirge", "2025-08-01",
            language="de"),
    oa_work(1013, "Prompt compression for retrieval augmented generation", "2025-09-01",
            has_abstract=False),
    oa_work(1014, "Placeholder record with an impossible future date", "2031-01-01"),
    # Would be a candidate, but published before target_from_date.
    oa_work(1015, "Federated distillation for heterogeneous edge devices", "2024-12-01",
            words=120, refs=range(2001, 2013)),
    # Filler on both sides of target_from_date; three share a date.
    oa_work(1016, "Active learning for low resource named entity recognition", "2023-03-14"),
    oa_work(1017, "Causal discovery from interventional single cell data", "2023-09-30"),
    oa_work(1018, "Neural radiance fields for dynamic urban scenes", "2024-05-05"),
    oa_work(1019, "Symbolic regression with transformer guided search", "2024-05-05"),
    oa_work(1020, "Spectral clustering of citation networks over time", "2024-05-05"),
    oa_work(1021, "Energy based models for out of distribution detection", "2024-08-19"),
    oa_work(1022, "Curriculum design for multi agent cooperative navigation", "2025-02-11"),
    oa_work(1023, "Differentiable physics for soft robot control", "2025-06-01"),
    oa_work(1024, "Benchmarking code generation models on scientific software", "2025-10-20"),
    oa_work(1025, "Topological data analysis of protein folding pathways", "2026-01-10"),
    # The targets' references: another subfield, older, reachable only by id.
    *[
        oa_work(2001 + i, f"A study of {topic}", f"{2018 + i % 5}-{1 + i % 9:02d}-15",
                subfield=OTHER_SUBFIELD)
        for i, topic in enumerate(_REFERENCE_TOPICS)
    ],
    # Same title as W1010 but 4 years earlier: not merged.
    oa_work(2013, _SAME_TITLE, "2020-04-01", subfield=OTHER_SUBFIELD),
    # A reference without an abstract: dropped and counted.
    oa_work(2014, "A reference known only by its title", "2019-05-01", subfield=OTHER_SUBFIELD,
            has_abstract=False),
    # An earlier version of target W1002 (same DOI), reachable only as a reference.
    oa_work(3001, _T2_TITLE, "2025-07-01", words=120, refs=range(2001, 2011), doi="10.1000/t2",
            subfield=OTHER_SUBFIELD),
    # Citers: three cite W1001 (a cap of 2 keeps the newest two), one cites W1003.
    oa_work(4001, "Routing sparse attention in legal document review", "2025-12-01",
            refs=[1001], subfield=OTHER_SUBFIELD),
    oa_work(4002, "Long context summarisation with learned routing", "2026-01-05",
            refs=[1001], subfield=OTHER_SUBFIELD),
    oa_work(4003, "A survey of efficient attention", "2025-10-15", refs=[1001],
            subfield=OTHER_SUBFIELD),
    oa_work(4004, "Calibrated reward models for preference learning", "2025-12-20",
            refs=[1003], subfield=OTHER_SUBFIELD),
]


class FakeOpenAlex:
    """In-memory OpenAlex /works endpoint for httpx.MockTransport.

    Supports the filters the corpus builder sends, sort=publication_date:desc
    (ties come back in `tie_order` by id, to prove the builder doesn't rely on
    it), cursor and plain paging, and select. Rejects what OpenAlex rejects
    (unknown filters, sorting by anything else such as id, per_page over 100)
    with HTTP 400. `fail_with` status codes are returned first, one per request.
    """

    FILTER_KEYS = frozenset(
        {"primary_topic.subfield.id", "has_abstract", "from_publication_date",
         "to_publication_date", "ids.openalex", "cites"}
    )

    def __init__(
        self,
        works: Sequence[dict[str, Any]] = FIXTURE_WORKS,
        *,
        tie_order: str = "desc",
        fail_with: Sequence[int] = (),
    ) -> None:
        self.works = list(works)
        self.tie_order = tie_order
        self.fail_with = list(fail_with)
        self.requests: list[dict[str, str]] = []
        self.auth_headers: list[str | None] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self, **kwargs: Any) -> OpenAlexClient:
        """An OpenAlexClient wired to this fake, with throttling and backoff sleeps off."""
        kwargs.setdefault("sleep", lambda seconds: None)
        kwargs.setdefault("max_requests_per_second", float("inf"))
        return OpenAlexClient(transport=self.transport, **kwargs)

    def handle(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.requests.append(params)
        self.auth_headers.append(request.headers.get("authorization"))
        if self.fail_with:
            return httpx.Response(self.fail_with.pop(0), json={"error": "simulated"})
        try:
            matches = self._sort(self._filter(params.get("filter", "")), params.get("sort"))
            per_page = int(params.get("per_page", "25"))
            if not 1 <= per_page <= 100:
                raise ValueError(f"per_page must be 1-100, got {per_page}")
            fields = params["select"].split(",") if "select" in params else None
            if fields and any("." in f for f in fields):
                raise ValueError("select takes top-level fields only")
        except ValueError as exc:
            return httpx.Response(400, json={"error": "Invalid query", "message": str(exc)})
        cursor = params.get("cursor")
        start = 0 if cursor in (None, "*") else int(cursor.removeprefix("offset:"))
        page = matches[start : start + per_page]
        next_cursor = f"offset:{start + per_page}" if cursor is not None and page else None
        results = [{k: w[k] for k in fields} if fields else w for w in page]
        meta = {"count": len(matches), "per_page": per_page, "next_cursor": next_cursor}
        return httpx.Response(200, json={"meta": meta, "results": results})

    def _filter(self, filters: str) -> list[dict[str, Any]]:
        clauses = [c.partition(":")[::2] for c in filters.split(",") if c]
        unknown = [key for key, _ in clauses if key not in self.FILTER_KEYS]
        if unknown:
            raise ValueError(f"unknown filter(s) {unknown}")
        return [w for w in self.works if all(self._match(w, k, v) for k, v in clauses)]

    @staticmethod
    def _match(work: dict[str, Any], key: str, value: str) -> bool:
        if key == "primary_topic.subfield.id":
            return work["primary_topic"]["subfield"]["id"].endswith(f"/{value}")
        if key == "has_abstract":
            return bool(work["abstract_inverted_index"]) == (value == "true")
        if key == "from_publication_date":
            return work["publication_date"] >= value
        if key == "to_publication_date":
            return work["publication_date"] <= value
        if key == "ids.openalex":
            return work["id"].rsplit("/", 1)[-1] in value.split("|")
        cited = {f"https://openalex.org/{v}" for v in value.split("|")}  # cites
        return bool(cited & set(work["referenced_works"]))

    def _sort(self, works: list[dict[str, Any]], sort: str | None) -> list[dict[str, Any]]:
        by_id = sorted(works, key=lambda w: int(w["id"].rsplit("W", 1)[-1]))
        if sort is None:
            return by_id
        if sort != "publication_date:desc":
            raise ValueError(f"can't sort by {sort!r}")
        if self.tie_order == "desc":
            by_id.reverse()
        return sorted(by_id, key=lambda w: w["publication_date"], reverse=True)  # stable


def corpus_config_dict(out_dir: Path, **corpus_overrides: Any) -> dict[str, Any]:
    """A valid config (as a YAML-shaped dict) for building the fixture corpus into out_dir."""

    def model(name: str, family: str) -> dict[str, Any]:
        return dataclasses.asdict(make_model_config(name=name, family=family))

    return {
        "corpus": {
            "filters": [SWEEP_FILTER],
            "language": "en",
            "min_abstract_words": 20,
            "max_citers_per_target": 2,
            "max_works_per_sweep": 1000,
            "out_dir": str(out_dir),
            **corpus_overrides,
        },
        "bench": {
            "n_targets": 2,
            "target_from_date": FIXTURE_TARGET_FROM,
            "max_attempts_per_target": 3,
            "max_distinctive_overlap": 0.15,
            "max_content_jaccard": 0.15,
            "seed": 7,
            "out_dir": str(out_dir / "bench"),
        },
        "retrieval": {
            "query_prefix": "query: ",
            "encoder_revision": FakeEncoder.revision,
            "max_queries": 5,
            "max_llm_calls_agent": 6,
            "max_queries_unconstrained": 15,
            "max_llm_calls_agent_unconstrained": 20,
            "index_dir": str(out_dir / "index"),
        },
        "run": {
            "out_dir": str(out_dir / "runs"),
            "cache_path": str(out_dir / "llm.sqlite"),
            "seed": 1,
        },
        "paraphraser": model("para-model", "family-a"),
        "judge": model("judge-model", "family-b"),
        "agent": model("agent-model", "family-c"),
    }


def corpus_config(out_dir: Path, **corpus_overrides: Any) -> Config:
    return config_from_dict(corpus_config_dict(out_dir, **corpus_overrides))
