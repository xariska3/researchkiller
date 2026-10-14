"""Phase 2 Step 3, the competitive evaluation (python -m rk compete): the systems' clients,
the cache and cost caps, the resolver, the stratified sample, and whole runs.

Every API (Anthropic, OpenRouter, Semantic Scholar, OpenAlex lookups, Crossref) is a
fake behind httpx.MockTransport. The live-eval run the sample comes from is a run on the
fixture benchmark (FakeLiveOpenAlex, FakeLLM agent, FakeEncoder), as in test_live_eval.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from bench_data import GOOD, T1, T2, T3, TITLES, corpus_records, judge_reply, target_rows
from conftest import FakeEncoder, FakeLLM
from live_fakes import FakeLiveOpenAlex
from test_live_eval import B1_RANKS, B5_RANKS, _config_dict, _paraphraser, _phase1_run, agent

from rk import cli
from rk.bench import make_bench
from rk.compete import evaluate as compete_eval
from rk.compete.evaluate import (
    OURS,
    RESULTS_FILE,
    Query,
    holm,
    make_plan,
    match_item,
    quotas_for,
    run_compete,
    stratified_sample,
)
from rk.compete.resolve import UNRESOLVED, VERIFIED, Resolution, Resolver, titles_agree
from rk.compete.systems import (
    CompeteError,
    Item,
    ResponseCache,
    SpendLedger,
    SystemCapReached,
    ask,
    connect_store,
    find_doi,
    make_system,
    method_cost,
    parse_items,
    request_key,
)
from rk.config import CompeteSystemConfig, Config, ConfigError, config_from_dict, load_config
from rk.corpus import freeze
from rk.evaluate import RUN_FILE, SUMMARY_JSON, SUMMARY_MD, EvalError
from rk.live import evaluate as live_evaluate
from rk.live.evaluate import run_live_eval
from rk.live.fusion import PROPOSAL_EQUAL
from rk.live.openalex_search import LiveSearchClient, OpenAlexCache

NEIGHBOUR = {"title": "A real neighbour paper", "doi": "10.9999/real"}
FAKE_PAPER = {"title": "Imaginary results on nothing at all", "doi": "10.1234/fake"}


def no_sleep(seconds: float) -> None:
    pass


def _target(prompt: str) -> str:
    """The fixture target whose proposal is in a prompt or query."""
    return next(t for t in (T1, T2, T3) if GOOD[t] in prompt)


# --- configs -------------------------------------------------------------------------------


def claude_dict(**overrides: Any) -> dict[str, Any]:
    return {
        "name": "claude",
        "kind": "anthropic",
        "model": "claude-test",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "TEST_ANTHROPIC_KEY",
        "web_search_tool": "web_search_20260318",
        "max_searches": 5,
        "price_in_per_m": 2.0,
        "price_out_per_m": 10.0,
        "price_per_search": 0.01,
        "reserve_per_query": 0.5,
        "cost_cap": 10.0,
        **overrides,
    }


def openai_dict(**overrides: Any) -> dict[str, Any]:
    return {
        "name": "openai",
        "kind": "openrouter",
        "model": "openai/test",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "TEST_OPENROUTER_KEY",
        "web_search_tool": "openrouter:web_search",
        "max_searches": 3,
        "price_in_per_m": 2.0,
        "price_out_per_m": 10.0,
        "price_per_search": 0.01,
        "reserve_per_query": 0.5,
        "cost_cap": 10.0,
        **overrides,
    }


def perplexity_dict(**overrides: Any) -> dict[str, Any]:
    return {
        "name": "perplexity",
        "kind": "openrouter",
        "model": "perplexity/sonar",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "TEST_OPENROUTER_KEY",
        "search_context_size": "low",
        "price_in_per_m": 1.0,
        "price_out_per_m": 1.0,
        "price_per_request": 0.005,
        "reserve_per_query": 0.03,
        "cost_cap": 1.0,
        **overrides,
    }


def s2_dict(kind: str = "s2-search", **overrides: Any) -> dict[str, Any]:
    return {
        "name": kind,
        "kind": kind,
        "base_url": "https://api.semanticscholar.org/graph/v1",
        "api_key_env": "TEST_S2_KEY",
        **overrides,
    }


def system_config(data: dict[str, Any]) -> CompeteSystemConfig:
    return CompeteSystemConfig(**data)


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "anthropic-key")
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "openrouter-key")
    monkeypatch.delenv("TEST_S2_KEY", raising=False)


def test_the_committed_config_has_the_compete_section() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "phase1.yaml")
    assert cfg.compete is not None
    assert [s.name for s in cfg.compete.systems] == [
        "claude",
        "openai",
        "perplexity",
        "s2-search",
        "s2-snippets",
    ]
    assert all(s.cost_cap > 0 for s in cfg.compete.systems if s.kind in ("anthropic", "openrouter"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "bing"}, "kind must be one of"),
        ({"cost_cap": 0.0}, "cost_cap > 0"),
        ({"reserve_per_query": 0.0}, "reserve_per_query > 0"),
        ({"web_search_tool": None}, "need web_search_tool"),
        ({"model": None}, "need a model"),
        ({"name": "a,b"}, "no commas"),
        ({"search_context_size": "huge"}, "search_context_size"),
        ({"max_searches": 0}, "max_searches must be >= 1"),
    ],
)
def test_system_configs_are_checked(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        system_config(claude_dict(**overrides))


def test_compete_config_checks_names_and_free_systems(tmp_path: Path) -> None:
    def load(systems: list[dict[str, Any]]) -> Config:
        data = _config_dict(tmp_path)
        data["compete"] = {"runs": ["runs/x"], "cache_path": "c.sqlite", "systems": systems}
        return config_from_dict(data)

    cfg = load([claude_dict(), s2_dict()])
    assert cfg.compete is not None and cfg.compete.n_queries == 100
    assert cfg.compete.systems[1].cost_cap == 0
    with pytest.raises(ConfigError, match="unique"):
        load([claude_dict(), claude_dict()])
    with pytest.raises(ConfigError, match="'ours'"):
        load([claude_dict(name="ours")])
    with pytest.raises(ConfigError, match="free systems have no cost_cap"):
        load([s2_dict(cost_cap=1.0)])
    with pytest.raises(ConfigError, match="the web_search_tool of openrouter systems"):
        load([openai_dict(web_search_tool="web_search_20260318")])


# --- parsing answers -----------------------------------------------------------------------


def test_find_doi_in_dois_and_links() -> None:
    assert find_doi("https://doi.org/10.1000/ABC.") == "10.1000/abc"
    assert find_doi("https://dl.acm.org/doi/10.1145/3442188.3445922") == "10.1145/3442188.3445922"
    assert find_doi("doi:10.18653/v1/2023.acl-long.1)") == "10.18653/v1/2023.acl-long.1"
    assert find_doi("https://arxiv.org/abs/2310.01234") is None
    assert find_doi("null") is None and find_doi(None) is None


def test_parse_items_reads_the_papers_json() -> None:
    papers = [
        {"title": " Sparse routing ", "year": "2024", "doi": "https://doi.org/10.1000/X."},
        {"title": None, "doi": None, "link": None},  # names nothing: skipped
        {"title": "Linked", "year": True, "doi": "none", "link": "https://arxiv.org/abs/2401.1"},
        {"title": "Not a link", "link": "n/a"},
        "not a paper",
    ]
    text = "Here they are:\n```json\n" + json.dumps({"papers": papers}) + "\n```"
    items, error = parse_items(text)
    assert error is None
    assert items == [
        Item("Sparse routing", 2024, "10.1000/x", None),
        Item("Linked", None, None, "https://arxiv.org/abs/2401.1"),
        Item("Not a link", None, None, None),
    ]
    bare = [{"title": f"Paper {i}"} for i in range(12)]
    items, error = parse_items(json.dumps(bare))
    assert error is None and [i.title for i in items] == [f"Paper {i}" for i in range(10)]
    assert parse_items("I found nothing relevant.")[0] == []
    assert "no JSON value" in (parse_items("I found nothing relevant.")[1] or "")
    assert parse_items('{"answer": 1}') == ([], "the answer's JSON has no list of papers")


# --- the fake APIs ---------------------------------------------------------------------------


def _papers_text(target: str) -> str:
    """Claude's answer: a real neighbour first, then the original by title only."""
    papers = [
        {**NEIGHBOUR, "year": 2024, "link": None},
        {"title": TITLES[target], "year": 2026, "doi": None, "link": None},
    ]
    return json.dumps({"papers": papers})


def anthropic_reply(text: str, *, stop: str = "end_turn", searches: int = 2) -> dict[str, Any]:
    middle = len(text) // 2  # the answer comes split over two text blocks, as with citations
    return {
        "content": [
            {"type": "text", "text": "I'll search for related work."},
            {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": []},
            {"type": "text", "text": text[:middle]},
            {"type": "text", "text": text[middle:], "citations": []},
        ],
        "stop_reason": stop,
        "usage": {
            "input_tokens": 10_000,
            "output_tokens": 500,
            "server_tool_use": {"web_search_requests": searches},
        },
    }


def chat_reply(text: str, *, cost: float | None = 0.006, searches: int | None = None) -> dict:
    usage: dict[str, Any] = {"prompt_tokens": 300, "completion_tokens": 700}
    if cost is not None:
        usage["cost"] = cost
    if searches is not None:
        usage["server_tool_use"] = {"web_search_requests": searches}
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": usage,
        "provider": "Perplexity",
    }


class FakeWorld:
    """Every API the competitive evaluation talks to, for httpx.MockTransport.

    Claude lists a real neighbour and the original (by title); OpenAI lists the
    original's title under the neighbour's DOI; Perplexity lists a paper that doesn't
    exist (or, for T3's proposal, answers without JSON); Semantic Scholar finds nothing.
    OpenAlex knows the neighbour's DOI and each original under its Crossref DOI;
    Crossref knows the originals' titles. `fail_with` status codes are returned first,
    one per request.
    """

    def __init__(self, *, fail_with: tuple[int, ...] = ()) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_with = list(fail_with)
        self.openalex = {
            f"doi:{NEIGHBOUR['doi']}": {"id": "https://openalex.org/W900", **NEIGHBOUR},
            **{
                f"doi:10.5555/{t.lower()}": {
                    "id": f"https://openalex.org/{t}",
                    "doi": f"https://doi.org/10.5555/{t.lower()}",
                    "title": TITLES[t],
                    "publication_year": 2026,
                }
                for t in (T1, T2, T3)
            },
        }

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def hosts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.requests:
            counts[r.url.host] = counts.get(r.url.host, 0) + 1
        return counts

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_with:
            return httpx.Response(self.fail_with.pop(0), json={"error": "simulated"})
        host = request.url.host
        if host == "api.anthropic.com":
            prompt = json.loads(request.content)["messages"][0]["content"]
            return httpx.Response(200, json=anthropic_reply(_papers_text(_target(prompt))))
        if host == "openrouter.ai":
            body = json.loads(request.content)
            target = _target(body["messages"][0]["content"])
            if body["model"].startswith("openai/"):
                paper = {"title": TITLES[target], "doi": NEIGHBOUR["doi"]}  # the wrong DOI
                return httpx.Response(200, json=chat_reply(json.dumps([paper]), searches=3))
            if target == T3:
                return httpx.Response(200, json=chat_reply("I could not find anything."))
            text = "Sure! " + json.dumps({"papers": [{**FAKE_PAPER, "year": 2023}]})
            return httpx.Response(200, json=chat_reply(text))
        if host == "api.semanticscholar.org":
            return httpx.Response(200, json={"total": 0, "offset": 0, "data": []})
        if host == "api.openalex.org":
            ident = urllib.parse.unquote(request.url.path.removeprefix("/works/"))
            work = self.openalex.get(ident)
            return httpx.Response(200, json=work) if work else httpx.Response(404, json={})
        if host == "api.crossref.org":
            title = request.url.params["query.bibliographic"]
            issued = {"date-parts": [[2026]]}
            items = [
                {"DOI": f"10.5555/{t.lower()}", "title": [TITLES[t]], "issued": issued}
                for t in (T1, T2, T3)
                if TITLES[t] == title
            ]
            return httpx.Response(200, json={"message": {"items": items}})
        raise AssertionError(f"unexpected request {request.url}")


def client(data: dict[str, Any], world: FakeWorld) -> Any:
    return make_system(system_config(data), transport=world.transport, sleep=no_sleep)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[tuple[ResponseCache, SpendLedger]]:
    conn = connect_store(tmp_path / "compete.sqlite")
    yield ResponseCache(conn), SpendLedger(conn)
    conn.close()


# --- the clients ---------------------------------------------------------------------------


def test_anthropic_request_continuation_usage_and_answer(
    store: tuple[ResponseCache, SpendLedger],
) -> None:
    replies = [
        anthropic_reply("", stop="pause_turn", searches=2),
        anthropic_reply(_papers_text(T1), searches=1),
    ]
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=replies[len(sent) - 1])

    claude = make_system(
        system_config(claude_dict()), transport=httpx.MockTransport(handle), sleep=no_sleep
    )
    cache, ledger = store
    answer = ask(claude, GOOD[T1], qid="A1", cache=cache, ledger=ledger)
    assert len(sent) == 2
    assert sent[0].headers["x-api-key"] == "anthropic-key"
    assert sent[0].headers["anthropic-version"] == "2023-06-01"
    first, second = (json.loads(r.content) for r in sent)
    assert first["tools"] == [{"type": "web_search_20260318", "name": "web_search", "max_uses": 5}]
    prompt = first["messages"][0]["content"]
    assert prompt.startswith(
        "Find the existing papers most closely related to this research proposal. List up to "
        "10, each with title, year, and DOI or link."
    )
    assert GOOD[T1] in prompt and '{"papers": [' in prompt
    # pause_turn: the paused assistant content goes back unchanged
    assert second["messages"][1] == {"role": "assistant", "content": replies[0]["content"]}
    reading = answer.reading
    assert reading.items == [
        Item(NEIGHBOUR["title"], 2024, NEIGHBOUR["doi"], None),
        Item(TITLES[T1], 2026, None, None),
    ]
    assert (reading.input_tokens, reading.output_tokens, reading.searches) == (20_000, 1000, 3)
    assert answer.cost == pytest.approx(20_000 * 2e-6 + 1000 * 10e-6 + 3 * 0.01)
    assert answer.money_spent == answer.cost and not answer.cached  # Anthropic reports no cost


def test_openrouter_requests_for_openai_and_perplexity(
    store: tuple[ResponseCache, SpendLedger],
) -> None:
    world = FakeWorld()
    cache, ledger = store
    openai = client(openai_dict(reasoning_effort="low"), world)
    answer = ask(openai, GOOD[T1], qid="A1", cache=cache, ledger=ledger)
    request = world.requests[-1]
    assert request.headers["authorization"] == "Bearer openrouter-key"
    body = json.loads(request.content)
    assert body["tools"] == [
        {"type": "openrouter:web_search", "parameters": {"engine": "native", "max_uses": 3}}
    ]
    assert body["reasoning"] == {"effort": "low"} and "web_search_options" not in body
    assert answer.reading.searches == 3
    assert answer.reading.billed_cost == 0.006 and answer.money_spent == 0.006
    assert answer.cost == pytest.approx(300 * 2e-6 + 700 * 10e-6 + 3 * 0.01)
    assert answer.reading.provider == "Perplexity"

    sonar = client(perplexity_dict(), world)
    answer = ask(sonar, GOOD[T2], qid="A2", cache=cache, ledger=ledger)
    body = json.loads(world.requests[-1].content)
    assert body["web_search_options"] == {"search_context_size": "low"} and "tools" not in body
    assert answer.reading.items == [Item(FAKE_PAPER["title"], 2023, FAKE_PAPER["doi"], None)]
    assert answer.cost == pytest.approx(300e-6 + 700e-6 + 0.005)  # tokens plus the request fee


def test_semantic_scholar_search_and_snippets(
    store: tuple[ResponseCache, SpendLedger], monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = [
        {"title": "DOI", "year": 2025, "externalIds": {"DOI": "10.1000/A"}, "url": "https://s"},
        {"title": "Preprint", "year": 2026, "externalIds": {"ArXiv": "2601.01234"}},
    ]
    snippets = [
        {"paper": {"corpusId": 7, "title": "Seven"}, "snippet": {"text": "a"}},
        {"paper": {"corpusId": 7, "title": "Seven"}, "snippet": {"text": "b"}},
        {"paper": {"corpusId": 8, "title": "Eight"}, "snippet": {"text": "c"}},
    ]
    statuses = [429]
    sent: list[httpx.Request] = []
    slept: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if statuses:
            return httpx.Response(statuses.pop(0), headers={"retry-after": "3"})
        if request.url.path.endswith("/paper/search"):
            return httpx.Response(200, json={"total": 2, "data": papers})
        return httpx.Response(200, json={"data": snippets})

    cache, ledger = store
    search = make_system(
        system_config(s2_dict()), transport=httpx.MockTransport(handle), sleep=slept.append
    )
    answer = ask(search, GOOD[T1], qid="A1", cache=cache, ledger=ledger)
    assert len(sent) == 2 and 3.0 in slept  # the 429 waited for its Retry-After
    assert dict(sent[-1].url.params) == {
        "query": GOOD[T1],
        "limit": "10",
        "fields": "title,year,externalIds,url",
    }
    assert "x-api-key" not in sent[-1].headers  # no key set: the shared pool
    assert answer.reading.items == [
        Item("DOI", 2025, "10.1000/a", "https://s"),
        Item("Preprint", 2026, "10.48550/arxiv.2601.01234", None),
    ]
    assert answer.cost == 0.0 and ledger.recorded("s2-search") == 0.0

    monkeypatch.setenv("TEST_S2_KEY", "s2-key")
    snippet_search = make_system(
        system_config(s2_dict("s2-snippets")), transport=httpx.MockTransport(handle), sleep=no_sleep
    )
    answer = ask(snippet_search, GOOD[T1], qid="A1", cache=cache, ledger=ledger)
    assert sent[-1].headers["x-api-key"] == "s2-key"
    assert dict(sent[-1].url.params) == {"query": GOOD[T1], "limit": "50"}
    assert answer.reading.items == [
        Item("Seven", None, None, "https://api.semanticscholar.org/CorpusId:7"),
        Item("Eight", None, None, "https://api.semanticscholar.org/CorpusId:8"),
    ]


# --- the cache and the cost caps -------------------------------------------------------------


def test_a_cached_answer_is_free_and_keeps_its_latency(
    store: tuple[ResponseCache, SpendLedger],
) -> None:
    world = FakeWorld()
    ticks = iter(range(100))
    claude = make_system(
        system_config(claude_dict()),
        transport=world.transport,
        sleep=no_sleep,
        clock=lambda: float(next(ticks)),
    )
    cache, ledger = store
    first = ask(claude, GOOD[T1], qid="A1", cache=cache, ledger=ledger)
    assert len(world.requests) == 1 and first.latency_s == 1.0
    again = ask(claude, GOOD[T1], qid="A1", cache=cache, ledger=ledger)
    assert len(world.requests) == 1  # served from the cache
    assert again.cached and again.money_spent == 0.0
    assert (again.cost, again.latency_s, again.reading) == (first.cost, 1.0, first.reading)
    assert ledger.recorded("claude") == pytest.approx(first.cost)
    assert request_key(claude.request(GOOD[T1])) in cache


def test_the_cost_cap_refuses_a_call_that_could_cross_it(
    store: tuple[ResponseCache, SpendLedger], tmp_path: Path
) -> None:
    world = FakeWorld()
    cache, ledger = store
    claude = client(claude_dict(reserve_per_query=0.04, cost_cap=0.07), world)
    ask(claude, GOOD[T1], qid="A1", cache=cache, ledger=ledger)  # costs $0.045
    with pytest.raises(SystemCapReached, match=r"\$0\.0450 of its \$0\.07 cost cap"):
        ask(claude, GOOD[T2], qid="A2", cache=cache, ledger=ledger)
    assert len(world.requests) == 1
    # The cap holds across runs: a ledger on a new connection reads what was spent.
    conn = connect_store(tmp_path / "compete.sqlite")
    with pytest.raises(SystemCapReached):
        SpendLedger(conn).charge("claude", 0.04, 0.07)
    conn.close()


def test_failed_calls_are_booked_only_if_they_may_have_been_billed(
    store: tuple[ResponseCache, SpendLedger],
) -> None:
    cache, ledger = store
    world = FakeWorld(fail_with=(500,) * 7)  # every attempt fails
    claude = client(claude_dict(reserve_per_query=0.5), world)
    with pytest.raises(CompeteError, match="giving up after 7 attempts"):
        ask(claude, GOOD[T1], qid="A1", cache=cache, ledger=ledger)
    assert ledger.recorded("claude") == 0.5  # sent without a 4xx: booked at its reservation
    world = FakeWorld(fail_with=(400,))
    claude = client(claude_dict(reserve_per_query=0.5), world)
    with pytest.raises(CompeteError, match="HTTP 400"):
        ask(claude, GOOD[T2], qid="A2", cache=cache, ledger=ledger)
    assert ledger.recorded("claude") == 0.5 and ledger.spent("claude") == 0.5  # released


def test_method_cost_counts_tokens_searches_and_the_request_fee() -> None:
    from rk.compete.systems import Reading

    reading = Reading([], "", None, input_tokens=1_000_000, output_tokens=100_000, searches=4)
    cfg = system_config(perplexity_dict(price_per_search=0.01))
    assert method_cost(cfg, reading) == pytest.approx(1.0 + 0.1 + 0.04 + 0.005)


# --- the resolver -------------------------------------------------------------------------------


def test_titles_agree() -> None:
    assert titles_agree("Attention Is All You Need", "attention is all you need.")
    full = "Calibrated reward models: a study of preference learning at scale"
    assert titles_agree("Calibrated reward models for preference learning at scale", full)
    assert titles_agree(  # a dropped subtitle
        "BERT: Pre-training of deep bidirectional transformers",
        "BERT pre-training of deep bidirectional transformers for language understanding",
    )
    assert not titles_agree("Graph networks for traffic forecasting", "Graph neural networks")
    assert not titles_agree("", "")


def test_resolution_routes_and_caching(tmp_path: Path) -> None:
    world = FakeWorld()
    conn = connect_store(tmp_path / "compete.sqlite")
    resolver = Resolver(
        conn,
        openalex_key="oa-key",
        crossref_mailto="me@example.org",
        transport=world.transport,
        sleep=no_sleep,
    )
    # A DOI whose work has the listed title.
    found = resolver.resolve(Item(NEIGHBOUR["title"], 2024, NEIGHBOUR["doi"], None))
    assert (found.status, found.route, found.openalex_id) == (VERIFIED, "openalex-doi", "W900")
    lookup = world.requests[-1]
    assert lookup.headers["authorization"] == "Bearer oa-key"
    assert lookup.url.params["select"] == "id,doi,title,publication_year"
    # A right DOI under a wrong title is a mismatch; the title then verifies through Crossref.
    found = resolver.resolve(Item(TITLES[T2], None, NEIGHBOUR["doi"], None))
    assert (found.status, found.route, found.doi_mismatch) == (VERIFIED, "crossref-title", True)
    assert (found.openalex_id, found.doi) == (T2, "10.5555/w102")
    crossref = next(r for r in world.requests if r.url.host == "api.crossref.org")
    assert crossref.url.params["mailto"] == "me@example.org"
    # Links: an OpenAlex work id, and an arXiv id (as arXiv's DataCite DOI).
    item = Item(None, None, None, "https://openalex.org/W900")
    assert list(Resolver.identifiers(item)) == [("openalex-id", "W900")]
    item = Item("Paper", None, None, "https://arxiv.org/pdf/2401.01234v2")
    assert list(Resolver.identifiers(item)) == [
        ("openalex-arxiv", "doi:10.48550/arxiv.2401.01234")
    ]
    item = Item("Paper", None, "10.1/x", "https://doi.org/10.1/x")
    assert list(Resolver.identifiers(item)) == [("openalex-doi", "doi:10.1/x")]  # once
    # Nothing anywhere: unresolved.
    missing = resolver.resolve(Item(FAKE_PAPER["title"], 2023, FAKE_PAPER["doi"], None))
    assert missing == Resolution(UNRESOLVED)
    # Every lookup is cached, misses included.
    sent = len(world.requests)
    resolver.resolve(Item(FAKE_PAPER["title"], 2023, FAKE_PAPER["doi"], None))
    resolver.resolve(Item(TITLES[T2], None, NEIGHBOUR["doi"], None))
    assert len(world.requests) == sent
    resolver.close()
    conn.close()


def test_a_listed_paper_matches_the_original_by_id_doi_or_title() -> None:
    original = {"id": "W1", "aliases": ["W2"], "doi": "10.1/orig", "title": "The Original"}
    unresolved = Resolution(UNRESOLVED)
    assert match_item(Item("the original!", None, None, None), unresolved, original) == "title"
    assert match_item(Item("Other", None, "10.1/orig", None), unresolved, original) == "doi"
    resolved = Resolution(VERIFIED, "crossref-title", "W2", "10.9/x", "Other", 2025)
    assert match_item(Item("Other", None, None, None), resolved, original) == "alias"
    assert match_item(Item("Other", None, None, None), unresolved, original) is None


# --- the sample -------------------------------------------------------------------------------


def test_quotas_are_proportional_with_largest_remainders() -> None:
    sizes = {"1": 48, "2-10": 30, "11-100": 24, "not found": 18}  # the dev set's buckets
    assert quotas_for(sizes, 100) == {"1": 40, "2-10": 25, "11-100": 20, "not found": 15}
    assert quotas_for({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert quotas_for({"a": 3, "b": 1}, 10) == {"a": 3, "b": 1}  # never more than there are
    assert sum(quotas_for(sizes, 37).values()) == 37


def test_the_stratified_sample_is_seeded() -> None:
    buckets = ["1"] * 10 + ["2-10"] * 6 + ["11-100"] * 3 + ["not found"] * 1
    frame = [Query(f"A{i:04d}", f"W{i}", "p", b, Path("r")) for i, b in enumerate(buckets)]
    sample, quotas = stratified_sample(frame, 10, seed=42)
    assert quotas == {"1": 5, "2-10": 3, "11-100": 2, "not found": 0}  # a remainder tie
    assert [sum(q.bucket == b for q in sample) for b in quotas] == list(quotas.values())
    assert stratified_sample(frame, 10, seed=42)[0] == sample
    assert stratified_sample(frame, 10, seed=7)[0] != sample
    assert len(stratified_sample(frame, 50, seed=42)[0]) == 20


def test_holm() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([]) == []


# --- whole runs -------------------------------------------------------------------------------


@pytest.fixture
def live_run(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """The fixture benchmark, a Phase 1 run, and a live-eval run with L1, L5 and L5w."""
    data = _config_dict(tmp_path)
    config = config_from_dict(data)
    freeze(config.corpus.out_dir, corpus_records(), target_rows(), {})
    stats = make_bench(config, _paraphraser(), FakeLLM(lambda messages: judge_reply()))
    assert stats["accepted"] == 3
    _phase1_run(config, B1_RANKS, B5_RANKS)
    assert live_evaluate.L5W_FUSION == PROPOSAL_EQUAL
    assert config.live is not None
    search = LiveSearchClient(
        config.live,
        OpenAlexCache(config.live.cache_path),
        max_age=None,
        transport=FakeLiveOpenAlex().transport,
        sleep=no_sleep,
        clock=lambda: 0.0,
    )
    done = run_live_eval(
        config, agent(), FakeEncoder(), search, configs=["L1", "L5", "L5w"], log_mlflow=False
    )
    return data, done.run_dir


def compete_config(
    data: dict[str, Any], run_dir: Path, tmp_path: Path, systems: list[dict[str, Any]]
) -> Config:
    data = {
        **data,
        "compete": {
            "runs": [str(run_dir)],
            "cache_path": str(tmp_path / "compete.sqlite"),
            "n_queries": 3,
            "systems": systems,
        },
    }
    return config_from_dict(data)


def run(
    cfg: Config, world: FakeWorld, *, systems: list[str] | None = None, encoder: Any = None
) -> compete_eval.CompeteRun:
    assert cfg.compete is not None
    plan = make_plan(cfg, systems=systems)
    conn = connect_store(cfg.compete.cache_path)
    clients = {
        s.name: make_system(s, transport=world.transport, sleep=no_sleep)
        for s in cfg.compete.systems
        if s.name in plan.systems
    }
    resolver = Resolver(conn, transport=world.transport, sleep=no_sleep)
    try:
        return run_compete(
            cfg,
            plan,
            clients=clients,
            resolver=resolver,
            conn=conn,
            encoder=encoder or FakeEncoder(),
        )
    finally:
        conn.close()


def _results(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lines = (run_dir / RESULTS_FILE).read_text(encoding="utf-8").splitlines()
    return {(r["system"], r["target_id"]): r for r in map(json.loads, lines)}


def test_a_whole_run_scores_every_system(
    live_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    data, live_dir = live_run
    systems = [claude_dict(), openai_dict(), perplexity_dict(), s2_dict()]
    cfg = compete_config(data, live_dir, tmp_path, systems)
    world = FakeWorld()
    done = run(cfg, world)
    rows = _results(done.run_dir)
    assert len(rows) == 5 * 3
    for t in (T1, T2, T3):
        claude = rows[("claude", t)]
        assert (claude["target_rank"], claude["match"], claude["hit_at_1"]) == (2, "id", 0)
        assert claude["n_unresolved"] == 0 and claude["hit_at_5"] == 1
        assert claude["items"][1]["resolution"]["route"] == "crossref-title"
        assert claude["cost"] == pytest.approx(0.045) and not claude["cached"]
        openai = rows[("openai", t)]  # the original's title under the wrong DOI
        assert (openai["target_rank"], openai["match"]) == (1, "id")
        assert (openai["n_doi_mismatch"], openai["n_unresolved"]) == (1, 0)
        assert rows[("s2-search", t)]["n_items"] == 0
        ours = rows[(OURS, t)]
        assert ours["reproduced"] is True and ours["n_unresolved"] == 0
        assert ours["items"][0]["link"].startswith("https://openalex.org/W")
    assert rows[("perplexity", T1)]["n_unresolved"] == 1
    assert rows[("perplexity", T3)]["parse_error"] is not None
    assert rows[("perplexity", T3)]["n_items"] == 0

    summary = json.loads((done.run_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    q = summary["quality"]
    assert q["claude"]["hit_at_5"] == 1.0 and q["claude"]["hit_at_1"] == 0.0
    assert q["perplexity"]["n_parse_failures"] == 1
    assert q["perplexity"]["unresolved_share"] == 1.0 and q[OURS]["unresolved_share"] == 0.0
    assert q["openai"]["doi_mismatch_share"] == 1.0
    assert summary["frame"] == {"1": 1, "2-10": 1, "11-100": 0, "not found": 1}
    assert [p["b"] for p in summary["paired"]] == ["claude", "openai", "perplexity", "s2-search"]
    assert all("p_holm" in p["mcnemar"] for p in summary["paired"])
    assert summary["spend"]["claude"]["total"] == pytest.approx(3 * 0.045)
    assert summary["spend"]["perplexity"]["total"] == pytest.approx(2 * 0.006 + 0.006)
    assert summary["ours_replay"] == {"recorded_l5w": 3, "reproduced": 3, "capped_checks": 0}
    meta = json.loads((done.run_dir / RUN_FILE).read_text(encoding="utf-8"))
    assert meta["kind"] == "compete" and len(meta["sample"]) == 3
    assert meta["prompt"]["name"] == "compete"
    markdown = (done.run_dir / SUMMARY_MD).read_text(encoding="utf-8")
    assert "## Listed papers that don't resolve (possibly fabricated)" in markdown
    assert "| ours vs claude | 3 |" in markdown

    # A second run asks nothing that was asked before: no API calls, nothing spent.
    sent = len(world.requests)
    again = run(cfg, world)
    assert len(world.requests) == sent
    assert again.summary["quality"]["claude"] == {**q["claude"], "money_spent": 0.0}


def test_a_system_over_its_cap_gets_no_more_answers(
    live_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    data, live_dir = live_run
    cfg = compete_config(
        data, live_dir, tmp_path, [claude_dict(reserve_per_query=0.04, cost_cap=0.07)]
    )
    world = FakeWorld()
    done = run(cfg, world, systems=["claude"])
    rows = list(_results(done.run_dir).values())
    assert [r["status"] for r in rows] == ["ok", "capped", "capped"]
    assert "cost cap" in rows[1]["error"]
    assert done.summary["quality"]["claude"]["n_capped"] == 2
    assert done.summary["paired"] == []  # ours wasn't asked
    assert world.hosts()["api.anthropic.com"] == 1


def test_missing_keys_stop_the_run_before_any_call(
    live_run: tuple[dict[str, Any], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, live_dir = live_run
    cfg = compete_config(data, live_dir, tmp_path, [claude_dict(), s2_dict()])
    monkeypatch.delenv("TEST_ANTHROPIC_KEY")
    world = FakeWorld()
    with pytest.raises(EvalError, match="claude needs TEST_ANTHROPIC_KEY"):
        run(cfg, world)
    assert world.requests == []
    with pytest.raises(EvalError, match=r"unknown systems \['bing'\]"):
        make_plan(cfg, systems=["bing"])


def test_compete_cli_dry_run_and_run(
    live_run: tuple[dict[str, Any], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, live_dir = live_run
    cfg = compete_config(data, live_dir, tmp_path, [claude_dict(), s2_dict()])
    config_path = tmp_path / "config.yaml"
    raw = {**data, "compete": {
        "runs": [str(live_dir)],
        "cache_path": str(tmp_path / "compete.sqlite"),
        "n_queries": 3,
        "systems": [claude_dict(), s2_dict()],
    }}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    world = FakeWorld()

    def fake_system(config: CompeteSystemConfig) -> Any:
        return make_system(config, transport=world.transport, sleep=no_sleep)

    class FakeResolver:
        @staticmethod
        def from_config(config: Config, conn: Any) -> Resolver:
            return Resolver(conn, transport=world.transport, sleep=no_sleep)

    monkeypatch.setattr(cli, "make_system", fake_system)
    monkeypatch.setattr(cli, "Resolver", FakeResolver)
    monkeypatch.setattr(cli, "make_encoder", lambda retrieval_cfg: FakeEncoder())

    assert cli.main(["compete", "--config", str(config_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 queries from the 3 the runs checked" in out
    assert "claude: 3 of 3 calls not cached, at most $1.50" in out
    assert "s2-search: 3 of 3 calls not cached; free" in out
    assert world.requests == []

    args = ["compete", "--config", str(config_path), "--limit", "2", "--systems", "ours,claude"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "rk compete: claude 2/2" in out and "Hit@5: ours" in out
    assert world.hosts()["api.anthropic.com"] == 2
    assert cfg.compete is not None and len(list(cfg.run.out_dir.glob("compete-*"))) == 1

    assert cli.main(["compete", "--config", str(config_path), "--systems", "bing"]) == 1
    assert "unknown systems ['bing']" in capsys.readouterr().err
