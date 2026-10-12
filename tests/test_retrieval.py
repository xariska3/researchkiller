from __future__ import annotations

import itertools
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from bench_data import ABSTRACTS, T1, T2, TITLES, corpus_records, target_rows
from conftest import FakeEncoder, FakeLLM, corpus_config_dict

from rk import cli, retrieval
from rk.agent import AgentRetriever, make_retrievers, parse_action
from rk.config import Config, config_from_dict
from rk.corpus import freeze
from rk.llm import load_prompt
from rk.retrieval import (
    INDEX_MANIFEST,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Index,
    RetrievalError,
    RewriteRetriever,
    analyze,
    build_index,
    fuse_runs,
    load_index,
    rrf,
)
from rk.text import content_stems

QUERY = "Condense very lengthy books by comparing each word only with a few learned groups."


def _config_dict(tmp_path: Path) -> dict[str, Any]:
    data = corpus_config_dict(tmp_path / "corpus")
    data["retrieval"]["index_dir"] = str(tmp_path / "index")
    return data


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    config = config_from_dict(_config_dict(tmp_path))
    freeze(config.corpus.out_dir, corpus_records(), target_rows(), {})
    return config


@pytest.fixture
def index(cfg: Config) -> Index:
    build_index(cfg, FakeEncoder())
    return load_index(cfg)


@pytest.fixture
def hybrid(index: Index) -> HybridRetriever:
    return HybridRetriever(BM25Retriever(index), DenseRetriever(index, FakeEncoder()))


def _agent(
    hybrid: Any, replies: Any, *, max_queries: int, max_calls: int
) -> tuple[AgentRetriever, FakeLLM]:
    llm = FakeLLM(replies)
    retriever = AgentRetriever(
        hybrid,
        llm,
        titles={r["id"]: r["title"] for r in corpus_records()},
        max_queries=max_queries,
        max_llm_calls=max_calls,
        prompt=load_prompt("agent_step"),
    )
    return retriever, llm


def _search(query: str) -> str:
    return json.dumps({"action": "search", "query": query})


class _CountingHybrid:
    """Wraps a HybridRetriever and records every query it ranks."""

    def __init__(self, inner: HybridRetriever) -> None:
        self.inner = inner
        self.ranked: list[str] = []

    def rank(self, query: str) -> list[str]:
        self.ranked.append(query)
        return self.inner.rank(query)


# --- indexes -----------------------------------------------------------------


def test_index_manifest_and_shared_tokenisation(cfg: Config, index: Index) -> None:
    manifest = json.loads((cfg.retrieval.index_dir / INDEX_MANIFEST).read_text(encoding="utf-8"))
    corpus_manifest = json.loads((cfg.corpus.out_dir / "manifest.json").read_text())
    assert manifest["corpus_sha256"] == corpus_manifest["files"]["corpus.jsonl"]["sha256"]
    assert (manifest["encoder"], manifest["device"], manifest["n_docs"]) == (
        "fake-encoder",
        "cpu",
        13,
    )
    assert manifest["doc_text"] == "title. abstract"
    assert index.doc_ids == sorted(index.doc_ids, key=lambda d: int(d[1:]))
    assert analyze(TITLES[T1]) == content_stems(TITLES[T1])  # same tokenisation as the leak filter


def test_index_refuses_a_corpus_sha_mismatch(cfg: Config, index: Index) -> None:
    shutil.rmtree(cfg.corpus.out_dir)  # rebuild the corpus with one more document
    extra = {"id": "W999", "aliases": [], "title": "New", "abstract": "A new document."}
    freeze(cfg.corpus.out_dir, [*corpus_records(), extra], target_rows(), {})
    with pytest.raises(RetrievalError, match="built for corpus sha256 .* rebuild the index"):
        load_index(cfg)


def test_index_is_never_overwritten(cfg: Config, index: Index) -> None:
    with pytest.raises(RetrievalError, match="refusing to overwrite"):
        build_index(cfg, FakeEncoder())


def test_dense_retriever_refuses_a_different_encoder(index: Index) -> None:
    other = FakeEncoder()
    other.name = "another-encoder"
    with pytest.raises(RetrievalError, match="built with encoder 'fake-encoder'"):
        DenseRetriever(index, other)


# --- B1-B3 -------------------------------------------------------------------


def test_bm25_finds_the_doc_with_an_exact_rare_term(index: Index) -> None:
    result = BM25Retriever(index).retrieve("attention that grows quadratically")
    assert result.doc_ids[0] == T1
    assert result.trace.queries == ["attention that grows quadratically"]
    assert result.trace.usage.calls == 0
    assert BM25Retriever(index).retrieve("the of and").doc_ids == []  # stopwords only


def test_dense_ranks_every_doc_and_hybrid_is_rrf_of_both(index: Index) -> None:
    bm25, dense = BM25Retriever(index), DenseRetriever(index, FakeEncoder())
    query = "reward models overconfident on human comparisons"
    assert len(dense.rank(query)) == 13
    assert dense.rank(query)[0] == T2
    hybrid = HybridRetriever(bm25, dense)
    assert hybrid.retrieve(query).doc_ids == rrf([bm25.rank(query), dense.rank(query)])


def test_rrf_matches_a_hand_computed_example() -> None:
    # W1: 1/61 + 1/62 = 0.03252, W3: 1/63 + 1/61 = 0.03227, W2: 1/62 = 0.01613, W4: 1/63.
    assert rrf([["W1", "W2", "W3"], ["W3", "W1", "W4"]]) == ["W1", "W3", "W2", "W4"]
    # Equal scores: lower doc id first, whatever the list order.
    assert rrf([["W9", "W2"], ["W2", "W9"]]) == ["W2", "W9"]
    assert rrf([["W2", "W9"], ["W9", "W2"]]) == ["W2", "W9"]
    assert len(rrf([[f"W{n}" for n in range(1, 151)]])) == 100


# --- B4 ----------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["Sure, here are some queries.", '{"query": "one"}'])
def test_b4_falls_back_to_the_original_query_on_a_parse_failure(
    reply: str, hybrid: HybridRetriever
) -> None:
    llm = FakeLLM([reply])
    b4 = RewriteRetriever(hybrid, llm, max_queries=3, prompt=load_prompt("rewrite"))
    result = b4.retrieve(QUERY)
    assert result.trace.queries == [QUERY]
    assert result.trace.errors[0].startswith("rewrite_parse_error")
    assert result.trace.usage.calls == 1
    assert result.doc_ids == hybrid.rank(QUERY)


def test_b4_caps_and_deduplicates_rewrites(hybrid: HybridRetriever) -> None:
    reply = json.dumps(
        {"queries": [" Fast  attention ", "fast attention", QUERY, "", 7, "memory saving", "x"]}
    )
    llm = FakeLLM([reply])
    b4 = RewriteRetriever(hybrid, llm, max_queries=3, prompt=load_prompt("rewrite"))

    result = b4.retrieve(QUERY)

    assert result.trace.queries == [QUERY, "Fast attention", "memory saving"]
    assert result.trace.budget_used == 3
    assert "Write up to 2 search queries" in llm.calls[0][-1]["content"]
    assert result.trace.prompt_hashes == {"rewrite": load_prompt("rewrite").sha256}
    assert result.doc_ids == fuse_runs([hybrid.rank(q) for q in result.trace.queries])


# --- B5 ----------------------------------------------------------------------


def test_b5_runs_the_original_first_and_honours_stop(hybrid: HybridRetriever) -> None:
    stop = '{"action": "stop", "reason": "covered"}'
    b5, _ = _agent(hybrid, [stop], max_queries=5, max_calls=6)
    result = b5.retrieve(QUERY)
    assert result.trace.queries == [QUERY]
    assert result.trace.usage.calls == 1
    assert result.trace.stop_reason == "agent_stop"
    assert result.trace.steps == [{"action": "stop", "reason": "covered"}]
    assert result.doc_ids == fuse_runs([hybrid.rank(QUERY)])


def test_b5_never_exceeds_max_queries(hybrid: HybridRetriever) -> None:
    counter = itertools.count(1)
    b5, llm = _agent(
        hybrid, lambda msgs: _search(f"new query {next(counter)}"), max_queries=3, max_calls=10
    )
    result = b5.retrieve(QUERY)
    assert result.trace.budget_used == len(result.trace.queries) == 3
    assert len(llm.calls) == result.trace.usage.calls == 2  # the original needs no call
    assert result.trace.stop_reason == "max_queries"


def test_b5_never_exceeds_the_llm_call_cap_and_repeats_are_not_rerun(
    hybrid: HybridRetriever,
) -> None:
    counting = _CountingHybrid(hybrid)
    b5, llm = _agent(
        counting, lambda msgs: _search(f"  {QUERY.upper()} "), max_queries=10, max_calls=3
    )
    result = b5.retrieve(QUERY)
    assert len(llm.calls) == 3
    assert result.trace.stop_reason == "max_llm_calls"
    assert counting.ranked == [QUERY]  # repeats aren't run again...
    assert result.trace.budget_used == 4  # ...but count against the budget
    assert [s["repeat"] for s in result.trace.steps] == [True, True, True]


def test_b5_repeats_can_exhaust_the_query_budget(hybrid: HybridRetriever) -> None:
    b5, llm = _agent(hybrid, lambda msgs: _search(QUERY), max_queries=2, max_calls=10)
    result = b5.retrieve(QUERY)
    assert (len(llm.calls), result.trace.budget_used, result.trace.queries) == (1, 2, [QUERY])
    assert result.trace.stop_reason == "max_queries"


def test_b5_parse_failure_stops_and_is_recorded(hybrid: HybridRetriever) -> None:
    b5, _ = _agent(hybrid, ["Let me search for transformers next."], max_queries=5, max_calls=6)
    result = b5.retrieve(QUERY)
    assert result.trace.stop_reason == "parse_error"
    assert result.trace.errors[0].startswith("agent_parse_error")
    assert result.trace.queries == [QUERY]


def test_b5_sees_only_the_top_ten_titles_of_each_query(hybrid: HybridRetriever) -> None:
    replies = [_search("reward model calibration"), '{"action": "stop"}']
    b5, llm = _agent(hybrid, replies, max_queries=5, max_calls=6)
    b5.retrieve(QUERY)
    second_prompt = llm.calls[1][-1]["content"]
    history = second_prompt.split("Searches so far")[1].split("You can run")[0]
    assert history.count("   - ") == 20  # two queries, ten titles each (13 docs in the corpus)
    titles = {r["id"]: r["title"] for r in corpus_records()}
    assert f"   - {titles[hybrid.rank(QUERY)[0]]}" in history
    assert all(abstract[:60] not in second_prompt for abstract in ABSTRACTS.values())


def test_parse_action() -> None:
    assert parse_action('{"action": "Search", "query": " a  b "}') == {
        "action": "search",
        "query": "a b",
    }
    assert parse_action('```json\n{"action": "stop"}\n```') == {"action": "stop", "reason": ""}
    for bad in ('{"action": "search"}', '{"action": "browse"}', "stop", "[1, 2]"):
        with pytest.raises((ValueError, TypeError)):
            parse_action(bad)


# --- shared fusion -----------------------------------------------------------


def test_b4_and_b5_share_the_fusion_function(
    hybrid: HybridRetriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    second = "memory efficient condensing of long texts"
    rewrites = FakeLLM([json.dumps({"queries": [second]})])
    b4 = RewriteRetriever(hybrid, rewrites, max_queries=2, prompt=load_prompt("rewrite"))
    b5, _ = _agent(hybrid, [_search(second)], max_queries=2, max_calls=5)
    fused: list[Sequence[Sequence[str]]] = []
    real = retrieval.fuse_runs

    def spy(runs: Sequence[Sequence[str]]) -> list[str]:
        fused.append(runs)
        return real(runs)

    monkeypatch.setattr(retrieval, "fuse_runs", spy)

    b4_result, b5_result = b4.retrieve(QUERY), b5.retrieve(QUERY)

    assert len(fused) == 2  # each called the one shared function...
    assert fused[0] == fused[1]  # ...on the same per-query lists...
    assert b4_result.doc_ids == b5_result.doc_ids  # ...so the same queries rank the same


# --- wiring, prompts, CLI ----------------------------------------------------


def test_make_retrievers_wires_all_six_with_config_budgets(cfg: Config, index: Index) -> None:
    retrievers = make_retrievers(cfg, index, FakeEncoder(), FakeLLM([]))
    assert list(retrievers) == ["B1", "B2", "B3", "B4", "B5", "B5-U"]
    r = cfg.retrieval
    b4: Any = retrievers["B4"]
    b5: Any = retrievers["B5"]
    b5u: Any = retrievers["B5-U"]
    assert b4.max_queries == r.max_queries
    assert (b5.max_queries, b5.max_llm_calls) == (r.max_queries, r.max_llm_calls_agent)
    assert (b5u.max_queries, b5u.max_llm_calls) == (
        r.max_queries_unconstrained,
        r.max_llm_calls_agent_unconstrained,
    )
    assert {name: ret.name for name, ret in retrievers.items()} == {n: n for n in retrievers}


@pytest.mark.parametrize("name", ["rewrite", "agent_step"])
def test_agent_prompts_frame_the_task_as_finding_prior_work(name: str) -> None:
    text = load_prompt(name).template.template.lower()
    assert "research proposal" in text and "published" in text
    assert "benchmark" not in text and "target" not in text


def test_cli_build_index(
    tmp_path: Path,
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_dict(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(cli, "make_encoder", lambda retrieval_cfg: FakeEncoder())

    assert cli.main(["build-index", "--config", str(config_path)]) == 0

    out = capsys.readouterr().out
    assert "indexed 13 documents" in out and "fake-encoder on cpu" in out
    assert load_index(cfg).doc_ids[0] == "W101"
