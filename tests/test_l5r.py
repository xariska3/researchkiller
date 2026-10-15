"""The default check, L5r, built into the live pipeline and live-eval.

L5r is L5w's check reranked by one listwise LLM call (rk.live.pipeline.with_rerank,
rk.live.reranker): Step 4's R1. The checks run on FakeLiveOpenAlex with test_live_eval's
FakeLLM agent and FakeEncoder; the reranker is a FakeLLM.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from bench_data import GOOD, T1, T2, T3, judge_reply, target_rows
from conftest import FakeEncoder, FakeLLM
from live_fakes import FakeLiveOpenAlex
from test_live_eval import (
    B1_RANKS,
    B5_RANKS,
    CHOSEN_L5W_FUSION,
    _config_dict,
    _paraphraser,
    _phase1_run,
    _qids,
    _rows,
    _search,
    _summary,
    agent,
)
from test_rerank import dated_corpus, numbered, original_first, rerank_dict

from rk import cli
from rk.bench import make_bench
from rk.config import Config, config_from_dict
from rk.corpus import freeze
from rk.evaluate import RUN_FILE, SUMMARY_MD, EvalError
from rk.live import rerank as live_rerank
from rk.live.evaluate import rerank_settings, run_live_eval
from rk.live.fusion import B5, PROPOSAL_EQUAL, final_ranking
from rk.live.openalex_search import LiveSearchClient, OpenAlexCache
from rk.live.pipeline import (
    DEFAULT_FUSION,
    FALLBACK,
    RERANKED,
    SKIPPED,
    LiveSearchResult,
    default_check,
    live_search,
    with_rerank,
)
from rk.live.rerank import PRODUCT_MODEL, R1, RESULTS_FILE, run_rerank
from rk.live.reranker import LISTWISE, candidates, render_prompt
from rk.llm import LLMError, load_prompt


def test_the_default_check_is_l5w_then_r1() -> None:
    assert DEFAULT_FUSION == CHOSEN_L5W_FUSION == PROPOSAL_EQUAL
    assert live_rerank.RERANK_CHOICE == R1
    assert live_rerank.CALLS[R1] == (LISTWISE, PRODUCT_MODEL)


# --- the rerank step in the pipeline ---------------------------------------------------------


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return config_from_dict({**_config_dict(tmp_path), "rerank": rerank_dict(tmp_path)})


def l5w_check(cfg: Config, fake: FakeLiveOpenAlex | None = None) -> LiveSearchResult:
    """A check of T1's proposal, its pool ranked as L5w ranks it."""
    search = _search(cfg, fake or FakeLiveOpenAlex())
    return live_search(
        GOOD[T1], cfg, search=search, encoder=FakeEncoder(), agent=agent(), fusion=DEFAULT_FUSION
    )


def reverse(messages: list[dict[str, str]]) -> str:
    """A listwise reply: every candidate number, last first."""
    return json.dumps({"ranking": sorted(numbered(messages[-1]["content"]).values(), reverse=True)})


def down(messages: list[dict[str, str]]) -> str:
    raise LLMError("upstream 503")


def ids(result: LiveSearchResult) -> list[str]:
    return [c.record["id"] for c in result.candidates]


def test_l5r_reranks_the_union_of_l5w_and_l5_and_l5w_follows(cfg: Config) -> None:
    assert cfg.rerank is not None and cfg.live is not None
    l5w = l5w_check(cfg)
    agent_calls = l5w.ledger.llm.calls
    reranker = FakeLLM(reverse)
    l5r = with_rerank(l5w, cfg, reranker, FakeEncoder())
    # R1's prompt, over the union of L5w's and L5's top `depth` in presented order.
    pool, queries = l5w.pool, l5w.trace.queries
    l5w_order = final_ranking(pool, queries, DEFAULT_FUSION)
    cands = candidates(l5w_order, final_ranking(pool, queries, B5), cfg.rerank.depth)
    by_id = {r["id"]: r for r in pool.records}
    records = [by_id[d] for d in cands]
    (call,) = reranker.calls
    assert call == [
        {
            "role": "user",
            "content": render_prompt(
                load_prompt(LISTWISE), GOOD[T1], records, cfg.rerank.abstract_chars
            ),
        }
    ]
    expected = list(dict.fromkeys([*reversed(cands), *l5w_order]))[: cfg.live.top_k]
    assert len(cands) > 1 and ids(l5r) == expected != ids(l5w)
    assert [c.rank for c in l5r.candidates] == list(range(1, len(expected) + 1))
    step = l5r.rerank
    assert step is not None
    assert (step.status, step.n_candidates, step.warning) == (RERANKED, len(cands), None)
    assert l5r.warnings == ()
    # The call is charged to the check, and L5w's ledger is left as it was.
    assert l5r.ledger.llm.calls == agent_calls + 1 and l5w.ledger.llm.calls == agent_calls
    assert l5r.ledger.method_cost == pytest.approx(
        l5w.ledger.method_cost + step.usage.method_cost
    )
    assert l5r.latency_s >= l5w.latency_s
    trace = l5r.to_dict()
    assert trace["rerank"]["status"] == RERANKED and trace["warnings"] == []


def test_the_default_check_is_l5w_search_then_the_rerank(cfg: Config) -> None:
    direct = with_rerank(l5w_check(cfg), cfg, FakeLLM(reverse), FakeEncoder())
    result = default_check(
        GOOD[T1],
        cfg,
        search=_search(cfg, FakeLiveOpenAlex()),
        encoder=FakeEncoder(),
        agent=agent(),
        reranker=FakeLLM(reverse),
    )
    assert result.fusion == DEFAULT_FUSION and result.rerank is not None
    assert result.rerank.status == RERANKED and ids(result) == ids(direct)


def test_an_unreadable_reply_keeps_l5w_order_with_a_warning(cfg: Config) -> None:
    l5w = l5w_check(cfg)
    l5r = with_rerank(l5w, cfg, FakeLLM(["Sorry, I can't rank these."]), FakeEncoder())
    assert ids(l5r) == ids(l5w)
    assert l5r.rerank is not None and l5r.rerank.status == FALLBACK
    (warning,) = l5r.warnings
    assert warning == (
        "Not reranked (the rerank reply had no readable ranking); the results keep the "
        "check's order."
    )
    assert l5r.rerank.warning == warning and l5r.to_dict()["warnings"] == [warning]
    assert l5r.ledger.llm.calls == l5w.ledger.llm.calls + 1  # the call was made and charged


def test_a_failed_call_falls_back_in_the_tool_and_raises_in_evaluation(cfg: Config) -> None:
    l5w = l5w_check(cfg)
    l5r = with_rerank(l5w, cfg, FakeLLM(down), FakeEncoder())
    assert ids(l5r) == ids(l5w)
    assert l5r.rerank is not None and l5r.rerank.status == FALLBACK
    assert "the rerank call failed: upstream 503" in l5r.warnings[0]
    assert l5r.ledger.llm.calls == l5w.ledger.llm.calls  # nothing answered, nothing charged
    with pytest.raises(LLMError, match="upstream 503"):
        with_rerank(l5w, cfg, FakeLLM(down), FakeEncoder(), strict=True)


def test_a_rerank_call_that_could_cross_the_cost_cap_is_not_made(cfg: Config) -> None:
    l5w = l5w_check(cfg)
    full = replace(l5w, ledger=replace(l5w.ledger, cap=l5w.ledger.method_cost))  # no room
    reranker = FakeLLM(reverse)
    l5r = with_rerank(full, cfg, reranker, FakeEncoder(), strict=True)
    assert reranker.calls == [] and ids(l5r) == ids(l5w)
    assert l5r.rerank is not None and l5r.rerank.status == FALLBACK
    assert "the rerank call could cost $" in l5r.warnings[0]


def test_a_check_without_candidates_skips_the_call(cfg: Config) -> None:
    empty = l5w_check(cfg, FakeLiveOpenAlex([]))
    reranker = FakeLLM([])
    l5r = with_rerank(empty, cfg, reranker, FakeEncoder())
    assert reranker.calls == [] and l5r.candidates == [] and l5r.warnings == ()
    assert l5r.rerank is not None and l5r.rerank.status == SKIPPED


def test_the_default_check_needs_a_rerank_section(tmp_path: Path) -> None:
    plain = config_from_dict(_config_dict(tmp_path))
    fake = FakeLiveOpenAlex()
    with pytest.raises(ValueError, match="rerank"):
        default_check(
            GOOD[T1],
            plain,
            search=_search(plain, fake),
            encoder=FakeEncoder(),
            agent=agent(),
            reranker=FakeLLM([]),
        )
    assert fake.requests == []  # refused before any search


# --- L5r in live-eval ----------------------------------------------------------------------


@pytest.fixture
def bench(tmp_path: Path) -> Config:
    """The fixture benchmark (dated, as the frozen corpus is) and a Phase 1 run."""
    config = config_from_dict({**_config_dict(tmp_path), "rerank": rerank_dict(tmp_path)})
    freeze(config.corpus.out_dir, dated_corpus(), target_rows(), {})
    stats = make_bench(config, _paraphraser(), FakeLLM(lambda messages: judge_reply()))
    assert stats["accepted"] == 3
    _phase1_run(config, B1_RANKS, B5_RANKS)
    return config


def live_eval(cfg: Config, reranker: FakeLLM | None, **kwargs: Any) -> Any:
    kwargs.setdefault("log_mlflow", False)
    search = _search(cfg, FakeLiveOpenAlex())
    return run_live_eval(cfg, agent(), FakeEncoder(), search, reranker=reranker, **kwargs)


def test_live_l5r_rows_are_the_offline_r1(bench: Config) -> None:
    live_reranker = FakeLLM(original_first)
    done = live_eval(bench, live_reranker, configs=["L5", "L5w", "L5r"])
    rows = _rows(bench, done.run_dir)
    offline_reranker = FakeLLM(original_first)
    replayed = run_rerank(
        bench,
        done.run_dir,
        encoder=FakeEncoder(),
        clients={PRODUCT_MODEL: offline_reranker},
        variants=[R1],
    )
    lines = (replayed.run_dir / RESULTS_FILE).read_text(encoding="utf-8").splitlines()
    r1 = {r["target_id"]: r for r in map(json.loads, lines) if r["config"] == R1}
    for target in (T1, T2, T3):
        l5r, l5w, l5 = rows[("L5r", target)], rows[("L5w", target)], rows[("L5", target)]
        assert l5r["status"] == "ok"
        assert json.loads(l5r["doc_ids"]) == r1[target]["doc_ids"]
        assert l5r["target_rank"] == r1[target]["target_rank"]
        assert l5r["llm_calls"] == l5w["llm_calls"] + 1
        assert l5r["method_cost"] == pytest.approx(l5w["method_cost"] + r1[target]["cost"])
        assert l5r["latency_s"] >= l5["latency_s"]
    assert any(rows[("L5r", t)]["target_rank"] == 1 for t in (T1, T2, T3))
    # The same requests: the tool's calls and R1's share their cache entries.
    prompts = sorted(c[-1]["content"] for c in live_reranker.calls)
    assert prompts == sorted(c[-1]["content"] for c in offline_reranker.calls)
    summary = _summary(done.run_dir)
    assert summary["rerank_steps"] == {"reranked": 3, "fallback": 0, "skipped": 0, "warnings": []}
    assert ("L5r", "L5w", 3) in [(p["a"], p["b"], p["n"]) for p in summary["paired_secondary"]]
    markdown = (done.run_dir / SUMMARY_MD).read_text(encoding="utf-8")
    assert "- L5r: the default check" in markdown
    assert "## L5r's rerank step" in markdown and "Reranked: 3." in markdown


def test_live_eval_counts_fallbacks_and_failed_calls_are_error_rows(bench: Config) -> None:
    def reply(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        if GOOD[T2] in prompt:
            raise LLMError("upstream 503")
        if GOOD[T3] in prompt:
            return "no ranking today"
        return original_first(messages)

    done = live_eval(bench, FakeLLM(reply), configs=["L5w", "L5r"])
    rows = _rows(bench, done.run_dir)
    failed = rows[("L5r", T2)]
    assert failed["status"] == "error" and "upstream 503" in failed["error"]
    assert rows[("L5w", T2)]["status"] == "ok"
    kept = rows[("L5r", T3)]
    assert kept["status"] == "ok" and kept["doc_ids"] == rows[("L5w", T3)]["doc_ids"]
    summary = _summary(done.run_dir)
    steps = summary["rerank_steps"]
    assert (steps["reranked"], steps["fallback"], steps["skipped"]) == (1, 1, 0)
    (warning,) = steps["warnings"]
    assert warning.startswith(f"{_qids(bench)[T3]}: Not reranked (the rerank reply had no")
    assert summary["errors"]["L5r"] == 1
    assert f"- {warning}" in (done.run_dir / SUMMARY_MD).read_text(encoding="utf-8")
    # The error row is retried like any other.
    again = live_eval(
        bench,
        FakeLLM(original_first),
        configs=["L5w", "L5r"],
        run_dir=done.run_dir,
        retry_errors=True,
    )
    assert _rows(bench, again.run_dir)[("L5r", T2)]["status"] == "ok"


def test_l5r_runs_only_with_a_reranker(bench: Config) -> None:
    with pytest.raises(EvalError, match="L5r needs"):
        live_eval(bench, None, configs=["L5r"])
    done = live_eval(bench, None)  # without a reranker, L5r is left out
    assert {config for config, _ in _rows(bench, done.run_dir)} == {"L0", "L1", "L5", "L5w"}
    assert _summary(done.run_dir)["rerank_steps"] is None


def test_l5r_rows_can_join_a_run_from_before_l5r(bench: Config) -> None:
    done = live_eval(bench, None, configs=["L5", "L5w"])
    meta_path = done.run_dir / RUN_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["rerank"]  # as a run started before L5r existed
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    added = live_eval(bench, FakeLLM(original_first), configs=["L5r"], run_dir=done.run_dir)
    assert added.run_dir == done.run_dir
    configs = [config for config, _ in _rows(bench, done.run_dir)]
    assert sorted(set(configs)) == ["L5", "L5r", "L5w"] and configs.count("L5r") == 3
    assert json.loads(meta_path.read_text(encoding="utf-8"))["rerank"] == rerank_settings(bench)
    # From then on, the run's L5r rows pin its rerank settings.
    assert bench.rerank is not None
    deeper = replace(bench, rerank=replace(bench.rerank, depth=5))
    with pytest.raises(EvalError, match="rerank changed"):
        live_eval(deeper, FakeLLM(original_first), configs=["L5r"], run_dir=done.run_dir)


# --- the command ---------------------------------------------------------------------------


def test_the_live_eval_command_gives_l5r_the_rerank_model(
    bench: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config_path = tmp_path / "config.yaml"
    data = {**_config_dict(tmp_path), "rerank": rerank_dict(tmp_path)}
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    fake = FakeLiveOpenAlex(budget_credits=10_000)

    class Search:
        @staticmethod
        def from_config(config: Config, cache: OpenAlexCache, *, evaluation: bool) -> Any:
            assert evaluation and config.live is not None
            return LiveSearchClient(
                config.live,
                cache,
                max_age=None,
                transport=fake.transport,
                sleep=lambda seconds: None,
                clock=lambda: 0.0,
            )

    reranker = FakeLLM(original_first)
    models: list[str] = []

    def client(model: Any, cache: Any, ledger: Any) -> FakeLLM:
        models.append(model.name)
        return reranker if model.name == "fake/reranker" else agent()

    monkeypatch.setattr(cli, "LiveSearchClient", Search)
    monkeypatch.setattr(cli, "LLMClient", client)
    monkeypatch.setattr(cli, "make_encoder", lambda retrieval_cfg: FakeEncoder())
    argv = ["live-eval", "--config", str(config_path), "--configs", "L5w,L5r"]
    assert cli.main(argv) == 0
    assert "3 of 3 queries done" in capsys.readouterr().out
    assert models == [bench.agent.name, "fake/reranker"] and len(reranker.calls) == 3
