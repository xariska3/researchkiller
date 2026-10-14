"""Phase 2 live evaluation (python -m rk live-eval): L0, L1, L5, budget stops, the summary.

OpenAlex is FakeLiveOpenAlex behind httpx.MockTransport, the agent a FakeLLM,
the encoder FakeEncoder; the Phase 1 run compared with is written by hand.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from bench_data import GOOD, T1, T2, T3, TITLES, corpus_records, judge_reply, target_rows
from conftest import FakeEncoder, FakeLLM, corpus_config_dict
from live_fakes import FakeLiveOpenAlex

from rk import cli
from rk.bench import make_bench
from rk.config import Config, config_from_dict
from rk.corpus import freeze
from rk.evaluate import PER_QUERY_DIR, RUN_FILE, SUMMARY_JSON, SUMMARY_MD, EvalError
from rk.evaluate import connect as connect_results
from rk.live.evaluate import (
    KIND,
    PHASE1_B5,
    TABLE,
    cut_row,
    first_found,
    original_rank,
    run_live_eval,
)
from rk.live.openalex_search import KEYWORD, SEMANTIC, LiveSearchClient, OpenAlexCache
from rk.live.openalex_search import SearchResponse as Response
from rk.live.pool import CandidatePool
from rk.llm import LLMError

PHASE1_RUN = "20261013-000000"
AGENT_QUERY = "learned clusters attention memory"  # every content stem is in T1's text


def _config_dict(tmp_path: Path, **live: Any) -> dict[str, Any]:
    data = corpus_config_dict(tmp_path / "corpus")
    data["bench"].update(n_targets=3, out_dir=str(tmp_path / "bench"))
    data["run"].update(
        out_dir=str(tmp_path / "runs"),
        results_db=str(tmp_path / "results.sqlite"),
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        mlflow_experiment="test-phase1",
    )
    data["live"] = {
        "cache_path": str(tmp_path / "openalex.sqlite"),
        "top_k": 10,
        "explain_top_k": 3,
        "phase1_run": PHASE1_RUN,
        "mlflow_experiment": "test-live",
        **live,
    }
    return data


def _paraphraser() -> FakeLLM:
    def reply(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        return next(GOOD[t] for t, title in TITLES.items() if f"Title: {title}" in prompt)

    return FakeLLM(reply)


def _qids(cfg: Config) -> dict[str, str]:
    """Target id -> qid of the benchmark's three queries."""
    lines = (cfg.bench.out_dir / "test_a.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["target_id"]: row["qid"] for row in map(json.loads, lines)}


def _phase1_run(
    cfg: Config,
    b1: dict[str, int | None],
    b5: dict[str, int | None],
    *,
    run_id: str = PHASE1_RUN,
    bench_sha: str | None = None,
) -> None:
    """A Phase 1 run with B1 and B5 rows at these ranks (target id -> rank)."""
    bench = (cfg.bench.out_dir / "test_a.jsonl").read_bytes()
    qids = _qids(cfg)
    with closing(connect_results(cfg.run.results_db)) as conn:
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                "runs/phase1",
                "2026-10-13T00:00:00+00:00",
                None,
                None,
                "0" * 64,
                bench_sha or hashlib.sha256(bench).hexdigest(),
                "{}",
            ),
        )
        for config, ranks in (("B1", b1), ("B5", b5)):
            for target, rank in ranks.items():
                hits = [int(rank is not None and rank <= k) for k in (1, 5, 10)]
                conn.execute(
                    "INSERT INTO query_results (run_id, config, qid, target_id, status, "
                    "target_rank, hit_at_1, hit_at_5, hit_at_10, reciprocal_rank, finished_at) "
                    "VALUES (?, ?, ?, ?, 'ok', ?, ?, ?, ?, ?, '2026-10-13T00:00:00+00:00')",
                    (run_id, config, qids[target], target, rank, *hits, 1 / rank if rank else 0.0),
                )
        conn.commit()


B1_RANKS = {T1: 1, T2: 4, T3: None}
B5_RANKS = {T1: 1, T2: 15, T3: 3}  # 15 is beyond top_k = 10


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Frozen fixture corpus, a FakeLLM benchmark of three queries, a Phase 1 run."""
    config = config_from_dict(_config_dict(tmp_path))
    freeze(config.corpus.out_dir, corpus_records(), target_rows(), {})
    stats = make_bench(config, _paraphraser(), FakeLLM(lambda messages: judge_reply()))
    assert stats["accepted"] == 3
    _phase1_run(config, B1_RANKS, B5_RANKS)
    return config


def agent(*, failing_proposal: str | None = None) -> FakeLLM:
    """B5's agent: one search (AGENT_QUERY), then stop. Raises for one proposal if asked."""

    def reply(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        if failing_proposal and failing_proposal in prompt:
            raise LLMError("simulated outage")
        if '\n2. "' in prompt:
            return json.dumps({"action": "stop", "reason": "covered"})
        return json.dumps({"action": "search", "query": AGENT_QUERY})

    return FakeLLM(reply)


def _search(cfg: Config, fake: FakeLiveOpenAlex) -> LiveSearchClient:
    assert cfg.live is not None
    return LiveSearchClient(
        cfg.live,
        OpenAlexCache(cfg.live.cache_path),
        max_age=None,  # an evaluation's cache never expires
        transport=fake.transport,
        sleep=lambda seconds: None,
        clock=lambda: 0.0,
    )


def _run(cfg: Config, fake: FakeLiveOpenAlex, llm: FakeLLM | None = None, **kwargs: Any) -> Any:
    kwargs.setdefault("log_mlflow", False)
    return run_live_eval(cfg, llm or agent(), FakeEncoder(), _search(cfg, fake), **kwargs)


def _rows(cfg: Config, run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    run_id = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))["run_id"]
    with closing(sqlite3.connect(cfg.run.results_db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {TABLE} WHERE run_id = ?", (run_id,))
        return {(r["config"], r["target_id"]): dict(r) for r in rows}


def _summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / SUMMARY_JSON).read_text(encoding="utf-8"))


def _semantic_calls(fake: FakeLiveOpenAlex, text: str) -> int:
    return sum(r.get("search.semantic") == text for r in fake.requests)


# --- ranks and first hits --------------------------------------------------------


def _record(work_id: str, **extra: Any) -> dict[str, Any]:
    return {"id": work_id, "aliases": [], "doi": None, "title": f"Title {work_id}", **extra}


def _response(kind: str, *ids: str) -> Response:
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    return Response("q", kind, [_record(i) for i in ids], len(ids), 0.001, False, now, {})


def test_ranks_count_within_the_top_k_and_any_version_matches() -> None:
    original = _record("W1", doi="10.1/x")
    ranked = [[_record("W5")], [_record("W6"), _record("W7", doi="10.1/x")], [_record("W1")]]
    assert original_rank(ranked, original, 10) == (2, "doi")  # a merged version's DOI
    assert original_rank(ranked, original, 1) == (None, None)


def test_the_first_search_to_return_the_original_is_the_earliest_run() -> None:
    original = _record("W1")
    searches = [
        (0, _response(SEMANTIC, "W5", "W6")),
        (1, _response(KEYWORD, "W5", "W6", "W1")),
        (1, _response(SEMANTIC, "W1")),
    ]
    assert first_found(searches, original) == (1, KEYWORD, 3)
    assert first_found(searches[:1], original) is None


def test_phase1_ranks_beyond_the_top_k_count_as_misses() -> None:
    assert cut_row({"target_rank": 3}, 10) == {
        "target_rank": 3,
        "hit_at_1": 0,
        "hit_at_5": 1,
        "hit_at_10": 1,
        "reciprocal_rank": pytest.approx(1 / 3),
    }
    assert cut_row({"target_rank": 15}, 10)["reciprocal_rank"] == 0.0
    assert cut_row({"target_rank": None}, 10)["target_rank"] is None


# --- the three configs -------------------------------------------------------------


def test_l0_l1_and_l5_rows(cfg: Config) -> None:
    fake = FakeLiveOpenAlex()
    done = _run(cfg, fake)
    assert (done.n_queries, done.n_done, done.budget_stop) == (3, 3, None)
    rows = _rows(cfg, done.run_dir)
    assert len(rows) == 9 and all(r["status"] == "ok" for r in rows.values())

    # L0: OpenAlex's own order, in which the original comes third here.
    l0 = rows[("L0", T1)]
    expected = [w["id"].rsplit("/", 1)[-1] for w in fake._semantic(GOOD[T1], fake.works)]
    assert json.loads(l0["doc_ids"]) == expected[:10] and expected.index(T1) == 2
    assert (l0["target_rank"], l0["match"], l0["hit_at_1"], l0["hit_at_5"]) == (3, "id", 0, 1)
    assert l0["reciprocal_rank"] == pytest.approx(1 / 3)
    assert (l0["found_query"], l0["found_kind"], l0["found_rank"]) == (0, SEMANTIC, 3)
    assert (l0["searches"], l0["searches_sent"], l0["llm_calls"]) == (1, 1, 0)
    assert (l0["openalex_cost"], l0["openalex_spent"], l0["llm_cost"]) == (0.001, 0.001, 0.0)

    # L1: the same results, from the cache, reranked by the pool.
    l1 = rows[("L1", T1)]
    pool = CandidatePool(FakeEncoder(), max_size=500)
    response = _search(cfg, fake).search(GOOD[T1], SEMANTIC)
    pool.add(response.records, query_index=0, kind=SEMANTIC)
    assert json.loads(l1["doc_ids"]) == pool.rank(GOOD[T1])[:10]
    assert set(json.loads(l1["doc_ids"])) <= set(expected)  # L0's full results, reordered
    assert (l1["searches_sent"], l1["openalex_cost"], l1["openalex_spent"]) == (0, 0.001, 0.0)

    # L5: the proposal (semantic only: it has over 32 words), then the agent's query
    # (keyword and semantic); the agent's query was cached after the first proposal.
    l5 = {t: rows[("L5", t)] for t in (T1, T2, T3)}
    assert [l5[t]["searches"] for t in (T1, T2, T3)] == [3, 3, 3]
    assert sorted(l5[t]["searches_sent"] for t in (T1, T2, T3)) == [0, 0, 2]
    first = l5[T1]
    assert (first["queries_issued"], first["llm_calls"], first["stop_reason"]) == (
        2,
        2,
        "agent_stop",
    )
    assert first["openalex_cost"] == pytest.approx(0.003)
    assert first["method_cost"] == pytest.approx(first["openalex_cost"] + first["llm_cost"])
    assert first["llm_cost"] > 0
    assert first["target_rank"] == json.loads(first["doc_ids"]).index(T1) + 1
    trace = json.loads(first["trace"])
    assert trace["queries"] == [GOOD[T1], AGENT_QUERY] and trace["searches"][0]["cached"]
    # One OpenAlex call per proposal for all three configs, plus the agent's two.
    assert [_semantic_calls(fake, GOOD[t]) for t in (T1, T2, T3)] == [1, 1, 1]
    assert len(fake.requests) == 5

    meta = json.loads((done.run_dir / RUN_FILE).read_text(encoding="utf-8"))
    assert meta["kind"] == KIND and meta["run_id"].startswith("live-")
    assert meta["models"]["encoder_revision"] == FakeEncoder.revision
    assert sorted(p.name for p in (done.run_dir / PER_QUERY_DIR).iterdir()) == [
        "L0.jsonl",
        "L1.jsonl",
        "L5.jsonl",
    ]


def test_an_original_only_the_agent_finds_is_credited_to_its_query(cfg: Config) -> None:
    fake = FakeLiveOpenAlex(hide_from_semantic={T1})
    done = _run(cfg, fake)
    rows = _rows(cfg, done.run_dir)
    for config in ("L0", "L1"):
        assert (rows[(config, T1)]["target_rank"], rows[(config, T1)]["found_query"]) == (
            None,
            None,
        )
    l5 = rows[("L5", T1)]
    assert (l5["found_query"], l5["found_kind"], l5["found_rank"]) == (1, KEYWORD, 1)
    assert l5["target_rank"] is not None
    found = _summary(done.run_dir)["found_by"]
    assert found["L0"]["not returned"] == {"n": 1, "in_top_k": 0}
    assert found["L5"]["agent query, keyword"] == {"n": 1, "in_top_k": 1}


# --- OpenAlex's daily budget -----------------------------------------------------


def test_a_spent_budget_stops_the_run_and_a_resume_finishes_it(cfg: Config) -> None:
    # 15 credits: the first proposal's search (10), then the agent's keyword search
    # can't be paid for. L0 and L1 of the first query finish; its L5 has no row.
    fake = FakeLiveOpenAlex(budget_credits=15)
    done = _run(cfg, fake)
    assert done.budget_stop is not None and done.budget_stop.resets_in_s == 3600
    assert (done.n_queries, done.n_done) == (3, 0)
    assert len(fake.requests) == 2  # the 429 was not retried
    rows = _rows(cfg, done.run_dir)
    first = min(_qids(cfg).values())
    assert {(c, r["qid"]) for (c, _), r in rows.items()} == {("L0", first), ("L1", first)}
    assert all(r["status"] == "ok" for r in rows.values())  # never an error row
    meta = json.loads((done.run_dir / RUN_FILE).read_text(encoding="utf-8"))
    assert [(s["qid"], s["config"]) for s in meta["budget_stops"]] == [(first, "L5")]
    summary = _summary(done.run_dir)
    assert summary["errors"] == {"L0": 0, "L1": 0} and len(summary["budget_stops"]) == 1
    assert "Stopped when OpenAlex's daily budget was spent" in (
        done.run_dir / SUMMARY_MD
    ).read_text(encoding="utf-8")

    fake.budget_credits = 10_000  # the next day
    resumed = _run(cfg, fake, run_dir=done.run_dir)
    assert (resumed.n_done, resumed.budget_stop) == (3, None)
    rows = _rows(cfg, done.run_dir)
    assert len(rows) == 9 and all(r["status"] == "ok" for r in rows.values())
    target = next(t for t, q in _qids(cfg).items() if q == first)
    assert _semantic_calls(fake, GOOD[target]) == 1  # its proposal came from the cache


def test_a_budget_429_without_headers_is_caught_by_the_rate_limit_check(cfg: Config) -> None:
    fake = FakeLiveOpenAlex(budget_credits=15, budget_headers=False)
    done = _run(cfg, fake)
    assert done.budget_stop is not None and "/rate-limit" in str(done.budget_stop)
    assert fake.rate_limit_calls == 1
    rows = _rows(cfg, done.run_dir)
    assert {c for c, _ in rows} == {"L0", "L1"}
    assert all(r["status"] == "ok" for r in rows.values())


# --- errors and resuming -------------------------------------------------------------


def test_other_openalex_errors_are_error_rows_and_can_be_retried(cfg: Config) -> None:
    fake = FakeLiveOpenAlex(fail_with=[400])  # the first search is rejected
    done = _run(cfg, fake)
    rows = _rows(cfg, done.run_dir)
    errors = {k for k, r in rows.items() if r["status"] == "error"}
    first = next(t for t, q in _qids(cfg).items() if q == min(_qids(cfg).values()))
    assert errors == {("L0", first)} and done.budget_stop is None
    assert rows[("L0", first)]["error"].startswith("OpenAlexError: HTTP 400")
    assert fake.rate_limit_calls == 1  # the budget was checked, and is fine
    assert _summary(done.run_dir)["errors"]["L0"] == 1

    retried = _run(cfg, fake, run_dir=done.run_dir, retry_errors=True)
    assert retried.budget_stop is None
    assert all(r["status"] == "ok" for r in _rows(cfg, done.run_dir).values())


def test_llm_errors_are_error_rows(cfg: Config) -> None:
    done = _run(cfg, FakeLiveOpenAlex(), agent(failing_proposal=GOOD[T2]))
    rows = _rows(cfg, done.run_dir)
    assert [k for k, r in rows.items() if r["status"] == "error"] == [("L5", T2)]
    assert rows[("L5", T2)]["error"] == "LLMError: simulated outage"
    assert _summary(done.run_dir)["quality"]["L5"]["n_ok"] == 2  # left out, as in Phase 1


def test_a_check_over_its_cost_cap_is_a_miss_counted_apart(cfg: Config, tmp_path: Path) -> None:
    capped = config_from_dict(_config_dict(tmp_path, cost_cap_per_check=0.0015))
    done = _run(capped, FakeLiveOpenAlex(), limit=1)
    rows = _rows(capped, done.run_dir)
    (row,) = [r for (c, _), r in rows.items() if c == "L5"]
    assert (row["status"], row["stop_reason"], row["hit_at_10"], row["reciprocal_rank"]) == (
        "ok",
        "cost_cap",
        0,
        0.0,
    )
    assert row["error"].startswith("CheckCostCapExceeded: an agent call could cost")
    # What it cost before it stopped: the proposal's search, from L0's cache.
    assert (row["searches"], row["searches_sent"], row["openalex_cost"]) == (1, 0, 0.001)
    assert row["llm_calls"] == 0 and row["latency_s"] >= 0
    assert rows[("L0", row["target_id"])]["status"] == "ok"  # one search fits the cap
    summary = _summary(done.run_dir)
    assert summary["quality"]["L5"] == {
        "n_ok": 1,
        "n_error": 0,
        "n_cost_cap": 1,
        "hit_at_1": 0.0,
        "hit_at_5": 0.0,
        "hit_at_10": 0.0,
        "mrr": 0.0,
    }
    assert summary["quality"]["L0"]["n_cost_cap"] == 0
    assert summary["found_by"]["L5"]["over its cost cap"] == {"n": 1, "in_top_k": 0}
    assert summary["effort"]["L5"]["queries"] is None  # not recorded for a capped check
    assert summary["effort"]["L5"]["method_cost"]["mean"] == pytest.approx(0.001)

    # A capped check is finished: --retry-errors doesn't run it again.
    llm = agent()
    _run(capped, FakeLiveOpenAlex(), llm, run_dir=done.run_dir, retry_errors=True, limit=1)
    assert llm.calls == []


def test_from_qid_starts_the_selection_in_seeded_order(cfg: Config) -> None:
    qids = sorted(_qids(cfg).values())
    done = _run(cfg, FakeLiveOpenAlex(), from_qid=qids[1][:5], limit=1)
    rows = _rows(cfg, done.run_dir)
    assert {r["qid"] for r in rows.values()} == {qids[1]}
    summary = _summary(done.run_dir)
    assert (summary["first_qid"], summary["last_qid"]) == (qids[1], qids[1])


def test_resume_refuses_a_changed_live_config_or_a_phase1_run(
    cfg: Config, tmp_path: Path
) -> None:
    done = _run(cfg, FakeLiveOpenAlex(), limit=1)
    changed = config_from_dict(_config_dict(tmp_path, results_per_query=20))
    with pytest.raises(EvalError, match="live changed"):
        _run(changed, FakeLiveOpenAlex(), run_dir=done.run_dir)
    phase1_dir = tmp_path / "phase1"
    phase1_dir.mkdir()
    (phase1_dir / RUN_FILE).write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
    with pytest.raises(EvalError, match="not a live-eval run"):
        _run(cfg, FakeLiveOpenAlex(), run_dir=phase1_dir)


def test_the_phase1_run_must_exist_and_share_the_benchmark(cfg: Config, tmp_path: Path) -> None:
    missing = config_from_dict(_config_dict(tmp_path, phase1_run="19990101-000000"))
    with pytest.raises(EvalError, match="not in the results database"):
        _run(missing, FakeLiveOpenAlex())
    _phase1_run(cfg, {}, {}, run_id="other-bench", bench_sha="f" * 64)
    other = config_from_dict(_config_dict(tmp_path, phase1_run="other-bench"))
    with pytest.raises(EvalError, match="another benchmark"):
        _run(other, FakeLiveOpenAlex())
    assert not cfg.run.out_dir.exists()  # no run directory for a run that can't start


# --- summary ----------------------------------------------------------------------


def test_the_summary_compares_with_phase1_b5_on_the_same_queries(cfg: Config) -> None:
    done = _run(cfg, FakeLiveOpenAlex())
    summary = _summary(done.run_dir)
    assert summary["configs"] == ["L0", "L1", "L5"]
    assert summary["compared"] == ["L0", "L1", "L5", PHASE1_B5]
    phase1 = summary["quality"][PHASE1_B5]
    assert (phase1["n_ok"], phase1["hit_at_5"], phase1["hit_at_10"]) == (
        3,
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
    )
    assert phase1["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)  # rank 15 is beyond top 10
    # Pre-declared: L5 vs L1 is primary; the rest are secondary.
    assert [(p["a"], p["b"], p["n"]) for p in summary["paired"]] == [("L5", "L1", 3)]
    assert [(p["a"], p["b"], p["n"]) for p in summary["paired_secondary"]] == [
        ("L1", "L0", 3),
        ("L5", "L0", 3),
        ("L5", PHASE1_B5, 3),
    ]
    difficulty = summary["difficulty"]
    assert difficulty["bucket_sizes"] == {"1": 1, "2-10": 1, "11-100": 0, "not found": 1}
    assert set(difficulty["hit_at_5"]) == {"L0", "L1", "L5", PHASE1_B5}
    assert difficulty["hit_at_5"][PHASE1_B5]["2-10"] == {"n": 1, "hit_at_5": 0.0}
    in_top_k = round(3 * summary["quality"]["L0"]["hit_at_10"])  # top_k is 10
    assert summary["matches"]["L0"] == {"id": in_top_k, "alias": 0, "doi": 0, "title": 0}
    assert summary["spent"]["L0"]["searches_sent"] == 3
    markdown = (done.run_dir / SUMMARY_MD).read_text(encoding="utf-8")
    for heading in (
        "Retrieval quality",
        "Effort per query",
        "Hit@5 by difficulty",
        "Primary comparison",
        "Secondary comparisons (exploratory)",
        "Which search first returned the original",
        "How the original matched",
        "Money spent",
        "Errors",
    ):
        assert f"## {heading}" in markdown
    header = "| Config | Queries | Errors | Cost-cap misses | Hit@1 | Hit@5 | Hit@10 | MRR@10 |"
    assert header in markdown and "| L5 vs L1 | 3 |" in markdown


def test_without_a_phase1_run_there_is_no_comparison(tmp_path: Path) -> None:
    data = _config_dict(tmp_path)
    del data["live"]["phase1_run"]
    config = config_from_dict(data)
    freeze(config.corpus.out_dir, corpus_records(), target_rows(), {})
    make_bench(config, _paraphraser(), FakeLLM(lambda messages: judge_reply()))
    summary = _summary(_run(config, FakeLiveOpenAlex(), limit=1).run_dir)
    assert summary["compared"] == summary["configs"] and summary["difficulty"] is None


# --- the command line and MLflow ---------------------------------------------------------


def test_live_eval_cli_with_mlflow_report_and_a_budget_stop(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mlflow import MlflowClient

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_dict(tmp_path)), encoding="utf-8")
    fake = FakeLiveOpenAlex(budget_credits=10_000)

    class Search:
        @staticmethod
        def from_config(config: Config, cache: OpenAlexCache, *, evaluation: bool) -> Any:
            assert evaluation and config.live is not None  # read as an evaluation
            return LiveSearchClient(
                config.live,
                cache,
                max_age=None,
                transport=fake.transport,
                sleep=lambda seconds: None,
                clock=lambda: 0.0,
            )

    monkeypatch.setattr(cli, "LiveSearchClient", Search)
    monkeypatch.setattr(cli, "LLMClient", lambda model, cache, ledger: agent())
    monkeypatch.setattr(cli, "make_encoder", lambda retrieval_cfg: FakeEncoder())

    assert cli.main(["live-eval", "--config", str(config_path), "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "OpenAlex budget left today: $1.0000 of $1" in out
    assert "2 of 2 queries done" in out
    (run_dir,) = cfg.run.out_dir.glob("live-*")
    summary = _summary(run_dir)

    client = MlflowClient(tracking_uri=cfg.run.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name("test-live")
    runs = client.search_runs([experiment.experiment_id])
    parent = next(r for r in runs if "mlflow.parentRunId" not in r.data.tags)
    children = {r.info.run_name: r for r in runs if "mlflow.parentRunId" in r.data.tags}
    assert set(children) == {"L0", "L1", "L5"}
    assert children["L5"].data.metrics["hit_at_5"] == summary["quality"]["L5"]["hit_at_5"]
    assert children["L5"].data.params["live.top_k"] == "10"
    assert {"L5_vs_L1.mcnemar_p", "L5_vs_L0.mcnemar_p"} <= set(parent.data.metrics)
    assert f"L5_vs_{PHASE1_B5}.delta_hit_at_5" in parent.data.metrics
    assert children["L5"].data.metrics["n_cost_cap"] == 0
    assert not Path("mlartifacts").exists() and not Path("mlruns").exists()

    (run_dir / SUMMARY_MD).unlink()
    assert cli.main(["report", str(run_dir)]) == 0
    assert _summary(run_dir) == summary and (run_dir / SUMMARY_MD).exists()

    # A spent budget: a clean stop, with the command that resumes the run.
    fake.budget_credits = 5
    args = ["live-eval", "--config", str(config_path), "--run-dir", str(run_dir)]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "stopped: OpenAlex's daily budget is spent" in out
    assert "2 of 3 queries done" in out and "resets in 1h 00m" in out
    hint = f"python -m rk live-eval --config {config_path} --run-dir {run_dir}"
    assert f"Resume with:\n  {hint}\n" in out


def test_live_eval_cli_checks_its_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["live-eval", "--config", "x.yaml", "--configs", "L0,L9"])
    assert "unknown live configs ['L9']" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["live-eval", "--config", "x.yaml", "--retry-errors"])
    assert "--retry-errors needs --run-dir" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["live-eval", "--config", "x.yaml", "--from-qid", "45"])
    assert "expected a qid prefix like A0045" in capsys.readouterr().err


def test_the_resume_hint_keeps_the_selection() -> None:
    config, run_dir = Path("configs/phase1.yaml"), Path("runs/live-1")
    hint = cli.resume_hint(config, run_dir, None, "A0045", 200, ["live.top_k=20"])
    assert hint == (
        f"python -m rk live-eval --config {config} --run-dir {run_dir} "
        '--from-qid A0045 --limit 200 --set "live.top_k=20"'
    )
