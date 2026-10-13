from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
import yaml
from bench_data import GOOD, T2, TITLES, corpus_records, judge_reply, target_rows
from conftest import FakeEncoder, FakeLLM, corpus_config_dict

from rk import cli
from rk.bench import _visit_order, make_bench, make_qid
from rk.config import Config, config_from_dict
from rk.corpus import freeze
from rk.evaluate import (
    CONFIG_ORDER,
    PER_QUERY_DIR,
    RUN_FILE,
    SUMMARY_JSON,
    SUMMARY_MD,
    EvalError,
    bucket,
    connect,
    git_state,
    mcnemar_exact,
    ok_row,
    paired_bootstrap,
    report,
    run_eval,
    target_rank,
)
from rk.llm import BudgetExceeded, LLMError
from rk.retrieval import Result, Trace, build_index

ORDER = [t["id"] for t in _visit_order(target_rows(), 7)]  # the bench seed in these configs
Q2 = make_qid(ORDER.index(T2), T2)
YES = judge_reply()
REWRITES = '{"queries": ["learned groups for efficient comparison", "memory saving condensation"]}'
SEARCH = '{"action": "search", "query": "condensing long texts with learned groups"}'
STOP = '{"action": "stop", "reason": "enough"}'


# --- statistics and ranks ----------------------------------------------------


def test_mcnemar_matches_hand_computed_examples() -> None:
    # b=1, c=6: n=7, P(X <= 1) = (C(7,0) + C(7,1)) / 2^7 = 8/128; two-sided = 0.125.
    assert mcnemar_exact(1, 6) == pytest.approx(0.125)
    assert mcnemar_exact(6, 1) == pytest.approx(0.125)
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact(5, 5) == 1.0  # capped at 1
    assert mcnemar_exact(0, 0) == 1.0


def test_bootstrap_is_seeded_and_covers_a_known_difference() -> None:
    diffs = [1] * 30 + [0] * 70  # mean difference 0.3
    ci = paired_bootstrap(diffs, seed=11)
    assert ci == paired_bootstrap(diffs, seed=11)
    assert ci is not None and 0.0 < ci[0] < 0.3 < ci[1] < 0.5
    assert paired_bootstrap([], seed=11) is None
    # 0/1 differences give means on a 0.01 grid, so the seed shows with continuous ones:
    continuous = [((i * 37) % 101) / 101 - 0.3 for i in range(100)]
    assert paired_bootstrap(continuous, seed=11) != paired_bootstrap(continuous, seed=12)


def test_target_rank_counts_the_target_or_any_alias() -> None:
    assert target_rank(["W5", "W7", "W1"], ["W1"]) == 3
    assert target_rank(["W5", "W8", "W1"], ["W1", "W8"]) == 2  # an alias earlier wins
    assert target_rank(["W5", "W7"], ["W1"]) is None
    assert target_rank([f"W{n}" for n in range(200, 301)] + ["W1"], ["W1"]) is None  # not in top 100


def test_ok_row_hits_and_reciprocal_rank() -> None:
    query = {"qid": "A-W1", "target_id": "W1", "target_aliases": ["W9"]}
    trace = Trace(queries=["q"], budget_used=1, latency_s=0.5)
    row = ok_row("run", "B1", query, Result(["W3", "W4", "W9", "W1"], trace))
    assert (row["target_rank"], row["hit_at_1"], row["hit_at_5"], row["hit_at_10"]) == (3, 0, 1, 1)
    assert row["reciprocal_rank"] == pytest.approx(1 / 3)
    missed = ok_row("run", "B1", query, Result(["W3"], trace))
    assert (missed["target_rank"], missed["hit_at_10"], missed["reciprocal_rank"]) == (None, 0, 0.0)


@pytest.mark.parametrize(
    ("rank", "name"), [(1, "1"), (2, "2-10"), (10, "2-10"), (11, "11-100"), (None, "not found")]
)
def test_difficulty_buckets(rank: int | None, name: str) -> None:
    assert bucket(rank) == name


# --- fixtures: index, bench and agent ----------------------------------------


def _config_dict(tmp_path: Path) -> dict[str, Any]:
    data = corpus_config_dict(tmp_path / "corpus")
    data["retrieval"]["index_dir"] = str(tmp_path / "index")
    data["bench"].update(n_targets=3, out_dir=str(tmp_path / "bench"))
    data["run"].update(
        out_dir=str(tmp_path / "runs"),
        results_db=str(tmp_path / "results.sqlite"),
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        mlflow_experiment="test-experiment",
    )
    return data


def _paraphraser() -> FakeLLM:
    def reply(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        return next(GOOD[t] for t, title in TITLES.items() if f"Title: {title}" in prompt)

    return FakeLLM(reply)


def agent(*, failing_proposal: str | None = None, billed_cost: float | None = None) -> FakeLLM:
    """Rewrites for B4; for B5, one search then stop. Raises for one proposal if asked."""

    def reply(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        if failing_proposal and failing_proposal in prompt:
            raise LLMError("simulated outage")
        if "Searches so far" in prompt:
            return STOP if '\n2. "' in prompt else SEARCH
        return REWRITES

    return FakeLLM(reply, billed_cost=billed_cost)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Frozen fixture corpus, FakeEncoder index and a FakeLLM benchmark of three queries."""
    config = config_from_dict(_config_dict(tmp_path))
    freeze(config.corpus.out_dir, corpus_records(), target_rows(), {})
    build_index(config, FakeEncoder())
    stats = make_bench(config, _paraphraser(), FakeLLM(lambda messages: YES))
    assert stats["accepted"] == 3
    return config


def _rows(cfg: Config, run_dir: Path) -> list[dict[str, Any]]:
    run_id = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))["run_id"]
    with closing(sqlite3.connect(cfg.run.results_db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM query_results WHERE run_id = ? ORDER BY config, qid", (run_id,)
        )
        return [dict(r) for r in rows]


def _proposals(llm: FakeLLM) -> set[str]:
    return {next((p for p in GOOD.values() if p in c[-1]["content"]), "?") for c in llm.calls}


# --- errors, retries, resume -------------------------------------------------


def test_llm_errors_mark_rows_and_are_left_out_of_paired_stats(cfg: Config) -> None:
    run_dir = run_eval(
        cfg, agent(failing_proposal=GOOD[T2]), FakeEncoder(), log_mlflow=False
    )

    rows = _rows(cfg, run_dir)
    errors = [(r["config"], r["qid"]) for r in rows if r["status"] == "error"]
    llm_configs = ["B4", "B4-D", "B5", "B5-D", "B5-U"]
    assert errors == [(c, Q2) for c in llm_configs]
    assert all(r["error"] == "LLMError: simulated outage" for r in rows if r["status"] == "error")
    summary = json.loads((run_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary["errors"] == {c: int(c in llm_configs) for c in CONFIG_ORDER}
    assert {(p["a"], p["b"]): p["n"] for p in summary["paired"]} == {
        ("B4", "B3"): 2,
        ("B5", "B4"): 2,
        ("B5-U", "B5"): 2,
    }
    assert {(p["a"], p["b"]): p["n"] for p in summary["paired_secondary"]} == {
        ("B2", "B3"): 3,
        ("B4-D", "B2"): 2,
        ("B5-D", "B2"): 2,
        ("B5-D", "B4-D"): 2,
    }
    assert summary["quality"]["B4"]["n_ok"] == 2


def test_retry_errors_redoes_only_the_errored_rows(cfg: Config) -> None:
    run_dir = run_eval(cfg, agent(failing_proposal=GOOD[T2]), FakeEncoder(), log_mlflow=False)
    before = {(r["config"], r["qid"]): r["finished_at"] for r in _rows(cfg, run_dir)}

    plain_rerun = agent()
    run_eval(cfg, plain_rerun, FakeEncoder(), run_dir=run_dir, log_mlflow=False)
    assert plain_rerun.calls == []  # without --retry-errors nothing is redone

    healthy = agent()
    run_eval(cfg, healthy, FakeEncoder(), run_dir=run_dir, retry_errors=True, log_mlflow=False)

    assert _proposals(healthy) == {GOOD[T2]}  # only the failed query's rows ran again
    after = _rows(cfg, run_dir)
    assert all(r["status"] == "ok" for r in after)
    redone = {
        (r["config"], r["qid"])
        for r in after
        if r["finished_at"] != before[(r["config"], r["qid"])]
    }
    assert redone <= {(c, Q2) for c in ("B4", "B4-D", "B5", "B5-D", "B5-U")}
    meta = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    assert [r["retry_errors"] for r in meta["resumed"]] == [False, True]


def test_rerunning_into_a_run_directory_resumes(cfg: Config) -> None:
    first = agent()
    run_dir = run_eval(cfg, first, FakeEncoder(), limit=1, log_mlflow=False)
    rows = _rows(cfg, run_dir)
    assert len(rows) == len(CONFIG_ORDER)  # one query, every config
    assert {r["qid"] for r in rows} == {make_qid(0, ORDER[0])}  # --limit follows the seeded order
    done_proposal = _proposals(first)

    later = agent()
    run_eval(cfg, later, FakeEncoder(), run_dir=run_dir, log_mlflow=False)

    assert len(_rows(cfg, run_dir)) == 3 * len(CONFIG_ORDER)
    assert _proposals(later).isdisjoint(done_proposal)  # the finished query wasn't redone


def test_new_configs_can_be_added_to_an_existing_run(cfg: Config) -> None:
    primary = ["B1", "B2", "B3", "B4", "B5", "B5-U"]
    run_dir = run_eval(cfg, agent(), FakeEncoder(), configs=primary, log_mlflow=False)
    before = {(r["config"], r["qid"]): r["finished_at"] for r in _rows(cfg, run_dir)}

    added = agent()
    run_eval(cfg, added, FakeEncoder(), configs=["B4-D", "B5-D"], run_dir=run_dir, log_mlflow=False)

    rows = _rows(cfg, run_dir)
    assert len(rows) == 3 * len(CONFIG_ORDER)
    kept = {(r["config"], r["qid"]): r["finished_at"] for r in rows if r["config"] in primary}
    assert kept == before  # the earlier configs' rows weren't redone
    assert added.calls  # the new ones were
    summary = json.loads((run_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary["configs"] == list(CONFIG_ORDER)
    assert [(p["a"], p["b"]) for p in summary["paired_secondary"]] == [
        ("B2", "B3"),
        ("B4-D", "B2"),
        ("B5-D", "B2"),
        ("B5-D", "B4-D"),
    ]
    markdown = (run_dir / SUMMARY_MD).read_text(encoding="utf-8")
    assert "| B5 vs B4 (primary) |" in markdown
    assert "### Secondary comparisons (dense base)" in markdown
    assert "B4-D and B5-D are secondary" in markdown


def test_billed_cost_is_reported_apart_from_the_computed_cost(cfg: Config) -> None:
    run_dir = run_eval(cfg, agent(billed_cost=0.01), FakeEncoder(), log_mlflow=False)

    b4 = [r for r in _rows(cfg, run_dir) if r["config"] == "B4"]
    assert all(r["billed_cost"] == pytest.approx(0.01 * r["llm_calls"]) for r in b4)
    summary = json.loads((run_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary["billed"]["B1"] == {"computed": 0.0, "billed": 0.0, "unrecorded_rows": 0}
    assert summary["billed"]["B4"]["billed"] == pytest.approx(0.03)
    computed = sum(r["money_spent"] for r in b4)
    assert summary["billed"]["B4"]["computed"] == pytest.approx(computed)
    # The cost tables stay on computed cost.
    assert summary["cost_quality"]["B4"]["mean_method_cost"] == pytest.approx(computed / 3)
    assert "## Billed cost" in (run_dir / SUMMARY_MD).read_text(encoding="utf-8")


def test_an_older_results_database_gets_the_billed_column(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE query_results (run_id TEXT, config TEXT, llm_calls INTEGER)")
        conn.execute("INSERT INTO query_results VALUES ('r', 'B4', 1)")
        conn.commit()

    with closing(connect(path)) as conn:
        columns = [r["name"] for r in conn.execute("PRAGMA table_info(query_results)")]
        assert "billed_cost" in columns
        assert conn.execute("SELECT billed_cost FROM query_results").fetchone()[0] is None


def test_rows_without_a_billed_cost_are_reported_as_not_recorded(cfg: Config) -> None:
    run_dir = run_eval(cfg, agent(), FakeEncoder(), log_mlflow=False)
    with closing(sqlite3.connect(cfg.run.results_db)) as conn:  # as rows from before billing
        conn.execute("UPDATE query_results SET billed_cost = NULL")
        conn.commit()

    summary = report(run_dir)

    assert summary["billed"]["B4"] == {
        "computed": pytest.approx(summary["billed"]["B4"]["computed"]),
        "billed": None,
        "unrecorded_rows": 3,
    }
    assert summary["billed"]["B1"]["billed"] == 0.0  # no LLM calls, so nothing to bill
    assert "– (3 rows not recorded)" in (run_dir / SUMMARY_MD).read_text(encoding="utf-8")


def test_budget_exceeded_stops_the_whole_run_after_writing_the_summary(cfg: Config) -> None:
    def broke(messages: list[dict[str, str]]) -> str:
        raise BudgetExceeded("cap reached")

    with pytest.raises(BudgetExceeded):
        run_eval(cfg, FakeLLM(broke), FakeEncoder(), log_mlflow=False)

    (run_dir,) = cfg.run.out_dir.iterdir()
    rows = _rows(cfg, run_dir)
    assert {r["config"] for r in rows} == {"B1", "B2", "B3"}  # B4 stopped the run
    assert all(r["status"] == "ok" for r in rows)  # not recorded as error rows
    assert (run_dir / SUMMARY_MD).exists()


def test_git_state_counts_only_tracked_changes_as_dirty(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    (tmp_path / "tracked.txt").write_text("one", encoding="utf-8")
    git("add", "tracked.txt")
    git("-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-m", "init")
    (tmp_path / "scratch.txt").write_text("notes", encoding="utf-8")

    commit, dirty, untracked = git_state(tmp_path)
    assert commit is not None and len(commit) == 40
    assert (dirty, untracked) == (False, ["scratch.txt"])  # untracked: listed, not dirty

    (tmp_path / "tracked.txt").write_text("two", encoding="utf-8")
    assert git_state(tmp_path)[1] is True


def test_resume_refuses_a_changed_retrieval_config(cfg: Config, tmp_path: Path) -> None:
    run_dir = run_eval(cfg, agent(), FakeEncoder(), limit=1, log_mlflow=False)
    data = _config_dict(tmp_path)
    data["retrieval"]["max_queries"] = 3
    with pytest.raises(EvalError, match="retrieval changed"):
        run_eval(config_from_dict(data), agent(), FakeEncoder(), run_dir=run_dir, log_mlflow=False)


def test_run_needs_an_intact_benchmark(cfg: Config) -> None:
    stats_path = cfg.bench.out_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["corpus_sha256"] = "0" * 64
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(EvalError, match="made from another corpus"):
        run_eval(cfg, agent(), FakeEncoder(), log_mlflow=False)


# --- end to end ----------------------------------------------------------------


def test_end_to_end_cli_run_and_report_with_mlflow(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mlflow import MlflowClient

    # An empty working directory, so the check at the end sees only what this test
    # wrote there (the repo root holds a real run's mlartifacts/).
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_dict(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(cli, "LLMClient", lambda model, cache, ledger: agent())
    monkeypatch.setattr(cli, "make_encoder", lambda retrieval_cfg: FakeEncoder())

    assert cli.main(["run", "--config", str(config_path)]) == 0
    (run_dir,) = cfg.run.out_dir.iterdir()
    assert "summary.md" in capsys.readouterr().out

    meta = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    recorded = {"git_commit", "git_dirty", "config", "corpus_sha256", "bench_sha256", "index"}
    assert recorded | {"prompt_hashes", "models", "mlflow"} <= set(meta)
    assert set(meta["prompt_hashes"]["retrieval"]) == {"rewrite", "agent_step"}
    assert meta["models"]["encoder_revision"] == FakeEncoder.revision
    rows = _rows(cfg, run_dir)
    assert len(rows) == 3 * len(CONFIG_ORDER) and all(r["status"] == "ok" for r in rows)
    assert sorted(p.name for p in (run_dir / PER_QUERY_DIR).iterdir()) == sorted(
        f"{c}.jsonl" for c in CONFIG_ORDER
    )
    summary = json.loads((run_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary["configs"] == list(CONFIG_ORDER)
    assert summary["difficulty"]["bucket_sizes"] and len(summary["paired"]) == 3
    assert len(summary["paired_secondary"]) == 4
    markdown = (run_dir / SUMMARY_MD).read_text(encoding="utf-8")
    headings = (
        "Retrieval quality",
        "Effort per query",
        "Hit@5 by difficulty",
        "Paired comparisons",
        "Cost and quality",
        "Billed cost",
        "Errors",
    )
    for heading in headings:
        assert f"## {heading}" in markdown

    # report rebuilds the same summary from the database
    (run_dir / SUMMARY_MD).unlink()
    assert cli.main(["report", str(run_dir)]) == 0
    assert report(run_dir) == summary
    assert (run_dir / SUMMARY_MD).read_text(encoding="utf-8") == markdown

    # MLflow: one parent run, one child per config, artifacts beside the temp database
    client = MlflowClient(tracking_uri=cfg.run.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name("test-experiment")
    assert experiment.artifact_location.startswith((tmp_path / "mlartifacts").as_uri())
    runs = client.search_runs([experiment.experiment_id])
    parent = next(r for r in runs if "mlflow.parentRunId" not in r.data.tags)
    children = {r.info.run_name: r for r in runs if "mlflow.parentRunId" in r.data.tags}
    assert set(children) == set(CONFIG_ORDER)
    assert all(r.data.tags["mlflow.parentRunId"] == parent.info.run_id for r in children.values())
    assert children["B4"].data.metrics["hit_at_5"] == summary["quality"]["B4"]["hit_at_5"]
    assert children["B5"].data.params["model.encoder_revision"] == FakeEncoder.revision
    assert "B5_vs_B4.mcnemar_p" in parent.data.metrics
    assert "B5-D_vs_B4-D.mcnemar_p" in parent.data.metrics  # secondary comparisons too
    child_artifacts = {a.path for a in client.list_artifacts(children["B1"].info.run_id)}
    assert "B1.jsonl" in child_artifacts
    parent_artifacts = {a.path for a in client.list_artifacts(parent.info.run_id)}
    assert {"summary.md", "summary.json"} <= parent_artifacts
    assert not Path("mlartifacts").exists() and not Path("mlruns").exists()  # nothing in the cwd
