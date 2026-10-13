"""Evaluation (spec: Evaluation): runner, SQL results, metrics, paired statistics, MLflow.

`run_eval` runs the retrieval configurations over the benchmark and writes one
`query_results` row per (run, config, query) to the results database as each
one finishes, so a rerun into the same run directory resumes, and
--retry-errors redoes only the rows whose LLM call failed. `summarize` builds
the summary from the database with SQL; paired statistics use only queries
that succeeded in both configs. Every run is logged to MLflow as a parent run
with one child run per config.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sqlite3
import subprocess
from collections.abc import Iterable, Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rk.agent import make_retrievers
from rk.bench import ACCEPTED_FILE, STATS_FILE
from rk.config import Config, RunConfig
from rk.corpus import CORPUS_FILE, load_frozen_corpus
from rk.llm import ChatModel, LLMError, load_prompt
from rk.retrieval import ENCODER_SETTINGS, TOP_K, Encoder, Result, load_index

CONFIG_ORDER = ("B1", "B2", "B3", "B4", "B5", "B5-U", "B4-D", "B5-D")
SECONDARY_CONFIGS = ("B4-D", "B5-D")  # B4 and B5 with B2 as the base instead of B3
PRIMARY_PAIR = ("B5", "B4")
PAIRS = (("B4", "B3"), PRIMARY_PAIR, ("B5-U", "B5"))  # (a, b): a minus b
SECONDARY_PAIRS = (("B2", "B3"), ("B4-D", "B2"), ("B5-D", "B2"), ("B5-D", "B4-D"))
HIT_KS = (1, 5, 10)
BOOTSTRAP_RESAMPLES = 10_000
BUCKETS = ("1", "2-10", "11-100", "not found")  # B1 rank of the target
EFFORT = ("queries", "llm_calls", "tokens", "method_cost", "latency_s")
RETRIEVAL_PROMPTS = ("rewrite", "agent_step")

RUN_FILE = "run.json"
CONFIG_FILE = "config.yaml"
SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"
PER_QUERY_DIR = "per_query"


class EvalError(RuntimeError):
    """An evaluation can't run or be reported as asked."""


# --- results database --------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_dir TEXT NOT NULL,
    created_at TEXT NOT NULL,
    git_commit TEXT,
    git_dirty INTEGER,
    corpus_sha256 TEXT NOT NULL,
    bench_sha256 TEXT NOT NULL,
    metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS query_results (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    config TEXT NOT NULL,
    qid TEXT NOT NULL,
    target_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error TEXT,
    target_rank INTEGER,
    hit_at_1 INTEGER,
    hit_at_5 INTEGER,
    hit_at_10 INTEGER,
    reciprocal_rank REAL,
    queries_issued INTEGER,
    llm_calls INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    method_cost REAL,
    money_spent REAL,
    billed_cost REAL,
    latency_s REAL,
    stop_reason TEXT,
    trace TEXT,
    doc_ids TEXT,
    finished_at TEXT NOT NULL,
    PRIMARY KEY (run_id, config, qid)
);
"""

_COLUMNS = (
    "run_id", "config", "qid", "target_id", "status", "error", "target_rank", "hit_at_1",
    "hit_at_5", "hit_at_10", "reciprocal_rank", "queries_issued", "llm_calls", "prompt_tokens",
    "completion_tokens", "method_cost", "money_spent", "billed_cost", "latency_s", "stop_reason",
    "trace", "doc_ids", "finished_at",
)


def connect(path: Path) -> sqlite3.Connection:
    """The results database, created with its tables if needed.

    A database from before billed costs were recorded gets the billed_cost
    column added; its earlier rows keep NULL there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(query_results)")}
    if "billed_cost" not in columns:
        conn.execute("ALTER TABLE query_results ADD COLUMN billed_cost REAL")
        conn.commit()
    return conn


def _put(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    placeholders = ", ".join("?" for _ in _COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO query_results ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
        [row.get(c) for c in _COLUMNS],
    )
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- per-query results -------------------------------------------------------


def target_rank(doc_ids: Sequence[str], target_ids: Iterable[str], k: int = TOP_K) -> int | None:
    """1-based position of the first doc that is the target or one of its aliases, else None."""
    wanted = set(target_ids)
    return next((i for i, d in enumerate(doc_ids[:k], start=1) if d in wanted), None)


def ok_row(run_id: str, config: str, query: dict[str, Any], result: Result) -> dict[str, Any]:
    rank = target_rank(result.doc_ids, [query["target_id"], *query["target_aliases"]])
    trace = result.trace
    return {
        "run_id": run_id,
        "config": config,
        "qid": query["qid"],
        "target_id": query["target_id"],
        "status": "ok",
        "target_rank": rank,
        **{f"hit_at_{k}": int(rank is not None and rank <= k) for k in HIT_KS},
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "queries_issued": trace.budget_used,
        "llm_calls": trace.usage.calls,
        "prompt_tokens": trace.usage.prompt_tokens,
        "completion_tokens": trace.usage.completion_tokens,
        "method_cost": trace.usage.method_cost,
        "money_spent": trace.usage.money_spent,
        "billed_cost": trace.usage.billed_cost,
        "latency_s": trace.latency_s,
        "stop_reason": trace.stop_reason,
        "trace": json.dumps(trace.to_dict(), ensure_ascii=False),
        "doc_ids": json.dumps(result.doc_ids),
        "finished_at": _now(),
    }


def error_row(run_id: str, config: str, query: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config": config,
        "qid": query["qid"],
        "target_id": query["target_id"],
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}"[:2000],
        "finished_at": _now(),
    }


# --- statistics --------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c (binomial, p=1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap(
    diffs: Sequence[float], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> list[float] | None:
    """Seeded percentile 95% CI [low, high] of the mean paired difference; None without data."""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(resamples, d.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]  # a list, so the summary round-trips through JSON


def _spread(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    p50, p95 = np.percentile(v, [50, 95])
    return {"mean": float(v.mean()), "p50": float(p50), "p95": float(p95)}


def bucket(rank: int | None) -> str:
    if rank is None:
        return "not found"
    if rank == 1:
        return "1"
    return "2-10" if rank <= 10 else "11-100"


# --- summary -----------------------------------------------------------------


def summarize(
    db_path: Path, run_id: str, *, seed: int, resumed: Sequence[dict[str, Any]] = ()
) -> dict[str, Any]:
    """Every table of the run summary, computed from the results database.

    `resumed` is run.json's list of resumes (when, at which commit), for the header.
    """
    with closing(connect(db_path)) as conn:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            raise EvalError(f"run {run_id} is not in {db_path}")
        quality = conn.execute(
            """
            SELECT config,
                   SUM(status = 'ok') AS n_ok,
                   SUM(status = 'error') AS n_error,
                   AVG(hit_at_1) AS hit_at_1,
                   AVG(hit_at_5) AS hit_at_5,
                   AVG(hit_at_10) AS hit_at_10,
                   AVG(reciprocal_rank) AS mrr
            FROM query_results WHERE run_id = ? GROUP BY config
            """,
            (run_id,),
        ).fetchall()
        ok = conn.execute(
            """
            SELECT config, qid, target_rank, hit_at_5, reciprocal_rank,
                   queries_issued AS queries, llm_calls,
                   prompt_tokens + completion_tokens AS tokens, method_cost, latency_s
            FROM query_results WHERE run_id = ? AND status = 'ok' ORDER BY config, qid
            """,
            (run_id,),
        ).fetchall()
        errors = conn.execute(
            "SELECT config, qid, error FROM query_results "
            "WHERE run_id = ? AND status = 'error' ORDER BY config, qid",
            (run_id,),
        ).fetchall()
        # Calls actually sent in this run, at the config prices and as billed. A row
        # without LLM calls cost nothing; one from before billing was recorded has NULL.
        spend = conn.execute(
            """
            SELECT config,
                   SUM(money_spent) AS computed,
                   SUM(COALESCE(billed_cost, CASE WHEN llm_calls = 0 THEN 0.0 END)) AS billed,
                   SUM(llm_calls > 0 AND billed_cost IS NULL) AS unrecorded
            FROM query_results WHERE run_id = ? AND status = 'ok' GROUP BY config
            """,
            (run_id,),
        ).fetchall()

    configs = sorted({r["config"] for r in quality}, key=CONFIG_ORDER.index)
    by_config: dict[str, dict[str, sqlite3.Row]] = {c: {} for c in configs}
    for row in ok:
        by_config[row["config"]][row["qid"]] = row

    effort = {
        c: {m: _spread([r[m] for r in by_config[c].values()]) for m in EFFORT} for c in configs
    }
    quality_by = {r["config"]: r for r in quality}
    spend_by = {r["config"]: r for r in spend}

    def billed(config: str) -> dict[str, Any]:
        row = spend_by.get(config)
        if row is None:  # no successful rows
            return {"computed": None, "billed": None, "unrecorded_rows": 0}
        return {
            "computed": row["computed"],
            "billed": None if row["unrecorded"] else row["billed"],
            "unrecorded_rows": row["unrecorded"],
        }

    def paired(pairs: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        return [
            _paired(a, b, by_config[a], by_config[b], seed=seed)
            for a, b in pairs
            if a in by_config and b in by_config
        ]

    def mean_of(config: str, metric: str) -> float | None:
        spread = effort[config][metric]
        return spread["mean"] if spread else None

    summary: dict[str, Any] = {
        "run_id": run_id,
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "git_dirty": None if run["git_dirty"] is None else bool(run["git_dirty"]),
        "resumed": [
            {key: r.get(key) for key in ("at", "git_commit", "git_dirty", "retry_errors")}
            for r in resumed
        ],
        "corpus_sha256": run["corpus_sha256"],
        "bench_sha256": run["bench_sha256"],
        "configs": configs,
        "quality": {
            c: {
                "n_ok": quality_by[c]["n_ok"],
                "n_error": quality_by[c]["n_error"],
                **{key: quality_by[c][key] for key in ("hit_at_1", "hit_at_5", "hit_at_10", "mrr")},
            }
            for c in configs
        },
        "effort": effort,
        "difficulty": _difficulty(by_config),
        "paired": paired(PAIRS),
        "paired_secondary": paired(SECONDARY_PAIRS),
        "cost_quality": {
            c: {
                "hit_at_5": quality_by[c]["hit_at_5"],
                "mrr": quality_by[c]["mrr"],
                "mean_method_cost": mean_of(c, "method_cost"),
                "mean_llm_calls": mean_of(c, "llm_calls"),
            }
            for c in configs
        },
        "billed": {c: billed(c) for c in configs},
        "errors": {c: sum(1 for e in errors if e["config"] == c) for c in configs},
        "error_examples": {
            c: [f"{e['qid']}: {e['error']}" for e in errors if e["config"] == c][:3]
            for c in configs
        },
    }
    return summary


def _difficulty(by_config: dict[str, dict[str, sqlite3.Row]]) -> dict[str, Any] | None:
    """Hit@5 per config within buckets of the target's B1 rank; None if B1 wasn't run."""
    if "B1" not in by_config:
        return None
    b1_bucket = {qid: bucket(row["target_rank"]) for qid, row in by_config["B1"].items()}
    counts = {b: sum(1 for v in b1_bucket.values() if v == b) for b in BUCKETS}
    table = {}
    for config, rows in by_config.items():
        cells = {}
        for b in BUCKETS:
            hits = [r["hit_at_5"] for qid, r in rows.items() if b1_bucket.get(qid) == b]
            cells[b] = {"n": len(hits), "hit_at_5": sum(hits) / len(hits) if hits else None}
        table[config] = cells
    return {"bucket_sizes": counts, "hit_at_5": table}


def _paired(
    a: str, b: str, rows_a: dict[str, sqlite3.Row], rows_b: dict[str, sqlite3.Row], *, seed: int
) -> dict[str, Any]:
    """a minus b on the queries that succeeded in both."""
    common = sorted(rows_a.keys() & rows_b.keys())
    d_hit = [rows_a[q]["hit_at_5"] - rows_b[q]["hit_at_5"] for q in common]
    d_rr = [rows_a[q]["reciprocal_rank"] - rows_b[q]["reciprocal_rank"] for q in common]
    a_only = sum(1 for d in d_hit if d > 0)
    b_only = sum(1 for d in d_hit if d < 0)
    return {
        "a": a,
        "b": b,
        "n": len(common),
        "delta_hit_at_5": float(np.mean(d_hit)) if common else None,
        "ci_hit_at_5": paired_bootstrap(d_hit, seed=seed),
        "delta_mrr": float(np.mean(d_rr)) if common else None,
        "ci_mrr": paired_bootstrap(d_rr, seed=seed),
        "mcnemar": {"a_only": a_only, "b_only": b_only, "p": mcnemar_exact(a_only, b_only)},
    }


def _f(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _ci(ci: Sequence[float] | None) -> str:
    return "–" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _paired_table(pairs: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| Comparison | n | ΔHit@5 [95% CI] | ΔMRR [95% CI] | McNemar (only A / only B) | p |",
        "|---|---|---|---|---|---|",
    ]
    if not pairs:
        lines.append("| (no comparable pair of configs in this run) | | | | | |")
    for p in pairs:
        m = p["mcnemar"]
        primary = " (primary)" if (p["a"], p["b"]) == PRIMARY_PAIR else ""
        lines.append(
            f"| {p['a']} vs {p['b']}{primary} | {p['n']} "
            f"| {_f(p['delta_hit_at_5'])} {_ci(p['ci_hit_at_5'])} "
            f"| {_f(p['delta_mrr'])} {_ci(p['ci_mrr'])} | {m['a_only']} / {m['b_only']} "
            f"| {m['p']:.3g} |"
        )
    return lines


def _resumes(resumed: Sequence[dict[str, Any]]) -> list[str]:
    """One header line with when, and at which commit, the run was resumed; none if never."""
    if not resumed:
        return []
    parts = []
    for r in resumed:
        dirty = " (uncommitted changes to tracked files)" if r["git_dirty"] else ""
        retry = ", retrying errors" if r["retry_errors"] else ""
        parts.append(f"{r['at']} at git {r['git_commit'] or 'unknown'}{dirty}{retry}")
    return ["", "Resumed " + "; ".join(parts) + "."]


def render_markdown(summary: dict[str, Any]) -> str:
    """summary.md: main, effort, difficulty, paired, cost-quality, billed and error tables."""
    dirty = " (uncommitted changes to tracked files)" if summary["git_dirty"] else ""
    provenance = (
        f"Git {summary['git_commit'] or 'unknown'}{dirty}; corpus sha256 "
        f"{summary['corpus_sha256'][:12]}; bench sha256 {summary['bench_sha256'][:12]}."
    )
    lines = [
        f"# Run {summary['run_id']}",
        "",
        provenance,
        *_resumes(summary["resumed"]),
        "",
        "## Retrieval quality",
        "",
        "| Config | Queries | Errors | Hit@1 | Hit@5 | Hit@10 | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in summary["configs"]:
        q = summary["quality"][c]
        lines.append(
            f"| {c} | {q['n_ok']} | {q['n_error']} | {_f(q['hit_at_1'])} | {_f(q['hit_at_5'])} "
            f"| {_f(q['hit_at_10'])} | {_f(q['mrr'])} |"
        )
    if any(c in SECONDARY_CONFIGS for c in summary["configs"]):
        lines += [
            "",
            (
                "B4-D and B5-D are secondary: B4 and B5 with B2 (dense) as the base retriever "
                "instead of B3."
            ),
        ]
    lines += [
        "",
        "## Effort per query (mean / p50 / p95)",
        "",
        "| Config | Queries issued | LLM calls | Tokens | Method cost ($) | Latency (s) |",
        "|---|---|---|---|---|---|",
    ]
    digits = {"queries": 1, "llm_calls": 1, "tokens": 0, "method_cost": 5, "latency_s": 3}
    for c in summary["configs"]:
        cells = []
        for m in EFFORT:
            s, d = summary["effort"][c][m], digits[m]
            stats = "–" if s is None else " / ".join(_f(s[k], d) for k in ("mean", "p50", "p95"))
            cells.append(stats)
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines += ["", "## Hit@5 by difficulty (the target's rank in B1)", ""]
    difficulty = summary["difficulty"]
    if difficulty is None:
        lines.append("B1 was not part of this run, so there is no difficulty breakdown.")
    else:
        sizes = difficulty["bucket_sizes"]
        lines.append("| Config | " + " | ".join(f"{b} (n={sizes[b]})" for b in BUCKETS) + " |")
        lines.append("|---|" + "---|" * len(BUCKETS))
        for c in summary["configs"]:
            cells = difficulty["hit_at_5"][c]
            lines.append(f"| {c} | " + " | ".join(_f(cells[b]["hit_at_5"]) for b in BUCKETS) + " |")

    lines += ["", "## Paired comparisons (queries that succeeded in both configs)", ""]
    lines += _paired_table(summary["paired"])
    if summary["paired_secondary"]:
        lines += ["", "### Secondary comparisons (dense base)", ""]
        lines += _paired_table(summary["paired_secondary"])
    lines += [
        "",
        "## Cost and quality",
        "",
        "| Config | Hit@5 | MRR | Mean method cost per query ($) | Mean LLM calls |",
        "|---|---|---|---|---|",
    ]
    for c in summary["configs"]:
        cq = summary["cost_quality"][c]
        lines.append(
            f"| {c} | {_f(cq['hit_at_5'])} | {_f(cq['mrr'])} | {_f(cq['mean_method_cost'], 5)} "
            f"| {_f(cq['mean_llm_calls'], 1)} |"
        )
    lines += [
        "",
        "## Billed cost",
        "",
        (
            "The tables above use computed cost: config prices times tokens for every LLM call, "
            "cache hits included, so configs compare on what the method costs. Below are only "
            "the calls this run actually sent (cache hits cost nothing), at the config prices "
            "and as the API billed them; the two differ when a provider bills other than its "
            "listed price. Rows run before billed costs were recorded have none (–)."
        ),
        "",
        "| Config | Calls sent, computed ($) | Billed ($) | Billed vs computed |",
        "|---|---|---|---|",
    ]
    for c in summary["configs"]:
        b = summary["billed"][c]
        billed, ratio = "–", "–"
        if b["billed"] is not None:
            billed = _f(b["billed"], 5)
            if b["computed"]:
                ratio = f"{b['billed'] / b['computed'] - 1:+.0%}"
        elif b["unrecorded_rows"]:
            billed = f"– ({b['unrecorded_rows']} rows not recorded)"
        lines.append(f"| {c} | {_f(b['computed'], 5)} | {billed} | {ratio} |")
    lines += ["", "## Errors", "", "| Config | Errors | Examples |", "|---|---|---|"]
    for c in summary["configs"]:
        examples = "; ".join(summary["error_examples"][c]).replace("|", "\\|")
        lines.append(f"| {c} | {summary['errors'][c]} | {examples} |")
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    (run_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / SUMMARY_MD).write_text(render_markdown(summary), encoding="utf-8")


def report(run_dir: Path) -> dict[str, Any]:
    """Regenerate summary.json and summary.md of a run from the results database."""
    meta_path = run_dir / RUN_FILE
    if not meta_path.exists():
        raise EvalError(f"{run_dir} is not a run directory (no {RUN_FILE})")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = summarize(
        Path(meta["results_db"]),
        meta["run_id"],
        seed=meta["seed"],
        resumed=meta.get("resumed", []),
    )
    write_summary(run_dir, summary)
    return summary


# --- running -----------------------------------------------------------------


def load_bench(
    bench_dir: Path, corpus_sha: str, corpus_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Accepted queries (by qid), bench stats and sha256 of test_a.jsonl, after integrity checks."""
    path, stats_path = bench_dir / ACCEPTED_FILE, bench_dir / STATS_FILE
    if not path.exists() or not stats_path.exists():
        raise EvalError(f"no benchmark in {bench_dir}: run `python -m rk make-bench` first")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("corpus_sha256") != corpus_sha:
        raise EvalError(f"the benchmark in {bench_dir} was made from another corpus")
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    rows.sort(key=lambda r: r["qid"])
    if not rows:
        raise EvalError(f"the benchmark in {bench_dir} has no accepted queries")
    unknown = [r["qid"] for r in rows if r["target_id"] not in corpus_ids]
    if unknown:
        raise EvalError(f"benchmark targets missing from the corpus: {unknown[:5]}")
    return rows, stats, hashlib.sha256(data).hexdigest()


MAX_UNTRACKED_LISTED = 200


def git_state(cwd: Path | None = None) -> tuple[str | None, bool | None, list[str] | None]:
    """HEAD commit, whether tracked files have uncommitted changes, and the untracked files.

    Untracked files (not git-ignored) don't make the tree dirty; they are only
    listed, as information. Everything is None if git isn't available.
    """

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=cwd
        )
        return done.stdout

    try:
        head = git("rev-parse", "HEAD").strip()
        tracked = git("status", "--porcelain", "--untracked-files=no")
        untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None, None, None
    return head, bool(tracked.strip()), sorted(untracked)[:MAX_UNTRACKED_LISTED]


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _fingerprint(meta: dict[str, Any]) -> dict[str, Any]:
    """What a resumed run must share with the original (the budget cap may change)."""
    config = meta["config"]
    return {
        "corpus_sha256": meta["corpus_sha256"],
        "bench_sha256": meta["bench_sha256"],
        "index": {k: meta["index"].get(k) for k in ("corpus_sha256", *ENCODER_SETTINGS)},
        "retrieval": config["retrieval"],
        "agent": config["agent"],
        "retrieval_prompts": meta["prompt_hashes"]["retrieval"],
    }


def run_eval(
    cfg: Config,
    agent: ChatModel,
    encoder: Encoder,
    *,
    configs: Sequence[str] | None = None,
    limit: int | None = None,
    retry_errors: bool = False,
    run_dir: Path | None = None,
    log_mlflow: bool = True,
) -> Path:
    """Run (or resume) an evaluation; return its run directory.

    Finished rows are skipped; with retry_errors, rows that ended in an LLM
    error are run again. An LLMError marks its row as an error and the run goes
    on; other exceptions (BudgetExceeded included) stop it, after the summary
    of what finished has been written.
    """
    corpus, _, corpus_manifest = load_frozen_corpus(cfg.corpus.out_dir)
    corpus_sha = corpus_manifest["files"][CORPUS_FILE]["sha256"]
    index = load_index(cfg)
    queries, bench_stats, bench_sha = load_bench(
        cfg.bench.out_dir, corpus_sha, {r["id"] for r in corpus}
    )
    queries = queries[:limit] if limit else queries
    selected = list(configs) if configs else list(CONFIG_ORDER)
    unknown = [c for c in selected if c not in CONFIG_ORDER]
    if unknown:
        raise EvalError(f"unknown configs {unknown}; choose from {list(CONFIG_ORDER)}")
    retrievers = make_retrievers(cfg, index, encoder, agent)

    commit, dirty, untracked = git_state()
    meta: dict[str, Any] = {
        "git_commit": commit,
        "git_dirty": dirty,  # tracked files only
        "git_untracked": untracked,  # information, not part of "dirty"
        "corpus_sha256": corpus_sha,
        "bench_sha256": bench_sha,
        "index": index.manifest,
        "prompt_hashes": {
            "bench": bench_stats.get("prompt_hashes", {}),
            "retrieval": {name: load_prompt(name).sha256 for name in RETRIEVAL_PROMPTS},
        },
        "models": {
            "paraphraser": cfg.paraphraser.name,
            "judge": cfg.judge.name,
            "agent": cfg.agent.name,
            **{key: index.manifest.get(key) for key in ENCODER_SETTINGS},
        },
        "config": _jsonable(dataclasses.asdict(cfg)),
        "seed": cfg.run.seed,
        "results_db": str(cfg.run.results_db.resolve()),
    }
    conn = connect(cfg.run.results_db)
    try:
        run_dir, run_meta = _open_run(conn, cfg, meta, run_dir, retry_errors=retry_errors)
        run_id = run_meta["run_id"]
        done = {
            (r["config"], r["qid"]): r["status"]
            for r in conn.execute(
                "SELECT config, qid, status FROM query_results WHERE run_id = ?", (run_id,)
            )
        }
        try:
            for config in selected:
                retriever = retrievers[config]
                for query in queries:
                    status = done.get((config, query["qid"]))
                    if status == "ok" or (status == "error" and not retry_errors):
                        continue
                    try:
                        row = ok_row(run_id, config, query, retriever.retrieve(query["query"]))
                    except LLMError as exc:
                        row = error_row(run_id, config, query, exc)
                    _put(conn, row)
        finally:
            export_jsonl(conn, run_id, run_dir)
            summary = summarize(
                cfg.run.results_db,
                run_id,
                seed=cfg.run.seed,
                resumed=run_meta.get("resumed", []),
            )
            write_summary(run_dir, summary)
    finally:
        conn.close()
    if log_mlflow:
        run_meta["mlflow"] = log_to_mlflow(run_dir, run_meta, cfg.run)
        (run_dir / RUN_FILE).write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    return run_dir


def _open_run(
    conn: sqlite3.Connection,
    cfg: Config,
    meta: dict[str, Any],
    run_dir: Path | None,
    *,
    retry_errors: bool,
) -> tuple[Path, dict[str, Any]]:
    """Create a new run directory and `runs` row, or check and reopen an existing run."""
    if run_dir is not None:
        meta_path = run_dir / RUN_FILE
        if not meta_path.exists():
            raise EvalError(f"{run_dir} is not a run directory (no {RUN_FILE})")
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        before, now = _fingerprint(run_meta), _fingerprint(meta)
        changed = [key for key in before if before[key] != now[key]]
        if changed:
            raise EvalError(f"can't resume {run_dir}: {', '.join(changed)} changed since it started")
        resumed = {
            "at": _now(),
            "git_commit": meta["git_commit"],
            "git_dirty": meta["git_dirty"],
            "git_untracked": meta["git_untracked"],
            "retry_errors": retry_errors,
        }
        run_meta.setdefault("resumed", []).append(resumed)
        meta_path.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
        return run_dir, run_meta
    if retry_errors:
        raise EvalError("--retry-errors needs the run directory to retry (--run-dir)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id, n = stamp, 1
    while (cfg.run.out_dir / run_id).exists():
        n += 1
        run_id = f"{stamp}-{n}"
    run_dir = cfg.run.out_dir / run_id
    (run_dir / PER_QUERY_DIR).mkdir(parents=True)
    run_meta = {"run_id": run_id, "created_at": _now(), **meta}
    (run_dir / RUN_FILE).write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    (run_dir / CONFIG_FILE).write_text(yaml.safe_dump(meta["config"], sort_keys=False), "utf-8")
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            str(run_dir.resolve()),
            run_meta["created_at"],
            meta["git_commit"],
            None if meta["git_dirty"] is None else int(meta["git_dirty"]),
            meta["corpus_sha256"],
            meta["bench_sha256"],
            json.dumps(run_meta),
        ),
    )
    conn.commit()
    return run_dir, run_meta


def export_jsonl(conn: sqlite3.Connection, run_id: str, run_dir: Path) -> list[Path]:
    """Write per_query/<config>.jsonl, a portable copy of the run's rows."""
    rows = conn.execute(
        "SELECT * FROM query_results WHERE run_id = ? ORDER BY config, qid", (run_id,)
    ).fetchall()
    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        record = dict(row)
        for key in ("trace", "doc_ids"):
            record[key] = json.loads(record[key]) if record[key] else None
        by_config.setdefault(record["config"], []).append(record)
    out = run_dir / PER_QUERY_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for config, records in by_config.items():
        path = out / f"{config}.jsonl"
        lines = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
        path.write_bytes(lines.encode("utf-8"))
        paths.append(path)
    return paths


# --- MLflow ------------------------------------------------------------------


def artifact_location(tracking_uri: str) -> str | None:
    """mlartifacts/ beside a SQLite tracking database, so artifacts never land in the cwd."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    db_path = Path(tracking_uri[len(prefix) :]).resolve()
    return (db_path.parent / "mlartifacts").as_uri()


def _metrics(summary: dict[str, Any], config: str) -> dict[str, float]:
    """Flat MLflow metrics of one config (names use only MLflow-safe characters)."""
    metrics: dict[str, float | None] = dict(summary["quality"][config])
    for name, spread in summary["effort"][config].items():
        for stat, value in (spread or {}).items():
            metrics[f"{name}_{stat}"] = value
    if summary["difficulty"] is not None:
        for b, cell in summary["difficulty"]["hit_at_5"][config].items():
            metrics[f"hit_at_5_b1_{b.replace(' ', '_').replace('-', '_')}"] = cell["hit_at_5"]
    return {k: float(v) for k, v in metrics.items() if v is not None}


def _params(meta: dict[str, Any]) -> dict[str, str]:
    retrieval = meta["config"]["retrieval"]
    params = {
        "run_id": meta["run_id"],
        "git_commit": meta["git_commit"],
        "git_dirty": meta["git_dirty"],
        "corpus_sha256": meta["corpus_sha256"],
        "bench_sha256": meta["bench_sha256"],
        **{f"model.{role}": name for role, name in meta["models"].items()},
        **{
            key: retrieval[key]
            for key in (
                "max_queries",
                "max_llm_calls_agent",
                "max_queries_unconstrained",
                "max_llm_calls_agent_unconstrained",
            )
        },
        **{f"prompt.{n}": h for group in meta["prompt_hashes"].values() for n, h in group.items()},
    }
    return {k: str(v) for k, v in params.items()}


def log_to_mlflow(run_dir: Path, meta: dict[str, Any], run_cfg: RunConfig) -> dict[str, Any]:
    """One parent MLflow run per benchmark run, one child per config; returns their ids.

    Params are set when a run is created; a resumed run reuses its MLflow runs
    and updates their metrics and artifacts.
    """
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    client = MlflowClient(tracking_uri=run_cfg.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(run_cfg.mlflow_experiment)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = client.create_experiment(
            run_cfg.mlflow_experiment,
            artifact_location=artifact_location(run_cfg.mlflow_tracking_uri),
        )
    summary = json.loads((run_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    ids = dict(meta.get("mlflow") or {})
    children = dict(ids.get("children") or {})
    params = [Param(k, v) for k, v in _params(meta).items()]
    stamp = int(datetime.now(timezone.utc).timestamp() * 1000)

    parent = ids.get("parent")
    if parent is None:
        parent = client.create_run(experiment_id, run_name=meta["run_id"]).info.run_id
        client.log_batch(parent, params=params)
    paired: dict[str, float | None] = {}
    for p in summary["paired"] + summary["paired_secondary"]:
        key = f"{p['a']}_vs_{p['b']}"
        paired[f"{key}.n"] = p["n"]
        paired[f"{key}.mcnemar_p"] = p["mcnemar"]["p"]
        for metric in ("hit_at_5", "mrr"):
            paired[f"{key}.delta_{metric}"] = p[f"delta_{metric}"]
            if p[f"ci_{metric}"]:
                low, high = p[f"ci_{metric}"]
                paired[f"{key}.delta_{metric}_ci_low"] = low
                paired[f"{key}.delta_{metric}_ci_high"] = high
    client.log_batch(
        parent,
        metrics=[Metric(k, float(v), stamp, 0) for k, v in paired.items() if v is not None],
    )
    for name in (SUMMARY_MD, SUMMARY_JSON, RUN_FILE):
        client.log_artifact(parent, str(run_dir / name))

    for config in summary["configs"]:
        child = children.get(config)
        if child is None:
            child = client.create_run(
                experiment_id, run_name=config, tags={"mlflow.parentRunId": parent}
            ).info.run_id
            client.log_batch(child, params=[*params, Param("config", config)])
            children[config] = child
        metrics = _metrics(summary, config)
        client.log_batch(child, metrics=[Metric(k, v, stamp, 0) for k, v in metrics.items()])
        client.log_artifact(child, str(run_dir / PER_QUERY_DIR / f"{config}.jsonl"))
        client.set_terminated(child)
    client.set_terminated(parent)
    return {"experiment_id": experiment_id, "parent": parent, "children": children}
