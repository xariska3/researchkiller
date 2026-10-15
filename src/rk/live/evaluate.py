"""Live evaluation (docs/phase2_spec.md, Evaluation): L0, L1 and L5 on the Phase 1 queries.

Three pre-declared configurations run on each Phase 1 benchmark proposal:

- L0: OpenAlex semantic search on the proposal alone, in OpenAlex's own order
  (one call, no LLM).
- L1: the same L0 results (the same call, from the cache), reranked locally
  by the pool's hybrid ranker (Phase 1's B3 with the results as the corpus).
- L5: the full live pipeline (rk.live.pipeline.live_search): B5's agent loop
  on live search, the candidate pool and the final RRF rerank.
- L5w: the same check as L5 (one agent loop serves both), with its final pool
  ranked by the fusion chosen on the dev set (L5W_FUSION, rk.live.fusion).

A result is the original paper if one of its versions matches it
(rk.live.pool.match_original: OpenAlex id, alias, DOI or exact normalised
title). Ranks count within the top live.top_k, the list the tool shows, so
MRR is MRR@top_k for every config, and Phase 1's B5 is cut the same way when
it is compared. Each row also records which search first returned the
original (the query, 0 for the proposal, and the kind of search), or that
none did.

Queries run one at a time, each through every config, and each row goes to
the results database as it finishes (table live_query_results; the run is a
row of Phase 1's `runs` table), so a rerun into the same run directory
resumes. The OpenAlex cache is read as an evaluation: a response never
expires, so a resumed or repeated run sees the same results. When OpenAlex's
daily budget is spent, the run stops: the unfinished query gets no row, and
resuming with --run-dir after midnight UTC redoes it, its earlier searches
coming from the cache. A check over its cost cap is a finished check that
found nothing: a miss (stop_reason "cost_cap"), counted apart in the summary.
Other OpenAlex errors and LLM errors are error rows. L5w shares L5's check,
so it shares its outcome too: a check that failed for L5 is an error row in
L5w as well, and is not run again.

The comparisons were pre-declared in docs/phase2_spec.md (Evaluation): L5 vs
L1 is Step 2's primary comparison, and L5w vs L5 and L5w vs L1 the test set's,
each decided by Hit@5 with the exact McNemar test, with MRR supporting; L1 vs
L0, L5 vs L0 and L5 vs Phase 1's B5 on the frozen corpus (live.phase1_run, on
the same queries) are secondary. All use Phase 1's
statistics (paired bootstrap CIs, exact McNemar on Hit@5). The summary also
breaks Hit@5 down by Phase 1's difficulty buckets (the target's rank in that
run's B1). Runs are logged to MLflow like Phase 1 runs, in
live.mlflow_experiment.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rk.bench import qid_position
from rk.config import Config, LiveConfig
from rk.corpus import CORPUS_FILE, OpenAlexBudgetExhausted, OpenAlexError, load_frozen_corpus
from rk.evaluate import (
    BUCKETS,
    HIT_KS,
    RUN_FILE,
    SUMMARY_JSON,
    SUMMARY_MD,
    EvalError,
    _difficulty,
    _f,
    _jsonable,
    _now,
    _open_run,
    _paired,
    _paired_table,
    _resumes,
    _spread,
    error_row,
    export_jsonl,
    git_state,
    load_bench,
    log_to_mlflow,
)
from rk.evaluate import connect as _connect_results
from rk.live.fusion import PROPOSAL_EQUAL
from rk.live.openalex_search import SEARCH_COST_USD, SEMANTIC, LiveSearchClient, SearchResponse
from rk.live.pipeline import (
    AGENT_PROMPT,
    CheckCostCapExceeded,
    CheckLedger,
    LiveSearchResult,
    live_search,
    with_fusion,
)
from rk.live.pool import CandidatePool, match_original
from rk.llm import ChatModel, LLMError, load_prompt
from rk.retrieval import ENCODER_SETTINGS, Encoder, encoder_settings
from rk.targets import Record

KIND = "live-eval"  # run.json "kind" of a live evaluation run
RUN_PREFIX = "live-"
TABLE = "live_query_results"
LIVE_CONFIGS = ("L0", "L1", "L5", "L5w")
PHASE1_B5 = "P1-B5"  # Phase 1's B5 on the frozen corpus, compared on the same queries
# L5w: L5's check with this final fusion, chosen on the dev set by the pre-declared rule
# (docs/phase2_spec.md): F1 proposal-equal had the best Hit@5 (0.750, tied with F2 and
# ahead on MRR@20) on runs/live-20261014-122144, against L5's 0.700.
L5W_FUSION: str | None = PROPOSAL_EQUAL
# Pre-declared in docs/phase2_spec.md; (a, b) is a minus b. L5 vs L1 is Step 2's primary
# comparison; L5w vs L5 and L5w vs L1 are the test set's.
PRIMARY_PAIRS = (("L5", "L1"), ("L5w", "L5"), ("L5w", "L1"))
SECONDARY_PAIRS = (("L1", "L0"), ("L5", "L0"), ("L5", PHASE1_B5))
COST_CAP = "cost_cap"  # stop_reason of a check over its cost cap: a miss
CONFIG_LABELS = {
    "L0": "OpenAlex semantic search on the proposal, in OpenAlex's order (1 call, no LLM)",
    "L1": "the L0 results reranked locally (BM25 + dense, RRF)",
    "L5": "the full live pipeline (B5's agent loop on live search, pool, final rerank)",
    "L5w": "L5's check (same agent loop and pool), its final pool ranked by the fusion",
    PHASE1_B5: "Phase 1's B5 on the frozen corpus, on the same queries",
}
EFFORT = (
    "queries",
    "searches",
    "llm_calls",
    "tokens",
    "openalex_cost",
    "llm_cost",
    "method_cost",
    "latency_s",
)
FOUND_BY = (
    "proposal, semantic",
    "proposal, keyword",
    "agent query, keyword",
    "agent query, semantic",
    "not returned",
    "over its cost cap",
)
MATCHES = ("id", "alias", "doi", "title")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    config TEXT NOT NULL,
    qid TEXT NOT NULL,
    target_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error TEXT,
    target_rank INTEGER,
    match TEXT,
    hit_at_1 INTEGER,
    hit_at_5 INTEGER,
    hit_at_10 INTEGER,
    reciprocal_rank REAL,
    found_query INTEGER,
    found_kind TEXT,
    found_rank INTEGER,
    queries_issued INTEGER,
    searches INTEGER,
    searches_sent INTEGER,
    pool_size INTEGER,
    llm_calls INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    openalex_cost REAL,
    openalex_spent REAL,
    llm_cost REAL,
    llm_billed_cost REAL,
    method_cost REAL,
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
    "run_id", "config", "qid", "target_id", "status", "error", "target_rank", "match",
    "hit_at_1", "hit_at_5", "hit_at_10", "reciprocal_rank", "found_query", "found_kind",
    "found_rank", "queries_issued", "searches", "searches_sent", "pool_size", "llm_calls",
    "prompt_tokens", "completion_tokens", "openalex_cost", "openalex_spent", "llm_cost",
    "llm_billed_cost", "method_cost", "billed_cost", "latency_s", "stop_reason", "trace",
    "doc_ids", "finished_at",
)


def connect(path: Path) -> sqlite3.Connection:
    """The results database: Phase 1's tables (`runs` is shared) and live_query_results."""
    conn = _connect_results(path)
    conn.executescript(_SCHEMA)
    return conn


def _put(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    placeholders = ", ".join("?" for _ in _COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO {TABLE} ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
        [row.get(c) for c in _COLUMNS],
    )
    conn.commit()


# --- one config on one query -------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """What one config returned for one proposal, before it is scored."""

    doc_ids: list[str]  # the final ranking, best first
    ranked: list[list[Record]]  # the same works, each as its versions (search results)
    searches: list[tuple[int, SearchResponse]]  # (query index, response), in the order run
    ledger: CheckLedger
    latency_s: float
    queries_issued: int  # queries counted against the budget (repeats count)
    pool_size: int
    stop_reason: str | None
    trace: dict[str, Any]


def _search_dict(query_index: int, s: SearchResponse) -> dict[str, Any]:
    return {
        "query_index": query_index,
        "kind": s.kind,
        "n_results": s.n_results,
        "n_usable": len(s.records),
        "cost_usd": s.cost_usd,
        "cached": s.cached,
        "fetched_at": s.fetched_at.isoformat(),
    }


def _proposal_search(
    proposal: str, search: LiveSearchClient, live: LiveConfig
) -> tuple[SearchResponse, CheckLedger]:
    ledger = CheckLedger(live.cost_cap_per_check)
    ledger.require(SEARCH_COST_USD, "a semantic search")
    response = search.search(proposal, SEMANTIC)
    ledger.add_search(response)
    return response, ledger


def run_l0(proposal: str, search: LiveSearchClient, live: LiveConfig) -> Outcome:
    """L0: OpenAlex semantic search on the proposal, in OpenAlex's own order."""
    start = time.perf_counter()
    response, ledger = _proposal_search(proposal, search, live)
    records = response.records
    return Outcome(
        doc_ids=[r["id"] for r in records],
        ranked=[[r] for r in records],
        searches=[(0, response)],
        ledger=ledger,
        latency_s=time.perf_counter() - start,
        queries_issued=1,
        pool_size=len(records),
        stop_reason=None,
        trace={"queries": [proposal], "searches": [_search_dict(0, response)]},
    )


def run_l1(proposal: str, search: LiveSearchClient, live: LiveConfig, encoder: Encoder) -> Outcome:
    """L1: the L0 results, merged into a pool and reranked by its hybrid ranker."""
    start = time.perf_counter()
    response, ledger = _proposal_search(proposal, search, live)
    pool = CandidatePool(encoder, max_size=live.max_pool_size)
    pool.add(response.records, query_index=0, kind=SEMANTIC)
    by_id = {r["id"]: r for r in pool.records}
    doc_ids = pool.rank(proposal)
    return Outcome(
        doc_ids=doc_ids,
        ranked=[pool.versions(by_id[d]) for d in doc_ids],
        searches=[(0, response)],
        ledger=ledger,
        latency_s=time.perf_counter() - start,
        queries_issued=1,
        pool_size=len(pool),
        stop_reason=None,
        trace={
            "queries": [proposal],
            "searches": [_search_dict(0, response)],
            "pool_size": len(pool),
            "pool_dropped": pool.dropped,
        },
    )


def check_outcome(result: LiveSearchResult) -> Outcome:
    """L5 (the full live pipeline) or L5w (the same check with another final fusion)."""
    queries = result.trace.queries
    return Outcome(
        doc_ids=[c.record["id"] for c in result.candidates],
        ranked=[c.versions for c in result.candidates],
        searches=[(queries.index(s.query), s) for s in result.searches],
        ledger=result.ledger,
        latency_s=result.latency_s,
        queries_issued=result.trace.budget_used,
        pool_size=result.pool_size,
        stop_reason=result.trace.stop_reason,
        trace=result.to_dict(),
    )


def original_rank(
    ranked: Sequence[Sequence[Record]], original: Record, k: int
) -> tuple[int | None, str | None]:
    """The original's rank (from 1) in the top k and how it matched, or (None, None)."""
    for rank, versions in enumerate(ranked[:k], start=1):
        how = match_original(versions, original)
        if how is not None:
            return rank, how
    return None, None


def first_found(
    searches: Sequence[tuple[int, SearchResponse]], original: Record
) -> tuple[int, str, int] | None:
    """The first search, in the order run, that returned the original: (query, kind, rank)."""
    for query_index, response in searches:
        for rank, record in enumerate(response.records, start=1):
            if match_original([record], original) is not None:
                return query_index, response.kind, rank
    return None


def ok_row(
    run_id: str,
    config: str,
    query: dict[str, Any],
    original: Record,
    outcome: Outcome,
    top_k: int,
) -> dict[str, Any]:
    rank, how = original_rank(outcome.ranked, original, top_k)
    found = first_found(outcome.searches, original)
    ledger = outcome.ledger
    return {
        "run_id": run_id,
        "config": config,
        "qid": query["qid"],
        "target_id": query["target_id"],
        "status": "ok",
        "target_rank": rank,
        "match": how,
        **{f"hit_at_{k}": int(rank is not None and rank <= k) for k in HIT_KS},
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "found_query": found[0] if found else None,
        "found_kind": found[1] if found else None,
        "found_rank": found[2] if found else None,
        "queries_issued": outcome.queries_issued,
        "searches": len(outcome.searches),
        "searches_sent": sum(not s.cached for _, s in outcome.searches),
        "pool_size": outcome.pool_size,
        "llm_calls": ledger.llm.calls,
        "prompt_tokens": ledger.llm.prompt_tokens,
        "completion_tokens": ledger.llm.completion_tokens,
        "openalex_cost": ledger.openalex_cost,
        "openalex_spent": ledger.openalex_spent,
        "llm_cost": ledger.llm.method_cost,
        "llm_billed_cost": ledger.llm.billed_cost,
        "method_cost": ledger.method_cost,
        "billed_cost": ledger.billed_cost,
        "latency_s": outcome.latency_s,
        "stop_reason": outcome.stop_reason,
        "trace": json.dumps(outcome.trace, ensure_ascii=False),
        "doc_ids": json.dumps(outcome.doc_ids[:top_k]),
        "finished_at": _now(),
    }


def capped_row(
    run_id: str,
    config: str,
    query: dict[str, Any],
    exc: CheckCostCapExceeded,
    latency_s: float,
) -> dict[str, Any]:
    """A check over its cost cap: it shows nothing, so it is a miss, with what it cost."""
    ledger = exc.ledger or CheckLedger(cap=0.0)
    return {
        "run_id": run_id,
        "config": config,
        "qid": query["qid"],
        "target_id": query["target_id"],
        "status": "ok",
        "error": f"{type(exc).__name__}: {exc}",
        **{f"hit_at_{k}": 0 for k in HIT_KS},
        "reciprocal_rank": 0.0,
        "searches": ledger.searches,
        "searches_sent": ledger.searches_sent,
        "llm_calls": ledger.llm.calls,
        "prompt_tokens": ledger.llm.prompt_tokens,
        "completion_tokens": ledger.llm.completion_tokens,
        "openalex_cost": ledger.openalex_cost,
        "openalex_spent": ledger.openalex_spent,
        "llm_cost": ledger.llm.method_cost,
        "llm_billed_cost": ledger.llm.billed_cost,
        "method_cost": ledger.method_cost,
        "billed_cost": ledger.billed_cost,
        "latency_s": latency_s,
        "stop_reason": COST_CAP,
        "finished_at": _now(),
    }


# --- the Phase 1 run compared with -------------------------------------------


def phase1_rows(
    conn: sqlite3.Connection, run_id: str, bench_sha: str
) -> dict[str, dict[str, sqlite3.Row]]:
    """B1 and B5 rows (status ok) of a Phase 1 run on the same benchmark, by config and qid."""
    run = conn.execute("SELECT bench_sha256 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        raise EvalError(
            f"the Phase 1 run {run_id} (live.phase1_run) is not in the results database"
        )
    if run["bench_sha256"] != bench_sha:
        raise EvalError(f"the Phase 1 run {run_id} (live.phase1_run) used another benchmark")
    found: dict[str, dict[str, sqlite3.Row]] = {"B1": {}, "B5": {}}
    for row in conn.execute(
        "SELECT config, qid, target_rank, hit_at_1, hit_at_5, hit_at_10 FROM query_results "
        "WHERE run_id = ? AND status = 'ok' AND config IN ('B1', 'B5')",
        (run_id,),
    ):
        found[row["config"]][row["qid"]] = row
    return found


def cut_row(row: sqlite3.Row, k: int) -> dict[str, Any]:
    """A Phase 1 row scored like a live one: a rank beyond the top k is a miss."""
    rank = row["target_rank"]
    kept = rank if rank is not None and rank <= k else None
    return {
        "target_rank": kept,
        **{f"hit_at_{h}": int(kept is not None and kept <= h) for h in HIT_KS},
        "reciprocal_rank": 1.0 / kept if kept else 0.0,
    }


# --- running -----------------------------------------------------------------


@dataclass(frozen=True)
class LiveEvalRun:
    run_dir: Path
    n_queries: int  # queries asked for in this call
    n_done: int  # of them, those with a row in every selected config
    budget_stop: OpenAlexBudgetExhausted | None  # why the run stopped early, if it did


def _fingerprint(meta: dict[str, Any]) -> dict[str, Any]:
    """What a resumed live run must share with the original."""
    config = meta["config"]
    return {
        "corpus_sha256": meta["corpus_sha256"],
        "bench_sha256": meta["bench_sha256"],
        "encoder": {k: meta["models"].get(k) for k in ENCODER_SETTINGS},
        "retrieval": config["retrieval"],
        "agent": config["agent"],
        "live": {k: v for k, v in config["live"].items() if k != "cache_ttl_days"},
        "prompts": meta["prompt_hashes"],
        "l5w_fusion": meta.get("l5w_fusion"),
    }


def spent_budget(search: LiveSearchClient) -> OpenAlexBudgetExhausted | None:
    """After a failed search: the stop to make if /rate-limit says no search fits today."""
    try:
        budget = search.budget()
    except OpenAlexError:
        return None
    if budget.remaining_usd >= SEARCH_COST_USD:
        return None
    return OpenAlexBudgetExhausted(
        f"OpenAlex's daily budget is spent: ${budget.remaining_usd:.4f} left (/rate-limit)",
        resets_in_s=budget.resets_in_s,
    )


def run_live_eval(
    cfg: Config,
    agent: ChatModel,
    encoder: Encoder,
    search: LiveSearchClient,
    *,
    configs: Sequence[str] | None = None,
    from_qid: str | None = None,
    limit: int | None = None,
    retry_errors: bool = False,
    run_dir: Path | None = None,
    log_mlflow: bool = True,
) -> LiveEvalRun:
    """Run (or resume) a live evaluation of the first `limit` benchmark queries.

    The queries are in seeded (qid) order, starting at `from_qid` (e.g. A0045)
    if given. Finished rows are skipped; with retry_errors, error rows are run
    again.
    `search` should read its cache as an evaluation
    (LiveSearchClient.from_config(..., evaluation=True)). A spent OpenAlex
    budget stops the run with no row for the unfinished query; other
    exceptions (BudgetExceeded included) stop it too, after the summary of
    what finished has been written.
    """
    live = cfg.live
    if live is None:
        raise EvalError("the config has no live: section")
    if live.top_k < max(HIT_KS):
        raise EvalError(f"live.top_k must be at least {max(HIT_KS)} to measure Hit@{max(HIT_KS)}")
    unknown = [c for c in configs or () if c not in LIVE_CONFIGS]
    if unknown:
        raise EvalError(f"unknown live configs {unknown}; choose from {list(LIVE_CONFIGS)}")
    available = [c for c in LIVE_CONFIGS if c != "L5w" or L5W_FUSION is not None]
    if configs and "L5w" in configs and L5W_FUSION is None:
        raise EvalError("L5w has no final fusion yet (rk.live.evaluate.L5W_FUSION)")
    selected = [c for c in available if not configs or c in configs]

    corpus, _, corpus_manifest = load_frozen_corpus(cfg.corpus.out_dir)
    corpus_sha = corpus_manifest["files"][CORPUS_FILE]["sha256"]
    by_id = {r["id"]: r for r in corpus}
    queries, _, bench_sha = load_bench(cfg.bench.out_dir, corpus_sha, set(by_id))
    if from_qid is not None:
        first = qid_position(from_qid)
        queries = [q for q in queries if qid_position(q["qid"]) >= first]
    queries = queries[:limit] if limit else queries
    originals = {q["qid"]: by_id[q["target_id"]] for q in queries}
    del corpus, by_id

    commit, dirty, untracked = git_state()
    meta: dict[str, Any] = {
        "kind": KIND,
        "git_commit": commit,
        "git_dirty": dirty,  # tracked files only
        "git_untracked": untracked,  # information, not part of "dirty"
        "corpus_sha256": corpus_sha,
        "bench_sha256": bench_sha,
        "phase1_run": live.phase1_run,
        "l5w_fusion": L5W_FUSION,
        "prompt_hashes": {"retrieval": {AGENT_PROMPT: load_prompt(AGENT_PROMPT).sha256}},
        "models": {"agent": cfg.agent.name, **encoder_settings(encoder)},
        "encoder_device": encoder.device,
        "config": _jsonable(dataclasses.asdict(cfg)),
        "seed": cfg.run.seed,
        "results_db": str(cfg.run.results_db.resolve()),
    }
    last: dict[str, Any] = {}  # the last proposal's live check, shared by L5 and L5w

    def check(proposal: str) -> LiveSearchResult:
        if last.get("proposal") != proposal:
            last.clear()
            last["proposal"] = proposal
            try:
                last["result"] = live_search(
                    proposal, cfg, search=search, encoder=encoder, agent=agent
                )
            # The same outcome for both: a miss over the cost cap, or an error row.
            except (CheckCostCapExceeded, OpenAlexError, LLMError) as exc:
                last["failed"] = exc
        if "failed" in last:
            raise last["failed"]
        return last["result"]

    def l5w(proposal: str) -> Outcome:
        if L5W_FUSION is None or live is None:
            raise EvalError("L5w has no final fusion yet (rk.live.evaluate.L5W_FUSION)")
        return check_outcome(with_fusion(check(proposal), L5W_FUSION, encoder, live))

    runners: dict[str, Callable[[str], Outcome]] = {
        "L0": lambda proposal: run_l0(proposal, search, live),
        "L1": lambda proposal: run_l1(proposal, search, live, encoder),
        "L5": lambda proposal: check_outcome(check(proposal)),
        "L5w": l5w,
    }
    if run_dir is not None and (run_dir / RUN_FILE).exists() and not is_live_run(run_dir):
        raise EvalError(f"{run_dir} is not a live-eval run")
    stop: OpenAlexBudgetExhausted | None = None
    conn = connect(cfg.run.results_db)
    try:
        if live.phase1_run is not None:
            phase1_rows(conn, live.phase1_run, bench_sha)  # fail now, not at the summary
        run_dir, run_meta = _open_run(
            conn,
            cfg,
            meta,
            run_dir,
            retry_errors=retry_errors,
            fingerprint=_fingerprint,
            prefix=RUN_PREFIX,
        )
        run_id = run_meta["run_id"]
        done = {
            (r["config"], r["qid"]): r["status"]
            for r in conn.execute(
                f"SELECT config, qid, status FROM {TABLE} WHERE run_id = ?", (run_id,)
            )
        }
        try:
            for query in queries:
                for config in selected:
                    status = done.get((config, query["qid"]))
                    if status == "ok" or (status == "error" and not retry_errors):
                        continue
                    start = time.perf_counter()
                    try:
                        outcome = runners[config](query["query"])
                        row = ok_row(
                            run_id, config, query, originals[query["qid"]], outcome, live.top_k
                        )
                    except OpenAlexBudgetExhausted as exc:
                        stop = exc
                    except CheckCostCapExceeded as exc:  # a miss, not an error
                        row = capped_row(run_id, config, query, exc, time.perf_counter() - start)
                    except OpenAlexError as exc:
                        stop = spent_budget(search)
                        row = error_row(run_id, config, query, exc)
                    except LLMError as exc:
                        row = error_row(run_id, config, query, exc)
                    if stop is not None:
                        run_meta.setdefault("budget_stops", []).append(
                            {
                                "at": _now(),
                                "qid": query["qid"],
                                "config": config,
                                "resets_in_s": stop.resets_in_s,
                                "message": str(stop),
                            }
                        )
                        (run_dir / RUN_FILE).write_text(
                            json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
                        )
                        break
                    _put(conn, row)
                    done[(config, query["qid"])] = row["status"]
                if stop is not None:
                    break
        finally:
            export_jsonl(conn, run_id, run_dir, table=TABLE)
            write_summary(run_dir, summarize(cfg.run.results_db, run_meta, seed=cfg.run.seed))
    finally:
        conn.close()
    if log_mlflow:
        run_cfg = dataclasses.replace(cfg.run, mlflow_experiment=live.mlflow_experiment)
        extra = {f"live.{k}": v for k, v in run_meta["config"]["live"].items()}
        run_meta["mlflow"] = log_to_mlflow(run_dir, run_meta, run_cfg, extra_params=extra)
        (run_dir / RUN_FILE).write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    n_done = sum(all((c, q["qid"]) in done for c in selected) for q in queries)
    return LiveEvalRun(run_dir, len(queries), n_done, stop)


def duration(seconds: float | None) -> str:
    """'5h 07m', or 'an unknown time'."""
    if seconds is None:
        return "an unknown time"
    minutes = max(0, round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m"


def describe_budget(search: LiveSearchClient) -> str:
    """One line on today's OpenAlex budget, from /rate-limit."""
    try:
        budget = search.budget()
    except OpenAlexError as exc:
        return f"OpenAlex budget unknown ({exc})"
    daily = "" if budget.daily_usd is None else f" of ${budget.daily_usd:g}"
    return (
        f"OpenAlex budget left today: ${budget.remaining_usd:.4f}{daily}; it resets in "
        f"{duration(budget.resets_in_s)} (midnight UTC)"
    )


# --- summary -----------------------------------------------------------------


def _found_by(row: sqlite3.Row) -> str:
    if row["stop_reason"] == COST_CAP:
        return "over its cost cap"
    if row["found_query"] is None:
        return "not returned"
    who = "proposal" if row["found_query"] == 0 else "agent query"
    return f"{who}, {row['found_kind']}"


def _effort_value(row: sqlite3.Row, metric: str) -> float | None:
    """One effort measure of a row; None where a check over its cost cap didn't record it."""
    if metric == "queries":
        return row["queries_issued"]
    if metric == "tokens":
        return row["prompt_tokens"] + row["completion_tokens"]
    return row[metric]


def _effort(rows: Sequence[sqlite3.Row], metric: str) -> dict[str, float] | None:
    return _spread([v for r in rows if (v := _effort_value(r, metric)) is not None])


def summarize(db_path: Path, meta: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Every table of a live run's summary, computed from the results database."""
    run_id = meta["run_id"]
    top_k = meta["config"]["live"]["top_k"]
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT * FROM {TABLE} WHERE run_id = ? ORDER BY config, qid", (run_id,)
        ).fetchall()
        phase1 = meta.get("phase1_run")
        reference = phase1_rows(conn, phase1, meta["bench_sha256"]) if phase1 else None

    configs = [c for c in LIVE_CONFIGS if any(r["config"] == c for r in rows)]
    ok: dict[str, dict[str, Any]] = {c: {} for c in configs}
    errors: dict[str, list[sqlite3.Row]] = {c: [] for c in configs}
    for row in rows:
        if row["status"] == "ok":
            ok[row["config"]][row["qid"]] = row
        else:
            errors[row["config"]].append(row)
    qids = sorted({r["qid"] for r in rows})
    scored: dict[str, dict[str, Any]] = dict(ok)
    compared = list(configs)
    difficulty = None
    if reference is not None:
        scored[PHASE1_B5] = {
            q: cut_row(reference["B5"][q], top_k) for q in qids if q in reference["B5"]
        }
        compared.append(PHASE1_B5)
        b1 = {q: reference["B1"][q] for q in qids if q in reference["B1"]}
        difficulty = _difficulty({"B1": b1, **{c: scored[c] for c in compared}}) if b1 else None
        if difficulty is not None:
            del difficulty["hit_at_5"]["B1"]  # the buckets' own config isn't compared

    def quality(config: str) -> dict[str, Any]:
        values = list(scored[config].values())

        def mean(key: str) -> float | None:
            return sum(v[key] for v in values) / len(values) if values else None

        return {
            "n_ok": len(values),
            "n_error": len(errors.get(config, [])),
            "n_cost_cap": sum(
                1 for r in ok.get(config, {}).values() if r["stop_reason"] == COST_CAP
            ),
            **{f"hit_at_{k}": mean(f"hit_at_{k}") for k in HIT_KS},
            "mrr": mean("reciprocal_rank"),
        }

    def paired(pairs: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        return [
            _paired(a, b, scored[a], scored[b], seed=seed)
            for a, b in pairs
            if a in scored and b in scored
        ]

    found = {}
    for c in configs:
        cells = {label: {"n": 0, "in_top_k": 0} for label in FOUND_BY}
        for row in ok[c].values():
            cell = cells[_found_by(row)]
            cell["n"] += 1
            cell["in_top_k"] += int(row["target_rank"] is not None)
        found[c] = cells

    return {
        "run_id": run_id,
        "kind": KIND,
        "created_at": meta["created_at"],
        "git_commit": meta["git_commit"],
        "git_dirty": meta["git_dirty"],
        "resumed": [
            {key: r.get(key) for key in ("at", "git_commit", "git_dirty", "retry_errors")}
            for r in meta.get("resumed", [])
        ],
        "budget_stops": [
            {key: s.get(key) for key in ("at", "qid", "config", "resets_in_s")}
            for s in meta.get("budget_stops", [])
        ],
        "corpus_sha256": meta["corpus_sha256"],
        "bench_sha256": meta["bench_sha256"],
        "phase1_run": phase1,
        "l5w_fusion": meta.get("l5w_fusion"),
        "top_k": top_k,
        "configs": configs,
        "compared": compared,
        "n_queries": len(qids),
        "first_qid": qids[0] if qids else None,
        "last_qid": qids[-1] if qids else None,
        "quality": {c: quality(c) for c in compared},
        "effort": {c: {m: _effort(list(ok[c].values()), m) for m in EFFORT} for c in configs},
        "difficulty": difficulty,
        "paired": paired(PRIMARY_PAIRS),
        "paired_secondary": paired(SECONDARY_PAIRS),
        "found_by": found,
        "matches": {
            c: {m: sum(1 for r in ok[c].values() if r["match"] == m) for m in MATCHES}
            for c in configs
        },
        "spent": {
            c: {
                "searches_sent": sum(r["searches_sent"] for r in ok[c].values()),
                "openalex": sum(r["openalex_spent"] for r in ok[c].values()),
                "llm_billed": sum(r["llm_billed_cost"] for r in ok[c].values()),
            }
            for c in configs
        },
        "errors": {c: len(errors[c]) for c in configs},
        "error_examples": {
            c: [f"{e['qid']}: {e['error']}" for e in errors[c]][:3] for c in configs
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """summary.md of a live run."""
    k = summary["top_k"]
    dirty = " (uncommitted changes to tracked files)" if summary["git_dirty"] else ""
    reference = summary["phase1_run"] or "none"
    lines = [
        f"# Live run {summary['run_id']}",
        "",
        (
            f"Git {summary['git_commit'] or 'unknown'}{dirty}; corpus sha256 "
            f"{summary['corpus_sha256'][:12]}; bench sha256 {summary['bench_sha256'][:12]}; "
            f"Phase 1 run compared: {reference}."
        ),
        *_resumes(summary["resumed"]),
    ]
    stops = summary["budget_stops"]
    if stops:
        when = "; ".join(f"{s['at']} (at {s['qid']}, {s['config']})" for s in stops)
        lines += ["", f"Stopped when OpenAlex's daily budget was spent: {when}."]
    lines += [
        "",
        *(
            f"- {c}: {CONFIG_LABELS[c]}"
            + (f" {summary['l5w_fusion']} (rk.live.fusion)" if c == "L5w" else "")
            for c in summary["compared"]
        ),
        "",
        (
            f"Ranks count within the top {k} (the list the tool shows), so MRR is MRR@{k} for "
            f"every row, {PHASE1_B5} included. {summary['n_queries']} queries have results "
            f"({summary['first_qid'] or '–'} to {summary['last_qid'] or '–'} in seeded order)."
        ),
        "",
        "## Retrieval quality",
        "",
        (
            "A check over its cost cap shows nothing, so it counts as a miss; such checks are "
            "counted under Cost-cap misses and included in Queries."
        ),
        "",
        f"| Config | Queries | Errors | Cost-cap misses | Hit@1 | Hit@5 | Hit@10 | MRR@{k} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in summary["compared"]:
        q = summary["quality"][c]
        lines.append(
            f"| {c} | {q['n_ok']} | {q['n_error']} | {q['n_cost_cap']} | {_f(q['hit_at_1'])} "
            f"| {_f(q['hit_at_5'])} | {_f(q['hit_at_10'])} | {_f(q['mrr'])} |"
        )
    lines += [
        "",
        "## Effort per query (mean / p50 / p95)",
        "",
        (
            "OpenAlex and LLM costs are method costs: what each call costs, cached or not. When "
            "L0 runs first, L1 and L5 read its proposal search from the cache, so their latency "
            "leaves out that OpenAlex call."
        ),
        "",
        (
            "| Config | Queries | Searches | LLM calls | Tokens | OpenAlex cost ($) | LLM cost ($) "
            "| Method cost ($) | Latency (s) |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    digits = {"queries": 1, "searches": 1, "llm_calls": 1, "tokens": 0, "latency_s": 2}
    for c in summary["configs"]:
        cells = []
        for m in EFFORT:
            s, d = summary["effort"][c][m], digits.get(m, 5)
            stats = (_f(s[x], d) for x in ("mean", "p50", "p95")) if s else ()
            cells.append(" / ".join(stats) or "–")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines += ["", "## Hit@5 by difficulty (the target's rank in Phase 1's B1)", ""]
    difficulty = summary["difficulty"]
    if difficulty is None:
        lines.append("No Phase 1 B1 rows for these queries, so there is no difficulty breakdown.")
    else:
        sizes = difficulty["bucket_sizes"]
        lines.append("| Config | " + " | ".join(f"{b} (n={sizes[b]})" for b in BUCKETS) + " |")
        lines.append("|---|" + "---|" * len(BUCKETS))
        for c in summary["compared"]:
            cells = difficulty["hit_at_5"][c]
            lines.append(f"| {c} | " + " | ".join(_f(cells[b]["hit_at_5"]) for b in BUCKETS) + " |")

    lines += [
        "",
        "## Pre-declared comparisons (queries that succeeded in both configs)",
        "",
        *_paired_table(summary["paired"]),
        "",
        (
            "Pre-declared in docs/phase2_spec.md: L5 vs L1 (Step 2's primary comparison), and "
            "on the test set L5w vs L5 and L5w vs L1. Hit@5 with the exact McNemar test "
            "decides; MRR supports."
        ),
        "",
        "## Secondary comparisons (exploratory)",
        "",
        *_paired_table(summary["paired_secondary"]),
        "",
        "## Which search first returned the original",
        "",
        f"Each cell: queries, and in brackets those with the original in the top {k}.",
        "",
        "| Config | " + " | ".join(FOUND_BY) + " |",
        "|---|" + "---|" * len(FOUND_BY),
    ]
    for c in summary["configs"]:
        cells = summary["found_by"][c]
        lines.append(
            f"| {c} | "
            + " | ".join(f"{cells[b]['n']} ({cells[b]['in_top_k']})" for b in FOUND_BY)
            + " |"
        )
    lines += [
        "",
        f"## How the original matched (in the top {k})",
        "",
        "| Config | " + " | ".join(MATCHES) + " |",
        "|---|" + "---|" * len(MATCHES),
    ]
    for c in summary["configs"]:
        counts = summary["matches"][c]
        lines.append(f"| {c} | " + " | ".join(str(counts[m]) for m in MATCHES) + " |")
    lines += [
        "",
        "## Money spent",
        "",
        (
            "Only calls this run sent (cache hits cost nothing): OpenAlex searches from its "
            "daily budget, and LLM calls as billed."
        ),
        "",
        "| Config | OpenAlex searches sent | OpenAlex ($) | LLM billed ($) |",
        "|---|---|---|---|",
    ]
    for c in summary["configs"]:
        s = summary["spent"][c]
        lines.append(
            f"| {c} | {s['searches_sent']} | {_f(s['openalex'], 4)} | {_f(s['llm_billed'], 4)} |"
        )
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


def is_live_run(run_dir: Path) -> bool:
    meta_path = run_dir / RUN_FILE
    if not meta_path.exists():
        return False
    return json.loads(meta_path.read_text(encoding="utf-8")).get("kind") == KIND


def report(run_dir: Path) -> dict[str, Any]:
    """Regenerate summary.json and summary.md of a live run from the results database."""
    meta = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    summary = summarize(Path(meta["results_db"]), meta, seed=meta["seed"])
    write_summary(run_dir, summary)
    return summary
