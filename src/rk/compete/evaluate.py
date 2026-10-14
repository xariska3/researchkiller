"""python -m rk compete: the automated head-to-head (docs/phase2_spec.md, Competitive evaluation).

Queries: compete.n_queries (100) Phase 1 queries drawn from those the live-eval runs
in compete.runs checked (the dev set and the test set), stratified by Phase 1's
difficulty buckets (the target's rank in live.phase1_run's B1: 1, 2-10, 11-100, not
found). Each bucket gets its share of the sample in proportion to its size (largest
remainders; ties go to the earlier bucket), filled by a seeded random draw (run.seed).
The sample is then put in a seeded random order, so --limit N takes N queries spread
over the buckets.

Systems: "ours" is the live tool as L5w (L5's check, its final pool ranked by the
fusion chosen on the dev set), replayed from each run's cached OpenAlex responses and
recorded agent queries (rk.live.replay): no OpenAlex or LLM calls. Its list is the
top 10; its cost and latency are the check's as the run recorded them (L5w's row when
the run has one, else L5's: the same check, ranked again in milliseconds). A check
over its cost cap lists nothing. Every other system answers through
rk.compete.systems.

Scoring: a listed paper is the original if its DOI, or the OpenAlex id or DOI it
resolved to, or its normalised title as listed or as resolved, matches the original
(rk.live.pool.match_original); the first such paper gives the rank, hence Hit@1/5/10
and MRR@10. Each paper the other systems list goes through rk.compete.resolve, and the
share that no free route verifies is the unresolved (possibly fabricated) share, with
DOI mismatches counted apart. Ours lists OpenAlex records, so it has none by
construction. An answer without the JSON asked for lists nothing (counted as a parse
failure). A system whose cost cap is reached gets no answer for the rest of the run,
and a call that fails after its retries is an error; both are counted apart and left
out of the statistics.

Statistics: Phase 1's (rk.evaluate._paired): ours against each other system on the
queries both answered, ΔHit@5 and ΔMRR@10 with paired bootstrap 95% CIs and the exact
McNemar test on Hit@5, with Holm-adjusted p-values across the systems; Hit@5 by
difficulty bucket.

A run writes runs/compete-<UTC time>/: run.json (sample, config, prompt hash, git
state), results.jsonl (one row per system and query, each listed paper with how it
resolved), summary.json and summary.md. Responses and lookups are cached in
compete.cache_path, so repeating or rerunning after a crash costs nothing for what was
already asked.
"""

from __future__ import annotations

import dataclasses
import json
import random
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rk.compete.resolve import UNRESOLVED, VERIFIED, Resolution, Resolver
from rk.compete.systems import (
    MAX_ITEMS,
    PROMPT,
    Answer,
    CompeteError,
    Item,
    ResponseCache,
    SpendLedger,
    System,
    SystemCapReached,
    ask,
    request_key,
)
from rk.config import Config
from rk.corpus import CORPUS_FILE, load_frozen_corpus
from rk.evaluate import (
    BUCKETS,
    HIT_KS,
    RUN_FILE,
    SUMMARY_JSON,
    SUMMARY_MD,
    EvalError,
    _f,
    _jsonable,
    _now,
    _paired,
    _spread,
    bucket,
    git_state,
    load_bench,
)
from rk.live import evaluate as live_evaluate
from rk.live.fusion import final_ranking
from rk.live.pool import match_original
from rk.live.replay import load_run, replay_run
from rk.llm import load_prompt
from rk.retrieval import Encoder
from rk.targets import Record

KIND = "compete"
RUN_PREFIX = "compete-"
OURS = "ours"
RESULTS_FILE = "results.jsonl"
EFFORT = ("cost", "latency_s", "searches")


@dataclass(frozen=True)
class Query:
    qid: str
    target_id: str
    proposal: str
    bucket: str  # Phase 1 difficulty: the target's rank in live.phase1_run's B1
    run_dir: Path  # the live-eval run that checked it


@dataclass(frozen=True)
class Plan:
    frame: list[Query]  # every query the runs checked
    quotas: dict[str, int]  # the sample's size per bucket
    sample: list[Query]  # in seeded order, cut to --limit
    originals: dict[str, Record]  # the sample's original papers, by target id
    systems: list[str]  # selected, in config order, ours first
    corpus_sha256: str
    bench_sha256: str


# --- the sample --------------------------------------------------------------------------


def quotas_for(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """n split over buckets in proportion to their sizes (largest remainders, ties in order)."""
    total = sum(sizes.values())
    n = min(n, total)
    if total == 0:
        return {b: 0 for b in sizes}
    exact = {b: n * s / total for b, s in sizes.items()}
    quotas = {b: int(e) for b, e in exact.items()}
    order = sorted(sizes, key=lambda b: (quotas[b] - exact[b], list(sizes).index(b)))
    for b in order[: n - sum(quotas.values())]:
        quotas[b] += 1
    return quotas


def stratified_sample(
    frame: Sequence[Query], n: int, seed: int
) -> tuple[list[Query], dict[str, int]]:
    """A seeded sample of n queries stratified by bucket, in a seeded random order."""
    ordered = sorted(frame, key=lambda q: q.qid)
    by_bucket = {b: [q for q in ordered if q.bucket == b] for b in BUCKETS}
    quotas = quotas_for({b: len(v) for b, v in by_bucket.items()}, n)
    rng = random.Random(seed)
    sample = [q for b in BUCKETS for q in rng.sample(by_bucket[b], quotas[b])]
    rng.shuffle(sample)
    return sample, quotas


def _checked_rows(meta: Mapping[str, Any]) -> dict[str, dict[str, sqlite3.Row]]:
    """A live run's finished rows (status ok, cost-cap misses included) by config and qid."""
    with closing(live_evaluate.connect(Path(meta["results_db"]))) as conn:
        rows = conn.execute(
            f"SELECT * FROM {live_evaluate.TABLE} WHERE run_id = ? AND status = 'ok'",
            (meta["run_id"],),
        ).fetchall()
    found: dict[str, dict[str, sqlite3.Row]] = {}
    for row in rows:
        found.setdefault(row["config"], {})[row["qid"]] = row
    return found


def make_plan(
    cfg: Config, *, systems: Sequence[str] | None = None, limit: int | None = None
) -> Plan:
    """The sample and the systems to ask: no calls are made."""
    compete, live = cfg.compete, cfg.live
    if compete is None:
        raise EvalError("the config has no compete: section")
    if live is None or live.phase1_run is None:
        raise EvalError("compete needs live.phase1_run, whose B1 gives the difficulty buckets")
    names = [OURS, *(s.name for s in compete.systems)]
    unknown = [s for s in systems or () if s not in names]
    if unknown:
        raise EvalError(f"unknown systems {unknown}; choose from {names}")
    chosen = [n for n in names if not systems or n in systems]

    corpus, _, manifest = load_frozen_corpus(cfg.corpus.out_dir)
    corpus_sha = manifest["files"][CORPUS_FILE]["sha256"]
    by_id = {r["id"]: r for r in corpus}
    queries, _, bench_sha = load_bench(cfg.bench.out_dir, corpus_sha, set(by_id))
    by_qid = {q["qid"]: q for q in queries}
    with closing(live_evaluate.connect(cfg.run.results_db)) as conn:
        b1 = live_evaluate.phase1_rows(conn, live.phase1_run, bench_sha)["B1"]
    frame: dict[str, Query] = {}
    for run_dir in compete.runs:
        meta, _, _ = load_run(run_dir)
        if meta["bench_sha256"] != bench_sha:
            raise EvalError(f"{run_dir} ran on another benchmark")
        for qid in sorted(_checked_rows(meta).get("L5", {})):
            if qid in frame:
                continue
            if qid not in b1:
                raise EvalError(f"{qid} has no B1 row in the Phase 1 run {live.phase1_run}")
            q = by_qid[qid]
            difficulty = bucket(b1[qid]["target_rank"])
            frame[qid] = Query(qid, q["target_id"], q["query"], difficulty, run_dir)
    sample, quotas = stratified_sample(list(frame.values()), compete.n_queries, cfg.run.seed)
    sample = sample[:limit] if limit else sample
    originals = {q.target_id: by_id[q.target_id] for q in sample}
    return Plan(list(frame.values()), quotas, sample, originals, chosen, corpus_sha, bench_sha)


# --- ours: the live tool, replayed -------------------------------------------------------


@dataclass(frozen=True)
class OursList:
    items: list[Item]
    target_rank: int | None
    match: str | None
    cost: float
    latency_s: float
    capped: bool  # the check went over its cost cap, so it lists nothing
    source: str  # the row its cost and latency come from
    reproduced: bool | None  # the replay equals the run's recorded L5w ranking, if it has one


def _our_item(record: Record) -> Item:
    return Item(
        record["title"],
        int(record["publication_date"][:4]),
        record.get("doi"),
        f"https://openalex.org/{record['id']}",
    )


def ours_lists(
    sample: Sequence[Query], originals: Mapping[str, Record], encoder: Encoder
) -> dict[str, OursList]:
    """L5w's top 10 for each sampled query, rebuilt from its run's cache (no calls)."""
    fusion = live_evaluate.L5W_FUSION
    if fusion is None:
        raise EvalError("L5w has no final fusion (rk.live.evaluate.L5W_FUSION)")
    by_run: dict[Path, list[Query]] = {}
    for q in sample:
        by_run.setdefault(q.run_dir, []).append(q)
    out: dict[str, OursList] = {}
    for run_dir, queries in by_run.items():
        meta, _, replays = replay_run(run_dir, encoder, qids={q.qid for q in queries})
        top_k = meta["config"]["live"]["top_k"]
        rows = _checked_rows(meta)
        replayed = {rp.qid: rp for rp in replays}
        for q in queries:
            l5w = rows.get("L5w", {}).get(q.qid)
            row = l5w if l5w is not None else rows["L5"][q.qid]
            source = f"{meta['run_id']} {'L5w' if l5w is not None else 'L5'}"
            cost, latency = row["method_cost"], row["latency_s"]
            if row["stop_reason"] == live_evaluate.COST_CAP:
                out[q.qid] = OursList([], None, None, cost, latency, True, source, None)
                continue
            rp = replayed[q.qid]
            ranking = final_ranking(rp.pool, rp.queries, fusion)
            by_id = {r["id"]: r for r in rp.pool.records}
            shown = ranking[:MAX_ITEMS]
            versions = [rp.pool.versions(by_id[d]) for d in shown]
            rank, how = live_evaluate.original_rank(versions, originals[q.target_id], MAX_ITEMS)
            reproduced = None if l5w is None else ranking[:top_k] == json.loads(l5w["doc_ids"])
            items = [_our_item(by_id[d]) for d in shown]
            out[q.qid] = OursList(items, rank, how, cost, latency, False, source, reproduced)
    return out


# --- scoring -----------------------------------------------------------------------------


def _scored(rank: int | None) -> dict[str, Any]:
    return {
        "target_rank": rank,
        **{f"hit_at_{k}": int(rank is not None and rank <= k) for k in HIT_KS},
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
    }


def match_item(item: Item, resolution: Resolution | None, original: Record) -> str | None:
    """How a listed paper is the original (id, alias, doi or title), as listed or resolved."""
    versions = [{"id": "", "aliases": [], "doi": item.doi, "title": item.title or ""}]
    if resolution is not None and resolution.status == VERIFIED:
        versions.append(
            {
                "id": resolution.openalex_id or "",
                "aliases": [],
                "doi": resolution.doi,
                "title": resolution.title or "",
            }
        )
    return match_original(versions, original)


def _base_row(system: str, query: Query, status: str) -> dict[str, Any]:
    return {
        "system": system,
        "qid": query.qid,
        "target_id": query.target_id,
        "bucket": query.bucket,
        "status": status,
        "finished_at": _now(),
    }


def ours_row(query: Query, ours: OursList) -> dict[str, Any]:
    items = [
        {"rank": i, **item.to_dict(), "resolution": None, "match": None}
        for i, item in enumerate(ours.items, start=1)
    ]
    if ours.target_rank is not None:
        items[ours.target_rank - 1]["match"] = ours.match
    return {
        **_base_row(OURS, query, "ok"),
        **_scored(ours.target_rank),
        "match": ours.match,
        "n_items": len(ours.items),
        "n_unresolved": 0,  # OpenAlex records, by construction
        "n_doi_mismatch": 0,
        "parse_error": None,
        "cost": ours.cost,
        "money_spent": 0.0,  # replayed from the cache
        "latency_s": ours.latency_s,
        "searches": None,
        "cached": True,
        "capped_check": ours.capped,
        "source": ours.source,
        "reproduced": ours.reproduced,
        "items": items,
    }


def answer_row(
    system: str,
    query: Query,
    answer: Answer,
    resolutions: Sequence[Resolution],
    original: Record,
) -> dict[str, Any]:
    reading = answer.reading
    rank, how = None, None
    items = []
    for i, (item, resolution) in enumerate(zip(reading.items, resolutions), start=1):
        matched = match_item(item, resolution, original)
        if matched and rank is None:
            rank, how = i, matched
        items.append(
            {"rank": i, **item.to_dict(), "resolution": resolution.to_dict(), "match": matched}
        )
    return {
        **_base_row(system, query, "ok"),
        **_scored(rank),
        "match": how,
        "n_items": len(items),
        "n_unresolved": sum(r.status == UNRESOLVED for r in resolutions),
        "n_doi_mismatch": sum(r.doi_mismatch for r in resolutions),
        "parse_error": reading.parse_error,
        "cost": answer.cost,
        "money_spent": answer.money_spent,
        "billed_cost": None if answer.cached else reading.billed_cost,
        "latency_s": answer.latency_s,
        "searches": reading.searches,
        "input_tokens": reading.input_tokens,
        "output_tokens": reading.output_tokens,
        "provider": reading.provider,
        "cached": answer.cached,
        "items": items,
        "answer_text": reading.text,
    }


def failed_row(system: str, query: Query, status: str, exc: BaseException) -> dict[str, Any]:
    return {**_base_row(system, query, status), "error": f"{type(exc).__name__}: {exc}"[:2000]}


# --- running -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CompeteRun:
    run_dir: Path
    summary: dict[str, Any]


def _new_run(cfg: Config, plan: Plan, clients: Mapping[str, System]) -> tuple[Path, dict[str, Any]]:
    assert cfg.compete is not None
    stamp = RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id, n = stamp, 1
    while (cfg.run.out_dir / run_id).exists():
        n += 1
        run_id = f"{stamp}-{n}"
    run_dir = cfg.run.out_dir / run_id
    run_dir.mkdir(parents=True)
    commit, dirty, untracked = git_state()
    prompt = load_prompt(PROMPT)
    meta: dict[str, Any] = {
        "run_id": run_id,
        "kind": KIND,
        "created_at": _now(),
        "git_commit": commit,
        "git_dirty": dirty,
        "git_untracked": untracked,
        "corpus_sha256": plan.corpus_sha256,
        "bench_sha256": plan.bench_sha256,
        "phase1_run": cfg.live.phase1_run if cfg.live else None,
        "l5w_fusion": live_evaluate.L5W_FUSION,
        "prompt": {"name": prompt.name, "sha256": prompt.sha256},
        "seed": cfg.run.seed,
        "frame": {b: sum(q.bucket == b for q in plan.frame) for b in BUCKETS},
        "quotas": plan.quotas,
        "sample": [{"qid": q.qid, "bucket": q.bucket, "run": str(q.run_dir)} for q in plan.sample],
        "systems": {
            name: "the live tool (L5w), replayed" if name == OURS else _jsonable(
                dataclasses.asdict(clients[name].config)
            )
            for name in plan.systems
        },
        "config": _jsonable(dataclasses.asdict(cfg.compete)),
    }
    (run_dir / RUN_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return run_dir, meta


def _append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _progress(row: Mapping[str, Any], i: int, n: int) -> str:
    head = f"{row['system']} {i}/{n} {row['qid']}"
    if row["status"] != "ok":
        return f"{head}: {row['status']}: {row['error'][:200]}"
    rank = row["target_rank"] or "-"
    spent = "cached" if row["cached"] else f"${row['money_spent']:.4f}"
    return (
        f"{head}: original at {rank}; {row['n_items']} listed, {row['n_unresolved']} "
        f"unresolved; {spent}; {row['latency_s']:.1f} s"
    )


def missing_keys(
    plan: Plan, clients: Mapping[str, System], cache: ResponseCache
) -> dict[str, str]:
    """System -> the unset env var it would need for calls that aren't cached."""
    missing = {}
    for name in plan.systems:
        if name == OURS:
            continue
        client = clients[name]
        env = client.missing_key()
        if env and any(request_key(client.request(q.proposal)) not in cache for q in plan.sample):
            missing[name] = env
    return missing


def run_compete(
    cfg: Config,
    plan: Plan,
    *,
    clients: Mapping[str, System],
    resolver: Resolver,
    conn: sqlite3.Connection,
    encoder: Encoder | None,
    progress: Callable[[str], None] = lambda line: None,
) -> CompeteRun:
    """Ask every selected system about every sampled query and write the run directory."""
    cache, ledger = ResponseCache(conn), SpendLedger(conn)
    missing = missing_keys(plan, clients, cache)
    if missing:
        needs = "; ".join(f"{s} needs {env}" for s, env in missing.items())
        raise EvalError(f"API keys missing for calls not in the cache: {needs}")
    if OURS in plan.systems and encoder is None:
        raise EvalError("ours is replayed with the encoder, but none was given")
    run_dir, meta = _new_run(cfg, plan, clients)
    path = run_dir / RESULTS_FILE
    rows: list[dict[str, Any]] = []
    n = len(plan.sample)
    try:
        if OURS in plan.systems and encoder is not None:
            lists = ours_lists(plan.sample, plan.originals, encoder)
            for i, q in enumerate(plan.sample, start=1):
                rows.append(ours_row(q, lists[q.qid]))
                _append(path, rows[-1])
                progress(_progress(rows[-1], i, n))
        for name in plan.systems:
            if name == OURS:
                continue
            capped: SystemCapReached | None = None
            for i, q in enumerate(plan.sample, start=1):
                if capped is not None:
                    row = failed_row(name, q, "capped", capped)
                else:
                    try:
                        answer = ask(
                            clients[name], q.proposal, qid=q.qid, cache=cache, ledger=ledger
                        )
                        resolutions = [resolver.resolve(item) for item in answer.reading.items]
                        row = answer_row(name, q, answer, resolutions, plan.originals[q.target_id])
                    except SystemCapReached as exc:
                        capped = exc
                        row = failed_row(name, q, "capped", exc)
                    except CompeteError as exc:
                        row = failed_row(name, q, "error", exc)
                rows.append(row)
                _append(path, row)
                progress(_progress(row, i, n))
    finally:
        spend = {
            name: {"total": ledger.recorded(name), "cap": clients[name].config.cost_cap}
            for name in plan.systems
            if name != OURS
        }
        summary = summarize(rows, meta=meta, systems=plan.systems, seed=cfg.run.seed, spend=spend)
        (run_dir / SUMMARY_JSON).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (run_dir / SUMMARY_MD).write_text(render_markdown(summary), encoding="utf-8")
    return CompeteRun(run_dir, summary)


# --- the summary -------------------------------------------------------------------------


def holm(ps: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the given order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted = [1.0] * len(ps)
    running = 0.0
    for position, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - position) * ps[i]))
        adjusted[i] = running
    return adjusted


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    meta: Mapping[str, Any],
    systems: Sequence[str],
    seed: int,
    spend: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    by_system = {s: [r for r in rows if r["system"] == s] for s in systems}
    ok = {s: {r["qid"]: r for r in by_system[s] if r["status"] == "ok"} for s in systems}

    def quality(system: str) -> dict[str, Any]:
        values = list(ok[system].values())
        n = len(values)
        items = sum(r["n_items"] for r in values)
        unresolved = sum(r["n_unresolved"] for r in values)
        mismatch = sum(r["n_doi_mismatch"] for r in values)

        def mean(key: str) -> float | None:
            return sum(r[key] for r in values) / n if n else None

        def count(status: str) -> int:
            return sum(r["status"] == status for r in by_system[system])

        return {
            "n_ok": n,
            "n_error": count("error"),
            "n_capped": count("capped"),
            "n_parse_failures": sum(r["parse_error"] is not None for r in values),
            "n_empty": sum(r["n_items"] == 0 for r in values),
            **{f"hit_at_{k}": mean(f"hit_at_{k}") for k in HIT_KS},
            "mrr": mean("reciprocal_rank"),
            "items": items,
            "mean_items": items / n if n else None,
            "unresolved": unresolved,
            "unresolved_share": unresolved / items if items else None,
            "doi_mismatch": mismatch,
            "doi_mismatch_share": mismatch / items if items else None,
            "money_spent": sum(r["money_spent"] for r in values),
        }

    def effort(system: str) -> dict[str, Any]:
        values = list(ok[system].values())
        return {m: _spread([r[m] for r in values if r.get(m) is not None]) for m in EFFORT}

    def by_bucket(system: str, b: str) -> dict[str, Any]:
        hits = [r["hit_at_5"] for r in ok[system].values() if r["bucket"] == b]
        return {"n": len(hits), "hit_at_5": sum(hits) / len(hits) if hits else None}

    difficulty = {s: {b: by_bucket(s, b) for b in BUCKETS} for s in systems}
    paired = []
    if OURS in ok:
        paired = [_paired(OURS, s, ok[OURS], ok[s], seed=seed) for s in systems if s != OURS]
        for pair, p in zip(paired, holm([pair["mcnemar"]["p"] for pair in paired])):
            pair["mcnemar"]["p_holm"] = p
    ours = list(ok.get(OURS, {}).values())
    return {
        "run_id": meta["run_id"],
        "kind": KIND,
        "created_at": meta["created_at"],
        "git_commit": meta["git_commit"],
        "git_dirty": meta["git_dirty"],
        "prompt": meta["prompt"],
        "seed": meta["seed"],
        "frame": meta["frame"],
        "quotas": meta["quotas"],
        "n_queries": len(meta["sample"]),
        "systems": list(systems),
        "system_labels": {s: _label(s, meta["systems"][s]) for s in systems},
        "quality": {s: quality(s) for s in systems},
        "effort": {s: effort(s) for s in systems},
        "spend": dict(spend),
        "difficulty": difficulty,
        "paired": paired,
        "ours_replay": {
            "recorded_l5w": sum(r["reproduced"] is not None for r in ours),
            "reproduced": sum(r["reproduced"] is True for r in ours),
            "capped_checks": sum(bool(r["capped_check"]) for r in ours),
        },
        "errors": {
            s: [f"{r['qid']}: {r['error']}" for r in by_system[s] if r["status"] == "error"][:3]
            for s in systems
        },
    }


def _label(name: str, system: Any) -> str:
    if name == OURS:
        return "the live tool (L5w), replayed from the live-eval caches"
    kind, model = system["kind"], system.get("model")
    if kind == "anthropic":
        return f"{model} with {system['web_search_tool']} (max {system['max_searches']} searches)"
    if kind == "openrouter" and system.get("web_search_tool"):
        return f"{model} on OpenRouter, native web search (max {system['max_searches']} searches)"
    if kind == "openrouter":
        context = system["search_context_size"] or "default"
        return f"{model} on OpenRouter (search context {context})"
    if kind == "s2-search":
        return "Semantic Scholar paper search, the proposal as the query"
    return "Semantic Scholar snippet search, the proposal as the query (each snippet's paper)"


def render_markdown(summary: Mapping[str, Any]) -> str:
    systems, q = summary["systems"], summary["quality"]
    dirty = " (uncommitted changes to tracked files)" if summary["git_dirty"] else ""
    frame, quotas = summary["frame"], summary["quotas"]
    strata = ", ".join(f"{b}: {quotas[b]} of {frame[b]}" for b in BUCKETS)
    lines = [
        f"# Competitive evaluation {summary['run_id']}",
        "",
        (
            f"Git {summary['git_commit'] or 'unknown'}{dirty}; prompt {summary['prompt']['name']} "
            f"sha256 {summary['prompt']['sha256'][:12]}; {summary['n_queries']} queries, "
            f"stratified by Phase 1 difficulty (bucket: sampled of checked, seed "
            f"{summary['seed']}): {strata}."
        ),
        "",
        *(f"- {s}: {summary['system_labels'][s]}" for s in systems),
        "",
        "## Finding the original",
        "",
        (
            "Errors and answers over a system's cost cap are left out; an answer without the "
            "JSON asked for lists nothing and counts as a miss (Unparsed)."
        ),
        "",
        (
            "| System | Answered | Errors | Over cost cap | Unparsed | Hit@1 | Hit@5 | Hit@10 "
            "| MRR@10 |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in systems:
        r = q[s]
        lines.append(
            f"| {s} | {r['n_ok']} | {r['n_error']} | {r['n_capped']} | {r['n_parse_failures']} "
            f"| {_f(r['hit_at_1'])} | {_f(r['hit_at_5'])} | {_f(r['hit_at_10'])} | {_f(r['mrr'])} |"
        )
    lines += [
        "",
        "## Listed papers that don't resolve (possibly fabricated)",
        "",
        (
            "Checked through free routes only (OpenAlex by id or DOI, Crossref by title; "
            "rk.compete.resolve). Ours lists OpenAlex records, so 0 by construction. Semantic "
            "Scholar lists papers from its own index, so its unresolved share estimates how "
            "often these routes miss a real paper."
        ),
        "",
        "| System | Papers listed | Per answer | Unresolved | Share | DOI mismatches | Share |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in systems:
        r = q[s]
        lines.append(
            f"| {s} | {r['items']} | {_f(r['mean_items'], 1)} | {r['unresolved']} "
            f"| {_f(r['unresolved_share'])} | {r['doi_mismatch']} | {_f(r['doi_mismatch_share'])} |"
        )
    lines += [
        "",
        "## Cost and latency per query (mean / p50 / p95)",
        "",
        (
            "Cost is the method cost at the config prices, cached answers included; ours is "
            "the live check's OpenAlex and LLM cost. Spent counts real calls in this run (as "
            "billed where the API reports it); the total is every compete run's, against the "
            "system's cap."
        ),
        "",
        (
            "| System | Cost ($) | Latency (s) | Web searches | Spent this run ($) "
            "| Spent in total ($) |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for s in systems:
        cells = []
        for m, d in (("cost", 4), ("latency_s", 1), ("searches", 1)):
            spread = summary["effort"][s][m]
            stats = (_f(spread[x], d) for x in ("mean", "p50", "p95")) if spread else ()
            cells.append(" / ".join(stats) or "–")
        spend = summary["spend"].get(s)
        cells.append(_f(q[s]["money_spent"], 4))
        cells.append(f"{spend['total']:.4f} of {spend['cap']:.2f}" if spend else "–")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Hit@5 by difficulty (the target's rank in Phase 1's B1)",
        "",
        "| System | " + " | ".join(f"{b} (n={quotas[b]})" for b in BUCKETS) + " |",
        "|---|" + "---|" * len(BUCKETS),
    ]
    for s in systems:
        cells = summary["difficulty"][s]
        lines.append(f"| {s} | " + " | ".join(_f(cells[b]["hit_at_5"]) for b in BUCKETS) + " |")
    lines += [
        "",
        "## Ours against each system (queries both answered)",
        "",
        (
            "| Comparison | n | ΔHit@5 [95% CI] | ΔMRR@10 [95% CI] | McNemar (only ours / only it) "
            "| p | p (Holm) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for p in summary["paired"]:
        m = p["mcnemar"]
        lines.append(
            f"| ours vs {p['b']} | {p['n']} | {_f(p['delta_hit_at_5'])} {_ci(p['ci_hit_at_5'])} "
            f"| {_f(p['delta_mrr'])} {_ci(p['ci_mrr'])} | {m['a_only']} / {m['b_only']} "
            f"| {m['p']:.3g} | {m['p_holm']:.3g} |"
        )
    if not summary["paired"]:
        lines.append("| (ours wasn't in this run) | | | | | | |")
    replay = summary["ours_replay"]
    lines += [
        "",
        "## Notes",
        "",
        (
            f"- Ours: the replayed L5w ranking equals the run's recorded L5w ranking in "
            f"{replay['reproduced']} of {replay['recorded_l5w']} checks that have one; "
            f"{replay['capped_checks']} checks were over their cost cap (an empty list)."
        ),
        *(
            f"- {s} errors: " + "; ".join(e.replace("|", "\\|") for e in summary["errors"][s])
            for s in systems
            if summary["errors"][s]
        ),
    ]
    return "\n".join(lines) + "\n"


def _ci(ci: Sequence[float] | None) -> str:
    return "–" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


# --- the plan, before anything is sent ---------------------------------------------------


def describe_plan(plan: Plan, clients: Mapping[str, System], conn: sqlite3.Connection) -> str:
    """The sample and, per system, what its uncached calls could cost at most (no calls)."""
    cache, ledger = ResponseCache(conn), SpendLedger(conn)
    sizes = {b: sum(q.bucket == b for q in plan.frame) for b in BUCKETS}
    drawn = {b: sum(q.bucket == b for q in plan.sample) for b in BUCKETS}
    lines = [
        f"rk compete: {len(plan.sample)} queries from the {len(plan.frame)} the runs checked; "
        "per Phase 1 bucket, sampled of checked: "
        + ", ".join(f"{b}: {drawn[b]} of {sizes[b]}" for b in BUCKETS)
    ]
    for name in plan.systems:
        if name == OURS:
            lines.append(f"  {OURS}: replayed from the live-eval caches; no calls")
            continue
        cfg = clients[name].config
        uncached = sum(
            request_key(clients[name].request(q.proposal)) not in cache for q in plan.sample
        )
        if cfg.cost_cap == 0:
            lines.append(f"  {name}: {uncached} of {len(plan.sample)} calls not cached; free")
            continue
        spent = ledger.recorded(name)
        lines.append(
            f"  {name}: {uncached} of {len(plan.sample)} calls not cached, at most "
            f"${uncached * cfg.reserve_per_query:.2f} (${cfg.reserve_per_query:.2f} reserved per "
            f"call); spent so far ${spent:.4f} of its ${cfg.cost_cap:.2f} cap"
        )
    return "\n".join(lines)
