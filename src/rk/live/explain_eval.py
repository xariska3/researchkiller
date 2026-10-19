"""The explainer's evaluation on dev queries (docs/phase2_spec.md, Step 5): report only.

`run_explain_eval` runs whole default checks (rk.live.check.run_check, strict) on the
first `limit` benchmark queries from `from_qid` (the dev set starts at A0045) and
measures, tuning nothing on them:
- **quote verification**: the share of explained papers whose evidence sentence was
  found in the abstract verbatim;
- **the original's overlap**: the level the original paper got when it is among the
  explained papers (it should be "high");
- **cost and time per check**, and how often a step fell back.

The searches read OpenAlex's cache as an evaluation does (they never expire), so on
queries live-eval has checked (with that run's live settings: a search whose filter
differs isn't cached), the searches and the agent and rerank calls come from the
caches; only the keyword lookups and the explainer calls are new. The check's own time
is then not a live time. With `timing_run` (Step 4's rerank run on the same checks),
each check's live time is estimated as its recorded check time (the live run's L5w
row, or L5's), plus its recorded rerank call, plus this run's snippet, keyword,
explainer and vocabulary stages.

With rerank.two_lists (issue #22), the checks have two lists; the recorded rerank call
is then R4's, every explained paper is counted once, and the summary adds the lists'
measures (papers explained, section B's papers and its "Also close in methods" line,
list B's fallbacks).

Writes runs/explain-<timestamp>/ (run.json, results.jsonl with one row per check,
reports/<qid>.json, summary.json, summary.md) and a CSV for hand review: the first
`review_n` checks, one row per top-REVIEW_TOP paper, never overwritten.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rk.bench import qid_position
from rk.config import Config
from rk.corpus import CORPUS_FILE, load_frozen_corpus
from rk.evaluate import (
    RUN_FILE,
    SUMMARY_JSON,
    SUMMARY_MD,
    EvalError,
    _f,
    _jsonable,
    _now,
    git_state,
    load_bench,
)
from rk.live.check import CheckRun, run_check
from rk.live.explain import EXPLAIN_PROMPT, EXPLAIN_SCHEMA_SHA256, EXPLAINED
from rk.live.openalex_search import LiveSearchClient
from rk.live.pipeline import AGENT_PROMPT, FALLBACK, RERANKED, SKIPPED, CheckCostCapExceeded
from rk.live.pool import match_original
from rk.live.report import (
    HIGH,
    NOT_EXPLAINED,
    OVERLAPS,
    UNVERIFIED,
    VERIFIED_EXACT,
    VERIFIED_MATCHED,
    CheckRequest,
    Paper,
)
from rk.live.rerank import R1, R4, _stats
from rk.live.reranker import LISTWISE, TWO_LISTS, TWO_LISTS_SCHEMA_SHA256
from rk.llm import ChatModel, load_prompt
from rk.retrieval import Encoder
from rk.targets import Record

KIND = "explain-eval"
RUN_PREFIX = "explain-"
RESULTS_FILE = "results.jsonl"
REPORTS_DIR = "reports"
DEV_FROM_QID = "A0045"  # the dev set: live-eval's run from A0045 (runs/live-20261014-122144)
REVIEW_FILE = Path("docs/evaluation/explainer_review.csv")
REVIEW_TOP = 5  # papers per check in the review CSV
REVIEW_COLUMNS = (
    "qid",
    "idea",
    "rank",
    "title",
    "overlap",
    "verified",
    "evidence",
    "differs",
    "correct?",
    "useful?",
    "false alarm?",
)
OVER_CAP = "over_cap"  # a check that stopped at its cost cap before the explainer


@dataclasses.dataclass(frozen=True)
class ExplainEvalRun:
    run_dir: Path
    summary: dict[str, Any]
    review_rows: int  # rows written to the review CSV


def recorded_times(
    timing_run: Path, results_db: Path, rerank_config: str = R1
) -> dict[str, dict[str, float | None]]:
    """Per qid: the live run's recorded check time and the rerank run's call time.

    `timing_run` is a rerank run (runs/rerank-<timestamp>); its run.json names the live
    run, whose rows are in `results_db`. The check time is its L5w row's, or its L5
    row's in a run from before L5w (the same check: L5w only ranks its pool another
    way). The call time is `rerank_config`'s (R1, or R4 for two lists); a cached call
    has no time (None).
    """
    meta = json.loads((timing_run / RUN_FILE).read_text(encoding="utf-8"))
    if meta.get("kind") != "rerank":
        raise EvalError(f"{timing_run} is not a rerank run")
    live_run = meta["live_run"]["run_id"]
    times: dict[str, dict[str, float | None]] = {}
    with closing(sqlite3.connect(results_db)) as conn:
        rows = conn.execute(
            "SELECT qid, config, latency_s FROM live_query_results "
            "WHERE run_id = ? AND config IN ('L5', 'L5w') AND status = 'ok' ORDER BY config",
            (live_run,),
        )
        for qid, _, latency in rows:  # L5w after L5, so it wins where both exist
            times[qid] = {"check_s": latency, "rerank_s": None}
    lines = (timing_run / "results.jsonl").read_text(encoding="utf-8").splitlines()
    for row in map(json.loads, lines):
        if row["config"] == rerank_config and row["qid"] in times:
            times[row["qid"]]["rerank_s"] = row.get("latency_s")
    return times


def _new_run(cfg: Config, meta: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    stamp = RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id, n = stamp, 1
    while (cfg.run.out_dir / run_id).exists():
        n += 1
        run_id = f"{stamp}-{n}"
    out_dir = cfg.run.out_dir / run_id
    (out_dir / REPORTS_DIR).mkdir(parents=True)
    run_meta = {"run_id": run_id, **meta}
    (out_dir / RUN_FILE).write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    return out_dir, run_meta


def _original_paper(papers: Sequence[Paper], original: Record) -> Paper | None:
    """The paper that is the original (any version's id or alias, DOI or title), if listed."""
    for p in papers:
        version = {"id": p.id, "aliases": p.aliases, "doi": p.doi, "title": p.title}
        if match_original([version], original) is not None:
            return p
    return None


def check_row(
    query: Mapping[str, Any],
    original: Record,
    run: CheckRun,
    times: Mapping[str, float | None] | None,
) -> dict[str, Any]:
    """One check's measures; a two-list check's explained papers count once each."""
    report = run.report
    explained_top = report.explained()
    explained = [p for p in explained_top if p.explanation is not None]
    found = _original_paper(report.papers, original)
    in_top = found is not None and found in report.shown()
    in_methods = _original_paper(report.methods or [], original)
    overlaps = {level: 0 for level in OVERLAPS}
    for p in explained:
        assert p.explanation is not None
        overlaps[p.explanation.overlap] += 1
    stages = report.stages
    tool_s = stages.snippets_s + stages.keywords_s + stages.explain_s + stages.vocabulary_s
    step = run.explain
    check_s = times.get("check_s") if times else None
    rerank_s = times.get("rerank_s") if times else None
    estimated = None if check_s is None or rerank_s is None else check_s + rerank_s + tool_s
    return {
        "qid": query["qid"],
        "target_id": query["target_id"],
        "status": "ok",
        "n_top": len(explained_top),
        "n_explained": len(explained),
        "n_exact": sum(p.status == VERIFIED_EXACT for p in explained),
        "n_matched": sum(p.status == VERIFIED_MATCHED for p in explained),
        "n_unverified": sum(p.status == UNVERIFIED for p in explained),
        "overlaps": overlaps,
        "high_not_original": sum(
            p.explanation is not None
            and p.explanation.overlap == HIGH
            and not (found is not None and p.same_work(found))
            for p in explained
        ),
        "original_rank": found.rank if found else None,
        "original_in_top": in_top,
        "original_overlap": (
            found.explanation.overlap if in_top and found and found.explanation else None
        ),
        "original_status": found.status if in_top and found else None,
        "lists": 1 if report.methods is None else 2,
        "n_methods": len(report.methods) if report.methods is not None else None,
        "n_also_close": len(report.also_in_methods or []),  # section B's line
        "original_methods_rank": in_methods.rank if in_methods else None,  # in section B
        "original_also_close": in_top and found is not None and report.in_methods(found),
        "methods_status": run.result.rerank.methods_status if run.result.rerank else None,
        "rerank": report.rerank,
        "explainer": report.explainer,
        "warnings": report.warnings,
        "method_cost": report.cost.method_cost,
        "billed_cost": report.cost.billed_cost,
        "explain_cost": step.usage.method_cost if step else 0.0,
        "explain_billed": step.usage.billed_cost if step else 0.0,
        "explain_prompt_tokens": step.usage.prompt_tokens if step else None,
        "explain_completion_tokens": step.usage.completion_tokens if step else None,
        "explain_cached": step.cached if step else None,
        "finish_reason": step.finish_reason if step else None,
        "keywords_cached": run.keywords_cached,
        "merged": [list(pair) for pair in run.merged],
        "n_papers": len(report.papers),
        "vocabulary_terms": sum(len(g.terms) for g in report.vocabulary),
        "stages": report.stages.model_dump(),
        "explainer_stage_s": tool_s,
        "recorded_check_s": check_s,
        "recorded_rerank_s": rerank_s,
        "estimated_live_s": estimated,
    }


def review_rows(
    query: Mapping[str, Any], report_papers: Sequence[Paper]
) -> list[dict[str, Any]]:
    """The review CSV's rows for one check: its top REVIEW_TOP papers, review columns empty."""
    rows = []
    for p in report_papers[:REVIEW_TOP]:
        e = p.explanation
        if e is not None and e.evidence is not None:
            evidence = e.evidence
        elif p.snippet is not None:
            evidence = f"[snippet] {p.snippet.sentence}"
        else:
            evidence = ""
        rows.append(
            {
                "qid": query["qid"],
                "idea": query["query"],
                "rank": p.rank,
                "title": p.title,
                "overlap": e.overlap if e else NOT_EXPLAINED,
                "verified": "" if e is None else (e.match if e.verified else "no"),
                "evidence": evidence,
                "differs": e.differs if e else "",
                "correct?": "",
                "useful?": "",
                "false alarm?": "",
            }
        )
    return rows


def run_explain_eval(
    cfg: Config,
    *,
    search: LiveSearchClient,
    encoder: Encoder,
    agent: ChatModel,
    reranker: ChatModel,
    explainer: ChatModel,
    from_qid: str = DEV_FROM_QID,
    limit: int = 20,
    review_n: int = 10,
    review_out: Path = REVIEW_FILE,
    timing_run: Path | None = None,
    progress: Callable[[str], None] = lambda line: None,
) -> ExplainEvalRun:
    """Whole default checks on `limit` queries from `from_qid`; see the module docstring.

    `search` should read its cache as an evaluation does. A failed explainer call
    (LLMError) stops the run; the rows written so far stay in results.jsonl, and a
    rerun replays every finished call from the cache.
    """
    if cfg.live is None or cfg.explain is None or cfg.rerank is None:
        raise EvalError("explain-eval needs the config's live:, rerank: and explain: sections")
    if review_out.exists():
        raise EvalError(f"{review_out} exists; it is never overwritten")
    corpus, _, manifest = load_frozen_corpus(cfg.corpus.out_dir)
    corpus_sha = manifest["files"][CORPUS_FILE]["sha256"]
    by_id = {r["id"]: r for r in corpus}
    queries, _, bench_sha = load_bench(cfg.bench.out_dir, corpus_sha, set(by_id))
    first = qid_position(from_qid)
    queries = [q for q in queries if qid_position(q["qid"]) >= first][:limit]
    if not queries:
        raise EvalError(f"no benchmark queries from {from_qid}")
    originals = {q["qid"]: by_id[q["target_id"]] for q in queries}
    del corpus, by_id
    two_lists = cfg.rerank.two_lists
    rerank_config = R4 if two_lists else R1
    times = (
        recorded_times(timing_run, cfg.run.results_db, rerank_config) if timing_run else None
    )
    rerank_prompt = TWO_LISTS if two_lists else LISTWISE
    prompts = {name: load_prompt(name).sha256 for name in (AGENT_PROMPT, rerank_prompt)}
    prompts[EXPLAIN_PROMPT] = load_prompt(EXPLAIN_PROMPT).sha256
    prompts["explain_schema"] = EXPLAIN_SCHEMA_SHA256
    if two_lists:
        prompts["rerank_schema"] = TWO_LISTS_SCHEMA_SHA256

    commit, dirty, untracked = git_state()
    explain_cfg = _jsonable(dataclasses.asdict(cfg.explain))
    out_dir, meta = _new_run(
        cfg,
        {
            "kind": KIND,
            "created_at": _now(),
            "git_commit": commit,
            "git_dirty": dirty,
            "git_untracked": untracked,
            "corpus_sha256": corpus_sha,
            "bench_sha256": bench_sha,
            "from_qid": from_qid,
            "limit": limit,
            "qids": [q["qid"] for q in queries],
            "review": {"file": str(review_out), "checks": review_n, "papers": REVIEW_TOP},
            "timing_run": str(timing_run) if timing_run else None,
            "timing_config": rerank_config if timing_run else None,
            "two_lists": two_lists,
            "models": {
                "agent": cfg.agent.name,
                "rerank": cfg.rerank.model.name,
                "explain": cfg.explain.model.name,
                "encoder": encoder.name,
            },
            "encoder_device": encoder.device,
            "prompts": prompts,
            "explain_top_k": cfg.live.explain_top_k,
            "config": {"explain": explain_cfg, "live": _jsonable(dataclasses.asdict(cfg.live))},
            "seed": cfg.run.seed,
        },
    )
    rows: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    reviewed = 0  # checks in the review CSV: the first review_n that finished
    for i, query in enumerate(queries, start=1):
        row: dict[str, Any]
        try:
            run = run_check(
                CheckRequest(text=query["query"]),
                cfg,
                search=search,
                encoder=encoder,
                agent=agent,
                reranker=reranker,
                explainer=explainer,
                strict=True,
            )
        except CheckCostCapExceeded as exc:
            row = {"qid": query["qid"], "target_id": query["target_id"], "status": OVER_CAP}
            row["error"] = str(exc)
        else:
            report_path = out_dir / REPORTS_DIR / f"{query['qid']}.json"
            report_path.write_text(run.report.model_dump_json(indent=2) + "\n", encoding="utf-8")
            row = check_row(query, originals[query["qid"]], run, (times or {}).get(query["qid"]))
            if reviewed < review_n:
                review.extend(review_rows(query, run.report.papers))
                reviewed += 1
        rows.append(row)
        with (out_dir / RESULTS_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if row["status"] == "ok":
            level = row["original_overlap"] or ("not in top" if not row["original_in_top"] else "-")
            progress(
                f"{i}/{len(queries)} {query['qid']}: original {level}; quotes "
                f"{row['n_exact']} exact, {row['n_matched']} matched, "
                f"{row['n_unverified']} unverified; ${row['method_cost']:.4f}"
            )
        else:
            progress(f"{i}/{len(queries)} {query['qid']}: {row['status']}")
    summary = summarize(rows, meta, cap=cfg.explain.cost_cap_per_check)
    (out_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / SUMMARY_MD).write_text(render_markdown(summary), encoding="utf-8")
    review_out.parent.mkdir(parents=True, exist_ok=True)
    with review_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(review)
    return ExplainEvalRun(out_dir, summary, len(review))


# --- the summary -----------------------------------------------------------------------------


def summarize(
    rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], *, cap: float
) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    explained = sum(r["n_explained"] for r in ok)
    kinds = {k: sum(r[f"n_{k}"] for r in ok) for k in ("exact", "matched", "unverified")}
    top = sum(r["n_top"] for r in ok)
    with_original = [r for r in ok if r["original_in_top"]]
    original_levels = {level: 0 for level in (*OVERLAPS, NOT_EXPLAINED)}
    for r in with_original:
        original_levels[r["original_overlap"] or NOT_EXPLAINED] += 1
    overlaps = {level: sum(r["overlaps"][level] for r in ok) for level in OVERLAPS}
    real = [r for r in ok if r["explain_cached"] is False]
    two = [r for r in ok if r.get("lists") == 2]

    def count(key: str, value: str) -> int:
        return sum(r.get(key) == value for r in ok)

    return {
        "run_id": meta["run_id"],
        "git_commit": meta["git_commit"],
        "qids": [meta["qids"][0], meta["qids"][-1]] if meta["qids"] else [],
        "n_checks": len(rows),
        "n_ok": len(ok),
        "over_cap": sum(r["status"] == OVER_CAP for r in rows),
        "quotes": {
            "papers": top,
            "explained": explained,
            **kinds,
            "rates": {k: n / explained if explained else None for k, n in kinds.items()},
        },
        "original": {
            "in_top": len(with_original),
            "not_in_top": len(ok) - len(with_original),
            "levels": original_levels,
            "high_rate": original_levels[HIGH] / len(with_original) if with_original else None,
            "quotes": {
                status: sum(r["original_status"] == status for r in with_original)
                for status in (VERIFIED_EXACT, VERIFIED_MATCHED, UNVERIFIED)
            },
        },
        "overlaps": overlaps,
        "high_not_original": sum(r["high_not_original"] for r in ok),
        "merged": [f"{r['qid']}: {a} + {b}" for r in ok for a, b in r["merged"]],
        "two_lists": (
            {
                "checks": len(two),
                "explained": _stats([r["n_top"] for r in two]),
                "section_b": _stats([r["n_methods"] for r in two]),
                "also_close": _stats([r["n_also_close"] for r in two]),
                "methods_list": {
                    s: count("methods_status", s) for s in (RERANKED, FALLBACK, SKIPPED)
                },
                "original_in_methods": sum(r["original_methods_rank"] is not None for r in two),
                "original_also_close": sum(bool(r["original_also_close"]) for r in two),
            }
            if two
            else None
        ),
        "steps": {
            "rerank": {s: count("rerank", s) for s in (RERANKED, FALLBACK, SKIPPED)},
            "explainer": {s: count("explainer", s) for s in (EXPLAINED, FALLBACK, SKIPPED)},
            "partial": sum(
                r["explainer"] == EXPLAINED and r["n_explained"] < r["n_top"] for r in ok
            ),
            "keywords_failed": sum(
                any(w.startswith("No OpenAlex keywords") for w in r["warnings"]) for r in ok
            ),
            "cut_off": sum(r["finish_reason"] == "length" for r in ok),
            "warnings": [f"{r['qid']}: {w}" for r in ok for w in r["warnings"]][:5],
        },
        "cost": {
            "cap": cap,
            "check": _stats([r["method_cost"] for r in ok]),
            "explain_call": _stats([r["explain_cost"] for r in ok]),
            "spent": sum(r["billed_cost"] for r in ok),
            "explain_spent": sum(r["explain_billed"] for r in ok),
        },
        "time": {
            "explain_call": _stats([r["stages"]["explain_s"] for r in real]),
            "explainer_stage": _stats([r["explainer_stage_s"] for r in real]),
            "snippets": _stats([r["stages"]["snippets_s"] for r in ok]),
            "estimated_live": _stats(
                [r["estimated_live_s"] for r in real if r["estimated_live_s"] is not None]
            ),
            "this_run": _stats([r["stages"]["total_s"] for r in ok]),
            "real_explain_calls": len(real),
        },
        "vocabulary_terms": _stats([r["vocabulary_terms"] for r in ok]),
    }


def _spread(s: Mapping[str, float] | None, digits: int) -> str:
    if not s:
        return "–"
    return " / ".join(_f(s[k], digits) for k in ("p50", "p95", "max"))


def render_markdown(summary: Mapping[str, Any]) -> str:
    q, o, steps = summary["quotes"], summary["original"], summary["steps"]
    cost, t = summary["cost"], summary["time"]
    first, last = summary["qids"] or ["–", "–"]
    quotes = ", ".join(
        f"{kind} {q[kind]} ({'–' if rate is None else f'{rate:.1%}'})"
        for kind, rate in q["rates"].items()
    )
    original_quotes = ", ".join(f"{status} {n}" for status, n in o["quotes"].items())
    high = "–" if o["high_rate"] is None else f"{o['high_rate']:.0%}"
    levels = ", ".join(f"{level} {n}" for level, n in o["levels"].items())
    overlaps = ", ".join(f"{level} {n}" for level, n in summary["overlaps"].items())
    rerank = ", ".join(f"{k} {v}" for k, v in steps["rerank"].items())
    explainer = ", ".join(f"{k} {v}" for k, v in steps["explainer"].items())
    lines = [
        f"# Explainer evaluation {summary['run_id']}",
        "",
        (
            f"Git {summary['git_commit']}. {summary['n_checks']} default checks with the "
            f"explainer ({first} to {last} in seeded order); {summary['n_ok']} finished, "
            f"{summary['over_cap']} stopped at the cost cap. Report only: nothing was tuned."
        ),
        "",
        "## Measures",
        "",
        (
            f"- **Quotes** of the {q['explained']} explained papers ({q['papers']} top "
            f"papers): {quotes}. Exact: verbatim in the abstract; matched: a sentence has "
            f"90% of its words; either way the abstract's sentence is shown."
        ),
        (
            f"- **The original's overlap** (when among the explained papers: {o['in_top']} "
            f"checks; {o['not_in_top']} without it): {levels}. High: {high}. Its quote: "
            f"{original_quotes}."
        ),
        (
            f"- **Overlap of every explained paper**: {overlaps}; "
            f"{summary['high_not_original']} papers other than the original were rated high."
        ),
        "",
        "## Steps",
        "",
        f"- Rerank: {rerank}.",
        (
            f"- Explainer: {explainer}; {steps['partial']} explained only some papers; "
            f"{steps['cut_off']} replies cut off."
        ),
        f"- Keyword lookups failed: {steps['keywords_failed']}.",
        (
            f"- Versions merged in the final lists (same text): {len(summary['merged'])}"
            + "".join(f"; {m}" for m in summary["merged"])
            + "."
        ),
    ]
    lines += [f"  - {w}" for w in steps["warnings"]]
    two = summary.get("two_lists")
    if two is not None:
        methods = ", ".join(f"{k} {v}" for k, v in two["methods_list"].items())
        lines += [
            "",
            "## Two lists (issue #22)",
            "",
            f"- Two-list checks: {two['checks']}.",
            (
                f"- Papers explained per check, each once (median / p95 / max): "
                f"{_spread(two['explained'], 0)}; in section B: "
                f"{_spread(two['section_b'], 0)}; section A papers on section B's \"Also "
                f"close in methods\" line: {_spread(two['also_close'], 0)}."
            ),
            f"- List B: {methods} (fallback: empty).",
            (
                f"- The original in section B: {two['original_in_methods']} checks; on its "
                f"\"Also close in methods\" line: {two['original_also_close']} checks."
            ),
        ]
    lines += [
        "",
        "## Cost and time per check (median / p95 / max)",
        "",
        f"- **Whole check** (method cost, cap ${cost['cap']:.2f}): ${_spread(cost['check'], 4)}.",
        f"- **Explainer call**: ${_spread(cost['explain_call'], 5)}.",
        (
            f"- **Spent in this run**: ${cost['spent']:.4f} "
            f"(explainer calls ${cost['explain_spent']:.4f})."
        ),
        (
            f"- **Explainer call time** (real calls, n = {t['real_explain_calls']}): "
            f"{_spread(t['explain_call'], 1)} s."
        ),
        (
            f"- **Explainer stage** (snippets, keywords, explainer, vocabulary; real calls): "
            f"{_spread(t['explainer_stage'], 1)} s."
        ),
        (
            f"- **Estimated live check** (recorded check + recorded rerank call + explainer "
            f"stage): {_spread(t['estimated_live'], 1)} s."
        ),
        (
            f"- This run's wall time, searches and rerank from the caches: "
            f"{_spread(t['this_run'], 1)} s."
        ),
        f"- Vocabulary terms per check: {_spread(summary['vocabulary_terms'], 0)}.",
    ]
    return "\n".join(lines) + "\n"
