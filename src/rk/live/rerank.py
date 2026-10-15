"""python -m rk rerank: LLM rerank variants of the live check, evaluated offline.

Pre-declared in docs/phase2_spec.md (Evaluation, rerank variants). Every L5 check
of a live-eval run is replayed from the cache (rk.live.replay): the same final
pool, ranked by L5's fusion and by L5w's. The candidates are the union of L5w's
and L5's top `rerank.depth` (30), taken in turn (L5w's first, then L5's first, and
so on, skipping works already taken). That is the *presented order*, and the
candidates are numbered in it. A reranker sees the proposal and each candidate's
number, title and abstract (cut at `rerank.abstract_chars`), nothing else.

Variants:
- R1, listwise: `rerank.model` returns every candidate number, closest first.
  Unknown or repeated numbers are dropped; missing ones follow in presented order.
- R2, listwise + first stage: R1's order and the presented order fused by RRF
  (k = 60). It reads R1's reply and makes no call of its own.
- R3, graded: `rerank.model` scores every candidate from 0 (unrelated) to 3 (the
  same research idea). Candidates are sorted by score, ties in presented order,
  and a missing score counts as 0.
- RS, the upper bound: R1 with `rerank.upper_model` (Claude Sonnet 5.5). It is not a
  product candidate.

R1's candidates, prompt and reply parsing are in rk.live.reranker, which the live
pipeline's L5r (rk.live.pipeline.with_rerank) calls too.

A reply with no readable JSON leaves the presented order and is counted. L5w's
works outside the candidates follow the reranked ones, and ranks count within
the run's live.top_k (20), so MRR is MRR@20. L5 and L5w are scored as replayed;
the run's recorded rankings are compared with them (the "reproduced" counts).

Selection (select_rerank): on the dev set, the product variant with the best
Hit@5 becomes L5r if it beats L5w (ties: MRR, then R1, R2, R3). RERANK_CHOICE
holds that choice. On the test set, L5r vs L5w decides: Hit@5 with the exact
McNemar test, MRR supporting.

A run writes runs/rerank-<UTC time>/: run.json, results.jsonl (one row per config
and check), summary.json and summary.md. Replies are cached by rk.llm, so running
again costs nothing for what was already asked.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rk.config import Config
from rk.evaluate import (
    HIT_KS,
    RUN_FILE,
    SUMMARY_JSON,
    SUMMARY_MD,
    EvalError,
    _ci,
    _f,
    _jsonable,
    _now,
    _paired,
    git_state,
)
from rk.live import evaluate as live_evaluate
from rk.live.fusion import B5, final_ranking
from rk.live.replay import Replay, _originals, _rows, _scored, load_run, replay_run
from rk.live.reranker import (
    LISTWISE,
    as_number,
    candidates,
    listwise_ranking,
    parse_order,
    render_prompt,
)
from rk.llm import BudgetExceeded, ChatModel, Completion, LLMError, Prompt, load_prompt, parse_json
from rk.retrieval import Encoder, rrf
from rk.targets import Record

KIND = "rerank"
RUN_PREFIX = "rerank-"
RESULTS_FILE = "results.jsonl"
L5, L5W = "L5", "L5w"
R1, R2, R3, RS = "R1", "R2", "R3", "RS"
PRODUCT = (R1, R2, R3)  # the candidates for L5r, in declared order
VARIANTS = (*PRODUCT, RS)
GRADED = "rerank_graded"  # R3's prompt; LISTWISE (R1, R2, RS) is rk.live.reranker's
PRODUCT_MODEL, UPPER_MODEL = "model", "upper_model"  # rerank: config keys
# Which call each variant reads: (prompt, model). R1 and R2 read the same reply.
CALLS = {
    R1: (LISTWISE, PRODUCT_MODEL),
    R2: (LISTWISE, PRODUCT_MODEL),
    R3: (GRADED, PRODUCT_MODEL),
    RS: (LISTWISE, UPPER_MODEL),
}
LABELS = {
    L5: "L5 (B5's fusion), replayed",
    L5W: "L5w (the default check), replayed",
    R1: "R1 listwise (product model)",
    R2: "R2 listwise + first stage (RRF)",
    R3: "R3 graded 0-3 (product model)",
    RS: "RS listwise, upper bound (upper model)",
}
MAX_SCORE = 3
UPPER_TIMEOUT_S = 600.0  # adaptive thinking can make an upper-model reply slow
REPLY_CHARS = 4000  # of each reply kept in results.jsonl
# Sonnet 5.5's training data runs to June 2026: originals published from this date
# on are ones it can't have read.
UNSEEN_FROM = date(2026, 7, 1)
# L5r: the rerank variant the dev set chose by select_rerank (docs/phase2_spec.md). On
# runs/live-20261014-122144 (runs/rerank-20261015-132644) R1 and R3 tied on Hit@5 (0.850,
# against L5w's 0.750), and R1 won the MRR@20 tie-break (0.842 against 0.758).
RERANK_CHOICE: str | None = R1


# --- replies (R1's candidates, prompt and parsing are in rk.live.reranker) ------------------


def parse_scores(text: str, n: int) -> dict[int, int] | None:
    """A graded reply's scores (0..3) by candidate number (1..n); None if nothing usable."""
    try:
        data = parse_json(text)
    except ValueError:
        return None
    if isinstance(data, dict) and isinstance(data.get("scores"), dict):
        data = data["scores"]
    if not isinstance(data, dict):
        return None
    scores: dict[int, int] = {}
    for key, value in data.items():
        number, score = as_number(key), as_number(value)
        if number is None or score is None:
            continue
        if 1 <= number <= n and 0 <= score <= MAX_SCORE:
            scores.setdefault(number, score)
    return scores or None


def fused_ranking(cands: Sequence[str], order: Sequence[int] | None) -> list[str]:
    """R2: RRF of the listwise order and the presented order."""
    return rrf([listwise_ranking(cands, order), list(cands)])


def graded_ranking(cands: Sequence[str], scores: Mapping[int, int] | None) -> list[str]:
    """R3: by score, highest first; ties (and missing scores, as 0) in presented order."""
    got = scores or {}
    numbers = sorted(range(1, len(cands) + 1), key=lambda n: (-got.get(n, 0), n))
    return [cands[n - 1] for n in numbers]


def rerank(variant: str, cands: Sequence[str], text: str) -> tuple[list[str], bool]:
    """The variant's ranking of the candidates from its reply, and whether the reply was read."""
    if variant == R3:
        scores = parse_scores(text, len(cands))
        return graded_ranking(cands, scores), scores is not None
    order = parse_order(text, len(cands))
    if variant == R2:
        return fused_ranking(cands, order), order is not None
    return listwise_ranking(cands, order), order is not None


def select_rerank(quality: Mapping[str, Mapping[str, Any]]) -> str | None:
    """L5r: the product variant with the best Hit@5 (ties: MRR, then declared order).

    None unless its Hit@5 is above L5w's.
    """
    best = max(
        PRODUCT,
        key=lambda v: (quality[v]["hit_at_5"], quality[v]["mrr"], -PRODUCT.index(v)),
    )
    return best if quality[best]["hit_at_5"] > quality[L5W]["hit_at_5"] else None


@dataclass(frozen=True)
class Reply:
    """One rerank call: the completion, or why there is none."""

    status: str  # "ok", "error" (the call failed) or "capped" (its ledger refused it)
    completion: Completion | None = None
    latency_s: float | None = None  # of a real call; None when cached
    error: str | None = None


def ask(client: ChatModel, text: str) -> Reply:
    start = time.perf_counter()
    try:
        completion = client.complete([{"role": "user", "content": text}])
    except BudgetExceeded as exc:
        return Reply("capped", error=str(exc))
    except LLMError as exc:
        return Reply("error", error=str(exc))
    latency = None if completion.cached else time.perf_counter() - start
    return Reply("ok", completion, latency)


# --- the run ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankRun:
    run_dir: Path
    summary: dict[str, Any]


def _new_run(
    cfg: Config, run_dir: Path, meta: Mapping[str, Any], variants: Sequence[str], qids: list[str]
) -> tuple[Path, dict[str, Any]]:
    assert cfg.rerank is not None
    stamp = RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id, n = stamp, 1
    while (cfg.run.out_dir / run_id).exists():
        n += 1
        run_id = f"{stamp}-{n}"
    out_dir = cfg.run.out_dir / run_id
    out_dir.mkdir(parents=True)
    commit, dirty, untracked = git_state()
    prompts = {name: load_prompt(name).sha256 for name in (LISTWISE, GRADED)}
    run_meta: dict[str, Any] = {
        "run_id": run_id,
        "kind": KIND,
        "created_at": _now(),
        "git_commit": commit,
        "git_dirty": dirty,
        "git_untracked": untracked,
        "live_run": {"dir": str(run_dir), "run_id": meta["run_id"]},
        "bench_sha256": meta["bench_sha256"],
        "l5w_fusion": live_evaluate.L5W_FUSION,
        "rerank_choice": RERANK_CHOICE,
        "variants": list(variants),
        "qids": qids,
        "prompts": prompts,
        "seed": cfg.run.seed,
        "config": _jsonable(dataclasses.asdict(cfg.rerank)),
    }
    (out_dir / RUN_FILE).write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    return out_dir, run_meta


def _append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass
class Calls:
    """The rerank calls of one run: the clients, and what their ledgers refused or spent."""

    clients: Mapping[str, ChatModel]
    prompts: Mapping[str, Prompt]
    abstract_chars: int
    capped: dict[str, str] = dataclasses.field(default_factory=dict)  # model key -> why
    spent: dict[str, float] = dataclasses.field(default_factory=dict)  # model key -> USD

    def ask(self, prompt_name: str, model_key: str, proposal: str, records: list[Record]) -> Reply:
        if model_key in self.capped:  # one refusal stops the model for the rest of the run
            return Reply("capped", error=self.capped[model_key])
        text = render_prompt(self.prompts[prompt_name], proposal, records, self.abstract_chars)
        reply = ask(self.clients[model_key], text)
        if reply.status == "capped":
            self.capped[model_key] = reply.error or "capped"
        if reply.completion is not None:
            spend = reply.completion.usage.billed_cost  # 0 for a cached reply
            self.spent[model_key] = self.spent.get(model_key, 0.0) + spend
        return reply


def rerank_check(
    rp: Replay,
    original: Record,
    *,
    recorded_l5w: list[str] | None,
    fusion: str,
    depth: int,
    top_k: int,
    check_cost: float,
    variants: Sequence[str],
    calls: Calls,
) -> tuple[list[dict[str, Any]], bool, bool]:
    """One replayed check's rows (L5, L5w, then each variant).

    Also returns whether the replayed L5 ranking equals the recorded one, and
    whether L5w's does (False when the run recorded no L5w ranking).
    """
    l5 = final_ranking(rp.pool, rp.queries, B5)
    l5w = final_ranking(rp.pool, rp.queries, fusion)
    cands = candidates(l5w, l5, depth)
    by_id = {r["id"]: r for r in rp.pool.records}

    def place(ranking: Sequence[str], k: int) -> tuple[int | None, str | None]:
        versions = [rp.pool.versions(by_id[d]) for d in ranking]
        return live_evaluate.original_rank(versions, original, k)

    base = {
        "qid": rp.qid,
        "target_id": rp.target_id,
        "published": original["publication_date"],
        "n_candidates": len(cands),
        "original_in_candidates": place(cands, len(cands))[0] is not None,
        "check_cost": check_cost,
    }

    def scored(config: str, ranking: Sequence[str], **extra: Any) -> dict[str, Any]:
        shown = list(ranking[:top_k])
        rank, how = place(shown, top_k)
        return {
            "config": config,
            **base,
            "status": "ok",
            **_scored(rank),
            "match": how,
            "doc_ids": shown,
            **extra,
            "finished_at": _now(),
        }

    rows = [
        scored(L5, l5, cost=0.0, total_cost=check_cost),
        scored(L5W, l5w, cost=0.0, total_cost=check_cost),
    ]
    records = [by_id[d] for d in cands]
    replies: dict[tuple[str, str], Reply] = {}
    for variant in variants:
        key = CALLS[variant]
        if key not in replies:
            replies[key] = calls.ask(*key, rp.queries[0], records)
        reply = replies[key]
        if reply.completion is None:
            failed = {"config": variant, **base, "status": reply.status, "error": reply.error}
            rows.append({**failed, "finished_at": _now()})
            continue
        completion = reply.completion
        reranked, readable = rerank(variant, cands, completion.text)
        cost = completion.usage.method_cost
        rows.append(
            scored(
                variant,
                list(dict.fromkeys([*reranked, *l5w])),  # L5w's other works follow
                readable=readable,
                cost=cost,
                total_cost=check_cost + cost,
                cached=completion.cached,
                latency_s=reply.latency_s,
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                finish_reason=completion.finish_reason,
                reply=completion.text[:REPLY_CHARS],
            )
        )
    return (
        rows,
        l5[:top_k] == rp.recorded_l5,
        recorded_l5w is not None and l5w[:top_k] == recorded_l5w,
    )


def run_rerank(
    cfg: Config,
    run_dir: Path,
    *,
    encoder: Encoder,
    clients: Mapping[str, ChatModel],
    variants: Sequence[str] = VARIANTS,
    limit: int | None = None,
    progress: Callable[[str], None] = lambda line: None,
) -> RerankRun:
    """Every variant on the run's replayed L5 checks (the first `limit` in qid order)."""
    if cfg.rerank is None:
        raise EvalError("the config has no rerank: section")
    fusion = live_evaluate.L5W_FUSION
    if fusion is None:
        raise EvalError("L5w has no final fusion (rk.live.evaluate.L5W_FUSION)")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown or not variants:
        raise EvalError(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    variants = [v for v in VARIANTS if v in variants]
    missing = sorted({CALLS[v][1] for v in variants} - set(clients))
    if missing:
        raise EvalError(f"no client for {missing}")
    meta, _, _ = load_run(run_dir)
    l5_rows, l5w_rows = _rows(meta, L5), _rows(meta, L5W)
    qids = sorted(l5_rows)[:limit] if limit else sorted(l5_rows)
    if not qids:
        raise EvalError(f"{run_dir} has no finished L5 checks to rerank")
    _, run_cfg, replays = replay_run(run_dir, encoder, qids=set(qids))
    top_k = meta["config"]["live"]["top_k"]
    cost_cap = meta["config"]["live"]["cost_cap_per_check"]
    originals = _originals(run_cfg, {rp.target_id for rp in replays})
    prompts = {name: load_prompt(name) for name in (LISTWISE, GRADED)}
    calls = Calls(clients, prompts, cfg.rerank.abstract_chars)
    out_dir, run_meta = _new_run(cfg, run_dir, meta, variants, qids)
    rows: list[dict[str, Any]] = []
    reproduced = {L5: 0, L5W: 0, "L5w_rows": 0}
    for i, rp in enumerate(replays, start=1):
        l5w_row = l5w_rows.get(rp.qid)
        done, same_l5, same_l5w = rerank_check(
            rp,
            originals[rp.target_id],
            recorded_l5w=json.loads(l5w_row["doc_ids"]) if l5w_row else None,
            fusion=fusion,
            depth=cfg.rerank.depth,
            top_k=top_k,
            check_cost=l5_rows[rp.qid]["method_cost"],
            variants=variants,
            calls=calls,
        )
        reproduced[L5] += same_l5
        reproduced[L5W] += same_l5w
        reproduced["L5w_rows"] += l5w_row is not None
        for row in done:
            _append(out_dir / RESULTS_FILE, row)
        rows.extend(done)
        ranks = ", ".join(
            f"{r['config']} {r.get('target_rank') or '-'}"
            if r["status"] == "ok"
            else f"{r['config']} {r['status']}"
            for r in done
        )
        inside = "in" if done[0]["original_in_candidates"] else "not in"
        progress(
            f"{i}/{len(replays)} {rp.qid}: {done[0]['n_candidates']} candidates "
            f"(original {inside}); rank {ranks}"
        )
    summary = summarize(
        rows,
        variants=variants,
        seed=cfg.run.seed,
        cost_cap=cost_cap,
        choice=RERANK_CHOICE,
        meta=run_meta,
        reproduced=reproduced,
        spent=calls.spent,
    )
    (out_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / SUMMARY_MD).write_text(render_markdown(summary), encoding="utf-8")
    return RerankRun(out_dir, summary)


# --- the summary -----------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    p50, p95 = np.percentile(v, [50, 95])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p50": float(p50),
        "p95": float(p95),
        "max": float(v.max()),
        "min": float(v.min()),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[str],
    seed: int,
    cost_cap: float,
    choice: str | None,
    meta: Mapping[str, Any],
    reproduced: Mapping[str, int],
    spent: Mapping[str, float],
) -> dict[str, Any]:
    configs = [L5, L5W, *variants]
    by = {c: [r for r in rows if r["config"] == c] for c in configs}
    ok = {c: {r["qid"]: r for r in by[c] if r["status"] == "ok"} for c in configs}

    def quality(c: str) -> dict[str, Any]:
        done = list(ok[c].values())
        n = len(done)

        def mean(key: str) -> float | None:
            return sum(r[key] for r in done) / n if n else None

        return {
            "n": n,
            "n_error": sum(r["status"] == "error" for r in by[c]),
            "n_capped": sum(r["status"] == "capped" for r in by[c]),
            "unreadable": sum(not r.get("readable", True) for r in done),
            **{f"hit_at_{k}": mean(f"hit_at_{k}") for k in HIT_KS},
            "mrr": mean("reciprocal_rank"),
        }

    checks = list(ok[L5W].values())
    qualities = {c: quality(c) for c in configs}
    summary: dict[str, Any] = {
        "run_id": meta["run_id"],
        "live_run": meta["live_run"],
        "git_commit": meta["git_commit"],
        "l5w_fusion": meta["l5w_fusion"],
        "variants": list(variants),
        "n_queries": len(checks),
        "reproduced": dict(reproduced),
        "quality": qualities,
        "ceiling": {
            "original_in_candidates": (
                sum(r["original_in_candidates"] for r in checks) / len(checks) if checks else None
            ),
            "candidates": _stats([r["n_candidates"] for r in checks]),
        },
        "vs_l5w": [_paired(v, L5W, ok[v], ok[L5W], seed=seed) for v in variants],
        "selected": (
            select_rerank(qualities)
            if all(v in variants and qualities[v]["n"] for v in PRODUCT)
            else None
        ),
        "choice": choice,
        "decides": (
            _paired(choice, L5W, ok[choice], ok[L5W], seed=seed) if choice in variants else None
        ),
        "upper_vs_choice": (
            _paired(RS, choice, ok[RS], ok[choice], seed=seed)
            if RS in variants and choice in variants and choice != RS
            else None
        ),
        "cost": {
            v: {
                "rerank": _stats([r["cost"] for r in ok[v].values()]),
                "check": _stats([r["total_cost"] for r in ok[v].values()]),
                "over_cap": sum(r["total_cost"] > cost_cap for r in ok[v].values()),
            }
            for v in configs
        },
        "cost_cap": cost_cap,
        "latency": {
            v: _stats([r["latency_s"] for r in ok[v].values() if r.get("latency_s") is not None])
            for v in variants
        },
        "spent": dict(spent),
    }
    unseen = UNSEEN_FROM.isoformat()

    def hit5_of(c: str, unseen_part: bool) -> dict[str, Any]:
        part = [r for r in ok[c].values() if (r["published"] >= unseen) == unseen_part]
        n = len(part)
        return {"n": n, "hit_at_5": sum(r["hit_at_5"] for r in part) / n if n else None}

    summary["by_publication"] = {
        f"before {unseen}": {c: hit5_of(c, False) for c in configs},
        f"from {unseen}": {c: hit5_of(c, True) for c in configs},
    }
    return summary


def _pair_row(p: Mapping[str, Any], note: str = "") -> str:
    m = p["mcnemar"]
    return (
        f"| {p['a']} vs {p['b']}{note} | {p['n']} | {_f(p['delta_hit_at_5'])} "
        f"{_ci(p['ci_hit_at_5'])} | {_f(p['delta_mrr'])} {_ci(p['ci_mrr'])} "
        f"| {m['a_only']} / {m['b_only']} | {m['p']:.3g} |"
    )


def render_markdown(summary: Mapping[str, Any]) -> str:
    n, rep = summary["n_queries"], summary["reproduced"]
    variants = summary["variants"]
    configs = [L5, L5W, *variants]
    ceiling = summary["ceiling"]
    cands = ceiling["candidates"] or {}
    lines = [
        f"# Rerank variants on {summary['live_run']['run_id']} ({summary['run_id']})",
        "",
        (
            f"Git {summary['git_commit']}; {n} checks replayed offline (L5 reproduced in "
            f"{rep[L5]} of {n}, L5w in {rep[L5W]} of {rep['L5w_rows']} recorded); L5w's fusion "
            f"{summary['l5w_fusion']}. Candidates per check: mean {_f(cands.get('mean'), 1)}, "
            f"min {_f(cands.get('min'), 0)}, max {_f(cands.get('max'), 0)}; the original is "
            f"among them in {_f(ceiling['original_in_candidates'])} of checks. Ranks count "
            "within the run's top_k, so MRR is MRR@20."
        ),
        "",
        "| Config | Checks | Errors | Over cap | Unreadable | Hit@1 | Hit@5 | Hit@10 | MRR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in configs:
        q = summary["quality"][c]
        lines.append(
            f"| {LABELS[c]} | {q['n']} | {q['n_error']} | {q['n_capped']} | {q['unreadable']} "
            f"| {_f(q['hit_at_1'])} | {_f(q['hit_at_5'])} | {_f(q['hit_at_10'])} "
            f"| {_f(q['mrr'])} |"
        )
    lines += [
        "",
        "## Each variant vs L5w",
        "",
        "| Comparison | n | ΔHit@5 [95% CI] | ΔMRR [95% CI] | Only A / only B | McNemar p |",
        "|---|---|---|---|---|---|",
    ]
    choice = summary["choice"]
    for p in summary["vs_l5w"]:
        lines.append(_pair_row(p, " (L5r: decides)" if p["a"] == choice else ""))
    if summary["upper_vs_choice"] is not None:
        lines.append(_pair_row(summary["upper_vs_choice"]))
    selected = summary["selected"]
    verdict = f"**{selected}**." if selected else "none beats L5w."
    lines += [
        "",
        "## Selection (docs/phase2_spec.md; meaningful on the dev set only)",
        "",
        (
            "The product variant with the best Hit@5 (ties: MRR, then R1, R2, R3), if it "
            f"beats L5w: {verdict}"
        ),
        "",
        f"RERANK_CHOICE (L5r) when this ran: {choice or 'not set'}.",
        "",
        "## Cost per check (USD: mean / p50 / p95 / max)",
        "",
        (
            "The check's cost is the recorded live check's plus the rerank call; the cap is "
            f"${summary['cost_cap']:.2f}."
        ),
        "",
        "| Config | Rerank call | Whole check | Over the cap | Rerank latency (s, real calls) |",
        "|---|---|---|---|---|",
    ]

    def spread(s: Mapping[str, float] | None, digits: int) -> str:
        if not s:
            return "–"
        return " / ".join(_f(s[k], digits) for k in ("mean", "p50", "p95", "max"))

    for c in configs:
        cost = summary["cost"][c]
        latency = summary["latency"].get(c)
        calls = f"{spread(latency, 1)} (n = {latency['n']})" if latency else "–"
        lines.append(
            f"| {c} | {spread(cost['rerank'], 5)} | {spread(cost['check'], 4)} "
            f"| {cost['over_cap']} | {calls} |"
        )
    lines += ["", "## Hit@5 by the original's publication date", ""]
    parts = summary["by_publication"]
    lines.append("| Config | " + " | ".join(parts) + " |")
    lines.append("|---|" + "---|" * len(parts))
    for c in configs:
        cells = [
            f"{_f(part[c]['hit_at_5'])} (n = {part[c]['n']})" for part in parts.values()
        ]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    spent = ", ".join(f"{k} ${v:.4f}" for k, v in summary["spent"].items()) or "nothing"
    lines += ["", f"Spent on real calls in this run: {spent}.", ""]
    return "\n".join(lines)
