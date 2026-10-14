"""Offline replay of a live-eval run: the dev-set fusion comparison and the misses export.

Each finished L5 row's trace lists the queries the agent ran and the searches
each one sent, in order. Replaying those searches from the OpenAlex cache
rebuilds the same candidate pool with no OpenAlex calls (the client is
offline: a search that isn't cached is an error) and no LLM calls (the agent's
queries come from the trace). The pool gets each query's results and is ranked
after each query, as in the live loop, so documents are encoded in the same
batches. The replay uses the run's own config from run.json, so the searches
carry the parameters they were cached under, and it counts how many replayed
L1 and L5 rankings match the ones the run recorded.

- evaluate_fusions: the pre-declared final-fusion variants (rk.live.fusion)
  on the run's queries, with Phase 1's statistics against L5 and L1, and the
  selection rule for L5w (docs/phase2_spec.md, Step 2 decisions).
- export_misses: a seeded random sample of the queries where L5 missed the
  original in its top 5, with L5's top 3 results, as a CSV to check by hand.
"""

from __future__ import annotations

import csv
import json
import random
import sqlite3
from collections.abc import Collection
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rk.config import Config, LiveConfig, config_from_dict
from rk.corpus import load_frozen_corpus
from rk.evaluate import HIT_KS, RUN_FILE, EvalError, _f, _paired, _paired_table
from rk.live.evaluate import COST_CAP, KIND, TABLE, connect, original_rank
from rk.live.fusion import B5, FUSIONS, VARIANTS, final_ranking, l1_ranking
from rk.live.openalex_search import LiveSearchClient, OpenAlexCache, SearchResponse
from rk.live.pipeline import proposal_records
from rk.live.pool import CandidatePool
from rk.retrieval import Encoder
from rk.targets import Record

FUSION_JSON = "fusion.json"
FUSION_MD = "fusion.md"
MISSES_FILE = Path("docs/evaluation/live_misses.csv")
MISSES_TOP = 3  # results shown per missed query
ABSTRACT_OPENING = 300  # characters
FUSION_LABELS = {
    B5: "L5 (b5: B5's fusion)",
    "proposal-equal": "F1 proposal-equal",
    "proposal-2x": "F2 proposal-2x",
    "l1-l5": "F3 l1-l5",
    "proposal-only": "F4 proposal-only",
}


@dataclass(frozen=True)
class Replay:
    """One L5 check rebuilt from the cache."""

    qid: str
    target_id: str
    queries: list[str]  # the proposal first
    pool: CandidatePool  # the final pool
    l1: list[str]  # L1's ranking: the proposal's search reranked on its own
    recorded_l5: list[str]  # the top live.top_k the run recorded
    recorded_l1: list[str] | None  # the same for L1, if the run has an L1 row


def load_run(run_dir: Path) -> tuple[dict[str, Any], Config, LiveConfig]:
    """A live run's run.json, the config it ran with, and its live: section."""
    meta_path = run_dir / RUN_FILE
    if not meta_path.exists():
        raise EvalError(f"{run_dir} is not a run directory (no {RUN_FILE})")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("kind") != KIND:
        raise EvalError(f"{run_dir} is not a live-eval run")
    cfg = config_from_dict(meta["config"])
    if cfg.live is None:
        raise EvalError(f"{run_dir} ran without a live: section")
    return meta, cfg, cfg.live


def _rows(meta: dict[str, Any], config: str) -> dict[str, sqlite3.Row]:
    """The run's finished rows of one config (status ok, not over the cost cap), by qid."""
    with closing(connect(Path(meta["results_db"]))) as conn:
        rows = conn.execute(
            f"SELECT * FROM {TABLE} WHERE run_id = ? AND config = ? AND status = 'ok'",
            (meta["run_id"], config),
        ).fetchall()
    return {r["qid"]: r for r in rows if r["stop_reason"] != COST_CAP}


def replay_run(
    run_dir: Path, encoder: Encoder, *, qids: Collection[str] | None = None
) -> tuple[dict[str, Any], Config, list[Replay]]:
    """Rebuild the run's L5 checks (those in `qids`, if given) from the cache, offline."""
    meta, cfg, live = load_run(run_dir)
    l5_rows, l1_rows = _rows(meta, "L5"), _rows(meta, "L1")
    replays = []
    with closing(OpenAlexCache(live.cache_path)) as cache:
        search = LiveSearchClient(live, cache, max_age=None, offline=True)
        for qid in sorted(l5_rows):
            if qids is not None and qid not in qids:
                continue
            row = l5_rows[qid]
            trace = json.loads(row["trace"])
            queries = trace["queries"]
            pool = CandidatePool(encoder, max_size=live.max_pool_size)
            responses: list[SearchResponse] = []
            for index, query in enumerate(queries):
                for s in (s for s in trace["searches"] if s["query"] == query):
                    response = search.search(query, s["kind"])
                    pool.add(response.records, query_index=index, kind=s["kind"])
                    responses.append(response)
                pool.rank(query)  # as the live loop did, so encoding batches match
            records = proposal_records(queries[0], responses)
            l1 = l1_ranking(records, queries[0], encoder, max_size=live.max_pool_size)
            l1_row = l1_rows.get(qid)
            replays.append(
                Replay(
                    qid=qid,
                    target_id=row["target_id"],
                    queries=queries,
                    pool=pool,
                    l1=l1,
                    recorded_l5=json.loads(row["doc_ids"]),
                    recorded_l1=json.loads(l1_row["doc_ids"]) if l1_row else None,
                )
            )
    return meta, cfg, replays


def _scored(rank: int | None) -> dict[str, Any]:
    return {
        "target_rank": rank,
        **{f"hit_at_{k}": int(rank is not None and rank <= k) for k in HIT_KS},
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
    }


def _originals(cfg: Config, target_ids: Collection[str]) -> dict[str, Record]:
    corpus, _, _ = load_frozen_corpus(cfg.corpus.out_dir)
    return {r["id"]: r for r in corpus if r["id"] in target_ids}


def select_fusion(quality: dict[str, dict[str, Any]]) -> str | None:
    """L5w's fusion: the variant with the best Hit@5 (ties: MRR@20, then declared order).

    None unless its Hit@5 is above L5's (b5).
    """
    best = max(
        VARIANTS,
        key=lambda f: (quality[f]["hit_at_5"], quality[f]["mrr"], -VARIANTS.index(f)),
    )
    return best if quality[best]["hit_at_5"] > quality[B5]["hit_at_5"] else None


def evaluate_fusions(run_dir: Path, encoder: Encoder) -> dict[str, Any]:
    """Every fusion on the run's replayed L5 checks; writes fusion.json and fusion.md."""
    meta, cfg, replays = replay_run(run_dir, encoder)
    if not replays:
        raise EvalError(f"{run_dir} has no finished L5 checks to replay")
    top_k = meta["config"]["live"]["top_k"]
    originals = _originals(cfg, {r.target_id for r in replays})
    scored: dict[str, dict[str, dict[str, Any]]] = {f: {} for f in FUSIONS}
    reproduced = {"L5": 0, "L1": 0, "L1_rows": 0}
    for rp in replays:
        by_id = {r["id"]: r for r in rp.pool.records}
        original = originals[rp.target_id]
        for fusion in FUSIONS:
            ranking = final_ranking(rp.pool, rp.queries, fusion, l1=rp.l1)
            if fusion == B5:
                reproduced["L5"] += ranking[:top_k] == rp.recorded_l5
            versions = [rp.pool.versions(by_id[d]) for d in ranking]
            scored[fusion][rp.qid] = _scored(original_rank(versions, original, top_k)[0])
        if rp.recorded_l1 is not None:
            reproduced["L1_rows"] += 1
            reproduced["L1"] += rp.l1[:top_k] == rp.recorded_l1
    # L1 is scored as the run recorded it (its replay is checked above).
    l1_rows = _rows(meta, "L1")
    scored["L1"] = {q: _scored(l1_rows[q]["target_rank"]) for q in scored[B5] if q in l1_rows}

    def quality(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
        def mean(key: str) -> float:
            return sum(r[key] for r in rows.values()) / len(rows)

        return {
            "n": len(rows),
            **{f"hit_at_{k}": mean(f"hit_at_{k}") for k in HIT_KS},
            "mrr": mean("reciprocal_rank"),
        }

    seed = cfg.run.seed
    summary = {
        "run_id": meta["run_id"],
        "top_k": top_k,
        "n_queries": len(replays),
        "reproduced": reproduced,
        "quality": {name: quality(rows) for name, rows in scored.items()},
        "vs_l5": [_paired(f, B5, scored[f], scored[B5], seed=seed) for f in VARIANTS],
        "vs_l1": [_paired(f, "L1", scored[f], scored["L1"], seed=seed) for f in FUSIONS],
    }
    summary["selected"] = select_fusion(summary["quality"])
    (run_dir / FUSION_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / FUSION_MD).write_text(render_fusions(summary), encoding="utf-8")
    return summary


def render_fusions(summary: dict[str, Any]) -> str:
    k, rep, n = summary["top_k"], summary["reproduced"], summary["n_queries"]
    lines = [
        f"# Final-fusion variants on {summary['run_id']} (offline replay)",
        "",
        (
            f"{n} L5 checks replayed from cached searches and recorded agent queries: no "
            f"OpenAlex or LLM calls. The replayed L5 ranking matches the recorded one in "
            f"{rep['L5']} of {n} checks, L1's in {rep['L1']} of {rep['L1_rows']}. Ranks count "
            f"within the top {k}."
        ),
        "",
        f"| Fusion | Queries | Hit@1 | Hit@5 | Hit@10 | MRR@{k} |",
        "|---|---|---|---|---|---|",
    ]
    for name in (*FUSIONS, "L1"):
        q = summary["quality"][name]
        label = FUSION_LABELS.get(name, "L1 (recorded)")
        lines.append(
            f"| {label} | {q['n']} | {_f(q['hit_at_1'])} | {_f(q['hit_at_5'])} "
            f"| {_f(q['hit_at_10'])} | {_f(q['mrr'])} |"
        )
    selected = summary["selected"]
    choice = (
        f"L5w = {FUSION_LABELS[selected]}."
        if selected
        else "No variant beats L5 on Hit@5, so L5 stays and there is no L5w."
    )
    return "\n".join(
        [
            *lines,
            "",
            "## Each variant vs L5 (exploratory: this is the selection set)",
            "",
            *_paired_table(summary["vs_l5"]),
            "",
            "## Each fusion vs L1",
            "",
            *_paired_table(summary["vs_l1"]),
            "",
            "## Selection (docs/phase2_spec.md)",
            "",
            f"The best variant by Hit@5 (ties: MRR@{k}, then declared order), if it beats L5: "
            + choice,
        ]
    ) + "\n"


MISSES_COLUMNS = (
    "qid",
    "proposal",
    "original_title",
    "original_rank",
    *(
        f"top{i}_{field}"
        for i in range(1, MISSES_TOP + 1)
        for field in ("title", "year", "abstract")
    ),
    "top1_close_prior_art",  # for the reader: is the top result genuinely close prior art?
    "note",
)


def opening(text: str, chars: int = ABSTRACT_OPENING) -> str:
    """The first `chars` characters of a text, cut at a word, with an ellipsis if cut."""
    if len(text) <= chars:
        return text
    return text[:chars].rsplit(" ", 1)[0] + " …"


def export_misses(
    run_dir: Path, encoder: Encoder, *, n: int, out_path: Path = MISSES_FILE
) -> int:
    """A seeded sample (run.seed) of `n` queries where L5 missed the original in its top 5.

    Each row has the proposal, the original's title and its L5 rank (or
    "none"), and L5's top 3 results (title, year, abstract opening), in qid
    order. The last two columns are left empty for the reader. The file is
    UTF-8 with a byte-order mark, and an existing file is never overwritten.
    Returns the number of rows.
    """
    if out_path.exists():
        raise EvalError(
            f"{out_path} already exists: refusing to overwrite it (it may hold answers)"
        )
    meta, cfg, _ = load_run(run_dir)
    l5 = _rows(meta, "L5")
    missed = sorted(q for q, r in l5.items() if not r["hit_at_5"])
    sample = sorted(random.Random(cfg.run.seed).sample(missed, min(n, len(missed))))
    _, _, replays = replay_run(run_dir, encoder, qids=set(sample))
    originals = _originals(cfg, {r.target_id for r in replays})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(MISSES_COLUMNS)
        for rp in replays:
            by_id = {r["id"]: r for r in rp.pool.records}
            rank = l5[rp.qid]["target_rank"]
            top = []
            for doc_id in rp.recorded_l5[:MISSES_TOP]:
                record = by_id[doc_id]
                year = record["publication_date"][:4]
                top += [record["title"], year, opening(record["abstract"])]
            top += [""] * (3 * MISSES_TOP - len(top))
            writer.writerow(
                [
                    rp.qid,
                    rp.queries[0],
                    originals[rp.target_id]["title"],
                    "none" if rank is None else rank,
                    *top,
                    "",
                    "",
                ]
            )
    return len(replays)
