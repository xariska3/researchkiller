"""The chart data: every system on the 80 test queries (exploratory).

usage: python scripts/test80_comparison.py [--out docs/evaluation/test80_comparison.csv]

Reads run files only (no calls): data/results.sqlite and the runs/ directories below.
Writes the CSV and prints the Markdown tables of docs/reports/phase2-step4-rerank.md,
section "Every system on the 80 test queries (exploratory)".

- L5r: the R1 rows of the test rerank run. Cost is the check's plus the rerank call's;
  latency is L5w's recorded check latency plus the rerank call's.
- L5w, L1: the live test run's rows.
- Opus, Sonnet, gpt-6-luna, Perplexity and both Semantic Scholar searches: compete runs.
  gpt-6-luna has Step 3's answers only (its over-cap queries are left out).
- Hit@1 and Hit@5 over each list's top 10. Unresolved references: listed papers that no
  free route verifies (0 for OpenAlex records). McNemar on Hit@5, L5r against each
  system on the queries both answered, with Holm's adjustment across the comparisons.
- L5r's Hit@5 misses per split (dev, test), by where the original was lost.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from rk.compete.evaluate import holm
from rk.evaluate import mcnemar_exact

RESULTS_DB = Path("data/results.sqlite")
LIVE_TEST = "live-20261015-000815"
RERANK_TEST = Path("runs/rerank-20261015-140923")
SPLITS = {  # split: (rerank run, live run id)
    "dev": (Path("runs/rerank-20261015-132644"), "live-20261014-122144"),
    "test": (RERANK_TEST, LIVE_TEST),
}
COMPETE = [  # label, compete run, system, note
    (
        "Claude Opus 5.5 + web search",
        Path("runs/compete-20261015-101555"),
        "claude-opus",
        "cached (Step 3 addendum)",
    ),
    (
        "Claude Sonnet 5.5 + web search",
        Path("runs/compete-20261015-162116"),
        "claude-sonnet",
        "cached (deep-check preview run), alone",
    ),
    (
        "gpt-6-luna + web search",
        Path("runs/compete-20261015-012022"),
        "openai-luna",
        "cached Step 3 answers only: the 39 test queries in Step 3's sample",
    ),
    ("Perplexity Sonar", Path("runs/compete-20261015-172548"), "perplexity", "search context low"),
    (
        "Semantic Scholar snippet search",
        Path("runs/compete-20261015-172548"),
        "s2-snippets",
        "the proposal as the query",
    ),
    (
        "Semantic Scholar paper search",
        Path("runs/compete-20261015-172548"),
        "s2-search",
        "the proposal as the query",
    ),
]
OURS = ("L5r", "L5w", "L1")  # OpenAlex records: nothing to resolve
TOP = 10
COLUMNS = [
    "system",
    "n",
    "hit_at_1",
    "hit_at_5",
    "unresolved_rate",
    "mean_cost_usd",
    "median_latency_s",
    "mean_latency_s",
    "l5r_only",
    "system_only",
    "mcnemar_p",
    "mcnemar_p_holm",
    "note",
]

Entry = dict[str, Any]  # rank (None outside the top 10), cost, latency, items, unresolved


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def rerank_rows(run_dir: Path, config: str) -> dict[str, dict[str, Any]]:
    return {r["qid"]: r for r in jsonl(run_dir / "results.jsonl") if r["config"] == config}


def live_rows(db: sqlite3.Connection, run_id: str) -> dict[str, dict[str, sqlite3.Row]]:
    rows: dict[str, dict[str, sqlite3.Row]] = {}
    for r in db.execute("SELECT * FROM live_query_results WHERE run_id = ?", (run_id,)):
        rows.setdefault(r["config"], {})[r["qid"]] = r
    return rows


def entry(
    rank: int | None, cost: float, latency: float, items: int = 0, unresolved: int = 0
) -> Entry:
    top = rank if rank is not None and rank <= TOP else None
    return {"rank": top, "cost": cost, "latency": latency, "items": items, "unresolved": unresolved}


def compete_entries(run_dir: Path, system: str, qids: set[str]) -> tuple[dict[str, Entry], int]:
    """A system's answers to the queries, and how many were over its cost cap."""
    rows = {
        r["qid"]: r
        for r in jsonl(run_dir / "results.jsonl")
        if r["system"] == system and r["qid"] in qids
    }
    ok = {q: r for q, r in rows.items() if r["status"] == "ok"}
    capped = sum(r["status"] == "capped" for r in rows.values())
    if len(rows) - len(ok) != capped:
        raise SystemExit(f"{run_dir} {system}: answers missing for a reason other than the cap")
    entries = {
        q: entry(r["target_rank"], r["cost"], r["latency_s"], r["n_items"], r["n_unresolved"])
        for q, r in ok.items()
    }
    return entries, capped


def hit(e: Entry, k: int) -> int:
    return int(e["rank"] is not None and e["rank"] <= k)


def table(db: sqlite3.Connection) -> list[dict[str, Any]]:
    r1 = rerank_rows(RERANK_TEST, "R1")
    live = live_rows(db, LIVE_TEST)
    qids = sorted(r1)
    if len(qids) != 80 or any(
        len(live[c]) != 80 or any(r["status"] != "ok" for r in live[c].values())
        for c in ("L1", "L5w")
    ):
        raise SystemExit("expected 80 finished test checks in every source")
    systems: dict[str, dict[str, Entry]] = {
        "L5r": {
            q: entry(
                r1[q]["target_rank"],
                r1[q]["total_cost"],
                live["L5w"][q]["latency_s"] + r1[q]["latency_s"],
            )
            for q in qids
        },
        **{
            c: {
                q: entry(
                    live[c][q]["target_rank"], live[c][q]["method_cost"], live[c][q]["latency_s"]
                )
                for q in qids
            }
            for c in ("L5w", "L1")
        },
    }
    notes = {
        "L5r": "default check: L5w + Qwen listwise rerank",
        "L5w": "live check before the reranker",
        "L1": "instant preview: one OpenAlex semantic search + local rerank, no LLM",
    }
    for label, run_dir, system, note in COMPETE:
        systems[label], capped = compete_entries(run_dir, system, set(qids))
        notes[label] = note + (f", less {capped} over its cost cap there" if capped else "")

    base = systems["L5r"]
    rows = []
    for name, data in systems.items():
        answered = sorted(data)
        n = len(answered)
        items = sum(data[q]["items"] for q in answered)
        unresolved = sum(data[q]["unresolved"] for q in answered)
        latency = np.array([data[q]["latency"] for q in answered], dtype=float)
        row: dict[str, Any] = {
            "system": name,
            "n": n,
            "hit_at_1": sum(hit(data[q], 1) for q in answered) / n,
            "hit_at_5": sum(hit(data[q], 5) for q in answered) / n,
            "unresolved_rate": 0.0 if name in OURS else (unresolved / items if items else None),
            "mean_cost_usd": float(np.mean([data[q]["cost"] for q in answered])),
            "median_latency_s": float(np.median(latency)),
            "mean_latency_s": float(latency.mean()),
            "note": notes[name],
        }
        if name != "L5r":
            a_only = sum(hit(base[q], 5) > hit(data[q], 5) for q in answered)
            b_only = sum(hit(base[q], 5) < hit(data[q], 5) for q in answered)
            row.update(l5r_only=a_only, system_only=b_only, mcnemar_p=mcnemar_exact(a_only, b_only))
        rows.append(row)
    compared = [r for r in rows if r["system"] != "L5r"]
    for r, p in zip(compared, holm([r["mcnemar_p"] for r in compared])):
        r["mcnemar_p_holm"] = p
    luna = systems["gpt-6-luna + web search"]
    on_luna = sum(hit(base[q], 5) for q in luna)
    print(f"L5r on gpt-6-luna's {len(luna)} queries: Hit@5 {on_luna} ({on_luna / len(luna):.3f})")
    return rows


def csv_value(key: str, value: Any) -> str:
    if value is None:
        return ""
    if key in ("hit_at_1", "hit_at_5", "unresolved_rate"):
        return f"{value:.4f}"
    if key == "mean_cost_usd":
        return f"{value:.6f}"
    if key.endswith("latency_s"):
        return f"{value:.2f}"
    if key.startswith("mcnemar"):
        return f"{value:.3g}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow([csv_value(c, r.get(c)) for c in COLUMNS])


def p_text(p: float) -> str:
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        (
            "| System | n | Hit@1 | Hit@5 | Unresolved references | Mean cost per query "
            "| Latency median / mean (s) | Hit@5: only L5r / only it | McNemar p | Holm p |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rate = r["unresolved_rate"]
        unresolved = "– (lists nothing)" if rate is None else "0"
        if rate:
            unresolved = f"{100 * rate:.1f}%"
        cost = "free" if r["mean_cost_usd"] == 0 else f"${r['mean_cost_usd']:.4f}"
        test = (
            "– | – | –"
            if r["system"] == "L5r"
            else f"{r['l5r_only']} / {r['system_only']} | {p_text(r['mcnemar_p'])} "
            f"| {p_text(r['mcnemar_p_holm'])}"
        )
        lines.append(
            f"| {r['system']} | {r['n']} | {r['hit_at_1']:.3f} | {r['hit_at_5']:.3f} "
            f"| {unresolved} | {cost} | {r['median_latency_s']:.1f} / {r['mean_latency_s']:.1f} "
            f"| {test} |"
        )
    return "\n".join(lines)


def misses(db: sqlite3.Connection) -> str:
    """L5r's Hit@5 misses per split, by where the original was lost."""
    lines = [
        (
            "| Split | n | L5r Hit@5 misses | Original never reached the candidates "
            "| Never returned by any OpenAlex search | Returned, but below the top 30 of both "
            "fusions | Among the candidates, reranked below 5 |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    totals = [0] * 6
    for split, (run_dir, live_id) in SPLITS.items():
        r1 = rerank_rows(run_dir, "R1")
        found = {q: r["found_query"] for q, r in live_rows(db, live_id)["L5"].items()}
        missed = [q for q, r in r1.items() if not r["hit_at_5"]]
        outside = [q for q in missed if not r1[q]["original_in_candidates"]]
        never = [q for q in outside if found[q] is None]
        returned = sorted(q[:5] for q in outside if found[q] is not None)
        inside = sorted(
            q[:5] + (f", at {r1[q]['target_rank']}" if r1[q]["target_rank"] else "")
            for q in missed
            if r1[q]["original_in_candidates"]
        )
        counts = [len(r1), len(missed), len(outside), len(never), len(returned), len(inside)]
        totals = [t + c for t, c in zip(totals, counts)]
        lines.append(
            f"| {split.capitalize()} | {' | '.join(str(c) for c in counts[:4])} "
            f"| {len(returned)} ({', '.join(returned)}) | {len(inside)} ({'; '.join(inside)}) |"
        )
    lines.append(f"| Both | {' | '.join(str(t) for t in totals)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation/test80_comparison.csv"))
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{RESULTS_DB.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = table(db)
    write_csv(rows, args.out)
    print(f"wrote {args.out}\n")
    print(markdown(rows))
    print()
    print(misses(db))


if __name__ == "__main__":
    main()
