"""L5r vs Claude Opus (web search) on the 80 test queries only. Read-only, exploratory.

L5r: R1 rows of the test rerank run. Opus: claude-opus rows of the 200-query compete run.
Both scored on their top 10; MRR@10.

One-off script of Phase 2 Step 4 (docs/reports/phase2-step4-rerank.md, "L5r against
Claude Opus on the test set"), kept as it was run on 2026-10-15; only line breaks and
type hints were added since. It reads only the two runs' results.jsonl files.

Run from the repository root: python scripts/l5r_vs_opus.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rk.evaluate import mcnemar_exact, paired_bootstrap

rows = [json.loads(x) for x in
        Path("runs/rerank-20261015-140923/results.jsonl").read_text("utf-8").splitlines()]
r1 = {r["qid"]: r for r in rows if r["config"] == "R1"}
l5w = {r["qid"]: r for r in rows if r["config"] == "L5w"}
assert len(r1) == 80 and all(r["status"] == "ok" for r in r1.values())
compete = [json.loads(x) for x in
           Path("runs/compete-20261015-101555/results.jsonl").read_text("utf-8").splitlines()]
opus = {r["qid"]: r for r in compete if r["system"] == "claude-opus" and r["qid"] in r1}
assert set(opus) == set(r1), (len(opus), len(r1))
print("opus statuses:", {s: sum(r["status"] == s for r in opus.values()) for s in ("ok",)},
      "parse errors:", sum(r["parse_error"] is not None for r in opus.values()))
qs = sorted(r1)


def rank10(r: dict[str, Any]) -> int | None:
    t = r["target_rank"]
    return t if t is not None and t <= 10 else None


def metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    return {
        "hit1": sum(r == 1 for r in ranks if r) / n,
        "hit5": sum(1 for r in ranks if r and r <= 5) / n,
        "hit10": sum(1 for r in ranks if r and r <= 10) / n,
        "mrr10": sum(1 / r for r in ranks if r) / n,
        "n_hit1": sum(r == 1 for r in ranks if r),
        "n_hit5": sum(1 for r in ranks if r and r <= 5),
        "n_hit10": sum(1 for r in ranks if r and r <= 10),
    }


ra = [rank10(r1[q]) for q in qs]
rb = [rank10(opus[q]) for q in qs]
rw = [rank10(l5w[q]) for q in qs]
for name, rk in (("L5r", ra), ("Opus", rb), ("L5w", rw)):
    m = metrics(rk)
    print(f"{name}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                  for k, v in m.items()))


def pair(a: list[int | None], b: list[int | None], k: int) -> tuple[Any, ...]:
    ha = [int(r is not None and r <= k) for r in a]
    hb = [int(r is not None and r <= k) for r in b]
    d = [x - y for x, y in zip(ha, hb)]
    x, y = sum(v > 0 for v in d), sum(v < 0 for v in d)
    ci = paired_bootstrap(d, seed=42)
    return np.mean(d), ci, x, y, mcnemar_exact(x, y)


for label, a, b in (("L5r vs Opus", ra, rb), ("L5w vs Opus", rw, rb)):
    for k in (1, 5, 10):
        dm, ci, x, y, p = pair(a, b, k)
        print(f"{label} Hit@{k}: d {dm:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] "
              f"only A/B {x}/{y} p={p:.3g}")
    drr = [(1 / x if x else 0) - (1 / y if y else 0) for x, y in zip(a, b)]
    ci = paired_bootstrap(drr, seed=42)
    print(f"{label} MRR@10: d {np.mean(drr):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")

either = sum(1 for x, y in zip(ra, rb) if (x and x <= 5) or (y and y <= 5))
print(f"either top5: {either}/80")
# Opus when it finds it: rank 1 share
found = [r for r in rb if r]
print(f"Opus found (top 10) {len(found)}, at rank 1 {sum(r == 1 for r in found)}")
# by date split, bucket
compete_sample = {r["qid"]: r["bucket"] for r in compete if r["system"] == "claude-opus"}
for b in ("1", "2-10", "11-100", "not found"):
    sel = [i for i, q in enumerate(qs) if compete_sample[q] == b]
    ha = sum(1 for i in sel if ra[i] and ra[i] <= 5)
    hb = sum(1 for i in sel if rb[i] and rb[i] <= 5)
    print(f"bucket {b} n={len(sel)}: L5r {ha}, Opus {hb}")
# Opus cost / latency on these 80, L5r cost
oc = [opus[q]["cost"] for q in qs]
ol = [opus[q]["latency_s"] for q in qs]
lc = [r1[q]["total_cost"] for q in qs]
print(f"Opus cost mean {np.mean(oc):.4f} p50 {np.median(oc):.4f}; latency mean {np.mean(ol):.1f}")
print(f"L5r cost mean {np.mean(lc):.4f}")
