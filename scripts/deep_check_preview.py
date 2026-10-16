"""Deep-check preview on the 80 test queries (docs/phase2_spec.md, Deep check). Read-only.

usage: python scripts/deep_check_preview.py COMPETE_RUN_DIR

L5r's top 10 (R1 rows of the test rerank run) and Sonnet's list (claude-sonnet rows of the
compete run) are interleaved, ours first; a paper already listed is skipped (same OpenAlex
id or alias, DOI or normalised title, over every version / the listed and resolved forms);
the merged list is cut to 10. A merged entry is the original if any paper folded into it
matched the original (L5r: the R1 row's target_rank; Sonnet: the compete row's match).

One-off script of Phase 2 Step 4 (docs/reports/phase2-step4-rerank.md, "Deep-check
preview on the test set"), kept as it was run on 2026-10-15 with
COMPETE_RUN_DIR = runs/compete-20261015-162116. It makes no API calls (the test checks
are replayed offline) and writes deep_check_preview.jsonl into COMPETE_RUN_DIR.
"""

import json
import sys
from pathlib import Path

import numpy as np

from rk.config import load_config
from rk.corpus import normalize_doi
from rk.evaluate import mcnemar_exact, paired_bootstrap
from rk.live.replay import replay_run
from rk.retrieval import make_encoder
from rk.text import normalize_title

TEST_LIVE = Path("runs/live-20261015-000815")
TEST_RERANK = Path("runs/rerank-20261015-140923")
K = 10
compete_dir = Path(sys.argv[1])


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()]


rr = jsonl(TEST_RERANK / "results.jsonl")
r1 = {r["qid"]: r for r in rr if r["config"] == "R1"}
cr = jsonl(compete_dir / "results.jsonl")
son = {r["qid"]: r for r in cr if r["system"] == "claude-sonnet"}
ours = {r["qid"]: r for r in cr if r["system"] == "ours"}
assert set(son) == set(r1) == set(ours), (len(son), len(r1), len(ours))
bad = {q: r["status"] for q, r in son.items() if r["status"] != "ok"}
print("sonnet rows not ok:", bad)
qs = sorted(r1)

cfg = load_config("configs/phase1.yaml")
_, _, replays = replay_run(TEST_LIVE, make_encoder(cfg.retrieval), qids=set(qs))
pools = {rp.qid: rp.pool for rp in replays}


def keys_of_versions(versions: list[dict]) -> set[str]:
    keys: set[str] = set()
    for v in versions:
        keys |= {f"id:{w}" for w in (v["id"], *v.get("aliases", [])) if w}
        doi = normalize_doi(v.get("doi"))
        if doi:
            keys.add(f"doi:{doi}")
        t = normalize_title(v.get("title") or "")
        if t:
            keys.add(f"title:{t}")
    return keys


def keys_of_item(item: dict) -> set[str]:
    keys: set[str] = set()
    res = item.get("resolution") or {}
    verified = res.get("status") == "verified"
    for doi in (item.get("doi"), res.get("doi") if verified else None):
        d = normalize_doi(doi)
        if d:
            keys.add(f"doi:{d}")
    if verified and res.get("openalex_id"):
        keys.add(f"id:{res['openalex_id']}")
    for title in (item.get("title"), res.get("title") if verified else None):
        t = normalize_title(title or "")
        if t:
            keys.add(f"title:{t}")
    return keys


def merge(ours_list: list[tuple], llm_list: list[tuple]) -> list[dict]:
    """Each input: (keys, is_original, source, unverified). Returns merged entries."""
    merged: list[dict] = []
    seq = []
    for i in range(max(len(ours_list), len(llm_list))):
        if i < len(ours_list):
            seq.append(ours_list[i])
        if i < len(llm_list):
            seq.append(llm_list[i])
    for keys, is_orig, source, unverified in seq:
        dup = next((m for m in merged if m["keys"] & keys), None)
        if dup is not None:
            dup["keys"] |= keys
            dup["original"] = dup["original"] or is_orig
            dup["folded"].append(source)
            continue
        if len(merged) == K:
            continue  # still fold duplicates into the kept 10, but add nothing new
        merged.append({"keys": set(keys), "original": is_orig, "source": source,
                       "unverified": unverified, "folded": []})
    return merged


out = []
for q in qs:
    row, srow, pool = r1[q], son[q], pools[q]
    by_id = {r["id"]: r for r in pool.records}
    top = row["doc_ids"][:K]
    t = row["target_rank"]
    ours_list = [(keys_of_versions(pool.versions(by_id[d])), t == i, "ours", False)
                 for i, d in enumerate(top, start=1)]
    items = srow.get("items") or [] if srow["status"] == "ok" else []
    llm_list = [(keys_of_item(it), it.get("match") is not None, "llm",
                 (it.get("resolution") or {}).get("status") != "verified") for it in items[:K]]
    merged = merge(ours_list, llm_list)
    rank = next((i for i, m in enumerate(merged, start=1) if m["original"]), None)
    l5r_rank = t if t is not None and t <= K else None
    s_rank = srow.get("target_rank")
    lat_l5r = ours[q]["latency_s"] + row["latency_s"]
    out.append({
        "qid": q, "bucket": ours[q]["bucket"], "deep": rank, "l5r": l5r_rank, "sonnet": s_rank,
        "n_merged": len(merged), "n_llm_in": sum(m["source"] == "llm" for m in merged),
        "n_unverified": sum(m["unverified"] for m in merged if m["source"] == "llm"),
        "cost_l5r": row["total_cost"], "cost_sonnet": srow["cost"],
        "cost": row["total_cost"] + srow["cost"],
        "spent": srow["money_spent"], "cached": srow["cached"],
        "lat_l5r": lat_l5r, "lat_sonnet": srow["latency_s"],
        "lat": max(lat_l5r, srow["latency_s"]),
        "searches": srow["searches"], "n_sonnet_items": len(items),
        "dups": sum(len(m["folded"]) for m in merged),
    })

Path(compete_dir / "deep_check_preview.jsonl").write_text(
    "".join(json.dumps(o) + "\n" for o in out), encoding="utf-8")
n = len(out)


def hits(key: str, k: int) -> list[int]:
    return [int(o[key] is not None and o[key] <= k) for o in out]


def mrr(key: str) -> list[float]:
    return [1 / o[key] if o[key] is not None and o[key] <= K else 0.0 for o in out]


print(f"n = {n}")
for key in ("deep", "l5r", "sonnet"):
    print(f"{key}: " + "  ".join(f"Hit@{k} {sum(hits(key, k))} ({np.mean(hits(key, k)):.3f})"
                                 for k in (1, 5, 10)) + f"  MRR@10 {np.mean(mrr(key)):.3f}")
for k in (1, 5, 10):
    d = [a - b for a, b in zip(hits("deep", k), hits("l5r", k))]
    x, y = sum(v > 0 for v in d), sum(v < 0 for v in d)
    ci = paired_bootstrap(d, seed=42)
    print(f"deep vs L5r Hit@{k}: {np.mean(d):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] "
          f"only deep / only L5r {x}/{y} McNemar p={mcnemar_exact(x, y):.3g}")
d = [a - b for a, b in zip(mrr("deep"), mrr("l5r"))]
ci = paired_bootstrap(d, seed=42)
print(f"deep vs L5r MRR@10: {np.mean(d):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
gain = [o["qid"][:5] + f"(l5r {o['l5r']}, son {o['sonnet']}, deep {o['deep']})"
        for o in out if (o["deep"] or 99) <= 5 < (o["l5r"] or 99)]
loss = [o["qid"][:5] + f"(l5r {o['l5r']}, son {o['sonnet']}, deep {o['deep']})"
        for o in out if (o["l5r"] or 99) <= 5 < (o["deep"] or 99)]
print("gains@5:", gain)
print("losses@5:", loss)
either5 = sum(1 for o in out if (o["l5r"] or 99) <= 5 or (o["sonnet"] or 99) <= 5)
either10 = sum(1 for o in out if o["l5r"] or (o["sonnet"] or 99) <= 10)
print(f"oracle either top5 {either5}, either top10 {either10}")
llm_in = sum(o["n_llm_in"] for o in out)
unv = sum(o["n_unverified"] for o in out)
tot = sum(o["n_merged"] for o in out)
print(f"merged entries {tot} (mean {tot / n:.2f}); from Sonnet {llm_in}; unverified {unv} "
      f"= {unv / tot:.3%} of merged; duplicates folded {sum(o['dups'] for o in out)}")
for key in ("cost", "cost_l5r", "cost_sonnet", "lat", "lat_l5r", "lat_sonnet", "searches"):
    v = np.array([o[key] for o in out], dtype=float)
    print(f"{key}: mean {v.mean():.4f} p50 {np.median(v):.4f} p95 {np.percentile(v, 95):.4f} "
          f"max {v.max():.4f}")
print(f"cached {sum(o['cached'] for o in out)}; new spend ${sum(o['spent'] for o in out):.4f}")
for b in ("1", "2-10", "11-100", "not found"):
    sel = [o for o in out if o["bucket"] == b]
    print(f"bucket {b} n={len(sel)}: deep {sum((o['deep'] or 99) <= 5 for o in sel)} "
          f"L5r {sum((o['l5r'] or 99) <= 5 for o in sel)} "
          f"Sonnet {sum((o['sonnet'] or 99) <= 5 for o in sel)}")
