"""L5r on the 80 test queries with the final-list merge of versions, as a check only.

The Step 5 product fix merges versions of one paper in a check's final list
(rk.live.pipeline.merge_versions); live-eval doesn't. This script measures what the
merge would change on the test set, at $0:
- it replays the test run's checks offline from the OpenAlex cache (rk.live.replay);
- takes each check's recorded L5r top 20 (data/results.sqlite);
- merges it as the tool does, and finds the original's rank again.

It makes no API calls: the replay is offline, and the rerank isn't redone (the L5r rows
hold its order). It prints Hit@1/5/10 and MRR@20 before and after the merge, every
merge, and every query whose original moved.

Run from the repository root: python scripts/l5r_merge_check.py
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from rk.corpus import load_frozen_corpus
from rk.evaluate import HIT_KS
from rk.live.evaluate import original_rank
from rk.live.pipeline import merge_versions, ranked_candidates
from rk.live.replay import load_run, replay_run
from rk.retrieval import make_encoder

RUN = Path("runs/live-20261015-000815")  # Step 2's test run, with L5r rows since #16


def scores(ranks: dict[str, int | None]) -> dict[str, float]:
    n = len(ranks)
    out = {f"Hit@{k}": sum(r is not None and r <= k for r in ranks.values()) / n for k in HIT_KS}
    out["MRR@20"] = sum(1 / r for r in ranks.values() if r) / n
    return out


def main() -> None:
    meta = json.loads((RUN / "run.json").read_text(encoding="utf-8"))
    with closing(sqlite3.connect(meta["results_db"])) as conn:
        rows = conn.execute(
            "SELECT qid, target_id, target_rank, doc_ids FROM live_query_results "
            "WHERE run_id = ? AND config = 'L5r' AND status = 'ok'",
            (meta["run_id"],),
        ).fetchall()
    l5r = {qid: (target, rank, json.loads(ids)) for qid, target, rank, ids in rows}
    _, run_cfg, _ = load_run(RUN)  # the encoder the run used: the replay ranks as it did
    _, cfg, replays = replay_run(RUN, make_encoder(run_cfg.retrieval), qids=set(l5r))
    assert cfg.live is not None
    top_k = cfg.live.top_k
    corpus, _, _ = load_frozen_corpus(cfg.corpus.out_dir)
    originals = {r["id"]: r for r in corpus if r["id"] in {t for t, _, _ in l5r.values()}}
    before: dict[str, int | None] = {}
    after: dict[str, int | None] = {}
    merges: list[str] = []
    for rp in replays:
        target, recorded, doc_ids = l5r[rp.qid]
        candidates = ranked_candidates(rp.pool, doc_ids, top_k)
        original = originals[target]
        before[rp.qid] = original_rank([c.versions for c in candidates], original, top_k)[0]
        assert before[rp.qid] == recorded, f"{rp.qid}: the replay doesn't reproduce L5r"
        merged, pairs = merge_versions(candidates)
        after[rp.qid] = original_rank([c.versions for c in merged], original, top_k)[0]
        merges += [f"{rp.qid}: kept {kept}, merged {other}" for kept, other in pairs]
    print(f"{len(before)} test checks; {len(merges)} merges in {len({m[:5] for m in merges})}")
    for name, ranks in (("before", before), ("after", after)):
        print(f"  {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in scores(ranks).items()))
    for line in merges:
        print("  " + line)
    moved = {q: (before[q], after[q]) for q in before if before[q] != after[q]}
    print(f"originals that moved: {len(moved)}")
    for qid, (old, new) in sorted(moved.items()):
        print(f"  {qid}: rank {old} -> {new}")


if __name__ == "__main__":
    main()
