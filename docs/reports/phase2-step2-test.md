# Phase 2 Step 2: the L5w test run (80 queries from A0187)

This is the test of L5w pre-declared in [docs/phase2_spec.md](../phase2_spec.md)
(Evaluation, final-fusion variants). L5w is **F1, proposal-equal**: L5's check
with its final pool ranked by a weighted RRF in which the proposal's ranking
weighs as much as all agent queries together. It was chosen on the 120-query
dev set ([phase2-step2-full.md](phase2-step2-full.md),
`runs/live-20261014-122144/fusion.md`). The test set decides, with L5w vs L5
and L5w vs L1 paired: Hit@5 with the exact McNemar test, MRR@20 supporting.

- **Run**: `runs/live-20261015-000815`. Every row is in `data/results.sqlite`
  (table `live_query_results`) and in MLflow (experiment
  `researchkiller-phase2-live`).
  - Started from `main` at eb6dc36 (PR #10 merged, clean tree) on
    2026-10-15 at 00:08 UTC.
  - Crashed at 00:36 UTC on query 70, A0272, on a bug that this run found
    (see [The crash and the resume](#the-crash-and-the-resume)).
  - Resumed at 00:47 UTC from ecdcc08, the fix, on branch
    `phase2-step2-test`, and finished at 00:53 UTC.
- **Commands**:
  - `python -m rk live-eval --config configs/phase1.yaml --configs L1,L5,L5w --from-qid A0187 --limit 80`
  - Resume:
    `python -m rk live-eval --config configs/phase1.yaml --configs L1,L5,L5w --run-dir runs/live-20261015-000815 --from-qid A0187 --limit 80 --retry-errors`
- **Queries**: the 80 queries after the dev set in seeded order, A0187 to
  A0287. All 80 are complete in all three configs.
  - There are no errors, no cost-cap misses and no budget stop.
  - A0187 had L0 and L1 rows in the dev run, but no L5 row, so it is not
    part of the dev set.
- **L5w runs live**: it ranks L5's own check (one agent loop serves both).
  On the dev set it was replayed offline.

## Summary

- **L5w vs L5 (pre-declared, decides): not significant.**
  - **Hit@5**: L5w hits on 4 queries where L5 misses, and the reverse
    happens on 3. ΔHit@5 = +0.013 (95% CI −0.050 to +0.075), McNemar p = 1.
  - **MRR@20** agrees: +0.018 (CI −0.016 to +0.052).
  - **Against the dev set**: there, L5w gained +0.050 (7 / 1, p = 0.07). On
    the test set the gain mostly disappears.
- **L5w vs L1 (pre-declared): not significant on Hit@5.**
  - **Hit@5**: +0.087 (CI +0.000 to +0.175), 11 / 4, p = 0.118.
  - **MRR@20** supports it: +0.098, with a CI above zero (+0.020 to
    +0.176).
  - **L5 vs L1**, for reference: +0.075, p = 0.263.
- **What the test shows**:
  - It doesn't confirm that L5w beats L5.
  - L5w isn't worse either. It is 1–2 points ahead on Hit@1, Hit@5, Hit@10
    and MRR@20.
  - Its trade is clear (see [Where L5w and L5 differ](#where-l5w-and-l5-differ)):
    it wins back originals that the proposal's own search ranked high, and
    loses originals that only the agent's queries found.
- **The gap to Phase 1's B5 remains.**
  - **Hit@5**: L5w 0.700 against 0.850 on the same queries (4 / 16,
    p = 0.012).
  - **Hit@10**: 0.775 against 0.925.
- **Cost and speed are well inside the spec.**
  - **Cost**: an L5 or L5w check costs $0.0077 on average (median
    $0.0076, max $0.0097).
  - **Time**: a standalone check takes 22.9 s at the median and 32.8 s at
    p95 (upper bounds).
  - **Cost-cap misses**: 0.
- **Budget**:
  - N = 80 from a fresh $1.00.
  - OpenAlex: $0.580 (580 searches).
  - LLM: $0.0320 billed.
  - $0.42 of today's OpenAlex budget is left.
- **Product note: L1 alone is not close enough to be the default mode.** It
  trails L5w by 8.7 points of Hit@5 on the test set (0.613 against 0.700;
  not significant at n = 80, but the MRR CI excludes zero) and by 11.7 on
  the dev set. It finds 3 of 15 originals in the hardest bucket, against L5's
  9. At $0.001 and about 2 s a check, it fits an instant first pass, with
  L5w (about $0.008 and 21 s) as the default.

## Dev and test side by side

| Config | Dev Hit@1 | Dev Hit@5 | Dev Hit@10 | Dev MRR@20 | Test Hit@1 | Test Hit@5 | Test Hit@10 | Test MRR@20 |
|---|---|---|---|---|---|---|---|---|
| L1 | 0.467 | 0.633 | 0.725 | 0.546 | 0.425 | 0.613 | 0.675 | 0.516 |
| L5 | 0.483 | 0.700 | 0.758 | 0.570 | 0.525 | 0.688 | 0.762 | 0.596 |
| L5w (F1 proposal-equal) | 0.508 | 0.750 | 0.817 | 0.617 | 0.537 | 0.700 | 0.775 | 0.614 |
| P1-B5 (frozen corpus, same queries) | 0.683 | 0.867 | 0.908 | 0.760 | 0.637 | 0.850 | 0.925 | 0.730 |

- **Dev**: the 120 complete queries of `runs/live-20261014-122144`
  (A0045–A0186).
  - L1, L5 and L5w are from `fusion.md`, the offline replay, which
    reproduced all 120 recorded L5 and L1 rankings.
  - P1-B5 is Phase 1's B5 cut to the top 20, as in
    [phase2-step2-full.md](phase2-step2-full.md).
- **Test**: the 80 queries of `runs/live-20261015-000815`, with every config
  run live.
- **MRR**: ranks count within the top 20 (the list the tool shows), so MRR is
  MRR@20 everywhere.
- **Dev-set advantage**: L5w's lead over L5 on the dev set was partly
  selection. Four variants were tried there, and the best one was kept.
  - **Dev**: +5.0 points of Hit@5 and +4.7 of MRR.
  - **Test**: +1.3 and +1.8.

## Pre-declared comparisons on the test set

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR@20 [95% CI] | Only A / only B | McNemar p |
|---|---|---|---|---|---|
| **L5w vs L5** | 80 | +0.013 [−0.050, +0.075] | +0.018 [−0.016, +0.052] | 4 / 3 | 1 |
| **L5w vs L1** | 80 | +0.087 [+0.000, +0.175] | +0.098 [+0.020, +0.176] | 11 / 4 | 0.118 |
| L5 vs L1 (Step 2's primary) | 80 | +0.075 [−0.037, +0.188] | +0.080 [−0.009, +0.171] | 13 / 7 | 0.263 |

- **Neither pre-declared comparison is significant on Hit@5.** For L5w vs
  L1, the Hit@5 CI touches zero.
- **The MRR CI for L5w vs L1 is above zero.** L5w ranks the original higher
  than L1 on average, but the Hit@5 test doesn't decide it at n = 80.

The same comparisons on the dev set (exploratory, since that is where L5w was
chosen; from `fusion.md`):

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR@20 [95% CI] | Only A / only B | McNemar p |
|---|---|---|---|---|---|
| L5w vs L5 | 120 | +0.050 [+0.008, +0.100] | +0.047 [+0.018, +0.079] | 7 / 1 | 0.070 |
| L5w vs L1 | 120 | +0.117 [+0.050, +0.192] | +0.071 [+0.006, +0.137] | 17 / 3 | 0.0026 |
| L5 vs L1 | 120 | +0.067 [−0.017, +0.150] | +0.024 [−0.054, +0.103] | 17 / 9 | 0.169 |

### Secondary (exploratory): against Phase 1's B5

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR@20 [95% CI] | Only A / only B | McNemar p |
|---|---|---|---|---|---|
| L5 vs P1-B5 | 80 | −0.163 [−0.263, −0.062] | −0.134 [−0.212, −0.060] | 3 / 16 | 0.0044 |
| L5w vs P1-B5 | 80 | −0.150 [−0.263, −0.050] | −0.116 [−0.196, −0.042] | 4 / 16 | 0.012 |

The L5w vs P1-B5 row isn't in the run's `summary.md`. I computed it with the
same functions (`paired_bootstrap`, seed 42, and `mcnemar_exact`).

## Where L5w and L5 differ

The 7 queries where exactly one of them has the original in its top 5:

| Query | Hit@5 in | Phase 1 B1 bucket | Rank L1 / L5 / L5w | Search that first returned it |
|---|---|---|---|---|
| A0195 | L5w | 11-100 | – / 9 / 2 | agent query 3, keyword |
| A0197 | L5w | 1 | 2 / 10 / 5 | the proposal, semantic |
| A0212 | L5w | 2-10 | 1 / 7 / 3 | the proposal, semantic |
| A0287 | L5w | 1 | 2 / 10 / 3 | the proposal, semantic |
| A0242 | L5 | not found | – / 2 / 8 | agent query 1, keyword |
| A0270 | L5 | not found | – / 5 / 6 | agent query 2, semantic |
| A0280 | L5 | not found | – / 3 / 7 | agent query 2, semantic |

(– means not in L1's top 20.)

- **Where L5w wins**:
  - In 3 of its 4 wins, L1 had the original at rank 1 or 2. L5's fusion
    pushed it to 7–10, and L5w's extra weight on the proposal brought it
    back to 3–5. This is the dilution the dev set showed.
  - In the fourth, A0195, L5w also lifted an original that only an agent
    query found, from 9 to 2.
- **Where L5w loses**: all 3 losses are in the hardest bucket, where Phase 1's
  BM25 never found the original.
  - Only an agent query returned the original there.
  - L5 ranked it 2–5. L5w, which trusts the proposal more, ranked it 6–8.

## Hit@5 by Phase 1 difficulty (the original's rank in Phase 1's B1)

| Config | 1 (n=34) | 2-10 (n=18) | 11-100 (n=13) | not found (n=15) |
|---|---|---|---|---|
| L1 | 0.824 (28) | 0.722 (13) | 0.385 (5) | 0.200 (3) |
| L5 | 0.853 (29) | 0.667 (12) | 0.385 (5) | **0.600 (9)** |
| L5w | **0.912 (31)** | 0.722 (13) | 0.462 (6) | 0.400 (6) |
| P1-B5 | 0.941 (32) | 0.889 (16) | 0.769 (10) | 0.667 (10) |

Hits are in brackets.

- **The three easier buckets**: L5w is at or above L5 in all of them.
- **The hardest bucket**: L5 is clearly better there: 9 of 15 against L5w's
  6 and L1's 3. The agent's queries pay off where the proposal's wording
  doesn't find the original, and L5w's weighting gives up part of that.
- **Against P1-B5**: L5w still trails it in every bucket.

## Where the original came from

L5 and L5w share the same check, so the same searches.

| | Queries |
|---|---|
| Returned by the proposal's semantic search | 58 |
| ... and in L1's / L5's / L5w's top 20 | 57 / 52 / 55 |
| Not returned by the proposal's search, but by an agent query (4 keyword, 10 semantic) | 14 |
| ... and in L5's / L5w's top 20 | 13 / 13 |
| Never returned by any search | 8 |

- **Agent queries matter more here than on the dev set**: they found the
  original for 14 of 80 test queries (17.5%), against 12 of 120 on the dev
  set (10%).
- **Never returned**: 8 of 80 (10%), against 16 of 120 (13%) on the dev set.
- **Matching**: every match was by OpenAlex id.

## Cost and latency

Per check (mean / p50 / p95). Method cost counts every call, cached or not.

| Config | Queries | Searches | LLM calls | OpenAlex cost ($) | LLM cost ($) | Method cost ($) | Latency (s) |
|---|---|---|---|---|---|---|---|
| L1 | 1 | 1 | 0 | 0.00100 | 0 | 0.00100 | 2.37 / 1.98 / 3.35 |
| L5 | 4.1 / 4 / 5 | 7.2 / 7 / 9 | 3.7 / 4 / 4 | 0.00725 / 0.00700 / 0.00900 | 0.00050 / 0.00052 / 0.00061 | 0.00775 / 0.00759 / 0.00955 | 20.9 / 21.0 / 30.3 |
| L5w | (L5's check) | | | | | 0.00775 / 0.00759 / 0.00955 | 21.2 / 21.3 / 30.5 |

- **Latency**: this run has no L0. So L1 sends the proposal's search itself,
  and its latency includes it, while L5 reads that search from the cache.
  - **Standalone L5 check**: L5's latency plus L1's is an upper bound,
    since it also counts L1's local rerank. It is 22.9 s at the median,
    32.8 s at p95 and 64.5 s at most. The spec asks for a median under 60 s.
  - **The slowest check**, A0287: OpenAlex took 15.9 s on the proposal's
    search, and the check took 48.6 s.
  - **L5w** adds about 0.3 s to L5: it re-ranks the same pool and makes no
    calls.
- **Cost**: an L5 or L5w check costs $0.0076 at the median and $0.0097 at
  most. The spec asks for a median under $0.03 and a $0.10 hard cap.
  - **Cost-cap misses**: 0 in every config.
- **The agent loop**:
  - The agent stopped on its own in 46 checks and used all 5 queries in 34.
    That's more often than on the dev set (36 of 120).
  - Pools held 280.5 works at the median (max 408), so none reached the
    500 cap.

## Budget used

- **Planning N** (with `budget_test.py`, the free `/rate-limit` lookup, at
  00:07 UTC):
  - $1.0000 of $1 was left.
  - N = min(80, floor((1.0000 − 0.05) / 0.0067)) = 80.
- **OpenAlex**: $0.580 used, from $1.0000 to $0.5020 at the crash and
  $0.4200 after the resume. That is 580 searches:
  - 80 proposal searches (L1);
  - 496 agent searches (L5);
  - 4 agent searches from A0272's crashed attempt. The retry read them from
    the cache, so no row counts them.
- **Per query**: $0.0072 (576 searches / 80), above the $0.0067 estimate.
  N = 80 still fit, with $0.42 to spare.
- **The run's "Money spent" table**: it shows L5w with the same 496
  searches and $0.0317 as L5. L5w shares L5's check, so these are the same
  calls, not extra spend.
- **LLM**: $0.0320 billed for 293 agent calls (qwen3-235b on OpenRouter).
  - The L5 rows record $0.0317. The other $0.0003 is the 3 agent calls of
    A0272's crashed attempt, which the retry read from the cache.
  - The cumulative ledger is now $5.4874 of $12.

## The crash and the resume

- **What happened**:
  - On A0272, the agent's third query was a keyword search for "synthetic
    Middle English generation neural network fine-tuning stylistic
    accuracy". It got OpenAlex's HTTP 504 `query_timeout` on all 6 attempts.
    OpenAlex says such requests aren't charged.
  - L5 correctly wrote an error row.
  - L5w then crashed with `KeyError: 'result'`.
- **The bug**:
  - The live check that L5 and L5w share remembered a successful check or a
    cost-cap miss, but not an OpenAlex or LLM error.
  - The dev set never hit this, because there L5w was replayed offline.
  - The 69 queries finished before the crash were kept, with their rows and
    summary.
- **The fix** (ecdcc08, `fix: L5w shares a failed L5 check as an error row
  instead of crashing`):
  - Any failure of the shared check is stored and raised again for L5w, so
    both configs get the same error row and the run goes on.
  - A regression test makes an agent search time out on every attempt. It
    failed on the old code with the same `KeyError`.
- **The resume** (`--run-dir … --retry-errors`, as agreed): A0272's L5
  check was retried and succeeded this time, so no error row is left.
  - The agent's first 3 LLM steps came from the cache, and so did the
    searches of its first 2 queries.
  - The third query's search went through this time, and the agent then ran
    a fourth query. L5 ranked the original 11th, and L5w missed it in the
    top 5 too.
  - The other 10 queries ran normally.
- **Strict behaviour kept for evaluation**: a failed agent search is an
  error row in both L5 and L5w, left out of paired comparisons and reported.
  In this run it ended up affecting no row.

## Deviations from the plan

1. **The run was split in two**: a crash at query 70 on a live-eval bug,
   then a resume from a fixed commit. `run.json` records both commits
   (`git_commit` eb6dc36, `resumed[0].git_commit` ecdcc08).
   - The fix changes only how a failed check is passed on to L5w.
   - It doesn't change any result of a check that succeeded, so the 69
     earlier queries are comparable.
2. **One extra commit**: the fix (ecdcc08) is on this branch, before the
   report, as agreed.
3. **An extra table**: L5w vs P1-B5 isn't in the run's summary. I computed
   it with the same functions and labelled it exploratory.

## Open questions

1. **L5w or L5 for the tool?** The test doesn't show L5w beating L5 (p = 1),
   nor losing to it.
   - **For L5w**: it is ahead on every measure, and it was the pre-declared
     choice.
   - **For L5**: it is clearly better in the hardest bucket (9 / 15 against
     6 / 15).
   - **Step 3's competitive evaluation**: it uses L5w as "ours", as set up
     before. Keep that, or switch to L5?
2. **The fusion trade-off**: L5w's wins and losses are mirror images
   (dilution against agent-only finds). A fusion that keeps the proposal's
   weight but doesn't demote originals that only an agent query found would
   be a new pre-declared variant, with a new test set. Is that worth a later
   step?
