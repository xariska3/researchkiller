# Phase 2 Step 2: live evaluation, full run (120 queries from A0045)

This is the full run of `python -m rk live-eval` under the plan pre-declared in
[docs/phase2_spec.md](../phase2_spec.md) (Evaluation, Step 2 decisions). The
30-query pilot is reported separately in
[phase2-step2-pilot.md](phase2-step2-pilot.md) and summarised again at the end.

- **Run**: `runs/live-20261014-122144`, from `main` at e8051f0 (PR #9 merged,
  clean tree), on 2026-10-14 from 12:21 to 12:58 UTC. Every row is in
  `data/results.sqlite` (table `live_query_results`) and in MLflow (experiment
  `researchkiller-phase2-live`).
- **Command**: `python -m rk live-eval --config configs/phase1.yaml --from-qid A0045 --limit 122`.
- **Queries**: the first 122 in seeded order from A0045 onward, so none of
  Phase 1's pilot or tuning targets. OpenAlex's daily budget ran out after
  120: queries A0045–A0186 are complete in all three configs. The 121st,
  A0187, has L0 and L1 rows and no L5 row. The 122nd was not started.
  Everything below uses the **120 complete queries**, unless a row says
  otherwise.

## Summary

- **Primary comparison, L5 vs L1: not significant.** L5 gains on 17 queries
  and loses on 9, for ΔHit@5 = +0.067 (95% CI −0.017 to +0.150, McNemar
  p = 0.169). MRR, the supporting measure, is flat: +0.024 (CI −0.054 to
  +0.103). The agent loop adds little over one OpenAlex call plus the local
  rerank.
- **The local rerank is the big step.** Reranking OpenAlex's 50 results (L1)
  lifts Hit@5 from 0.225 (L0, OpenAlex's own order) to 0.633: 51 queries
  gained, 1 lost.
- **Live recall is well below Phase 1's B5 on the frozen corpus.**
  - **Hit@10**: 0.758 for L5 against B5's 0.908 on the same queries, short of
    the spec's live-recall bar.
  - **Hit@5**: 0.700 against 0.867. B5 hits where L5 doesn't on 21 queries,
    and the reverse on 1 (p = 1e-5).
  - **Why**: of those 21, 15 are ranking problems inside the much larger live
    pool, 4 originals were never returned by OpenAlex, and 2 were excluded by
    the type filter.
- **Cost and speed are well inside the spec.**
  - **Cost**: an L5 check costs $0.0071 on average (median $0.0075, p95
    $0.0096, cap $0.10).
  - **Time**: a standalone check takes about 18.6 s at the median (p95
    23.4 s).
  - **Cost-cap misses**: 0 in all configs.
- **Budget**: 122 queries were planned from today's $0.7999. They needed
  $0.0066 each, not the estimated $0.0061, so the run hit the real budget 429
  after 120, and stopped cleanly as designed. That 429 is shown below.

## Retrieval quality (120 queries)

| Config | Queries | Errors | Cost-cap misses | Hit@1 | Hit@5 | Hit@10 | MRR@20 |
|---|---|---|---|---|---|---|---|
| L0: OpenAlex semantic, its order | 120 | 0 | 0 | 0.042 | 0.225 | 0.425 | 0.135 |
| L1: L0 reranked locally | 120 | 0 | 0 | 0.467 | 0.633 | 0.725 | 0.546 |
| L5: full live pipeline | 120 | 0 | 0 | 0.483 | 0.700 | 0.758 | 0.570 |
| P1-B5: Phase 1 B5, frozen corpus | 120 | 0 | 0 | 0.683 | 0.867 | 0.908 | 0.760 |

- **MRR@20**: ranks count within the top 20 (the list the tool shows), so MRR
  is MRR@20, with P1-B5 cut the same way.
- **Cost-cap misses**: a check over its cost cap would count as a miss here
  (pre-declared); there were none.
- **summary.md** also counts A0187 for L0 and L1 (121 queries): L0 Hit@5
  0.223, L1 0.636.

## Primary comparison: L5 vs L1 (pre-declared)

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR@20 [95% CI] | Only L5 / only L1 | McNemar p |
|---|---|---|---|---|---|
| L5 vs L1 | 120 | +0.067 [−0.017, +0.150] | +0.024 [−0.054, +0.103] | 17 / 9 | 0.169 |

Hit@5 with the exact McNemar test decides, and it is not significant at
n = 120. MRR doesn't move either.

- **Where L5 wins (17 queries)**:
  - In 5, only the agent's queries found the original (L1 never saw it), and
    L5 ranked it in the top 5: A0075, A0076, A0081, A0120 and A0145.
  - In 12, L1 had it at ranks 7–17 and the fused ranking lifted it into the
    top 5. Example: A0050 went from 9 to 1.
- **Where L5 loses (9 queries)**:
  - L1 had the original in its top 5, at rank 1 in 4 of these queries.
  - L5's final fusion (RRF over every query's ranking of the pool) pushed it
    down: to ranks 7–14 in 6 queries, and out of the top 20 in 3 (A0073,
    A0082, A0128).
  - With 4 or 5 queries, the proposal's own ranking is one vote of five. The
    agent's queries rank new candidates above the original.

### Secondary comparisons (exploratory)

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR@20 [95% CI] | Only A / only B | McNemar p |
|---|---|---|---|---|---|
| L1 vs L0 | 121 | +0.413 [+0.322, +0.504] | +0.411 [+0.343, +0.480] | 51 / 1 | 2.4e-14 |
| L5 vs L0 | 120 | +0.475 [+0.383, +0.567] | +0.435 [+0.362, +0.509] | 58 / 1 | 2.1e-16 |
| L5 vs P1-B5 | 120 | −0.167 [−0.234, −0.100] | −0.190 [−0.250, −0.133] | 1 / 21 | 1.1e-5 |

(L1 vs L0 includes A0187, where both have rows.)

## Hit@5 by Phase 1 difficulty (the original's rank in Phase 1's B1)

| Config | 1 (n=48) | 2-10 (n=30) | 11-100 (n=24) | not found (n=18) |
|---|---|---|---|---|
| L0 | 0.396 | 0.167 | 0.125 | 0.000 |
| L1 | 0.854 | 0.667 | 0.542 | 0.111 |
| L5 | 0.875 | 0.700 | 0.542 | 0.444 |
| P1-B5 | 1.000 | 0.867 | 0.750 | 0.667 |

- **Where the agent pays off**: the hardest bucket, where Phase 1's BM25
  never found the original. There, L5 gets 8 of 18 and L1 gets 2 of 18, the
  same pattern as the pilot.
- **Elsewhere**: L5 and L1 are close.
- **Against P1-B5**: L5 trails it in every bucket.

## Where the original came from

| | Queries |
|---|---|
| Returned by the proposal's semantic search (first search of every check) | 92 |
| ... and in L0's top 20 / L1's top 20 / L5's top 20 | 72 / 92 / 87 |
| Not returned by the proposal's search, but by an agent query (7 keyword, 5 semantic) | 12 |
| ... and in L5's top 20 | 10 |
| Never returned by any search | 16 |

- **OpenAlex's ranking of the original**: in the proposal's 50 results it
  ranked the original at a median of 9.5. Ranks 1–5: 27 queries. Ranks 6–20:
  45. Ranks 21–50: 20.
  - L1's rerank brings all 20 of those below rank 20 into its top 20.
  - L5 brings 17 of them.
- **The 16 never returned**, checked with free singleton lookups of all 121
  originals. All are still in OpenAlex under the same id, with an abstract.
  - **13** have an allowed type (articles, conference papers, one
    dissertation). Neither the proposal's semantic search nor the agent's 2–4
    keyword and semantic queries returned them in their top 50. These are
    pure retrieval misses.
  - **3 can't be returned at all**, because the type filter
    (`live.work_types`) drops their OpenAlex type:
    - A0051, "PentaCMD: 299,329 English-to-terminal-command pairs …", typed
      `dataset`. Phase 1's B5 had it at rank 1.
    - A0105, "Computational Linguistics and Natural Language Processing for
      Languages of Africa", typed `reference-entry`. Rank 9.
    - A0116, "ExSMuV: [Ex]ploration software for [S]ummarized [Mu]ltimedia
      [V]ertical search results", typed `software-paper`. Rank 2.

  Phase 1's corpus had no type filter, so these were valid benchmark targets.
- **Matching**: every match but one was by OpenAlex id; one L1 match was by
  exact normalised title.

## Why L5 trails Phase 1's B5

B5 had Hit@5 and L5 didn't on 21 queries:

| Reason | Queries |
|---|---|
| L5 returned the original but ranked it 6–20 | 11 |
| L5 returned it, but it fell out of the final top 20 | 4 |
| No search returned it | 4 |
| The type filter excludes it | 2 |

- **Mostly ranking, not retrieval.** In 15 of the 21, the original was in
  L5's pool and ranked too low.
  - **A bigger haystack**: the frozen corpus held 50,130 works: 37,971 from
    four topics since 2025, 11,962 of their references and 197 citers. The
    live pool is drawn from all of OpenAlex, so older papers on the same idea
    compete.
  - **Fusion dilution**: in 11 of these 15, L1 ranked the original higher
    than L5 did.
- **Some competitors may be legitimate.** A closer older paper is legitimate
  prior art, but this benchmark only credits the original. So part of the gap
  may be the metric rather than the tool. We haven't measured how much.

## Cost and latency

Per check (mean / p50 / p95). Method cost counts every call, cached or not.

| Config | Queries | Searches | LLM calls | OpenAlex cost ($) | LLM cost ($) | Method cost ($) | Latency (s) |
|---|---|---|---|---|---|---|---|
| L0 | 1 | 1 | 0 | 0.0010 | 0 | 0.0010 | 1.45 / 1.33 / 2.22 |
| L1 | 1 | 1 | 0 | 0.0010 | 0 | 0.0010 | 0.58 / 0.58 / 0.71 |
| L5 | 3.8 / 4 / 5 | 6.6 / 7 / 9 | 3.5 / 4 / 4 | 0.0066 / 0.0070 / 0.0090 | 0.0005 / 0.0005 / 0.0006 | 0.0071 / 0.0075 / 0.0096 | 16.0 / 17.1 / 22.4 |

- **Standalone latency**: L1's and L5's latencies leave out the proposal
  search they read from L0's cache. A standalone L5 check (its latency plus
  L0's) takes 18.6 s at the median, 23.4 s at p95 and 27.3 s at most. The
  spec asks for a median under 60 s.
- **Cost**: L5 costs $0.0075 at the median and $0.0096 at most, about 93%
  OpenAlex. The spec asks for a median under $0.03 and a $0.10 hard cap.
- **The agent loop**:
  - The agent stopped on its own in 84 checks and used all 5 queries in 36.
    That's more than in the pilot (3.8 queries on average, against 3.3).
  - Its queries had a median of 10 words.
  - One of 337 keyword searches came back empty.
  - Pools held 250 works at the median (max 384), so none hit the 500 cap.
- **Cost-cap misses**: 0.

## Budget used

- **Planning N** (point 1 of your instructions):
  - `/rate-limit` before the run reported $0.7999 left of the $1 daily
    budget.
  - N = floor((0.7999 − 0.05) / 0.0061) = 122.
- **Spent**:
  - OpenAlex: 795 searches, $0.7950 ($0.0066 per query). That was one L0
    call plus L5's agent searches. L1 and L5's first search came from the
    cache.
  - LLM: $0.0487 billed. The cumulative ledger is now $5.4554 of $12.
- **The estimate was low.** $0.0061 was the pilot's L5 method cost. The money
  a query actually spends on OpenAlex is L0's call plus L5's agent searches,
  and this run's agent issued more queries than the pilot's. So the $0.05
  margin was used up, and the run stopped 2 queries short of 122. For the
  next part, $0.0067 per query is a better estimate.

### The real budget-stop 429

The run stopped at 12:57:58 UTC, on A0187's L5 check (its first agent
search):

```
rk live-eval: stopped: OpenAlex's daily budget is spent: 9 credits left, and a call costs 10.
120 of 122 queries done; the unfinished query has no row. The budget resets in 11h 02m
(midnight UTC). Summary so far in runs\live-20261014-122144\summary.md. Resume with:
  python -m rk live-eval --config configs\phase1.yaml --run-dir runs\live-20261014-122144 --from-qid A0045 --limit 122
```

- **How the run handled it**: no error row was written, the stop is recorded
  in `run.json` and `summary.md`, and the exit code was 0.
- **The full response**: to record it, I sent one more search afterwards. It
  was refused at no cost (`credits_used` stayed at 9,991):

```
HTTP 429, content-type: application/json
retry-after: 39569
x-ratelimit-cost-required-usd: 0.001
x-ratelimit-credits-required: 10
x-ratelimit-limit: 10000
x-ratelimit-limit-usd: 1
x-ratelimit-onetime-remaining: 0
x-ratelimit-prepaid-remaining-usd: 0
x-ratelimit-remaining: 9
x-ratelimit-remaining-usd: 0.0009
x-ratelimit-reset: 39569
{"error":"Rate limit exceeded","message":"Insufficient budget. This request costs $0.001
but you only have $0.0009 remaining. Resets at midnight UTC. Need more? Add funds at
https://openalex.org/pricing","retryAfter":39569,"costUsd":0.001,"dailyRemainingUsd":0.0009,
"prepaidRemainingUsd":0,"creditsRequired":10,"creditsRemaining":9,"onetimeCreditsRemaining":0}
```

- **It matches the rule built in Step 2**: `X-RateLimit-Remaining` (9) is
  below the call's cost (10 credits), so the 429 raised
  `OpenAlexBudgetExhausted` at once, without retries.
- **The 429 also carries more direct signals**, which the docs didn't
  mention: `X-RateLimit-Credits-Required`, `Retry-After` until midnight UTC,
  and the message "Insufficient budget".

### Continuing toward 200

- **What's left**: 80 queries, A0187's L5 check plus 79 new ones. At $0.0067
  each that's about $0.54, so it fits in one day's budget after the reset.
- **The command**:
  `python -m rk live-eval --config configs/phase1.yaml --run-dir runs/live-20261014-122144 --from-qid A0045 --limit 200`.
  The printed resume hint repeats `--limit 122`; pass 200 to go on.

## The 30-query pilot, shown separately

These are queries A0000–A0038 (Phase 1's tuning targets) from run
`runs/live-20261014-114302`. I regenerated its summary with the pre-declared
framing; the numbers are unchanged.

| Config | Hit@1 | Hit@5 | Hit@10 | MRR@20 |
|---|---|---|---|---|
| L0 | 0.100 | 0.200 | 0.467 | 0.177 |
| L1 | 0.367 | 0.700 | 0.767 | 0.491 |
| L5 | 0.567 | 0.833 | 0.900 | 0.661 |
| P1-B5 | 0.633 | 0.833 | 0.900 | 0.705 |

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR@20 [95% CI] | Only A / only B | McNemar p |
|---|---|---|---|---|---|
| L5 vs L1 (primary) | 30 | +0.133 [−0.033, +0.300] | +0.171 [+0.068, +0.284] | 6 / 2 | 0.289 |
| L1 vs L0 | 30 | +0.500 [+0.333, +0.667] | +0.313 [+0.180, +0.450] | 15 / 0 | 6.1e-5 |
| L5 vs L0 | 30 | +0.633 [+0.467, +0.800] | +0.484 [+0.342, +0.627] | 19 / 0 | 3.8e-6 |
| L5 vs P1-B5 | 30 | 0.000 [−0.133, +0.133] | −0.044 [−0.153, +0.064] | 2 / 2 | 1 |

The full run is harder for every config than the pilot. The pilot's match
with Phase 1's B5 doesn't hold on the 120 queries from A0045: the gap to B5 is
−0.167 Hit@5 there. The tuning targets were not a representative sample.

## Deviations from the plan

1. **N reached**: 120 queries were completed of the 122 computed. The
   per-query estimate ($0.0061) was below the actual $0.0066, so the budget
   ran out 2 queries early. The run stopped cleanly on the real 429; the
   tables use the 120 complete queries.
2. **Report branch**: the run was done from `main` (e8051f0) as instructed.
   This report is committed on a new branch, `phase2-step2-full-report`, not
   pushed.
3. **One extra OpenAlex call**: after the run, I sent one deliberate request
   to record the full 429. It was refused at no cost. I also made free
   singleton lookups of all 121 originals.

## Open questions

1. **Continue toward 200**: shall I resume tomorrow (after 00:00 UTC) with
   `--limit 200`? It needs about $0.54 of OpenAlex budget and about $0.03 of
   LLM spend.
2. **Work types**: 3 of 120 originals (2.5%) are typed `dataset`,
   `reference-entry` and `software-paper`, which the whitelist drops, so the
   live tool can't find them. Should `software-paper` (and perhaps `dataset`,
   for dataset papers) join `live.work_types`? Or should the benchmark leave
   such targets out of live recall?
3. **Fusion diluting the proposal**: L5 lost 9 queries to L1 because the
   final RRF treats the proposal's ranking as one vote among 4–5. A possible
   fix for a later step is to weight the proposal's run more, or keep L1's top
   hits. That's a new pre-declared config, not a change to L5 now. Do you want
   it evaluated?
4. **Live recall vs Phase 1**: L5's Hit@10 (0.758) is below B5's 0.908. Part
   of the gap may be legitimate older prior art outranking the original. Do
   you want a small hand check: for a sample of L5's misses, is the top
   result a genuinely close earlier paper?
