# Phase 2 Step 2 pilot: live evaluation on 30 queries

This step adds `python -m rk live-eval`, which runs three pre-declared live
configurations on the Phase 1 benchmark queries, and runs it on the first 30
queries. The design is in [docs/phase2_spec.md](../phase2_spec.md) (Evaluation).

- **Branch**: `phase2-step2-live-eval`, from `main` at 0bae675 (PR #8 merged).
- **Run**: `runs/live-20261014-114302`, at commit c5bbe17 (clean tree), on
  2026-10-14 from 11:43 to 11:50 UTC. Every row is also in
  `data/results.sqlite` (table `live_query_results`) and in MLflow
  (experiment `researchkiller-phase2-live`).
- **Command**: `python -m rk live-eval --config configs/phase1.yaml --limit 30`.

## Summary

- **On these 30 queries, the full live pipeline (L5) matches Phase 1's B5 on
  the frozen corpus**: Hit@5 0.833 and Hit@10 0.900 for both. L5 searches all
  of OpenAlex; B5 searched a frozen corpus of four topics.
- **OpenAlex's own ranking is the weak part.** Its semantic search returned the
  original among its 50 results for 26 of 30 proposals, but usually not near
  the top (median rank 10). L0 (OpenAlex's order) gets Hit@5 0.200.
- **The local rerank does most of the work.** Reranking the same 50 results
  (L1) lifts Hit@5 to 0.700 (+0.500, McNemar p = 6e-5). The agent loop (L5)
  adds +0.133 Hit@5 over L1 (6 queries gained, 2 lost; p = 0.29, not
  significant at n = 30) and +0.171 MRR (CI +0.068 to +0.284).
- **Cost and speed are well inside the spec**: L5 costs $0.0061 per check on
  average (p95 $0.0096; cap $0.10), about 93% of it OpenAlex. It takes 12.8 s
  on average (p95 21.6 s), plus about 1.5 s for the proposal search it read
  from L0's cache. The spec's targets are a median under $0.03 and under 60 s.
- **Budget stops** are built and tested with a fake OpenAlex, but a real spent
  budget hasn't been seen yet: the pilot used $0.17 of the $1 daily budget.

These 30 queries are the first 30 qids (A0000–A0038). They are Phase 1's
pilot targets, on which the bench design was tuned, so they are not a random
sample.

## What was built

- **`python -m rk live-eval --config ... [--limit N] [--configs L0,L1,L5]
  [--run-dir DIR] [--retry-errors]`** (`src/rk/live/evaluate.py`):
  - **L0**: OpenAlex semantic search on the proposal, in OpenAlex's order (one
    call, no LLM). The cached responses confirm OpenAlex sorts these results by
    `relevance_score` (21 of 21 checked).
  - **L1**: the same call (read from the cache) with its results merged into a
    candidate pool and reranked by the pool's hybrid ranker (BM25 +
    Qwen3-Embedding, RRF).
  - **L5**: the full live pipeline from Step 1: the B5 agent loop on live search,
    the pool and the final RRF rerank, top 20.
- **Order**: queries run one at a time, each through L0, L1 and L5, so L1 and
  L5 reuse L0's proposal search, and a budget stop leaves whole queries.
- **Scoring**: a result is the original if one of its merged versions matches
  it (`match_original`: OpenAlex id, alias, DOI or exact normalised title).
  Ranks count within the top `live.top_k` (20), the list the tool shows, so
  MRR is MRR@20 everywhere.
- **Each row records**:
  - Hit@1/5/10, MRR, the rank, and how the original matched;
  - the search that first returned the original (its query, 0 for the
    proposal, and its kind) or that none did;
  - OpenAlex and LLM cost, apart, both as method cost and as money spent;
  - latency, pool size, queries and searches, and the full trace.
- **Phase 1 comparison**: `live.phase1_run` (`20261013-185712`, the full run)
  gives Phase 1's B5 (shown as P1-B5) on the same queries. Its ranks are cut at
  20 like the live ones, and the run must have used the same benchmark. The
  same run's B1 gives the difficulty buckets.
- **Statistics**: L5 vs L0 and L5 vs L1 are paired as in Phase 1 (seeded
  bootstrap CIs, exact McNemar on Hit@5). L1 vs L0 and L5 vs P1-B5 are
  exploratory.
- **Outputs**:
  - rows in `live_query_results`; the run itself is a row of Phase 1's `runs`
    table;
  - `per_query/*.jsonl`, `summary.md` and `summary.json` in the run directory;
  - MLflow, like Phase 1: one parent run, one child per config.

  `python -m rk report runs/live-...` rebuilds a live run's summary.
- **OpenAlex cache**: read as an evaluation (`evaluation=True`), so a response
  never expires and a resumed or repeated run sees the same results.

### OpenAlex's daily budget

OpenAlex answers both a spent daily budget and a per-second overrun with HTTP
429, and the docs give no way to tell them apart in the body. Every response
does carry `X-RateLimit-Remaining` (credits left today; 1 credit = $0.0001) and
`X-RateLimit-Reset` (seconds to midnight UTC), and the `/rate-limit` endpoint
is free. Probes on 2026-10-14 confirmed the headers, the endpoint's fields and
that it costs 0 credits.

- **Detecting it**:
  - A 429 whose `X-RateLimit-Remaining` can't pay for the call raises
    `OpenAlexBudgetExhausted` at once (in `rk.corpus.OpenAlexClient`, so the
    corpus build gets it too). Any other 429 is retried as before.
  - If an OpenAlex call fails some other way, `live-eval` asks `/rate-limit`.
    When no search fits today, that also counts as a budget stop. This covers
    a budget 429 that comes without the header.
- **What a budget stop does**:
  - The run stops; the unfinished query gets no row, so it is never an error
    row.
  - The stop is recorded in `run.json` and shown in `summary.md`.
  - The summary so far is written, and MLflow is updated.
  - It prints the reset time and the command to resume:
    `python -m rk live-eval --config configs/phase1.yaml --run-dir runs/live-...`.
    The resume skips finished rows and redoes the unfinished query, whose
    earlier searches come from the cache.
- **Tested with the fake OpenAlex**:
  - a stop in the middle of a query;
  - a resume the next day;
  - a 429 without headers, caught by the `/rate-limit` check;
  - a per-second 429, which is still retried;
  - the CLI message.
- **At startup**, `live-eval` prints today's budget. The pilot started with
  $0.9699 left.

## Results

| Config | Queries | Errors | Hit@1 | Hit@5 | Hit@10 | MRR@20 |
|---|---|---|---|---|---|---|
| L0: OpenAlex semantic, its order | 30 | 0 | 0.100 | 0.200 | 0.467 | 0.177 |
| L1: L0 reranked locally | 30 | 0 | 0.367 | 0.700 | 0.767 | 0.491 |
| L5: full live pipeline | 30 | 0 | 0.567 | 0.833 | 0.900 | 0.661 |
| P1-B5: Phase 1 B5, frozen corpus | 30 | 0 | 0.633 | 0.833 | 0.900 | 0.705 |

P1-B5's MRR here is MRR@20. Phase 1's own reports use MRR@100.

### Paired comparisons

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR@20 [95% CI] | Only A / only B | McNemar p |
|---|---|---|---|---|---|
| L5 vs L0 | 30 | +0.633 [+0.467, +0.800] | +0.484 [+0.342, +0.627] | 19 / 0 | 3.8e-6 |
| L5 vs L1 | 30 | +0.133 [−0.033, +0.300] | +0.171 [+0.068, +0.284] | 6 / 2 | 0.289 |
| L1 vs L0 (exploratory) | 30 | +0.500 [+0.333, +0.667] | +0.313 [+0.180, +0.450] | 15 / 0 | 6.1e-5 |
| L5 vs P1-B5 (exploratory) | 30 | 0.000 [−0.133, +0.133] | −0.044 [−0.153, +0.064] | 2 / 2 | 1 |

### Hit@5 by Phase 1 difficulty (the original's rank in Phase 1's B1)

| Config | 1 (n=10) | 2-10 (n=9) | 11-100 (n=5) | not found (n=6) |
|---|---|---|---|---|
| L0 | 0.400 | 0.111 | 0.200 | 0.000 |
| L1 | 0.800 | 0.889 | 0.800 | 0.167 |
| L5 | 1.000 | 0.889 | 0.600 | 0.667 |
| P1-B5 | 1.000 | 1.000 | 0.600 | 0.500 |

The buckets hold 5–10 queries each, so these are descriptive only. The agent
matters most on the hardest bucket: the originals that Phase 1's BM25 never
found. There, L5 gets 4 of 6 and L1 gets 1 of 6.

### Which search first returned the original

| Config | Proposal, semantic | Agent query, keyword | Agent query, semantic | Never returned |
|---|---|---|---|---|
| L0 / L1 | 26 (20 / 24 in the top 20) | – | – | 4 |
| L5 | 26 (26 in the top 20) | 1 (1) | 0 | 3 |

(Proposals have 45–77 words, so they get semantic search only; the
"proposal, keyword" column is always 0 and is left out.)

- **Where OpenAlex ranked the original** in the proposal's 50 results: median
  10. Ranks 1–5: 6 queries. Ranks 6–20: 14. Ranks 21–49: 6.
  - L0 keeps OpenAlex's order, so it misses the six below rank 20.
  - L1's rerank brings 4 of those 6 into the top 20.
  - L5 brings all 6 into the top 20.
- **One original only the agent found**: A0017. Its first agent query, a
  keyword search, returned the original at rank 1, and L5 ranked it first.
  L0 and L1 never saw it.
- **Three originals were never returned** by any search: A0013, A0030 and
  A0036. Free singleton lookups show all 30 originals are in OpenAlex with an
  abstract, an allowed type (25 `article`, 5 `conference-paper`) and the same
  id as in the frozen corpus. So the type filter cost nothing here. Neither
  the proposal's semantic search nor the agent's 3–4 queries returned these
  three in their top 50. Phase 1's B5 found two of them in the frozen corpus
  (ranks 8 and 5), where far fewer competing papers exist.
- **Every match was by OpenAlex id**: no alias, DOI or title matches were
  needed on these 30.

### Where L5 and L1 differ

- **L5 better on Hit@5 (6 queries)**:
  - A0011: L1 rank 14, L5 rank 4.
  - A0017: L1 missed it, L5 rank 1 (the agent's keyword search).
  - A0019: 6 → 4.
  - A0021: L1 missed it, L5 rank 2.
  - A0022: L1 missed it, L5 rank 5.
  - A0035: 9 → 2.

  For A0021 and A0022, OpenAlex had the original at ranks 42 and 43 of the
  proposal's search. The agent's queries pulled it up in the fused ranking.
- **L1 better (2 queries)**:
  - A0018: L1 rank 3, L5 rank 8.
  - A0038: 4 → 10.

  In both, the agent's queries found other close papers, and the final RRF
  over all queries pushed the original down.

### Per query

Ranks in the top 20 ("–": not in the top 20). "First returned by" is the
search, and that search's rank of the original. The B1 bucket is Phase 1's.

| Query | B1 bucket | P1-B5 | L0 | L1 | L5 | L5: first returned by | L5 queries | L5 pool |
|---|---|---|---|---|---|---|---|---|
| A0000 | 1 | 1 | 1 | 1 | 1 | proposal semantic #1 | 5 | 277 |
| A0002 | 11-100 | 1 | – | 3 | 1 | proposal semantic #29 | 4 | 259 |
| A0004 | 2-10 | 1 | 8 | 1 | 1 | proposal semantic #8 | 4 | 296 |
| A0006 | 2-10 | 1 | 10 | 3 | 3 | proposal semantic #10 | 1 | 41 |
| A0008 | 1 | 1 | 1 | 1 | 1 | proposal semantic #1 | 1 | 45 |
| A0009 | 1 | 1 | 14 | 3 | 3 | proposal semantic #14 | 4 | 263 |
| A0010 | 1 | 1 | 13 | 5 | 1 | proposal semantic #13 | 5 | 309 |
| A0011 | not found | 3 | 20 | 14 | 4 | proposal semantic #20 | 3 | 206 |
| A0012 | 1 | 1 | 16 | 1 | 1 | proposal semantic #16 | 3 | 221 |
| A0013 | 11-100 | 8 | – | – | – | never | 5 | 219 |
| A0016 | 2-10 | 1 | 10 | 1 | 1 | proposal semantic #10 | 3 | 186 |
| A0017 | 1 | 3 | – | – | 1 | query 1 keyword #1 | 5 | 300 |
| A0018 | 11-100 | – | 10 | 3 | 8 | proposal semantic #10 | 2 | 103 |
| A0019 | 1 | 1 | – | 6 | 4 | proposal semantic #49 | 4 | 281 |
| A0020 | 2-10 | 1 | – | 3 | 1 | proposal semantic #21 | 5 | 341 |
| A0021 | not found | 3 | – | – | 2 | proposal semantic #42 | 3 | 214 |
| A0022 | not found | – | – | – | 5 | proposal semantic #43 | 5 | 319 |
| A0023 | 2-10 | 1 | 8 | 2 | 1 | proposal semantic #8 | 5 | 392 |
| A0025 | 1 | 1 | 20 | 1 | 1 | proposal semantic #20 | 2 | 128 |
| A0026 | 1 | 1 | 4 | 1 | 1 | proposal semantic #4 | 3 | 179 |
| A0027 | 2-10 | 1 | 6 | 1 | 1 | proposal semantic #6 | 2 | 121 |
| A0030 | not found | – | – | – | – | never | 4 | 240 |
| A0031 | not found | 6 | 6 | 4 | 4 | proposal semantic #6 | 1 | 42 |
| A0032 | 11-100 | 3 | 9 | 1 | 1 | proposal semantic #9 | 1 | 41 |
| A0033 | 1 | 1 | 5 | 1 | 1 | proposal semantic #5 | 1 | 41 |
| A0034 | 11-100 | 1 | 1 | 2 | 1 | proposal semantic #1 | 5 | 322 |
| A0035 | 2-10 | 1 | – | 9 | 2 | proposal semantic #22 | 2 | 103 |
| A0036 | not found | 5 | – | – | – | never | 5 | 315 |
| A0037 | 2-10 | 1 | 2 | 1 | 1 | proposal semantic #2 | 3 | 219 |
| A0038 | 2-10 | 3 | 17 | 4 | 10 | proposal semantic #17 | 4 | 246 |

(P1-B5 ranks beyond 20 show as "–": A0018 was at 85 and A0022 at 25 in Phase 1.)

## Cost and latency

Per check (mean / p50 / p95). Method cost counts every call, including those
served from the cache.

| Config | Searches | LLM calls | OpenAlex cost ($) | LLM cost ($) | Method cost ($) | Latency (s) |
|---|---|---|---|---|---|---|
| L0 | 1 / 1 / 1 | 0 | 0.0010 | 0 | 0.0010 | 1.51 / 1.36 / 2.33 |
| L1 | 1 / 1 / 1 | 0 | 0.0010 | 0 | 0.0010 | 0.59 / 0.58 / 0.63 |
| L5 | 5.7 / 6 / 9 | 3.0 / 3.5 / 4 | 0.0057 / 0.0060 / 0.0090 | 0.0004 / 0.0005 / 0.0006 | 0.0061 / 0.0065 / 0.0096 | 12.8 / 14.5 / 21.6 |

- **Latency**: L1's and L5's leave out the proposal search they read from L0's
  cache (1.4 s at the median, 2.7 s at most). A standalone L5 check would take
  about 16 s at the median, and under 30 s at worst here.
- **The agent loop**:
  - The agent stopped on its own in 21 checks and used all 5 queries in 9.
  - In 5 checks it stopped after the proposal alone, so L5 ranked like L1.
  - Its queries had 6–22 words (median 12). Keyword searches never came back
    empty: every one of the 70 returned a full page of 50.
  - Pools held 220 works at the median (max 392), so none hit the 500 cap.
- **Money spent in this run**:
  - OpenAlex: 170 searches, $0.17 of the daily $1.
  - LLM: $0.0107 billed. The cumulative ledger is now $5.4064 of $12.
- **Encoder**: Qwen3-Embedding ran on CUDA (GRID A100-40C).

## The full run (not started)

- **OpenAlex budget**:
  - On average a query sends 5.7 searches ($0.0057): one for L0, and L5's
    agent queries.
  - The other 470 queries need about $2.70 of OpenAlex budget, so about three
    days on the free key ($1 a day; $0.80 is left today).
  - The run would stop at each day's limit and resume with `--run-dir`.
- **LLM**: about $0.17.
- **Compute**: about 2 hours in total at the pilot's pace.

## Reproducibility

- OpenAlex responses are cached and read as an evaluation, so the first
  response ever fetched is always the one used.
- Agent calls are in the LLM cache.
- So a rerun or a resume replays the same searches and agent steps for free.
  It differs only if the encoder's float16 GPU arithmetic does, as in Phase 1.
- `run.json` records:
  - the commit, the config and the prompt hashes;
  - the encoder settings and the device;
  - the Phase 1 run compared with;
  - any resumes and budget stops.

## Deviations from the spec, and why

1. **Answers commit**: I committed your answers 2–4 (work types plus review,
   book and report; the spec decisions) on this branch, as 41d5609 before the
   Step 2 code, since PR #8 was already merged.
2. **MRR@20, not MRR@100**: ranks count within the top 20, the list the tool
   shows, for every config and for P1-B5. Hit@1/5/10 are unaffected.
3. **New config keys**:
   - `live.phase1_run` (`"20261013-185712"`) names the Phase 1 run to compare
     with.
   - `live.mlflow_experiment` (`researchkiller-phase2-live`) keeps live runs
     apart from Phase 1's in MLflow.
4. **Options and commands beyond the spec**:
   - `--retry-errors` works as in Phase 1's `run`.
   - `rk report` also rebuilds a live run's summary.
5. **Order of work**: queries run one at a time through L0, L1 and L5. Phase 1
   ran config by config. This way L1 and L5 reuse L0's call, and a budget stop
   leaves whole queries. The cost is that L1's and L5's latency leaves out that
   one call.
6. **Error rows**: other OpenAlex errors, LLM errors and a check over its $0.10
   cap are error rows, left out of the metrics as in Phase 1. None happened in
   the pilot.
7. **L0 order**: L0 keeps OpenAlex's raw order and doesn't merge versions; L1
   and L5 merge them in the pool.
8. **Changes to Phase 1 code**, with Phase 1 behaviour unchanged (all Phase 1
   tests pass):
   - `rk.corpus.OpenAlexClient` stops at once on a budget 429 and can read
     `/rate-limit`.
   - `rk.evaluate` takes a run-id prefix and a fingerprint for resumes, a
     table name for the JSONL export, and extra MLflow params.
9. **Candidate versions**: `live_search` candidates now keep their merged
   versions, so matching checks every version.
10. **Tests**: the fake OpenAlex moved to `tests/live_fakes.py` and gained a
    daily budget.

## Open questions

1. **The full run**: should I run all 500 now? It needs about three days of the
   free OpenAlex budget (or a prepaid top-up) and about $0.17 of LLM spend.
2. **Primary comparison for the full run**: you asked for L5 vs L0 and L5 vs
   L1. The pilot shows L0 is a weak baseline: reranking alone adds +0.50 Hit@5.
   Should L5 vs L1 be the primary comparison, with L5 vs L0 secondary?
3. **The budget-stop rule** has not been seen against a real spent budget. The
   full run will hit it on its first day, and I'll report what the 429 looked
   like.
4. **A check over its cost cap** is an error row now. For the live-recall
   criterion, should it count as a miss instead? None happened here; the
   largest check cost $0.0097.
5. **Phase 1's pilot targets**: the pilot's queries are Phase 1's tuning
   targets. In the full-run report, should I also give the robustness subset
   (A0045 onward), as Phase 1 did?
