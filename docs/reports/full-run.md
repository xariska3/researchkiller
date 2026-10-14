# Full run (Step 7)

This report covers every item in [docs/analysis_plan.md](../analysis_plan.md)
for the Phase 1 full run. The bench it ran on is described in
[full-bench.md](full-bench.md).

- **Run**: `runs/20261013-185712` (run id 20261013-185712), at commit c027718
  on `main` with a clean working tree.
- **Inputs**: bench sha256 18f33b6067f5 (500 queries); corpus sha256
  1615016d878a.
- **Command**: `python -m rk run --config configs/phase1.yaml`, all eight
  configs.
- **Timing**: it ran from 11:56 to 17:01 local time (5 h 05 min). It wasn't
  interrupted, so it was never resumed.
- **Report**: `python -m rk report runs/20261013-185712` then rewrote
  `summary.md` and `summary.json` from the results database (5.8 s, 8
  configs, 0 error rows). The tables below are copied from that `summary.md`.
- **Stop rule**: the $10 rule never triggered; cumulative spend ended at
  $5.39.

## Data

- **Queries**: all 500 accepted queries in `data/bench/test_a.jsonl`. The bench
  reached `n_targets` before the frozen targets ran out.
- **Configs**: all eight on every query, which makes 4,000 rows. The settings
  are those in `configs/phase1.yaml`:
  - `max_queries` 5 and `max_llm_calls_agent` 6; 15 and 20 for B5-U;
  - agent `qwen/qwen3-235b-a22b-2507` at temperature 0;
  - `run.seed` 42.
- **Metrics**: Hit@1, Hit@5, Hit@10 and MRR within the top 100. The target or
  any of its aliases counts.
- **Errors**: 0 in every config, so every paired statistic uses all 500
  queries (465 in the robustness subset).

## Retrieval quality

| Config | Queries | Errors | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|
| B1 | 500 | 0 | 0.376 | 0.562 | 0.620 | 0.466 |
| B2 | 500 | 0 | 0.584 | 0.782 | 0.842 | 0.678 |
| B3 | 500 | 0 | 0.568 | 0.780 | 0.834 | 0.664 |
| B4 | 500 | 0 | 0.552 | 0.832 | 0.894 | 0.672 |
| B5 | 500 | 0 | 0.638 | 0.858 | 0.902 | 0.733 |
| B5-U | 500 | 0 | 0.644 | 0.850 | 0.890 | 0.737 |
| B4-D | 500 | 0 | 0.470 | 0.714 | 0.800 | 0.581 |
| B5-D | 500 | 0 | 0.542 | 0.752 | 0.812 | 0.641 |

B4-D and B5-D are B4 and B5 with B2 (dense) as the base retriever instead of
B3 (hybrid).

## Primary comparison

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR [95% CI] | McNemar (only B5 / only B4) | p |
|---|---|---|---|---|---|
| B5 vs B4 | 500 | +0.026 [-0.006, +0.058] | +0.061 [+0.028, +0.093] | 40 / 27 | 0.142 |

**Decision**: Hit@5 with the McNemar test decides, at α = 0.05.
- **Hit@5**: p = 0.142, so B5 is not significantly better than B4 on Hit@5.
  The difference is +2.6 points. Its CI runs from a 0.6-point loss to a
  5.8-point gain.
- **MRR (supporting)**: ΔMRR is +0.061 [+0.028, +0.093], and its CI excludes
  zero. B5 ranks the target higher than B4 does: Hit@1 is 0.638 against
  0.552. Under the decision rule, MRR doesn't change the outcome.
- **Pilot**: the 30-query pilot gave ΔHit@5 0.000 [-0.133, +0.133].

## Secondary comparisons (exploratory)

These comparisons are exploratory and have no multiple-comparison correction.
"A vs B" means A minus B. A difference is called significant when its 95% CI
excludes zero; a CI that ends exactly at zero doesn't exclude it.

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR [95% CI] | McNemar (only A / only B) | p | CI excludes 0 |
|---|---|---|---|---|---|---|
| B4 vs B3 | 500 | +0.052 [+0.016, +0.088] | +0.008 [-0.026, +0.042] | 56 / 30 | 0.00667 | Hit@5 |
| B5-U vs B5 | 500 | -0.008 [-0.032, +0.016] | +0.004 [-0.018, +0.027] | 16 / 20 | 0.618 | – |
| B2 vs B3 | 500 | +0.002 [-0.034, +0.038] | +0.014 [-0.017, +0.045] | 43 / 42 | 1 | – |
| B4-D vs B2 | 500 | -0.068 [-0.102, -0.034] | -0.097 [-0.130, -0.065] | 22 / 56 | 0.000149 | Hit@5, MRR |
| B5-D vs B2 | 500 | -0.030 [-0.066, +0.004] | -0.037 [-0.069, -0.006] | 33 / 48 | 0.119 | MRR |
| B5-D vs B4-D | 500 | +0.038 [+0.002, +0.074] | +0.060 [+0.027, +0.094] | 50 / 31 | 0.0448 | Hit@5, MRR |
| B1 vs B2 | 500 | -0.220 [-0.266, -0.174] | -0.212 [-0.254, -0.171] | 27 / 137 | 6.39e-19 | Hit@5, MRR |
| B4 vs B2 | 500 | +0.050 [+0.018, +0.084] | -0.006 [-0.038, +0.027] | 49 / 24 | 0.00463 | Hit@5 |
| B5 vs B2 | 500 | +0.076 [+0.040, +0.112] | +0.055 [+0.022, +0.088] | 62 / 24 | 5.08e-05 | Hit@5, MRR |
| B5-U vs B2 | 500 | +0.068 [+0.032, +0.104] | +0.059 [+0.025, +0.094] | 61 / 27 | 0.000374 | Hit@5, MRR |

The plan's questions:
- **B4 vs B3: does one-shot rewriting help?** It helps on Hit@5 (+0.052), but
  not on MRR (+0.008, not significant). Rewriting brings the target into the
  top 5 and top 10 more often: Hit@10 rises from 0.834 to 0.894. It places it
  first slightly less often: Hit@1 falls from 0.568 to 0.552.
- **B5-U vs B5: what does a larger budget buy?** Nothing measurable: ΔHit@5
  is -0.008 and ΔMRR +0.004, both not significant. B5-U makes twice the LLM
  calls (7.0 against 3.5 per query) at 3.1 times the method cost.
- **B2 vs B3: dense alone against the hybrid.** No difference: ΔHit@5 is
  +0.002 and ΔMRR +0.014, both not significant.
- **B4 and B5 on a dense base**:
  - B4-D is significantly worse than its own base B2, by -0.068 on Hit@5 and
    -0.097 on MRR.
  - B5-D recovers part of that loss: it beats B4-D by +0.038 on Hit@5 (CI
    [+0.002, +0.074], McNemar p 0.045) and +0.060 on MRR.
  - B5-D still stays below B2: -0.030 on Hit@5 (not significant) and -0.037
    on MRR (significant).
- **Every config vs B2**:
  - B1 is far below B2 (-0.220 on Hit@5).
  - B4 is above B2 on Hit@5 only (+0.050).
  - B5 and B5-U are above B2 on both metrics: B5 by +0.076 on Hit@5 and
    +0.055 on MRR; B5-U by +0.068 and +0.059.
  - B3 vs B2 is the B2 vs B3 row above with the sign flipped: no difference.

## Difficulty

A query's difficulty is the rank of its target in B1. This table is
descriptive only, and no query was filtered on it.

| Config | 1 (n=188) | 2-10 (n=122) | 11-100 (n=100) | not found (n=90) |
|---|---|---|---|---|
| B1 | 1.000 | 0.762 | 0.000 | 0.000 |
| B2 | 0.920 | 0.811 | 0.750 | 0.489 |
| B3 | 0.989 | 0.951 | 0.790 | 0.100 |
| B4 | 0.941 | 0.861 | 0.860 | 0.533 |
| B5 | 0.973 | 0.885 | 0.820 | 0.622 |
| B5-U | 0.931 | 0.852 | 0.840 | 0.689 |
| B4-D | 0.819 | 0.721 | 0.690 | 0.511 |
| B5-D | 0.846 | 0.746 | 0.710 | 0.611 |

- **Easy queries**: where BM25 already ranks the target first or in the top
  10, B3 (hybrid) is the strongest config. It scores 0.989 on rank-1 queries,
  behind only B1 itself (1.000), and 0.951 on ranks 2–10, the best of all.
- **Queries BM25 misses entirely**: B3 falls to 0.100 there, while B2 (dense)
  reaches 0.489. B5-U is the best on these queries (0.689), then B5 (0.622)
  and B5-D (0.611).

## Robustness: without the pilot targets (exploratory)

The tables below repeat the analysis on the queries from qid A0045 onward,
leaving out the 35 pilot-target queries that the bench design was tuned on.

| Config | Queries | Errors | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|
| B1 | 465 | 0 | 0.378 | 0.561 | 0.619 | 0.468 |
| B2 | 465 | 0 | 0.585 | 0.778 | 0.839 | 0.677 |
| B3 | 465 | 0 | 0.574 | 0.783 | 0.839 | 0.668 |
| B4 | 465 | 0 | 0.555 | 0.830 | 0.895 | 0.674 |
| B5 | 465 | 0 | 0.641 | 0.858 | 0.901 | 0.735 |
| B5-U | 465 | 0 | 0.643 | 0.843 | 0.886 | 0.734 |
| B4-D | 465 | 0 | 0.467 | 0.712 | 0.798 | 0.578 |
| B5-D | 465 | 0 | 0.535 | 0.748 | 0.811 | 0.636 |

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR [95% CI] | McNemar (only B5 / only B4) | p |
|---|---|---|---|---|---|
| B5 vs B4 | 465 | +0.028 [-0.006, +0.060] | +0.061 [+0.028, +0.095] | 38 / 25 | 0.13 |

The pilot targets don't drive the results:
- No config's Hit@5 moves by more than 0.007 between the two tables. The
  largest change is B5-U, from 0.850 to 0.843.
- The primary comparison reads the same as on all 500 queries: Hit@5 is not
  significant, and the MRR CI excludes zero.

## Exhaustion check (exploratory)

**Groups**:
- 500 accepted and 86 exhausted targets (all 4 attempts rejected);
- left out: 33 targets skipped as not research contributions, and 0
  unfinished ones.

Differences are exhausted minus accepted, with a seeded bootstrap 95% CI
(10,000 resamples within each group) and a permutation p-value (10,000
permutations).

| Measure | Exhausted (n=86) | Accepted (n=500) | Difference [95% CI] | p |
|---|---|---|---|---|
| Abstract length (words) | 215.8 | 207.3 | +8.5 [-3.7, +21.7] | 0.179 |
| Share of sentences with a number | 0.212 | 0.178 | +0.034 [-0.007, +0.074] | 0.0949 |
| Share of sentences with a results cue | 0.280 | 0.271 | +0.009 [-0.024, +0.043] | 0.6 |

| Subfield | Accepted | Exhausted | Exhaustion rate |
|---|---|---|---|
| Artificial Intelligence | 442 | 76 | 0.147 |
| Information Systems | 58 | 10 | 0.147 |

The exhaustion rate of Information Systems minus Artificial Intelligence is
0.000 [-0.087, +0.093], p 1.

| Topic (descriptive only) | Accepted | Exhausted | Exhaustion rate |
|---|---|---|---|
| Topic Modeling | 317 | 61 | 0.161 |
| Natural Language Processing Techniques | 89 | 6 | 0.063 |
| Information Retrieval and Search Behavior | 58 | 10 | 0.147 |
| Speech and dialogue systems | 36 | 9 | 0.200 |

**Why the exhausted targets' 344 attempts were rejected**:
- the leak checks rejected 275 of them, the judge only 69;
- 42 of the 86 exhausted targets had at least one judge rejection;
- the facets the judge failed were: captures: method 2, domain 26, claim 26;
  adds_nothing: domain 8, claim 16.

| Rejection reason | Attempts |
|---|---|
| acronym_in_paraphrase | 40 |
| source_name | 42 |
| title_stem | 162 |
| idf_overlap | 12 |
| content_jaccard | 19 |
| judge_captures | 48 |
| judge_adds_nothing | 21 |

**Reading**, by the plan's rule: the bench would be biased if the exhausted
targets had more results or number sentences, with a CI that excludes zero.
- **Results-cue sentences**: their share doesn't differ (+0.009 [-0.024,
  +0.043]).
- **Number sentences**: their share leans the biased way (+0.034), but the CI
  includes zero ([-0.007, +0.074], p 0.095).
- **Conclusion**: the rule's condition isn't met, so the plan requires no
  limitation in the README. The bench stays frozen, as the plan says it would
  either way.
- **Subfield**: the exhaustion rate is the same in both subfields (0.147).
- **What exhausts targets**: mostly the leak checks (title_stem alone
  rejected 162 of the 344 attempts), not the judge.

## Effort per query (mean / p50 / p95)

| Config | Queries issued | LLM calls | Tokens | Method cost ($) | Latency (s) |
|---|---|---|---|---|---|
| B1 | 1.0 / 1.0 / 1.0 | 0.0 / 0.0 / 0.0 | 0 / 0 / 0 | 0.00000 / 0.00000 / 0.00000 | 0.007 / 0.007 / 0.007 |
| B2 | 1.0 / 1.0 / 1.0 | 0.0 / 0.0 / 0.0 | 0 / 0 / 0 | 0.00000 / 0.00000 / 0.00000 | 0.087 / 0.078 / 0.135 |
| B3 | 1.0 / 1.0 / 1.0 | 0.0 / 0.0 / 0.0 | 0 / 0 / 0 | 0.00000 / 0.00000 / 0.00000 | 0.093 / 0.086 / 0.137 |
| B4 | 5.0 / 5.0 / 5.0 | 1.0 / 1.0 / 1.0 | 259 / 259 / 277 | 0.00007 / 0.00007 / 0.00008 | 2.935 / 2.550 / 5.279 |
| B5 | 3.9 / 4.0 / 5.0 | 3.5 / 4.0 / 4.0 | 2922 / 3274 / 3682 | 0.00048 / 0.00053 / 0.00060 | 8.593 / 7.231 / 20.624 |
| B5-U | 7.0 / 5.0 / 14.0 | 7.0 / 5.0 / 14.0 | 9800 / 5226 / 27240 | 0.00151 / 0.00084 / 0.00405 | 16.150 / 12.451 / 39.392 |
| B4-D | 5.0 / 5.0 / 5.0 | 1.0 / 1.0 / 1.0 | 259 / 259 / 277 | 0.00007 / 0.00007 / 0.00008 | 0.471 / 0.455 / 0.624 |
| B5-D | 3.9 / 4.0 / 5.0 | 3.5 / 4.0 / 4.0 | 2908 / 3262 / 3705 | 0.00048 / 0.00053 / 0.00060 | 8.066 / 6.815 / 20.127 |

B4-D's latency is low because it uses B4's rewrite, which was already cached.
Its LLM calls and cost still count, since method cost includes cache hits.

## Cost and quality (computed cost)

| Config | Hit@5 | MRR | Mean method cost per query ($) | Mean LLM calls |
|---|---|---|---|---|
| B1 | 0.562 | 0.466 | 0.00000 | 0.0 |
| B2 | 0.782 | 0.678 | 0.00000 | 0.0 |
| B3 | 0.780 | 0.664 | 0.00000 | 0.0 |
| B4 | 0.832 | 0.672 | 0.00007 | 1.0 |
| B5 | 0.858 | 0.733 | 0.00048 | 3.5 |
| B5-U | 0.850 | 0.737 | 0.00151 | 7.0 |
| B4-D | 0.714 | 0.581 | 0.00007 | 1.0 |
| B5-D | 0.752 | 0.641 | 0.00048 | 3.5 |

## Billed cost

These figures cover only the calls this run actually sent, so cache hits cost
nothing.
- **Pilot-run cache**: the first 30 queries' agent calls were cache hits from
  the pilot run.
- **B4-D**: it sent no calls at all.

| Config | Calls sent, computed ($) | Billed ($) | Billed vs computed |
|---|---|---|---|
| B1 | 0.00000 | 0.00000 | – |
| B2 | 0.00000 | 0.00000 | – |
| B3 | 0.00000 | 0.00000 | – |
| B4 | 0.03439 | 0.02969 | -14% |
| B5 | 0.22508 | 0.19078 | -15% |
| B5-U | 0.71246 | 0.51313 | -28% |
| B4-D | 0.00000 | 0.00000 | – |
| B5-D | 0.22555 | 0.19413 | -14% |

**Totals**:
- **The run**: 7,107 agent calls sent, at $1.1975 computed and $0.9277
  billed (Parasail). That is the change in the spend ledger.
- **Bench and run together**: this step's `make-bench` cost $1.3796 and the
  run $0.9277, so $2.31 in all, against the ~$2.5 projected.
- **Cumulative spend**: $5.3946, under the $12 budget cap and the $10 stop
  rule.

Parasail billed 14–28% below its listed price, and the gap is largest for
B5-U, which has the longest conversations. I haven't checked why.

## Errors

| Config | Errors |
|---|---|
| B1 | 0 |
| B2 | 0 |
| B3 | 0 |
| B4 | 0 |
| B5 | 0 |
| B5-U | 0 |
| B4-D | 0 |
| B5-D | 0 |

## Reporting rule and amendments

- **Reporting rule**: every config and every comparison in the plan is
  reported above. No config, query or comparison was dropped.
- **Amendments**: none. The plan wasn't changed after the full run started.
