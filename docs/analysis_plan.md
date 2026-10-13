# Analysis plan for the Phase 1 full run

Written on 2026-10-13, after the Step 6 pilot (`runs/20261013-121507`, 30
queries) and before the full run. This file fixes what the full run reports and
how. The design itself is in [phase1_spec.md](phase1_spec.md).

## Data

- **Queries**: every accepted query in `data/bench/test_a.jsonl` once
  `make-bench` has finished. That is 500 queries, or fewer if the frozen
  targets run out first. No targets are added.
- **Configs**: all eight (B1, B2, B3, B4, B5, B5-U, B4-D, B5-D) on every query,
  in one run. Settings are those in `configs/phase1.yaml`: `max_queries` 5,
  `max_llm_calls_agent` 6, and 15 / 20 for B5-U; agent
  `qwen/qwen3-235b-a22b-2507` at temperature 0; `run.seed` 42. All LLM calls
  are cached.
- **Metrics**: Hit@1, Hit@5, Hit@10 and MRR, where the target or any of its
  aliases counts, and only within the top 100.
- **Errors**: a row that ends in an LLM error is counted in the error table.
  A paired statistic uses only the queries that succeeded in both configs.

## Primary comparison

**B5 vs B4** (B5 minus B4), on Hit@5 and MRR:
- the mean paired difference, with a seeded paired-bootstrap 95% CI (10,000
  resamples, `run.seed`);
- an exact two-sided McNemar test on Hit@5 (only-B5 vs only-B4 hits).

**Decision rule**: Hit@5 with the McNemar test decides the primary comparison,
at α = 0.05. ΔHit@5 and its CI give the size of the difference. MRR supports
the result: ΔMRR and its CI are reported, but they don't decide it.

## Secondary comparisons

These use the same statistics as the primary comparison and are always given
as "A vs B", meaning A minus B.

- B4 vs B3: does one-shot rewriting help?
- B5-U vs B5: what does a larger budget buy?
- B2 vs B3: dense alone against the hybrid.
- B4-D vs B2, B5-D vs B2, B5-D vs B4-D: B4 and B5 on a dense base.
- Every config vs B2, the strongest no-LLM baseline in the pilot: B1, B3, B4,
  B5, B5-U, B4-D and B5-D vs B2. B3 vs B2 is B2 vs B3 with the sign flipped, so
  it is reported once, as B2 vs B3.

## Difficulty

This analysis is reported as in the pilot. A query's difficulty is the rank of
its target in B1, in four buckets: 1, 2–10, 11–100, and not found in the top
100. For each config the report gives Hit@5 within each bucket, plus the bucket
sizes. Difficulty is descriptive only. Queries are never filtered on it.

## Robustness: without the pilot targets

The bench design (prompts, leak filter thresholds, judge model) was tuned on
the first targets of the seeded visiting order. These are qids A0000–A0044,
which cover 45 targets:
- 40 were paraphrased while the design was tuned: 35 accepted and 5
  exhausted;
- 5 were skipped as not research contributions.

The 35 accepted ones are the only queries among them. They are the whole bench
as of commit 66e6146 (`test_a.jsonl` sha256 4c4df58bf66c). The 30 pilot-run
queries, the human check and the judge validation all come from them.

The main table (Hit@1/5/10 and MRR per config, with n) and the primary
comparison are repeated on the queries from qid A0045 onward. That is about 465
if the bench reaches 500.

## Exhaustion check

The human check found that the judge wrongly rejects proposals for stating
aims rather than the paper's findings. Wrong rejections can exhaust a target's
4 attempts. This check asks whether that biases which papers survive into the
bench.

**Groups**: every target `make-bench` paraphrased, split into two groups:
- accepted: the target has a query in `test_a.jsonl`;
- exhausted: all `max_attempts_per_target` (4) attempts were rejected.

Skipped targets (no attempt) are left out. So is any target whose attempts are
unfinished, if there is one; such targets are counted. The pilot targets are
included, since the bench refuses outputs made with other prompts or models.

**Measures**, all taken from the target's abstract in the frozen corpus:
1. **Abstract length**: whitespace-separated words (`len(abstract.split())`,
   as in the target filter).
2. **Share of number sentences**: the fraction of the abstract's sentences
   (split as `rk.bench.sentences` splits them) that contain a number. A number
   is a digit run that isn't joined to a letter or a hyphen:
   `(?<![\w-])\d+(?:[.,]\d+)*%?(?![\w-])`. So "12.5%", "0.83" and "2025" count;
   "GPT-4", "F1" and "3D" don't.
3. **Share of results sentences**: the fraction of sentences that contain a
   results cue. Cues are matched case-insensitively and as whole words:
   result, results, show, shows, showed, shown, demonstrate, demonstrates,
   demonstrated, outperform, outperforms, outperformed, outperforming, achieve,
   achieves, achieved, achieving, surpass, surpasses, surpassed, reveal,
   reveals, revealed, find, finds, found, finding, findings, indicate,
   indicates, indicated, improves, improved, improvement, improvements,
   significantly, state-of-the-art.
4. **Subfield**: OpenAlex `primary_topic.subfield`. Among the 750 frozen
   targets it is either Artificial Intelligence (666) or Information Systems
   (84). The finer breakdown is the primary topic: Topic Modeling 487, Natural
   Language Processing Techniques 121, Information Retrieval and Search
   Behavior 84, Speech and dialogue systems 58. Information Systems is exactly
   the Information Retrieval topic.

These definitions are proxies, fixed now, before any of the new data are seen.
The implementation must match them.

**Statistics**:
- **Measures 1–3**: the mean in each group, and the difference (exhausted minus
  accepted) with a seeded bootstrap 95% CI. The CI uses 10,000 resamples within
  each group and `run.seed`. A two-sided permutation p-value comes from 10,000
  label permutations, also seeded with `run.seed`: p = (1 + the number of
  permutations whose |difference| is at least the observed one) / 10,001.
- **Subfield**: the exhaustion rate per subfield, i.e. exhausted / (accepted +
  exhausted). Its difference (Information Systems minus Artificial
  Intelligence) gets the same CI and permutation test.
- **Topic**: the exhaustion rate per topic, with counts. This is descriptive
  only.
- **Descriptive**: for the exhausted targets, the rejection reason of each
  attempt (leak filter or judge, and the judge's failed facets), to show how
  much exhaustion is the judge's doing.

**Reading**: the judge's aims-vs-findings pattern would bias the bench if the
exhausted targets have more results or number sentences, with a CI that
excludes zero. In that case the README states the bias and its size as a
limitation. The bench stays frozen either way: no target is re-run, re-judged
or added.

## Interpretation and reporting

- The primary comparison is decided as stated above. Elsewhere, a difference
  is called significant when its 95% CI excludes zero. For Hit@5, the McNemar
  p-value is reported alongside.
- Only the primary comparison is confirmatory. The secondary comparisons, the
  difficulty and robustness analyses and the exhaustion check are exploratory.
  They are reported without a multiple-comparison correction and labelled as
  exploratory.
- **Reporting rule**: all configs and all comparisons in this plan are
  reported, whatever the results. No config, query or comparison is dropped
  after the results are seen.
- Also reported, as specified:
  - effort per query (mean / p50 / p95);
  - the cost–quality table, on computed cost;
  - billed cost, kept separate;
  - errors.
- Any change to this plan after the full run starts is added as a dated
  amendment at the end of this file, with its reason. The text above stays as
  it is.

## Implementation

`python -m rk run` (at its end) and `python -m rk report` produce everything
in this plan in `summary.md` and `summary.json`. The sections of `summary.md`:
- Retrieval quality;
- Effort per query;
- Hit@5 by difficulty;
- Primary comparison, holding only B5 vs B4, with the decision rule;
- Secondary comparisons (exploratory), in the order listed above;
- Robustness: without the pilot targets (exploratory);
- Exhaustion check (exploratory);
- Cost and quality;
- Billed cost;
- Errors.

The definitions are in `rk.exhaustion` (`NUMBER_RE`, `RESULTS_CUES`) and in
`rk.evaluate` (`PRIMARY_PAIR`, `SECONDARY_PAIRS`, `PILOT_POSITIONS`).

The exhaustion check reads the bench and frozen corpus through the run's
config snapshot. It runs only if `test_a.jsonl` still has the sha256 the run
recorded; otherwise the summary says it wasn't computed.
