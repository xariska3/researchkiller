# Full bench (Step 7)

Run on 2026-10-13 at commit c027718 (`main`, after the Step 6 merge):

```
python -m rk make-bench --config configs/phase1.yaml
```

The command took no `--limit`. It resumed the frozen pilot bench in `data/bench`
(35 accepted queries, `test_a.jsonl` sha256 4c4df58bf66c) and continued in the
seeded order until `n_targets` (500) queries were accepted. It ran from 10:16 to
11:56 local time (1 h 40 min) and exited with code 0. The $10 stop
rule never triggered: cumulative spend ended at $4.47.

The finished `test_a.jsonl` has sha256 18f33b6067f5
(18f33b6067f538d488e7e8475a1958389ec75fe76dc4d222eda9eabbf1921dfc).

## Targets

The bench reached its goal of 500 accepted queries. Along the way it visited
the first 619 of the 750 frozen targets in seeded order (qids A0000–A0618). The
last 131 were never visited.

| Group | All | Pilot targets (A0000–A0044) | From A0045 on |
|---|---|---|---|
| Accepted | 500 | 35 | 465 |
| Exhausted (all 4 attempts rejected) | 86 | 5 | 81 |
| Skipped (not a research contribution) | 33 | 5 | 28 |
| Unfinished | 0 | 0 | 0 |
| Visited | 619 | 45 | 574 |

**Exhaustion rate**: 86 of the 586 paraphrased targets (14.7%) were
exhausted. The rate was 5 / 40 (12.5%) on the pilot targets and 81 / 546
(14.8%) after them.

**Attempts**: 1,211 in all (500 accepted, 711 rejected). The attempt on which
each accepted target was accepted:

| Attempt | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Accepted | 235 | 197 | 34 | 34 |
| Rejected | 351 | 154 | 120 | 86 |

**Composition of the accepted queries**:
- by subfield: Artificial Intelligence 442, Information Systems 58;
- by primary topic: Topic Modeling 317, Natural Language Processing Techniques
  89, Information Retrieval and Search Behavior 58, Speech and dialogue systems
  36.

## Rejections by reason

The checks run in the order below, and the first one that fails is the
rejection reason. The table counts all 711 rejected attempts, including those
of targets that were accepted later. The 33 skipped targets are not attempts
and are counted separately.

| Reason | Check | Attempts | Share |
|---|---|---|---|
| length | length | 0 | 0.0% |
| acronym_in_paraphrase | leak filter | 66 | 9.3% |
| source_acronym | leak filter | 0 | 0.0% |
| source_name | leak filter | 53 | 7.5% |
| title_stem | leak filter | 303 | 42.6% |
| idf_overlap | leak filter | 36 | 5.1% |
| content_jaccard | leak filter | 100 | 14.1% |
| judge_parse_error | judge | 0 | 0.0% |
| judge_captures | judge | 92 | 12.9% |
| judge_adds_nothing | judge | 61 | 8.6% |
| **All** | | **711** | |
| not_research_contribution (targets skipped) | before any attempt | 33 targets | |

- **The leak filter rejected 558 attempts (78.5%)**: title_stem alone
  rejected 303. It was the most common reason on every attempt, and 165 of
  its 303 rejections came on the first attempt, before any retry prompt
  quoted a reason.
- **The judge rejected the other 153 (21.5%)**: it saw 653 attempts, those
  that passed the leak filter, and accepted 500 of them (76.6%). It never
  failed to parse.
- **Facets the judge failed**: a rejection can name several.

| Direction | Problem | Method | Domain | Claim |
|---|---|---|---|---|
| captures | 0 | 4 | 46 | 54 |
| adds_nothing | 0 | 0 | 22 | 48 |

- **Exhausted targets**: their 344 rejected attempts were mostly leak-filter
  rejections; only 69 (20%) came from the judge.
  - 42 of the 86 exhausted targets had at least one judge rejection.
  - 44 were exhausted by the leak filter alone.
  - 1 was rejected by the judge on all four attempts.
  - title_stem rejected 162 of the 344 attempts.

The exhaustion check in the analysis plan (abstract measures and subfield,
exhausted vs accepted) is part of the full-run report,
[full-run.md](full-run.md).

## Money

| | Calls | Billed ($) |
|---|---|---|
| Paraphraser, `meta-llama/llama-4-maverick` (Novita) | 1,136 | 0.2503 |
| Judge, `mistralai/mistral-medium-3.1` (Mistral) | 1,127 | 1.1293 |
| **This make-bench run** | **2,263** | **1.3796** |

- **This run**: it cost $1.38. That's the change in the spend ledger
  (`data/cache/llm.sqlite`), against the $1.39 projected before the run. Every
  paraphraser call was served by Novita and every judge call by Mistral, the
  pinned endpoints.
- **The whole bench**, from `stats.json` and including the 35 pilot queries:
  - method cost $1.4912 (paraphraser $0.2667, judge $1.2245);
  - $1.4780 at config prices for the calls actually sent. 61 paraphraser
    calls of the pilot part were cache hits.
- **Cumulative spend across all commands**: $4.4669, up from $3.0873. The
  budget cap is $12 and the stop rule is $10.

## Notes

- **The $10 stop rule**: a watcher script outside `rk` checked the spend
  ledger every 5 s while `make-bench` ran. Had spend passed $10, it would have
  stopped the command once the attempt in progress was written. It never
  triggered.
- **The pilot run's report**: `test_a.jsonl` has changed, so regenerating
  `runs/20261013-121507` now shows its exhaustion check as not computed. Its
  `summary.md` from before the bench was finished keeps the pilot-bench
  version.
