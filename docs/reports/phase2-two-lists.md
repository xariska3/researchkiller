# Two lists (issue #22): closest in contribution, closest in methods

Every full check now returns two ranked lists from the same candidates, from one
rerank call:
- **A, "Closest in contribution (possible prior art)"**: ranked by overlap with what
  the idea claims as new (its problem and claim). Papers that share only tools,
  datasets or software rank lower.
- **B, "Closest in methods (worth citing)"**: ranked by overlap in method, tools and
  data.

The evaluation was declared in docs/phase2_spec.md (Evaluation, two lists) and
committed (10de9d2) before any call, with the code (523320a). It is report only:
nothing was tuned, and the test set was not used.

## Summary

- **List A passes.** On the 120 dev checks, its Hit@5 is 0.850 against the 0.84 line,
  and its Hit@1 is 0.833. Both equal L5r's: list A placed the original exactly where
  L5r did in all 120 checks. So two lists are now the default full check
  (`rerank.two_lists: true`).
- **Cost per check**: median $0.0133, max $0.0181, under the $0.03 cap. Step 5's
  single list had about the same median ($0.0136).
  - *Later note (2026-10-19): with section B's 6 papers no longer in section A, it
    is median $0.0156, max $0.0183; see
    [Section B without repeats](#section-b-without-repeats-2026-10-19).*
- **Time per check, estimated**: median 40.1 s, max 57.3 s. That is about 8 s more
  than the 32.4 s of Step 5's check of the fixes, and the max is close to the spec's
  60 s.
  - **The rerank call** takes 11.9 s (median) against R1's 5.4 s in Step 4, since it
    writes two lines and two rankings.
  - **The explainer stage** takes 11.0 s (median) against 9.8 s, for 9 papers
    instead of 10 but up to 14.
  - *Later note (2026-10-19): with 14 papers in every check, it is median 45.4 s,
    max 60.8 s.*
- **Robustness**: nothing fell back in 120 checks. Every reply was readable, with
  both lists, and no explainer reply was cut off.
- **Spend**: $0.72 of the $2.00 cap, over the pilot and both dev runs.

## What changed

- **The rerank call** (`rk.live.pipeline.with_rerank`, `two_lists`): the same model,
  settings, candidates and cut abstracts as L5r. The prompt
  (`src/rk/prompts/rerank_two_lists.txt`) asks the model to state, in one line each,
  what the idea claims as new and what methods it uses, then to rank all candidates
  twice. The reply is strict JSON, enforced by the API's schema (the parasail
  endpoint lists `structured_outputs`).
  - **List A** is the check's main list, with L5w's other works after it, as in L5r.
  - **List B** has the candidates the reply leaves out after it, in presented order.
  - **Fallback**: if the call fails, could cross the cap, or list A can't be read,
    list A keeps the check's order (L5w's, as L5r does) and list B is empty, with one
    warning. A readable list A without list B keeps list A, with a warning.
- **The explainer**: one call for the top 8 of list A and the top 6 of list B
  (`explain.contribution_top_k`, `explain.methods_top_k`). A paper in both is
  explained once and shown in both. The quote checks and the $0.03 cap are unchanged,
  and the vocabulary map uses every explained paper.
  - *Later note (2026-10-19): list B's 6 are now its first 6 that aren't in list A's
    top 8, so every check explains 14 papers and none is shown twice; see
    [Section B without repeats](#section-b-without-repeats-2026-10-19).*
- **The report** (terminal, Markdown export, result page): section A, then section B,
  each with its heading and a one-line description.
  - **Section A** keeps the report's order (overlap level, then rank) and its "Also
    found" list (ranks 9 to 20). A paper that is also in section B gets an "also in
    methods" note.
    - *Later note (2026-10-19): the note now marks the section A papers named on
      section B's "Also close in methods" line.*
  - **Section B** is in its own rank order.
    - *Later note (2026-10-19): it now shows list B's first 6 papers that aren't in
      section A, under a line naming the section A papers list B ranks among them.*
  - **The report's JSON** also keeps the reply's two lines (`idea`). They are not
    shown.
    - *Later note (2026-10-19): they are now shown at the top of the results, under
      "How the tool read your idea".*
- **The web app**: list B's papers are stored in `candidates` and `explanations`
  too, in a new `ranking` column, which is added to an existing database. A stored
  one-list report isn't reused for a two-list check.
- **Unchanged**: the quick check, and live-eval's L5r.

## The evaluation

### Pilot

`python -m rk rerank runs/live-20261014-114302 --variants R1,R4 --limit 3`
(`runs/rerank-20261016-164025`; Phase 1 tuning targets, neither dev nor test).
- **Replies**: all 3 readable, both lists complete (35-38 numbers each), and the
  lines on point.
- **The call**: $0.0017 against R1's $0.0015, and 14-16 s.
- **Ranks**: list A ranked the original first in all 3.

### List A on the 120 dev checks

`python -m rk rerank runs/live-20261014-122144 --variants R1,R4`
(`runs/rerank-20261016-164915`). R4 is the two-list call scored on list A; R1 (L5r)
came from the cache at $0. The replay reproduced all 120 recorded L5 rankings.

| Config | Hit@1 | Hit@5 | Hit@10 | MRR@20 |
|---|---|---|---|---|
| L5w | 0.508 | 0.750 | 0.817 | 0.617 |
| L5r (R1) | 0.833 | 0.850 | 0.850 | 0.842 |
| **List A (R4)** | **0.833** | **0.850** | **0.850** | **0.842** |

- **Against L5r**: 0 queries gained and 0 lost (McNemar p = 1). The original is at
  the same rank in all 120.
- **The ceiling** is unchanged: the original is among the candidates in 103 of 120.
  Of the 18 misses, 17 are checks whose original never reached the candidates; in the
  other (A0109), both lists ranked it below 20.
- **List B** was readable in all 120 replies.
- **The call**: $0.0016 (median; p95 $0.0019, max $0.0021) against R1's $0.0014, and
  11.9 s (median; p95 16.8 s, max 20.2 s).

### Cost and time per check

`python -m rk explain-eval --from-qid A0045 --limit 120 --timing-run
runs/rerank-20261016-164915`, with `rerank.two_lists=true` and the dev run's work
types (`runs/explain-20261016-171526`). The searches and the agent and rerank calls
came from the caches; the keyword lookups and the 120 explainer calls were new.

| Per check | Median | p95 | Max |
|---|---|---|---|
| Whole check's cost (cap $0.03) | $0.0133 | $0.0167 | $0.0181 |
| Explainer call's cost | $0.0041 | $0.0062 | $0.00675 |
| Estimated live time | 40.1 s | 51.9 s | 57.3 s |
| Explainer stage | 11.0 s | 20.4 s | 27.1 s |
| Papers explained | 9 | 13 | 14 |
| Of list B's 6, also in list A's 8 | 5 | 6 | 6 |

- **Estimated live time**: as in Step 5, the dev run's recorded check time, plus R4's
  recorded call time, plus this run's snippet, keyword, explainer and vocabulary
  stages.
- **The cap**: no call was refused at the cap, including the explainer calls of the
  3 checks with 14 papers, and the most expensive check cost $0.0181.
- **Quotes**, of 1,142 explained papers: 91.6% exact, 7.3% matched, 1.1% unverified.
  - **On the same 20 queries as Step 5's check of the fixes** (A0071 to A0094): 93.7%
    exact, against 95.5% there.
  - **By list**: list A's papers 92.1% exact, list B's other papers 89.0%.
- **The original**: in list A's explained top 8 in 102 checks, and rated high in all
  102. It is in list B's top 6 in 99 of them, and never in list B alone.
- **Other papers rated high**: 58 of 1,142 (5.1%), against 6 of 200 (3.0%) in Step
  5's check of the fixes.

## What this doesn't show

- **Whether issue #22's case is fixed.** The benchmark's ideas are paraphrases of one
  paper, so their contribution and their methods point to the same papers: list A
  ranked every original as L5r did, and list B's top 6 overlapped list A's top 8 by
  5 papers (median). The real user's satellite IoT check, where shared tools
  outranked the contribution, is not in the repo; running it again would show
  whether list A now puts the satellite measurement papers first.
- **List B's quality.** There is no ground truth for "closest in methods", so it is
  not scored.
- **List A on the test set.** As declared, the test set wasn't used. The README's
  results table is still L5r's (single list) on the 80 test queries.
  - *Later note (2026-10-19): the README now says so under the table, with list A's
    dev result. Its other cost and time figures are now the two-list check's (about
    $0.016 and 45 s, medians on the 120 dev checks).*

## Deviations and choices

- **The fallback** keeps the check's order (L5w's) for list A, as L5r's fallback
  does now. It doesn't make a second call with L5r's prompt, since the spec asks for
  one call.
- **`python -m rk rerank`** keeps Step 4's four variants as its default; R4 runs only
  when named (`--variants R1,R4`), so a plain run still reproduces Step 4.
- **The explainer's warning** for a missing explanation names a list B paper by its
  rank in list B ("rank 2 of the methods list"), since positions in the call are not
  ranks any more.
- **`max_tokens` stays 3,500** for the explainer: the longest reply in Step 5 was
  about 186 tokens a paper, so 14 papers fit, and no reply was cut off.
  - *Later note (2026-10-19): with 14 papers in every check, the longest reply was
    2,603 tokens, and none was cut off.*

## Section B without repeats (2026-10-19)

After reviewing this PR, the author asked for three changes:
- section B shows the top 6 methods papers that aren't in section A's top 8, each
  fully explained, with one line naming the section A papers that also rank high in
  methods;
- the reranker's two lines appear at the top of the results and in the export;
- the README describes the two lists.

The re-run of the cost check was declared in docs/phase2_spec.md (Evaluation, two
lists) and committed with the code (bc0a276) before any call. It passes when every
check costs under the $0.03 cap with no explainer call refused at it.

### What changed

- **Section B** (`rk.live.check.methods_section`) has list B's first 6 papers that
  aren't in section A's top 8. They are numbered by their rank in list B. The explainer
  call now has 14 papers: list A's top 8, then section B.
- **The line** "Also close in methods: #1, #3 above" names, by their numbers in
  section A, the section A papers that list B ranks among section B's. So down to
  section B's last paper, every paper of list B is either in section B or named on the
  line.
  - Those papers keep the "also in methods" note in section A.
  - A check whose list B can't be read has neither section B nor the line, and says
    so as before.
  - *Later note (2026-10-19): at the author's request, three or more consecutive
    numbers show as a range, as in "Also close in methods: #1 to #8 above". Two are
    still listed ("#1, #2").*
- **"How the tool read your idea"**: the page, the export and the terminal open the
  results with the reply's two lines, as "Contribution: ..." and "Methods: ...".
  - A missing line is left out.
  - A check without the lines (one list, or a fallback) has no block.
- **The config**: with two lists, `contribution_top_k + methods_top_k` must be at most
  `live.top_k` (8 + 6 against 20), so that list B has 6 papers outside section A when
  the pool is big enough.
- **The README**: "How it works" describes the two lists. A line under the results
  table, which stays the single list's, gives list A's dev result.

### Cost and time per check

The same command and settings as the cost and time run above, at bc0a276
(`runs/explain-20261019-130131`). Of the 120 explainer calls, 117 were new. The other 3
came from the cache: in those checks, list B's top 6 shared no paper with list A's top
8, so the explained papers were the same as before.

| Per check | Before: median | Before: max | Now: median | Now: max |
|---|---|---|---|---|
| Whole check's cost (cap $0.03) | $0.0133 | $0.0181 | $0.0156 | $0.0183 |
| Explainer call's cost | $0.0041 | $0.00675 | $0.0064 | $0.0072 |
| Estimated live time | 40.1 s | 57.3 s | 45.4 s | 60.8 s |
| Explainer stage | 11.0 s | 27.1 s | 16.1 s | 24.8 s |
| Papers explained | 9 | 14 | 14 | 14 |

- **It passes.** All 120 checks finished under the cap, and no call was refused at it.
  - **Fallbacks**: none.
  - **Cut-off replies**: none; the longest reply was 2,603 tokens of 3,500.
- **Cost**: $0.0023 more a check at the median, for 14 papers instead of 9. The max
  barely moved: the most expensive check before already explained 14.
- **Time**: 5.3 s more at the median.
  - The max, 60.8 s, is above the 57.3 s the author accepted and the spec's 60 s.
  - 2 of 120 checks are estimated at 60 s or more.
  - As declared, nothing was changed for time.
  - *Later note (2026-10-19): the author accepted the 60.8 s max.*
- **Section B**: 6 papers in all 120 checks.
  - **The line** names all 8 of section A's papers in 65 checks (the median is 8),
    and none in 3.
  - **Where section B starts**: in those 65 checks, list B's top 8 are list A's top
    8, so section B is list B's ranks 9 to 14. The benchmark's ideas paraphrase one
    paper, so contribution and methods point to the same papers.
- **Quotes**, of 1,680 explained papers: 92.3% exact, 6.6% matched, 1.1% unverified.
  - **By section**: section A 93.2% exact, section B 91.0%.
  - **On A0071 to A0094**: 94.3%, against 93.7% before and 95.5% in Step 5.
- **The original**: rated high in all 102 checks that explained it.
  - It is on section B's line in 99 of them.
  - It is never in section B itself.
- **Other papers rated high**: 83 of 1,680 (4.9%), of which 13 are among section B's
  720 (1.8%).
- **Spend**: $0.74 of LLM calls, against the $1.50 cap. The two-lists work has now
  spent $1.46 in all.

## Open questions

1. **The README**: "How it works" still says the top 10 are explained, and the
   results table is the single list's. Should I update the text for two lists now,
   and say the table is L5r's, or leave it for you?
   - *Later note (2026-10-19): answered. "How it works" now describes the two lists,
     the table stays, and a line under it gives list A's dev result.*
2. **The idea lines**: should the page show the reranker's two lines (what it read as
   the contribution and the methods) above each section? They are in the JSON only.
   - *Later note (2026-10-19): answered. They are shown once, at the top of the
     results and the export, under "How the tool read your idea".*
3. **The time**: the max estimate (57.3 s) is close to 60 s. If that matters, list B
   could be cut to fewer candidates, or the explainer split into two calls; neither
   is tested.
   - *Later note (2026-10-19): answered. 57 s is acceptable, so nothing was changed
     for time.*
4. **Repeats in section B**: as asked, a paper in both lists is shown in full in
   both. In 57 of the 120 dev checks, all 6 of section B's papers are also in section
   A, so the page shows them twice. Should section B show those as a title and a
   pointer to section A instead?
   - *Later note (2026-10-19): answered. Section B now shows the top 6 methods
     papers that aren't in section A's top 8, each fully explained, and one line
     names the section A papers that also rank high in methods.*
