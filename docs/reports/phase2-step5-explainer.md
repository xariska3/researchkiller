# Phase 2 Step 5: what a check shows for its top papers

Every check now returns its top 10 papers with an explanation a researcher can check
at a glance:
- **a snippet**: the abstract sentence closest to the idea;
- **an overlap explanation** from one LLM call: overlap level, a verdict per facet,
  a quote from the abstract, and what differs;
- **a vocabulary map** of the terms the literature uses that the idea doesn't.

`python -m rk check --config configs/phase1.yaml --file idea.txt [--quick]` prints
the report and saves it as JSON under `runs/check-<timestamp>/`. The first evaluation
below is report only: nothing was tuned on it.

**Update (2026-10-15)**:
- **Hand review** of 50 rows by Claude, not the author: 0 false alarms; correct 40
  yes, 9 partly, 1 no; useful 31, 17, 2.
- **Product fixes**: five, chosen after that review and the first 20 dev checks (see
  Product fixes).
- **Check of the fixes** on 20 other dev checks:
  - **Quotes**: 95.5% exact, 4.0% matched, 0.5% unverified.
  - **The original**: in the top 10 and rated high in all 20 checks.
  - **Cost**: median $0.0136.
  - **Duplicates**: 4 real duplicates merged.
- **Most of the text below describes the first run.** "What a check does now" has
  been updated for the fixes.

## Summary of the first run

- **Quotes**: 168 of 200 explained papers (84.0%) have an evidence sentence found
  verbatim in the abstract.
  - **The other 32 are not invented.** Each is a real sentence of that abstract,
    edited by the model: a lead-in dropped ("To address this, we propose" became "We
    propose"), the end cut off, or a text artefact of the source fixed.
  - **As specified**, those papers show their snippet instead, marked "quote
    unverified".
- **The original paper**: in the top 10 of 16 of the 20 checks (at rank 1 each
  time), and rated **high in all 16**. Its quote was verified in 14.
  - **This is a consistency check, not independent evidence**: the same Mistral
    model accepted each benchmark proposal as keeping all four facets of its
    original.
- **Cost per check**, with the explainer: median $0.0133, p95 $0.0159, max $0.0161,
  under the $0.03 cap. The explainer call is $0.0045 of it.
- **Time per check**, estimated: median 31.1 s, p95 38.1 s, max 40.1 s, under the
  spec's 60 s.
  - The explainer stage is 9.7 s of it (median).
  - A fully live check, in the first pilot, took 35.0 s.
- **Robustness**: no step fell back in 20 checks. Every rerank reply and explainer
  reply was readable, no reply was cut off, every paper got an explanation, and no
  keyword lookup failed.
- **Hand review**: [docs/evaluation/explainer_review.csv](../evaluation/explainer_review.csv),
  the top 5 papers of 10 checks. It is now filled in; see Hand review.

## What a check does now

| Step | Default check | Quick check (`--quick`) |
|---|---|---|
| Retrieval | L5r (Step 4) | L1: one OpenAlex semantic search, reranked locally |
| Versions of one paper merged in the final list (product fix) | yes | yes |
| Snippets (top 10) | yes | yes |
| Keyword lookup (top 10) | yes | yes |
| Explainer call (top 10) | yes | no |
| Vocabulary map | keywords, titles, the agent's queries | keywords, titles |
| LLM calls | agent, rerank, explainer | none |

- **Snippet** (`rk.live.explain.snippets`): the abstract is split into sentences by
  rule (not after "et al.", "e.g." or an initial). Each sentence is scored against
  the idea with the check's Qwen3-Embedding encoder, the idea encoded as a query.
  - **Product fixes**: a full stop followed directly by a capital ("datasets.We")
    ends a sentence too, and a snippet is cut at 350 characters, at a word, with "…".
- **Explainer** (`rk.live.explain.explain`): one call to Mistral Medium 3.1, the
  validated Phase 1 judge (`explain.model`: the judge's endpoint and settings, with
  `max_tokens` 3,500).
  - **Input**: the idea and the 10 papers (title, and the abstract cut at 2,000
    characters, as for the rerank call).
  - **Prompt**: `rk/prompts/explain_overlap.txt`, built from the judge's facet
    definitions (problem, method, domain, claim).
  - **Answer**: strict JSON, enforced by the API's schema. Per paper: an overlap
    level (high, partial or low); same or different for each facet, with a reason;
    one evidence sentence; and one line on what differs.
- **Quote check** (first run): the evidence must be in the abstract verbatim, after
  normalising only whitespace and the form of quotation marks. Quotation marks around
  the whole quote are ignored, and a quote must be at least 5 words long.
  - If it fails, the quote is dropped and the paper is marked unverified, and its
    snippet is shown.
  - **Now** (product fix, `locate_quote`): the abstract's own sentence is always
    shown, never the explainer's version.
    - **verified-exact**: the quote is in that sentence verbatim.
    - **verified-matched**: no verbatim match, but the sentence has at least 90% of
      the quote's words.
    - **unverified**: neither, or a quote under 5 words. The snippet is shown.
- **"Differs"** (product fix): a substantive difference, or exactly "No substantive
  difference found.". It is never publication status or "the proposal is a plan".
- **Fallbacks**: if the explainer call fails, could cross the cap, or can't be read,
  the papers keep their snippets and the check carries a warning. The whole check
  never fails at this stage. A missing or invalid entry costs only its own paper.
- **Vocabulary map** (`rk.live.vocab`): rule-based, with no LLM.
  - **Sources**: the top papers' OpenAlex keywords, their titles, and the agent's
    queries.
  - **Terms**: a phrase is a run of words between stopwords and punctuation. It
    counts if it has a content word the idea lacks, and at least one top paper uses
    it.
  - **Product fix**: phrases also end at using, based, via, with, for, by and
    through, and a phrase ending in one ("graph-based") is dropped.
  - **Ranking**: terms are ranked by how many top papers use them, then grouped by
    source, at most 10 in all.
  - **Keywords need one extra call**: the searches don't ask for keywords, and
    adding them to the searches would change every cached search request. So the
    keywords come from one OpenAlex list call by id ($0.0001), cached like the
    searches.
- **Report** (`rk.live.report`, Pydantic):
  - `CheckRequest` takes an idea of 40–500 words, an optional "published before"
    date, and the quick flag.
  - `CheckReport` always has the header "Closest prior work found; no close match
    is not proof of novelty.". It also holds the coverage line, the papers, the
    vocabulary, the warnings, the cost, the stage times, the models and the prompt
    hashes.
  - The terminal report lists the explained papers by overlap level, then rank, as
    the spec says, then ranks 11–20 by title.
- **Budget**: `explain.cost_cap_per_check` ($0.03) caps the whole check in
  `rk check`. live-eval keeps its $0.10 cap (`live.cost_cap_per_check`), so older
  runs can still be resumed.

## The evaluation (declared in docs/phase2_spec.md before any explainer call)

- **Run**: `runs/explain-20261015-212527`, at commit 4287207 with a clean tree:

  ```
  python -m rk explain-eval --config configs/phase1.yaml --limit 20 --review 10 --timing-run runs/rerank-20261015-132644 --set run.budget_cap=6.57 --set live.work_types=[article,preprint,conference-paper,book-chapter,dissertation,review,book,report]
  ```

- **Queries**: the first 20 dev queries in seeded order, A0045 to A0070.
- **Replay**: the searches and the agent and rerank calls replay Step 2's and Step
  4's dev checks from the caches; only the explainer calls and the keyword lookups
  are new.
  - **The work types**: the dev run predates `software-paper` in `live.work_types`.
    With today's list, its searches aren't in the cache (the first pilot found this),
    so the run uses the dev run's own list.
- **Time**: the check's own time is not a live time, since most of it came from the
  caches. Each check's live time is estimated as:
  - its recorded check time (the dev run's L5 row, the same check as L5w);
  - plus its recorded rerank call (Step 4's R1 row);
  - plus this run's snippet, keyword, explainer and vocabulary stages.
  - **A0045 is left out of the time measures**: its explainer reply came from the
    second pilot's cache. So the time measures cover 19 real calls.

### Measures

| 20 dev checks | Result |
|---|---|
| Quote verification | **168 / 200 (84.0%)** |
| Papers explained | 200 / 200 (no fallback, no partial reply) |
| Original in the explained top 10 | 16 / 20, all at rank 1 |
| Original's overlap level | **high 16**, partial 0, low 0 |
| Original's quote verified | 14 / 16 |
| Overlap of all 200 papers | high 24, partial 75, low 101 |
| Papers other than the original rated high | 8: 5 in A0049, 2 in A0059, 1 in A0069 |
| Vocabulary terms per check | 10 in every check |

- **The 4 checks without the original** (A0049, A0051, A0057, A0064) match Step 4:
  in those, retrieval didn't bring the original into the top 10.
  - **A0049's 5 "high" papers** are all KV-cache-sharing systems for multi-agent
    LLMs, the proposal's topic. Whether they are real prior art or false alarms is
    for the hand review: A0049 is in the CSV.

### Why 32 quotes failed

I read all 32 next to their abstracts. Every one is a real sentence of that paper's
abstract, edited:

| What the model changed | Quotes |
|---|---|
| Dropped a lead-in and capitalised the new start ("To address this, we propose" → "We propose") | 14 |
| Fixed a text artefact of the source: hyphens lost in OpenAlex's text ("customeragent", "semisupervised"), LaTeX (`\textbf{pr}ivileged`), broken spacing ("F eedback"), quotation marks inside the sentence | 8 |
| Cut off the end of the sentence | 5 |
| Left out a parenthesis, a model list or a citation | 3 |
| Changed only the case of the first letter | 1 |
| Verbatim, but 4 words (the minimum is 5) | 1 |

- **None was invented or taken from another paper.** The check works as specified,
  but in this run it held back 16% of quotes for edits that leave the meaning intact.
- **Options** are under Open questions.

### Cost and time per check

| Median / p95 / max | |
|---|---|
| Whole check, method cost (cap $0.03) | $0.0133 / $0.0159 / $0.0161 |
| Explainer call | $0.0045 / $0.0050 / $0.0051 |
| Explainer call time (19 real calls) | 8.9 / 13.6 / 14.9 s |
| Explainer stage: snippets, keywords, explainer, vocabulary | 9.7 / 14.2 / 15.7 s |
| **Estimated live check** | **31.1 / 38.1 / 40.1 s** |

- **Snippets** take 0.35 s a check on the GPU (median). The keyword lookup and the
  vocabulary map take under a second together.
- **The explainer's answer** is about 1,600 output tokens. That is most of its cost
  ($2 per million output tokens), and well under its 3,500-token limit.
- **A fully live check**: the first pilot (`runs/explain-20261015-212213`, A0045 with
  today's work types) sent every search and call again.
  - **Time**: 35.0 s in all (searches and agent 15.2 s, rerank 9.7 s, explainer
    9.0 s).
  - **Cost**: $0.0116.

### The vocabulary map (observations, not measured)

- **Mostly useful terms**: on A0049, "KV cache reuse", "multi-agent LLM systems" and
  "KV cache sharing", where the proposal says "shared memory of previously processed
  text". On A0060, "chinese word segmentation" and "POS tagging", where the proposal
  says "Chinese fill-in-the-blank exercises".
- **Some noise**:
  - query fragments that end in a generic word ("order using", "cloze questions
    using");
  - lone words ("operators", "LLMs").
- **Not changed here**: nothing is tuned on this evaluation. A rule that ends phrases
  at "using", "based" or "via" would remove the first kind.

## Hand review

[docs/evaluation/explainer_review.csv](../evaluation/explainer_review.csv) has 50 rows:
the first 10 checks of the first run (A0045 to A0058), the top 5 papers of each.
- **Columns**: qid, idea, rank, title, overlap, verified, evidence and differs (from
  the first run's code), then the review: correct?, useful?, false alarm?, notes and
  reviewer.
- **Reviewer**: Claude (an AI assistant), not the author.

| 50 rows | yes | partly | no |
|---|---|---|---|
| Correct? (the overlap level and the explanation) | 40 | 9 | 1 |
| Useful? (to a researcher checking the idea) | 31 | 17 | 2 |
| False alarm? | 0 | – | 50 |

- **No false alarms.** A0049's 5 papers rated high were the main worry, since its
  original wasn't in the top 10.
  - **Four are real prior art**: KV-cache reuse or sharing across agents.
  - **One is "slightly generous"** (PReCache, multi-LoRA cache sharing; correct:
    partly).
- **What the 10 not-fully-correct rows got wrong**, from the notes:
  - **5 overlap levels off by one step**, in both directions. The 1 "no" is A0054's
    rank 2: rated partial, though it is the same metric and should be high.
  - **4 "differs" lines**:
    - 2 meta: "already published; proposal is a plan" (A0054 rank 1) and "proposal
      is a plan without implementation" (A0058 rank 1);
    - 2 inaccurate: A0049 rank 5, and A0051 rank 4 ("no dataset", where the
      competition released data).
  - **1 duplicate**: A0053's ranks 4 and 5 are two versions of one paper. Rank 5 is
    also one of the 2 rows rated not useful.
- **Snippets**: two were whole abstracts (A0056 rank 4, A0058 rank 4), because the
  splitter missed a full stop with no space after it ("datasets.We").
- **Fixed since**: the meta "differs", the duplicate and the whole-abstract snippets
  (Product fixes, items 3–5).
- **Not fixed**: the overlap level's calibration, off by one step in 5 of 50 rows.

## Product fixes (chosen after the first run and the hand review)

These were decided on 2026-10-15, after looking at the first 20 dev checks and their
hand review, so they are product fixes, not results of a blind evaluation. They were
checked on 20 other dev checks, as declared in docs/phase2_spec.md before the run.

1. **Quotes** (`rk.live.explain.locate_quote`):
   - **The prompt** asks for the sentence exactly, from its first word, keeping any
     opening words, symbols, markup and capitals.
   - **The code** shows the abstract's own sentence, never the explainer's version:
     verified-exact if the quote is in it verbatim, verified-matched if it has at
     least 90% of the quote's words, otherwise unverified and the snippet is shown.
   - **The 5-word minimum** stays.
2. **Vocabulary** (`rk.live.vocab`): phrases end at using, based, via, with, for, by
   and through, keywords included, and a phrase ending in one ("graph-based") is
   dropped.
3. **"Differs"**: a substantive difference (in problem, method, domain or claim), or
   exactly "No substantive difference found."; never publication status, the
   proposal being a plan, or the proposal not specifying something yet.
   - The code puts the fixed line into the prompt (`NO_DIFFERENCE`), and rewrites a
     near-miss such as a missing full stop to it.
4. **Sentences and snippets** (`rk.text.sentences`): a full stop followed directly
   by a capital ends a sentence ("datasets.We"). A snippet is cut at 350
   characters, at a word, with "…".
5. **Versions of one paper in the final list** (`rk.live.pipeline.merge_same_text`):
   two results merge when their titles' content stems have a Jaccard of at least 0.7
   and their abstracts share most sentences, with or without a shared author. The
   entry keeps the earlier place in the list.
   - **Which version it shows** (changed after PR #17, by your decision): the
     earliest-published version, with the others as its aliases. Before, it showed
     the one earlier in the list.
   - **Shared sentences**: more than half of the shorter abstract's sentences, where
     a sentence is shared if the other abstract has one with a word-set Jaccard of at
     least 0.5.
   - **Why not identical sentences**: the example, A0053's ranks 4 and 5, is a
     preprint and its journal version. Every sentence is reworded a little, so they
     share 0 identical sentences, but 67% at a Jaccard of 0.5.
   - **On the first run's lists**, the rule merges that pair and 3 exact duplicates.
     It leaves apart the 4 other pairs whose titles reach 0.7: those share at most
     33% of sentences.
   - **The pool and live-eval are unchanged**, and the list can be shorter than 20
     after a merge.

### The check of the fixes: 20 other dev checks

- **Run**: `runs/explain-20261015-215122`, at commit e0d40e0 with a clean tree:

  ```
  python -m rk explain-eval --config configs/phase1.yaml --from-qid A0071 --limit 20 --review 10 --review-out runs/explain-review-A0071.csv --timing-run runs/rerank-20261015-132644 --set run.budget_cap=6.46 --set live.work_types=[article,preprint,conference-paper,book-chapter,dissertation,review,book,report]
  ```

- **Queries**: the next 20 dev queries in seeded order, A0071 to A0094, none of
  them in the first run.
- **Setup**: the same replay and timing as the first run. All 20 explainer calls
  were real.

| | First run, A0045–A0070 (before the fixes) | Check, A0071–A0094 (after) |
|---|---|---|
| Quotes exact | 168 / 200 (84.0%) | **191 / 200 (95.5%)** |
| Quotes matched (shown as the abstract's sentence) | – (no such rule; 28 of the 32 would have matched) | 8 (4.0%) |
| Quotes unverified (snippet shown) | 32 (16.0%) | **1 (0.5%)** |
| Original in the explained top 10 | 16 / 20 | 20 / 20 |
| Original rated high | 16 / 16 | 20 / 20 (quote exact 19, matched 1) |
| Whole check, median / p95 / max | $0.0133 / $0.0159 / $0.0161 | $0.0136 / $0.0159 / $0.0162 |
| Explainer call, median | $0.0045 | $0.0047 |
| Estimated live check, median / p95 / max | 31.1 / 38.1 / 40.1 s | 32.4 / 37.0 / 40.5 s |
| Fallbacks | none | none |

- **The two runs have different queries**, so this is not a controlled comparison.
- **Most of the gain is the prompt**: 95.5% of quotes are now exact.
- **The 8 matched quotes** are the abstract's sentences from which the explainer:
  - dropped a lead-in ("Inspired by this observation, in this work we tackle" →
    "in this work we tackle");
  - changed the case of the first letter;
  - mended "pro-pose", a non-ASCII hyphen or a stray "\n".
- **The 1 unverified quote** (A0078 rank 2) shows its snippet.
- **"Differs"**: 23 of the 200 say "No substantive difference found.", and none cites
  publication status or "the proposal is a plan". One says "proposal does not
  specify", which the prompt also asks it to avoid.
- **Snippets**: 10 were cut at 350 characters; none is longer than 351 characters
  with the "…".
- **Vocabulary**: 10 terms per check, and none of the 200 ends in a function word.
- **Merges**: 4, in 2 checks, all real versions of one paper. The pool's own rules
  missed them, for one of three reasons: titles too short for its exact-title rule,
  years too far apart, or author names written differently ("Alokaili, A."). The
  four:
  - "Automatic Generation of Topic Labels" (two records, same day);
  - "Tagging and Tag Recommendation" (2019 and 2020);
  - "Probabilistic models for semantic visualization…" (a 2020 dissertation and
    its 2017 article);
  - "3D-SE Viewer…" (2007 and 2012).
- **"The earlier" meant earlier in the list in this run.** In the third pair, that
  kept the 2020 dissertation and made the 2017 article its alias. Since PR #17, the
  2017 article is shown, and the dissertation is its alias.
- **Cost**: the explainer call is $0.0002 more, from the longer prompt. Merging and
  snippets cost nothing.

### L5r on the test set with the final-list merge (a check only)

live-eval doesn't merge versions, so Step 4's test numbers come from unmerged lists.
`scripts/l5r_merge_check.py` measures the difference offline, at $0:
- it replays each of the 80 test checks' pools from the OpenAlex cache
  (`runs/live-20261015-000815`);
- takes its recorded L5r top 20, merges it as the tool now does (earliest-published
  version shown), and finds the original's rank again.

The replay reproduces every recorded L5r rank before the merge.

| 80 test queries | Hit@1 | Hit@5 | Hit@10 | MRR@20 |
|---|---|---|---|---|
| L5r as recorded (no merge) | 0.825 | 0.863 | 0.875 | 0.842 |
| L5r with the final-list merge | **0.838** | 0.863 | 0.875 | 0.848 |

- **Merges**: 9, in 9 checks.
  - **Only one moved an original**: A0211, from rank 2 to rank 1. Its original is the
    2026-09 article, and rank 1 was the same paper's 2026-03 preprint. The merge shows
    the preprint and lists the article as its alias, so the original now counts at
    rank 1.
  - **8 of the 9 are real versions**: preprints and their articles, conference papers
    and their journal versions, and two citation records with garbled titles.
  - **One is probably a false merge**: "NTCIR-4 Patent Retrieval Experiments at
    RICOH" (2004) and "NTCIR-5 …" (2015). The title rule drops bare numbers, and the
    group reused most of its abstract. It is outside the top 5 and isn't an original.
- **Hit@5 doesn't change**, so Step 4's conclusions stand.

## Spend

| Ledger | Spent | Total / cap |
|---|---|---|
| `data/cache/llm.sqlite` (OpenRouter: Mistral, and Qwen in the first pilot) | **$0.1855**: $0.0942 for the first run and its pilots, $0.0913 for the check of the fixes | $6.26 of $12 |
| OpenAlex (free daily budget) | $0.0091: 5 searches in the first pilot, 41 keyword lookups | – |

- **Within the declared caps**: the first run's $0.50 (`run.budget_cap` 6.57, the
  ledger's $6.0745 plus $0.4955), and the check's $0.30 (6.46).
- **By run**:
  - the first pilot: $0.0064, including the agent and rerank calls its fresh
    searches needed;
  - the second pilot: $0.0045;
  - the run: $0.0833 of explainer calls (19 new) and $0.0019 of keyword lookups.

## Deviations

1. **The keywords come from an extra OpenAlex call**, one list call per check
   ($0.0001). Asking for them in the searches would invalidate every cached search.
2. **The vocabulary map is grouped by source** (keywords, titles, the agent's
   queries), not by meaning ("also called", "related method"): without an LLM, the
   source is what the code knows. A term must be used by at least one top paper, so
   a query term no paper uses is dropped.
3. **The quote check also requires 5 words**, and ignores quotation marks around the
   whole quote. Only whitespace and the form of quotation marks are normalised, as
   specified. In this run, the 5-word rule held back one verbatim 4-word quote.
4. **The $0.03 cap applies to `rk check`.** live-eval keeps its $0.10, so runs
   recorded under it can still be resumed.
5. **The evaluation used the dev run's work types**, so that its checks replay from
   the caches (see The evaluation).
6. **Time is estimated** from recorded times plus the measured explainer stage, since
   the searches came from the caches; 19 of the 20 explainer calls were real.
7. **Small additions**:
   - `--before DATE` on `rk check`, the spec's optional input;
   - a paper's aliases in the report schema, for matching;
   - Pydantic declared in `pyproject.toml`, where it was already installed through
     mlflow.

## Open questions

**From the first run, answered on 2026-10-15:**
1. **Quotes**: the prompt fix plus the 90% match, as options (b) and (c), keeping the
   5-word minimum. Done: Product fixes, item 1.
2. **An independent check of "high"**: the hand review found 0 false alarms in 50
   rows, and 4 of A0049's 5 high papers are real prior art.
3. **Vocabulary**: end phrases at function words. Done: item 2.

**From the product fixes, answered after PR #17:**
1. **Overlap calibration** (5 of 50 levels one step off): accepted, with no tuning.
2. **The merge** keeps the earliest-published version, with the others as aliases.
   Done (`rk.live.pipeline.merge_versions`).
3. **live-eval and the merge**: checked once, offline, on the test set, without
   changing live-eval. Hit@1 is 0.838 instead of 0.825, and Hit@5 is unchanged at
   0.863 (see above).
