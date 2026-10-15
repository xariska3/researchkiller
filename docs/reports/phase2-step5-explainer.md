# Phase 2 Step 5: what a check shows for its top papers

Every check now returns its top 10 papers with an explanation a researcher can check
at a glance:
- **a snippet**: the abstract sentence closest to the idea;
- **an overlap explanation** from one LLM call: overlap level, a verdict per facet,
  a quote from the abstract, and what differs;
- **a vocabulary map** of the terms the literature uses that the idea doesn't.

`python -m rk check --config configs/phase1.yaml --file idea.txt [--quick]` prints
the report and saves it as JSON under `runs/check-<timestamp>/`. The evaluation below
is report only: nothing was tuned on it.

## Summary

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
- **Hand review**: [docs/evaluation/explainer_review.csv](../evaluation/explainer_review.csv)
  has 50 rows to fill in: the top 5 papers of 10 checks.

## What a check does now

| Step | Default check | Quick check (`--quick`) |
|---|---|---|
| Retrieval | L5r (Step 4) | L1: one OpenAlex semantic search, reranked locally |
| Snippets (top 10) | yes | yes |
| Keyword lookup (top 10) | yes | yes |
| Explainer call (top 10) | yes | no |
| Vocabulary map | keywords, titles, the agent's queries | keywords, titles |
| LLM calls | agent, rerank, explainer | none |

- **Snippet** (`rk.live.explain.snippets`): the abstract is split into sentences by
  rule (not after "et al.", "e.g." or an initial). Each sentence is scored against
  the idea with the check's Qwen3-Embedding encoder, the idea encoded as a query.
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
- **Quote check**: the evidence must be in the abstract verbatim, after normalising
  only whitespace and the form of quotation marks. Quotation marks around the whole
  quote are ignored, and a quote must be at least 5 words long.
  - If it fails, the quote is dropped and the paper is marked unverified, and its
    snippet is shown.
- **Fallbacks**: if the explainer call fails, could cross the cap, or can't be read,
  the papers keep their snippets and the check carries a warning. The whole check
  never fails at this stage. A missing or invalid entry costs only its own paper.
- **Vocabulary map** (`rk.live.vocab`): rule-based, with no LLM.
  - **Sources**: the top papers' OpenAlex keywords, their titles, and the agent's
    queries.
  - **Terms**: a phrase is a run of words between stopwords and punctuation. It
    counts if it has a content word the idea lacks, and at least one top paper uses
    it.
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
the first 10 checks (A0045 to A0058), the top 5 papers of each.
- **Columns**: qid, idea, rank, title, overlap, verified, evidence, differs.
- **Evidence**: the verified quote, or "[snippet] …" where the quote failed.
- **Empty columns to fill in**: correct?, useful?, false alarm?.

## Spend

| Ledger | Spent | Total / cap |
|---|---|---|
| `data/cache/llm.sqlite` (OpenRouter: Mistral, and Qwen in the first pilot) | **$0.0942** | $6.17 of $12 |
| OpenAlex (free daily budget) | $0.0071: 5 searches in the first pilot, 21 keyword lookups | – |

- **Within the declared $0.50 cap**: every run had `run.budget_cap` 6.57, the
  ledger's $6.0745 plus $0.4955.
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

1. **Quote verification at 84%**: all 32 failures are lightly edited real sentences.
   - **(a) Keep the rule as it is.** Those papers show their snippet.
   - **(b) Change the prompt**: copy the sentence from its first word, keeping any
     opening clause and markup. This would be measured on new dev queries, and
     probably fixes some but not all.
   - **(c) Match the quote to the abstract's sentences.** Find the sentence it comes
     from (one that has at least 90% of the quote's words) and show that sentence,
     verbatim. Every quote shown would still be the abstract's own text.
     - **In this run** it would recover 28 of the 32, computed offline from the run's
       reports.
     - **But it relaxes your "verbatim substring" rule**, so it is your decision.
2. **An independent check of "high"**: the 16 / 16 is expected, since Mistral built
   the benchmark. The hand review's "correct?" and "false alarm?" columns are the real
   test, especially A0049's 5 high papers.
3. **Vocabulary noise**: should phrases end at "using", "based" and "via"? This would
   be a later change, judged on new queries.
