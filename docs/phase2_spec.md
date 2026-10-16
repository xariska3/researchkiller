# Phase 2 spec: ResearchKiller as a live tool for researchers

## Goal
A researcher pastes an abstract or idea description and, in under a minute and
for a few cents, gets back:
1. the closest existing papers from the whole literature (live OpenAlex),
2. for each one, what overlaps and what differs (problem, method, domain, claim),
   backed by a sentence quoted verbatim from that paper's abstract,
3. a vocabulary map: other names the idea goes by in the literature.

The tool never declares an idea "novel". It reports the closest prior work it
found and how close it is.

Phase 1 already decided the search design (docs/reports/full-run.md): an agent
with a capped budget (B5 settings) over hybrid retrieval. Phase 2 reuses that
design live, and keeps the Phase 1 benchmark as a regression test.

## Success criteria (definition of done)
- **Live recall:** on the 500 Phase 1 queries, the live tool finds the original
  paper in its top 10 at least as often as B5 did on the frozen corpus (0.90),
  or we report honestly how far below it is and why.
- **Explanations are grounded:** 100% of quoted evidence sentences are verbatim
  substrings of the cited abstract (checked in code, not by the LLM).
- **Hand check:** 10 real reports read by the author; record for each whether
  the top results and overlap explanations were useful.
  - *Later note (2026-10-16): the hand check was done on 10 benchmark checks
    (the top 5 papers of each, 50 rows), reviewed by Claude rather than the
    author; see [the Step 5 report](reports/phase2-step5-explainer.md).*
- **Cost and speed:** median under $0.03 and under 60 s per check; hard cap of
  $0.10 per check.
  - *Later note (2026-10-16): the tool caps a whole check at $0.03
    (`explain.cost_cap_per_check`); $0.10 stays the cap of live-eval's checks.*
- **Competitive check:** on the same queries, compared with ChatGPT and Claude
  with web search, Perplexity and Semantic Scholar on recall, fabricated
  references, cost and speed (see "Competitive evaluation").
- A web page + API that runs locally with one command.

## Pipeline (per check)
1. **Input:** idea text, 40–500 words. Optional: a "published before" date, so
   the user can also test an old idea fairly.
2. **Query planning:** reuse the B5 agent loop (same prompt family, same caps:
   max 5 queries, 6 LLM calls), but its search tool is live retrieval (step 3)
   and it sees the top-10 titles from the live pool.
3. **Live retrieval:** for each query, OpenAlex `/works` search with
   `has_abstract:true`, about 50 results per query. Check the current OpenAlex
   docs for a semantic or vector search option; if one exists, also use it and
   merge. Dedupe by id/DOI/title into a candidate pool (typically 100–250).
   Cache every OpenAlex response on disk.
   *Step 1 finding:* OpenAlex has semantic search (`search.semantic`: at most
   50 results, 1 request/s, $0.001 per call, the same price as keyword search).
   Keyword search ANDs every word, so queries longer than
   `live.keyword_max_words` (32; in practice the user's text) get semantic
   search only. Details in `src/rk/live/openalex_search.py`.
   *Step 1 decisions:*
   - **Work types:** keep `article`, `preprint`, `conference-paper`,
     `book-chapter`, `dissertation`, `review`, `book`, `report` and (since
     Step 2) `software-paper` (`live.work_types`; a survey, tech report or
     software paper on the same idea is prior art too). Drop `dataset`, `erratum`, `peer-review`, `paratext`, `letter`,
     `editorial` and other non-papers. Searches send a `type:` filter, and any
     other type that comes back is dropped locally.
   - **Near-duplicates:** besides Phase 1's dedupe (shared id or DOI, or the
     same long title at most 2 years apart), two records merge only if all
     three hold: title Jaccard >= 0.8 on content stems, at least one shared
     author, and publication years at most 1 apart. Authors are the same if
     they share an OpenAlex author id, or if every token of the shorter name
     (2 tokens or more) is in the other name, since an author id can be
     missing. The earliest version keeps its id; the others become aliases.
   - **Cache:** every OpenAlex response is kept. The live tool refetches a
     response older than `live.cache_ttl_days` (7). Evaluation runs read the
     first response ever cached and never expire, so they are reproducible.
4. **Rerank locally:** BM25 + the Qwen3-Embedding dense score over the
   candidate pool, fused with the same RRF function as Phase 1 (hybrid,
   because Phase 1 showed rewrites do worse on a dense-only base). Keep the top 20.
   *Steps 2 and 4 results:* the final fusion is L5w's (proposal-equal). Then one
   call to the agent model reranks the union of L5w's and L5's top 30 listwise
   (R1, giving L5r; see Evaluation).
5. **Overlap explainer (top 10):** reuse the validated facet judge (Mistral Medium 3.1),
   reframed: compare the user's idea with each paper facet by facet (problem,
   method, domain, claim). Output per paper, as strict JSON:
   overlap level (high / partial / low), per-facet same/different with a short
   reason, one evidence sentence copied from the abstract, and "what differs"
   in one line. Code verifies the evidence sentence is a verbatim substring;
   if not, drop the quote and mark the item "unverified".
6. **Vocabulary map:** terms from the agent's queries and from the top papers'
   titles/keywords that don't appear in the user's text, grouped (for example
   "also called", "related method"). Deterministic where possible; at most one
   cheap LLM call.
7. **Report:** ordered by overlap level then rank. Header line: "Closest prior
   work found", plus the coverage note "searched N candidates from OpenAlex;
   absence of a match is not proof of novelty".

## Architecture
- `src/rk/live/`: `openalex_search.py`, `pool.py` (dedupe, rerank),
  `explain.py`, `vocab.py`, `pipeline.py` (runs one check end to end, with a
  per-check cost ledger and the $0.10 cap).
- **API (FastAPI):**
  - `POST /api/checks` {text, before_date?} → {check_id}; work runs in the background.
  - `GET /api/checks/{id}` → status (queued/running/done/failed), progress stage,
    and the report when done.
  - `GET /` → one simple HTML page (Jinja2 template, plain JS polling, no
    frontend framework): text box, submit, live progress, report rendering.
- **Background jobs:** a single worker thread with a queue, kept minimal.
  Persist job state so a restart doesn't lose finished reports.
- **Database (SQLite via SQLAlchemy):** tables `checks` (id, text hash, status,
  timings, cost, model versions, prompt hashes), `candidates` (check_id, paper
  id, ranks, scores), `explanations` (check_id, paper id, overlap level, facets
  JSON, quote, verified flag). Same database as the LLM cache/spend file is fine.
- **Schemas:** Pydantic models for the API and for the explainer output.
- **Config:** a `live:` section in the YAML (results per query, pool size,
  top-k to explain, per-check cost cap, timeouts). *Step 1:* the section is in
  `configs/phase1.yaml`. The cap is on a check's method cost (OpenAlex plus
  LLM), and a call that could cross it fails the check.
- **Embeddings:** GPU if available, CPU fallback; record which one in the check.

## Evaluation
- `python -m rk live-eval --limit N`: runs steps 2–4 (no explainer) on Phase 1
  queries, live, and reports Hit@1/5/10 and MRR of the original paper in the
  whole literature, plus cost and latency per check. Writes to the results database and
  MLflow like Phase 1 runs.
  *Step 1 decision (matching the original):* a result is the original paper if
  any of its merged versions has the original's OpenAlex id or one of its
  aliases, its DOI, or its exact normalised title (`rk.live.pool.match_original`).
  *Step 2 decisions, pre-declared on 2026-10-14 before the full run:*
  - **Configs**: L0 is OpenAlex semantic search on the proposal alone, in
    OpenAlex's own order (1 call, no LLM). L1 is the same results reranked
    locally (hybrid BM25 + dense, RRF). L5 is the full live pipeline (steps
    2–4). Ranks count within the top `live.top_k` (20), so MRR is MRR@20.
  - **Primary comparison**: L5 vs L1. Hit@5 with the exact McNemar test
    decides; MRR supports (paired bootstrap 95% CIs, as in Phase 1).
  - **Secondary comparisons (exploratory)**: L1 vs L0, L5 vs L0, and L5 vs
    Phase 1's B5 on the frozen corpus, on the same queries (`live.phase1_run`).
  - **Queries**: the full run uses the first N queries in seeded order from
    A0045 onward, which leaves out Phase 1's pilot and tuning targets. N is what
    fits in the day's OpenAlex budget (about $0.0061 per query, keeping a
    $0.05 margin), up to 200. A resume with `--run-dir` continues toward 200
    when the budget is topped up or resets.
  - **Pilot**: the 30-query pilot (A0000–A0038, all Phase 1 tuning targets) is
    reported separately from the full run.
  - **Cost cap**: a check over its cost cap counts as a miss for live recall,
    and the number of such checks is reported separately.
  *Final-fusion variants, pre-declared on 2026-10-14 before evaluating them:*
  - **Dev set**: the 120 complete queries of `runs/live-20261014-122144`
    (A0045–A0186).
  - **Offline replay**: every check is replayed from its cached OpenAlex
    responses (no new calls), with the agent's queries taken from the stored
    trace (no new LLM calls). The agent loop and the final pool are L5's; only
    the final ranking changes.
  - **Variants**: n is the number of queries run, the proposal first. Each
    query's ranking is the pool's hybrid ranking of the final pool, as in L5.
    - **F1, proposal-equal**: weighted RRF. The proposal's ranking has weight
      n − 1, equal to all agent queries combined (1 if the agent ran none),
      and each agent query has weight 1.
    - **F2, proposal-2x**: weighted RRF, with weight 2 for the proposal and 1
      for each agent query.
    - **F3, l1-l5**: RRF of two lists, L1's (the proposal's own search,
      reranked; mapped onto the final pool's merged works) and L5's final list.
    - **F4, proposal-only**: the final pool ranked by the proposal alone.
  - **Selection**: the variant with the highest Hit@5 on the dev set becomes
    **L5w**, if its Hit@5 is above L5's. Ties go to the higher MRR@20, then to
    the order above. If no variant beats L5, L5 stays and there is no L5w. The
    dev-set comparisons with L5 and L1 are reported, but they are exploratory,
    since this is the set the choice is made on.
  - **Test set**: the next 80 queries after the dev set in seeded order (from
    A0187), run live with L1, L5 and L5w into a new run directory, sized to the
    day's OpenAlex budget. On it, L5w vs L5 and L5w vs L1 are paired: Hit@5
    with the McNemar test decides, and MRR supports.
  - **Work types**: `software-paper` joins `live.work_types` for the test run.
    One dev-set original was a software paper the filter dropped.
  - **Selection result (2026-10-14)**, from `python -m rk live-fusion` on the
    dev set. The replay reproduced all 120 recorded L5 and L1 rankings.
    - **Hit@5**: L5 0.700, F1 0.750, F2 0.750, F3 0.717, F4 0.667.
    - **Choice**: F1 and F2 tie on Hit@5, and F1 wins the MRR@20 tie-break
      (0.617 against 0.594), so **L5w = F1, proposal-equal**.
    - **Against L5 on the dev set**: +0.050 Hit@5 (7 queries gained, 1 lost,
      McNemar p = 0.07). This is exploratory; the test set decides.
  - **Test result (2026-10-15)**: L5w vs L5 +0.013 Hit@5 (4 / 3, p = 1), not
    confirmed, but L5w isn't worse. L5w stays the default check
    (docs/reports/phase2-step2-test.md).
  *Rerank variants (Step 4), pre-declared on 2026-10-15 before evaluating them:*
  - **Why**: the original is sometimes in the final pool but ranked below the
    top 5. In Step 3, 2 of the 3 examples where other systems found the original
    and ours didn't were ranking misses.
  - **Prompts**: `src/rk/prompts/rerank_listwise.txt` and `rerank_graded.txt`,
    committed with the code before any call.
  - **Offline, on cached checks**: every check is replayed as for the fusion
    variants (cached OpenAlex responses, the agent's recorded queries), so the
    pools and the L5 and L5w rankings are the recorded ones. Only the reranker's
    LLM calls are new.
  - **Candidates**: the union of L5w's and L5's top 30 in the final pool, so
    originals that only the agent's queries ranked high aren't lost. They are
    taken in turn, L5w's first, then L5's first, and so on, skipping works
    already taken. This is the *presented order*, and the candidates are
    numbered in it. A pool with fewer works gives fewer candidates.
  - **What the reranker sees**: the proposal, and for each candidate its number,
    title and abstract. Abstracts are cut at 2,000 characters, at a word
    boundary (6.6% of corpus abstracts are longer). No years, authors or venues.
  - **Product variants**: the agent model (`qwen/qwen3-235b-a22b-2507`, on the
    agent's pinned endpoint, temperature 0, seed 42, at most 1,000 output
    tokens). Its training cutoff (2025-07-21) is before every original.
    - **R1, listwise**: one call. The model returns every candidate number,
      closest first, as JSON; the paper that describes the same research idea
      (the same problem, approach and main claim) comes first. The ranking is
      that order. Unknown or repeated numbers are dropped, and missing ones
      follow in presented order.
    - **R2, listwise + first stage**: R1's order fused with the presented order
      by RRF (k = 60, equal weights), so a candidate the first stage ranked high
      isn't lost to one bad LLM judgment. It uses R1's calls, with no new ones.
    - **R3, graded**: one call. The model gives every candidate a score as JSON:
      - 3 = the same research idea;
      - 2 = the same problem and a closely similar approach;
      - 1 = the same broad topic;
      - 0 = unrelated.

      Candidates are sorted by score, ties in presented order. A missing score
      counts as 0.
    - **Unreadable replies**: a reply with no readable JSON leaves the
      presented order. Such replies are counted and reported.
    - **The rest of the list**: L5w's works outside the candidates follow the
      reranked ones. Ranks count within the top 20, so MRR is MRR@20, as before.
  - **Upper bound, RS**: R1's prompt and parsing, with Claude Sonnet 5.5
    (Anthropic's OpenAI-compatible endpoint). It is not a product candidate and
    can't be selected.
    - **Settings**: temperature 1 (Claude 5 models refuse less), its default
      adaptive thinking, and at most 16,000 output tokens.
    - **Cap**: $20, in its own ledger
      (`data/cache/rerank_sonnet.sqlite`).
    - **Memorisation**: Sonnet 5.5's training data runs to June 2026, so it may
      have read originals published before July. Its results are therefore
      also reported for originals published from 2026-07-01, which it can't
      have seen.
  - **Pilot**: 3 queries of the Step 2 pilot run (`runs/live-20261014-114302`,
    Phase 1 tuning targets, neither dev nor test), with every variant. It
    checks parsing, tokens, cost and latency. If RS would cost more than $20
    on the 200 queries, stop and report.
  - **Selection, on the dev set** (the 120 queries of
    `runs/live-20261014-122144`):
    - The product variant with the best Hit@5 becomes **L5r**, if its Hit@5 is
      above L5w's. Ties go to the higher MRR@20, then to the order R1, R2, R3.
    - If no variant beats L5w, there is no L5r and L5w stays.
    - The dev-set comparisons are exploratory.
  - **Test, on the test set** (the 80 queries of `runs/live-20261015-000815`):
    - **Decides**: L5r vs L5w, with ΔHit@5 and the exact McNemar test at
      α = 0.05. MRR@20 supports, with a paired bootstrap 95% CI.
    - **If significant and positive**: the reranker joins the default check.
    - **If positive but not significant**: reported as not confirmed, and the
      decision is left to the author.
    - **Exploratory**: the other variants and RS against L5w, RS against L5r,
      and RS on the originals from 2026-07-01.
  - **Reported on both sets**:
    - Hit@1/5/10 and MRR@20 for L5, L5w, R1–R3 and RS;
    - the share of queries whose original is among the candidates (the
      ceiling of any reranker);
    - the number of candidates;
    - unreadable replies;
    - the reranker's cost per check (mean, median, p95, max) and the whole
      check's cost against the $0.10 cap;
    - the rerank call's latency.
  - **Spend**: the R1–R3 calls go to `run.budget_cap`'s ledger
    (`data/cache/llm.sqlite`), at about $1 for 400 calls. RS has its own cap.
  - **Pilot result (2026-10-15)**, `runs/rerank-20261015-131744`: every reply
    was readable. A check cost $0.0014 more with Qwen, which adds 5–9 s. RS cost
    $0.031 a check, about $6 for the 200, so it went ahead.
  - **Selection result (2026-10-15)**, from `python -m rk rerank` on the dev
    set (`runs/rerank-20261015-132644`). The replay reproduced all 120 recorded
    L5 rankings.
    - **Hit@5**: L5 0.700, L5w 0.750, R1 0.850, R2 0.808, R3 0.850, and RS
      0.858 (not eligible).
    - **The ceiling**: the original was among the candidates in 103 of 120
      checks (0.858).
    - **Choice**: R1 and R3 tie on Hit@5, and R1 wins the MRR@20 tie-break
      (0.842 against 0.758), so **L5r = R1, listwise**.
    - **Against L5w on the dev set**: +0.100 Hit@5 (13 queries gained, 1
      lost, McNemar p = 0.002). This is exploratory; the test set decides.
  - **Test result (2026-10-15)**, `runs/rerank-20261015-140923`.
    - **L5r vs L5w**: Hit@5 0.863 against 0.700, +0.163 (13 / 0, p = 0.0002),
      and MRR@20 +0.228. This is confirmed.
    - **So**: the default check becomes L5r (L5w, then R1's rerank) once the
      live pipeline has it.
    - **RS** (0.875) is one query ahead of R1.
    - **Report**: docs/reports/phase2-step4-rerank.md.
- Explainer check: run 10 full checks, export a CSV for hand review (results
  useful? explanations correct?), and report the quote-verification rate.
  *Step 5 decisions, declared on 2026-10-15 before any explainer call:*
  - **The command**: `python -m rk explain-eval` runs whole default checks with
    the explainer (`rk.live.check.run_check`) on the **first 20 dev queries**
    in seeded order, from A0045. Report only: nothing is tuned on it.
  - **Caches**: the searches read OpenAlex's cache as an evaluation does, so
    the searches and the agent and rerank calls replay Step 2's and Step 4's
    dev checks. The keyword lookups and the explainer calls are new.
  - **Measures**:
    - **the quote-verification rate**: explained papers whose evidence
      sentence is in the abstract verbatim (whitespace and quotation marks
      normalised, at least 5 words);
    - **the original's overlap level** when it is among the explained top 10
      (it should be "high");
    - **cost per check**, with the explainer: median, p95 and max;
    - **time per check**, estimated as the recorded check time (the dev
      live run's L5w row), plus the recorded rerank call (Step 4's dev rerank
      run), plus this run's snippet, keyword, explainer and vocabulary stages;
    - **fallbacks**: the rerank and explainer steps, partial explanations and
      failed keyword lookups.
  - **Caveat**: every benchmark proposal was accepted by the same Mistral
    model, as keeping all four facets of its original. So "the original is
    high" is expected, and is a consistency check, not an independent one.
  - **Hand review**: `docs/evaluation/explainer_review.csv` holds the first
    10 of the 20 checks, one row per top-5 paper, with empty columns for
    correct?, useful? and false alarm?.
  - **Spend cap**: $0.50 of new LLM spend; the expected spend is about $0.12.
  - **Result (2026-10-15)**, `runs/explain-20261015-212527`:
    - **Quotes**: 168 of 200 verified (84.0%). The other 32 are all edited real
      sentences: a lead-in dropped, the end cut off, or a source artefact fixed.
    - **The original**: rated high in 16 of 16 checks that had it in the top 10.
    - **Cost**: median $0.0133 a check (max $0.0161).
    - **Time**: about 31 s estimated live (median).
    - **Fallbacks**: none.
    - **Run settings**: the run used the dev run's `live.work_types` (from before
      `software-paper`), so that its searches replay from the cache.
    - **Report**: docs/reports/phase2-step5-explainer.md.
  - **Hand review (2026-10-15)**: `docs/evaluation/explainer_review.csv`, 50
    rows, reviewed by Claude (not the author).
    - **False alarms**: 0 / 50.
    - **Correct**: 40 yes, 9 partly, 1 no.
    - **Useful**: 31 yes, 17 partly, 2 no.
  - **Product fixes, decided on 2026-10-15** after looking at those 20 dev
    checks and the hand review, so they are labelled as such:
    1. **Quotes**:
       - The prompt asks for the evidence sentence exactly, from its first word.
       - The code shows the abstract's own sentence, never the explainer's
         version: verified-exact if the quote is in it verbatim, verified-matched
         if it has at least 90% of the quote's words, otherwise unverified (the
         snippet).
       - The 5-word minimum stays.
    2. **Vocabulary**: phrases end at the function words using, based, via,
       with, for, by and through, and a phrase ending in one is dropped.
    3. **"Differs"**: either a substantive difference, or exactly "No
       substantive difference found." It is never publication status, or the
       proposal being a plan.
    4. **Sentences**: a full stop followed directly by a capital ends a sentence
       ("datasets.We"). A snippet is cut at about 350 characters, at a word,
       with "…".
    5. **Near-duplicates in a check's final list**: two results merge when their
       titles' content stems have a Jaccard of at least 0.7 and their abstracts
       share most sentences, with or without a shared author.
       - **Shared sentences**: more than half of the shorter abstract's
         sentences, where a sentence is shared if the other abstract has one
         with a word-set Jaccard of at least 0.5.
       - **Why not exact**: the example (A0053, ranks 4 and 5) rewords every
         sentence a little.
       - **The merge**: the entry keeps the earlier place in the list. *Changed
         after PR #17*: it shows the earliest-published version, and the others
         become its aliases. live-eval doesn't merge.
       - **On the test set** (a check only, offline): it changes L5r's Hit@1
         from 0.825 to 0.838, and leaves Hit@5 at 0.863
         (`scripts/l5r_merge_check.py`).
  - **Check of the fixes, declared before the run**: `explain-eval` on the next
    20 dev queries in seeded order, from A0071, so none of the first 20.
    - **Settings**: the dev run's work types and the timing run, as before.
    - **Report**: the exact, matched and unverified rates of the quotes, and the
      cost per check, with the first run's measures alongside.
    - **Review CSV**: under runs/, since the hand-reviewed one stays as it is.
    - **Spend cap**: $0.30, against an expected $0.10.
    - **Result (2026-10-15)**, `runs/explain-20261015-215122`:
      - **Quotes**: 95.5% exact, 4.0% matched, 0.5% unverified (191, 8 and 1
        of 200).
      - **The original**: in the top 10 and rated high in 20 of 20.
      - **Cost**: median $0.0136 a check (max $0.0162).
      - **Merges**: 4, all real versions of one paper.
      - **"Differs"**: none cites publication status or "the proposal is a
        plan".
      - **Spend**: $0.091.
- The Phase 1 frozen benchmark stays runnable as a regression test.

## Competitive evaluation: is it better than what researchers already use?
The question a researcher asks is "why not just ask ChatGPT?". We answer it
with numbers.

**Automated head-to-head** (100 Phase 1 queries, stratified by the Phase 1
difficulty buckets, same prompt for every system: "Find the existing papers
most closely related to this research proposal. List up to 10, each with
title, year and link."):
- ResearchKiller (live tool)
- an OpenAI model with its web search tool
- a Claude model with its web search tool
- Perplexity Sonar (API)
- Semantic Scholar search API (plain academic search baseline, free)

Check current APIs and prices first (via OpenRouter where possible, otherwise
direct). Metrics per system:
- **Recall:** is the original paper in the list (match by DOI/OpenAlex id, else
  normalised title)? Report Hit@1/5/10.
- **Fabricated references:** share of listed papers that don't exist (each
  title/DOI looked up in OpenAlex). Ours is 0 by construction; measure the others.
- **Cost and latency** per query.
Same paired statistics as Phase 1. Report every system, whatever the outcome.

**Manual spot check** (2–3 of the same queries, by hand, as qualitative
examples): the ChatGPT and Claude apps in deep-research mode (fresh temporary
chat, memory off, web search on), plus Undermind/Elicit/Consensus where a free
tier allows. Record whether the original paper appears, time taken, and what
each tool offers that ours doesn't. No scraping or automation of these tools.
Before the automated run, check current prices and set a per-system cost cap.

**Landscape doc**: what each existing tool does (Elicit,
Consensus, Undermind, Semantic Scholar, Google Scholar Labs, ChatGPT/Claude deep
research, Perplexity; research systems: Idea Novelty Checker, OpenScholar,
PaperQA2, AI Scientist's novelty check), price, speed, and where ResearchKiller
is different. Candidate differentiators to confirm or reject with the numbers:
prior art found under different terminology, facet-level overlap with verified
quotes, vocabulary map, measured recall, ~1 cent and under a minute per check,
no fabricated citations.

*Later note (2026-10-16): the landscape doc wasn't written; Step 3 ran the
automated head-to-head only.*

## Steps (one Claude Code session each, each on its own branch + PR)
1. Live retrieval: OpenAlex search client (checked against current docs),
   candidate pool, local rerank, B5 loop on the live pool. Mocked tests.
2. Live eval: pilot on 30 queries, then all 500. Report. *(Decided in Step 2:
   after the pilot, up to 200 queries from A0045 onward, as the OpenAlex
   budget allows; see Evaluation.)*
3. Competitive evaluation: landscape doc, automated head-to-head on 100
   queries, manual comparison sheet for the author.
4. Rerank: up to three cheap LLM rerank variants and a Sonnet upper bound,
   evaluated offline on the cached checks (chosen on dev, confirmed on test).
   *(Added on 2026-10-15, before the explainer; see Evaluation.)*
   *Built into the live pipeline on 2026-10-15, after PR #15:*
   - **The default check**: `rk.live.pipeline.default_check` is L5r, L5w's check
     followed by R1's listwise call (`with_rerank`). The call is charged to the
     check, under its $0.10 cap.
   - **Fallback**: if the call fails, could cross the cap, or its reply has no
     readable ranking, the check keeps L5w's order and carries a warning in the
     result (`warnings`). Step 4's offline R1 kept the presented order for an
     unreadable reply, but none was unreadable there.
   - **Evaluation**: live-eval has an L5r config. As for agent searches, it is
     strict: a failed rerank call is an error row. An unreadable reply or a call
     over the cap falls back as in the tool, and the summary counts these
     fallbacks.
5. Overlap explainer + quote verification + vocabulary map + report schema.
   Tests with FakeLLM. Hand check on 10 checks.
   *Built on 2026-10-15* (`rk.live.check`, `rk.live.explain`, `rk.live.vocab`,
   `rk.live.report`; `python -m rk check --file idea.txt [--quick]`):
   - **Snippets**: every top-10 paper gets the abstract sentence closest to
     the idea (Qwen3-Embedding), in the quick check too.
   - **The explainer**: one call to the judge's model (Mistral Medium 3.1,
     `explain.model`), strict JSON per paper. A quote not found in the abstract
     is dropped, and the paper is marked unverified. A failed or unreadable
     call leaves the snippets, with a warning.
   - **The vocabulary map**: rule-based, with no LLM. Terms come from the top
     papers' OpenAlex keywords (one extra list call by id, $0.0001), their
     titles and the agent's queries, grouped by source and capped at 10.
   - **The cap**: a whole check of `rk check` is capped at
     `explain.cost_cap_per_check` ($0.03); live-eval keeps
     `live.cost_cap_per_check`.
6. FastAPI + SQLite + background worker + web page. Tests with TestClient.
   *Decided on 2026-10-15:* a failed or timed-out agent search is skipped with
   a warning instead of failing the whole check. Evaluation stays strict: there,
   a failed agent search is an error row.
   *Built on 2026-10-15* (`rk.web`; `python -m rk serve`):
   - **Checks**: the web app runs them through `rk.live.check.run_check`, as
     `rk check` does, one at a time in a background thread.
   - **Storage**: SQLite (`web.db_path`), tables `checks`, `candidates` and
     `explanations`.
   - **Reuse**: the same text, mode and date within `web.history_days` (7)
     reuses the stored report at $0.
   - **Daily cap**: `web.daily_cost_cap` ($1.00) counts money spent today
     (UTC), plus $0.03 held for each check in progress.
   - **Refusals**: OpenAlex's spent daily budget refuses new checks until it
     resets.
   - **Restarts**: unfinished checks are marked failed, never re-run.
   - **Agent searches**: the tool (`skip_failed_searches`) skips a failed one
     with a warning; `strict` runs (evaluation) raise.
   - **Venues**: they come from the same list call as the keywords
     (`LiveSearchClient.details`, now for all 20 papers).
7. User README ("how to run, what it does, what it doesn't"), screenshots,
   demo. Optional: Dockerfile built and tested in GitHub Actions.

**Backlog** (decided on 2026-10-15):
- **Retrieval improvements** are deferred until after the explainer and the
  web app. Of the originals missing from L5r's candidates, 24 of 27 on the dev
  and test sets (8 of 10 on the test set) never came back from any OpenAlex
  search.

## Product ideas (not built yet)
- **Quick check vs thorough check** (from the Step 2 pilot): a "quick check"
  mode is L1: one OpenAlex semantic call and the local rerank, with no LLM, in
  about 1–2 s. It would sit next to the "thorough check", L5, the full
  pipeline. In the pilot, L1 had Hit@5 0.700 against L5's 0.833.
  - *Later note (2026-10-16): the quick check is now built, as a mode of its
    own: `python -m rk check --quick`, and the web app's Quick option.*
  *Decided on 2026-10-15, after the L5w test run:*
  - **Default check**: L5w. The reranker joins it if it wins on the test set
    in the rerank evaluation that comes before the explainer. *(It did, on
    2026-10-15, so the default is L5r: L5w with R1's listwise rerank, built in
    as `rk.live.pipeline.default_check`; see Steps, 4.)*
  - **Preview**: L1 is shown at once, as a first preview, while the full check
    runs. L1 alone is not the default: it trailed L5w by 8.7 points of Hit@5 on
    the test set and by 11.7 on the dev set.
    - *Later note (2026-10-16): as built, the quick check is a separate mode the
      user picks; the full check doesn't show L1 first.*
- **Deep check** (pre-declared on 2026-10-15, after Step 3; not built yet): an
  opt-in mode that adds one web-search LLM call to the default check.
  *Decided on 2026-10-15, after the preview below:* it isn't built for now.
  - **Why**: in Step 3 the two kinds of system were complementary. Ours found
    39 of 41 originals in the easiest bucket, Opus 10 of 16 in the hardest.
    Ours' or the LLM's top 5 held the original on 87 of 100 queries with Opus
    and 85 with Sonnet, against 72 for ours alone (exploratory).
  - **The call**: Claude Sonnet 5.5 with web search (`max_searches` 10), given
    Step 3's prompt (`src/rk/prompts/compete.txt`). It runs in parallel with
    the default check. Sonnet is chosen over Opus for its cost, at about $0.16
    against $0.29 a query, with 85 against 87 of 100 together.
  - **Merging**: the two top-10 lists are interleaved, starting with ours'
    first paper.
    - A paper already listed is skipped. Papers are the same if they share an
      OpenAlex id, a DOI or a normalised title, as in
      `rk.live.pool.match_original`.
    - Every LLM paper goes through Step 3's free resolution routes. One that
      no route verifies is kept but marked "unverified". Ours' papers are
      OpenAlex records.
    - The merged list is cut to 10.
  - **Evaluation (later)**: the deep check (A) against the default check alone
    (B), paired on the same queries.
    - **Primary**: Hit@5 with the exact McNemar test.
    - **Also reported**: Hit@10, MRR@10, the share of unverified papers in the
      merged top 10, and cost and latency per check.
    - **Queries**: new ones, from A0288 onward in seeded order. The 200
      queries checked in Step 2 can be scored for free from the caches, but
      only as an exploratory preview, since this design came from looking at
      them.
  - **Cost**: about $0.17 a check, over the $0.10 hard cap of a default
    check, so the deep check needs a cap of its own.
  - **Exploratory preview on the test set** (requested on 2026-10-15, after
    Step 4; recorded here before any call). It decides nothing: 39 of these
    queries were in Step 3's sample, which the design came from.
    - **Queries**: the 80 test queries of `runs/live-20261015-000815`.
    - **The default check is now L5r**: its top 10 come from Step 4's test
      run (`runs/rerank-20261015-140923`, R1).
    - **Sonnet's answers**: from a compete run on the same 80 queries, with
      Step 3's settings. 39 are in Step 3's cache; 41 are new calls.
    - **Cap**: $11.61 of new spend, lower than the requested $15 (the compete
      group cap set to $88).
    - **Reported**: the deep check against L5r alone, on Hit@1, Hit@5 (with
      the exact McNemar test), Hit@10, MRR@10, the unverified share of the
      merged top 10, and cost and latency per check.
    - **Hit@1 is equal by construction**: the merged list starts with L5r's
      first paper.
    - **Result** (`runs/compete-20261015-162116`; see
      [the rerank report](reports/phase2-step4-rerank.md)): Hit@5 0.900
      against L5r's 0.863 (4 / 1, McNemar p = 0.375), at $0.166 a check and
      $6.45 of new spend. Nothing is decided.

## Out of scope for Phase 2
Docker as a requirement, user accounts, scoop alerts, `.bib` missing-citation check, the MCP server,
full-text PDFs, hosting on the public internet. These are candidates for Phase 3.

## Known limitations to state
- Coverage is limited to OpenAlex works with abstracts.
- Explanations come from an LLM; quotes are verified, interpretations are not.
- A "no close match" result is not evidence of novelty.