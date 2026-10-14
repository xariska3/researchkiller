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
- **Cost and speed:** median under $0.03 and under 60 s per check; hard cap of
  $0.10 per check.
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
- Explainer check: run 10 full checks, export a CSV for hand review (results
  useful? explanations correct?), and report the quote-verification rate.
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

**Landscape doc** (docs/landscape.md): what each existing tool does (Elicit,
Consensus, Undermind, Semantic Scholar, Google Scholar Labs, ChatGPT/Claude deep
research, Perplexity; research systems: Idea Novelty Checker, OpenScholar,
PaperQA2, AI Scientist's novelty check), price, speed, and where ResearchKiller
is different. Candidate differentiators to confirm or reject with the numbers:
prior art found under different terminology, facet-level overlap with verified
quotes, vocabulary map, measured recall, ~1 cent and under a minute per check,
no fabricated citations.

## Steps (one Claude Code session each, each on its own branch + PR)
1. Live retrieval: OpenAlex search client (checked against current docs),
   candidate pool, local rerank, B5 loop on the live pool. Mocked tests.
2. Live eval: pilot on 30 queries, then all 500. Report. *(Decided in Step 2:
   after the pilot, up to 200 queries from A0045 onward, as the OpenAlex
   budget allows; see Evaluation.)*
3. Competitive evaluation: landscape doc, automated head-to-head on 100
   queries, manual comparison sheet for the author.
4. Overlap explainer + quote verification + vocabulary map + report schema.
   Tests with FakeLLM. Hand check on 10 checks.
5. FastAPI + SQLite + background worker + web page. Tests with TestClient.
6. User README ("how to run, what it does, what it doesn't"), screenshots,
   demo. Optional: Dockerfile built and tested in GitHub Actions.

## Product ideas (not built yet)
- **Quick check vs thorough check** (from the Step 2 pilot): a "quick check"
  mode is L1: one OpenAlex semantic call and the local rerank, with no LLM, in
  about 1–2 s. It would sit next to the "thorough check", L5, the full
  pipeline. In the pilot, L1 had Hit@5 0.700 against L5's 0.833.

## Out of scope for Phase 2
Docker as a requirement, user accounts, scoop alerts, `.bib` missing-citation check, the MCP server,
full-text PDFs, hosting on the public internet. These are candidates for Phase 3.

## Known limitations to state
- Coverage is limited to OpenAlex works with abstracts.
- Explanations come from an LLM; quotes are verified, interpretations are not.
- A "no close match" result is not evidence of novelty.