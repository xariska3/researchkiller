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
- A web page + API that runs locally with one command (Docker Compose).

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
- Explainer check: run 10 full checks, export a CSV for hand review (results
  useful? explanations correct?), and report the quote-verification rate.
- The Phase 1 frozen benchmark stays runnable as a regression test.

## Steps (one Claude Code session each, each on its own branch + PR)
1. Live retrieval: OpenAlex search client (checked against current docs),
   candidate pool, local rerank, B5 loop on the live pool. Mocked tests.
2. Live eval: pilot on 30 queries, then all 500. Report.
3. Overlap explainer + quote verification + vocabulary map + report schema.
   Tests with FakeLLM. Hand check on 10 checks.
4. FastAPI + SQLite + background worker + web page. Tests with TestClient.
5. Docker Compose (app + MLflow UI), README for users ("how to run, what it
   does, what it doesn't"), screenshots, demo.

## Out of scope for Phase 2
User accounts, scoop alerts, `.bib` missing-citation check, the MCP server,
full-text PDFs, hosting on the public internet. These are candidates for Phase 3.

## Known limitations to state
- Coverage is limited to OpenAlex works with abstracts.
- Explanations come from an LLM; quotes are verified, interpretations are not.
- A "no close match" result is not evidence of novelty.