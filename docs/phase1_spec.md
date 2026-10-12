# Phase 1 spec: terminology-shift retrieval benchmark

## Goal
Measure whether agentic query planning finds prior art better than cheaper
retrieval, when the query describes a paper's contribution in **different
terminology**. Phase 1 builds the benchmark and five retrieval configurations.
No critic, no API server, no deployment yet.

The key comparison is **B4 vs B5** at the same query budget.

## Repo layout (keep it this small)
```
researchkiller/
+-- CLAUDE.md
+-- README.md
+-- pyproject.toml
+-- .gitignore            # data/, runs/, .env, .venv/, __pycache__/, *.sqlite, mlruns/
+-- .github/workflows/ci.yml  # Python 3.10: ruff check, then pytest
+-- configs/phase1.yaml
+-- docs/phase1_spec.md
+-- src/rk/
¦   +-- __init__.py
¦   +-- config.py         # YAML -> dataclasses
¦   +-- llm.py            # OpenAI-compatible client, cache, cost ledger
¦   +-- text.py           # tokenising, stemming, stopwords, acronyms
¦   +-- corpus.py         # OpenAlex client, corpus build, dedupe, freeze
¦   +-- targets.py        # target selection
¦   +-- bench.py          # paraphrase generation + filters
¦   +-- retrieval.py      # B1-B4
¦   +-- agent.py          # B5
¦   +-- evaluate.py       # runner, metrics, paired statistics
¦   +-- cli.py            # CLI, run as `python -m rk`
¦   +-- __main__.py       # enables `python -m rk`
+-- tests/
    +-- conftest.py       # FakeLLM, FakeEncoder, tiny fixture corpus
    +-- test_*.py
```
`data/` and `runs/` are created at runtime and git-ignored.

## Dependencies
Core: `httpx`, `numpy`, `pyyaml`, `bm25s`, `faiss-cpu`.
Optional extras: `dense = [sentence-transformers]`, `tracking = [mlflow]`, `dev = [pytest, ruff]`.

## Config (`configs/phase1.yaml`)
Sections: `corpus`, `bench`, `retrieval`, `run`, and three model roles:
`paraphraser`, `judge`, `agent`. Each model has `name, family, base_url,
api_key_env, price_in_per_m, price_out_per_m, temperature, max_tokens, seed,
training_cutoff` (ISO date).

**Model-independence check (fail loudly):** `paraphraser`, `judge` and `agent`
must all come from different families. Reason: a same-family agent can "undo"
its sibling's word choices, a same-family judge rubber-stamps them, and a judge
sharing the agent's family selects paraphrases the agent finds natural.

**Training-cutoff check (fail loudly):** `bench.target_from_date` must be later
than every model's `training_cutoff`.

## LLM client (`llm.py`)
- OpenAI-compatible `/chat/completions` over httpx (works with local vLLM or a hosted API).
- SQLite cache keyed by hash of (model, messages, temperature, seed, max_tokens).
- Usage tracking returns two numbers: **method cost** (counted even on cache
  hits, so reported cost reflects the method) and **money spent** (real calls only).
- `Ledger` with a hard cap (`run.budget_cap`, default 20) on spend **cumulative
  across all commands**: every real call appends a row (timestamp, command,
  model, prompt_tokens, completion_tokens, cost) to a `spend` table in the same
  SQLite database as the cache, and the ledger reads the running total at
  startup. Charge a pessimistic estimate *before* each real call; raise
  `BudgetExceeded` if over.
- Retries with exponential backoff on 429/5xx/transport errors.
- `parse_json()` helper that tolerates code fences.

## Corpus (`corpus.py`, `targets.py`)
Source: OpenAlex `/works`. Optional `mailto` and `api_key` (from env). Respect
10 req/s. Cursor paging, `per_page=200`, `select` only the fields needed
(id, doi, title, publication_date, abstract_inverted_index, referenced_works,
primary_topic, cited_by_count, language).

Build order:
1. **Subfield sweeps**: each config filter string (e.g.
   `primary_topic.subfield.id:1702,from_publication_date:2023-01-01`),
   always add `has_abstract:true`. Request results sorted by publication_date
   **descending** (newest first, so targets always survive truncation), ties
   by id, and stop after `corpus.max_works_per_sweep` (default 25000), so
   truncation is deterministic. Check the OpenAlex docs for which sort fields
   exist; if `id` isn't sortable, sort ties by (date, id) locally after
   fetching. Rebuild abstracts from the inverted index.
   Drop docs under `min_abstract_words` or wrong language.
2. **Dedupe**: merge two works only if they share a DOI, or if their
   normalised titles match AND the title has at least 5 content words AND
   their publication years are at most 2 apart. Earliest version keeps its id;
   other ids become `aliases` (a hit on any alias counts as a hit).
3. **Select targets** (`targets.py`): origin = sweep, published on or after
   `bench.target_from_date` (later than every model's `training_cutoff`, checked at load),
   abstract 100–450 words, at least 10 references. Seeded sample of
   `1.5 × n_targets` (oversample; some will fail the filters). Fail loudly if
   no target candidates remain after selection.
4. **Citation expansion**: fetch every target's references (batch
   `ids.openalex:W1|W2|...`, 50 per call) and up to `max_citers_per_target`
   citers (`cites:Wxxx`). References are the hard near-miss distractors.
   Dedupe again and re-map target ids through aliases.
5. **Freeze**: write `corpus.jsonl`, `targets.jsonl`, `manifest.json`
   (filters, counts by origin, API calls, sha256 of corpus file, timestamp,
   and per sweep: `truncated` (bool), `oldest_date`, `newest_date`, `n_fetched`).
   Refuse to overwrite an existing manifest.

## Benchmark generation (`bench.py`), Test A only
For each target, up to `max_attempts_per_target` tries:
1. **Paraphraser** writes a 2–4 sentence research *proposal* (as if written
   before the work) with the same research question, method and claimed
   contribution, using different terminology. No model, dataset or method
   names, no acronyms, no title words. Attempt `a` (from 0) uses seed
   `paraphraser.seed + a`, so every attempt has its own cache key, and every
   retry prompt includes the previous attempt's rejection reason.
2. **Lexical-leak filter** (deterministic, reject if any fails):
   - any distinctive title term (content stems minus stopwords and generic science words) present
   - any acronym/named artefact from the title or abstract present
   - more than a fraction `max_distinctive_overlap` (in [0, 1]; 0.15 = at most 3)
     of the abstract's top-20 IDF terms present
   - content-stem Jaccard with title+abstract above `max_content_jaccard`
3. **Equivalence filter**: the **judge** model (different family) checks
   both directions, as JSON yes/no + reason:
   - Does the proposal capture the original's core problem, method and claim?
   - Does the proposal add nothing the original doesn't support?
   Both must be yes. **Never use embedding similarity as the filter**: it
   would keep only paraphrases dense retrieval handles well and bias the benchmark.
4. Record, but **never filter on**, the target's BM25 rank for the paraphrase
   (a difficulty statistic). Filtering on it would bias against B1.

Output `data/bench/test_a.jsonl`: `qid, target_id, target_aliases, query,
attempt, leak_stats, judge_verdicts, bm25_rank`. Also write a rejection log
with reasons and a short stats file (accepted / rejected by reason).
Stop at `n_targets` accepted queries.

## Retrieval configurations (`retrieval.py`, `agent.py`)
All return a ranked list of up to 100 doc ids plus a trace
(queries issued, LLM usage, latency).

| ID | Strategy |
|---|---|
| B1 | BM25 (`bm25s`), single query |
| B2 | Dense: sentence-transformers (default `BAAI/bge-small-en-v1.5`, query prefix from config), normalised vectors, FAISS `IndexFlatIP` |
| B3 | Hybrid: reciprocal rank fusion (k=60) of B1 and B2 top-100 |
| B4 | One LLM call (agent model) proposes up to `max_queries - 1` rewrites; run B3 on the original + rewrites; RRF-fuse all lists |
| B5 | Agent loop: issue a query, see top-10 titles, decide next query or stop. Same `max_queries` and a `max_llm_calls_agent` cap. Fuse exactly as B4 |
| B5-U | B5 with the larger unconstrained budget, reported separately |

The encoder is pluggable so tests can use a deterministic fake. Indexes are
built once per frozen corpus and saved to `data/index/`.

## Evaluation (`evaluate.py`)
- Per config: Hit@1, Hit@5, Hit@10, MRR (over top-100). Hit = target id or any alias.
- Per query: queries issued, LLM calls, prompt/completion tokens, method cost, latency.
  Report mean, p50, p95.
- **Paired statistics for B5 vs B4** (and B4 vs B3): paired bootstrap
  (10,000 resamples, seeded) 95% CI on ΔHit@5 and ΔMRR, plus exact McNemar on Hit@5.
- Outputs per run in `runs/<timestamp>/`: `per_query/<config>.jsonl`, `summary.json`,
  `summary.md` (one table), and the config used.
- MLflow logging behind `run.mlflow: true` only (optional dependency).

## CLI
```
python -m rk build-corpus  --config configs/phase1.yaml
python -m rk build-index   --config ...
python -m rk make-bench    --config ...
python -m rk run           --config ... [--configs B1,B4,B5] [--limit N]
python -m rk report        runs/<timestamp>
```
The `rk` console script is equivalent, but launcher executables can be blocked
on Windows, so docs always use `python -m rk`.

## Steps (one per Claude Code session)
1. Scaffold: layout, `pyproject.toml`, `.gitignore`, `config.py`, `text.py`, `llm.py`, `conftest.py`, tests.
2. Corpus + targets with mocked OpenAlex tests.
3. Benchmark generator + filters, tests with FakeLLM.
4. Retrieval B1–B5, tests on the fixture corpus.
5. Evaluation runner, metrics, paired stats, CLI wiring, end-to-end test on fixtures.
6. Pilot on real data (small: ~30 targets) and fix what breaks.
7. Full run, README with results and limitations, push.

## Out of scope for Phase 1
Test B (false prior-art accusations; needs independent novelty labels such as
RINoBench), the critic, LangGraph, FastAPI, Docker, live web search.

## Known limitations to state in the README
- Single-domain paraphrases don't test cross-domain mechanism matching.
- Post-cutoff targets reduce, but don't eliminate, memorisation.
- An LLM equivalence judge can still accept subtly shifted meanings.
- B4/B5 parity is on queries issued, not tokens; B5 spends more tokens by design, so cost is always reported next to quality.