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
¦   +-- llm.py            # OpenAI-compatible client, cache, cost ledger, prompt loading
¦   +-- prompts/          # versioned prompt templates (*.txt)
¦   +-- text.py           # tokenising, stemming, stopwords, acronyms
¦   +-- corpus.py         # OpenAlex client, corpus build, dedupe, freeze
¦   +-- targets.py        # target selection
¦   +-- bench.py          # paraphrase generation + filters
¦   +-- retrieval.py      # indexes, B1-B4, fusion
¦   +-- agent.py          # B5, B5-U, all six configs wired from the config
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
Source: OpenAlex `/works`. Optional `api_key` (sent as a Bearer header; a
free key gives 10× the keyless daily budget) and `mailto` (ignored by OpenAlex
since February 2026), both from env. Throttle to the documented limit (100
req/s). Cursor paging, `per_page=100` (the documented maximum; 200 is
deprecated), `select` only the fields needed (id, doi, title,
publication_date, abstract_inverted_index, referenced_works, primary_topic,
cited_by_count, language). The API facts and their doc URLs are in the
comment at the top of `corpus.py`.

Build order:
1. **Subfield sweeps**: each config filter string (e.g.
   `primary_topic.subfield.id:1702,from_publication_date:2023-01-01`),
   always add `has_abstract:true` and `to_publication_date:<build date>`
   (newest-first results otherwise start with bogus future dates). Request
   results sorted by publication_date **descending** (newest first, so targets
   always survive truncation) and keep `corpus.max_works_per_sweep` (default
   25000). OpenAlex can't sort by `id` (HTTP 400), so past the cap keep reading
   while the publication date at the cut lasts, then order by (date desc, id
   asc) locally and cut: truncation is deterministic. Rebuild abstracts from
   the inverted index. Drop docs under `min_abstract_words` or wrong language.
2. **Dedupe**: merge two works only if they share a DOI, or if their
   normalised titles match AND the title has at least 5 content words AND
   their publication years are at most 2 apart. Earliest version keeps its id;
   other ids become `aliases` (a hit on any alias counts as a hit). A merged
   record keeps the earliest version's fields, with references unioned over
   all versions.
3. **Select targets** (`targets.py`): origin = sweep, published on or after
   `bench.target_from_date` (later than every model's `training_cutoff`, checked at load),
   abstract 100–450 words, at least 10 references. Seeded sample of
   `1.5 × n_targets` (oversample; some will fail the filters). Fail loudly if
   no target candidates remain after selection.
4. **Citation expansion**: fetch every target's references (batch
   `ids.openalex:W1|W2|...`, 50 per call) and up to `max_citers_per_target`
   citers (`cites:Wxxx`, newest first, cut like a sweep). References are the
   hard near-miss distractors. Expansion docs pass the same abstract, language
   and `min_abstract_words` filters as sweeps; references without an abstract
   (and ids OpenAlex doesn't return) are counted per origin under `dropped` in
   the manifest. Dedupe again, re-map
   target ids through aliases, and re-check remapped targets against the
   target rules (an earlier version may predate `target_from_date`).
5. **Freeze**: write `corpus.jsonl`, `targets.jsonl`, `manifest.json`
   (filters, counts by origin, API calls, sha256 of corpus file, timestamp,
   and per sweep: `truncated` (bool), `oldest_date`, `newest_date`, `n_fetched`,
   `n_kept`).
   Refuse to overwrite an existing manifest.

## Benchmark generation (`bench.py`), Test A only
Prompts are versioned text files in `src/rk/prompts/` (`paraphrase.txt`,
`judge_captures.txt`, `judge_adds_nothing.txt`); each prompt's sha256 is
recorded in every output row and in `stats.json`. Targets are visited in a
seeded random order (`bench.seed`). For each target, up to
`max_attempts_per_target` tries; the first failing check is the rejection
reason:
1. **Paraphraser** gets the title and abstract and writes a research
   *proposal* (as if written before the work) with the same research question,
   method and claimed contribution, using different terminology. No model,
   dataset or method names, no acronyms, no title words. Attempt `a` (from 0)
   uses seed `paraphraser.seed + a`, so every attempt has its own cache key,
   and every retry prompt includes the previous attempt's rejection reason.
2. **Length**: 2–4 sentences and 40–120 words, else `length`.
3. **Lexical-leak filter** (deterministic), in this order:
   1. any acronym/named artefact in the proposal at all (`acronym_in_paraphrase`)
   2. any acronym/named artefact from the title or abstract, case-insensitive
      (`source_acronym`; ones spelling a stopword, like "IT" or "US", are skipped)
   3. any distinctive title stem: title content stems minus generic science
      words (`title_stem`)
   4. more than a fraction `max_distinctive_overlap` (in [0, 1]; 0.15 = at most
      3) of the abstract's top-20 IDF stems, IDF over the frozen corpus, ties
      to stems more frequent in the abstract (`idf_overlap`)
   5. content-stem Jaccard with title+abstract above `max_content_jaccard`,
      generic science words excluded on both sides with the same rule as in
      3 (`content_jaccard`)
4. **Equivalence filter**: the **judge** model (different family) answers
   each direction in a **separate call**, so one answer can't anchor the other,
   as JSON `{"verdict": "yes"|"no", "reason": "..."}`:
   - Does the proposal capture the original's core problem, method and claim?
     (`judge_captures` on "no")
   - Does the proposal add nothing the original doesn't support?
     (`judge_adds_nothing` on "no"; not asked after a first "no")
   An unreadable answer is a rejection (`judge_parse_error`), never a pass.
   **Never use embedding similarity as the filter**: it would keep only
   paraphrases dense retrieval handles well and bias the benchmark.

Outputs in `bench.out_dir`: `test_a.jsonl` (`qid, target_id, target_aliases,
query, attempt, leak_stats, judge_verdicts, prompt_hashes`, plus the attempt's
LLM `usage`), `rejections.jsonl` (one row per rejected attempt with `reason`
and `detail`), and `stats.json` (accepted, rejected by reason, attempts per
target, LLM usage and method cost). Rows are appended as attempts finish, so a
rerun resumes: targets already accepted or exhausted are skipped, an
interrupted target continues with its next attempt, and outputs made with other
prompts, models or another corpus are refused. Stop at `n_targets` accepted
queries. The BM25 rank of each target is not computed here (see Evaluation).

## Retrieval configurations (`retrieval.py`, `agent.py`)
All implement one interface, `retrieve(query)`, returning a ranked list of up
to 100 doc ids plus a trace (queries issued, LLM calls, prompt/completion
tokens, method cost, latency, errors, prompt hashes). Every ranking breaks ties
by doc id.

| ID | Strategy |
|---|---|
| B1 | BM25 (`bm25s`), single query; only docs with a positive score |
| B2 | Dense: sentence-transformers (default `BAAI/bge-small-en-v1.5`, query prefix from config), normalised float32 vectors, FAISS `IndexFlatIP` |
| B3 | Hybrid: reciprocal rank fusion (k=60) of B1 and B2 top-100 |
| B4 | One agent-model call returns JSON `{"queries": [...]}`: at most `max_queries - 1` rewrites, duplicates removed. Run B3 on the original + rewrites; RRF-fuse all lists. On a parse failure, the original query alone, with the error recorded |
| B5 | The original query runs first with no LLM call. Each agent step sees the queries so far with the top-10 titles of each (titles only, to bound tokens) and returns `{"action": "search", "query": ...}` or `{"action": "stop", "reason": ...}`. Hard limits: `max_queries` queries in total and `max_llm_calls_agent` calls. A repeated query counts toward the budget but isn't re-run; a parse failure stops the loop and is recorded. Fused with exactly the same function as B4 |
| B5-U | B5 with the larger unconstrained budgets, reported separately |

Prompts (`rewrite.txt`, `agent_step.txt`) frame the task as finding prior work
for a research proposal and never mention a benchmark or a target paper.

**Indexes** (`python -m rk build-index`): document text is "title. abstract".
BM25 gets the project's own tokenisation (`content_tokens` + `stem`). The
encoder sits behind a small interface (`encode_queries`, `encode_docs`) so
tests use a deterministic fake; it runs on the GPU if torch sees CUDA. Indexes
are built once per frozen corpus and saved to `data/index/` with a manifest
(corpus sha256, encoder, device, doc count, build time); loading refuses an
index whose corpus sha256 doesn't match the frozen corpus.

## Evaluation (`evaluate.py`)
- Per config: Hit@1, Hit@5, Hit@10, MRR (over top-100). Hit = target id or any alias.
- Per query, a difficulty column: the target's rank in the B1 (BM25) results.
  Recorded, **never filtered on** (filtering on it would bias against B1).
- Per query: queries issued, LLM calls, prompt/completion tokens, method cost, latency.
  Report mean, p50, p95.
- **Paired statistics for B5 vs B4** (and B4 vs B3): paired bootstrap
  (10,000 resamples, seeded) 95% CI on ΔHit@5 and ΔMRR, plus exact McNemar on Hit@5.
- Outputs per run in `runs/<timestamp>/`: `per_query/<config>.jsonl`, `summary.json`,
  `summary.md` (one table), and the config used.
- MLflow logging behind `run.mlflow: true` only (optional dependency).

## CLI
```
python -m rk build-corpus  --config configs/phase1.yaml [--max-works N]
python -m rk build-index   --config ...
python -m rk make-bench    --config ... [--limit N] [--show]
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