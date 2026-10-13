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
+-- .gitignore            # data/, runs/, .env, .venv/, __pycache__/, *.sqlite, mlflow.db, mlartifacts/
+-- .env.example          # the API key variables, empty; copy to .env (git-ignored)
+-- .github/workflows/ci.yml  # Python 3.10: ruff check, then pytest
+-- configs/phase1.yaml
+-- docs/phase1_spec.md
+-- src/rk/
¦   +-- __init__.py
¦   +-- config.py         # YAML -> dataclasses
¦   +-- llm.py            # OpenAI-compatible client, cache, cost ledger, prompt loading
¦   +-- prompts/          # versioned prompt templates (*.txt)
¦   +-- data/             # curated lists (domain_entities.txt, domain_fields.txt)
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
Core: `httpx`, `numpy`, `pyyaml`, `bm25s`, `faiss-cpu`, `mlflow`, `python-dotenv`.
Optional extras: `dense = [sentence-transformers>=2.7.0, transformers>=4.51.0]`
(the Qwen3-Embedding model card's minimums), `dev = [pytest, ruff]`.

## Config (`configs/phase1.yaml`)
Sections: `corpus`, `bench`, `retrieval`, `run`, and three model roles:
`paraphraser`, `judge`, `agent`. Each model has `name, family, base_url,
api_key_env, price_in_per_m, price_out_per_m, temperature, max_tokens, seed,
training_cutoff` (ISO date), and an optional OpenRouter `provider` (see LLM
client).

API keys are read from the environment variables the config names. At
startup `python -m rk` also loads `.env` from the working directory (the repo
root) with python-dotenv; variables already set in the environment win.
`.env.example` lists the variables with empty values.

**Model-independence check (fail loudly):** `paraphraser`, `judge` and `agent`
must all come from different families. Reason: a same-family agent can "undo"
its sibling's word choices, a same-family judge rubber-stamps them, and a judge
sharing the agent's family selects paraphrases the agent finds natural.

**Training-cutoff check (fail loudly):** `bench.target_from_date` must be later
than every model's `training_cutoff`.

## LLM client (`llm.py`)
- OpenAI-compatible `/chat/completions` over httpx (works with local vLLM or a hosted API).
- SQLite cache keyed by hash of (model, messages, temperature, seed, max_tokens),
  plus the provider routing when a provider is pinned.
- **Provider pinning (OpenRouter):** a model's `provider` (an endpoint slug
  from OpenRouter's provider list, e.g. `novita/fp8`) is sent as
  `"provider": {"order": [slug], "allow_fallbacks": false,
  "require_parameters": true}`, so only that endpoint serves the model, and
  OpenRouter refuses the call (a 4xx) rather than drop a parameter such as
  `seed` that the endpoint doesn't support. Every request sends `X-OpenRouter-Metadata:
  enabled`; the provider OpenRouter reports (the `selected` endpoint in
  `openrouter_metadata`, else the top-level `provider` string) is stored with
  the cached reply, returned on cache hits too, and recorded in every trace
  and spend row.
- Usage tracking returns two numbers: **method cost** (counted even on cache
  hits, so reported cost reflects the method) and **money spent** (real calls only).
- `Ledger` with a hard cap (`run.budget_cap`, default 20) on spend **cumulative
  across all commands**: every real call appends a row (timestamp, command,
  model, prompt_tokens, completion_tokens, cost, billed_cost, provider) to a
  `spend` table in the same SQLite database as the cache, and the ledger reads
  the running total at startup. `cost` is computed from the config prices;
  `billed_cost` is what the API reports it charged (OpenRouter's `usage.cost`,
  in USD, returned on every non-streaming reply without being requested), or
  NULL if it reports nothing. A row counts against the cap at `billed_cost`
  when present, else at `cost`. Charge a pessimistic estimate *before* each
  real call; raise `BudgetExceeded` if over.
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
`max_attempts_per_target` tries (4 in configs/phase1.yaml: with 3, 14 of 40
pilot targets ran out, and exhaustion biases the benchmark toward papers that
are easy to paraphrase); the first failing check is the rejection
reason:
0. **Not a research contribution** (checked once, before any LLM call): a
   target whose title contains "survey", or whose title or abstract describes
   the paper itself as one ("this/our/the present survey|review|tutorial",
   "we survey|review", "is/are reviewed", "PRISMA"), is skipped with reason
   `not_research_contribution`. A survey's proposal is generic and matches
   many papers equally well, so the target isn't the one right answer. Only
   self-description counts, so papers *about* reviews (Cochrane reviews, peer
   review, literature-review tooling) are kept. Found in the Step 6 pilot,
   where 4 of 35 accepted targets were surveys, reviews or tutorials; OpenAlex's
   `type` field marked none of them as reviews.
1. **Paraphraser** gets the title and abstract and writes a research
   *proposal* (as if written before the work) with the same research question,
   method and claimed contribution, using different terminology. No model,
   dataset, method or organisation names, no acronyms, no rare or distinctive
   title words (common field and task terms are fine; see the leak filter),
   and a language, nationality, country or region, or a research field or
   application domain (physics, maritime, clinical, ...), that the title
   names must be kept (the prompt lists the ones found in the title). These
   domain terms come from two curated lists, `src/rk/data/domain_entities.txt`
   and `src/rk/data/domain_fields.txt` (matched whole-word and
   case-insensitively, except all-caps names and names marked `=` because they
   are also common words; ML and method words are left out). Attempt `a` (from 0)
   uses seed `paraphraser.seed + a`, so every attempt has its own cache key,
   and every retry prompt includes the previous attempt's rejection reason.
2. **Length**: 2–4 sentences and 40–120 words, else `length`.
3. **Lexical-leak filter** (deterministic). Domain terms are first blanked out
   of proposal, title and abstract, so keeping one is never a leak. Then, in
   this order:
   1. any acronym/named artefact in the proposal at all (`acronym_in_paraphrase`)
   2. any acronym/named artefact from the title or abstract (`source_acronym`):
      matched case-insensitively, except mixed-case forms ("STandard", "LoRA"),
      which must match as written; ones spelling a stopword ("IT", "US") or a
      common corpus word are skipped. Common means at least 0.5% of corpus
      abstracts write it in lower case, and not less often than they
      capitalise it (the `source_name` statistic): "STEP" and the "STandard"
      it abbreviates no longer ban the words step and standard, while gpt
      (0.24%), llm and rag stay banned in lower case too.
   3. any capitalised name from the abstract ("Bilibili") that the proposal
      uses mid-sentence (`source_name`). A word counts as a name if the corpus
      abstracts capitalise it mid-sentence more often than they write it in
      lower case, so title-cased common words ("Retrieval-Augmented
      Generation") don't count; stopwords and generic science words never do.
   4. any rare title stem: a title content stem (generic science words
      excluded) whose document frequency in the frozen corpus is below
      `bench.title_stem_max_df`, a fraction of documents (`title_stem`). Common
      task words ("translation", "question answering", "knowledge graph") are
      allowed; rare, distinctive ones and names stay banned. 1.0 bans every
      title stem. configs/phase1.yaml uses 0.02, chosen from the pilot's 215
      title stems (median df 3.6%): names and paper-defining terms sit below
      2% (abduction 0.09%, faithfulness 1.3%), task words well above
      (translation 9.5%, question 15.5%, graph 6.8%).
   5. more than a fraction `max_distinctive_overlap` (in [0, 1]; 0.15 = at most
      3) of the abstract's top-20 IDF stems, IDF over the frozen corpus, ties
      to stems more frequent in the abstract (`idf_overlap`)
   6. content-stem Jaccard with title+abstract above `max_content_jaccard`,
      generic science words excluded on both sides with the same rule as in
      4 (`content_jaccard`)
4. **Equivalence filter**: the **judge** model (different family) answers
   each direction in a **separate call**, so one answer can't anchor the
   other. In each call it first describes four facets of the paper and of the
   proposal (problem; method; domain or population, including language or
   region; main claim), then compares them facet by facet, and answers JSON
   `{"facets": ..., "match": {facet: {"match": "yes"|"no", "reason"}}, "verdict",
   "reason"}`:
   - Does the proposal keep each facet? "no" if one is left out, changed, or
     stated so generally that it no longer tells this work apart from close
     sibling work (e.g. it drops the specific domain, population, data type or
     language); the prompt gives one short example (`judge_captures`).
   - Does the proposal change or add to any facet, or state one too
     generally (the same test and example)? (`judge_adds_nothing`; not asked
     after a first "no"). Testing generality only in "captures" was tried in
     the pilot and caught 7/30 generalised corruptions instead of 24/30.
   Both prompts say that names of datasets, benchmarks, models, methods,
   platforms and organisations, and numbers, are removed on purpose and never
   make a facet fail, and that a described stand-in counts as specific (the
   pilot judge kept demanding names the leak filter bans).
   The answer format is enforced by the API: the judge is called with
   `response_format` = a strict JSON schema (OpenAI-style structured outputs,
   which the pinned endpoint must list; with `require_parameters` OpenRouter
   refuses rather than ignores it), and the schema's sha256 is recorded with
   the prompt hashes. The verdict is "yes" only if the judge says yes **and** every facet
   matches; a facet marked "no" rejects whatever the judge concluded, and the
   rejection message (quoted in the retry prompt) names each failed facet with
   its reason. A reply without a yes/no for every facet and the verdict is a
   rejection (`judge_parse_error`), never a pass. `stats.json` counts judge
   rejections per facet (`judge_failed_facets`). **Never use embedding
   similarity as the filter**: it would keep only paraphrases dense retrieval
   handles well and bias the benchmark.

**Judge validation** (`python -m rk validate-judge --config ... --n 30`): for
the first N accepted pairs (qid order), the paraphraser writes four
corruptions of the proposal (`prompts/corrupt.txt`), each with exactly one
change: a different domain or population, a different method, a different
expected outcome or contribution (what the work will show or deliver; merely
negating a "whether X" question doesn't count), or so general it no longer
identifies the work. The
judge answers both directions for the originals and the corruptions (a
variant passes only if both say yes). Outputs in `data/judge_validation/`:
`pairs.jsonl`, `corruptions.jsonl`, `verdicts.jsonl` and `summary.json`
(acceptance rate on originals, rejection rate per corruption type, which
direction and facet caught it, cost). The originals passed this judge in
make-bench, so they are accepted by construction; the corruptions' rejection
rates are what measure the judge.

A qid is the target's position in the seeded visiting order plus its id
(`A0007-W123`), so sorting by qid gives the seeded order and the first N
queries (`run --limit N`) are a seeded random sample, not the lowest ids.

Outputs in `bench.out_dir`: `test_a.jsonl` (`qid, target_id, target_aliases,
query, attempt, leak_stats, judge_verdicts, prompt_hashes`, plus the attempt's
LLM `usage` and, per role, the `providers` that served its calls),
`rejections.jsonl` (one row per rejected attempt with `reason` and `detail`,
plus one row per skipped target), and `stats.json` (accepted, rejected by
reason, skipped, attempts per target, LLM usage, calls per provider and method
cost; skipped targets had no attempt, so they aren't in the attempt counts).
`--limit N` caps the targets paraphrased in a run; skipped ones don't count. Rows are appended as attempts finish, so a
rerun resumes: targets already accepted or exhausted are skipped, an
interrupted target continues with its next attempt, and outputs made with other
prompts, models or another corpus are refused. Stop at `n_targets` accepted
queries. If the frozen targets run out first, the bench is smaller: no targets
are added, since their references aren't in the corpus and they would be
easier. The BM25 rank of each target is not computed here (see Evaluation).

## Retrieval configurations (`retrieval.py`, `agent.py`)
All implement one interface, `retrieve(query)`, returning a ranked list of up
to 100 doc ids plus a trace (queries issued, LLM calls and the provider that
served each, prompt/completion tokens, method cost, latency, errors, prompt
hashes). Every ranking breaks ties by doc id.

| ID | Strategy |
|---|---|
| B1 | BM25 (`bm25s`), single query; only docs with a positive score |
| B2 | Dense: sentence-transformers (default `BAAI/bge-small-en-v1.5`; `configs/phase1.yaml` uses `Qwen/Qwen3-Embedding-0.6B`), pinned to the Hugging Face commit `retrieval.encoder_revision`, run in `encoder_dtype` (default float32) with inputs truncated at `encoder_max_seq_length` tokens (default: the model's own). Queries get `query_prefix` (for Qwen3-Embedding, the model card's `Instruct: <task>\nQuery:` format), documents no prefix. Normalised float32 vectors, FAISS `IndexFlatIP` |
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
(corpus sha256, encoder, revision, dtype and max sequence length, device, doc
count, build time); loading refuses an index whose corpus sha256 doesn't match
the frozen corpus, and dense retrieval refuses an encoder whose name, revision,
dtype or max sequence length differs from the index's.

## Evaluation (`evaluate.py`)
- **Runner** (`python -m rk run`): loads the frozen corpus, index and bench (all
  integrity-checked), creates `runs/<timestamp>/` and records run metadata: git
  commit and whether tracked files have uncommitted changes (untracked files
  are listed separately, as information), config snapshot, corpus and bench
  sha256, index manifest, prompt hashes, model names. Queries run in qid
  order (the seeded order), so `--limit N` is a seeded random sample.
- Per query and config: target rank (first position of the target or any
  alias in the top-100, else none), Hit@1/5/10, reciprocal rank, and the trace
  fields (queries issued, LLM calls, prompt/completion tokens, method cost,
  latency, stop reason).
- **Errors**: an LLM error marks that row `status=error` and the run continues
  (other errors, including `BudgetExceeded`, stop it after the summary is
  written). `--retry-errors` with `--run-dir` redoes only errored rows; a rerun
  into the same `--run-dir` skips finished rows, and refuses if the corpus,
  bench, index, retrieval config, agent model or retrieval prompts changed.
- **Results database** (`run.results_db`, default `data/results.sqlite`):
  tables `runs` (run metadata) and `query_results` (one row per run, config and
  query), written as rows finish. `per_query/<config>.jsonl` is a portable
  copy. `python -m rk report` rebuilds the summary from the database with SQL.
- **Metrics**: per config Hit@1/5/10 and MRR, and mean/p50/p95 of queries
  issued, LLM calls, tokens, method cost and latency (successful rows only).
- **Difficulty**: each query's B1 target rank is its difficulty (recorded,
  **never filtered on**). Hit@5 per config within B1-rank buckets 1, 2–10,
  11–100 and not found (only when B1 is part of the run).
- **Paired comparisons**: B4 vs B3, B5 vs B4, B5-U vs B5, on queries that
  succeeded in both configs: seeded paired bootstrap (10,000 resamples,
  `run.seed`) 95% CI on ΔHit@5 and ΔMRR, and an exact McNemar test on Hit@5.
- A cost–quality table: Hit@5 and MRR next to mean method cost per query.
- Outputs per run in `runs/<timestamp>/`: `run.json`, `config.yaml`,
  `per_query/<config>.jsonl`, `summary.json` and `summary.md` (main,
  effort, difficulty, paired, cost–quality and error tables).
- **MLflow** (core dependency) with the documented default SQLite backend
  (`run.mlflow_tracking_uri`, default `sqlite:///mlflow.db` at the repo root,
  artifacts in `mlartifacts/` beside it): one parent run per benchmark run and
  one child run per config, with params (models, budgets, prompt hashes, corpus
  sha, encoder and revision, git commit), metrics, and artifacts (per-query
  JSONL, summary). UI: `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## CLI
```
python -m rk build-corpus  --config configs/phase1.yaml [--max-works N]
python -m rk build-index   --config ...
python -m rk make-bench    --config ... [--limit N] [--show]
python -m rk validate-judge --config ... [--n 30] [--out-dir data/judge_validation]
python -m rk export-human-check --config ... [--n 20] [--out data/human_check.csv] [--validation-dir DIR]
python -m rk run           --config ... [--configs B1,B4,B5] [--limit N] [--run-dir runs/<timestamp>] [--retry-errors]
python -m rk report        runs/<timestamp>
```
Every command that takes `--config` also takes `--set SECTION.KEY=VALUE`
(repeatable; YAML values) to override one config value for that run, e.g.
`--set judge.name=openai/gpt-4.1`; the result is validated like the file.
`export-human-check` writes pairs to a CSV for checking by hand (`part, qid,
title, abstract, proposal, judge_reason, verdict, note`, the last two empty),
and never overwrites one. Part A is a seeded random sample (`bench.seed`) of
`--n` accepted pairs: does the proposal describe the same paper? With
`--validation-dir`, part B adds every original that the configured judge
rejected in that `validate-judge` run, with the judge's reason per failed
facet: is the rejection right? A run made with another judge is refused.

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