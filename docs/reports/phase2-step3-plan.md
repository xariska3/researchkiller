# Phase 2 Step 3 plan: the competitive evaluation, before any paid call

This step adds `python -m rk compete`: it puts the live tool next to Claude and an
OpenAI model with their web search tools, Perplexity Sonar and Semantic Scholar on 100
Phase 1 queries. The design follows [docs/phase2_spec.md](../phase2_spec.md)
(Competitive evaluation) and the instructions for this step. The command is built and
tested with fakes. **Nothing has been sent to a paid API yet.** The cost estimates below
are for your go.

- **Branch**: `phase2-step3-competitive`, from `main` at eb6dc36. `main` and tonight's
  00:07 UTC test run are untouched.
- **Command**: `python -m rk compete --config configs/phase1.yaml [--limit N]
  [--systems ours,claude-sonnet,...] [--dry-run] [--save-sample CSV]`
  (`src/rk/compete/`).

## Update after your answers (2026-10-14)

This section supersedes the Summary and Decisions below where they differ.

**Systems, in run order**:

| System | What it is | Cap |
|---|---|---|
| `claude-sonnet` | Sonnet 5.5, all 100 queries first | $40 |
| `claude-opus` | Opus 5.5, on the same 100 queries | $55 |
| `openai-luna` | "OpenAI smaller model (gpt-6-luna); flagship not tested for budget reasons". Called directly through the OpenAI Responses API with `{"type": "web_search"}` and `max_tool_calls: 5`, to match Claude's 5 searches. | $4.50 |
| `perplexity` | Sonar on OpenRouter | $1 |
| `s2-search`, `s2-snippets` | Semantic Scholar, with the key, at 0.5 requests/s | free |

- **Combined Anthropic cap**: both Claude systems belong to the spend group `anthropic`,
  capped in code at $89. A call that could take their combined spend past it is refused,
  so Opus gets at most min($55, $89 − Sonnet's spend). The workspace's $95 spend limit
  is a second guard.

**Free key checks** (no tokens):
- **Anthropic**: the workspace-scoped key is accepted (Models API HTTP 200). It reports
  web search and code execution as supported for `claude-sonnet-5-5` and
  `claude-opus-5-5`.
- **OpenAI**: the key is accepted and `gpt-6-luna` is available. OpenAI has no balance
  endpoint for API keys; the balance is $5.33 as you reported.
- **OpenRouter**: $6.64 left.
- **Semantic Scholar**:
  - The key works (HTTP 200 after a pause). Keyless requests are still refused with 429,
    and the first keyed request got one too, so the client's backoff on 429 matters.
  - Snippet search on the A0049 proposal returned 50 passages from 40 distinct, on-topic
    papers.

**OpenAI web search, direct**
([guide](https://developers.openai.com/api/docs/guides/tools-web-search),
[pricing](https://developers.openai.com/api/docs/pricing),
[gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna)):
- **Calling it**: Responses API only for `gpt-6-luna`, which reasons at medium effort by
  default. `max_tool_calls` caps the built-in tool calls in one response.
- **Billing**: each `web_search_call` (search, open page or find in page) is a tool
  call. "Search actions incur a tool call cost": $10 per 1,000 calls plus the search
  content tokens at model rates ($0.10/$0.50 per million).
- **Our counting**: the docs don't say whether opening a page is billed like a search, so
  every `web_search_call` is counted at $0.01.

**Re-estimate**:

| System | Per query: low / central / high | 100 queries, central (range) | Cap | Pilot stop (over 2x central) |
|---|---|---|---|---|
| claude-sonnet | $0.11 / $0.22 / $0.35 | $22 ($11–35) | $40 | $0.44 per query |
| claude-opus | $0.19 / $0.40 / $0.65 | $40 ($19–65) | $55, and the group's $89 | $0.80 per query |
| openai-luna | $0.022 / $0.035 / $0.059 | $3.50 ($2.20–5.90) | $4.50 | $0.07 per query |
| perplexity | $0.006 / $0.0065 / $0.008 | $0.65 ($0.60–0.80) | $1 | $0.013 per query |

- **gpt-6-luna**: 2–5 tool calls, 10k–50k input tokens and 2k–8k output tokens per
  query. Above about $0.044 a query (4 or more calls every time), its cap stops it before
  100 queries.
- **The pilot** costs about $2. Sonnet and Opus together are about $62 at the central
  estimates.
- **OpenRouter** keeps at least $5.60 for our own agent calls.

**Sample**: `--save-sample docs/evaluation/compete_sample.csv` writes the sample in its
seeded order. The step-1 dry run will use it once tonight's test run is in
`compete.runs`.

## Summary

- **Estimated cost for 100 queries**:

  | System | Central | Range | Cap |
  |---|---|---|---|
  | Claude Sonnet 5.5 | about $22 | $11–35 | $40 |
  | OpenAI GPT-6.1 Sol | about $12 | $6–23 | $4.50 |
  | Perplexity Sonar | about $0.65 | $0.60–0.80 | $1 |
  | Semantic Scholar | free | – | – |
  | Ours | free | – | – |

  The Sonnet cap is 40% of the $100 promotional credit. Ours is replayed from the
  live-eval caches.
- **OpenAI doesn't fit the OpenRouter balance.** The balance is $6.64 on 2026-10-14
  ($25.00 total, $18.36 used). With a cap of $4.50, about 35–40 of the 100 queries would
  run. A top-up of about $15 runs all 100. The alternative is a cheaper model:
  `openai/gpt-6-luna`, about $3.50 per 100 queries.
- **The Anthropic key in `.env` is refused.** A free Models API call (no tokens) returned
  HTTP 400: "This API key is not scoped to a workspace, so this request must include the
  anthropic-workspace-id header…". I expect the Messages API to refuse it the same way,
  so Claude can't run until this is fixed. I haven't changed any code for it. Proposed
  fix: create a key inside a workspace (ideally a dedicated one with a spend limit of
  about $95, as a second guard on the credit) and put it in `.env`. The alternative is
  for me to add an optional `ANTHROPIC_WORKSPACE_ID` that the client sends as the
  `anthropic-workspace-id` header.
- **Semantic Scholar's paper search finds nothing for a whole proposal.** A free probe
  sent a 57-word proposal as the query and got 0 results. Without a key, most requests
  got HTTP 429. So the spec's baseline (paper search) is joined by its snippet search,
  which matches passages of text. Every snippet probe got HTTP 429, so it is untested
  on our proposals. Both are free.

## API facts checked (2026-10-14)

**Claude with web search** (Anthropic Messages API):
- **Sources**: [web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool),
  [pricing](https://platform.claude.com/docs/en/about-claude/pricing).
- **Tool versions**:
  - `web_search_20250305`: basic search.
  - `web_search_20260209`: adds dynamic filtering. Claude filters the results with
    code before they enter its context, which cuts input tokens. Claude 4.6 and later
    support it.
  - `web_search_20260318`: adds `response_inclusion`.
  - We use `web_search_20260318` with its default dynamic filtering.
- **`max_uses`** caps searches per request. Search tool errors come back inside an
  HTTP 200 response. A long turn can end with `stop_reason: "pause_turn"`; sending the
  assistant content back continues it (the client does this, up to 3 times).
- **Price**:
  - $10 per 1,000 searches, plus tokens.
  - Search results count as input tokens, in the search iterations of a turn and in
    later turns.
  - Code execution is free with `web_search_20260209` and later.
  - Opus 5.5 $4/$20 per million input/output tokens; Sonnet 5.5 $2/$10; Haiku 5.5
    $0.10/$0.50. The 4.7+ tokenizer makes about 30% more tokens for the same text.
- **Promotional credits**:
  - [API credits for Max and Team plans](https://platform.claude.com/docs/en/about-claude/api-credits-for-subscribers)
    cover the Claude API, Managed Agents, the Agent SDK and the playground. They don't
    cover Claude Code. They are drawn first, then any purchased credits.
  - When none are left, requests fail with "credit balance is too low".
  - Neither that page nor the [credit terms](https://www.anthropic.com/legal/credit-terms)
    list an exclusion for server tools such as web search. I can't confirm coverage
    for your particular offer. The Console's Cost page will show it after the pilot.

**OpenAI with web search, via OpenRouter**:
- **Sources**: [server tool](https://openrouter.ai/docs/guides/features/server-tools/web-search.md),
  [usage accounting](https://openrouter.ai/docs/guides/guides/usage-accounting),
  [OpenAI pricing](https://developers.openai.com/api/docs/pricing).
- **The tool**: the `openrouter:web_search` server tool with `engine: "native"` runs
  OpenAI's own search (GPT-5 and later). The `web` plugin and `:online` are
  deprecated.
- **Limits and counts**: `max_uses` caps searches, and
  `usage.server_tool_use.web_search_requests` counts them. `usage.cost` is "the total
  amount charged to your account", on every response.
- **Price**:
  - OpenAI: $10 per 1,000 web search calls, plus search content tokens at model
    rates.
  - OpenRouter's model list ([/api/v1/models](https://openrouter.ai/api/v1/models)):
    - `openai/gpt-6.1-sol`: $2/$10.
    - `openai/gpt-6-luna`: $0.10/$0.50.
    - `openai/gpt-6-astra`: $10/$50.
    - Each lists `web_search` at $0.01.

**Perplexity Sonar**:
- **Sources**: [pricing](https://docs.perplexity.ai/getting-started/pricing),
  OpenRouter `perplexity/sonar`.
- **Price**:
  - $1/$1 per million tokens.
  - Request fee: $5, $8 or $12 per 1,000 requests at search context low, medium or
    high. Low is the default and is what we use.
  - Citation tokens are not billed for Sonar.
- **On OpenRouter**: $1/$1 and $0.005 per request. Sonar searches on every request, so
  it needs no tool.

**Semantic Scholar** ([OpenAPI spec](https://api.semanticscholar.org/graph/v1/swagger.json)):
- **Endpoints**: `/paper/search` (relevance-ranked, plain-text query) and
  `/snippet/search` (~500-word passages from titles, abstracts and body text, each
  with its paper).
- **Access**: free; a key goes in `x-api-key`.
- **Probes**: without a key, 11 of 12 requests got HTTP 429. The one that returned had
  0 results for a 57-word proposal.

**Checking that listed papers exist, through free routes only**:
- **OpenAlex singleton lookups**: one work by id or DOI is free and doesn't count
  against the daily budget ([example costs](https://help.openalex.org/access/example-costs/)).
  No OpenAlex search is used.
- **Crossref title search**: `/works?query.bibliographic=` is free. The public pool's
  rate limit is 5 with 1 request at a time; a `mailto` gives the polite pool's 10 with
  3 ([access](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/)).

## What was built

- **`python -m rk compete`** (`src/rk/compete/evaluate.py`).
  - **Queries**:
    - Drawn from the queries that the runs in `compete.runs` checked.
    - Stratified by Phase 1 difficulty: the target's rank in `live.phase1_run`'s B1
      (1, 2–10, 11–100, not found).
    - Each bucket's share is proportional (largest remainders), drawn with seed 42, and
      the sample is listed in seeded order. `--limit N` therefore takes N queries spread
      over the buckets.
    - `--dry-run` prints the sample and each system's uncached calls, sending nothing.
  - **Ours** is L5w, replayed offline from each run's cached OpenAlex responses and
    recorded agent queries, with no OpenAlex or LLM calls.
    - Its list is the top 10.
    - Its cost and latency come from the check's recorded row: L5w's on the test run,
      L5's on the dev run (the same check).
    - The replay is compared with the recorded L5w ranking wherever the run has one.
  - **Scoring**:
    - **Original found**: a listed paper is the original if its DOI, the OpenAlex id or
      DOI it resolved to, or its normalised title matches the original (as in
      live-eval). The first match gives the rank: Hit@1/5/10 and MRR@10.
    - **Unresolved (possibly fabricated)**: listed papers no free route verifies.
    - **DOI mismatches** (a DOI that leads to another paper) are counted separately.
    - **Ours** lists OpenAlex records, so it has no unresolved papers by construction.
    - **Unparsed answers**: an answer without the JSON asked for lists nothing and
      counts as a miss.
  - **Statistics**: Phase 1's: ours vs each system on the queries both answered. ΔHit@5
    and ΔMRR@10 get paired bootstrap 95% CIs, and Hit@5 the exact McNemar test with Holm
    adjustment across systems. Hit@5 is also broken down by difficulty bucket.
  - **Outputs**: `runs/compete-<UTC time>/` with `run.json`, `results.jsonl`,
    `summary.json` and `summary.md`. Each row of `results.jsonl` holds every listed
    paper with how it resolved and the raw answer text.
- **Systems** (`src/rk/compete/systems.py`):
  - **Clients**: Anthropic Messages API with the web search tool; OpenRouter chat
    completions, with the server tool (OpenAI) or without (Perplexity); Semantic
    Scholar paper and snippet search.
  - **Caching**: every response is cached in `data/cache/compete.sqlite` under the
    request's sha256, with the call's latency. A rerun is free and reports the same
    latency.
  - **Cost caps**:
    - Each system has a cumulative cost cap (table `compete_spend` in the same file).
    - Before each real call the client reserves `reserve_per_query` and refuses a call
      that could cross the cap.
    - A failed call that may have been billed is booked at its reservation, as in
      `rk.llm.Ledger`.
    - Retries back off on 429, 5xx and transport errors, honouring Retry-After.
- **Resolver** (`src/rk/compete/resolve.py`). It tries each listed paper in this order:
  1. an OpenAlex id in its link;
  2. its DOI, or one in its link;
  3. an arXiv link, as arXiv's DataCite DOI;
  4. its title in Crossref (top 5), with the matched DOI then looked up in OpenAlex.

  Titles agree if their normalised forms are equal, their content stems have Jaccard
  similarity of at least 0.8, or one title's stems (at least 4) are all in the other's.
  Every lookup is cached, misses included.
- **Prompt** (`src/rk/prompts/compete.txt`, the same for every LLM system):

  ```
  Find the existing papers most closely related to this research proposal. List up to 10, each
  with title, year, and DOI or link.

  Answer with JSON only, the most closely related paper first, in this form:
  {"papers": [{"title": "paper title", "year": 2024, "doi": "10.xxxx/xxxx or null", "link": "https://... or null"}]}

  Research proposal: <the Phase 1 proposal>
  ```

  Semantic Scholar's two searches get the proposal itself as the query.
- **Config**: a `compete:` section in `configs/phase1.yaml` holds the systems, prices,
  reserves and caps. `.env.example` adds `ANTHROPIC_API_KEY`, and optional
  `SEMANTIC_SCHOLAR_API_KEY` and `CROSSREF_MAILTO`.

**Dry run on the dev set alone** (the test run isn't in `compete.runs` yet):
- **Sample**: 100 of the 120 checked queries. Per bucket: 1: 40 of 48, 2–10: 25 of 30,
  11–100: 20 of 24, not found: 15 of 18.
- **Calls**: every system's 100 calls are uncached, and nothing has been spent.
- **Prompt size**: about 750 characters on average; proposals run 47–74 words.

## Cost estimates for 100 queries

The token counts are assumptions. The three-query pilot (below) measures them before
the full run.

| System | Model and settings | Per query: low / central / high | 100 queries: central (range) | Cap |
|---|---|---|---|---|
| claude | Sonnet 5.5, `web_search_20260318`, max 5 searches | $0.11 / $0.22 / $0.35 | **$22** ($11–35) | $40 |
| openai | `openai/gpt-6.1-sol`, native search, max 5 searches | $0.06 / $0.12 / $0.23 | **$12** ($6–23) | $4.50 |
| perplexity | `perplexity/sonar`, context low | $0.006 / $0.0065 / $0.008 | **$0.65** ($0.60–0.80) | $1 |
| s2-search, s2-snippets | Semantic Scholar | free | free | – |
| ours | L5w, replayed | free (the checks were paid in Step 2: $0.0071 mean on the dev set) | free | – |
| resolver | OpenAlex singleton, Crossref | free | free | – |

**Assumptions** per query:
- **Claude**:
  - 3–5 searches.
  - Billed input of 30k–120k tokens: results count as input, and every sampling step
    rereads the context. Dynamic filtering trims the results.
  - 2k–6k output tokens (filtering code plus the answer).
  - Central: 70k input, 4k output, 4 searches = $0.14 + $0.04 + $0.04.
- **OpenAI**:
  - 2–5 searches.
  - 10k–50k input tokens (search content at model rates).
  - 2k–8k output tokens, reasoning included.
  - Central: 25k input, 4k output, 3 searches = $0.05 + $0.04 + $0.03.
- **Perplexity**: about 250 input and 800–1,500 output tokens, plus the $0.005 request
  fee.

**Options**:

| Option | Low / central / high per query | 100 queries | Cap needed |
|---|---|---|---|
| Opus 5.5 instead of, or as well as, Sonnet 5.5 | $0.19 / $0.40 / $0.65 | about $40 ($19–65) | about $55, so both Claude models fit under $95 of the credit |
| `openai/gpt-6-luna`, mostly search fees | – | about $3.50 ($2–6) | – |
| `openai/gpt-5.4-mini` | – | about $7 | – |

**Budgets**:
- **Anthropic** (promotional credit, $100, expires 2026-10-18 UTC): Sonnet 5.5 at most
  $40.
- **OpenRouter** ($6.64):
  - tonight's test run: about $0.03 (the Qwen agent cost $0.06 for the 150 live queries
    so far);
  - Perplexity: at most $1;
  - OpenAI: at most $4.50;
  - that leaves at least $1.10.
- **Ledger**: the compete caps are kept apart from `run.budget_cap` and
  `data/cache/llm.sqlite`, which this command never writes.

**Expected latency** per query: Claude 30–90 s, OpenAI 20–60 s, Perplexity 3–10 s,
Semantic Scholar under 2 s plus any 429 waits. Ours had a p50 of 17 s on the dev set.
Systems run one after another, so the full run takes about 3 hours.

## Pre-declared analysis

These are fixed before any paid call.

- **Sample**: 100 queries, stratified as above, from the dev run
  (`runs/live-20261014-122144`) and tonight's test run once it is added to
  `compete.runs`. Seed 42.
- **Primary**: ours vs each system on Hit@5, by the exact McNemar test with Holm
  adjustment across the systems. MRR@10 supports, with paired bootstrap 95% CIs.
- **Also reported** for every system, whatever the outcome:
  - Hit@1 and Hit@10;
  - the unresolved share (an upper bound on fabrication) and the DOI-mismatch share;
  - cost and latency per query;
  - Hit@5 by difficulty bucket;
  - errors, over-cap and unparsed answers.
- **Calibrating the unresolved share**: Semantic Scholar lists real papers from its
  own index, so its unresolved share estimates how often the free routes miss a real
  paper. An arXiv-only preprint named without a link is the known blind spot, since
  Crossref doesn't index arXiv.

## Proposed run, after your go

1. After tonight's test run, add its directory to `compete.runs`. This is a config
   change, committed on this branch. Then run `--dry-run` to fix the final sample.
2. Pilot: `--limit 3` with every system, about $1 in total. Compare the measured cost
   per query with the estimates and check that the answers parse. If a system costs
   more than twice its central estimate, stop and report.
3. Full run of 100 queries (cached pilot answers cost nothing again). Then a short
   report, `docs/reports/phase2-step3-competitive.md`.
4. The landscape doc and the manual comparison sheet (also Step 3 in the spec) are not
   part of this step.

## Decisions for you

1. **Anthropic key**: a workspace-scoped key, or the optional workspace-id header?
2. **Claude model**: Sonnet 5.5 only (default; the same token prices as GPT-6.1 Sol), or
   Opus 5.5 as well with the rest of the credit?
3. **OpenAI**: one of
   - top up OpenRouter by about $15 for GPT-6.1 Sol on all 100 queries;
   - switch to `openai/gpt-6-luna`;
   - keep GPT-6.1 Sol and let the $4.50 cap stop it after about 35–40 queries.
4. **Semantic Scholar**: will you request a free API key? Keep both variants?

## Deviations from the spec

1. **Prompt**:
   - The spec asked for "title, year and link". Your instruction asked for "title,
     year, and DOI or link", plus JSON; I used your wording.
   - I added "the most closely related paper first", so that Hit@1 and Hit@5 mean
     something.
2. **Checking whether a paper exists**:
   - The spec looks every title and DOI up in OpenAlex. Following your instruction,
     only free routes are used: OpenAlex by id or DOI, and Crossref by title. OpenAlex
     title search, which is paid, isn't used.
   - So "unresolved" is an upper bound on fabrication, calibrated by Semantic Scholar's
     rate.
   - arXiv links are looked up in OpenAlex by arXiv's DOI. That is still an OpenAlex DOI
     lookup.
3. **Semantic Scholar**: snippet search is added beside paper search, since paper search
   returns nothing for a whole proposal.
4. **Ours**:
   - L5w, the variant chosen on the dev set.
   - On dev queries its cost and latency are L5's recorded ones, since the dev run has
     no L5w rows. It is the same check; the re-rank takes milliseconds.
5. **Errors and over-cap answers** are left out of the statistics, as live-eval leaves
   out errors. Unparsed answers count as misses.
6. **Storage**:
   - Compete runs write a run directory only. Unlike live-eval, they write no
     results-database table and no MLflow run.
   - Spend is tracked per system in `data/cache/compete.sqlite`, not in
     `llm.sqlite`'s cumulative ledger.

## Addendum: Opus on all 200 checked queries (pre-declared 2026-10-15)

Fixed after the 100-query results ([phase2-step3-results.md](phase2-step3-results.md))
and before any call for the other 100 queries.

- **Your decision**: run Opus on the 100 checked queries that are not in
  `docs/evaluation/compete_sample.csv`, with the same settings (`max_searches` 10, the
  same prompt and tool), capped at $40, within the Anthropic group cap. Ours is replayed
  from the cache.
- **Command**: `python -m rk compete --config configs/phase1.yaml
  --set compete.n_queries=200 --systems ours,claude-opus`.
  - With 200, the sample is the whole frame: every query the dev and test runs checked
    (rank 1: 82, ranks 2–10: 48, ranks 11–100: 37, not found: 33).
  - Opus's answers to the first 100 come from the cache. The dry run counts 100
    uncached calls.
- **Cap**: Opus's `cost_cap` is 69.9621, which is the $29.962188 already spent plus $40.
  The group cap stays at $89, with $42.42 left. If the cap stops the run before 200,
  the over-cap queries are left out of the statistics and counted, and the completed N
  is reported.
- **Primary**: ours vs Opus on Hit@5 over all 200 queries (those both answered), by the
  exact two-sided McNemar test at α = 0.05. This is one test, so there is no Holm
  adjustment. MRR@10 supports, with a paired bootstrap 95% CI.
- **Also reported**:
  - the other 100 queries on their own (descriptive);
  - Hit@1 and Hit@10;
  - Hit@5 by difficulty bucket;
  - the unresolved and DOI-mismatch shares;
  - cost, latency and searches per query;
  - the spend.
- **Weights**: none. The 200 are every checked query, so the bucket mix is the frame's
  rather than the 100-query sample's.
