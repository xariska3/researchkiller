# Phase 2 Step 3: competitive evaluation results (100 queries)

"Why not just ask ChatGPT?" This step answers it with numbers. On 100 Phase 1
proposals, each system had to name the paper the proposal was paraphrased from:

- the live tool (**ours**, L5w);
- Claude Sonnet 5.5 and Opus 5.5 with web search;
- OpenAI's gpt-6-luna with web search;
- Perplexity Sonar;
- Semantic Scholar's paper search and snippet search.

The design and the analysis were fixed before any paid call, in
[phase2-step3-plan.md](phase2-step3-plan.md) (Pre-declared analysis).

- **Run**: `runs/compete-20261015-012022`, from branch
  `phase2-step3-competitive` at 5a299ac, on 2026-10-15 from 01:20 to 04:22
  UTC.
  - **Command**: `python -m rk compete --config configs/phase1.yaml`.
  - **Reused from the cache**: the 3 Claude answers of the re-pilot, the 3
    OpenAI answers of the pilot, and every Perplexity and Semantic Scholar
    answer from the earlier 100-query run of those systems
    (`runs/compete-20261015-011630`).
- **Queries**: `docs/evaluation/compete_sample.csv`.
  - 100 of the 200 queries that the two live runs checked: the dev set
    `runs/live-20261014-122144` and the test set `runs/live-20261015-000815`.
    61 come from the dev set and 39 from the test set.
  - Stratified by Phase 1 difficulty, the original's rank in Phase 1's B1:
    rank 1: 41, ranks 2–10: 24, ranks 11–100: 19, not found: 16. Seed 42.
- **Same question for every LLM system** (`src/rk/prompts/compete.txt`):
  "Find the existing papers most closely related to this research proposal.
  List up to 10, each with title, year, and DOI or link", as JSON, the most
  closely related first. Semantic Scholar gets the proposal itself as the
  query.
- **Ours**: L5w, the tool's default check, replayed offline from the
  live-eval caches. Its list is the top 10.
  - The replay reproduced all 39 recorded L5w rankings of the test run.
  - Its cost and latency are the live check's.
- **Scoring**:
  - **Found**: a listed paper is the original if its DOI, the OpenAlex
    record it resolves to, or its normalised title matches the original.
    The first match gives the rank: Hit@1/5/10 and MRR@10.
  - **Unresolved**: a listed paper that no free route verifies (OpenAlex by
    id or DOI, arXiv's DOI in OpenAlex, Crossref by title).

## Summary

- **Hit@5**: ours 0.720, gpt-6-luna 0.648 (on 91 queries), Claude Opus
  0.610, Claude Sonnet 0.580, Semantic Scholar snippets 0.390, Perplexity
  0.240, Semantic Scholar paper search 0.000.
- **Pre-declared test** (McNemar on Hit@5, Holm across the six systems):
  - **Clearly ahead** of Perplexity, both Semantic Scholar searches
    (Holm p ≤ 1.5e-7).
  - **Ahead, but not significantly**, of the three web-search LLMs:
    - Sonnet: +0.14 (27 / 13, p = 0.039, Holm 0.115).
    - Opus: +0.11 (26 / 15, p = 0.117, Holm 0.235).
    - gpt-6-luna: +0.07 on its 91 (21 / 15, p = 0.405).
    - At n = 100, none of these gaps survives the correction.
- **The two kinds of system are complementary.**
  - Ours finds the originals that are easy for a lexical search: 39 of 41
    where Phase 1's BM25 had them first, against 25–26 for Claude.
  - The web-search LLMs do better on the hardest queries: where BM25 never
    found the original, Opus got 10 of 16, gpt-6-luna 8 of 14, Sonnet 7 of
    16, and ours 4 of 16.
  - Ours and Opus together put the original in the top 5 on 87 of 100
    queries (exploratory).
- **Ranking**:
  - When an LLM system finds the original, it nearly always lists it first:
    Sonnet in 52 of 58, Opus in 57 of 61, gpt-6-luna in 59 of 59.
  - Ours spreads its finds over ranks 1–10: 54 at rank 1, 72 in the top 5,
    81 in the top 10.
- **Possibly fabricated papers**: the share of listed papers that don't
  resolve, which is an upper bound.
  - Sonnet 8.0%, Opus 6.9%, gpt-6-luna 9.1%, Perplexity 39.4%. Ours: 0, by
    construction.
  - DOIs or links that lead to another paper: 2.5–3.3% for Claude and
    gpt-6-luna, 6.5% for Perplexity.
  - **Calibration**: Semantic Scholar's papers are all real, yet 24.5% of
    them don't resolve by title. Claude's unresolved share among
    title-checked items (28–30%) is close to that baseline. gpt-6-luna's (57%)
    and Perplexity's (50%) are about twice it.
- **Cost and time per query**:

  | System | Cost | Time |
  |---|---|---|
  | Ours | $0.0075 | 18 s |
  | Sonnet | $0.164 (22×) | 22 s |
  | Opus | $0.293 (39×) | 32 s |
  | gpt-6-luna | $0.057 | 44 s |
  | Perplexity | $0.0056 | 5 s |
  | Semantic Scholar | free | 1–5 s |
- **Searches per query** (mean):

  | System | Searches |
  |---|---|
  | Ours | 7.0 OpenAlex searches (4.0 queries) |
  | Sonnet | 7.3 web searches |
  | Opus | 6.8 |
  | gpt-6-luna | 5.2 |
  | Perplexity | not reported (it searches on every request) |
  | Semantic Scholar | one call |
- **OpenAI** reached its cap after 91 answered queries. The other 9 are over
  the cap: they are counted and left out of its statistics.
- **Spend**: $52.32 in all.
  - Anthropic: $46.58 of the $89 group cap.
  - OpenAI: $5.18, of which $5.02 is new spend, under your $5.05.
  - Perplexity: $0.56.

## Finding the original

Errors and answers over a system's cost cap are left out. An answer without
the JSON asked for would count as a miss (Unparsed); there were none.

| System | Answered | Errors | Over cost cap | Unparsed | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|---|---|---|
| **ours** (L5w) | 100 | 0 | 0 | 0 | 0.540 | **0.720** | **0.810** | 0.623 |
| claude-sonnet (Sonnet 5.5, web search) | 100 | 0 | 0 | 0 | 0.520 | 0.580 | 0.580 | 0.544 |
| claude-opus (Opus 5.5, web search) | 100 | 0 | 0 | 0 | 0.570 | 0.610 | 0.610 | 0.585 |
| openai-luna (gpt-6-luna, web search)¹ | 91 | 0 | 9 | 0 | **0.648** | 0.648 | 0.648 | **0.648** |
| perplexity (Sonar, context low) | 100 | 0 | 0 | 0 | 0.210 | 0.240 | 0.250 | 0.222 |
| s2-search (Semantic Scholar paper search) | 100 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| s2-snippets (Semantic Scholar snippet search) | 100 | 0 | 0 | 0 | 0.260 | 0.390 | 0.420 | 0.314 |

¹ "OpenAI smaller model (gpt-6-luna); flagship not tested for budget reasons".
On the same 91 queries, ours has Hit@5 0.714.

- **Semantic Scholar's paper search** returned nothing for any whole
  proposal (0 papers in 100 answers), as the free probe had suggested.
- **Its snippet search** works: it finds the original in the top 10 for 42
  of 100.

## Ours against each system (pre-declared)

On the queries both answered: paired bootstrap 95% CIs, exact McNemar on
Hit@5, and Holm's adjustment across the six systems.

| Comparison | n | ΔHit@5 [95% CI] | ΔMRR@10 [95% CI] | Only ours / only it | McNemar p | Holm p |
|---|---|---|---|---|---|---|
| ours vs claude-sonnet | 100 | +0.140 [+0.020, +0.260] | +0.079 [−0.038, +0.194] | 27 / 13 | 0.039 | 0.115 |
| ours vs claude-opus | 100 | +0.110 [−0.020, +0.230] | +0.038 [−0.088, +0.159] | 26 / 15 | 0.117 | 0.235 |
| ours vs openai-luna | 91 | +0.066 [−0.066, +0.198] | −0.031 [−0.156, +0.097] | 21 / 15 | 0.405 | 0.405 |
| ours vs perplexity | 100 | +0.480 [+0.360, +0.600] | +0.401 [+0.291, +0.511] | 54 / 6 | 9.7e-11 | 4.9e-10 |
| ours vs s2-search | 100 | +0.720 [+0.630, +0.800] | +0.623 [+0.536, +0.705] | 72 / 0 | 4.2e-22 | 2.5e-21 |
| ours vs s2-snippets | 100 | +0.330 [+0.230, +0.430] | +0.309 [+0.208, +0.408] | 36 / 3 | 3.6e-8 | 1.4e-7 |

- **Against Claude and gpt-6-luna**: ours leads on Hit@5 in every
  comparison, but no lead survives Holm.
  - Against Sonnet, the unadjusted p is 0.039, and the Hit@5 CI excludes
    zero.
  - **MRR@10** doesn't separate ours from any of the three. It is even
    slightly lower than gpt-6-luna's: when gpt-6-luna finds the original, it
    lists it first.
- **Against Perplexity and Semantic Scholar**: ours is clearly better.

## Hit@5 by difficulty (the original's rank in Phase 1's B1)

| System | 1 (n=41) | 2-10 (n=24) | 11-100 (n=19) | not found (n=16) |
|---|---|---|---|---|
| ours | **0.951** (39) | **0.750** (18) | 0.579 (11) | 0.250 (4) |
| claude-sonnet | 0.634 (26) | 0.542 (13) | 0.632 (12) | 0.438 (7) |
| claude-opus | 0.610 (25) | 0.500 (12) | **0.737** (14) | **0.625** (10) |
| openai-luna (n = 37 / 23 / 17 / 14) | 0.676 (25) | 0.609 (14) | 0.706 (12) | 0.571 (8) |
| perplexity | 0.244 (10) | 0.250 (6) | 0.263 (5) | 0.188 (3) |
| s2-search | 0 | 0 | 0 | 0 |
| s2-snippets | 0.512 (21) | 0.583 (14) | 0.211 (4) | 0 |

Hits are in brackets.

- **Ours tracks lexical difficulty.** It falls from 0.95 to 0.25 as Phase
  1's BM25 finds the original less easily.
- **The web-search LLMs barely do.** Opus does better on the hardest two
  buckets than on the easiest. Web search engines match on more than the
  proposal's words (titles, citations, popularity), so a paraphrase that
  defeats BM25 doesn't defeat them as often.
- **Together (exploratory)**: the original is in ours' or the other
  system's top 5 for:
  - 87 of 100 queries with Opus;
  - 85 with Sonnet;
  - 80 of 91 with gpt-6-luna.

  The three LLM systems alone, together, reach 75.
- **Publication date (exploratory)**: on the 12 most recent originals
  (published September–October 2026, a few weeks before the run), ours has
  10 and the LLM systems 5–7. Earlier quarters show no clear pattern.

## Listed papers that don't resolve (possibly fabricated)

| System | Papers listed | Per answer | Unresolved | Share | DOI/link to another paper | Share | Title-checked items | Unresolved among them |
|---|---|---|---|---|---|---|---|---|
| ours | 1000 | 10.0 | 0 | 0 | 0 | 0 | – | – |
| claude-sonnet | 981 | 9.8 | 78 | 8.0% | 25 | 2.5% | 262 (27%) | 29.8% |
| claude-opus | 993 | 9.9 | 69 | 6.9% | 33 | 3.3% | 249 (25%) | 27.7% |
| openai-luna | 836 | 9.2 | 76 | 9.1% | 28 | 3.3% | 134 (16%) | 56.7% |
| perplexity | 964 | 9.6 | 380 | 39.4% | 63 | 6.5% | 761 (79%) | 49.9% |
| s2-snippets | 1000 | 10.0 | 245 | 24.5% | 0 | 0 | 1000 (100%) | 24.5% |

- **Unresolved is an upper bound on fabrication.** Only free routes are used:
  1. an OpenAlex id, or the DOI or arXiv id in the answer, looked up in
     OpenAlex;
  2. failing that, the title in Crossref.

  Most LLM-listed papers carried a DOI or arXiv link and resolved that way.
- **Title-checked items** are those that fell through to Crossref's title
  search: verified there, or unresolved.
  - **The calibration**: Semantic Scholar's snippet papers are all real and
    come without a DOI, so all of them go through the title search. 24.5% of
    them fail it, which is how often that route misses a real paper.
  - **Claude**: among its title-checked items, the miss rate (28–30%) is
    close to that baseline. Most of its 7–8% unresolved papers are probably
    real papers the free routes miss.
  - **gpt-6-luna (57%) and Perplexity (50%)**: about twice the baseline.
    They likely list papers that don't exist, or web pages that aren't
    papers.
  - **Caveat**: LLM titles can be slightly reworded, which also makes a real
    paper fail the title match. So even this comparison is an upper bound.
- **A DOI or link that leads to another paper** is a definite error.
  - In 2.5–3.3% of the LLM-listed papers, and 6.5% of Perplexity's, the
    identifier given points elsewhere.
  - Example from the pilot: Sonnet gave 10.18653/v1/2020.emnlp-main.696 for a
    paper whose DOI is …main.51.

## Searches per query

| System | What a search is | Mean / p50 / p95 / max | Total | Notes |
|---|---|---|---|---|
| ours | OpenAlex search (keyword or semantic) | 7.0 / 7 / 9 / 9 | 702 | 4.0 queries per check: the proposal (semantic search only) and 3.0 agent queries (keyword and semantic each) |
| claude-sonnet | web search (server tool) | 7.3 / 7 / 10 / 10 | 730 | max 10; 1 answer with no search; 9 answers had a batch refused at the limit |
| claude-opus | web search (server tool) | 6.8 / 6 / 10 / 10 | 681 | max 10; 1 answer with no search; 47 answers had a batch refused at the limit |
| openai-luna | `web_search_call` items | 5.2 / 5 / 6 / 6 | 469 | `max_tool_calls` 5, but up to 6 items per answer, counted as returned |
| perplexity | – | not reported | – | Sonar searches on every request |
| s2-search, s2-snippets | one API call | 1 | 100 each | |

- **How Claude searches**: with dynamic filtering, Claude writes code that
  sends its searches in batches of 6–8. A batch that would take the
  answer's searches past `max_uses` is refused whole.
  - At 10, Opus still hit the limit in 47 of 100 answers. It wanted more
    than 10 searches about half the time.
  - Sonnet hit it in 9.
  - The searches counted above are those that ran.
- **Unequal limits**: Claude got 10 searches, gpt-6-luna 5 tool calls (see
  [Pilots](#pilots-and-changes-before-the-full-run)).

## Cost and latency per query

Method cost at the config prices, cached answers included (mean / p50 / p95).

| System | Cost ($) | Latency (s) | Spent in this run ($) | Spent in total ($), all compete runs |
|---|---|---|---|---|
| ours | 0.0075 / 0.0076 / 0.0096 | 18.3 / 18.4 / 27.5 | 0 (replayed) | – (the live checks cost $0.0075 each in Step 2) |
| claude-sonnet | 0.164 / 0.160 / 0.200 | 21.6 / 20.3 / 25.2 | 15.96 | 16.62 of 40 |
| claude-opus | 0.293 / 0.280 / 0.421 | 32.1 / 32.0 / 43.5 | 28.41 | 29.96 of 55 |
| openai-luna | 0.057 / 0.056 / 0.066 | 44.2 / 44.0 / 58.0 | 5.02 | 5.18 of 5.21 |
| perplexity | 0.0056 / 0.0056 / 0.0057 | 5.0 / 4.9 / 6.5 | 0 (cached) | 0.56 of 1 |
| s2-search | 0 | 0.7 / 0.7 / 0.8 | 0 | – |
| s2-snippets | 0 | 4.5 / 4.5 / 6.0 | 0 | – |

- **Ours is the cheapest system that searches well**: 22× cheaper than
  Sonnet, 39× cheaper than Opus and about 8× cheaper than gpt-6-luna.
  - It is about as fast as Sonnet (about 20 s at the median, counting its
    proposal search).
  - It is faster than Opus and gpt-6-luna.
- **Ours' latency is slightly low.** It leaves out the proposal's own search,
  which the live runs read from L0's or L1's cache. That search takes about
  1.4–2.4 s: the mean latency of L0 and of L1.
- **Latency is the API call's time.** It excludes our resolver lookups,
  which are free.
- **Against the plan's estimates** (central, per query):
  - Sonnet $0.164 against $0.22;
  - Opus $0.293 against $0.40;
  - gpt-6-luna $0.057 against $0.035, 1.6× and under the 2× rule;
  - Perplexity $0.0056 against $0.0065.

## Examples

### Ours found the original, no other system did (3 of 11 such queries)

Every system answered these three. "Found" means anywhere in the list.

- **A0054** (Phase 1 B1 rank 1). Original: "The degree of knowledge gain
  metric: Evaluating how searchers acquire new information while searching
  online", published 2026-09-22.
  - **Proposal**: "We propose a new way to measure how much a person learns
    while looking for information on the internet. By applying principles from
    information theory to user behavior, we will estimate the decrease in
    uncertainty without needing surveys or tests…"
  - **Ours**: rank 1.
  - **The others**: Sonnet (7 searches), Opus (8) and Perplexity all put a
    neighbouring paper first, "Evaluating Knowledge Gain in Search
    Environments: An Exploratory Study…". gpt-6-luna (6 searches) listed
    knowledge-gain papers from 2018–2021.
  - **Why it's likely missed**: the original was three weeks old, already in
    OpenAlex but apparently not yet found by web search.
- **A0257** (B1: not found). Original: "Dissecting GraphRAG: A Modular
  Analysis of Knowledge Structuring for Factoid Question Answering" (TACL,
  2026).
  - **Proposal**: "We intend to examine how different architectural
    components affect a retrieval-augmented system for answering factual
    queries using structured data…"
  - **Ours**: rank 1, in the bucket where Phase 1's BM25 never found it.
  - **The others** listed the well-known GraphRAG papers instead: "From Local
    to Global", "In-depth Analysis of Graph-based RAG in a Unified
    Framework", LightRAG, StructRAG.
- **A0284** (B1 ranks 2–10). Original: "Query-adaptive graph
  retrieval-augmented generation for traceable power-quality knowledge
  reasoning", published 2026-09-17.
  - **Ours**: rank 1.
  - **The others**: Sonnet, Opus and gpt-6-luna all found close neighbours
    (HG-RAG and KG-Anchored RAG for power systems) but not the original.
    Perplexity listed general power-quality pages.

### Others found the original, ours didn't (3 of 15 such queries)

- **A0119** (B1 ranks 11–100). Original: "Event-based news embedding:
  leveraging entities, themes, and historical context" (2026).
  - **The others**: Sonnet, Opus and gpt-6-luna listed it first, Perplexity
    second and S2 snippets third.
  - **Ours missed it on ranking.** OpenAlex's semantic search on the
    proposal returned it at rank 47 of 50, and L1's rerank lifted it to 14.
    But the final fusion of L5 and L5w left it out of the top 10.
- **A0218** (B1 ranks 2–10). Original: "BanglaVoice: A curated sentence-level
  annotated dataset for active, passive, and middle voice in Bangla…" (2026).
  - **The others**: Opus, gpt-6-luna and Perplexity listed it first, Sonnet
    third and S2 snippets seventh.
  - **Ours missed it on retrieval.** None of our 9 OpenAlex searches
    returned it; the pool held 307 other works.
- **A0147** (B1: not found). Original: "Factually Consistent Prompting with
  LLMs for Cross-Lingual Dialogue Summarization" (2026).
  - **The others**: all four LLM systems listed it first.
  - **Ours missed it on ranking.** OpenAlex returned it at 23; L1 ranked it
    12 and L5 13. L5w's top 10 doesn't include it.

The pattern across the 15:
- **The LLM systems' web search often lands on the exact paper**, and they
  then list it first.
- **Ours more often has the paper in its pool but ranks it 11th or lower.**
  That's the same ranking gap that separates live L5 from Phase 1's B5
  ([phase2-step2-full.md](phase2-step2-full.md)).

## Pilots and changes before the full run

1. **Pilot, all systems, 3 queries** (`runs/compete-20261015-005927`, Claude
   at `max_searches` 5).
   - **Cost**: every system was under 2× its central estimate:
     - Sonnet $0.066 a query;
     - Opus $0.209;
     - gpt-6-luna $0.052;
     - Perplexity $0.0056.
   - **The problem**: Claude's first code batch (6–8 searches) went over
     `max_uses` 5 and was refused whole, with "Server tool use limit exceeded
     during code execution". This happened in all 6 Claude answers.
     - Sonnet answered 2 of 3 with no search at all, and gave 3 DOIs that
       point to other papers.
     - I stopped and reported before the full run.
2. **Your decisions**, committed in 5a299ac (config only):
   - Both Claude systems get `max_searches` 10, keeping dynamic filtering.
   - OpenAI's cap allows $5.05 of new spend, with no top-up.
3. **Claude re-pilot, 3 queries** (`runs/compete-20261015-011624`):
   - Sonnet $0.153 a query (max $0.163; stop above $0.44), with 6–10 searches.
   - Opus $0.308 a query (max $0.467; stop above $0.80), with 5–10
     searches.
   - Both passed the 2× rule.
4. **Semantic Scholar, Perplexity and ours**: run on all 100 while you
   decided (`runs/compete-20261015-011630`), and reused from the cache in the
   full run. The numbers are identical.
5. **Full run**: all systems, one run directory.

## Spend

| Account | System | Pilot | Re-pilot | Full runs | Total | Cap |
|---|---|---|---|---|---|---|
| Anthropic (credit) | claude-sonnet | 0.1984 | 0.4596 | 15.9577 | 16.6157 | 40 |
| Anthropic (credit) | claude-opus | 0.6255 | 0.9247 | 28.4120 | 29.9622 | 55 |
| Anthropic group | both | | | | 46.5778 | 89 |
| OpenAI | openai-luna | 0.1560 | – | 5.0213 | 5.1773 | 5.2059 (pilot + $5.05) |
| OpenRouter | perplexity | 0.0169 | – | 0.5475 | 0.5644 | 1 |
| Semantic Scholar, Crossref, OpenAlex lookups | – | | | | 0 | – |

- **Total**: $52.32.
- **The first pilot's Claude answers** ($0.82) are not used in the results.
  The re-pilot changed the request, so they weren't reused.
- **What's left**:
  - Anthropic: $42.42 of the group cap, from the credit that expires on
    2026-10-18 UTC.
  - OpenAI: $0.03 of its cap.
- **No LLM spend in `data/cache/llm.sqlite`**: compete keeps its own ledger
  in `data/cache/compete.sqlite`.

## Deviations from the plan

1. **Claude's search limit**: 10 for Claude (your decision after the pilot),
   5 tool calls for gpt-6-luna. The comparison is therefore not at equal
   search budgets. Each system's actual counts are reported above.
2. **OpenAI completed 91 of 100**: the cap stopped it as you specified. Its
   9 over-cap queries are counted and left out, so its comparison is on 91
   queries.
3. **More than one run directory**: the pilot, the re-pilot and the early
   run of the unaffected systems are separate. The full run reused their
   cached answers, and every number above comes from the one full run.
4. **Report name**: `phase2-step3-results.md`, as you asked; the plan said
   `phase2-step3-competitive.md`.
5. **Extra exploratory analyses**, not pre-declared and labelled as such:
   - ours and each system together;
   - the title-route calibration of the unresolved share;
   - the publication-date breakdown.
6. **Ours is L5w** as planned. The test run didn't confirm L5w over L5
   ([phase2-step2-test.md](phase2-step2-test.md), on branch
   `phase2-step2-test`). L5 replayed instead might score slightly
   differently. This wasn't run.

## Open questions

1. **Power**: with 100 queries, ours' lead over Claude and gpt-6-luna doesn't
   survive Holm. The other 100 checked queries would double n.
   - **Cost**: about $16 for Sonnet and $29 for Opus, more than the $42.42
     left on the Anthropic credit for both. One Claude model would fit.
   - **Deadline**: the credit expires on 2026-10-18 UTC.
   - Run one?
2. **Equal search budgets**: rerunning gpt-6-luna at 10 tool calls would
   need an OpenAI top-up of roughly $8–10 (my estimate). Worth it, or report
   the asymmetry as is?
3. **A combined mode**: ours plus one web-search LLM call would have put the
   original in the top 5 for 85–87 of 100 queries, against 72 for ours
   alone. It would cost about $0.17 (with Sonnet) instead of $0.0075.
   Should a pre-declared hybrid config be a later step?
4. **Ours' ranking misses**: in 2 of the 3 reverse examples, OpenAlex
   returned the original but our final ranking dropped it. This is the
   Step 2 gap again. Should improving the rerank come before the explainer
   step?
