# Phase 2 Step 1: live retrieval

This step adds live retrieval over OpenAlex: a search client, a candidate pool
with a local rerank, and Phase 1's B5 agent loop running on that pool, plus a
`live:` config section. The design is in [docs/phase2_spec.md](../phase2_spec.md).

- **Branch**: `phase2-step1-live-retrieval`, from `main` at 99e3ef0.
- **Smoke test**: on 3 real Phase 1 queries, the original paper was at **rank 1
  of the live top 20** every time. Each check took 12–20 s and cost $0.005–0.010.

## OpenAlex, as documented on 2026-10-13

I checked the current docs and confirmed the key points with six live probes.
The same notes, with every URL, are at the top of
`src/rk/live/openalex_search.py`.

| Topic | What the docs say | What the probes showed |
|---|---|---|
| Keyword search ([searching](https://help.openalex.org/api/searching/)) | `search=` on `/works` covers title, abstract, full text and keywords. Words are stemmed, and words without AND/OR/NOT are ANDed. Results carry a `relevance_score` and are sorted by it. URLs are capped at about 4 KB. | A 57-word proposal as a keyword query matched 6 off-topic works. |
| Semantic search ([semantic search](https://help.openalex.org/api/semantic-search/), [guide](https://help.openalex.org/guides/semantic-search)) | `search.semantic=` embeds the query and every work's title and abstract with GTE Large EN and ranks by cosine. It uses the first 2,000 characters, returns at most 50 results, and allows 1 request per second. Most filters work with it. It can't be combined with another search parameter or with rerank. | `per_page=100` gives HTTP 400 ("cannot exceed 50"). `page=2` works within the 50. The score is the cosine (0.79–0.93). `has_abstract:true` works. The same proposal found its original paper at rank 1. |
| Filters ([filters](https://help.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists)) | `has_abstract:true`; `to_publication_date` is inclusive, so "published before D" is D − 1 day. | – |
| Paging ([paging](https://help.openalex.org/api/paging/)) | `per_page` is 1–100, and page × per_page can't exceed 10,000; cursor paging goes beyond that. | One page per search is enough here. |
| Cost ([example costs](https://help.openalex.org/access/example-costs/), [pricing](https://help.openalex.org/access/pricing/)) | Per 1,000 calls: singleton lookups are free, list+filter $0.10, search $1, semantic search $1, rerank +$1, content download $10. A free key gives $1 a day, $0.10 without a key, reset at midnight UTC. | Each search reported `meta.cost_usd` 0.001 and 10 credits in the headers. A rejected call cost 0. |
| Rate limits ([authentication](https://help.openalex.org/api-reference/authentication)) | Requests per second: 10 without a key, 30 with a free key, 50 on an annual plan; semantic search 1. Both an empty budget and a per-second overrun return HTTP 429, and a per-second 429 costs nothing. `X-RateLimit-*` headers and `/rate-limit` report the budget. | – |

OpenAlex also offers `rerank=true`, its own reranking of the top 100. We don't
use it, since the spec reranks locally.

## What was built

- **`src/rk/live/openalex_search.py`**:
  - `LiveSearchClient`: keyword and semantic search with `has_abstract:true`
    and an optional "published before" date. Each kind is throttled to its own
    documented rate (keyword 10/s, semantic 1/s), and semantic queries are cut
    to 2,000 characters. Retries and backoff come from Phase 1's
    `OpenAlexClient`.
  - `OpenAlexCache`: every response is cached in SQLite
    (`data/cache/openalex.sqlite`), keyed by its request parameters.
  - Cost is read from `meta.cost_usd`, falling back to the documented $0.001.
- **`src/rk/live/pool.py`, `CandidatePool`**:
  - **Dedupe**: versions of a work are merged with Phase 1's corpus dedupe
    (shared id, shared DOI, or the same long title at most two years apart),
    and every search hit is kept per work.
  - **Rerank**: the pool is ranked exactly as Phase 1's B3 ranks the frozen
    corpus, with the pool as the corpus. It uses the same `BM25Retriever`,
    `DenseRetriever` (Qwen3-Embedding) and `HybridRetriever` (RRF, k = 60)
    classes. A test checks this ranking against Phase 1 on the same documents.
  - **Vectors**: each document is encoded once per check.
  - **Size cap**: at most `max_pool_size` works are kept, in order of first
    arrival.
- **`src/rk/live/pipeline.py`, `live_search()`**:
  - **Agent loop**: Phase 1's `AgentRetriever` runs unchanged, with the same
    prompt and B5 caps (5 queries, 6 LLM calls). Its search tool is
    `LiveRanker`: each query is searched on OpenAlex, the results join the
    pool, and the agent sees the top-10 titles of the pool ranked for that
    query.
  - **Final ranking**: B5's own fusion (RRF over every query's hybrid ranking)
    over the final pool; the top 20 are kept.
  - **Cost cap**: each check has a ledger with a hard $0.10 cap on its method
    cost (OpenAlex plus LLM). Before each search it reserves the documented
    price, and before each agent call Phase 1's pessimistic bound. A call that
    could cross the cap raises `CheckCostCapExceeded`.
  - **Output**: the result records the queries, steps, searches, pool size,
    candidates with their hits, cost, time and encoder device; `to_dict()`
    serialises it.
- **`live:` section** in `configs/phase1.yaml` (`rk.config.LiveConfig`):

| Key | Value |
|---|---|
| `cache_path` | `data/cache/openalex.sqlite` |
| `results_per_query` | 50 |
| `semantic_search` | true |
| `semantic_results_per_query` | 50 |
| `keyword_max_words` | 32 |
| `max_pool_size` | 500 |
| `top_k` | 20 |
| `explain_top_k` | 10 |
| `cost_cap_per_check` | 0.10 |
| `openalex_timeout_s` | 30 |
| `work_types` | article, preprint, conference-paper, book-chapter, dissertation (added after review) |
| `cache_ttl_days` | 7 (added after review) |

  The section is optional, so the Phase 1 commands don't need it.

### Design choices

- **The user's text gets semantic search only.** Keyword search ANDs every
  word, so a whole proposal matches almost nothing: 6 off-topic works in the
  probe. The cut-off is `keyword_max_words` = 32. In the Phase 1 full run,
  agent queries had 6–21 words and proposals 45–77.
- **The agent sees the pool, not one search.** Each query's titles come from
  the cumulative pool ranked for that query, as the spec says ("the top-10
  titles from the live pool").
- **The final ranking re-ranks every query against the final pool.** This
  matches Phase 1, where each query's ranking covered the whole corpus. The
  rankings the agent saw mid-loop each covered only the pool as it was then.
- **The cost cap is on method cost.** Method cost counts cache hits too, so a
  replayed check stops where the original would.

## Tests

- **Mocked**: 25 new tests in `tests/test_live.py`. They use
  `FakeLiveOpenAlex` behind `httpx.MockTransport`: its keyword search ANDs
  stems and its semantic search uses FakeEncoder cosine. The agent is FakeLLM.
  They cover:
  - the parameters each search sends, including the 2,000-character cut and
    the 50-result semantic page;
  - the "published before" filter;
  - the disk cache across clients;
  - per-kind throttling and 429 retries, and the fallback price;
  - dedupe with merged hits, the pool size cap, and encoding once;
  - pool ranking identical to Phase 1's B3 and B5 fusion on the same documents;
  - the full B5 loop on the live pool, including which queries get which
    searches and the B5 caps;
  - cost reporting and a free cached replay;
  - the cost cap on both searches and agent calls;
  - the config checks.
- **Network**: `test_live_openalex_keyword_and_semantic_search`
  (`@pytest.mark.network`) makes one keyword and one semantic search against
  the real API. It **passed** ($0.002).
- **Totals**: `ruff check .` is clean. `pytest`: **395 passed**, 0 failed, 2
  deselected (the network tests). `pytest -m network tests/test_live.py`:
  1 passed.

## Smoke test: 3 real Phase 1 queries

I ran the first three bench queries in qid order (A0000, A0002, A0004; A0001
and A0003 aren't in the bench). The run used:
- the real agent, `qwen/qwen3-235b-a22b-2507` through OpenRouter, with B5 caps;
- live OpenAlex with the free key;
- Qwen3-Embedding-0.6B on the GPU (A100).

Loading the encoder took 13.7 s once and isn't in the per-check times. "The
original paper" counts if its OpenAlex id is the candidate's id or one of its
merged aliases.

| Query | Agent queries | Searches | Pool | Original paper in the live top 20 | Time (s) | Method cost ($) | OpenAlex ($) | LLM ($) |
|---|---|---|---|---|---|---|---|---|
| A0000 | 2, then stop | 5 (2 keyword, 3 semantic) | 169 | **1** | 14.0 | 0.0054 | 0.0050 | 0.00043 |
| A0002 | 2, then stop | 5 (2 keyword, 3 semantic) | 207 | **1** | 11.8 | 0.0053 | 0.0050 | 0.00034 |
| A0004 | 4 (max_queries) | 9 (4 keyword, 5 semantic) | 354 | **1** | 19.7 | 0.0096 | 0.0090 | 0.00056 |

The agent's queries (query 0 is the proposal itself):
- **A0000**, original paper "From search to GenAI queries: Global trends in
  physics information-seeking across topics and regions":
  1. impact of generative AI on student information-seeking behavior in
     science education: longitudinal analysis of search engine and Wikipedia
     traffic
  2. generative AI impact on student science information seeking patterns:
     longitudinal web traffic analysis by country and topic
- **A0002**, original paper "TopicSelectNet: A Hybrid Approach to Abstractive
  Summarization using Dynamic Content Selection":
  1. abstractive legal text summarization using thematic clustering and
     content selection
  2. abstractive legal summarization theme-based clustering content selection
     diversity
- **A0004**, original paper "Textual overlap rather than domain alignment: A
  comparative study of fine-tuning strategies for specialised machine
  translation with large language models":
  1. fine-tuning strategies for Chinese-English political text translation
     selective vs full parameter updates
  2. parameter-efficient fine-tuning political domain adaptation
     Chinese-English machine translation
  3. comparative study full fine-tuning vs parameter-efficient adaptation
     political text translation Chinese English
  4. impact of fine-tuning methods on Chinese-English political text
     translation: distinguishing data repetition from domain similarity

What the smoke test shows:
- **Recall: nothing yet.** Phase 1's B5 also ranked all three at 1 on the
  frozen corpus, so these are easy queries. Live recall over the 500 queries is
  Step 2.
- **Who found the paper.** Semantic search on the proposal found all three
  originals, at OpenAlex ranks 1, 29 and 8. Keyword search on agent queries
  found two of them: A0002 at rank 1, and A0004 at ranks 3 and 2. For A0004,
  no single search ranked the paper first; the fused local rerank put it at 1.
- **Speed and cost** are within the spec's targets (median under 60 s and
  $0.03 per check). OpenAlex is about 90% of the cost: $0.001 per search, so
  5–9 searches cost $0.005–0.009. The agent costs $0.0003–0.0006.
- **Dedupe merged versions.** For A0000, OpenAlex holds four records of the
  original paper (W7128640141, an arXiv DOI, plus W7128648086, W7166562055 and
  W7167075703). The pool merged them, the earliest version kept its id, and the
  bench's target W7166562055 matched as an alias.
- **Some near-duplicates and non-papers get through.**
  - For A0002, ranks 2 and 3 are two records of one work with different titles
    ("BB25HLegalSum: Leveraging BM25 …" and "BB25HLegalSum : a method for
    …"). The title rule doesn't merge them.
  - For A0004, rank 2 is a dataset record, "Minimal replication dataset for
    'Textual Overlap Rather Than Domain Alignment …'".

### Spend

| | Calls | Cost ($) | Where it's counted |
|---|---|---|---|
| Agent LLM, smoke test | 10 | 0.0012 billed | Spend ledger, command `live-smoke`; total now $5.3958 |
| OpenAlex, smoke test | 19 searches | 0.019 | OpenAlex's free daily budget, not our ledger |
| OpenAlex, probes | 5 paid (1 rejected, free) | 0.005 | same |
| OpenAlex, network test | 2 | 0.002 | same |

## After review: answers 1–4

These changes are on the same branch; the smoke test above ran before them and
was not run again.

1. **LangGraph.** The search agent stays the plain B5 loop. `CLAUDE.md` now
   says LangGraph is optional, and only if it adds value to the end-to-end
   check workflow later.
2. **Pool filtering.**
   - **Work types.** Searches ask OpenAlex only for the types in
     `live.work_types` (a `type:` filter), and any other type that comes back
     is dropped as well. The types follow
     [OpenAlex's vocabulary](https://help.openalex.org/data/work-types/):
     proceedings papers are `conference-paper`, and `proceedings-article`
     matched no works. A probe ($0.002) confirmed that the filter works with
     both kinds of search.
     - **Dropped:** datasets, errata, peer reviews and paratext, as asked, and
       with them `review` (literature reviews), `book`, `report`, `letter`,
       `editorial` and the other non-papers.
     - **The A0004 replication dataset** is now dropped. Its title Jaccard
       with the paper is 0.84, and it shares an author and a year, so without
       the type rule it would have merged into the paper as the earlier
       version.
   - **Near-duplicates.** Records with different titles merge only if all
     three hold:
     - title Jaccard ≥ 0.8, on content stems (the tokenisation BM25 and the
       leak filter use);
     - at least one shared author;
     - publication years at most 1 apart.

     Two authors are the same if they share an OpenAlex author id, or if
     every token of the shorter name (2 tokens or more) is in the other name.
     The name rule is needed for the A0002 case, BB25HLegalSum: a 2023
     conference paper and the author's 2024 dissertation. Their title Jaccard
     is 0.9, but the dissertation's only author has no OpenAlex id and is
     written "Andrade, Leonardo Bonalume de". The pair now merges, and the
     2023 paper keeps its id.
   - **Phase 1's merges are unchanged:** shared id or DOI, or the same long
     title at most 2 years apart.
   - The tests use the real metadata of these records, from free singleton
     lookups.
3. **Cache TTL.** The cache now keeps every response it fetches.
   - The live tool reads the newest one if it is younger than
     `live.cache_ttl_days` (7), and fetches again otherwise.
   - Evaluation runs (`LiveSearchClient.from_config(..., evaluation=True)`)
     read the first response ever cached, so they never expire and a rerun
     sees the same data.
   - The cache table is new (`openalex_responses`). The smoke test's
     `responses` table is left in the file but no longer read; its entries
     wouldn't match anyway, since the requests now carry the type filter and
     authors.
4. **Matching the original.** `rk.live.pool.match_original` reports how a
   work is the original paper, checking all its merged versions: by OpenAlex
   id, by any alias on either side, by DOI, or by exact normalised title.
   Step 2's evaluation will use it.
