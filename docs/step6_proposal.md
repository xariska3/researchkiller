# Step 6a: proposed real configuration

Researched on 2026-10-12. **No LLM money was spent.** Sources of the numbers:

- **OpenAlex:** 38 list calls (2 for topic sizes, 16 for candidate counts and abstract samples, 20 to fetch 2,000 real abstracts). They were made without a key, because `OPENALEX_API_KEY` was not visible to this session (not in the process, user or machine environment).
- **OpenRouter:** its public model list (`/api/v1/models`).
- **Model cards and provider docs:** linked below.
- **Encoder timings:** measured on the project's GPU (GRID A100-40C).

All counts are as of 2026-10-12 and will grow until the corpus is built.

## Key numbers

| | Proposal |
|---|---|
| Topics | T10028, T10181, T12031, T10286 (LLMs, NLP, dialogue, retrieval) |
| Sweep corpus | ≈ 41.6k English works with abstracts since 2025-01-01 (+ ≈ 10k references/citers) |
| Eligible targets from `target_from_date` | ≈ 2,650 (need 450 = 1.5 × 300) |
| Paraphraser | `meta-llama/llama-4-maverick`, $0.188 / $0.652 per 1M, cutoff Aug 2024 |
| Judge | `openai/gpt-4.1-mini`, $0.40 / $1.60 per 1M, cutoff 2024-06-01 |
| Agent | `qwen/qwen3-235b-a22b-2507`, $0.09 / $0.55 per 1M, no published cutoff (released 2025-07-21) |
| `target_from_date` | 2025-10-01 |
| Encoder | `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, fp16: 133 docs/s, ≈ 6.5 min for the corpus |
| Cost, 30 queries | bench ≤ $0.07 + run ≤ $0.10 |
| Cost, 300 queries | bench ≤ $0.71 + run ≤ $1.01 (worst case) |
| `budget_cap` | 12 |

## 1. Topics

### Largest topics since 2025-01-01 (works with abstracts)

#### Subfield 1702 (Artificial Intelligence): 495,670 works

| # | Topic | Works |
|---|---|---|
| 1 | T10028 Topic Modeling | 58,213 |
| 2 | T10682 Quantum Computing Algorithms and Architecture | 28,755 |
| 3 | T10764 Privacy-Preserving Technologies in Data | 22,385 |
| 4 | T11689 Adversarial Robustness in Machine Learning | 20,482 |
| 5 | T10462 Reinforcement Learning in Robotics | 19,703 |
| 6 | T13904 Artificial Intelligence Applications | 19,089 |
| 7 | T10020 Quantum Information and Cryptography | 18,736 |
| 8 | T10664 Sentiment Analysis and Opinion Mining | 17,122 |
| 9 | T10862 AI in cancer detection | 16,216 |
| 10 | T12026 Explainable Artificial Intelligence (XAI) | 16,108 |
| 11 | T10181 Natural Language Processing Techniques | 15,121 |
| 12 | T12128 AI in Service Interactions | 12,921 |
| 13 | T11307 Domain Adaptation and Few-Shot Learning | 12,729 |
| 14 | T10201 Speech Recognition and Synthesis | 11,786 |
| 15 | T10100 Metaheuristic Optimization Algorithms Research | 10,773 |
| 16 | T10126 Logic, programming, and type systems | 8,940 |
| 17 | T10237 Cryptography and Data Security | 8,769 |
| 18 | T11273 Advanced Graph Neural Networks | 8,137 |
| 19 | T10320 Neural Networks and Applications | 7,302 |
| 20 | T11902 Intelligent Tutoring Systems and Adaptive Learning | 7,207 |
| 21 | T13702 Machine Learning in Healthcare | 6,769 |
| 22 | T11276 Solar Radiation and Photovoltaics | 6,505 |
| 23 | T12031 Speech and dialogue systems | 6,477 |
| 24 | T11512 Anomaly Detection Techniques and Applications | 6,248 |
| 25 | T11652 Imbalanced Data Classification Techniques | 6,122 |
| 26 | T10215 Semantic Web and Ontologies | 5,748 |
| 27 | T12262 Hate Speech and Cyberbullying Detection | 5,308 |
| 28 | T11424 Security and Verification in Computing | 5,262 |
| 29 | T11010 Logic, Reasoning, and Knowledge | 5,181 |
| 30 | T10951 Cryptographic Implementations and Security | 4,993 |

#### Subfield 1710 (Information Systems): 392,930 works

| # | Topic | Works |
|---|---|---|
| 1 | T14414 Artificial Intelligence in Education | 71,213 |
| 2 | T10270 Blockchain Technology Applications and Security | 30,444 |
| 3 | T10260 Software Engineering Research | 18,358 |
| 4 | T11937 Research Data Management Practices | 17,893 |
| 5 | T10101 Cloud Computing and Resource Management | 17,074 |
| 6 | T10734 Information and Cyber Security | 16,185 |
| 7 | T12338 Digital literacy in education | 16,132 |
| 8 | T10203 Recommender Systems and Techniques | 15,954 |
| 9 | T14516 English Language Learning and Teaching | 14,694 |
| 10 | T12799 Mobile and Web Applications | 12,179 |
| 11 | T10430 Software Engineering Techniques and Practices | 10,608 |
| 12 | T13602 Educational Methods and Media Use | 10,468 |
| 13 | T14496 Educational Technology and E-Learning | 9,815 |
| 14 | T13673 Library Science and Information | 9,703 |
| 15 | T10286 Information Retrieval and Search Behavior | 8,404 |
| 16 | T13373 Data Mining and Machine Learning Applications | 8,137 |
| 17 | T12519 Cybercrime and Law Enforcement Studies | 7,178 |
| 18 | T14216 Multimedia Learning Systems | 7,060 |
| 19 | T11644 Spam and Phishing Detection | 6,721 |
| 20 | T14103 Educational Management and Quality | 6,092 |
| 21 | T12408 Educational Innovations and Challenges | 5,846 |
| 22 | T12395 Educational Innovations and Technology | 5,118 |
| 23 | T12479 Web Application Security Vulnerabilities | 4,525 |
| 24 | T11800 User Authentication and Security Systems | 3,947 |
| 25 | T13386 Educational Methods and Teacher Development | 3,717 |
| 26 | T13081 Digital Education and Society | 3,531 |
| 27 | T11370 Mobile Learning in Education | 3,207 |
| 28 | T13194 ICT in Developing Communities | 3,009 |
| 29 | T13038 Internet of Things and AI | 3,003 |
| 30 | T14498 Educational Technology in Learning | 2,905 |

### Proposed area: language models, NLP, dialogue and retrieval

Four topics, one sweep each:

| Topic | All since 2025-01-01 | English since 2025-01-01 | Eligible from 2025-09-01 | Eligible from 2025-10-01 | Eligible from 2026-01-01 | Abstract 100–450 words |
|---|---|---|---|---|---|---|
| T10028 Topic Modeling | 58,213 | 29,049 | 1,875 | 1,742 | 1,294 | 97% |
| T10181 Natural Language Processing Techniques | 15,121 | 6,181 | 493 | 456 | 351 | 98% |
| T12031 Speech and dialogue systems | 6,477 | 2,582 | 241 | 224 | 160 | 97% |
| T10286 Information Retrieval and Search Behavior | 8,404 | 3,783 | 329 | 300 | 210 | 98% |
| **Total** | **88,215** | **41,595** | **2,938** | **2,722** | **2,015** | |

"Eligible" counts English works with an abstract and at least 10 references
(`referenced_works_count:>9`), published from the given date to today. These
are the target rules in `targets.py` that OpenAlex can filter on. The last
column is the share of a 100-work random sample of those works (seed 42, from
2025-09-01) whose abstract has 100–450 words, the one rule that needs the
abstract text. Expected eligible targets from **2025-10-01**: 2,722 × ≈ 97% ≈
**2,650**, far above the 450 needed (1.5 × 300). Dedupe will lower that a
little.

**Filter strings** (the builder adds `has_abstract:true` and
`to_publication_date:<build date>`):

```
primary_topic.id:T10028,language:en,from_publication_date:2025-01-01
primary_topic.id:T10181,language:en,from_publication_date:2025-01-01
primary_topic.id:T12031,language:en,from_publication_date:2025-01-01
primary_topic.id:T10286,language:en,from_publication_date:2025-01-01
```

**Expected sweep corpus:** ≈ 41.6k English works now, more by the build date.
With `max_works_per_sweep: 35000` no sweep should be truncated; the largest,
T10028, is ≈ 29k. Citation expansion for 450 sampled targets adds an estimated
≈ 10k references and citers, so the indexed corpus will be ≈ 50k documents.

**Why these four:**
- **Coherent and dense.** All four cover language models, NLP methods, dialogue
  and retrieval, so references and near-miss distractors overlap.
- **T10028 is mostly LLM work despite its label.** OpenAlex's classifier puts
  much LLM-application work here. Sampled recent titles: "Automatic
  classification of medical literature leveraging pre-trained large language
  models"; "A unified RAG-based framework for intelligent exploration of Argo
  and BGC-Argo oceanographic data"; "Logic, inference, understanding:
  cross-domain generalization for generative language models".
- **The other three** contribute, for example, "Morphology-Aware Segmentation
  and Tokenization for Turkic Languages" (T10181), "A State-Aware Dual-Tool
  Architecture for Improved Tool Use in Multi-Turn LLM Dialogue" (T12031), and
  "Hi-RAG: A Hierarchical Retrieval-Augmented Generation Framework for
  Scalable and Generalisable Tool Selection" (T10286).
- **Left out:**
  - Sentiment analysis (T10664) and recommender systems (T10203) are large but
    pull the corpus toward applied classification and recommendation.
  - Semantic Web (T10215) and Logic & Reasoning (T11010) are small and only
    loosely related. T11010 also has the lowest abstract-length pass rate
    (78%).

**Observations that matter for the build:**

1. **Only ≈ 47% of works are tagged English** (41,595 of 88,215). The corpus
   drops the rest, so `language:en` belongs in the filter itself: then
   `max_works_per_sweep` counts only usable works.
2. **"At least 10 references" is the rule that binds.** Roughly one in ten
   recent English works passes it, because OpenAlex often lacks reference
   lists, especially for the newest works. The estimate: 2,722 eligible from
   2025-10-01, against ≈ 24k English works in that period at the 2025–2026
   average rate. In the 2,000 newest works (Aug–Oct 2026) the median
   reference count is 0 and only 17% have 10 or more. The abstract-length rule
   passes 97–98%.
3. **Recent targets have few citers.** Targets from late 2025 onwards mostly
   have few or none yet, so the citer part of the expansion will be small.

## 2. Models

Three different families, all via OpenRouter (`https://openrouter.ai/api/v1`,
key in `OPENROUTER_API_KEY`):

| Role | OpenRouter id | Family | $ / 1M in | $ / 1M out | Training cutoff | Source |
|---|---|---|---|---|---|---|
| Judge | `openai/gpt-4.1-mini` | openai | 0.40 | 1.60 | 2024-06-01 | [OpenAI docs](https://developers.openai.com/api/docs/models/gpt-4.1-mini): "Jun 01, 2024 knowledge cutoff"; not a reasoning model |
| Paraphraser | `meta-llama/llama-4-maverick` | llama | 0.188 | 0.652 | August 2024 (config 2024-08-31) | [Model card](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct): "The pretraining data has a cutoff of August 2024"; released 2025-04-05 |
| Agent | `qwen/qwen3-235b-a22b-2507` | qwen | 0.09 | 0.55 | **none published**; release date 2025-07-21 used | [Model card](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507): "supports only non-thinking mode", no cutoff; release date from the [Hugging Face API](https://huggingface.co/api/models/Qwen/Qwen3-235B-A22B-Instruct-2507) (`createdAt` 2025-07-21) |

All three list `temperature`, `max_tokens` and `seed` among OpenRouter's
supported parameters, and none lists `reasoning`. Prices are OpenRouter's
listed prices on 2026-10-12.

**Alternatives considered:**
- **Judge:**
  - GPT-5.x, GPT-6.x and the o-series are reasoning models; OpenRouter lists no
    `temperature` for them, so they fail the requirement.
  - `gpt-4o-mini` ($0.15 / $0.60) is cheaper but weaker, with an Oct 2023
    cutoff.
  - `gpt-4.1` is stronger at 5× the price.
- **Paraphraser:**
  - `meta-llama/llama-3.3-70b-instruct` ($0.22 / $0.50, cutoff Dec 2023) is
    the fallback if Maverick paraphrases poorly in the pilot.
  - Newer Mistral models (`mistral-large-2512`, `mistral-medium-3.1`) cost more
    and would push `target_from_date` later.
- **Agent:**
  - `qwen3-30b-a3b-instruct-2507` ($0.10 / $0.30) is a smaller option.
  - Qwen 3.5 and later are listed with reasoning.

**`target_from_date`: 2025-10-01.**
- The latest cutoff is the agent's release date, 2025-07-21; 2025-10-01 leaves a margin of more than 2 months.
- It still leaves ≈ 2,650 eligible targets.
- If you want a wider margin against the preprint/memorisation risk, 2026-01-01 still leaves ≈ 1,950 (2,015 × 97%).

## 3. Encoder

Encoding the same 2,000 real "title. abstract" texts (newest English works in
the four topics) on the GRID A100-40C. The check retrieves each paper's
abstract (without its title) from the paper's title, among the 2,000; it's a
quality proxy, not the benchmark task:

| Encoder | Revision | Precision | Params | Time for 2,000 | Docs/s | Hit@1 | MRR | Peak GPU memory |
|---|---|---|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 (current) | 5c38ec7c… | fp32 | 33M | 5.3 s | 374 | 0.759 | 0.811 | 0.8 GB |
| BAAI/bge-large-en-v1.5 | d4aa6901… | fp32 | 335M | 38.9 s | 51 | 0.772 | 0.818 | 3.1 GB |
| BAAI/bge-large-en-v1.5 | d4aa6901… | fp16 | 335M | 6.4 s | 314 | 0.774 | 0.821 | 1.1 GB |
| Qwen/Qwen3-Embedding-0.6B | 97b0c614… | fp32 | 596M | 57.8 s | 35 | 0.808 | 0.856 | 8.2 GB |
| **Qwen/Qwen3-Embedding-0.6B** | **97b0c614…** | **fp16** | 596M | **15.0 s** | **133** | **0.809** | **0.857** | 5.2 GB |

The bge-small row used batch 64; the others used batch 32 with
`max_seq_length = 512`. Abstracts fit in 512 tokens: the p90 is 324 words,
about 430 tokens.

**Proposal: `Qwen/Qwen3-Embedding-0.6B`**
- **Revision:** pinned to `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- **Precision and length:** run in fp16 with `max_seq_length = 512`.
- **Quality:** +5.0 points Hit@1 and +4.6 MRR over bge-small on the proxy.
  bge-large gains only +1.5.
- **No remote code:** it works with the installed sentence-transformers 6.1 /
  transformers 5.19.
- **Query prefix:** the model's documented `Instruct: …\nQuery:` format,
  `"Instruct: Given a research proposal, retrieve scientific papers that
  describe the same research idea\nQuery:"`, framed like the agent prompts.

**Extrapolation:** ≈ 50–52k documents ÷ 133 docs/s ≈ **6.5 minutes** in fp16
(fp32 would take ≈ 25 minutes). BM25 indexing takes seconds.

**A code change is needed before using it (not part of this step):**
`SentenceTransformerEncoder` loads models in fp32 with the model's default
maximum length.
- **Out of memory in fp32:** with Qwen3-Embedding at batch 64 and its 32k
  default maximum length, this ran out of GPU memory during this test.
- **Fix:** add two retrieval settings, `encoder_dtype` (e.g. `float16`) and
  `encoder_max_seq_length` (e.g. `512`), pass them to the encoder, and record
  them in the index manifest.

## 4. Cost estimate

**Tokens:** counted with the Qwen3 tokenizer on the real prompt files, rendered
with 300 of the real abstracts and titles. A typical 72-token proposal stands
in for the paraphraser's output. The Llama 4 and GPT-4.1 tokenizers should be
within roughly ±15% for English text.

| Item | Tokens |
|---|---|
| Abstract (mean) | 319 |
| Title (mean) | 21 |
| Paraphrase prompt (mean; a retry adds ≈ 31) | 536 / 567 |
| Each judge prompt (mean) | 549 |
| B4 rewrite prompt | 209 |
| B5 step prompt with k queries in the history (10 real titles each) | k=1: 527, 2: 807, 3: 1,064, 4: 1,332, …, 14: 3,757 |

**Assumed outputs:** paraphrase 100 tokens; judge 30 per call; rewrite 80;
agent step 25. Each chat message also gets 8 tokens of chat-template
overhead.

### Benchmark generation

Each attempt is priced as an upper bound: one paraphrase call plus both judge
calls. Attempts that fail the length or leak checks never reach the judge.

- Paraphraser: 567 in × $0.188/M + 100 out × $0.652/M = $0.000172
- Judge, per call: 549 in × $0.40/M + 30 out × $1.60/M = $0.000268
- One attempt: $0.000172 + 2 × $0.000268 = **$0.000707**

With at most 3 attempts per target and each attempt accepted with
probability p:

| Scenario | Attempts per target | P(target accepted) | Attempts per accepted query | 30 queries | 300 queries |
|---|---|---|---|---|---|
| central, p = 0.5 | 1 + 0.5 + 0.25 = 1.75 | 0.875 | 2.00 | **$0.042** | **$0.42** |
| pessimistic, p = 0.3 | 1 + 0.7 + 0.49 = 2.19 | 0.657 | 3.33 | **$0.071** | **$0.71** |

### Evaluation run

B1–B3 cost nothing. Prices are the agent's, $0.09/M in and $0.55/M out, in the
worst case where every allowed search is used:

- B4: 1 call, 209 in + 80 out = $0.000063
- B5 (`max_queries` 5 → at most 4 agent calls): 527 + 807 + 1,064 + 1,332 = 3,730 in + 100 out = $0.000391
- B5-U (`max_queries` 15 → at most 14 calls): prompts 527 … 3,757, sum 30,139 in + 350 out = $0.002905
- Per query: **$0.003359** → 30 queries **$0.10**, 300 queries **$1.01**

### Totals

| | 30 queries (pilot) | 300 queries (full) |
|---|---|---|
| Bench (pessimistic) | $0.07 | $0.71 |
| Run (worst case) | $0.10 | $1.01 |
| **Total** | **≈ $0.17** | **≈ $1.72** |

Both totals sit far below `budget_cap: 12`, even doubled for tokenizer and
provider-price variance. Reruns from the cache cost nothing, though method
cost is still reported. The ledger reserves a byte-based upper bound per call
before sending; the largest reservation (the 14th B5-U step, ≈ 15 KB of JSON)
is ≈ $0.0016.

**OpenAlex calls for the full build:**
- ≈ 420 sweep pages;
- ≈ 250 reference batches;
- ≈ 700 citer queries.

That is ≈ 1,400 list calls ≈ $0.14, which exceeds the keyless daily budget
($0.10). The build therefore needs `OPENALEX_API_KEY` in the environment ($1/day).

## 5. Proposed config changes (not applied)

The proposed file loads under the current strict loader: three distinct
families, and `target_from_date` later than every cutoff.

```diff
--- a/configs/phase1.yaml
+++ b/configs/phase1.yaml
@@ -5,12 +5,15 @@
 
 corpus:
-  # OpenAlex /works filter strings, one subfield sweep each.
-  # `has_abstract:true` is always added by the corpus builder.
+  # OpenAlex /works filter strings, one sweep per topic (docs/step6_proposal.md).
+  # `has_abstract:true` and `to_publication_date:<build date>` are always added.
   filters:
-    - "primary_topic.subfield.id:1702,from_publication_date:2023-01-01"
+    - "primary_topic.id:T10028,language:en,from_publication_date:2025-01-01"   # Topic Modeling (LLM applications)
+    - "primary_topic.id:T10181,language:en,from_publication_date:2025-01-01"   # NLP Techniques
+    - "primary_topic.id:T12031,language:en,from_publication_date:2025-01-01"   # Speech and dialogue systems
+    - "primary_topic.id:T10286,language:en,from_publication_date:2025-01-01"   # Information Retrieval and Search Behavior
   language: en
   min_abstract_words: 50          # placeholder
   max_citers_per_target: 20       # placeholder
-  max_works_per_sweep: 25000      # per filter string, newest first (publication_date desc)
+  max_works_per_sweep: 35000      # safety net; the largest sweep (T10028) had ~29k on 2026-10-12
   mailto_env: OPENALEX_MAILTO     # optional; OpenAlex has ignored mailto since Feb 2026
   api_key_env: OPENALEX_API_KEY   # optional, but a free key gives 10x the daily budget
@@ -18,6 +21,6 @@ corpus:
 
 bench:
-  n_targets: 200                  # placeholder (the Step 6 pilot uses ~30)
-  target_from_date: 2025-06-01    # must be later than every model's training_cutoff
+  n_targets: 300                  # the pilot is the first ~30 in seeded order (--limit)
+  target_from_date: 2025-10-01    # must be later than every model's training_cutoff
   max_attempts_per_target: 3      # placeholder
   max_distinctive_overlap: 0.15   # fraction of the abstract's top-20 IDF terms (0.15 = 3)
@@ -27,7 +30,7 @@ bench:
 
 retrieval:
-  encoder: BAAI/bge-small-en-v1.5
-  encoder_revision: 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a   # HF commit (main as of 2024-02-22)
-  query_prefix: "Represent this sentence for searching relevant passages: "
+  encoder: Qwen/Qwen3-Embedding-0.6B
+  encoder_revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3   # HF commit (main on 2026-10-12)
+  query_prefix: "Instruct: Given a research proposal, retrieve scientific papers that describe the same research idea\nQuery:"
   max_queries: 5                          # placeholder; B4 and B5
   max_llm_calls_agent: 6                  # placeholder; B5
@@ -40,5 +43,5 @@ run:
   cache_path: data/cache/llm.sqlite   # LLM cache + cumulative spend table
   seed: 42
-  budget_cap: 20.0                # USD; hard cap on real LLM spend across all commands
+  budget_cap: 12.0                # USD; hard cap on real LLM spend across all commands
   results_db: data/results.sqlite # runs + query_results tables, read by `rk report`
   mlflow_tracking_uri: sqlite:///mlflow.db   # repo root; artifacts go to mlartifacts/ beside it
@@ -47,40 +50,40 @@ run:
 # Model roles. All three must come from different model families, and
 # bench.target_from_date must be later than every training_cutoff (both checked
-# at load time). Names, endpoints, prices and cutoffs are placeholders: verify
-# them before Step 6. Where no cutoff is published, the release date is used,
-# since it bounds the cutoff from above.
+# at load time). Prices are OpenRouter's listed prices on 2026-10-12. Where no
+# cutoff is published, the release date is used, since it bounds the cutoff
+# from above. Sources: docs/step6_proposal.md.
 paraphraser:
-  name: qwen/qwen-2.5-72b-instruct
-  family: qwen
+  name: meta-llama/llama-4-maverick
+  family: llama
   base_url: https://openrouter.ai/api/v1
   api_key_env: OPENROUTER_API_KEY
-  price_in_per_m: 0.40
-  price_out_per_m: 0.40
+  price_in_per_m: 0.188
+  price_out_per_m: 0.652
   temperature: 0.7
   max_tokens: 400
   seed: 42
-  training_cutoff: 2024-09-19     # release date; no published cutoff
+  training_cutoff: 2024-08-31     # model card: "a cutoff of August 2024"
 
 judge:
-  name: google/gemma-3-27b-it
-  family: gemma
+  name: openai/gpt-4.1-mini
+  family: openai
   base_url: https://openrouter.ai/api/v1
   api_key_env: OPENROUTER_API_KEY
-  price_in_per_m: 0.20
-  price_out_per_m: 0.40
+  price_in_per_m: 0.40
+  price_out_per_m: 1.60
   temperature: 0.0
   max_tokens: 300
   seed: 42
-  training_cutoff: 2024-08-31     # model card: August 2024
+  training_cutoff: 2024-06-01     # OpenAI docs: "Jun 01, 2024 knowledge cutoff"
 
 agent:
-  name: meta-llama/llama-3.3-70b-instruct
-  family: llama
+  name: qwen/qwen3-235b-a22b-2507
+  family: qwen
   base_url: https://openrouter.ai/api/v1
   api_key_env: OPENROUTER_API_KEY
-  price_in_per_m: 0.20
-  price_out_per_m: 0.40
+  price_in_per_m: 0.09
+  price_out_per_m: 0.55
   temperature: 0.0
   max_tokens: 400
   seed: 42
-  training_cutoff: 2023-12-31     # model card: December 2023
+  training_cutoff: 2025-07-21     # release date (HF createdAt); no published cutoff
```

## Suggested plan for Step 6b

1. Set `OPENALEX_API_KEY` and `OPENROUTER_API_KEY` in the environment the commands run in.
2. Apply the config diff, plus the encoder `encoder_dtype` / `encoder_max_seq_length` code change from section 3.
3. Build the corpus **once with `n_targets: 300`**, which samples 450 targets.
   The pilot and the full run then share one frozen corpus and index. Since
   qids follow the seeded order, the pilot is a random subset of the full
   benchmark.
4. Pilot: `python -m rk make-bench --config configs/phase1.yaml --limit 40 --show`
   should give ≈ 30 accepted queries. Then
   `python -m rk run --config configs/phase1.yaml --limit 30`.
5. Calibrate the placeholders (leak thresholds, budgets) on the pilot, then
   resume the same bench to 300.

## Risks and open points

- **OpenRouter routing:** OpenRouter may route a model to different providers.
  Their prices and quantisation can differ, and some providers ignore `seed`.
  The cache makes reruns reproducible, but first runs can vary. Pinning a
  provider needs a small client change: the `provider` field in the request
  body.
- **Billed vs counted cost:** the ledger counts cost from the config prices,
  not OpenRouter's billed amount. The $12 cap leaves a wide margin over the
  ≈ $2 estimate.
- **Preprints:** a target published after 2025-10-01 may have an earlier
  preprint that a model saw in training. A later `target_from_date`
  (2026-01-01) narrows this risk.
- **Growing counts:** all counts are from 2026-10-12 and will grow until the
  build date.

## Step 6b: what was applied

The proposal was approved with changes, and `configs/phase1.yaml` now holds:

- `target_from_date: 2026-01-01`, which leaves ≈ 2,015 eligible targets
  (≈ 1,950 after the abstract-length rule).
- `n_targets: 500`, so 750 targets are sampled.
- `budget_cap: 12`.
- `language:en` in every filter string.

**Encoder:**
- **dtype and length:** `encoder_dtype: float16` and `encoder_max_seq_length: 512`, both recorded in the index manifest.
- **Queries:** the model card's format, `Instruct: <task>\nQuery:<query>`.
- **Documents:** no prefix.
- **Requirements:** the card requires `transformers>=4.51.0` and `sentence-transformers>=2.7.0`. These are now the lower bounds of the dense extra; 5.19.0 and 6.1.0 are installed.

**Pinned OpenRouter providers**, from each model's endpoint list on 2026-10-12. Each is sent with fallbacks disabled.

| Role | Endpoint | $/M in / out | Why |
|---|---|---|---|
| Paraphraser | `novita/fp8` | 0.27 / 0.85 | Cheapest endpoint that lists `seed`. DigitalOcean ($0.188 / $0.652) doesn't list it; Parasail fp8 and Google Vertex cost more. |
| Judge | `openai` | 0.40 / 1.60 | First party. Azure has the same price. |
| Agent | `parasail/fp8` | 0.14 / 0.80 | 100% uptime over the last 30 minutes. DeepInfra fp8 ($0.09 / $0.55) was degraded (status −5, 53% uptime). GMICloud ($0.087 / $0.35) had no uptime data. |

The proposal's estimate assumed the cheapest endpoints. Here it is again with the pinned prices and the same token counts:

| | 30 queries (pilot) | 500 queries (full) |
|---|---|---|
| Bench, pessimistic ($0.000773 per attempt) | $0.08 | $1.29 |
| Run, worst case ($0.005195 per query) | $0.16 | $2.60 |
| **Total** | **≈ $0.23** | **≈ $3.89** |

**OpenAlex calls for the build**, scaled from section 4 to 750 sampled targets:
- ≈ 420 sweep pages;
- ≈ 420 reference batches;
- ≈ 1,170 citer queries.

That is ≈ 2,000 calls ≈ $0.20, within the keyed daily budget.

**After review:**
- **Strict routing:** the provider routing also sends `require_parameters: true`. All three pinned endpoints list `temperature`, `max_tokens` and `seed` (checked 2026-10-12).
- **Billed cost:** the spend table now stores OpenRouter's billed `usage.cost` in `billed_cost`, next to the computed `cost`. The ledger counts the billed cost when present, which settles the "billed vs counted cost" risk above.
- **Pilot size:** the pilot uses `make-bench --limit 35`.
