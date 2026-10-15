# Phase 2 Step 4: LLM reranking of the live check

The live check often had the original in its pool but below the top 5. This step tries
an LLM reranker on top of it. Every live check is replayed from its cache, the union of
L5w's and L5's top 30 goes to the reranker, and the rerank variants are scored against
L5w, the default check. The variants and the analysis were pre-declared in
[docs/phase2_spec.md](../phase2_spec.md) (Evaluation, rerank variants; commit 6b404fb)
before any code was written or call made. One variant was chosen on the dev set and
confirmed on the test set, the same discipline as L5w's choice in Step 2.

- **Branch**: `phase2-step4-rerank`, from `main` at f33f637 (PRs #11–#13 merged).
- **Command**: `python -m rk rerank --config configs/phase1.yaml RUN_DIR [--variants ...]
  [--limit N]` (`src/rk/live/rerank.py`, built at 4b6f7ed).
- **Runs**, each from a clean tree:
  - **Pilot**: `runs/rerank-20261015-131744`, the first 3 checks of the Step 2 pilot run
    (`runs/live-20261014-114302`, Phase 1 tuning targets, neither dev nor test), every
    variant.
  - **Dev**: `runs/rerank-20261015-132644`, the 120 checks of
    `runs/live-20261014-122144`, every variant (at 4b6f7ed).
  - **Test**: `runs/rerank-20261015-140923`, the 80 checks of
    `runs/live-20261015-000815`, every variant (at 36344eb, after the dev choice was
    committed).
- **Replay**: no OpenAlex or agent calls. The replay reproduced all 120 recorded L5
  rankings on dev, and all 80 L5 and 80 L5w rankings on test.

## The variants

- **Candidates**: L5w's and L5's top 30, taken in turn (L5w's first, then L5's first,
  and so on), without repeats. There were 30 to 48 per check, 36 on average.
- **What the reranker sees**: the proposal, and each candidate's number, title and
  abstract (cut at 2,000 characters). No years, authors or venues.
- **Product variants**: the agent model, `qwen/qwen3-235b-a22b-2507`, with a training
  cutoff of 2025-07-21, before every original. Temperature 0, seed 42.
  - **R1, listwise**: one call returns every candidate number, closest first.
  - **R2, listwise + first stage**: R1's order fused with the presented order by RRF.
    It has no call of its own.
  - **R3, graded**: one call scores every candidate 0–3 (3 = the same research idea).
- **RS, the upper bound**: R1's prompt with Claude Sonnet 5.5. It is not a product
  candidate.
- **The rest of the list**: L5w's other works follow the reranked candidates. Ranks
  count within the top 20, so MRR is MRR@20.

## Summary

- **The test decides for the reranker.** L5r = R1 (listwise, chosen on dev) beats L5w
  on the test set:
  - **Hit@5**: 0.863 against 0.700, +0.163 [95% CI +0.087, +0.250]. It gains 13 queries
    and loses none, McNemar p = 0.00024.
  - **MRR@20**: +0.228 [+0.149, +0.312].
  - **Hit@1**: 0.825 against 0.537.
  - By the pre-declared rule (significant and positive), **the reranker joins the
    default check**: default = L5w + R1, i.e. L5r.
- **The dev set showed the same**: R1 0.850 against L5w's 0.750 (13 / 1, p = 0.002).
- **Sonnet adds almost nothing over Qwen.** RS vs R1: +1 query on dev and +1 on test
  (p = 1 both).
  - **Both are at the ceiling.** When the original is among the candidates, R1 puts
    it in the top 10 every time on test (70 of 70) and on 102 of 103 dev queries.
  - **A stronger reranker can't help more.** What is left is the 27 of 200 queries
    (13.5%) whose original isn't among the candidates.
- **The remaining misses are retrieval misses.** Of the 27 queries (dev and test) whose
  original isn't among the candidates, 24 never came back from any OpenAlex search; 3
  came back but ranked below 30 in both fusions (exploratory).
- **The live gap to Phase 1 is closed (exploratory).** On the same queries, L5r's
  Hit@5 is 102 of 120 against Phase 1's frozen-corpus B5 at 104 on dev, and 69 against
  68 of 80 on test. In Step 2, L5w trailed B5 by 15 points on test.
- **Cost and time**: the rerank call adds $0.0014 and about 6 s to a check.
  - **Cost**: a whole check costs $0.0085–0.0092 on average and at most $0.0115, under
    the $0.10 cap.
  - **Time**: the call takes 5.7–6.1 s on average (p95 8.2–8.5 s).
  - **RS**: $0.032 a call, 23 times R1, but 2 s.
- **No sign of memorisation by Sonnet.** On originals published from July 2026, which
  Sonnet can't have read, RS and R1 have the same Hit@5: 30 of 31 on dev and 21 of 24
  on test.
- **Spend**:
  - Qwen: $0.59, in `llm.sqlite`, now $6.07 of $12.
  - Sonnet: $6.53 of its $20 cap.

## Finding the original

| Config | Dev Hit@1 | Dev Hit@5 | Dev Hit@10 | Dev MRR@20 | Test Hit@1 | Test Hit@5 | Test Hit@10 | Test MRR@20 |
|---|---|---|---|---|---|---|---|---|
| L5 (replayed) | 0.483 | 0.700 | 0.758 | 0.570 | 0.525 | 0.688 | 0.762 | 0.596 |
| L5w, the default check (replayed) | 0.508 | 0.750 | 0.817 | 0.617 | 0.537 | 0.700 | 0.775 | 0.614 |
| **R1 listwise = L5r** | **0.833** | 0.850 | 0.850 | **0.842** | 0.825 | 0.863 | **0.875** | 0.842 |
| R2 listwise + first stage | 0.592 | 0.808 | 0.850 | 0.688 | 0.600 | 0.812 | 0.850 | 0.688 |
| R3 graded | 0.675 | 0.850 | 0.850 | 0.758 | 0.688 | 0.850 | 0.875 | 0.765 |
| RS (Sonnet, upper bound) | 0.825 | **0.858** | **0.858** | 0.840 | **0.863** | **0.875** | 0.875 | **0.869** |
| *Ceiling: original among the candidates* | | *0.858* | | | | *0.875* | | |
| *Phase 1 B5, frozen corpus (same queries)* | *0.683* | *0.867* | *0.908* | *0.760* | *0.637* | *0.850* | *0.925* | *0.730* |

- **Errors and unreadable replies**: none in any variant, on any set. Every reply was
  JSON and listed or scored the candidates, and every finish reason was `stop`.
- **The baselines match Step 2**: L5w's dev numbers are the F1 numbers from Step 2's
  replay, and its test numbers are the test run's live ones.

## Selection on the dev set (pre-declared)

- **R1 and R3 tie on Hit@5** at 0.850, both above L5w's 0.750.
- **R1 wins the tie-break**: MRR@20 0.842 against 0.758. So **L5r = R1**, committed in
  `rk.live.rerank.RERANK_CHOICE` (36344eb) before the test run.
- **RS** had the highest Hit@5 (0.858), but it isn't eligible.

| Dev comparison (exploratory) | n | ΔHit@5 [95% CI] | ΔMRR@20 [95% CI] | Only A / only B | McNemar p |
|---|---|---|---|---|---|
| R1 vs L5w | 120 | +0.100 [+0.042, +0.158] | +0.225 [+0.166, +0.286] | 13 / 1 | 0.0018 |
| R2 vs L5w | 120 | +0.058 [+0.017, +0.108] | +0.071 [+0.046, +0.100] | 8 / 1 | 0.039 |
| R3 vs L5w | 120 | +0.100 [+0.042, +0.158] | +0.141 [+0.099, +0.187] | 13 / 1 | 0.0018 |
| RS vs L5w | 120 | +0.108 [+0.058, +0.167] | +0.223 [+0.166, +0.283] | 13 / 0 | 0.00024 |
| RS vs R1 | 120 | +0.008 [+0.000, +0.025] | −0.001 [−0.021, +0.017] | 1 / 0 | 1 |

## The test (pre-declared)

| Test comparison | n | ΔHit@5 [95% CI] | ΔMRR@20 [95% CI] | Only A / only B | McNemar p |
|---|---|---|---|---|---|
| **L5r (R1) vs L5w (decides)** | 80 | **+0.163 [+0.087, +0.250]** | **+0.228 [+0.149, +0.312]** | **13 / 0** | **0.00024** |
| R2 vs L5w (exploratory) | 80 | +0.113 [+0.050, +0.188] | +0.074 [+0.040, +0.115] | 9 / 0 | 0.0039 |
| R3 vs L5w (exploratory) | 80 | +0.150 [+0.075, +0.237] | +0.151 [+0.098, +0.210] | 12 / 0 | 0.00049 |
| RS vs L5w (exploratory) | 80 | +0.175 [+0.100, +0.263] | +0.255 [+0.175, +0.341] | 14 / 0 | 0.00012 |
| RS vs L5r (exploratory) | 80 | +0.013 [+0.000, +0.037] | +0.027 [+0.000, +0.061] | 1 / 0 | 1 |

- **Decision**: L5r's Hit@5 is significantly above L5w's (p = 0.00024 < 0.05), so the
  reranker joins the default check, as pre-declared.
- **Where the 13 gains came from**: L5w had the original at ranks 6–18 in 11 of them,
  and below its top 20 in 2. R1 puts it first in 12 of the 13 and fifth in one
  (A0216).
- **The union with L5 mattered twice.** In A0249, L5w had the original below its top
  20 and L5 at 16; in A0252, neither had it in the top 20, only in the top 30. In both,
  R1 ranked it first.
- **R1 moved no original down out of the top 5** on test. On dev it did so once
  (A0109, below).

## How the variants differ

- **Listwise is the better form for Hit@1.** R1 puts the original first in 82–83% of
  queries, R3 in 67–69%.
  - **R3's ties**: in half the checks it gives a 3 to more than one candidate (2.5 on
    average), and ties go to the presented order.
  - The original lost such a tie to a near neighbour in 20 dev and 14 test checks.
  - **R2's fusion**: it keeps the first stage's vote, which pulls the original back
    down when the first stage had it low. It is the weakest of the three, but never
    worse than L5w on test.
- **R1's dev loss is a data error, not a ranking error.** In A0109, the original's
  OpenAlex record (W7132857027) has the title of one paper ("A Career Interview
  Dialogue System using Large Language Model-based Dynamic Slot Generation") and the
  abstract of another (Typed Petri Nets for neural networks).
  - The proposal paraphrases the abstract. R1 ranked the record low, likely because of
    the unrelated title, while RS put it second.
  - The benchmark is frozen, so the query stays in.
- **Examples of gains**:
  - **A0156** (dev): Chinese building fire-protection codes, recognising domain
    entities for compliance checking. L5w had the original at 19, R1 at 1.
  - **A0193** (test): phrase-level embedding expansion for Korean multi-vector
    retrieval. L5w had it at 18, R1 at 1.

## Hit@5 by the original's publication date

Sonnet 5.5's training data runs to June 2026, so it may have read originals published
before July (pre-declared split).

| Config | Dev, before 2026-07 (n = 89) | Dev, from 2026-07 (n = 31) | Test, before 2026-07 (n = 56) | Test, from 2026-07 (n = 24) |
|---|---|---|---|---|
| L5w | 63 | 27 | 38 | 18 |
| R1 = L5r | 72 | 30 | 48 | 21 |
| R3 | 72 | 30 | 47 | 21 |
| RS | 73 | 30 | 49 | 21 |

- **No sign of memorisation**: RS and R1 tie on the originals Sonnet can't have seen,
  and RS's one-query lead on each set is in the earlier part.

## Cost and latency per check

| Config | Rerank call: mean / p95 / max | Whole check: mean / max | Over the $0.10 cap | Rerank call latency: mean / p95 | Tokens in / out (mean) |
|---|---|---|---|---|---|
| L5w (dev / test) | – | $0.0071 / $0.0096; $0.0077 / $0.0097 | 0 | – | – |
| R1 = L5r, dev | $0.00139 / $0.00167 / $0.00185 | $0.0085 / $0.0114 | 0 | 5.7 s / 8.5 s | 9,135 / 138 |
| R1 = L5r, test | $0.00144 / $0.00165 / $0.00191 | $0.0092 / $0.0115 | 0 | 6.1 s / 8.2 s | 9,471 / 139 |
| R3, test | $0.00153 / $0.00173 / $0.00202 | $0.0093 / $0.0116 | 0 | 9.5 s / 13.1 s | 9,487 / 248 |
| RS, test | $0.0329 / $0.0385 / $0.0426 | $0.0407 / $0.0521 | 0 | 2.1 s / 2.9 s | 15,885 / 117 |

- **The whole check's cost** is the recorded live check's (OpenAlex and agent), plus
  the rerank call.
- **L5r costs about 20% more than L5w** and stays far under the spec's $0.03 median
  and $0.10 cap.
- **Time**: Step 2's test run measured a standalone check at 22.9 s median (32.8 s at
  p95). With the rerank call, L5r is about 29 s median, under the spec's 60 s.
- **RS answered without visible thinking**: about 120 output tokens in about 2 s, on
  Anthropic's OpenAI-compatible endpoint with its defaults.
  - Thinking might change its Hit@1 a little.
  - It couldn't raise its Hit@10, which is already the ceiling on both sets.

## Exploratory context

- **Against Phase 1's B5** (frozen corpus, same queries): R1 vs B5 is 5 / 7 on dev
  (p = 0.77) and 10 / 9 on test (p = 1). The live tool now finds the original about as
  often as Phase 1's best configuration did on the frozen corpus.
- **Against the web-search LLMs of Step 3**, on the same 200 checked queries:
  - **Hit@5**: L5r 171 (0.855), L5w 146 (0.730), Opus 122 (0.610). L5r vs Opus is
    61 / 12.
  - **Hit@1**: L5r 166, Opus 115. Opus's strength in Step 3 was its rank 1, and L5r
    now beats it there too.
  - Both comparisons use the same scoring as Step 3, and neither was pre-declared.
- **What's left is retrieval**: 24 of the 27 originals missing from the candidates
  never came back from any search the check ran.
  - The other 3 came back but ranked 22–47 in the proposal's semantic search, and
    below 30 in both fusions (A0119, A0220, A0230).
  - A deeper candidate list could catch those 3. Nothing in the ranking can catch the
    24.

## Caveat: the benchmark suits an LLM reranker

Each proposal is a paraphrase of its original's abstract, with Phase 1's leak filters
keeping the word overlap low. A reranker that reads the proposal and every abstract can
recognise the paraphrase by its meaning: the same problem, approach and claim.

- **A real researcher's idea is not a paraphrase of one paper.** The gain on real
  checks, in finding the closest prior work, may be smaller.
- **The cost is a 6 s call** that can be dropped if it doesn't help. Step 5's hand
  check of 10 real checks should look at whether L5r's top results are the most useful
  ones.

## Spend

| Ledger | Pilot | Dev | Test | Total this step | Ledger total / cap |
|---|---|---|---|---|---|
| `data/cache/llm.sqlite` (Qwen via OpenRouter) | $0.0085 | $0.3424 | $0.2363 | **$0.587** | $6.07 / $12 |
| `data/cache/rerank_sonnet.sqlite` (Anthropic credit) | $0.0919 | $3.8004 | $2.6350 | **$6.527** | $6.53 / $20 |

- **Anthropic credit**: by our count, $82.92 of the $100 credit is now spent (Step 3's
  $76.39 plus this step's). The credit expires on 2026-10-18 UTC, and the workspace
  spend limit is $95. Anthropic doesn't report a billed cost, so the console's figure
  may differ slightly.
- **OpenAlex**: none.

## Deviations from the plan

1. **Sonnet's thinking**: the declared settings were "its default adaptive thinking".
   Through the OpenAI-compatible endpoint, Sonnet answered without visible thinking
   (about 120 output tokens). I kept the declared settings rather than turning thinking
   on.
2. **The pilot's queries** were the first 3 of the Step 2 pilot run in qid order
   (A0000, A0002, A0004), as `--limit 3` takes them.
3. **Exploratory analyses** (the Phase 1 B5 comparison, the retrieval misses and the
   comparison with Opus on the 200) were not pre-declared, and are labelled as such.
4. **Nothing in the live tool changed.** L5r is evaluated offline only: the live
   pipeline and live-eval still have L5w as their best config.

## Open questions

1. **Building L5r into the live pipeline**: the rerank call after the final fusion,
   with the $0.10 cap covering it, and an L5r config in live-eval. Should this happen
   at the start of the explainer step, which needs the final top 10, or as its own
   small step?
2. **Should a reranker failure fall back to L5w?** In the product, an unreadable or
   failed rerank reply could fall back to L5w's order with a warning, as decided for
   failed agent searches. In the evaluation, unreadable replies already keep the
   presented order (there were none).
3. **What remains is retrieval**: 24 of 27 remaining misses never came back from
   OpenAlex. Is that worth a later step (more or different agent queries), or is
   0.85–0.86 Hit@5 enough for Phase 2?
4. **The remaining Anthropic credit**: about $17 by our count, expiring 2026-10-18.
   The deep-check evaluation needs Sonnet with web search on new queries, about $0.17
   a query, so about 100 queries would fit. Use it for that before it expires, or let
   it go?
