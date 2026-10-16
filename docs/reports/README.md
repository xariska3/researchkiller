# Reports

These are the working reports written by the coding agent (Claude Code) at each step,
addressed to the author and kept as written, as a dated record of what was decided and
found. 'You' is the author; 'I' is the agent. Current numbers are in the
[main README](../../README.md).

In order (dates of the first commit of each):

1. [../step6_proposal.md](../step6_proposal.md) (2026-10-12): Phase 1 Step 6a, the
   proposed real configuration (topics, models, encoder, costs), kept in `docs/`.
2. [full-bench.md](full-bench.md) (2026-10-13): Phase 1, building the full benchmark of
   terminology-shifted proposals.
3. [full-run.md](full-run.md) (2026-10-13): Phase 1, every retrieval configuration on the
   500-query benchmark, on a frozen corpus.
4. [phase2-step1.md](phase2-step1.md) (2026-10-13): Phase 2 Step 1, live retrieval over
   OpenAlex.
5. [phase2-step2-pilot.md](phase2-step2-pilot.md) (2026-10-14): Step 2 pilot, the live
   check on 30 queries.
6. [phase2-step2-full.md](phase2-step2-full.md) (2026-10-14): Step 2, the live check on
   the 120-query development set.
7. [phase2-step2-test.md](phase2-step2-test.md) (2026-10-14): Step 2, the test run of the
   chosen check (L5w) on 80 held-out queries.
8. [phase2-step3-plan.md](phase2-step3-plan.md) (2026-10-14): Step 3 plan, the
   comparison with other systems, before any paid call.
9. [phase2-step3-results.md](phase2-step3-results.md) (2026-10-14): Step 3 results,
   against web-search LLMs, Perplexity and Semantic Scholar.
10. [phase2-step4-rerank.md](phase2-step4-rerank.md) (2026-10-15): Step 4, the LLM
    reranker (L5r), and every system on the 80 test queries.
11. [phase2-step5-explainer.md](phase2-step5-explainer.md) (2026-10-15): Step 5, what a
    check shows (snippets, overlap explanations, vocabulary map), the hand review and
    the more detailed limitations.
12. [phase2-two-lists.md](phase2-two-lists.md) (2026-10-16): issue #22, two lists from
    one rerank call (closest in contribution, closest in methods), evaluated on the
    dev set.
