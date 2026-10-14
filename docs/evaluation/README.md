# Hand checks

- `human_check_results.csv`: the Phase 1 benchmark (below).
- `live_misses.csv`: Phase 2 live retrieval's misses (at the end).

# Human check of the benchmark

`human_check_results.csv` is a hand review of the frozen bench design (Step 6
pilot, 2026-10-13; judge `mistralai/mistral-medium-3.1`). It was written by

```
python -m rk export-human-check --config configs/phase1.yaml --n 20 --validation-dir data/judge_validation_mistral_medium
```

and filled in by hand. Columns: `part, qid, title, abstract, proposal,
judge_reason, verdict, note`. `judge_reason` is the judge's own reason; `verdict`
and `note` are the reviewer's.

## The two parts

- **Part A** (20 rows): a seeded random sample of the pairs the judge
  accepted in the 40-target pilot bench. Question: does the proposal describe
  the same paper? Verdict `yes`, `borderline` or `no`.
- **Part B** (10 rows): every original proposal that the judge rejected in
  `validate-judge` (30 pairs, drawn from an earlier pilot bench, so their
  proposals differ from part A's), with the judge's reason per failed facet.
  Question: is the rejection right? Verdict `right` or `wrong`.

## Review rule

A proposal is written before the work is done, so it states aims, questions
and planned methods. It can't state findings. A proposal is never wrong for
leaving out the paper's results, such as outperforming a baseline or a
negative finding.

## Results

- **Part A: 19/20 correct, plus 1 borderline over-generalisation (A0030).**
  Its proposal (retrieval, validation and "specialised neural architectures")
  fits many chain-of-thought papers.
- **Part B: 9/10 of the judge's rejections were wrong.** Most penalised a
  proposal for stating aims rather than findings. The one right rejection is
  the same target, A0030: its earlier proposal was too general to identify
  the paper.

# Live misses (Phase 2 Step 2)

`live_misses.csv` holds 10 queries where the live pipeline (L5) missed the
original paper in its top 5. They are a seeded random sample (`run.seed`, 42)
of the 36 such queries in the 120-query dev run
`runs/live-20261014-122144`. It was written by

```
python -m rk live-misses runs/live-20261014-122144
```

which rebuilds those checks offline from the cached OpenAlex responses: no
new calls.

- **Columns**:
  - `qid`, `proposal`, `original_title`;
  - `original_rank`: the original's rank in L5's top 20, or `none`;
  - for each of L5's top 3 results, `topN_title`, `topN_year` and
    `topN_abstract` (the first 300 characters);
  - `top1_close_prior_art` and `note`: left empty for the reviewer.
- **Question**: is L5's top result genuinely close prior art for the
  proposal, so that the miss is partly the metric's (it credits only the
  original), not the tool's?
