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
