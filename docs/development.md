# Development and the benchmark

How to run the tests, the Phase 1 benchmark and the Phase 2 evaluations. The tool
itself is described in the [README](../README.md).

## Setup

Create a Python 3.10+ virtualenv, run `pip install -e ".[dev,dense]"`, then
`ruff check .` and `pytest`. The tests never touch the network. For GPU encoding on
Windows, first install a CUDA build of torch
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`), since the
PyPI wheel is CPU-only.

API keys go in `.env` at the repo root (copy `.env.example`; `.env` is git-ignored).
`python -m rk` loads the file at startup; variables already set in the environment
take precedence.
- `OPENROUTER_API_KEY`: the paraphraser, judge, agent, reranker and explainer, and the
  Perplexity system of `compete`.
- `OPENALEX_API_KEY`: optional, but a free key gives 10x the keyless daily budget.
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY`: only for
  `python -m rk compete`, the head-to-head against other systems.

## Phase 1: the benchmark on a frozen corpus

```
python -m rk build-corpus --config configs/phase1.yaml [--max-works N]
python -m rk build-index  --config configs/phase1.yaml
python -m rk make-bench   --config configs/phase1.yaml [--limit N] [--show]
python -m rk validate-judge --config configs/phase1.yaml [--n 30]
python -m rk export-human-check --config configs/phase1.yaml [--n 20]
python -m rk run          --config configs/phase1.yaml [--configs B1,B4,B5] [--limit N]
python -m rk report       runs/<timestamp>
```

A run stores one row per configuration and query in `data/results.sqlite` (tables
`runs` and `query_results`), writes a portable copy to `runs/<timestamp>/per_query/`,
and summarises it in `summary.md` and `summary.json`. To resume an interrupted run, or
add configurations to a run, pass `--run-dir runs/<timestamp>`; add `--retry-errors`
to redo the rows that ended in an LLM error.

The design is in [phase1_spec.md](phase1_spec.md), and the results are in
[reports/full-run.md](reports/full-run.md) and
[reports/full-bench.md](reports/full-bench.md).

## Phase 2: the live tool and its evaluations

- `python -m rk check --config configs/phase1.yaml --file idea.txt [--quick]`: one
  check in the terminal.
- `python -m rk serve [--host H] [--port P]`: the web app.
- `python -m rk live-eval`, `rerank`, `explain-eval` and `compete`: the evaluations
  in [reports/](reports/). Each report gives its exact command.

The design is in [phase2_spec.md](phase2_spec.md).

## Experiment tracking

Every benchmark run is logged to MLflow, one parent run per benchmark run and one
child run per configuration, in the SQLite database `mlflow.db` at the repo root, with
artifacts in `mlartifacts/` beside it. Open the UI with

```
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

and browse to http://127.0.0.1:5000.
