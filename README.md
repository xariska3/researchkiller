# ResearchKiller

ResearchKiller is a prior-art retrieval benchmark and, later, an adversarial
research-proposal reviewer. Phase 1 asks whether agentic query planning finds
prior art better than cheaper retrieval when the query describes a paper's
contribution in different terminology. It builds a frozen OpenAlex corpus,
rewrites recent target papers as terminology-shifted research proposals, and
compares BM25, dense, hybrid, one-shot LLM query rewriting and an agent loop at
the same query budget. The design is in [docs/phase1_spec.md](docs/phase1_spec.md).

## Status and setup

The pipeline (corpus, benchmark, indexes, evaluation) is implemented and tested
on fixtures; there are no results yet (pilot in Step 6, full run and write-up
in Step 7). To set up, create a Python 3.10+ virtualenv, run
`pip install -e ".[dev,dense]"`, then `ruff check .` and `pytest`. For GPU
encoding on Windows, first install a CUDA build of torch
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`), since
the PyPI wheel is CPU-only.

API keys go in `.env` at the repo root (copy `.env.example`; `.env` is
git-ignored): `OPENROUTER_API_KEY` for the paraphraser, judge and agent, and
optionally `OPENALEX_API_KEY`. `python -m rk` loads the file at startup;
variables already set in the environment take precedence.

## Running

```
python -m rk build-corpus --config configs/phase1.yaml [--max-works N]
python -m rk build-index  --config configs/phase1.yaml
python -m rk make-bench   --config configs/phase1.yaml [--limit N] [--show]
python -m rk run          --config configs/phase1.yaml [--configs B1,B4,B5] [--limit N]
python -m rk report       runs/<timestamp>
```

A run stores one row per configuration and query in `data/results.sqlite`
(tables `runs` and `query_results`), writes a portable copy to
`runs/<timestamp>/per_query/`, and summarises it in `summary.md` and
`summary.json`. To resume an interrupted run, pass `--run-dir runs/<timestamp>`;
add `--retry-errors` to redo the rows that ended in an LLM error.

## Experiment tracking

Every run is logged to MLflow, one parent run per benchmark run and one child
run per configuration, in the SQLite database `mlflow.db` at the repo root,
with artifacts in `mlartifacts/` beside it. Open the UI with

```
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

and browse to http://127.0.0.1:5000.
