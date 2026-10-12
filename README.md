# ResearchKiller

ResearchKiller is a prior-art retrieval benchmark and, later, an adversarial
research-proposal reviewer. Phase 1 asks whether agentic query planning finds
prior art better than cheaper retrieval when the query describes a paper's
contribution in different terminology. It builds a frozen OpenAlex corpus,
rewrites recent target papers as terminology-shifted research proposals, and
compares BM25, dense, hybrid, one-shot LLM query rewriting and an agent loop at
the same query budget. The design is in [docs/phase1_spec.md](docs/phase1_spec.md).

This README is a placeholder. The project is at Step 1 (scaffold): the CLI
commands (`python -m rk build-corpus --config configs/phase1.yaml` and so on)
are stubs and there are no results yet. To set up, create a Python 3.10+
virtualenv, run `pip install -e ".[dev]"` (add the `dense` extra for
sentence-transformers and `tracking` for MLflow), then run `ruff check .` and
`pytest`. For GPU encoding on Windows, first install a CUDA build of torch
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`), since
the PyPI wheel is CPU-only. Results, limitations and reproduction instructions will be added in
Step 7.
