# ResearchKiller: working rules for Claude Code

ResearchKiller is a prior-art retrieval benchmark and, later, an adversarial
research-proposal reviewer. The full Phase 1 design is in
`docs/phase1_spec.md`. Read it before any work.

## How we work
- An instructor reviews your output between steps. Do **only the step you are
  asked for**. Don't start the next step.
- If the spec is ambiguous or a design choice looks wrong, stop and say so in
  your summary rather than guessing.
- Any deviation from the spec must be listed explicitly in your summary.

## End-of-step summary (always, in this format)
1. Files created / changed (one line each)
2. Commands run and results (tests: passed / failed counts)
3. Deviations from the spec, and why
4. Open questions for the instructor

## Code rules
- Python 3.10+, `src/` layout, package name `rk`. Type hints everywhere.
- Keep it minimal: no new dependencies beyond the spec without asking.
- No frameworks for the agent in Phase 1 (plain Python loop). LangGraph comes in Phase 2.
- Tests must never touch the network: mock OpenAlex with `httpx.MockTransport`,
  use the fake LLM and fake encoder from `tests/conftest.py`.
- Every run must be reproducible: fixed seeds, cached LLM calls, frozen corpus.

## Git rules
- Make one local commit per completed step, with a conventional message
  (`feat: ...`, `test: ...`, `docs: ...`).
- Never push unless explicitly asked.
- Never commit `data/`, `runs/`, `.env`, caches, model weights or virtualenvs.