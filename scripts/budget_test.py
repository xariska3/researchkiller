"""Size the test run (C): today's OpenAlex budget from the free /rate-limit, N queries that fit.

$0.0067 per query (measured in runs/live-20261014-122144: one L1 call plus L5's agent
searches; L5w adds none), $0.05 kept back, at most 80. Exits 1 if the budget hasn't reset.

One-off script of Phase 2 Step 2 (docs/reports/phase2-step2-test.md, Budget used), kept
as it was run on 2026-10-14 (the two `with` statements since joined into one, for ruff).
The /rate-limit lookup is free.

Run from the repository root: python scripts/budget_test.py
"""

import math
import sys
from contextlib import closing
from pathlib import Path

from rk.cli import load_env_file
from rk.config import load_config
from rk.live.openalex_search import LiveSearchClient, OpenAlexCache

PER_QUERY, MARGIN, MAX_N = 0.0067, 0.05, 80

load_env_file()
cfg = load_config(Path("configs/phase1.yaml"))
assert cfg.live is not None
with (
    closing(OpenAlexCache(cfg.live.cache_path)) as cache,
    closing(LiveSearchClient.from_config(cfg, cache, evaluation=True)) as search,
):
    budget = search.budget()
n = min(MAX_N, math.floor((budget.remaining_usd - MARGIN) / PER_QUERY))
print(f"remaining ${budget.remaining_usd:.4f} of ${budget.daily_usd}; resets in "
      f"{budget.resets_in_s / 3600:.2f} h; N = min({MAX_N}, floor(({budget.remaining_usd:.4f} - "
      f"{MARGIN}) / {PER_QUERY})) = {n}")
sys.exit(0 if n >= 1 else 1)
