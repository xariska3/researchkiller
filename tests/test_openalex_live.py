"""Live OpenAlex smoke test. Deselected by default; run with `pytest -m network`.

Uses OPENALEX_API_KEY (via configs/phase1.yaml) if it is set; one list request
costs $0.0001 of the daily budget.
"""

from __future__ import annotations

import itertools
from contextlib import closing
from datetime import date
from pathlib import Path

import pytest

from rk.config import load_config
from rk.corpus import NEWEST_FIRST, OpenAlexClient, sweep_filter, to_record

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase1.yaml"


@pytest.mark.network
def test_live_openalex_returns_five_recent_works_with_abstracts() -> None:
    corpus_cfg = load_config(REPO_CONFIG).corpus
    today = date.today()
    filters = sweep_filter(corpus_cfg.filters[0], today)

    with closing(OpenAlexClient.from_config(corpus_cfg)) as client:
        works = list(itertools.islice(client.iter_works(filters, sort=NEWEST_FIRST, per_page=5), 5))
        calls = client.calls

    records = [to_record(w, "sweep") for w in works]
    assert len(records) == 5
    assert calls == 1
    assert all(r["id"].startswith("W") and r["title"] for r in records)
    assert all(r["publication_date"] <= today.isoformat() for r in records)
    assert all(len(r["abstract"].split()) > 0 for r in records)
    dates = [r["publication_date"] for r in records]
    assert dates == sorted(dates, reverse=True)  # newest first
