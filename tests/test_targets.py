from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from rk.config import BenchConfig
from rk.targets import TargetSelectionError, select_targets, target_problems

CUTOFF = date(2025, 6, 1)


def _record(num: int, **overrides: Any) -> dict[str, Any]:
    record = {
        "id": f"W{num}",
        "aliases": [],
        "origin": "sweep",
        "publication_date": "2025-09-01",
        "abstract": " ".join(["word"] * 120),
        "referenced_works": [f"W{n}" for n in range(10)],
    }
    return {**record, **overrides}


def _bench(n_targets: int = 4, seed: int = 7) -> BenchConfig:
    return BenchConfig(
        n_targets=n_targets,
        target_from_date=CUTOFF,
        max_attempts_per_target=3,
        max_distinctive_overlap=0.15,
        max_content_jaccard=0.15,
        seed=seed,
        out_dir=Path("data/bench"),
    )


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"origin": "reference"}, "not from a sweep"),
        ({"publication_date": "2025-05-31"}, "published before 2025-06-01"),
        ({"abstract": " ".join(["word"] * 99)}, "abstract not 100-450 words"),
        ({"abstract": " ".join(["word"] * 451)}, "abstract not 100-450 words"),
        ({"referenced_works": ["W1"] * 9}, "fewer than 10 references"),
    ],
)
def test_each_target_rule(overrides: dict[str, Any], problem: str) -> None:
    assert target_problems(_record(1), CUTOFF) == []
    assert target_problems(_record(1, **overrides), CUTOFF) == [problem]


def test_boundaries_are_inclusive() -> None:
    edge = _record(
        1,
        publication_date="2025-06-01",
        abstract=" ".join(["word"] * 450),
        referenced_works=["W1"] * 10,
    )
    assert target_problems(edge, CUTOFF) == []


def test_samples_one_and_a_half_times_n_targets_with_a_seed() -> None:
    records = [_record(n) for n in range(100, 120)] + [_record(5, origin="citer")]

    first = select_targets(records, _bench(n_targets=4, seed=7))
    again = select_targets(list(reversed(records)), _bench(n_targets=4, seed=7))
    other = select_targets(records, _bench(n_targets=4, seed=8))

    assert first.n_candidates == 20
    assert len(first.targets) == 6  # ceil(1.5 * 4)
    ids = [t["id"] for t in first.targets]
    assert ids == sorted(ids, key=lambda i: int(i[1:]))
    assert [t["id"] for t in again.targets] == ids  # input order doesn't matter
    assert [t["id"] for t in other.targets] != ids


def test_takes_every_candidate_when_there_are_fewer_than_the_sample() -> None:
    selection = select_targets([_record(1), _record(2)], _bench(n_targets=4))
    assert [t["id"] for t in selection.targets] == ["W1", "W2"]


def test_fails_loudly_without_candidates() -> None:
    records = [_record(1, publication_date="2024-01-01"), _record(2, origin="citer")]
    with pytest.raises(TargetSelectionError) as exc_info:
        select_targets(records, _bench())
    message = str(exc_info.value)
    assert "no target candidates among 2 documents" in message
    assert "published before 2025-06-01: 1" in message
    assert "not from a sweep: 1" in message
