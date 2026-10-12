"""Target selection (spec: Corpus, step 3).

A target is a sweep document published on or after bench.target_from_date,
with a 100-450 word abstract and at least 10 references (counted from its
OpenAlex metadata, not from what is in the corpus). A seeded sample of
1.5 x n_targets candidates is drawn, since some targets will later fail the
benchmark filters.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from rk.config import BenchConfig

TARGET_MIN_ABSTRACT_WORDS = 100
TARGET_MAX_ABSTRACT_WORDS = 450
TARGET_MIN_REFERENCES = 10
OVERSAMPLE = 1.5

Record = dict[str, Any]


class TargetSelectionError(RuntimeError):
    """No document qualifies as a target."""


def id_number(work_id: str) -> int:
    """Numeric part of an OpenAlex work id ("W123" -> 123), for a stable id order."""
    return int(work_id.lstrip("W"))


def target_problems(record: Record, target_from_date: date) -> list[str]:
    """Why `record` can't be a target; empty if it can."""
    problems = []
    if record["origin"] != "sweep":
        problems.append("not from a sweep")
    if date.fromisoformat(record["publication_date"]) < target_from_date:
        problems.append(f"published before {target_from_date}")
    n_words = len(record["abstract"].split())
    if not TARGET_MIN_ABSTRACT_WORDS <= n_words <= TARGET_MAX_ABSTRACT_WORDS:
        problems.append(
            f"abstract not {TARGET_MIN_ABSTRACT_WORDS}-{TARGET_MAX_ABSTRACT_WORDS} words"
        )
    if len(record["referenced_works"]) < TARGET_MIN_REFERENCES:
        problems.append(f"fewer than {TARGET_MIN_REFERENCES} references")
    return problems


@dataclass(frozen=True)
class Selection:
    targets: list[Record]  # sampled candidates, in id order
    n_candidates: int


def select_targets(records: Iterable[Record], bench: BenchConfig) -> Selection:
    """Seeded sample of ceil(1.5 x n_targets) candidates; fails loudly if there are none."""
    records = list(records)
    candidates = sorted(
        (r for r in records if not target_problems(r, bench.target_from_date)),
        key=lambda r: id_number(r["id"]),
    )
    if not candidates:
        raise TargetSelectionError(_explain(records, bench.target_from_date))
    k = min(len(candidates), math.ceil(OVERSAMPLE * bench.n_targets))
    sample = random.Random(bench.seed).sample(candidates, k)
    return Selection(sorted(sample, key=lambda r: id_number(r["id"])), len(candidates))


def _explain(records: Sequence[Record], target_from_date: date) -> str:
    failing: Counter[str] = Counter()
    for record in records:
        failing.update(target_problems(record, target_from_date))
    reasons = ", ".join(f"{reason}: {n}" for reason, n in sorted(failing.items())) or "none"
    return (
        f"no target candidates among {len(records)} documents "
        f"(documents failing each rule: {reasons})"
    )
