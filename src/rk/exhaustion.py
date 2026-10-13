"""Exhaustion check (docs/analysis_plan.md): do the targets the bench exhausts differ?

The human check found the judge rejecting proposals for stating aims rather
than the paper's findings, and such rejections can use up a target's attempts.
This compares the targets make-bench exhausted with the ones it accepted, on
the measures the analysis plan fixed before the full run: abstract length, the
share of sentences with a number, the share with a results cue, and subfield
(and, descriptively, topic). It also lists why the exhausted targets' attempts
were rejected.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rk.bench import (
    ACCEPTED_FILE,
    FACETS,
    JUDGE_DIRECTIONS,
    NOT_RESEARCH,
    REASONS,
    REJECTIONS_FILE,
    sentences,
)
from rk.corpus import CORPUS_FILE, CorpusError, load_frozen_corpus

RESAMPLES = 10_000
# A digit run not joined to a letter or a hyphen: "12.5%", "0.83" and "2025" count;
# "GPT-4", "F1" and "3D" don't.
NUMBER_RE = re.compile(r"(?<![\w-])\d+(?:[.,]\d+)*%?(?![\w-])")
RESULTS_CUES = (
    "result", "results", "show", "shows", "showed", "shown", "demonstrate", "demonstrates",
    "demonstrated", "outperform", "outperforms", "outperformed", "outperforming", "achieve",
    "achieves", "achieved", "achieving", "surpass", "surpasses", "surpassed", "reveal",
    "reveals", "revealed", "find", "finds", "found", "finding", "findings", "indicate",
    "indicates", "indicated", "improves", "improved", "improvement", "improvements",
    "significantly", "state-of-the-art",
)
RESULTS_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(cue) for cue in RESULTS_CUES) + r")\b", re.IGNORECASE
)
MEASURES = ("abstract_words", "number_sentence_share", "results_sentence_share")
SUBFIELD_PAIR = ("Information Systems", "Artificial Intelligence")  # rate difference: a minus b
JUDGE_REASONS = ("judge_parse_error", *(f"judge_{d}" for d in JUDGE_DIRECTIONS))


def abstract_measures(abstract: str) -> dict[str, float]:
    """Whitespace words, and the shares of sentences with a number and with a results cue."""
    parts = sentences(abstract) or [abstract]
    return {
        "abstract_words": float(len(abstract.split())),
        "number_sentence_share": sum(bool(NUMBER_RE.search(s)) for s in parts) / len(parts),
        "results_sentence_share": sum(bool(RESULTS_RE.search(s)) for s in parts) / len(parts),
    }


def two_group(
    a: Sequence[float], b: Sequence[float], *, seed: int, resamples: int = RESAMPLES
) -> dict[str, Any]:
    """Means of two independent groups and a minus b, with a 95% CI and a p-value.

    The CI is a seeded percentile bootstrap that resamples within each group;
    the two-sided permutation p-value is (1 + permutations whose |difference|
    is at least the observed one) / (1 + resamples). Without data in both
    groups, the difference, CI and p are None.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out: dict[str, Any] = {
        "n_a": int(x.size),
        "n_b": int(y.size),
        "mean_a": float(x.mean()) if x.size else None,
        "mean_b": float(y.mean()) if y.size else None,
        "diff": None,
        "ci": None,
        "p": None,
    }
    if not x.size or not y.size:
        return out
    rng = np.random.default_rng(seed)
    diff = float(x.mean() - y.mean())
    boot = (
        x[rng.integers(0, x.size, size=(resamples, x.size))].mean(axis=1)
        - y[rng.integers(0, y.size, size=(resamples, y.size))].mean(axis=1)
    )
    low, high = np.percentile(boot, [2.5, 97.5])
    shuffled = rng.permuted(np.tile(np.concatenate([x, y]), (resamples, 1)), axis=1)
    perm = shuffled[:, : x.size].mean(axis=1) - shuffled[:, x.size :].mean(axis=1)
    extreme = int(np.sum(np.abs(perm) >= abs(diff) - 1e-9))  # tolerance for rounding
    out.update(diff=diff, ci=[float(low), float(high)], p=(1 + extreme) / (1 + resamples))
    return out


def target_groups(
    accepted_rows: Sequence[dict[str, Any]],
    rejected_rows: Sequence[dict[str, Any]],
    max_attempts: int,
) -> dict[str, list[str]]:
    """Target ids that were accepted, exhausted, are unfinished, or were skipped.

    Exhausted means every one of `max_attempts` attempts was rejected (as in
    make-bench's stats); unfinished means fewer were, and none accepted.
    """
    accepted = {r["target_id"] for r in accepted_rows}
    skipped = {r["target_id"] for r in rejected_rows if r["reason"] == NOT_RESEARCH}
    attempts = Counter(r["target_id"] for r in rejected_rows if r["reason"] != NOT_RESEARCH)
    open_ = {t: n for t, n in attempts.items() if t not in accepted}
    return {
        "accepted": sorted(accepted),
        "exhausted": sorted(t for t, n in open_.items() if n >= max_attempts),
        "unfinished": sorted(t for t, n in open_.items() if n < max_attempts),
        "skipped": sorted(skipped),
    }


def _label(record: dict[str, Any], key: str) -> str:
    """The record's primary-topic subfield or topic name, "unknown" if it has none."""
    topic = record.get("primary_topic") or {}
    return topic.get(key) or "unknown"


def _rates(labels: dict[str, str], exhausted: set[str]) -> dict[str, dict[str, Any]]:
    """Accepted, exhausted and exhaustion rate per label, the largest groups first."""
    groups: dict[str, list[str]] = {}
    for target, label in labels.items():
        groups.setdefault(label, []).append(target)
    rates = {}
    for label, ids in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        n_exhausted = sum(t in exhausted for t in ids)
        rates[label] = {
            "accepted": len(ids) - n_exhausted,
            "exhausted": n_exhausted,
            "rate": n_exhausted / len(ids),
        }
    return rates


def _rejections(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Why the exhausted targets' attempts were rejected: per reason, and the judge's facets."""
    by_reason = Counter(r["reason"] for r in rows)
    facets = {d: Counter[str]() for d in JUDGE_DIRECTIONS}
    for r in rows:
        direction = r["reason"].removeprefix("judge_")
        if direction in facets:
            facets[direction].update(r["judge_verdicts"][direction].get("failed", []))
    judged = [r for r in rows if r["reason"] in JUDGE_REASONS]
    return {
        "attempts": len(rows),
        "judge_attempts": len(judged),
        "targets_with_a_judge_rejection": len({r["target_id"] for r in judged}),
        "by_reason": {reason: by_reason[reason] for reason in REASONS if by_reason[reason]},
        "judge_failed_facets": {d: {f: facets[d][f] for f in FACETS} for d in JUDGE_DIRECTIONS},
    }


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def exhaustion_check(
    bench_dir: Path,
    corpus_dir: Path,
    *,
    max_attempts: int,
    bench_sha256: str,
    corpus_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """The exhaustion check of the bench a run used, or {"unavailable": why}.

    It needs that run's bench (test_a.jsonl with the run's sha256) and frozen
    corpus. Differences are exhausted minus accepted; for subfield, the
    exhaustion rate of SUBFIELD_PAIR[0] minus SUBFIELD_PAIR[1].
    """
    accepted_path = bench_dir / ACCEPTED_FILE
    if not accepted_path.exists():
        return {"unavailable": f"no benchmark in {bench_dir}"}
    data = accepted_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != bench_sha256:
        return {"unavailable": f"{accepted_path} has changed since this run"}
    try:
        corpus, _, manifest = load_frozen_corpus(corpus_dir)
    except CorpusError as exc:
        return {"unavailable": str(exc)}
    if manifest["files"][CORPUS_FILE]["sha256"] != corpus_sha256:
        return {"unavailable": f"the frozen corpus in {corpus_dir} isn't the one this run used"}
    rejections_path = bench_dir / REJECTIONS_FILE
    accepted_rows = _jsonl(data)
    rejected_rows = _jsonl(rejections_path.read_bytes()) if rejections_path.exists() else []

    groups = target_groups(accepted_rows, rejected_rows, max_attempts)
    accepted, exhausted = groups["accepted"], groups["exhausted"]
    by_id = {r["id"]: r for r in corpus}
    measured = {t: abstract_measures(by_id[t]["abstract"]) for t in [*accepted, *exhausted]}
    exhausted_set = set(exhausted)
    subfields = {t: _label(by_id[t], "subfield") for t in measured}
    topics = {t: _label(by_id[t], "display_name") for t in measured}

    def exhausted_in(subfield: str) -> list[int]:
        return [int(t in exhausted_set) for t, s in subfields.items() if s == subfield]

    return {
        "max_attempts": max_attempts,
        "n": {group: len(ids) for group, ids in groups.items()},
        "measures": {  # a = exhausted, b = accepted
            m: two_group(
                [measured[t][m] for t in exhausted], [measured[t][m] for t in accepted], seed=seed
            )
            for m in MEASURES
        },
        "subfield": {
            "rates": _rates(subfields, exhausted_set),
            "difference": {
                "a": SUBFIELD_PAIR[0],
                "b": SUBFIELD_PAIR[1],
                **two_group(
                    exhausted_in(SUBFIELD_PAIR[0]), exhausted_in(SUBFIELD_PAIR[1]), seed=seed
                ),
            },
        },
        "topic": {"rates": _rates(topics, exhausted_set)},
        "exhausted_rejections": _rejections(
            [r for r in rejected_rows if r["target_id"] in exhausted_set]
        ),
    }
