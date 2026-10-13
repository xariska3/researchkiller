from __future__ import annotations

from typing import Any

import pytest

from rk.bench import make_qid, qid_position
from rk.exhaustion import NUMBER_RE, RESULTS_RE, abstract_measures, target_groups, two_group


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("12.5%", True),
        ("0.83", True),
        ("2025", True),
        ("by 5.", True),
        ("1,000 users", True),
        ("GPT-4", False),
        ("F1", False),
        ("3D", False),
        ("Llama3", False),
        ("five percent", False),
    ],
)
def test_numbers_are_digit_runs_not_joined_to_letters_or_hyphens(text: str, number: bool) -> None:
    assert bool(NUMBER_RE.search(text)) is number


@pytest.mark.parametrize(
    ("text", "cue"),
    [
        ("Results show a large gap.", True),
        ("Experiments demonstrated it.", True),
        ("It achieves state-of-the-art accuracy.", True),
        ("We find that it helps.", True),
        ("It significantly reduces memory.", True),
        ("We aim to improve recall.", False),  # "improve" (an aim) isn't a cue; "improves" is
        ("A showcase of the founders' work.", False),  # whole words only
        ("We propose a method.", False),
    ],
)
def test_results_cues_are_whole_words_from_the_plan(text: str, cue: bool) -> None:
    assert bool(RESULTS_RE.search(text)) is cue


def test_abstract_measures_count_words_and_sentence_shares() -> None:
    abstract = (
        "We study long inputs. Our model reaches 91.2% accuracy. "
        "Results show it beats GPT-4 on 3 tasks. Code is released."
    )
    assert abstract_measures(abstract) == {
        "abstract_words": 20.0,
        "number_sentence_share": 2 / 4,  # "91.2%", "3"; "GPT-4" doesn't count
        "results_sentence_share": 1 / 4,  # "Results", "show"
    }


def test_two_group_difference_ci_and_permutation_p() -> None:
    separated = two_group([1.0] * 10, [0.0] * 10, seed=3)
    assert (separated["n_a"], separated["n_b"], separated["diff"]) == (10, 10, 1.0)
    assert separated["ci"] == [1.0, 1.0]
    assert separated["p"] < 0.001  # only 2 of C(20, 10) labelings are as extreme
    same = two_group([0.0, 1.0] * 5, [1.0, 0.0] * 5, seed=3)
    assert (same["diff"], same["p"]) == (0.0, 1.0)
    assert same["ci"][0] < 0.0 < same["ci"][1]
    values = ([0.3, 0.9, 0.4, 0.8, 0.5], [0.2, 0.1, 0.6, 0.3])
    assert two_group(*values, seed=3) == two_group(*values, seed=3)  # seeded
    assert two_group(*values, seed=3)["ci"] != two_group(*values, seed=4)["ci"]
    empty = two_group([], [1.0], seed=3)
    assert (empty["mean_a"], empty["mean_b"], empty["diff"], empty["ci"], empty["p"]) == (
        None,
        1.0,
        None,
        None,
        None,
    )


def _rejected(target: str, reason: str, attempt: int = 0) -> dict[str, Any]:
    return {"target_id": target, "attempt": attempt, "reason": reason}


def test_targets_are_grouped_as_make_bench_counts_them() -> None:
    accepted = [{"target_id": "W1"}, {"target_id": "W2"}]
    rejected = [
        _rejected("W2", "title_stem"),  # rejected once, then accepted
        *(_rejected("W3", "judge_captures", a) for a in range(4)),  # exhausted
        *(_rejected("W4", "idf_overlap", a) for a in range(2)),  # unfinished
        _rejected("W5", "not_research_contribution"),  # skipped
    ]
    assert target_groups(accepted, rejected, max_attempts=4) == {
        "accepted": ["W1", "W2"],
        "exhausted": ["W3"],
        "unfinished": ["W4"],
        "skipped": ["W5"],
    }


def test_qid_position_reads_back_make_qid() -> None:
    assert qid_position(make_qid(44, "W7212357336")) == 44
    assert qid_position("A0045-W1") == 45
    assert qid_position(make_qid(12, "W9", width=5)) == 12
