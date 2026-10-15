"""L5r's rerank: one listwise LLM call over a live check's candidates (Phase 2 Step 4).

This is R1 of docs/phase2_spec.md (Evaluation, rerank variants), chosen on the dev
set and confirmed on the test set (docs/reports/phase2-step4-rerank.md). The
candidates are the union of L5w's and L5's top `rerank.depth`, taken in turn (L5w's
first, then L5's first, and so on), without repeats: the presented order, in which
they are numbered. The model sees the proposal and each candidate's number, title
and abstract (cut at `rerank.abstract_chars`), nothing else, and returns every
number, closest first. Unknown or repeated numbers are dropped; candidates it leaves
out follow in presented order.

rk.live.pipeline.with_rerank makes the call in a live check; rk.live.rerank replays
it, with the other variants, on cached checks.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from rk.llm import Prompt, parse_json
from rk.targets import Record
from rk.text import opening

LISTWISE = "rerank_listwise"  # the prompt (rk/prompts/rerank_listwise.txt)


def candidates(l5w: Sequence[str], l5: Sequence[str], depth: int) -> list[str]:
    """The union of both top `depth`s, taken in turn (L5w's first), without repeats."""
    taken: dict[str, None] = {}
    for pair in itertools.zip_longest(l5w[:depth], l5[:depth]):
        for doc_id in pair:
            if doc_id is not None:
                taken.setdefault(doc_id, None)
    return list(taken)


def render_candidates(records: Sequence[Record], abstract_chars: int) -> str:
    """Each candidate as its number, title and abstract (cut at a word), in the given order."""
    return "\n\n".join(
        f"[{n}] Title: {r['title']}\nAbstract: {opening(r['abstract'], abstract_chars)}"
        for n, r in enumerate(records, start=1)
    )


def render_prompt(
    prompt: Prompt, proposal: str, records: Sequence[Record], abstract_chars: int
) -> str:
    return prompt.render(
        n=str(len(records)),
        proposal=proposal,
        candidates=render_candidates(records, abstract_chars),
    )


def as_number(value: Any) -> int | None:
    """An integer from a JSON number or a string of digits; None otherwise."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def parse_order(text: str, n: int) -> list[int] | None:
    """A listwise reply's candidate numbers (1..n), in order; None if nothing usable."""
    try:
        data = parse_json(text)
    except ValueError:
        return None
    if isinstance(data, dict):
        data = data.get("ranking")
    if not isinstance(data, list):
        return None
    order: dict[int, None] = {}
    for item in data:
        number = as_number(item)
        if number is not None and 1 <= number <= n:
            order.setdefault(number, None)
    return list(order) or None


def listwise_ranking(cands: Sequence[str], order: Sequence[int] | None) -> list[str]:
    """The candidates in the reply's order, then the ones it left out, in presented order."""
    chosen = [cands[n - 1] for n in order or ()]
    return list(dict.fromkeys([*chosen, *cands]))
