"""What a check says about its top papers (docs/phase2_spec.md, Pipeline 5; Step 5).

- **Snippets** (`snippets`, free): for each paper, the abstract sentence closest to the
  idea, by the check's encoder (the idea as a query, the sentences as documents).
- **Explanations** (`explain`): one call to the explainer model (Phase 1's validated
  judge model) with the idea and every paper (number, title, abstract cut at
  explain.abstract_chars). The answer is strict JSON (EXPLAIN_RESPONSE_FORMAT), one
  entry per paper: an overlap level, a same/different verdict and reason per facet, one
  sentence copied from the abstract as evidence, and one line on what differs. Each
  entry is read on its own, so one bad entry costs only its paper.
- **Quote check** (`verify_quote`): the evidence must be in the abstract verbatim, up
  to whitespace and the form of quotation marks, and at least MIN_QUOTE_WORDS words
  long. If it isn't, the quote is dropped and the paper is marked unverified; its
  snippet stays.

The call is charged to the check's ledger under its cost cap. If it could cross the
cap, fails, or its answer can't be read, the check keeps its papers and snippets and
gets a warning; with `strict` (evaluation), a failed call raises instead.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from pydantic import ValidationError

from rk.config import ModelConfig
from rk.live.pipeline import CappedChatModel, CheckCostCapExceeded, CheckLedger
from rk.live.report import FACETS, OVERLAPS, Explanation, Facets, Snippet
from rk.live.reranker import as_number, render_candidates
from rk.llm import ChatModel, Completion, LLMError, Usage, load_prompt, parse_json
from rk.retrieval import Encoder
from rk.targets import Record
from rk.text import sentences

EXPLAIN_PROMPT = "explain_overlap"  # rk/prompts/explain_overlap.txt
MIN_QUOTE_WORDS = 5  # shorter evidence shows nothing, so it isn't verified
# What the explainer step did (ExplainStep.status).
EXPLAINED = "explained"  # at least one paper has an explanation
FALLBACK = "fallback"  # none has: snippets only, with a warning
SKIPPED = "skipped"  # no papers, so no call


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    """A JSON-schema object in strict mode: every property required, no others."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
# The explainer's answer, enforced by the API (json_schema, strict) as the judge's is;
# the facets come before the overlap level, so the model compares before it grades.
EXPLAIN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "overlap_explanations",
        "strict": True,
        "schema": _strict_object(
            {
                "papers": {
                    "type": "array",
                    "items": _strict_object(
                        {
                            "paper": {"type": "integer"},
                            "facets": _strict_object(
                                {
                                    f: _strict_object(
                                        {
                                            "verdict": {
                                                "type": "string",
                                                "enum": ["same", "different"],
                                            },
                                            "reason": _STRING,
                                        }
                                    )
                                    for f in FACETS
                                }
                            ),
                            "overlap": {"type": "string", "enum": list(OVERLAPS)},
                            "evidence": _STRING,
                            "differs": _STRING,
                        }
                    ),
                }
            }
        ),
    },
}
EXPLAIN_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(EXPLAIN_RESPONSE_FORMAT, sort_keys=True).encode("utf-8")
).hexdigest()


# --- snippets ------------------------------------------------------------------------------


def snippets(text: str, records: Sequence[Record], encoder: Encoder) -> list[Snippet | None]:
    """Each record's abstract sentence closest to `text` (None if it has no sentence).

    Ties go to the earlier sentence. All sentences are encoded in one batch.
    """
    per_record = [sentences(r["abstract"]) for r in records]
    flat = [s for found in per_record for s in found]
    if not flat:
        return [None] * len(records)
    query = encoder.encode_queries([text])[0]
    similarity = encoder.encode_docs(flat) @ query
    out: list[Snippet | None] = []
    start = 0
    for found in per_record:
        if not found:
            out.append(None)
            continue
        scores = similarity[start : start + len(found)]
        best = int(np.argmax(scores))  # the first maximum
        out.append(Snippet(sentence=found[best], similarity=round(float(scores[best]), 4)))
        start += len(found)
    return out


# --- the quote check -----------------------------------------------------------------------

_QUOTE_MARKS = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"',
                              "„": '"', "‟": '"'})


def normalise_quote(text: str) -> str:
    """Whitespace collapsed to single spaces, and curly quotation marks made straight."""
    return " ".join(text.translate(_QUOTE_MARKS).split())


def unwrapped(quote: str) -> str:
    """The quote normalised, without quotation marks wrapped around all of it."""
    q = normalise_quote(quote)
    if len(q) >= 2 and q[0] == q[-1] and q[0] in "\"'":
        q = q[1:-1].strip()
    return q


def verify_quote(quote: str, abstract: str) -> bool:
    """Whether `quote` is in `abstract` verbatim, up to whitespace and quotation marks.

    Quotation marks wrapped around the whole quote are ignored, and a quote of fewer
    than MIN_QUOTE_WORDS words is not verified.
    """
    q = unwrapped(quote)
    return len(q.split()) >= MIN_QUOTE_WORDS and q in normalise_quote(abstract)


# --- the explainer call --------------------------------------------------------------------


def render_prompt(text: str, records: Sequence[Record], abstract_chars: int) -> str:
    return load_prompt(EXPLAIN_PROMPT).render(
        n=str(len(records)),
        proposal=text,
        papers=render_candidates(records, abstract_chars),
    )


def parse_explanations(reply: str, records: Sequence[Record]) -> list[Explanation | None] | None:
    """One explanation per record, in order (None where the reply has no usable entry).

    None if the reply has no readable list of papers. An entry is usable if its paper
    number is new and in range and its fields are valid; its quote is checked against
    that paper's abstract.
    """
    try:
        data = parse_json(reply)
    except ValueError:
        return None
    if isinstance(data, dict):
        data = data.get("papers")
    if not isinstance(data, list):
        return None
    found: list[Explanation | None] = [None] * len(records)
    for item in data:
        if not isinstance(item, dict):
            continue
        number = as_number(item.get("paper"))
        if number is None or not 1 <= number <= len(records) or found[number - 1] is not None:
            continue
        quote = item.get("evidence")
        if not isinstance(quote, str):
            continue
        try:
            facets = Facets.model_validate(item.get("facets"))
            verified = verify_quote(quote, records[number - 1]["abstract"])
            found[number - 1] = Explanation(
                overlap=item.get("overlap"),
                facets=facets,
                evidence=unwrapped(quote) if verified else None,
                verified=verified,
                quote=quote,
                differs=item.get("differs"),
            )
        except ValidationError:
            continue
    return found


@dataclass(frozen=True)
class ExplainStep:
    """What the explainer step did for a check."""

    status: str  # EXPLAINED, FALLBACK (snippets only) or SKIPPED (no papers)
    explanations: list[Explanation | None]  # one per paper, in order
    ledger: CheckLedger  # the check's, with the call charged
    latency_s: float
    usage: Usage  # the call's, if it was answered
    cached: bool  # the call was served from the LLM cache
    finish_reason: str | None = None
    warning: str | None = None


def explain(
    text: str,
    records: Sequence[Record],
    explainer: ChatModel,
    model: ModelConfig,
    ledger: CheckLedger,
    *,
    abstract_chars: int,
    strict: bool = False,
) -> ExplainStep:
    """One explainer call for every record, charged to a copy of the check's ledger.

    If the call could cross the check's cap, fails (LLMError), or its reply can't be
    read, no paper is explained and the step carries a warning; papers whose entry is
    missing or invalid get a warning too. With `strict`, a failed call raises.
    """
    start = time.perf_counter()
    ledger = replace(ledger)  # the caller's stays as it was
    if not records:
        return ExplainStep(SKIPPED, [], ledger, time.perf_counter() - start, Usage(), False)
    capped = CappedChatModel(explainer, model, ledger, what="the explainer call")
    messages = [{"role": "user", "content": render_prompt(text, records, abstract_chars)}]
    completion: Completion | None = None
    explanations: list[Explanation | None] | None = None
    problem: str | None = None
    try:
        completion = capped.complete(messages, response_format=EXPLAIN_RESPONSE_FORMAT)
    except CheckCostCapExceeded as exc:  # "the explainer call could cost $x, and the check ..."
        problem = str(exc)
    except LLMError as exc:
        if strict:
            raise
        problem = f"the explainer call failed: {exc}"
    if completion is not None:
        explanations = parse_explanations(completion.text, records)
        if explanations is None or not any(explanations):
            cut = " (it was cut off)" if completion.finish_reason == "length" else ""
            problem = f"the explainer's reply could not be read{cut}"
    warning: str | None = None
    if problem is not None or explanations is None or not any(explanations):
        explanations = [None] * len(records)
        status = FALLBACK
        warning = f"Not explained ({problem}); the papers show their snippets only."
    else:
        status = EXPLAINED
        missing = [n for n, e in enumerate(explanations, start=1) if e is None]
        if missing:
            listed = ", ".join(str(n) for n in missing)
            warning = (
                f"No explanation for the paper(s) at rank {listed}; they show their snippets "
                f"only."
            )
    return ExplainStep(
        status=status,
        explanations=explanations,
        ledger=ledger,
        latency_s=time.perf_counter() - start,
        usage=completion.usage if completion else Usage(),
        cached=completion.cached if completion else False,
        finish_reason=completion.finish_reason if completion else None,
        warning=warning,
    )
