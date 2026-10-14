"""A fake OpenAlex /works search for the Phase 2 tests (httpx.MockTransport).

The works are the bench_data corpus records as OpenAlex returns them.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import httpx
from bench_data import corpus_records
from conftest import FakeEncoder

from rk.text import content_stems, reconstruct_abstract

SEARCH_CREDITS = 10  # what one keyword or semantic search costs (1 credit = $0.0001)
RESETS_IN_S = 3600


def oa_work(record: dict[str, Any], published: str = "2025-03-01", **extra: Any) -> dict[str, Any]:
    """A bench_data corpus record as OpenAlex returns it (abstract as an inverted index)."""
    inverted: dict[str, list[int]] = {}
    for position, word in enumerate(record["abstract"].split()):
        inverted.setdefault(word, []).append(position)
    return {
        "id": f"https://openalex.org/{record['id']}",
        "doi": None,
        "title": record["title"],
        "publication_date": published,
        "abstract_inverted_index": inverted,
        "relevance_score": 1.0,
        "language": "en",
        "type": "article",
        "authorships": [],
        **extra,
    }


WORKS = [oa_work(r) for r in corpus_records()]


class FakeLiveOpenAlex:
    """In-memory /works search and /rate-limit for httpx.MockTransport.

    Keyword search matches works whose title and abstract contain every content
    stem of the query (OpenAlex ANDs the words), most stem occurrences first.
    Semantic search ranks works by FakeEncoder cosine, leaving out the ids in
    `hide_from_semantic`. Both apply has_abstract, type (unless `honour_types`
    is off) and to_publication_date, honour per_page and select, report
    meta.cost_usd, and reject what OpenAlex rejects (semantic per_page over 50)
    with HTTP 400. `fail_with` status codes are returned first, one per request.

    With `budget_credits`, each search costs SEARCH_CREDITS of a daily budget,
    and every response carries the X-RateLimit headers (unless `budget_headers`
    is off); a search the budget can't pay for gets HTTP 429, as OpenAlex
    answers once the day's budget is spent. /rate-limit reports the budget.
    """

    def __init__(
        self,
        works: Sequence[dict[str, Any]] = WORKS,
        *,
        cost: float | None = 0.001,
        fail_with: Sequence[int] = (),
        honour_types: bool = True,
        hide_from_semantic: Collection[str] = (),
        budget_credits: int | None = None,
        budget_headers: bool = True,
    ) -> None:
        self.works = list(works)
        self.cost = cost
        self.fail_with = list(fail_with)
        self.honour_types = honour_types
        self.hide_from_semantic = {f"https://openalex.org/{w}" for w in hide_from_semantic}
        self.budget_credits = budget_credits
        self.budget_headers = budget_headers
        self.requests: list[dict[str, str]] = []  # /works only
        self.rate_limit_calls = 0
        self.encoder = FakeEncoder()

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _headers(self) -> dict[str, str]:
        if self.budget_credits is None or not self.budget_headers:
            return {}
        return {
            "x-ratelimit-limit": "10000",
            "x-ratelimit-remaining": str(self.budget_credits),
            "x-ratelimit-reset": str(RESETS_IN_S),
        }

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rate-limit":
            self.rate_limit_calls += 1
            credits = 10000 if self.budget_credits is None else self.budget_credits
            budget = {
                "daily_budget_usd": 1,
                "daily_remaining_usd": credits / 10000,
                "prepaid_remaining_usd": 0,
                "resets_in_seconds": RESETS_IN_S,
            }
            return httpx.Response(200, json={"rate_limit": budget})
        params = dict(request.url.params)
        self.requests.append(params)
        if self.fail_with:
            return httpx.Response(self.fail_with.pop(0), json={"error": "simulated"})
        if self.budget_credits is not None and self.budget_credits < SEARCH_CREDITS:
            return httpx.Response(429, json={"error": "rate limited"}, headers=self._headers())
        per_page = int(params.get("per_page", "25"))
        works = [w for w in self.works if self._passes(w, params.get("filter", ""))]
        if "search.semantic" in params:
            if per_page > 50:
                message = f"per_page cannot exceed 50 for semantic search. Received {per_page}."
                return httpx.Response(400, json={"error": "Invalid", "message": message})
            shown = [w for w in works if w["id"] not in self.hide_from_semantic]
            ranked = self._semantic(params["search.semantic"], shown)
        elif "search" in params:
            ranked = self._keyword(params["search"], works)
        else:
            return httpx.Response(400, json={"error": "no search parameter"})
        if self.budget_credits is not None:
            self.budget_credits -= SEARCH_CREDITS
        fields = params["select"].split(",")
        results = [{k: w.get(k) for k in fields} for w in ranked[:per_page]]
        meta: dict[str, Any] = {"count": len(ranked), "per_page": per_page}
        if self.cost is not None:
            meta["cost_usd"] = self.cost
        return httpx.Response(200, json={"meta": meta, "results": results}, headers=self._headers())

    @staticmethod
    def _text(work: dict[str, Any]) -> str:
        return f"{work['title']}. {reconstruct_abstract(work['abstract_inverted_index'])}"

    def _passes(self, work: dict[str, Any], filters: str) -> bool:
        for clause in filter(None, filters.split(",")):
            key, _, value = clause.partition(":")
            if key == "has_abstract" and bool(work["abstract_inverted_index"]) != (value == "true"):
                return False
            if key == "type" and self.honour_types and work["type"] not in value.split("|"):
                return False
            if key == "to_publication_date" and work["publication_date"] > value:
                return False
        return True

    def _keyword(self, query: str, works: list[dict[str, Any]]) -> list[dict[str, Any]]:
        wanted = set(content_stems(query))
        scored = []
        for work in works:
            stems = content_stems(self._text(work))
            if wanted and wanted <= set(stems):
                scored.append((-sum(s in wanted for s in stems), work["id"], work))
        return [w for *_, w in sorted(scored, key=lambda s: s[:2])]

    def _semantic(self, query: str, works: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not works:
            return []
        q = self.encoder.encode_queries([query])[0]
        docs = self.encoder.encode_docs([self._text(w) for w in works])
        sims = docs @ q
        order = sorted(range(len(works)), key=lambda i: (-sims[i], works[i]["id"]))
        return [works[i] for i in order if sims[i] > 0]
