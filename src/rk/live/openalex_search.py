"""Live OpenAlex search: keyword and semantic /works search, with every response cached on disk.

OpenAlex API facts this module relies on, checked 2026-10-13 against the docs
and with six live probes (one Phase 1 proposal and a short query):

- Keyword search (https://help.openalex.org/api/searching/): `search=` on
  /works matches title, abstract, full text and keywords, with stemming and
  stop-word removal; words not joined by AND/OR/NOT are ANDed. Results carry a
  `relevance_score` (text similarity plus a capped citation boost) and are
  sorted by it. Scoped variants (search.title, search.title_and_abstract,
  search.title_abstract_keywords) exist; we use plain `search`. URLs are
  limited to about 4 KB. `rerank=true` reorders the top 100 with OpenAlex's own
  model for $1 per 1,000 extra; we rerank locally instead. Probe: a 57-word
  proposal as a keyword query matched 6 off-topic works (all words ANDed over
  full text), so queries longer than live.keyword_max_words get semantic search
  only. Agent queries in the Phase 1 full run had 6-21 words, proposals 45-77.
- Semantic search (https://help.openalex.org/api/semantic-search/,
  https://help.openalex.org/guides/semantic-search): `search.semantic=` on
  /works embeds the query with GTE Large EN (1,024 dimensions) and ranks works
  by cosine similarity to their embedded title and abstract. The first 2,000
  characters are used (longer input is truncated, so we truncate before
  sending), at most 50 results come back, and it is limited to 1 request per
  second. Most filters work with it (not cited_by_count or the country-code
  filters); it can't be combined with another search parameter or rerank.
  Probes: per_page=100 returns HTTP 400 "per_page cannot exceed 50 for
  semantic search"; page=2 with per_page=25 returns results 26-50; the
  relevance_score is the cosine (0.79-0.93 in the probe); has_abstract:true
  works with it. The Phase 1 proposal found its original paper at rank 1.
- Filters (https://help.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists):
  has_abstract:true keeps works with an abstract; to_publication_date:<date>
  is inclusive, so "published before D" is to_publication_date:D-1.
- Work types (https://help.openalex.org/data/work-types/, updated 2026-08-17):
  25 values. We keep live.work_types (article, preprint, conference-paper,
  book-chapter, dissertation, review, book, report, software-paper) with a type: filter, and
  drop any other type that comes back anyway (dataset, erratum, peer-review,
  paratext, letter, editorial, other, ...). Proceedings papers are
  `conference-paper`, newly added with low coverage (`proceedings-article`
  matched 0 works); `review` (a literature review) is not `peer-review` (a
  referee report). Probe: the type: filter works with keyword and semantic
  search; semantic search filters after retrieval, so it can return fewer
  than 50 (43 in the probe).
- Authors: `authorships[].author` has an id and a display_name; the id can be
  null (a dissertation's only author in the A0002 smoke case had none).
- Paging (https://help.openalex.org/api/paging/): per_page 1-100 (default 25);
  page x per_page at most 10,000; cursor paging beyond that. One page per
  query is enough here (50 results).
- Cost (https://help.openalex.org/access/example-costs/,
  https://help.openalex.org/access/pricing/): per 1,000 calls, singleton
  lookups are free, list+filter $0.10, search $1, semantic search $1, rerank
  +$1, content download $10. A free key gives $1 of usage per day, $0.10
  without a key; the budget resets at midnight UTC; prepaid usage can be added.
  Probes: every keyword and semantic call reported `meta.cost_usd` 0.001 and
  the headers x-ratelimit-cost-usd 0.001 and x-ratelimit-credits-used 10
  (1 credit = $0.0001); a rejected call (HTTP 400) cost 0.
- Rate limits and auth (https://help.openalex.org/api-reference/authentication,
  https://help.openalex.org/how-to-use-the-api/rate-limits-and-authentication):
  10 requests/s without a key, 30 with a free key, 50 on an annual plan;
  semantic search 1 request/s; searches with more than 5 boolean operators
  1-25/s. Both a spent daily budget and a per-second overrun return HTTP 429;
  a per-second 429 costs nothing, so we back off and retry (rk.corpus
  OpenAlexClient). The key goes in an api_key parameter or an
  "Authorization: Bearer" header (we use the header). Headers
  X-RateLimit-Limit/-Remaining/-Credits-Used/-Reset report the budget, and
  /rate-limit returns it for a key. A 429 whose X-RateLimit-Remaining can't
  pay for one search raises rk.corpus.OpenAlexBudgetExhausted (not retried);
  `budget()` reads /rate-limit, which is free.

The cache keeps every response it ever fetched. The live tool reads the newest
one if it is younger than live.cache_ttl_days and fetches again otherwise;
evaluation runs read the first one ever cached, so they never expire and a
rerun sees the same responses.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from rk.config import SEMANTIC_MAX_RESULTS, Config, LiveConfig
from rk.corpus import OpenAlexClient, OpenAlexError, short_id, to_record
from rk.targets import Record

KEYWORD, SEMANTIC = "keyword", "semantic"
SEARCH_COST_USD = 0.001  # documented price of one keyword or semantic search call
SEMANTIC_MAX_CHARS = 2000  # OpenAlex uses only the first 2,000 characters
KEYWORD_REQUESTS_PER_SECOND = 10.0  # the keyless limit, so it holds with or without a key
SEMANTIC_REQUESTS_PER_SECOND = 1.0
SELECT = (
    "id,doi,title,publication_date,abstract_inverted_index,relevance_score,language,type,"
    "authorships"
)
ORIGIN = "live"  # the record origin rk.corpus.dedupe sees

# Every response ever fetched, one row each (the Step 1 smoke test's `responses`
# table had one row per request and is no longer read).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS openalex_responses (
    key TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    params TEXT NOT NULL,
    response TEXT NOT NULL,
    cost_usd REAL NOT NULL,
    PRIMARY KEY (key, fetched_at)
)
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CachedResponse:
    response: dict[str, Any]
    cost_usd: float  # what the call cost when it was fetched
    fetched_at: datetime


class OpenAlexCache:
    """SQLite store of every OpenAlex /works response, keyed by the request parameters."""

    def __init__(self, path: str | Path, *, now: Callable[[], datetime] = _utc_now) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self.now = now

    @staticmethod
    def key(params: dict[str, str]) -> str:
        blob = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(
        self, params: dict[str, str], *, max_age: timedelta | None = None
    ) -> CachedResponse | None:
        """A cached response, or None.

        With max_age None, the first response ever cached (it never expires);
        otherwise the newest one, if it is at most max_age old.
        """
        order = "ASC" if max_age is None else "DESC"
        row = self._conn.execute(
            "SELECT response, cost_usd, fetched_at FROM openalex_responses WHERE key = ?"
            f" ORDER BY fetched_at {order} LIMIT 1",
            (self.key(params),),
        ).fetchone()
        if row is None:
            return None
        cached = CachedResponse(json.loads(row[0]), float(row[1]), datetime.fromisoformat(row[2]))
        if max_age is not None and self.now() - cached.fetched_at > max_age:
            return None
        return cached

    def put(
        self, params: dict[str, str], response: dict[str, Any], cost_usd: float
    ) -> datetime:
        """Add a response (older ones are kept); returns when it was fetched."""
        fetched_at = self.now()
        self._conn.execute(
            "INSERT INTO openalex_responses (key, fetched_at, params, response, cost_usd)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                self.key(params),
                fetched_at.isoformat(),
                json.dumps(params, sort_keys=True, ensure_ascii=False),
                json.dumps(response, ensure_ascii=False),
                cost_usd,
            ),
        )
        self._conn.commit()
        return fetched_at

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM openalex_responses").fetchone()[0])

    def close(self) -> None:
        self._conn.close()


@dataclass(frozen=True)
class SearchResponse:
    """One keyword or semantic search: the usable records, in OpenAlex's order."""

    query: str
    kind: str  # KEYWORD or SEMANTIC
    records: list[Record]
    n_results: int  # works returned, before dropping unusable ones (live_record)
    cost_usd: float  # what the call costs (method cost), also when served from the cache
    cached: bool
    fetched_at: datetime  # when OpenAlex sent this response
    params: dict[str, str] = field(repr=False)

    @property
    def money_spent(self) -> float:
        return 0.0 if self.cached else self.cost_usd


@dataclass(frozen=True)
class Budget:
    """The API key's OpenAlex budget, from /rate-limit (USD)."""

    remaining_usd: float  # what is left today, plus any prepaid balance
    daily_usd: float | None  # the daily budget
    resets_in_s: float | None  # seconds until midnight UTC


def _number(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def authors(work: dict[str, Any]) -> list[dict[str, str | None]]:
    """The work's authors in order: OpenAlex author id (may be None) and display name."""
    found = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        author_id = author.get("id")
        found.append(
            {"id": short_id(author_id) if author_id else None, "name": author.get("display_name")}
        )
    return found


def live_record(work: dict[str, Any], types: Collection[str] | None = None) -> Record | None:
    """A search result as a corpus-style record with its type and authors.

    None without an id, a date or an abstract, or if its type isn't in `types`.
    """
    if not work.get("id") or not work.get("publication_date"):
        return None
    if types is not None and work.get("type") not in types:
        return None
    record = to_record(work, ORIGIN)
    if not record["abstract"].strip():
        return None
    return {**record, "type": work.get("type"), "authors": authors(work)}


def search_filter(before: date | None, types: Collection[str] = ()) -> str:
    """has_abstract:true, the kept types, and to_publication_date:<day before> for "before"."""
    clauses = ["has_abstract:true"]
    if types:
        clauses.append(f"type:{'|'.join(types)}")
    if before is not None:
        clauses.append(f"to_publication_date:{(before - timedelta(days=1)).isoformat()}")
    return ",".join(clauses)


class LiveSearchClient:
    """Keyword and semantic /works search, each throttled to its own documented rate.

    Every successful response is cached by its request parameters, so a repeated
    search costs nothing and replays offline. `max_age` None (evaluation) reads
    the first response ever cached; otherwise a cached response older than
    max_age is fetched again. With `offline`, a search that isn't cached raises
    OpenAlexError instead of being sent. Retries and backoff come from
    rk.corpus.OpenAlexClient.
    """

    def __init__(
        self,
        live: LiveConfig,
        cache: OpenAlexCache,
        *,
        max_age: timedelta | None = None,
        offline: bool = False,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.live = live
        self.cache = cache
        self.max_age = max_age
        self.offline = offline
        self.calls = 0  # searches answered, cached or not

        def client(rate: float) -> OpenAlexClient:
            return OpenAlexClient(
                api_key=api_key,
                transport=transport,
                max_requests_per_second=rate,
                timeout=live.openalex_timeout_s,
                call_cost_usd=SEARCH_COST_USD,
                sleep=sleep,
                clock=clock,
            )

        self._clients = {
            KEYWORD: client(KEYWORD_REQUESTS_PER_SECOND),
            SEMANTIC: client(SEMANTIC_REQUESTS_PER_SECOND),
        }

    @classmethod
    def from_config(
        cls, cfg: Config, cache: OpenAlexCache, *, evaluation: bool = False, **kwargs: Any
    ) -> LiveSearchClient:
        """Client with the OpenAlex API key from the env var corpus.api_key_env names.

        The live tool's cached responses expire after live.cache_ttl_days; an
        evaluation run's never do.
        """
        live = cfg.live
        if live is None:
            raise ValueError("the config has no live: section")
        env = cfg.corpus.api_key_env
        api_key = os.environ.get(env) if env else None
        ttl = None if evaluation or live.cache_ttl_days is None else live.cache_ttl_days
        max_age = None if ttl is None else timedelta(days=ttl)
        return cls(live, cache, max_age=max_age, api_key=api_key or None, **kwargs)

    @property
    def authenticated(self) -> bool:
        return self._clients[KEYWORD].authenticated

    def params(self, query: str, kind: str, before: date | None = None) -> dict[str, str]:
        """The /works parameters of one search, as sent and as cached."""
        if kind == KEYWORD:
            search = {"search": query}
            per_page = self.live.results_per_query
        elif kind == SEMANTIC:
            search = {"search.semantic": query[:SEMANTIC_MAX_CHARS]}
            per_page = min(self.live.semantic_results_per_query, SEMANTIC_MAX_RESULTS)
        else:
            raise ValueError(f"unknown search kind {kind!r}")
        return {
            **search,
            "filter": search_filter(before, self.live.work_types),
            "per_page": str(per_page),
            "select": SELECT,
        }

    def search(self, query: str, kind: str, *, before: date | None = None) -> SearchResponse:
        """One page of results for `query`, from the cache when possible."""
        params = self.params(query, kind, before)
        hit = self.cache.get(params, max_age=self.max_age)
        if hit is None and self.offline:
            raise OpenAlexError(f"offline, and this {kind} search isn't cached: {query[:100]!r}")
        if hit is None:
            data = self._clients[kind].get_works(params)
            cost = _reported_cost(data)
            fetched_at = self.cache.put(params, data, cost)
        else:
            data, cost, fetched_at = hit.response, hit.cost_usd, hit.fetched_at
        self.calls += 1
        works = data.get("results") or []
        records = [r for w in works if (r := live_record(w, self.live.work_types)) is not None]
        return SearchResponse(
            query, kind, records, len(works), cost, hit is not None, fetched_at, params
        )

    def budget(self) -> Budget:
        """Today's OpenAlex budget for the API key (a free /rate-limit call)."""
        data = self._clients[KEYWORD].rate_limit()
        remaining = _number(data, "daily_remaining_usd")
        if remaining is None:
            raise OpenAlexError(f"/rate-limit has no daily_remaining_usd: {str(data)[:300]}")
        return Budget(
            remaining_usd=remaining + (_number(data, "prepaid_remaining_usd") or 0.0),
            daily_usd=_number(data, "daily_budget_usd"),
            resets_in_s=_number(data, "resets_in_seconds"),
        )

    def close(self) -> None:
        for client in self._clients.values():
            client.close()


def _reported_cost(data: dict[str, Any]) -> float:
    """meta.cost_usd as OpenAlex reports it, or the documented price if it's missing."""
    value = (data.get("meta") or {}).get("cost_usd")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return SEARCH_COST_USD
