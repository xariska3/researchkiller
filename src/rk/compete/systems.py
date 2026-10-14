"""The systems compared with the live tool, behind one interface, cached and cost-capped.

Every LLM system gets the same prompt (rk/prompts/compete.txt: the spec's request,
up to 10 papers with title, year and DOI or link, as JSON, closest first); Semantic
Scholar's searches get the proposal itself as the query. Every response is stored in
the compete cache (SQLite, compete.cache_path) under the sha256 of the request, with
the latency of the call that fetched it, so a repeated run replays it for free.
Before a real call, the system's reserve_per_query is charged to its spend in the
same file, and the call is refused (SystemCapReached) if that could take the system's
total over its cost_cap; afterwards the call's cost replaces the reservation. As in
rk.llm.Ledger, a call that fails after a request went out without a 4xx coming back
is booked at its reservation, since it may have been billed.

API facts, checked on 2026-10-14 against the docs:

- Anthropic web search
  (https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool and
  .../about-claude/pricing): tool types web_search_20250305 (basic),
  web_search_20260209 (dynamic filtering: Claude filters results with code before
  they reach its context; Claude 4.6 and later) and web_search_20260318 (adds
  response_inclusion). `max_uses` caps the searches. $10 per 1,000 searches plus
  tokens: search results count as input tokens, and code execution is free with
  web_search_20260209 or later. usage reports input and output tokens and
  server_tool_use.web_search_requests; tool errors come inside a 200 response. A long
  turn can stop with stop_reason "pause_turn"; sending the assistant content back
  continues it. Per million input/output tokens: Opus 5.5 $4/$20, Sonnet 5.5 $2/$10,
  Haiku 5.5 $0.10/$0.50.
- OpenRouter (https://openrouter.ai/docs/guides/features/server-tools/web-search.md,
  .../guides/usage-accounting): the `openrouter:web_search` server tool (the `web`
  plugin and `:online` are deprecated) with engine "native" runs the provider's own
  search (OpenAI: GPT-5 and later); max_uses caps the searches and
  usage.server_tool_use.web_search_requests counts them. OpenAI charges $10 per 1,000
  web search calls plus the search content tokens at model rates
  (https://developers.openai.com/api/docs/pricing). usage.cost, on every response, is
  "the total amount charged to your account".
- Perplexity Sonar (https://docs.perplexity.ai/getting-started/pricing): $1/$1 per
  million tokens plus $5/$8/$12 per 1,000 requests at search context low/medium/high
  (low is the default); OpenRouter lists perplexity/sonar at $1/$1 and $0.005 per
  request. It searches on every request; no tool is needed.
- Semantic Scholar (https://api.semanticscholar.org/graph/v1/swagger.json): GET
  /paper/search ranks papers for a plain-text query (at most 100 per page); GET
  /snippet/search returns ~500-word passages from titles, abstracts and body text,
  best first, each with its paper's corpusId and title. Free; a key goes in
  `x-api-key`. Probes on 2026-10-14 without a key: most requests got HTTP 429 (a
  shared pool), and a 57-word proposal as a paper-search query matched 0 papers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx

from rk.config import CompeteSystemConfig
from rk.llm import (
    METADATA_HEADER,
    billed_cost,
    load_prompt,
    parse_json,
    retry_after_seconds,
    served_provider,
)

PROMPT = "compete"  # rk/prompts/compete.txt
MAX_ITEMS = 10  # the prompt asks for up to 10 papers
ANTHROPIC_VERSION = "2023-06-01"
MAX_CONTINUATIONS = 3  # pause_turn continuations of one Anthropic answer
S2_FIELDS = "title,year,externalIds,url"
S2_SNIPPETS = 50  # snippets asked for, so that 10 distinct papers usually come back
NOT_TOOL_BLOCKS = ("text", "thinking", "redacted_thinking")  # Anthropic content blocks
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_DOI_TRAILING = ".,;:)]}>\"'"


class CompeteError(RuntimeError):
    """A system call failed: rejected, malformed, or retries exhausted."""


class SystemCapReached(RuntimeError):
    """A real call could take a system's spend over its cost cap."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_prompt(proposal: str) -> str:
    return load_prompt(PROMPT).render(proposal=proposal)


def find_doi(text: str | None) -> str | None:
    """The first DOI in `text` (a DOI, a doi.org link or any URL holding one), lower-cased."""
    if not text:
        return None
    match = _DOI_RE.search(text)
    if match is None:
        return None
    return match.group(0).rstrip(_DOI_TRAILING).lower() or None


# --- what a system lists ---------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One paper as a system listed it; its rank is its position in the list, from 1."""

    title: str | None
    year: int | None
    doi: str | None  # lower-cased, without a doi.org prefix
    link: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _text(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"\s*\d{4}\s*", value):
        return int(value)
    return None


def to_item(paper: Any) -> Item | None:
    """A listed paper from the answer's JSON, or None if it names nothing to look up."""
    if not isinstance(paper, dict):
        return None
    link = _text(paper.get("link")) or _text(paper.get("url"))
    if link is not None and not link.lower().startswith(("http://", "https://")):
        link = None
    item = Item(
        title=_text(paper.get("title")),
        year=_year(paper.get("year")),
        doi=find_doi(_text(paper.get("doi"))),
        link=link,
    )
    return item if item.title or item.doi or item.link else None


def parse_items(text: str) -> tuple[list[Item], str | None]:
    """The papers an answer lists (at most MAX_ITEMS), and why parsing failed, if it did.

    The answer should be {"papers": [...]}; a bare list is accepted too.
    """
    try:
        data = parse_json(text)
    except ValueError as exc:
        return [], str(exc)
    papers = data.get("papers") if isinstance(data, dict) else data
    if not isinstance(papers, list):
        return [], "the answer's JSON has no list of papers"
    items = [item for paper in papers if (item := to_item(paper)) is not None]
    return items[:MAX_ITEMS], None


@dataclass(frozen=True)
class Reading:
    """What a system's raw responses say: its list and its usage."""

    items: list[Item]
    text: str  # the answer text that was parsed ("" for the search APIs)
    parse_error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    billed_cost: float | None = None  # what the API reported charging, if it does
    provider: str | None = None  # who served it, as OpenRouter reports


@dataclass(frozen=True)
class Answer:
    reading: Reading
    cost: float  # method cost at the config prices, counted on cache hits too
    money_spent: float  # real calls only: billed if the API reported it, else `cost`
    latency_s: float  # the call that fetched it (its successful attempts), also on cache hits
    cached: bool


def method_cost(cfg: CompeteSystemConfig, reading: Reading) -> float:
    """USD cost of an answer at the config prices: tokens, searches and the request fee."""
    tokens = reading.input_tokens * cfg.price_in_per_m + reading.output_tokens * cfg.price_out_per_m
    return tokens / 1_000_000 + reading.searches * cfg.price_per_search + cfg.price_per_request


# --- storage: responses and spend --------------------------------------------------------

_RESPONSES_SCHEMA = """
CREATE TABLE IF NOT EXISTS compete_responses (
    key TEXT PRIMARY KEY,
    system TEXT NOT NULL,
    request TEXT NOT NULL,
    responses TEXT NOT NULL,
    latency_s REAL NOT NULL,
    created_at TEXT NOT NULL
)
"""

_SPEND_SCHEMA = """
CREATE TABLE IF NOT EXISTS compete_spend (
    timestamp TEXT NOT NULL,
    system TEXT NOT NULL,
    qid TEXT,
    cost REAL NOT NULL,
    billed_cost REAL
)
"""


def connect_store(path: str | Path) -> sqlite3.Connection:
    """The compete cache file (created if missing); tables are made by their users."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path))


def request_key(request: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a request (method, URL, body or parameters)."""
    blob = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """Every system's raw responses, keyed by request_key, with the latency of the call."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.execute(_RESPONSES_SCHEMA)
        conn.commit()

    def get(self, key: str) -> tuple[list[dict[str, Any]], float] | None:
        row = self._conn.execute(
            "SELECT responses, latency_s FROM compete_responses WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else (json.loads(row[0]), float(row[1]))

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def put(
        self,
        key: str,
        system: str,
        request: Mapping[str, Any],
        responses: Sequence[Mapping[str, Any]],
        latency_s: float,
    ) -> None:
        """Store a paid answer at once, so it survives a crash."""
        self._conn.execute(
            "INSERT OR REPLACE INTO compete_responses"
            " (key, system, request, responses, latency_s, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                system,
                json.dumps(request, sort_keys=True, ensure_ascii=False),
                json.dumps(list(responses), ensure_ascii=False),
                latency_s,
                _utc_now(),
            ),
        )
        self._conn.commit()


class SpendLedger:
    """Cumulative spend per system, with the system's cost cap (table compete_spend).

    A row's spend is its billed_cost when the API reported one, else its cost. The
    totals are read from the table on every check, so they add up across runs.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._reserved: dict[str, float] = {}
        conn.execute(_SPEND_SCHEMA)
        conn.commit()

    def recorded(self, system: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(COALESCE(billed_cost, cost)), 0.0) FROM compete_spend"
            " WHERE system = ?",
            (system,),
        ).fetchone()
        return float(row[0])

    def spent(self, system: str) -> float:
        """Recorded spend plus reservations for calls in flight."""
        return self.recorded(system) + self._reserved.get(system, 0.0)

    def charge(self, system: str, amount: float, cap: float) -> None:
        """Reserve `amount` for a real call, or raise SystemCapReached if it might not fit."""
        spent = self.spent(system)
        if spent + amount > cap:
            raise SystemCapReached(
                f"{system}: a call could cost up to ${amount:.4f}, but ${spent:.4f} of its "
                f"${cap:.2f} cost cap is spent"
            )
        self._reserved[system] = self._reserved.get(system, 0.0) + amount

    def record(
        self,
        system: str,
        *,
        reserved: float,
        qid: str | None,
        cost: float,
        billed_cost: float | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO compete_spend (timestamp, system, qid, cost, billed_cost)"
            " VALUES (?, ?, ?, ?, ?)",
            (_utc_now(), system, qid, cost, billed_cost),
        )
        self._conn.commit()
        self.release(system, reserved)

    def release(self, system: str, reserved: float) -> None:
        self._reserved[system] = self._reserved.get(system, 0.0) - reserved


# --- HTTP --------------------------------------------------------------------------------


class Billing:
    """Whether a real call may have been billed: a request went out and no 4xx came back."""

    def __init__(self) -> None:
        self.possible = False


class Http:
    """Throttled JSON requests: 429, 5xx and transport errors back off and retry.

    The latency of a call is the time of its successful attempt, without the
    waits between attempts.
    """

    def __init__(
        self,
        *,
        requests_per_second: float,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 6,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.calls = 0  # requests sent, retries included
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._interval = 1.0 / requests_per_second
        self._next_slot = 0.0
        self._sleep = sleep
        self._clock = clock
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def call(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
        params: Mapping[str, str] | None = None,
        billing: Billing | None = None,
        missing_ok: bool = False,
    ) -> tuple[Any, float]:
        """(the response's JSON, seconds); with missing_ok, a 404 gives (None, seconds)."""
        billing = billing or Billing()
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.calls += 1
            retry_after: float | None = None
            billed_before = billing.possible
            billing.possible = True
            start = self._clock()
            try:
                response = self._client.request(
                    method, url, headers=headers, json=json_body, params=params
                )
            except httpx.TransportError as exc:  # timeouts are transport errors in httpx
                problem = f"{type(exc).__name__}: {exc}"
            else:
                seconds = self._clock() - start
                status = response.status_code
                if 400 <= status < 500:
                    billing.possible = billed_before  # rejected, so not billed
                if status == 404 and missing_ok:
                    return None, seconds
                if status == 429 or status >= 500:
                    problem = f"HTTP {status}: {response.text[:300]}"
                    retry_after = retry_after_seconds(response)
                elif response.is_error:
                    raise CompeteError(f"{method} {url}: HTTP {status}: {response.text[:500]}")
                else:
                    try:
                        return response.json(), seconds
                    except ValueError as exc:
                        raise CompeteError(f"{method} {url}: the response is not JSON") from exc
            if attempt == self.max_retries:
                raise CompeteError(
                    f"{method} {url}: giving up after {attempt + 1} attempts; last error: {problem}"
                )
            delay = self.backoff_base * 2**attempt
            if retry_after is not None:
                delay = max(delay, retry_after)
            self._sleep(min(delay, self.backoff_max))
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        now = self._clock()
        if now < self._next_slot:
            self._sleep(self._next_slot - now)
            now = self._next_slot
        self._next_slot = now + self._interval


# --- the systems -------------------------------------------------------------------------


class System(Protocol):
    config: CompeteSystemConfig

    def request(self, proposal: str) -> dict[str, Any]:
        """The request for one proposal, as sent and as cached (no secrets in it)."""

    def missing_key(self) -> str | None:
        """The env var a real call needs but that isn't set, if any."""

    def send(
        self, request: Mapping[str, Any], billing: Billing
    ) -> tuple[list[dict[str, Any]], float]:
        """The raw responses for a request, and the seconds they took."""

    def read(self, responses: Sequence[Mapping[str, Any]]) -> Reading:
        """The list and the usage in raw responses (fresh or cached)."""


class _Client:
    def __init__(
        self,
        config: CompeteSystemConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.http = Http(
            requests_per_second=config.requests_per_second,
            timeout=config.timeout_s,
            transport=transport,
            sleep=sleep,
            clock=clock,
        )

    @property
    def url(self) -> str:
        return self.config.base_url.rstrip("/")

    def api_key(self) -> str | None:
        env = self.config.api_key_env
        return (os.environ.get(env) or None) if env else None

    def missing_key(self) -> str | None:
        env = self.config.api_key_env
        return env if env and self.api_key() is None else None

    def close(self) -> None:
        self.http.close()


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


class AnthropicWebSearch(_Client):
    """A Claude model with Anthropic's web search tool (Messages API)."""

    def request(self, proposal: str) -> dict[str, Any]:
        cfg = self.config
        tool: dict[str, Any] = {"type": cfg.web_search_tool, "name": "web_search"}
        if cfg.max_searches is not None:
            tool["max_uses"] = cfg.max_searches
        return {
            "method": "POST",
            "url": f"{self.url}/messages",
            "json": {
                "model": cfg.model,
                "max_tokens": cfg.max_tokens,
                "messages": [{"role": "user", "content": render_prompt(proposal)}],
                "tools": [tool],
            },
        }

    def send(
        self, request: Mapping[str, Any], billing: Billing
    ) -> tuple[list[dict[str, Any]], float]:
        api_key = self.api_key()
        if api_key is None:
            raise CompeteError(f"environment variable {self.config.api_key_env} is not set")
        headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
        body = request["json"]
        messages = list(body["messages"])
        responses: list[dict[str, Any]] = []
        seconds = 0.0
        for _ in range(MAX_CONTINUATIONS + 1):
            data, took = self.http.call(
                "POST",
                request["url"],
                headers=headers,
                json_body={**body, "messages": messages},
                billing=billing,
            )
            if not isinstance(data, dict):
                raise CompeteError(f"{self.config.name}: unexpected response {str(data)[:300]}")
            responses.append(data)
            seconds += took
            if data.get("stop_reason") != "pause_turn":
                break
            messages = [*messages, {"role": "assistant", "content": data.get("content") or []}]
        return responses, seconds

    def read(self, responses: Sequence[Mapping[str, Any]]) -> Reading:
        """The answer is the text after the last tool block (all text if there is none)."""
        blocks = [b for r in responses for b in r.get("content") or [] if isinstance(b, dict)]
        tools = [i for i, b in enumerate(blocks) if b.get("type") not in NOT_TOOL_BLOCKS]
        after = blocks[tools[-1] + 1 :] if tools else blocks
        text = "".join(b.get("text") or "" for b in after if b.get("type") == "text")
        if not text.strip():
            text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
        input_tokens = output_tokens = searches = 0
        for r in responses:
            usage = r.get("usage")
            if not isinstance(usage, dict):
                raise CompeteError(f"{self.config.name}: a response has no usage")
            # Prompt caching isn't used; any cache tokens are counted at the input price.
            input_tokens += sum(
                _count(usage.get(k))
                for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            )
            output_tokens += _count(usage.get("output_tokens"))
            searches += _count((usage.get("server_tool_use") or {}).get("web_search_requests"))
        items, error = parse_items(text)
        return Reading(items, text, error, input_tokens, output_tokens, searches)


class OpenRouterChat(_Client):
    """A model behind OpenRouter's chat completions, with the web search server tool or not."""

    def request(self, proposal: str) -> dict[str, Any]:
        cfg = self.config
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": render_prompt(proposal)}],
            "max_tokens": cfg.max_tokens,
        }
        if cfg.web_search_tool is not None:
            parameters: dict[str, Any] = {"engine": "native"}  # the provider's own search
            if cfg.max_searches is not None:
                parameters["max_uses"] = cfg.max_searches
            if cfg.search_context_size is not None:
                parameters["search_context_size"] = cfg.search_context_size
            body["tools"] = [{"type": cfg.web_search_tool, "parameters": parameters}]
        elif cfg.search_context_size is not None:
            body["web_search_options"] = {"search_context_size": cfg.search_context_size}
        if cfg.reasoning_effort is not None:
            body["reasoning"] = {"effort": cfg.reasoning_effort}
        return {"method": "POST", "url": f"{self.url}/chat/completions", "json": body}

    def send(
        self, request: Mapping[str, Any], billing: Billing
    ) -> tuple[list[dict[str, Any]], float]:
        api_key = self.api_key()
        if api_key is None:
            raise CompeteError(f"environment variable {self.config.api_key_env} is not set")
        headers = {"Authorization": f"Bearer {api_key}", **METADATA_HEADER}
        data, seconds = self.http.call(
            "POST", request["url"], headers=headers, json_body=request["json"], billing=billing
        )
        if not isinstance(data, dict):
            raise CompeteError(f"{self.config.name}: unexpected response {str(data)[:300]}")
        return [data], seconds

    def read(self, responses: Sequence[Mapping[str, Any]]) -> Reading:
        (data,) = responses
        try:
            text = data["choices"][0]["message"].get("content") or ""
            usage = data["usage"]
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise CompeteError(
                f"{self.config.name}: malformed chat completion: {str(data)[:300]}"
            ) from exc
        searches = _count((usage.get("server_tool_use") or {}).get("web_search_requests"))
        items, error = parse_items(text) if isinstance(text, str) else ([], "no text content")
        return Reading(
            items,
            text if isinstance(text, str) else "",
            error,
            prompt_tokens,
            completion_tokens,
            searches,
            billed_cost(data),
            served_provider(data),
        )


class _SemanticScholar(_Client):
    def missing_key(self) -> str | None:
        return None  # the key is optional

    def send(
        self, request: Mapping[str, Any], billing: Billing
    ) -> tuple[list[dict[str, Any]], float]:
        api_key = self.api_key()
        headers = {"x-api-key": api_key} if api_key else {}
        data, seconds = self.http.call(
            "GET", request["url"], headers=headers, params=request["params"], billing=billing
        )
        if not isinstance(data, dict):
            raise CompeteError(f"{self.config.name}: unexpected response {str(data)[:300]}")
        return [data], seconds


def s2_item(paper: Mapping[str, Any]) -> Item:
    """A Semantic Scholar paper: its DOI, or its arXiv id as arXiv's DataCite DOI."""
    ids = paper.get("externalIds") or {}
    doi = find_doi(_text(ids.get("DOI")))
    arxiv = _text(ids.get("ArXiv"))
    if doi is None and arxiv:
        doi = f"10.48550/arxiv.{arxiv.lower()}"
    return Item(_text(paper.get("title")), _year(paper.get("year")), doi, _text(paper.get("url")))


class SemanticScholarSearch(_SemanticScholar):
    """Semantic Scholar's paper search (relevance-ranked) with the proposal as the query."""

    def request(self, proposal: str) -> dict[str, Any]:
        params = {"query": proposal, "limit": str(MAX_ITEMS), "fields": S2_FIELDS}
        return {"method": "GET", "url": f"{self.url}/paper/search", "params": params}

    def read(self, responses: Sequence[Mapping[str, Any]]) -> Reading:
        papers = responses[0].get("data") or []
        items = [s2_item(p) for p in papers if isinstance(p, dict)]
        return Reading([i for i in items if i.title or i.doi][:MAX_ITEMS], "")


class SemanticScholarSnippets(_SemanticScholar):
    """Semantic Scholar's snippet search with the proposal as the query: each snippet's
    paper, in order of its best snippet."""

    def request(self, proposal: str) -> dict[str, Any]:
        params = {"query": proposal, "limit": str(S2_SNIPPETS)}
        return {"method": "GET", "url": f"{self.url}/snippet/search", "params": params}

    def read(self, responses: Sequence[Mapping[str, Any]]) -> Reading:
        items: list[Item] = []
        seen: set[str] = set()
        for snippet in responses[0].get("data") or []:
            paper = snippet.get("paper") if isinstance(snippet, dict) else None
            if not isinstance(paper, dict):
                continue
            corpus_id = str(paper.get("corpusId") or "")
            title = _text(paper.get("title"))
            key = corpus_id or (title or "")
            if not key or key in seen or title is None:
                continue
            seen.add(key)
            link = f"https://api.semanticscholar.org/CorpusId:{corpus_id}" if corpus_id else None
            items.append(Item(title, None, None, link))
        return Reading(items[:MAX_ITEMS], "")


CLIENTS: dict[str, type[_Client]] = {
    "anthropic": AnthropicWebSearch,
    "openrouter": OpenRouterChat,
    "s2-search": SemanticScholarSearch,
    "s2-snippets": SemanticScholarSnippets,
}


def make_system(
    config: CompeteSystemConfig,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> _Client:
    return CLIENTS[config.kind](config, transport=transport, sleep=sleep, clock=clock)


def ask(
    system: System,
    proposal: str,
    *,
    qid: str,
    cache: ResponseCache,
    ledger: SpendLedger,
) -> Answer:
    """One system's answer to one proposal: from the cache, or a real call under its cap."""
    cfg = system.config
    request = system.request(proposal)
    key = request_key(request)
    hit = cache.get(key)
    if hit is not None:
        responses, latency = hit
        reading = system.read(responses)
        return Answer(reading, method_cost(cfg, reading), 0.0, latency, cached=True)
    reserve = cfg.reserve_per_query
    ledger.charge(cfg.name, reserve, cfg.cost_cap)
    billing = Billing()
    try:
        responses, latency = system.send(request, billing)
        reading = system.read(responses)
    except BaseException:  # includes KeyboardInterrupt: book, then re-raise
        if billing.possible:
            ledger.record(cfg.name, reserved=reserve, qid=qid, cost=reserve)
        else:
            ledger.release(cfg.name, reserve)
        raise
    cost = method_cost(cfg, reading)
    billed = reading.billed_cost
    ledger.record(cfg.name, reserved=reserve, qid=qid, cost=cost, billed_cost=billed)
    cache.put(key, cfg.name, request, responses, latency)
    return Answer(reading, cost, cost if billed is None else billed, latency, cached=False)
