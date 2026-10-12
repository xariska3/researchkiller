"""OpenAI-compatible chat client with a SQLite response cache and a hard-capped cost ledger.

Cost accounting keeps two numbers (see Usage). *Method cost* is what the method
costs from scratch and is counted on cache hits too, so reported cost reflects
the method. *Money spent* counts real API calls only. Before every real call the
client charges a pessimistic estimate to the Ledger, which refuses calls that
could take cumulative spend (across all commands, persisted next to the cache)
over its cap. Cache hits never touch the network, the ledger or the API key, so
a cached run replays offline and for free.

On OpenRouter, a model's `provider` pins one endpoint with fallbacks disabled
and every request parameter required
(https://openrouter.ai/docs/guides/routing/provider-selection), and every
request asks for router metadata (https://openrouter.ai/docs/guides/features/router-metadata)
so the provider that served each call is stored with the reply and its spend row.
The billed cost OpenRouter returns in `usage.cost` (USD, on every non-streaming
reply: https://openrouter.ai/docs/guides/guides/usage-accounting) is stored
next to the cost computed from the config prices, and the ledger counts it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from string import Template
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from rk.config import ModelConfig

if TYPE_CHECKING:  # annotation only: typing.Self needs Python 3.11
    from typing_extensions import Self

Message = Mapping[str, str]  # {"role": ..., "content": ...}

METADATA_HEADER = {"X-OpenRouter-Metadata": "enabled"}  # other servers ignore it


class _Closing:
    """`with` support for classes that hold a resource released by close()."""

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class LLMError(RuntimeError):
    """A completion failed: bad request, malformed response, or retries exhausted."""


class BudgetExceeded(RuntimeError):
    """A real API call could take total spend over the ledger's cap."""


@dataclass(frozen=True)
class Usage:
    """Token and cost totals for one or more completions (USD)."""

    calls: int = 0  # completions served, cached or not
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    method_cost: float = 0.0  # counted on cache hits too
    money_spent: float = 0.0  # real API calls only

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            calls=self.calls + other.calls,
            cache_hits=self.cache_hits + other.cache_hits,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            method_cost=self.method_cost + other.method_cost,
            money_spent=self.money_spent + other.money_spent,
        )


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    cached: bool
    model: str
    finish_reason: str | None
    provider: str | None = None  # who served it, as OpenRouter reports (also on cache hits)


class ChatModel(Protocol):
    """Anything that answers chat messages: LLMClient, or FakeLLM in tests."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> Completion: ...


@dataclass(frozen=True)
class Prompt:
    """A versioned prompt template from rk/prompts, with the sha256 of its text."""

    name: str
    template: Template  # $placeholders, so JSON braces need no escaping
    sha256: str

    def render(self, **values: str) -> str:
        return self.template.substitute(values)


def load_prompt(name: str) -> Prompt:
    """Load rk/prompts/<name>.txt.

    The file is read with universal newlines, so the text sent to the model and
    its hash don't depend on the line endings of the checkout.
    """
    path = resources.files("rk").joinpath("prompts").joinpath(f"{name}.txt")
    text = path.read_text(encoding="utf-8")
    return Prompt(name, Template(text), hashlib.sha256(text.encode("utf-8")).hexdigest())


def token_cost(model: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost of a call with these token counts at the model's prices."""
    return (
        prompt_tokens * model.price_in_per_m + completion_tokens * model.price_out_per_m
    ) / 1_000_000


def prompt_token_bound(messages: Sequence[Message]) -> int:
    """Upper bound on prompt tokens: the UTF-8 size of the JSON-encoded messages.

    Byte-level tokenizers emit at most one token per byte, and the JSON syntax
    more than covers the chat-template overhead.
    """
    return len(json.dumps([dict(m) for m in messages], ensure_ascii=False).encode("utf-8"))


def pessimistic_cost(model: ModelConfig, messages: Sequence[Message]) -> float:
    """Upper bound on a real call's cost (prompt bound + max_tokens), charged before sending."""
    return token_cost(model, prompt_token_bound(messages), model.max_tokens)


def cache_key(
    model: str,
    messages: Sequence[Message],
    temperature: float,
    seed: int,
    max_tokens: int,
    provider: Mapping[str, Any] | None = None,
    response_format: Mapping[str, Any] | None = None,
) -> str:
    """sha256 of the canonical JSON of (model, messages, temperature, seed, max_tokens).

    A provider pin (OpenRouter routing) and a response format (e.g. a JSON
    schema) are part of the key when set, so a reply is never served for a
    request that differs in either; without them the key is unchanged.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(m) for m in messages],
        "temperature": float(temperature),
        "seed": int(seed),
        "max_tokens": int(max_tokens),
    }
    if provider is not None:
        payload["provider"] = dict(provider)
    if response_format is not None:
        payload["response_format"] = dict(response_format)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedReply:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    provider: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS completions (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    text TEXT NOT NULL,
    finish_reason TEXT,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    provider TEXT
)
"""


_SPEND_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    timestamp TEXT NOT NULL,
    command TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost REAL NOT NULL,
    billed_cost REAL,
    provider TEXT
)
"""


def _connect(path: str | Path, schema: str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(schema)
    conn.commit()
    return conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LLMCache(_Closing):
    """SQLite store of completions keyed by cache_key(); one file can serve every model."""

    def __init__(self, path: str | Path) -> None:
        self._conn = _connect(path, _SCHEMA)

    def get(self, key: str) -> CachedReply | None:
        row = self._conn.execute(
            "SELECT text, finish_reason, prompt_tokens, completion_tokens, provider"
            " FROM completions WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else CachedReply(*row)

    def put(self, key: str, request: Mapping[str, Any], reply: CachedReply) -> None:
        """Store a reply with its full request, committing at once so paid calls survive a crash."""
        self._conn.execute(
            "INSERT OR REPLACE INTO completions (key, model, request, text, finish_reason,"
            " prompt_tokens, completion_tokens, created_at, provider)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                request["model"],
                json.dumps(request, sort_keys=True, ensure_ascii=False),
                reply.text,
                reply.finish_reason,
                reply.prompt_tokens,
                reply.completion_tokens,
                _utc_now(),
                reply.provider,
            ),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0])

    def close(self) -> None:
        self._conn.close()


class Ledger(_Closing):
    """Cumulative LLM spend with a hard cap, persisted in the `spend` table.

    Use the same SQLite file as the LLMCache. Spend adds up across every command
    and process using that file: the running total is read at startup. Before a
    real call the client reserves a pessimistic estimate (`charge`); afterwards it
    `record`s what the call cost, which releases the reservation and appends a
    spend row. A row's spend is its `billed_cost` (what the API reports it
    charged) when present, else its `cost` (computed from the config prices).
    A failed call is recorded at its estimate if any request was sent
    without a 4xx coming back (timeouts, transport errors, 5xx, malformed 200s,
    Ctrl-C mid-request), since it may have been billed; otherwise its reservation
    is just `release`d. Spend by a command running concurrently is only seen
    after a restart, so run commands one at a time.
    """

    def __init__(self, path: str | Path, cap: float = 20.0, *, command: str) -> None:
        if cap < 0:
            raise ValueError("cap must be >= 0")
        self.cap = cap
        self.command = command
        self._conn = _connect(path, _SPEND_SCHEMA)
        row = self._conn.execute(
            "SELECT COALESCE(SUM(COALESCE(billed_cost, cost)), 0.0) FROM spend"
        ).fetchone()
        self._recorded = float(row[0])
        self._reserved = 0.0

    @property
    def spent(self) -> float:
        """Recorded spend across all commands, plus reservations for calls in flight."""
        return self._recorded + self._reserved

    @property
    def remaining(self) -> float:
        return self.cap - self.spent

    def charge(self, amount: float, *, label: str = "call") -> None:
        """Reserve `amount` for a real call, or raise BudgetExceeded if it might not fit."""
        if self.spent + amount > self.cap:
            raise BudgetExceeded(
                f"{label} could cost up to ${amount:.6f}, but only ${self.remaining:.6f} "
                f"of the ${self.cap:.2f} cumulative budget cap is left"
            )
        self._reserved += amount

    def record(
        self,
        *,
        reserved: float,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        billed_cost: float | None = None,
        provider: str | None = None,
    ) -> None:
        """Release a reservation and persist what the call cost, and who served it if known.

        `cost` is computed from the config prices; `billed_cost`, if the API
        reported one, is what it charged and is what counts against the cap.
        """
        self._conn.execute(
            "INSERT INTO spend (timestamp, command, model, prompt_tokens, completion_tokens,"
            " cost, billed_cost, provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _utc_now(),
                self.command,
                model,
                prompt_tokens,
                completion_tokens,
                cost,
                billed_cost,
                provider,
            ),
        )
        self._conn.commit()
        self._reserved -= reserved
        self._recorded += cost if billed_cost is None else billed_cost

    def release(self, reserved: float) -> None:
        """Drop a reservation for a call that cost nothing."""
        self._reserved -= reserved

    def close(self) -> None:
        self._conn.close()


class LLMClient(_Closing):
    """Chat completions for one model role, cached and charged to a shared ledger."""

    def __init__(
        self,
        model: ModelConfig,
        cache: LLMCache,
        ledger: Ledger,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.cache = cache
        self.ledger = ledger
        self.usage = Usage()  # running total over this client's completions
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._sleep = sleep
        self._http = httpx.Client(base_url=model.base_url, timeout=timeout, transport=transport)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> Completion:
        """Answer `messages`, from the cache when possible.

        `seed` overrides the model's configured seed, e.g. to get a fresh sample
        on a retry. `response_format` is sent as the OpenAI-style field (e.g.
        {"type": "json_schema", ...}); with a pinned provider, require_parameters
        makes OpenRouter refuse an endpoint that can't honour it. Both are part
        of the cache key.
        """
        request: dict[str, Any] = {
            "model": self.model.name,
            "messages": _normalize_messages(messages),
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "seed": self.model.seed if seed is None else seed,
        }
        if self.model.provider is not None:
            request["provider"] = provider_routing(self.model.provider)
        if response_format is not None:
            request["response_format"] = dict(response_format)
        key = cache_key(**request)
        reply = self.cache.get(key)
        cached = reply is not None
        if reply is None:
            reply = self._real_call(request)
            self.cache.put(key, request, reply)
        cost = token_cost(self.model, reply.prompt_tokens, reply.completion_tokens)
        usage = Usage(
            calls=1,
            cache_hits=int(cached),
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            method_cost=cost,
            money_spent=0.0 if cached else cost,
        )
        self.usage += usage
        return Completion(
            text=reply.text,
            usage=usage,
            cached=cached,
            model=self.model.name,
            finish_reason=reply.finish_reason,
            provider=reply.provider,
        )

    def _real_call(self, request: dict[str, Any]) -> CachedReply:
        api_key = os.environ.get(self.model.api_key_env)
        if not api_key:
            raise LLMError(
                f"environment variable {self.model.api_key_env} is not set "
                f"(needed for {self.model.name})"
            )
        prompt_bound = prompt_token_bound(request["messages"])
        if "response_format" in request:  # a schema is sent to the model too
            prompt_bound += len(json.dumps(request["response_format"]).encode("utf-8"))
        estimate = token_cost(self.model, prompt_bound, self.model.max_tokens)
        self.ledger.charge(estimate, label=self.model.name)
        billing = _Billing()
        try:
            data = self._post(request, api_key, billing)
            reply = _parse_reply(data, self.model.name)
        except BaseException:  # includes KeyboardInterrupt: book, then re-raise
            if billing.possible:
                self.ledger.record(
                    reserved=estimate,
                    model=self.model.name,
                    prompt_tokens=prompt_bound,
                    completion_tokens=self.model.max_tokens,
                    cost=estimate,
                )
            else:
                self.ledger.release(estimate)
            raise
        self.ledger.record(
            reserved=estimate,
            model=self.model.name,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            cost=token_cost(self.model, reply.prompt_tokens, reply.completion_tokens),
            billed_cost=billed_cost(data),
            provider=reply.provider,
        )
        return reply

    def _post(self, request: dict[str, Any], api_key: str, billing: _Billing) -> Any:
        """POST /chat/completions, retrying 429, 5xx and transport errors with backoff.

        Every request counts as possibly billed from the moment it is sent; only a
        4xx response clears that again. `billing` carries the verdict out, even
        when the call ends in an exception.
        """
        headers = {"Authorization": f"Bearer {api_key}", **METADATA_HEADER}
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            billed_before = billing.possible
            billing.possible = True
            try:
                response = self._http.post("/chat/completions", json=request, headers=headers)
            except httpx.TransportError as exc:  # timeouts are transport errors in httpx
                problem = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if 400 <= status < 500:
                    billing.possible = billed_before  # rejected, so not billed
                if status == 429 or status >= 500:
                    problem = f"HTTP {status}: {response.text[:300]}"
                    retry_after = retry_after_seconds(response)
                elif response.is_error:
                    raise LLMError(f"{self.model.name}: HTTP {status}: {response.text[:500]}")
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise LLMError(f"{self.model.name}: response is not JSON") from exc
            if attempt == self.max_retries:
                raise LLMError(
                    f"{self.model.name}: giving up after {attempt + 1} attempts; "
                    f"last error: {problem}"
                )
            delay = self.backoff_base * 2**attempt
            if retry_after is not None:
                delay = max(delay, retry_after)
            self._sleep(min(delay, self.backoff_max))
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._http.close()


class _Billing:
    """Whether a real call may have been billed: a request went out and no 4xx came back."""

    def __init__(self) -> None:
        self.possible = False


def _normalize_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError(f"each message needs string 'role' and 'content': {message!r}")
        out.append({"role": role, "content": content})
    if not out:
        raise ValueError("messages must not be empty")
    return out


def _parse_reply(data: Any, model_name: str) -> CachedReply:
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data["usage"]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"{model_name}: malformed chat completion: {str(data)[:300]}") from exc
    if not isinstance(text, str):
        raise LLMError(f"{model_name}: chat completion has no text content")
    return CachedReply(
        text=text,
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider=served_provider(data),
    )


def provider_routing(slug: str) -> dict[str, Any]:
    """OpenRouter's `provider` request field: only this endpoint, no fallbacks.

    With require_parameters, OpenRouter refuses the request (a 4xx, so neither
    retried nor booked) rather than send it to an endpoint that would ignore a
    parameter such as `seed`.
    """
    return {"order": [slug], "allow_fallbacks": False, "require_parameters": True}


def billed_cost(data: Mapping[str, Any]) -> float | None:
    """What the API says it charged for a completion (OpenRouter's usage.cost, USD), else None."""
    try:
        cost = data["usage"]["cost"]
    except (KeyError, TypeError):
        return None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    return float(cost) if math.isfinite(cost) and cost >= 0 else None


def served_provider(data: Mapping[str, Any]) -> str | None:
    """The provider OpenRouter reports as having served a completion, else None.

    The documented place is the `selected` entry of
    openrouter_metadata.endpoints.available (sent because of METADATA_HEADER);
    responses also carry a top-level "provider" string, used as the fallback.
    """
    try:
        available = data["openrouter_metadata"]["endpoints"]["available"]
        selected = [e["provider"] for e in available if e.get("selected") is True]
    except (KeyError, TypeError, AttributeError):
        selected = []
    provider = selected[0] if selected else data.get("provider")
    return provider if isinstance(provider, str) else None


def retry_after_seconds(response: httpx.Response) -> float | None:
    """A numeric Retry-After header in seconds, if the server sent one."""
    try:
        return max(0.0, float(response.headers["retry-after"]))
    except (KeyError, ValueError):
        return None


_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


def parse_json(text: str) -> Any:
    """Parse JSON from model output, tolerating code fences and surrounding prose.

    Tries a fenced block first, then the whole text, then the first `{` or `[`
    from which a complete JSON value decodes (which also handles an unclosed
    fence from a truncated reply). Raises ValueError if nothing parses.
    """
    fenced = _FENCE_RE.search(text)
    candidates = ([fenced.group(1)] if fenced else []) + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for start, ch in enumerate(candidate):
            if ch in "{[":
                try:
                    return _DECODER.raw_decode(candidate, start)[0]
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no JSON value found in model output: {text[:200]!r}")
