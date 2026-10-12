"""OpenAI-compatible chat client with a SQLite response cache and a hard-capped cost ledger.

Cost accounting keeps two numbers (see Usage). *Method cost* is what the method
costs from scratch and is counted on cache hits too, so reported cost reflects
the method. *Money spent* counts real API calls only. Before every real call the
client charges a pessimistic estimate to the shared Ledger, which refuses calls
that could take total spend over its cap. Cache hits never touch the network,
the ledger or the API key, so a cached run replays offline and for free.
"""

from __future__ import annotations

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
from types import TracebackType
from typing import Any, Protocol

import httpx

from rk.config import ModelConfig

Message = Mapping[str, str]  # {"role": ..., "content": ...}


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


class ChatModel(Protocol):
    """Anything that answers chat messages: LLMClient, or FakeLLM in tests."""

    def complete(self, messages: Sequence[Message], *, seed: int | None = None) -> Completion: ...


def token_cost(model: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost of a call with these token counts at the model's prices."""
    return (
        prompt_tokens * model.price_in_per_m + completion_tokens * model.price_out_per_m
    ) / 1_000_000


def pessimistic_cost(model: ModelConfig, messages: Sequence[Message]) -> float:
    """Upper bound on a real call's cost, charged to the ledger before sending.

    Prompt tokens are bounded by the UTF-8 size of the JSON-encoded messages:
    byte-level tokenizers emit at most one token per byte, and the JSON syntax
    more than covers the chat-template overhead. Completion tokens are bounded by
    max_tokens.
    """
    encoded = json.dumps([dict(m) for m in messages], ensure_ascii=False).encode("utf-8")
    return token_cost(model, len(encoded), model.max_tokens)


def cache_key(
    model: str, messages: Sequence[Message], temperature: float, seed: int, max_tokens: int
) -> str:
    """sha256 of the canonical JSON of (model, messages, temperature, seed, max_tokens)."""
    payload = {
        "model": model,
        "messages": [dict(m) for m in messages],
        "temperature": float(temperature),
        "seed": int(seed),
        "max_tokens": int(max_tokens),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedReply:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS completions (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    text TEXT NOT NULL,
    finish_reason TEXT,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL
)
"""


class LLMCache:
    """SQLite store of completions keyed by cache_key(); one file can serve every model."""

    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> CachedReply | None:
        row = self._conn.execute(
            "SELECT text, finish_reason, prompt_tokens, completion_tokens"
            " FROM completions WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else CachedReply(*row)

    def put(self, key: str, request: Mapping[str, Any], reply: CachedReply) -> None:
        """Store a reply with its full request, committing at once so paid calls survive a crash."""
        self._conn.execute(
            "INSERT OR REPLACE INTO completions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                request["model"],
                json.dumps(request, sort_keys=True, ensure_ascii=False),
                reply.text,
                reply.finish_reason,
                reply.prompt_tokens,
                reply.completion_tokens,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> LLMCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class Ledger:
    """Spend tracker with a hard cap, shared by every LLMClient in a process.

    Before a real call the client charges a pessimistic estimate; when the call
    returns, the estimate is settled to the actual cost. A call that fails keeps
    its estimate, since it may still have been billed.
    """

    def __init__(self, cap: float = 20.0) -> None:
        if cap < 0:
            raise ValueError("cap must be >= 0")
        self.cap = cap
        self.spent = 0.0

    @property
    def remaining(self) -> float:
        return self.cap - self.spent

    def charge(self, amount: float, *, label: str = "call") -> None:
        if self.spent + amount > self.cap:
            raise BudgetExceeded(
                f"{label} could cost up to ${amount:.6f}, but only ${self.remaining:.6f} "
                f"of the ${self.cap:.2f} budget cap is left"
            )
        self.spent += amount

    def settle(self, estimate: float, actual: float) -> None:
        self.spent += actual - estimate


class LLMClient:
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

    def complete(self, messages: Sequence[Message], *, seed: int | None = None) -> Completion:
        """Answer `messages`, from the cache when possible.

        `seed` overrides the model's configured seed, e.g. to get a fresh sample
        on a retry; it is part of the cache key.
        """
        request: dict[str, Any] = {
            "model": self.model.name,
            "messages": _normalize_messages(messages),
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "seed": self.model.seed if seed is None else seed,
        }
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
        )

    def _real_call(self, request: dict[str, Any]) -> CachedReply:
        api_key = os.environ.get(self.model.api_key_env)
        if not api_key:
            raise LLMError(
                f"environment variable {self.model.api_key_env} is not set "
                f"(needed for {self.model.name})"
            )
        estimate = pessimistic_cost(self.model, request["messages"])
        self.ledger.charge(estimate, label=self.model.name)
        reply = _parse_reply(self._post(request, api_key), self.model.name)
        actual = token_cost(self.model, reply.prompt_tokens, reply.completion_tokens)
        self.ledger.settle(estimate, actual)
        return reply

    def _post(self, request: dict[str, Any], api_key: str) -> Any:
        """POST /chat/completions, retrying 429, 5xx and transport errors with backoff."""
        headers = {"Authorization": f"Bearer {api_key}"}
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            try:
                response = self._http.post("/chat/completions", json=request, headers=headers)
            except httpx.TransportError as exc:
                problem = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status == 429 or status >= 500:
                    problem = f"HTTP {status}: {response.text[:300]}"
                    retry_after = _retry_after_seconds(response)
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

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _normalize_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"each message needs string 'role' and 'content': {message!r}")
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
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
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
