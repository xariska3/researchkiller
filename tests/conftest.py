"""Shared test doubles: a scripted FakeLLM and a deterministic FakeEncoder.

Tests never touch the network: real httpx transports are blocked for every
test, and LLM traffic goes through FakeLLM. Import the doubles with
``from conftest import FakeLLM, FakeEncoder, make_model_config``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from rk.config import ModelConfig
from rk.llm import Completion, LLMCache, Message, Usage, token_cost
from rk.text import content_stems

TEST_API_KEY_ENV = "RK_TEST_API_KEY"


def make_model_config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "name": "fake-model",
        "family": "fake",
        "base_url": "http://llm.test/v1",
        "api_key_env": TEST_API_KEY_ENV,
        "price_in_per_m": 1.0,
        "price_out_per_m": 2.0,
        "temperature": 0.0,
        "max_tokens": 64,
        "seed": 7,
    }
    values.update(overrides)
    return ModelConfig(**values)


class FakeLLM:
    """Scripted stand-in for an OpenAI-compatible chat model.

    `replies` is a list of reply strings, served in order, or a function from the
    messages to a reply. Token usage is whitespace word counts, so tests can
    predict it. Use it directly as a ChatModel (`complete`) or under a real
    LLMClient via `transport`, which answers like a /chat/completions endpoint.
    Every prompt received is recorded in `calls`.
    """

    def __init__(
        self,
        replies: Sequence[str] | Callable[[list[dict[str, str]]], str],
        *,
        model: ModelConfig | None = None,
    ) -> None:
        self.model = model or make_model_config()
        self._replies = replies if callable(replies) else list(replies)
        self.calls: list[list[dict[str, str]]] = []

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    def _reply(self, messages: Sequence[Message]) -> tuple[str, int, int]:
        msgs = [dict(m) for m in messages]
        self.calls.append(msgs)
        if callable(self._replies):
            text = self._replies(msgs)
        elif self._replies:
            text = self._replies.pop(0)
        else:
            raise AssertionError(f"FakeLLM script exhausted after {len(self.calls) - 1} replies")
        prompt_tokens = sum(self.count_tokens(m["content"]) for m in msgs)
        return text, prompt_tokens, self.count_tokens(text)

    def complete(self, messages: Sequence[Message], *, seed: int | None = None) -> Completion:
        text, prompt_tokens, completion_tokens = self._reply(messages)
        cost = token_cost(self.model, prompt_tokens, completion_tokens)
        usage = Usage(
            calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            method_cost=cost,
            money_spent=cost,
        )
        return Completion(
            text=text, usage=usage, cached=False, model=self.model.name, finish_reason="stop"
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        """httpx.MockTransport handler speaking the /chat/completions protocol."""
        body = json.loads(request.content)
        text, prompt_tokens, completion_tokens = self._reply(body["messages"])
        return httpx.Response(
            200,
            json={
                "id": f"fake-{len(self.calls)}",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)


class FakeEncoder:
    """Deterministic bag-of-stems encoder: feature-hashed, L2-normalised float32.

    Texts that share content stems get higher cosine similarity, so retrieval
    tests behave sensibly. Uses blake2b, not hash(), so vectors are identical
    across processes. Empty texts encode to the zero vector.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for s in content_stems(text):
                digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
                vectors[row, int.from_bytes(digest, "little") % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.divide(vectors, norms, out=vectors, where=norms > 0)
        return vectors


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"tests must not touch the network (tried {request.url})")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(TEST_API_KEY_ENV, "sk-test")
    return "sk-test"


@pytest.fixture
def llm_cache(tmp_path: Path) -> Iterator[LLMCache]:
    with LLMCache(tmp_path / "llm.sqlite") as cache:
        yield cache
