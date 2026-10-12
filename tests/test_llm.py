from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import TEST_API_KEY_ENV, FakeLLM, make_model_config

from rk.llm import (
    BudgetExceeded,
    LLMCache,
    LLMClient,
    LLMError,
    Ledger,
    Usage,
    cache_key,
    parse_json,
    pessimistic_cost,
    token_cost,
)

MSGS = [{"role": "user", "content": "What is the answer?"}]  # 4 fake tokens


def test_cache_hit_reuses_reply_and_still_counts_method_cost(
    llm_cache: LLMCache, api_key: str
) -> None:
    model = make_model_config()
    fake = FakeLLM(["The answer is 42."])
    client = LLMClient(model, llm_cache, Ledger(cap=1.0), transport=fake.transport)

    first = client.complete(MSGS)
    second = client.complete(MSGS)

    assert len(fake.calls) == 1  # the second answer came from the cache
    assert (first.cached, second.cached) == (False, True)
    assert second.text == first.text == "The answer is 42."
    cost = token_cost(model, 4, 4)
    assert first.usage == Usage(
        calls=1, prompt_tokens=4, completion_tokens=4, method_cost=cost, money_spent=cost
    )
    assert second.usage == Usage(
        calls=1, cache_hits=1, prompt_tokens=4, completion_tokens=4, method_cost=cost
    )
    assert client.usage == first.usage + second.usage
    assert client.usage.method_cost == pytest.approx(2 * cost)
    assert client.usage.money_spent == pytest.approx(cost)


def test_cache_persists_and_replays_without_api_key_or_budget(
    tmp_path: Path, api_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "llm.sqlite"
    model = make_model_config()
    with LLMCache(path) as cache:
        LLMClient(model, cache, Ledger(), transport=FakeLLM(["cached reply"]).transport).complete(
            MSGS
        )

    monkeypatch.delenv(TEST_API_KEY_ENV)
    silent = FakeLLM([])  # would fail the test if asked
    with LLMCache(path) as cache:
        client = LLMClient(model, cache, Ledger(cap=0.0), transport=silent.transport)
        out = client.complete(MSGS)

    assert out.cached and out.text == "cached reply"
    assert out.usage.money_spent == 0.0 and out.usage.method_cost > 0.0
    assert silent.calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other-model"},
        {"messages": [{"role": "user", "content": "What is the question?"}]},
        {"temperature": 0.5},
        {"seed": 8},
        {"max_tokens": 65},
    ],
)
def test_cache_key_covers_every_request_field(change: dict[str, Any]) -> None:
    base: dict[str, Any] = {
        "model": "m",
        "messages": MSGS,
        "temperature": 0.0,
        "seed": 7,
        "max_tokens": 64,
    }
    assert cache_key(**base) == cache_key(**{**base, "temperature": 0})  # int vs float
    assert cache_key(**base) != cache_key(**{**base, **change})


def test_seed_override_is_a_separate_cache_entry(llm_cache: LLMCache, api_key: str) -> None:
    fake = FakeLLM(["first sample", "second sample"])
    client = LLMClient(make_model_config(), llm_cache, Ledger(), transport=fake.transport)
    assert client.complete(MSGS).text == "first sample"
    assert client.complete(MSGS, seed=8).text == "second sample"
    assert client.complete(MSGS, seed=8).cached
    assert len(fake.calls) == 2


def test_ledger_refuses_call_over_cap_before_sending(llm_cache: LLMCache, api_key: str) -> None:
    model = make_model_config()
    fake = FakeLLM(["never sent"])
    ledger = Ledger(cap=0.5 * pessimistic_cost(model, MSGS))
    client = LLMClient(model, llm_cache, ledger, transport=fake.transport)

    with pytest.raises(BudgetExceeded):
        client.complete(MSGS)

    assert fake.calls == []
    assert ledger.spent == 0.0
    assert len(llm_cache) == 0


def test_ledger_charges_estimate_before_call_then_settles_to_actual(
    llm_cache: LLMCache, api_key: str
) -> None:
    model = make_model_config()
    estimate = pessimistic_cost(model, MSGS)
    ledger = Ledger(cap=1.0)
    spent_during_call: list[float] = []

    def reply(messages: list[dict[str, str]]) -> str:
        spent_during_call.append(ledger.spent)
        return "a short reply"

    client = LLMClient(model, llm_cache, ledger, transport=FakeLLM(reply).transport)
    out = client.complete(MSGS)

    assert spent_during_call == [pytest.approx(estimate)]
    assert out.usage.money_spent < estimate  # the estimate really is pessimistic
    assert ledger.spent == pytest.approx(out.usage.money_spent)


def test_ledger_hard_cap() -> None:
    assert Ledger().cap == 20.0
    ledger = Ledger(cap=1.0)
    ledger.charge(0.6)
    with pytest.raises(BudgetExceeded, match="budget cap"):
        ledger.charge(0.5)
    assert ledger.spent == pytest.approx(0.6)
    ledger.settle(estimate=0.6, actual=0.1)
    assert ledger.remaining == pytest.approx(0.9)
    ledger.charge(0.5)  # fits now


def test_retries_429_5xx_and_transport_errors_with_backoff(
    llm_cache: LLMCache, api_key: str
) -> None:
    fake = FakeLLM(["finally"])
    failures: list[httpx.Response | None] = [
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(503),
        None,  # connection error
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if failures:
            failure = failures.pop(0)
            if failure is None:
                raise httpx.ConnectError("connection refused", request=request)
            return failure
        return fake.handle(request)

    sleeps: list[float] = []
    client = LLMClient(
        make_model_config(),
        llm_cache,
        Ledger(),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    assert client.complete(MSGS).text == "finally"
    assert sleeps == [3.0, 2.0, 4.0]  # Retry-After beats backoff 1.0, then 2.0, 4.0


def test_client_errors_are_not_retried(llm_cache: LLMCache, api_key: str) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400, json={"error": "bad request"})

    sleeps: list[float] = []
    client = LLMClient(
        make_model_config(),
        llm_cache,
        Ledger(),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    with pytest.raises(LLMError, match="HTTP 400"):
        client.complete(MSGS)
    assert len(seen) == 1 and sleeps == []


def test_gives_up_after_max_retries_and_keeps_estimate_charged(
    llm_cache: LLMCache, api_key: str
) -> None:
    model = make_model_config()
    ledger = Ledger()
    sleeps: list[float] = []
    client = LLMClient(
        model,
        llm_cache,
        ledger,
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        max_retries=2,
        sleep=sleeps.append,
    )
    with pytest.raises(LLMError, match="giving up after 3 attempts"):
        client.complete(MSGS)
    assert sleeps == [1.0, 2.0]
    assert ledger.spent == pytest.approx(pessimistic_cost(model, MSGS))
    assert len(llm_cache) == 0


def test_missing_api_key_fails_before_charging(
    llm_cache: LLMCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TEST_API_KEY_ENV, raising=False)
    ledger = Ledger()
    client = LLMClient(make_model_config(), llm_cache, ledger, transport=FakeLLM(["x"]).transport)
    with pytest.raises(LLMError, match=TEST_API_KEY_ENV):
        client.complete(MSGS)
    assert ledger.spent == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"verdict": "yes"}', {"verdict": "yes"}),
        ('```json\n{"verdict": "yes"}\n```', {"verdict": "yes"}),
        ("```\n[1, 2]\n```", [1, 2]),
        ('Here it is:\n```JSON\n{"a": 1}\n```\nHope this helps.', {"a": 1}),
        ('Sure! {"a": {"b": [1, 2]}} -- done', {"a": {"b": [1, 2]}}),
        ('Options [maybe]: {"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}', {"a": 1}),  # unclosed fence from a truncated reply
    ],
)
def test_parse_json_tolerates_code_fences_and_prose(text: str, expected: Any) -> None:
    assert parse_json(text) == expected


@pytest.mark.parametrize("text", ["no json here", "```json\n{not json}\n```", ""])
def test_parse_json_raises_when_nothing_parses(text: str) -> None:
    with pytest.raises(ValueError, match="no JSON value"):
        parse_json(text)
