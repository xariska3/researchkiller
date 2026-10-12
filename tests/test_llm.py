from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import TEST_API_KEY_ENV, FakeLLM, make_model_config

from rk.llm import (
    BudgetExceeded,
    Ledger,
    LLMCache,
    LLMClient,
    LLMError,
    Usage,
    cache_key,
    parse_json,
    pessimistic_cost,
    prompt_token_bound,
    token_cost,
)

MSGS = [{"role": "user", "content": "What is the answer?"}]  # 4 fake tokens
OTHER_MSGS = [{"role": "user", "content": "Something else entirely?"}]

MakeLedger = Callable[..., Ledger]


def _spend_rows(db_path: Path) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(
            "SELECT command, model, prompt_tokens, completion_tokens, cost FROM spend"
        ).fetchall()


def test_cache_hit_reuses_reply_and_still_counts_method_cost(
    llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
) -> None:
    model = make_model_config()
    fake = FakeLLM(["The answer is 42."])
    client = LLMClient(model, llm_cache, make_ledger(cap=1.0), transport=fake.transport)

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
    db_path: Path, api_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = make_model_config()
    with LLMCache(db_path) as cache, Ledger(db_path, command="make-bench") as ledger:
        LLMClient(model, cache, ledger, transport=FakeLLM(["cached reply"]).transport).complete(
            MSGS
        )

    monkeypatch.delenv(TEST_API_KEY_ENV)
    silent = FakeLLM([])  # would fail the test if asked
    with LLMCache(db_path) as cache, Ledger(db_path, cap=0.0, command="run") as ledger:
        out = LLMClient(model, cache, ledger, transport=silent.transport).complete(MSGS)

    assert out.cached and out.text == "cached reply"
    assert out.usage.money_spent == 0.0 and out.usage.method_cost > 0.0
    assert silent.calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other-model"},
        {"messages": OTHER_MSGS},
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


def test_seed_override_is_a_separate_cache_entry(
    llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
) -> None:
    fake = FakeLLM(["first sample", "second sample"])
    client = LLMClient(make_model_config(), llm_cache, make_ledger(), transport=fake.transport)
    assert client.complete(MSGS).text == "first sample"
    assert client.complete(MSGS, seed=8).text == "second sample"
    assert client.complete(MSGS, seed=8).cached
    assert len(fake.calls) == 2


def test_ledger_refuses_call_over_cap_before_sending(
    db_path: Path, llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
) -> None:
    model = make_model_config()
    fake = FakeLLM(["never sent"])
    ledger = make_ledger(cap=0.5 * pessimistic_cost(model, MSGS))
    client = LLMClient(model, llm_cache, ledger, transport=fake.transport)

    with pytest.raises(BudgetExceeded):
        client.complete(MSGS)

    assert fake.calls == []
    assert ledger.spent == 0.0
    assert len(llm_cache) == 0
    assert _spend_rows(db_path) == []


def test_ledger_reserves_estimate_during_call_then_records_actual(
    db_path: Path, llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
) -> None:
    model = make_model_config()
    estimate = pessimistic_cost(model, MSGS)
    ledger = make_ledger(cap=1.0)
    spent_during_call: list[float] = []

    def reply(messages: list[dict[str, str]]) -> str:
        spent_during_call.append(ledger.spent)
        return "a short reply"

    client = LLMClient(model, llm_cache, ledger, transport=FakeLLM(reply).transport)
    out = client.complete(MSGS)

    assert spent_during_call == [pytest.approx(estimate)]
    assert out.usage.money_spent < estimate  # the estimate really is pessimistic
    assert ledger.spent == pytest.approx(out.usage.money_spent)
    assert _spend_rows(db_path) == [("test", "fake-model", 4, 3, token_cost(model, 4, 3))]


def test_ledger_hard_cap_and_reservations() -> None:
    with Ledger(":memory:", command="test") as ledger:
        assert ledger.cap == 20.0
    with Ledger(":memory:", cap=1.0, command="test") as ledger:
        ledger.charge(0.6)
        with pytest.raises(BudgetExceeded, match="budget cap"):
            ledger.charge(0.5)
        assert ledger.spent == pytest.approx(0.6)
        ledger.record(reserved=0.6, model="m", prompt_tokens=10, completion_tokens=5, cost=0.1)
        assert ledger.remaining == pytest.approx(0.9)
        ledger.charge(0.5)  # fits now


def test_spend_persists_across_ledger_instances(db_path: Path) -> None:
    with Ledger(db_path, cap=1.0, command="make-bench") as first:
        first.charge(0.5)
        first.record(reserved=0.5, model="m", prompt_tokens=1000, completion_tokens=200, cost=0.3)
        assert first.spent == pytest.approx(0.3)

    with Ledger(db_path, cap=1.0, command="run") as second:
        assert second.spent == pytest.approx(0.3)  # running total read at startup
        with pytest.raises(BudgetExceeded, match="cumulative"):
            second.charge(0.8)
        second.charge(0.6)  # still fits next to the earlier 0.3

    assert _spend_rows(db_path) == [("make-bench", "m", 1000, 200, 0.3)]


def test_budget_is_cumulative_across_commands_through_the_client(
    db_path: Path, api_key: str
) -> None:
    model = make_model_config()
    with LLMCache(db_path) as cache, Ledger(db_path, cap=1.0, command="make-bench") as ledger:
        fake = FakeLLM(["four words of reply"])
        first = LLMClient(model, cache, ledger, transport=fake.transport).complete(MSGS)

    # The next command's cap leaves less than one more pessimistic call.
    cap = first.usage.money_spent + 0.5 * pessimistic_cost(model, OTHER_MSGS)
    silent = FakeLLM(["never sent"])
    with LLMCache(db_path) as cache, Ledger(db_path, cap=cap, command="run") as ledger:
        client = LLMClient(model, cache, ledger, transport=silent.transport)
        assert client.complete(MSGS).cached  # replays stay free
        with pytest.raises(BudgetExceeded):
            client.complete(OTHER_MSGS)

    assert silent.calls == []
    assert _spend_rows(db_path) == [("make-bench", "fake-model", 4, 4, first.usage.money_spent)]


def test_retries_429_5xx_and_transport_errors_with_backoff(
    llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
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
        make_ledger(),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    assert client.complete(MSGS).text == "finally"
    assert sleeps == [3.0, 2.0, 4.0]  # Retry-After beats backoff 1.0, then 2.0, 4.0


def test_client_errors_are_not_retried(
    llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400, json={"error": "bad request"})

    sleeps: list[float] = []
    client = LLMClient(
        make_model_config(),
        llm_cache,
        make_ledger(),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    with pytest.raises(LLMError, match="HTTP 400"):
        client.complete(MSGS)
    assert len(seen) == 1 and sleeps == []


Outcome = int | type[httpx.TransportError]


def _failing_transport(*outcomes: Outcome) -> httpx.MockTransport:
    """Answer each request with the next status code or transport error; the last repeats."""
    queue = list(outcomes)

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, int):
            return httpx.Response(outcome, json={"error": "simulated"})
        raise outcome("simulated failure", request=request)

    return httpx.MockTransport(handler)


def _failing_client(
    llm_cache: LLMCache, ledger: Ledger, *outcomes: Outcome, sleeps: list[float] | None = None
) -> LLMClient:
    return LLMClient(
        make_model_config(),
        llm_cache,
        ledger,
        transport=_failing_transport(*outcomes),
        max_retries=2,
        sleep=(sleeps if sleeps is not None else []).append,
    )


def test_gives_up_after_max_retries(
    llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
) -> None:
    sleeps: list[float] = []
    client = _failing_client(llm_cache, make_ledger(), 500, sleeps=sleeps)
    with pytest.raises(LLMError, match="giving up after 3 attempts"):
        client.complete(MSGS)
    assert sleeps == [1.0, 2.0]
    assert len(llm_cache) == 0


@pytest.mark.parametrize("failure", [500, 503, httpx.ReadTimeout, httpx.ConnectError])
def test_failed_call_is_booked_for_timeouts_transport_errors_and_5xx(
    failure: Outcome, db_path: Path, llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
) -> None:
    model = make_model_config()
    ledger = make_ledger()
    with pytest.raises(LLMError) as exc_info:
        _failing_client(llm_cache, ledger, failure).complete(MSGS)

    estimate = pessimistic_cost(model, MSGS)
    assert exc_info.value.maybe_billed
    assert ledger.spent == pytest.approx(estimate)
    assert _spend_rows(db_path) == [
        ("test", "fake-model", prompt_token_bound(MSGS), model.max_tokens, estimate)
    ]


@pytest.mark.parametrize("failure", [400, 401, 404, 429])
def test_failed_call_is_not_booked_for_4xx_including_429(
    failure: int, db_path: Path, llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
) -> None:
    ledger = make_ledger()
    with pytest.raises(LLMError) as exc_info:
        _failing_client(llm_cache, ledger, failure).complete(MSGS)

    assert not exc_info.value.maybe_billed
    assert ledger.spent == 0.0  # reservation released
    assert _spend_rows(db_path) == []


def test_failed_call_is_booked_if_any_attempt_may_have_been_billed(
    db_path: Path, llm_cache: LLMCache, make_ledger: MakeLedger, api_key: str
) -> None:
    ledger = make_ledger()
    with pytest.raises(LLMError, match="HTTP 400"):
        _failing_client(llm_cache, ledger, 503, 400).complete(MSGS)
    assert len(_spend_rows(db_path)) == 1


def test_missing_api_key_fails_before_charging(
    db_path: Path, llm_cache: LLMCache, make_ledger: MakeLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TEST_API_KEY_ENV, raising=False)
    ledger = make_ledger()
    client = LLMClient(make_model_config(), llm_cache, ledger, transport=FakeLLM(["x"]).transport)
    with pytest.raises(LLMError, match=TEST_API_KEY_ENV):
        client.complete(MSGS)
    assert ledger.spent == 0.0
    assert _spend_rows(db_path) == []


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
