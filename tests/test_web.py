"""The web app (rk.web): API, pages, worker and limits, with TestClient and the fakes.

Checks run in the app's real worker thread on FakeLiveOpenAlex, test_live_eval's
FakeLLM agent, test_rerank's FakeLLM reranker, test_explain's FakeLLM explainer and
FakeEncoder; the tests poll the API until a check finishes.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from bench_data import GOOD, T1, T2, TITLES
from conftest import TEST_API_KEY_ENV, FakeEncoder, FakeLLM
from fastapi.testclient import TestClient
from live_fakes import FakeLiveOpenAlex
from sqlalchemy import update
from test_explain import VENUE_OF_T1, config_dict, explainer_reply, works
from test_live_eval import AGENT_QUERY, _search
from test_live_eval import agent as fake_agent
from test_rerank import original_first

from rk import cli
from rk.config import Config, config_from_dict
from rk.live.report import HEADER, CheckReport, CheckSubmission
from rk.llm import LLMError
from rk.web.app import RESTART_MESSAGE, QuietPolls, create_app
from rk.web.db import DONE, FAILED, CheckRow, Store, utc_now
from rk.web.worker import Services


def web_cfg(tmp_path: Path, **web: Any) -> Config:
    data = config_dict(tmp_path)
    data["web"] = {"db_path": str(tmp_path / "web.sqlite"), **web}
    return config_from_dict(data)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return web_cfg(tmp_path)


@dataclass
class Fakes:
    """The services a check runs on, as fakes the tests can inspect."""

    openalex: FakeLiveOpenAlex = field(default_factory=lambda: FakeLiveOpenAlex(works()))
    agent: FakeLLM = field(default_factory=fake_agent)
    reranker: FakeLLM = field(default_factory=lambda: FakeLLM(original_first))
    explainer: FakeLLM = field(default_factory=lambda: FakeLLM(explainer_reply))

    def llm_calls(self) -> int:
        return len(self.agent.calls) + len(self.reranker.calls) + len(self.explainer.calls)


def services(cfg: Config, fakes: Fakes) -> Any:
    @contextmanager
    def factory() -> Iterator[Services]:
        yield Services(
            search=_search(cfg, fakes.openalex),
            encoder=FakeEncoder(),
            agent=fakes.agent,
            reranker=fakes.reranker,
            explainer=fakes.explainer,
        )

    return factory


@contextmanager
def serving(cfg: Config, fakes: Fakes | None = None) -> Iterator[TestClient]:
    app = create_app(cfg, services=services(cfg, fakes or Fakes()))
    with TestClient(app) as client:
        yield client


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


@pytest.fixture
def client(cfg: Config, fakes: Fakes) -> Iterator[TestClient]:
    with serving(cfg, fakes) as test_client:
        yield test_client


def submit(client: TestClient, text: str = GOOD[T1], **body: Any) -> httpx.Response:
    return client.post("/api/checks", json={"text": text, **body})


def wait(client: TestClient, check_id: str, timeout: float = 30.0) -> dict[str, Any]:
    """Poll the status until the check is done or failed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/checks/{check_id}").json()
        if status["status"] in (DONE, FAILED):
            return status
        time.sleep(0.02)
    raise AssertionError(f"check {check_id} didn't finish in {timeout} s")


def check(client: TestClient, text: str = GOOD[T1], **body: Any) -> dict[str, Any]:
    response = submit(client, text, **body)
    assert response.status_code == 202, response.text
    return wait(client, response.json()["check_id"])


# --- a check through the API and the pages --------------------------------------------------


def test_a_full_check_is_submitted_polled_and_shown(client: TestClient, cfg: Config) -> None:
    response = submit(client)
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "queued" and accepted["from_history"] is False
    status = wait(client, accepted["check_id"])
    assert status["status"] == DONE and status["stage"] == DONE and status["error"] is None
    report = CheckReport.model_validate(status["report"])
    assert report.header == HEADER and report.mode == "default"
    assert report.papers[0].id == T1 and report.papers[0].venue == VENUE_OF_T1
    assert status["billed_cost"] == pytest.approx(report.cost.billed_cost)
    assert status["billed_cost"] > 0
    # Stored in the three tables.
    assert cfg.web is not None
    store = Store(cfg.web.db_path)
    candidates = store.candidates(accepted["check_id"])
    explanations = store.explanations(accepted["check_id"])
    assert [c.paper_id for c in candidates] == [p.id for p in report.papers]
    assert [e.rank for e in explanations] == [1, 2, 3]  # the test config explains 3
    assert explanations[0].overlap == "high" and explanations[0].match == "exact"
    # The result page.
    page = client.get(f"/checks/{accepted['check_id']}")
    assert page.status_code == 200
    html = page.text
    assert "<h1>Closest prior work found</h1>" in html and report.coverage in html
    assert "No close match is not proof of novelty." in html
    assert TITLES[T1] in html and VENUE_OF_T1 in html and "high overlap" in html
    assert "exact quote" in html and "<b>Differs:</b>" in html
    assert re.search(r"<b>Problem</b>: <span class=\"verdict \w+\">\w+</span>\.\s+\w", html)
    assert "—" not in html  # a facet's verdict and reason are two sentences
    assert "Words the literature uses that your text doesn't" in html
    assert f'/api/checks/{accepted["check_id"]}/export?format=md' in html
    assert "Candidates searched:" in html and cfg.agent.name in html
    # The recent checks, in the API and on the front page.
    (recent,) = client.get("/api/checks").json()
    assert recent["id"] == accepted["check_id"] and recent["status"] == DONE
    assert GOOD[T1].startswith(recent["first_words"].removesuffix(" …"))
    front = client.get("/").text
    assert f'href="/checks/{accepted["check_id"]}"' in front and "Recent checks" in front
    assert "It never declares an idea novel: a missing match is not proof of novelty." in front


def test_a_quick_check_makes_no_llm_call(client: TestClient, fakes: Fakes) -> None:
    status = check(client, mode="quick")
    assert status["status"] == DONE and fakes.llm_calls() == 0
    report = CheckReport.model_validate(status["report"])
    assert report.mode == "quick" and all(p.explanation is None for p in report.papers)
    assert report.papers[0].snippet is not None


def test_the_idea_length_is_checked_with_a_plain_message(client: TestClient) -> None:
    short = submit(client, "Too short to check.")
    assert short.status_code == 422
    assert short.json() == {"detail": "the idea must be 40-500 words long; it has 4"}
    long = submit(client, "word " * 501)
    assert long.json()["detail"] == "the idea must be 40-500 words long; it has 501"
    mode = submit(client, mode="slow")
    assert mode.status_code == 422 and mode.json()["detail"].startswith("mode: ")
    assert client.get("/api/checks").json() == []


def test_a_report_exports_as_markdown_and_json(client: TestClient) -> None:
    check_id = check(client)["id"]
    md = client.get(f"/api/checks/{check_id}/export", params={"format": "md"})
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown")
    assert md.headers["content-disposition"].endswith('.md"')
    assert md.text.startswith(f"# {HEADER}") and TITLES[T1] in md.text
    assert "(high overlap)" in md.text and "**Differs**" in md.text
    assert re.search(r"^- \*\*Problem\*\*: (same|different)\. \w", md.text, re.MULTILINE)
    assert "—" not in md.text
    js = client.get(f"/api/checks/{check_id}/export", params={"format": "json"})
    assert js.headers["content-type"] == "application/json"
    report = CheckReport.model_validate_json(js.text)
    assert report.papers[0].id == T1
    assert client.get(f"/api/checks/{check_id}/export?format=pdf").status_code == 422
    assert client.get("/api/checks/nope/export").status_code == 404
    assert client.get("/api/checks/nope").status_code == 404


def test_the_same_check_within_7_days_is_reused_at_no_cost(
    client: TestClient, fakes: Fakes, cfg: Config
) -> None:
    first = check(client)
    calls = fakes.llm_calls()
    again = submit(client, "  " + GOOD[T1].replace(" ", "  ") + "\n")  # same, spaced out
    assert again.status_code == 202
    accepted = again.json()
    assert accepted["from_history"] is True and accepted["status"] == DONE
    reused = client.get(f"/api/checks/{accepted['check_id']}").json()
    assert reused["from_history_of"] == first["id"] and reused["billed_cost"] == 0
    assert reused["report"] == first["report"] and fakes.llm_calls() == calls
    assert "From history: the same check ran on" in client.get(
        f"/checks/{accepted['check_id']}"
    ).text
    # Another mode, or a date, is another check.
    assert submit(client, mode="quick").json()["from_history"] is False
    dated = submit(client, before_date="2025-06-01").json()
    assert dated["from_history"] is False
    wait(client, dated["check_id"])
    # Older than 7 days: run again.
    assert cfg.web is not None
    with Store(cfg.web.db_path).session() as session:
        session.execute(update(CheckRow).values(created_at=utc_now() - timedelta(days=8)))
    assert submit(client).json()["from_history"] is False


def test_a_check_that_could_cross_the_daily_cap_is_refused(tmp_path: Path) -> None:
    cfg = web_cfg(tmp_path, daily_cost_cap=0.05)
    assert cfg.web is not None
    store = Store(cfg.web.db_path)
    spent = store.create(CheckSubmission(text=GOOD[T2]))
    store.fail(spent.id, "it spent before it failed", billed_cost=0.03)
    fakes = Fakes()
    with serving(cfg, fakes) as client:
        refused = submit(client)
        assert refused.status_code == 429
        detail = refused.json()["detail"]
        assert detail.startswith("The web app's daily spend cap of $0.05 is reached: $0.0300")
        assert "raise web.daily_cost_cap" in detail
        # Yesterday's spending doesn't count.
        with store.session() as session:
            session.execute(update(CheckRow).values(created_at=utc_now() - timedelta(days=1)))
        assert submit(client).status_code == 202
    assert fakes.llm_calls() > 0


def test_a_restart_fails_unfinished_checks_and_never_reruns_them(
    cfg: Config, fakes: Fakes
) -> None:
    assert cfg.web is not None
    store = Store(cfg.web.db_path)
    queued = store.create(CheckSubmission(text=GOOD[T1]))
    running = store.create(CheckSubmission(text=GOOD[T2]))
    store.start(running.id)
    with serving(cfg, fakes) as client:
        for row in (queued, running):
            status = client.get(f"/api/checks/{row.id}").json()
            assert status["status"] == FAILED and status["error"] == RESTART_MESSAGE
        assert "It was not re-run" in client.get(f"/checks/{queued.id}").text
    assert fakes.llm_calls() == 0 and fakes.openalex.requests == []


def test_user_text_is_escaped_in_the_pages(client: TestClient) -> None:
    text = "<script>alert('pwned')</script> " + GOOD[T1]
    check_id = check(client, text)["id"]
    for html in (client.get(f"/checks/{check_id}").text, client.get("/").text):
        assert "<script>alert(" not in html
        assert "&lt;script&gt;alert(&#39;pwned&#39;)&lt;/script&gt;" in html


def test_a_spent_openalex_budget_fails_the_check_and_refuses_new_ones(
    cfg: Config,
) -> None:
    fakes = Fakes(openalex=FakeLiveOpenAlex(works(), budget_credits=0))
    with serving(cfg, fakes) as client:
        status = check(client)
        assert status["status"] == FAILED
        assert status["error"].startswith("OpenAlex's daily search budget is spent")
        assert "new checks can start then" in status["error"]
        refused = submit(client, GOOD[T2])
        assert refused.status_code == 503
        assert refused.json()["detail"].startswith("OpenAlex's daily search budget is spent")
    assert fakes.llm_calls() == 0


def test_api_keys_never_reach_responses_or_pages(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-or-v1-0123456789abcdef"
    monkeypatch.setenv(TEST_API_KEY_ENV, secret)

    def leaky(messages: list[dict[str, str]]) -> str:
        raise LLMError(f"HTTP 401 for key {secret}")

    fakes = Fakes(explainer=FakeLLM(leaky))
    with serving(cfg, fakes) as client:
        status = check(client)
        assert status["status"] == DONE
        assert any("[redacted]" in w for w in status["warnings"])
        pages = [
            json.dumps(status),
            client.get(f"/checks/{status['id']}").text,
            client.get(f"/api/checks/{status['id']}/export?format=md").text,
            client.get("/api/checks").text,
        ]
        assert all(secret not in page for page in pages)


def test_a_failed_agent_search_is_skipped_with_a_warning(cfg: Config) -> None:
    class AgentQueryTimesOut(FakeLiveOpenAlex):
        def handle(self, request: httpx.Request) -> httpx.Response:
            params = dict(request.url.params)
            if AGENT_QUERY in (params.get("search"), params.get("search.semantic")):
                return httpx.Response(504, json={"error": "query_timeout"})
            return super().handle(request)

    fakes = Fakes(openalex=AgentQueryTimesOut(works()))
    with serving(cfg, fakes) as client:
        status = check(client)
    assert status["status"] == DONE
    assert any(w.startswith("An agent search failed and was skipped") for w in status["warnings"])


def test_the_server_loads_the_encoder_before_taking_checks(cfg: Config, fakes: Fakes) -> None:
    loaded = threading.Event()
    build = services(cfg, fakes)

    @contextmanager
    def slow() -> Iterator[Services]:
        time.sleep(0.3)  # loading the encoder
        with build() as built:
            loaded.set()
            yield built

    with TestClient(create_app(cfg, services=slow)) as client:
        assert loaded.is_set()  # the start waited for it
        assert check(client)["status"] == DONE


def test_status_polls_are_left_out_of_the_access_log(client: TestClient) -> None:
    def line(method: str, path: str, status: int) -> logging.LogRecord:
        args = ("127.0.0.1:50000", method, path, "1.1", status)
        return logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 0, '%s - "%s %s HTTP/%s" %d', args, None
        )

    quiet = QuietPolls()
    assert not quiet.filter(line("GET", "/api/checks/0a1b2c", 200))
    assert quiet.filter(line("POST", "/api/checks", 202))
    assert quiet.filter(line("GET", "/api/checks", 200))
    assert quiet.filter(line("GET", "/checks/0a1b2c", 200))
    assert quiet.filter(line("GET", "/api/checks/0a1b2c/export?format=md", 200))
    assert quiet.filter(line("GET", "/api/checks/nope", 404))
    filters = logging.getLogger("uvicorn.access").filters
    assert [type(f) for f in filters].count(QuietPolls) == 1  # installed once by the start


# --- the command ----------------------------------------------------------------------------


def test_serve_binds_to_localhost_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn
    import yaml

    config_path = tmp_path / "config.yaml"
    data = config_dict(tmp_path)
    data["web"] = {"db_path": str(tmp_path / "web.sqlite")}
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    assert cli.main(["serve", "--config", str(config_path)]) == 0
    assert calls == [{"host": "127.0.0.1", "port": 8000, "log_level": "info"}]
    assert "rk serve: http://127.0.0.1:8000" in capsys.readouterr().out
    argv = ["serve", "--config", str(config_path), "--host", "0.0.0.0", "--port", "9"]
    assert cli.main(argv) == 0
    assert calls[-1]["host"] == "0.0.0.0" and calls[-1]["port"] == 9
