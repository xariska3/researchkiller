"""The web app's worker: one background thread running queued checks, one at a time.

Each check runs through rk.live.check.run_check, the function `python -m rk check`
calls, so the web app has no pipeline of its own. The worker thread builds the
services it needs (LLM clients, OpenAlex search, the encoder) itself, through a
factory: their SQLite connections belong to the thread that opened them. `ready` is
set once they are built, so the app can load the encoder before taking checks.

A check's stage is written to the database as it runs. Errors become a short
message on the check, with every configured API key's value removed (`Redactor`).
OpenAlex's spent daily budget is recorded with its reset time, so the app can refuse
new checks until then.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager
from dataclasses import dataclass

from rk.config import Config
from rk.corpus import OpenAlexBudgetExhausted, OpenAlexError
from rk.live.check import run_check
from rk.live.openalex_search import LiveSearchClient, OpenAlexCache
from rk.live.pipeline import CheckCostCapExceeded
from rk.live.report import CheckReport, CheckRequest
from rk.llm import BudgetExceeded, ChatModel, Ledger, LLMCache, LLMClient, LLMError
from rk.retrieval import Encoder, make_encoder
from rk.web.db import OPENALEX_BUDGET, Store, reset_time, utc_now

log = logging.getLogger("rk.web")


@dataclass
class Services:
    """What a check needs; built in the worker thread."""

    search: LiveSearchClient
    encoder: Encoder
    agent: ChatModel | None = None  # None: quick checks only
    reranker: ChatModel | None = None
    explainer: ChatModel | None = None
    ledger: Ledger | None = None  # the cumulative LLM ledger, to count a failed check's spend


ServicesFactory = Callable[[], AbstractContextManager[Services]]


@contextmanager
def live_services(cfg: Config) -> Iterator[Services]:
    """The real services: cached LLM clients on the shared ledger, OpenAlex, the encoder."""
    if cfg.live is None or cfg.rerank is None or cfg.explain is None:
        raise ValueError("the web app needs the config's live:, rerank: and explain: sections")
    with ExitStack() as stack:
        cache = stack.enter_context(LLMCache(cfg.run.cache_path))
        ledger = stack.enter_context(Ledger(cfg.run.cache_path, cfg.run.budget_cap, command="web"))
        agent = stack.enter_context(closing(LLMClient(cfg.agent, cache, ledger)))
        reranker = stack.enter_context(closing(LLMClient(cfg.rerank.model, cache, ledger)))
        explainer = stack.enter_context(closing(LLMClient(cfg.explain.model, cache, ledger)))
        responses = stack.enter_context(closing(OpenAlexCache(cfg.live.cache_path)))
        search = stack.enter_context(closing(LiveSearchClient.from_config(cfg, responses)))
        yield Services(search, make_encoder(cfg.retrieval), agent, reranker, explainer, ledger)


def api_key_names(cfg: Config) -> list[str]:
    """Every environment variable the config names as holding an API key."""
    names = {m.api_key_env for m in (cfg.paraphraser, cfg.judge, cfg.agent)}
    if cfg.rerank is not None:
        names |= {cfg.rerank.model.api_key_env, cfg.rerank.upper_model.api_key_env}
    if cfg.explain is not None:
        names.add(cfg.explain.model.api_key_env)
    if cfg.corpus.api_key_env:
        names.add(cfg.corpus.api_key_env)
    if cfg.compete is not None:
        names |= {s.api_key_env for s in cfg.compete.systems if s.api_key_env}
    return sorted(names)


class Redactor:
    """Removes the values of the given environment variables (API keys) from text."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)

    def __call__(self, text: str) -> str:
        for name in self.names:
            value = os.environ.get(name)
            if value and len(value) >= 8:
                text = text.replace(value, "[redacted]")
        return text


def _duration(seconds: float) -> str:
    minutes = max(0, round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m"


class Worker:
    """One thread that runs the queued checks in order."""

    def __init__(
        self,
        cfg: Config,
        store: Store,
        services: ServicesFactory,
        *,
        redact: Callable[[str], str] = lambda text: text,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.services = services
        self.redact = redact
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="rk-web-worker", daemon=True)
        self.startup_error: str | None = None
        self.ready = threading.Event()  # set once the services are built, or failed to

    def start(self) -> None:
        self.thread.start()

    def submit(self, check_id: str) -> None:
        self.queue.put(check_id)

    def stop(self, timeout: float = 30.0) -> None:
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join(timeout)

    def _run(self) -> None:
        with ExitStack() as stack:
            try:
                services = stack.enter_context(self.services())
            except Exception as exc:  # the services couldn't start: fail what comes
                self.startup_error = self.redact(f"{type(exc).__name__}: {exc}")
                log.exception("the check worker could not start: %s", self.startup_error)
                self.ready.set()
                while (check_id := self.queue.get()) is not None:
                    self.store.fail(
                        check_id, f"The server couldn't start its checks: {self.startup_error}"
                    )
                return
            self.ready.set()
            while (check_id := self.queue.get()) is not None:
                try:
                    self.run_one(services, check_id)
                except Exception:  # e.g. the database: keep the worker for the next check
                    log.exception("check %s: the worker failed", check_id)

    def run_one(self, services: Services, check_id: str) -> None:
        """Run one queued check and store its report, or a clear error."""
        row = self.store.get(check_id)
        if row is None:
            return
        self.store.start(check_id)
        spent_before = services.ledger.spent if services.ledger is not None else 0.0
        request = CheckRequest(text=row.text, before=row.before_date, quick=row.mode == "quick")

        def spent() -> float:
            return services.ledger.spent - spent_before if services.ledger is not None else 0.0

        try:
            run = run_check(
                request,
                self.cfg,
                search=services.search,
                encoder=services.encoder,
                agent=services.agent,
                reranker=services.reranker,
                explainer=services.explainer,
                progress=lambda stage: self.store.set_stage(check_id, stage),
            )
        except OpenAlexBudgetExhausted as exc:
            retry = reset_time(utc_now(), exc.resets_in_s)
            wait = _duration((retry - utc_now()).total_seconds())
            self.store.fail(
                check_id,
                f"OpenAlex's daily search budget is spent, so the check couldn't search. It "
                f"resets at {retry:%H:%M} UTC (in {wait}); new checks can start then.",
                code=OPENALEX_BUDGET,
                retry_after=retry,
                billed_cost=spent(),
            )
        except CheckCostCapExceeded as exc:
            self.store.fail(
                check_id,
                self.redact(f"The check stopped at its cost cap: {exc}"),
                billed_cost=spent(),
            )
        except BudgetExceeded as exc:
            self.store.fail(
                check_id,
                self.redact(f"The project's total LLM spending cap (run.budget_cap) is reached: "
                            f"{exc}"),
                billed_cost=spent(),
            )
        except (OpenAlexError, LLMError) as exc:
            self.store.fail(
                check_id,
                self.redact(f"The check failed: {exc}"),
                billed_cost=spent(),
            )
        except Exception as exc:  # a bug: keep the server up, say what happened
            message = self.redact(f"{type(exc).__name__}: {exc}")
            log.exception("check %s failed: %s", check_id, message)
            self.store.fail(check_id, f"Internal error: {message}", billed_cost=spent())
        else:
            self.store.finish(check_id, redacted(run.report, self.redact))


def redacted(report: CheckReport, redact: Callable[[str], str]) -> CheckReport:
    """The report with API keys removed from its warnings (they quote error messages)."""
    return report.model_copy(update={"warnings": [redact(w) for w in report.warnings]})
