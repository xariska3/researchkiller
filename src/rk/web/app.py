"""The web app (docs/phase2_spec.md, Architecture; Step 6): `python -m rk serve`.

API:
- POST /api/checks {text, mode: "full"|"quick", before_date?} -> {check_id, status,
  from_history}. The idea must be 40-500 words (422 with a plain message otherwise).
  The same text, mode and date within web.history_days reuses the stored report at
  $0. A check is refused (429) when it could take the day's spend over
  web.daily_cost_cap, and (503) while OpenAlex's daily budget is spent.
- GET /api/checks/{id}: status, stage, warnings, error, and the report when done.
- GET /api/checks: the recent checks.
- GET /api/checks/{id}/export?format=md|json: the report as a file.

Pages: "/" (the form, live progress, recent checks) and "/checks/{id}" (the result),
from Jinja2 templates with autoescaping; plain JavaScript polls the API.

Checks run one at a time in a background thread (rk.web.worker). On start, checks
left queued or running by a previous server are marked failed, never re-run, so a
restart can't spend money unasked.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import BaseModel

from rk.config import Config, WebConfig
from rk.live.report import (
    FACET_LABELS,
    FACETS,
    HEADER,
    MAX_WORDS,
    MIN_WORDS,
    CheckReport,
    CheckSubmission,
    render_markdown,
)
from rk.web.db import DONE, FAILED, CheckRow, Store, history_key, start_of_day, utc_now
from rk.web.worker import Redactor, ServicesFactory, Worker, api_key_names, live_services

RESTART_MESSAGE = (
    "The server stopped before this check finished. It was not re-run, so that a restart "
    "never spends money unasked: submit it again."
)


class Accepted(BaseModel):
    check_id: str
    status: str
    from_history: bool


class CheckStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "failed"]
    stage: str | None
    mode: str
    created_at: datetime
    finished_at: datetime | None
    warnings: list[str]
    error: str | None
    from_history_of: str | None
    billed_cost: float  # money this check spent ($0 from history)
    report: CheckReport | None


class CheckSummary(BaseModel):
    id: str
    first_words: str
    mode: str
    status: str
    billed_cost: float
    created_at: datetime
    from_history: bool


def _status(row: CheckRow) -> CheckStatus:
    return CheckStatus(
        id=row.id,
        status=row.status,
        stage=row.stage,
        mode=row.mode,
        created_at=row.created_at,
        finished_at=row.finished_at,
        warnings=row.warnings or [],
        error=row.error,
        from_history_of=row.from_history_of,
        billed_cost=row.billed_cost,
        report=row.report() if row.status == DONE else None,
    )


def _summary(row: CheckRow) -> CheckSummary:
    return CheckSummary(
        id=row.id,
        first_words=row.first_words,
        mode=row.mode,
        status=row.status,
        billed_cost=row.billed_cost,
        created_at=row.created_at,
        from_history=row.from_history_of is not None,
    )


def _validation_message(exc: RequestValidationError) -> str:
    """The first problem with a request body, in words ("the idea must be 40-500 ...")."""
    for error in exc.errors():
        message = str(error.get("msg", ""))
        field = ".".join(str(p) for p in error.get("loc", ()) if p != "body")
        message = message.removeprefix("Value error, ")
        return f"{field}: {message}" if field and field != "text" else message
    return "the request is not valid"


def create_app(
    cfg: Config,
    *,
    store: Store | None = None,
    services: ServicesFactory | None = None,
) -> FastAPI:
    """The app; `store` and `services` default to the config's database and live services."""
    web = cfg.web or WebConfig()
    if cfg.explain is None:
        raise ValueError("the web app needs the config's explain: section")
    per_check_cap = cfg.explain.cost_cap_per_check
    store = store or Store(web.db_path)
    redact = Redactor(api_key_names(cfg))
    worker = Worker(
        cfg, store, services or (lambda: live_services(cfg)), redact=redact
    )
    templates = Environment(
        loader=PackageLoader("rk.web", "templates"),
        autoescape=select_autoescape(default=True, default_for_string=True),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store.fail_unfinished(RESTART_MESSAGE)
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="ResearchKiller", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store, app.state.worker = store, worker

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _validation_message(exc)})

    def page(name: str, **values: Any) -> HTMLResponse:
        html = templates.get_template(name).render(
            header=HEADER,
            min_words=MIN_WORDS,
            max_words=MAX_WORDS,
            facets=FACETS,
            facet_labels=FACET_LABELS,
            **values,
        )
        return HTMLResponse(html)

    def refusal(now: datetime) -> HTTPException | None:
        """Why a new check can't start now, if it can't."""
        block = store.openalex_block(now)
        if block is not None and block.retry_after is not None:
            minutes = max(1, round((block.retry_after - now).total_seconds() / 60))
            return HTTPException(
                503,
                f"OpenAlex's daily search budget is spent. It resets at "
                f"{block.retry_after:%H:%M} UTC (in {minutes // 60}h {minutes % 60:02d}m); "
                f"new checks can start then.",
            )
        spent = store.spent_since(start_of_day(now))
        held = store.unfinished() * per_check_cap
        if spent + held + per_check_cap > web.daily_cost_cap:
            return HTTPException(
                429,
                f"The web app's daily spend cap of ${web.daily_cost_cap:.2f} is reached: "
                f"${spent:.4f} spent today (UTC) and ${held:.2f} held for checks in "
                f"progress, and a check can cost up to ${per_check_cap:.2f}. Try again "
                f"after midnight UTC, or raise web.daily_cost_cap in the config.",
            )
        return None

    @app.post("/api/checks", response_model=Accepted, status_code=202)
    def submit(submission: CheckSubmission) -> Accepted:
        now = utc_now()
        since = now - timedelta(days=web.history_days)
        previous = store.history(history_key(submission), since)
        if previous is not None:
            row = store.from_history(previous)
            return Accepted(check_id=row.id, status=row.status, from_history=True)
        problem = refusal(now)
        if problem is not None:
            raise problem
        row = store.create(submission)
        worker.submit(row.id)
        return Accepted(check_id=row.id, status=row.status, from_history=False)

    @app.get("/api/checks", response_model=list[CheckSummary])
    def recent(limit: int = Query(20, ge=1, le=200)) -> list[CheckSummary]:
        return [_summary(row) for row in store.recent(limit)]

    def found(check_id: str) -> CheckRow:
        row = store.get(check_id)
        if row is None:
            raise HTTPException(404, f"no check {check_id}")
        return row

    @app.get("/api/checks/{check_id}", response_model=CheckStatus)
    def status(check_id: str) -> CheckStatus:
        return _status(found(check_id))

    @app.get("/api/checks/{check_id}/export")
    def export(check_id: str, format: Literal["md", "json"] = "md") -> Response:
        row = found(check_id)
        report = row.report()
        if row.status != DONE or report is None:
            raise HTTPException(409, f"check {check_id} has no report (status {row.status})")
        name = f"researchkiller-check-{check_id[:8]}.{format}"
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        if format == "json":
            body, media_type = report.model_dump_json(indent=2), "application/json"
        else:
            body, media_type = render_markdown(report), "text/markdown; charset=utf-8"
        return Response(content=body, media_type=media_type, headers=headers)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return page("index.html", recent=store.recent(20))

    @app.get("/checks/{check_id}", response_class=HTMLResponse)
    def result(check_id: str) -> HTMLResponse:
        row = found(check_id)
        report = row.report() if row.status == DONE else None
        shown = report.shown() if report else []
        return page(
            "check.html",
            row=row,
            report=report,
            shown=shown,
            rest=[p for p in report.papers if p not in shown] if report else [],
            source=store.get(row.from_history_of) if row.from_history_of else None,
            failed=row.status == FAILED,
        )

    return app
