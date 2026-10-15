"""The web app's database: SQLite via SQLAlchemy (docs/phase2_spec.md, Architecture).

Three tables:
- `checks`: one row per check: its text and mode, status (queued, running, done,
  failed), current stage, warnings or error, cost, timings, models, prompt hashes,
  and the whole report as JSON.
- `candidates`: one row per listed paper of a finished check (rank, ids, title,
  venue, snippet).
- `explanations`: one row per explained paper (overlap level, facets, evidence and
  how it was found, the explainer's own quote, "differs").

A check reused from history is a new `checks` row with the stored report, marked
`from_history_of`, and costs nothing. Times are UTC, stored without a time zone.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from rk.live.report import CheckReport, CheckSubmission

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
UNFINISHED = (QUEUED, RUNNING)
OPENALEX_BUDGET = "openalex_budget"  # checks.error_code when OpenAlex's daily budget ran out


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def history_key(submission: CheckSubmission) -> str:
    """What makes two checks the same: the text (whitespace collapsed), mode and date."""
    key = [" ".join(submission.text.split()), submission.mode, str(submission.before_date)]
    return hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()


class Base(DeclarativeBase):
    pass


class CheckRow(Base):
    __tablename__ = "checks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), index=True)
    stage: Mapped[str | None] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(8))  # full or quick
    text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(64), index=True)  # history_key
    before_date: Mapped[date | None] = mapped_column(Date)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(32))
    retry_after: Mapped[datetime | None] = mapped_column(DateTime)  # OpenAlex's reset
    from_history_of: Mapped[str | None] = mapped_column(String(32))
    report_json: Mapped[str | None] = mapped_column(Text)
    method_cost: Mapped[float | None] = mapped_column(Float)
    billed_cost: Mapped[float] = mapped_column(Float, default=0.0)  # money this row spent
    total_s: Mapped[float | None] = mapped_column(Float)
    n_candidates: Mapped[int | None] = mapped_column(Integer)
    models: Mapped[dict[str, str] | None] = mapped_column(JSON)
    prompt_hashes: Mapped[dict[str, str] | None] = mapped_column(JSON)

    def report(self) -> CheckReport | None:
        return CheckReport.model_validate_json(self.report_json) if self.report_json else None

    @property
    def first_words(self) -> str:
        words = self.text.split()
        return " ".join(words[:12]) + (" …" if len(words) > 12 else "")


class CandidateRow(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    check_id: Mapped[str] = mapped_column(ForeignKey("checks.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    paper_id: Mapped[str] = mapped_column(String(32))
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    title: Mapped[str] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(String(256))
    publication_date: Mapped[str] = mapped_column(String(10))
    venue: Mapped[str | None] = mapped_column(Text)
    snippet: Mapped[str | None] = mapped_column(Text)
    snippet_similarity: Mapped[float | None] = mapped_column(Float)


class ExplanationRow(Base):
    __tablename__ = "explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    check_id: Mapped[str] = mapped_column(ForeignKey("checks.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    paper_id: Mapped[str] = mapped_column(String(32))
    overlap: Mapped[str] = mapped_column(String(8))
    facets: Mapped[dict[str, Any]] = mapped_column(JSON)
    match: Mapped[str] = mapped_column(String(16))  # exact, matched or unverified
    evidence: Mapped[str | None] = mapped_column(Text)  # the abstract's sentence shown
    quote: Mapped[str] = mapped_column(Text)  # the explainer's own version
    differs: Mapped[str] = mapped_column(Text)


class Store:
    """The web app's checks in SQLite; every method is one short transaction."""

    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30}
        )
        Base.metadata.create_all(self.engine)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._sessions.begin() as session:
            yield session

    def close(self) -> None:
        self.engine.dispose()

    # --- writing ---------------------------------------------------------------------------

    def create(self, submission: CheckSubmission) -> CheckRow:
        row = CheckRow(
            id=uuid.uuid4().hex,
            created_at=utc_now(),
            status=QUEUED,
            stage=QUEUED,
            mode=submission.mode,
            text=submission.text,
            text_hash=history_key(submission),
            before_date=submission.before_date,
            warnings=[],
            billed_cost=0.0,
        )
        with self.session() as session:
            session.add(row)
        return row

    def from_history(self, source: CheckRow) -> CheckRow:
        """A new finished row with `source`'s report: nothing is run, nothing spent."""
        now = utc_now()
        row = CheckRow(
            id=uuid.uuid4().hex,
            created_at=now,
            started_at=now,
            finished_at=now,
            status=DONE,
            stage=DONE,
            mode=source.mode,
            text=source.text,
            text_hash=source.text_hash,
            before_date=source.before_date,
            warnings=list(source.warnings),
            from_history_of=source.from_history_of or source.id,
            report_json=source.report_json,
            method_cost=source.method_cost,
            billed_cost=0.0,
            total_s=0.0,
            n_candidates=source.n_candidates,
            models=source.models,
            prompt_hashes=source.prompt_hashes,
        )
        with self.session() as session:
            session.add(row)
        return row

    def _update(self, check_id: str, **values: Any) -> None:
        with self.session() as session:
            session.execute(update(CheckRow).where(CheckRow.id == check_id).values(**values))

    def start(self, check_id: str) -> None:
        self._update(check_id, status=RUNNING, stage=RUNNING, started_at=utc_now())

    def set_stage(self, check_id: str, stage: str) -> None:
        self._update(check_id, stage=stage[:64])

    def finish(self, check_id: str, report: CheckReport) -> None:
        """Mark a check done and store its report, papers and explanations."""
        with self.session() as session:
            session.execute(
                update(CheckRow)
                .where(CheckRow.id == check_id)
                .values(
                    status=DONE,
                    stage=DONE,
                    finished_at=utc_now(),
                    warnings=list(report.warnings),
                    report_json=report.model_dump_json(),
                    method_cost=report.cost.method_cost,
                    billed_cost=report.cost.billed_cost,
                    total_s=report.stages.total_s,
                    n_candidates=report.n_candidates,
                    models=report.models,
                    prompt_hashes=report.prompt_hashes,
                )
            )
            for paper in report.papers:
                session.add(
                    CandidateRow(
                        check_id=check_id,
                        rank=paper.rank,
                        paper_id=paper.id,
                        aliases=paper.aliases,
                        title=paper.title,
                        doi=paper.doi,
                        publication_date=paper.publication_date,
                        venue=paper.venue,
                        snippet=paper.snippet.sentence if paper.snippet else None,
                        snippet_similarity=paper.snippet.similarity if paper.snippet else None,
                    )
                )
                e = paper.explanation
                if e is not None:
                    session.add(
                        ExplanationRow(
                            check_id=check_id,
                            rank=paper.rank,
                            paper_id=paper.id,
                            overlap=e.overlap,
                            facets=e.facets.model_dump(),
                            match=e.match,
                            evidence=e.evidence,
                            quote=e.quote,
                            differs=e.differs,
                        )
                    )

    def fail(
        self,
        check_id: str,
        error: str,
        *,
        code: str | None = None,
        retry_after: datetime | None = None,
        billed_cost: float = 0.0,
    ) -> None:
        self._update(
            check_id,
            status=FAILED,
            stage=FAILED,
            finished_at=utc_now(),
            error=error,
            error_code=code,
            retry_after=retry_after,
            billed_cost=billed_cost,
        )

    def fail_unfinished(self, error: str) -> int:
        """Mark every queued or running check failed (after a restart); how many there were."""
        with self.session() as session:
            result = session.execute(
                update(CheckRow)
                .where(CheckRow.status.in_(UNFINISHED))
                .values(status=FAILED, stage=FAILED, finished_at=utc_now(), error=error)
            )
            return int(result.rowcount or 0)

    # --- reading ---------------------------------------------------------------------------

    def get(self, check_id: str) -> CheckRow | None:
        with self.session() as session:
            return session.get(CheckRow, check_id)

    def recent(self, limit: int = 20) -> list[CheckRow]:
        with self.session() as session:
            query = select(CheckRow).order_by(CheckRow.created_at.desc()).limit(limit)
            return list(session.scalars(query))

    def history(self, key: str, since: datetime) -> CheckRow | None:
        """The latest finished check with this history key created since `since`."""
        with self.session() as session:
            query = (
                select(CheckRow)
                .where(CheckRow.text_hash == key, CheckRow.status == DONE)
                .where(CheckRow.created_at >= since, CheckRow.report_json.is_not(None))
                .order_by(CheckRow.created_at.desc())
                .limit(1)
            )
            return session.scalars(query).first()

    def spent_since(self, since: datetime) -> float:
        """Money spent by the checks created since `since`."""
        with self.session() as session:
            total = session.scalar(
                select(func.coalesce(func.sum(CheckRow.billed_cost), 0.0)).where(
                    CheckRow.created_at >= since
                )
            )
            return float(total or 0.0)

    def unfinished(self) -> int:
        with self.session() as session:
            count = session.scalar(
                select(func.count()).select_from(CheckRow).where(CheckRow.status.in_(UNFINISHED))
            )
            return int(count or 0)

    def openalex_block(self, now: datetime) -> CheckRow | None:
        """The latest check that found OpenAlex's daily budget spent, if it hasn't reset yet."""
        with self.session() as session:
            query = (
                select(CheckRow)
                .where(CheckRow.error_code == OPENALEX_BUDGET, CheckRow.retry_after > now)
                .order_by(CheckRow.retry_after.desc())
                .limit(1)
            )
            return session.scalars(query).first()

    def candidates(self, check_id: str) -> list[CandidateRow]:
        with self.session() as session:
            query = select(CandidateRow).where(CandidateRow.check_id == check_id)
            return list(session.scalars(query.order_by(CandidateRow.rank)))

    def explanations(self, check_id: str) -> list[ExplanationRow]:
        with self.session() as session:
            query = select(ExplanationRow).where(ExplanationRow.check_id == check_id)
            return list(session.scalars(query.order_by(ExplanationRow.rank)))


def start_of_day(now: datetime) -> datetime:
    return datetime(now.year, now.month, now.day)


def reset_time(now: datetime, seconds: float | None) -> datetime:
    """When a budget that resets in `seconds` (else at the next midnight UTC) resets."""
    if seconds is not None:
        return now + timedelta(seconds=seconds)
    return start_of_day(now) + timedelta(days=1)
