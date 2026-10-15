"""The report of one check (docs/phase2_spec.md, Pipeline 7; Step 5): Pydantic models.

CheckRequest is what a user sends (the API's POST body); CheckReport is everything a
check returns, as `python -m rk check` saves it and as the API will serve it. The
header is always HEADER: a check lists the closest work it found, and finding nothing
close doesn't prove an idea new. `render_text` prints a report for the terminal.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

HEADER = "Closest prior work found; no close match is not proof of novelty."
MIN_WORDS, MAX_WORDS = 40, 500  # the idea's length (docs/phase2_spec.md, Pipeline 1)
DEFAULT, QUICK = "default", "quick"  # the check's mode
HIGH, PARTIAL, LOW = "high", "partial", "low"
OVERLAPS = (HIGH, PARTIAL, LOW)  # closest first
SAME, DIFFERENT = "same", "different"
FACETS = ("problem", "method", "domain", "claim")
FACET_LABELS = {"problem": "Problem", "method": "Method", "domain": "Domain", "claim": "Claim"}
# How the explainer's quote was found in the abstract (Explanation.match, rk.live.explain).
EXACT = "exact"  # verbatim in one of its sentences
MATCHED = "matched"  # one of its sentences has at least 90% of the quote's words
NO_MATCH = "unverified"  # neither: no evidence shown
# What the report shows for a paper (Paper.status).
VERIFIED_EXACT = "verified-exact"  # an explanation; evidence: the sentence the quote is in
VERIFIED_MATCHED = "verified-matched"  # an explanation; evidence: the sentence it matched
UNVERIFIED = "unverified"  # an explanation without evidence: the snippet is shown
NOT_EXPLAINED = "not explained"  # the snippet only


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def idea_text(text: str) -> str:
    """The idea without surrounding whitespace, if it is MIN_WORDS-MAX_WORDS words long."""
    n = len(text.split())
    if not MIN_WORDS <= n <= MAX_WORDS:
        raise ValueError(f"the idea must be {MIN_WORDS}-{MAX_WORDS} words long; it has {n}")
    return text.strip()


class CheckRequest(_Model):
    """One idea to check: 40-500 words, optionally only against work published before a date."""

    text: str
    before: date | None = None
    quick: bool = False  # L1: one OpenAlex search and the local rerank, no LLM

    @field_validator("text")
    @classmethod
    def _length(cls, text: str) -> str:
        return idea_text(text)


class CheckSubmission(_Model):
    """The web API's POST body (rk.web): the idea, the mode and an optional date."""

    text: str
    mode: Literal["full", "quick"] = "full"  # full: the default check; quick: L1, no LLM
    before_date: date | None = None  # only works published before this date

    @field_validator("text")
    @classmethod
    def _length(cls, text: str) -> str:
        return idea_text(text)

    def request(self) -> CheckRequest:
        return CheckRequest(text=self.text, before=self.before_date, quick=self.mode == "quick")


class FacetVerdict(_Model):
    verdict: Literal["same", "different"]
    reason: str


class Facets(_Model):
    problem: FacetVerdict
    method: FacetVerdict
    domain: FacetVerdict
    claim: FacetVerdict


class Explanation(_Model):
    """How one paper overlaps the idea, from the explainer call.

    The evidence shown is always the abstract's own sentence, never the explainer's
    version of it (`quote`, kept for the record).
    """

    overlap: Literal["high", "partial", "low"]
    facets: Facets
    match: Literal["exact", "matched", "unverified"]  # how the quote was found
    evidence: str | None  # the abstract's sentence the quote was found in; None if none
    quote: str  # the sentence the explainer gave as evidence, as it gave it
    differs: str  # one line on what the paper does differently

    @property
    def verified(self) -> bool:
        return self.match != NO_MATCH


class Snippet(_Model):
    """The abstract sentence closest to the idea (Qwen3-Embedding cosine)."""

    sentence: str
    similarity: float


class Paper(_Model):
    rank: int  # from 1, in the check's final order
    id: str  # OpenAlex work id
    aliases: list[str]  # ids of versions merged into it
    url: str
    doi: str | None
    title: str
    publication_date: str
    type: str | None
    authors: list[str]
    venue: str | None = None  # the journal or conference (OpenAlex primary_location.source)
    snippet: Snippet | None = None  # the top live.explain_top_k only
    explanation: Explanation | None = None  # the top live.explain_top_k, default check only

    @property
    def status(self) -> str:
        if self.explanation is None:
            return NOT_EXPLAINED
        match = self.explanation.match
        return {EXACT: VERIFIED_EXACT, MATCHED: VERIFIED_MATCHED}.get(match, UNVERIFIED)

    @property
    def link(self) -> str:
        """The DOI's address if it has one, else its OpenAlex page."""
        return f"https://doi.org/{self.doi}" if self.doi else self.url

    @property
    def year(self) -> str:
        return self.publication_date[:4]

    def evidence(self) -> tuple[str, str] | None:
        """The text to show as evidence and its label, or None if there is nothing."""
        e = self.explanation
        if e is not None and e.evidence is not None:
            label = "exact quote" if e.match == EXACT else "matched to the abstract"
            return e.evidence, label
        if self.snippet is not None:
            label = "unverified quote: snippet shown" if e is not None else "snippet"
            return self.snippet.sentence, label
        return None


class VocabularyGroup(_Model):
    source: Literal["keywords", "titles", "queries"]
    label: str
    terms: list[str]


class Cost(_Model):
    """The check's cost (USD): method cost counts cache hits too; billed is money spent."""

    method_cost: float
    billed_cost: float
    openalex_cost: float
    llm_method_cost: float
    cap: float


class Stages(_Model):
    """Seconds per stage; search includes the agent loop, the rerank its LLM call."""

    search_s: float
    rerank_s: float
    snippets_s: float
    keywords_s: float
    explain_s: float
    vocabulary_s: float
    total_s: float


class CheckReport(_Model):
    header: Literal["Closest prior work found; no close match is not proof of novelty."] = HEADER
    coverage: str  # how much was searched
    mode: Literal["default", "quick"]
    request: CheckRequest
    papers: list[Paper]  # the top live.top_k, in rank order
    vocabulary: list[VocabularyGroup]
    warnings: list[str]  # for the user: a step that fell back, a lookup that failed
    queries: list[str]  # the agent's own queries (not the idea)
    n_candidates: int  # works in the final pool
    n_searches: int
    rerank: str | None  # the rerank step's status (rk.live.pipeline), None in quick mode
    explainer: str | None  # the explainer's status (rk.live.explain), None in quick mode
    cost: Cost
    stages: Stages
    models: dict[str, str]
    prompt_hashes: dict[str, str]
    encoder: str
    encoder_device: str
    created_at: datetime

    def shown(self) -> list[Paper]:
        """The explained part of the list, as the report shows it: by overlap, then rank."""
        top = [p for p in self.papers if p.snippet is not None or p.explanation is not None]

        def key(p: Paper) -> tuple[int, int]:
            level = OVERLAPS.index(p.explanation.overlap) if p.explanation else len(OVERLAPS)
            return level, p.rank

        return sorted(top, key=key)


def _year(paper: Paper) -> str:
    return paper.publication_date[:4]


def render_text(report: CheckReport) -> str:
    """The report for a terminal: the explained papers by overlap, then the rest by rank."""
    lines = [report.header, report.coverage]
    mode = "Quick check (no LLM)" if report.mode == QUICK else "Default check"
    lines.append(
        f"{mode}; cost ${report.cost.method_cost:.4f} (spent ${report.cost.billed_cost:.4f}); "
        f"{report.stages.total_s:.1f} s."
    )
    for warning in report.warnings:
        lines.append(f"Warning: {warning}")
    shown = report.shown()
    for paper in shown:
        lines.append("")
        explanation = paper.explanation
        level = f"[{explanation.overlap} overlap] " if explanation else ""
        lines.append(f"#{paper.rank} {level}{paper.title} ({_year(paper)})")
        lines.append(f"   {paper.url}" + (f"  doi:{paper.doi}" if paper.doi else ""))
        if explanation is not None and explanation.evidence is not None:
            label = "Evidence" if explanation.match == EXACT else "Evidence (matched)"
            lines.append(f'   {label}: "{explanation.evidence}"')
        elif paper.snippet is not None:
            label = "Snippet, quote unverified" if explanation is not None else "Snippet"
            lines.append(f'   {label}: "{paper.snippet.sentence}"')
        if explanation is not None:
            for facet in FACETS:
                verdict = getattr(explanation.facets, facet)
                lines.append(f"   {FACET_LABELS[facet]}: {verdict.verdict}. {verdict.reason}")
            lines.append(f"   Differs: {explanation.differs}")
    rest = [p for p in report.papers if p not in shown]
    if rest:
        lines += ["", f"Also found (ranks {rest[0].rank}-{rest[-1].rank}):"]
        lines += [f"#{p.rank} {p.title} ({_year(p)})" for p in rest]
    if report.vocabulary:
        lines += ["", "Words the literature uses that your text doesn't:"]
        lines += [f"   {g.label}: {', '.join(g.terms)}" for g in report.vocabulary]
    return "\n".join(lines) + "\n"


def render_markdown(report: CheckReport) -> str:
    """The report as Markdown (the web app's export): the same order as the result page."""
    mode = "Quick check (no LLM)" if report.mode == QUICK else "Full check"
    lines = [
        f"# {report.header}",
        "",
        report.coverage,
        "",
        f"**Idea** ({mode}"
        + (f", published before {report.request.before}" if report.request.before else "")
        + f"): {report.request.text}",
    ]
    if report.warnings:
        lines += ["", "**Warnings**", ""] + [f"- {w}" for w in report.warnings]
    shown = report.shown()
    for paper in shown:
        e = paper.explanation
        level = f" ({e.overlap} overlap)" if e else ""
        venue = f", {paper.venue}" if paper.venue else ""
        lines += ["", f"## {paper.rank}. [{paper.title}]({paper.link}){level}", ""]
        lines.append(f"{paper.year}{venue}")
        evidence = paper.evidence()
        if evidence is not None:
            lines += ["", f"> {evidence[0]}", "", f"*{evidence[1]}*"]
        if e is not None:
            lines.append("")
            for facet in FACETS:
                verdict = getattr(e.facets, facet)
                lines.append(f"- **{FACET_LABELS[facet]}**: {verdict.verdict}. {verdict.reason}")
            lines += ["", f"**Differs**: {e.differs}"]
    rest = [p for p in report.papers if p not in shown]
    if rest:
        lines += ["", "## Also found", ""]
        lines += [f"{p.rank}. [{p.title}]({p.link}) ({p.year})" for p in rest]
    if report.vocabulary:
        lines += ["", "## Words the literature uses that your text doesn't", ""]
        lines += [f"- **{g.label}**: {', '.join(g.terms)}" for g in report.vocabulary]
    models = ", ".join(f"{role} {name}" for role, name in report.models.items())
    lines += [
        "",
        "---",
        "",
        (
            f"Cost ${report.cost.method_cost:.4f} (spent ${report.cost.billed_cost:.4f}); "
            f"{report.stages.total_s:.1f} s; {report.n_candidates} candidates from "
            f"{report.n_searches} searches; models: {models}; "
            f"{report.created_at.strftime('%Y-%m-%d %H:%M')} UTC."
        ),
    ]
    return "\n".join(lines) + "\n"
