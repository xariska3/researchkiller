"""One whole check, as the tool runs it (docs/phase2_spec.md, Pipeline; Step 5).

`run_check` runs the default check (L5r, rk.live.pipeline.default_check) or the quick
check (L1, rk.live.pipeline.quick_check, no LLM), and merges versions of one paper in
its final list (rk.live.pipeline.merge_same_text). Then, for the top
live.explain_top_k papers:
1. a snippet each (rk.live.explain.snippets);
2. their OpenAlex keywords, for the vocabulary map (one list call);
3. the explainer call (default check only; rk.live.explain.explain);
4. the vocabulary map (rk.live.vocab).

The whole check runs under explain.cost_cap_per_check. Nothing after the retrieval can
fail it: a keyword lookup or an explainer call that fails or could cross the cap leaves
the papers as they are, with a warning. With `strict` (evaluation), a failed explainer
call raises instead.

A two-list check (rerank.two_lists, issue #22) explains, in the same call, the top
explain.contribution_top_k of list A (section A) and section B: the first
explain.methods_top_k papers of list B that aren't in section A (`methods_section`).
The section A papers that list B ranks among section B's are named on its "Also close
in methods" line instead of being shown twice.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from rk.config import Config
from rk.corpus import LIST_CALL_USD, OpenAlexError
from rk.live.explain import (
    EXPLAIN_PROMPT,
    EXPLAIN_SCHEMA_SHA256,
    ExplainStep,
    explain,
    snippets,
)
from rk.live.openalex_search import LiveSearchClient
from rk.live.pipeline import (
    Candidate,
    CheckCostCapExceeded,
    LiveSearchResult,
    default_check,
    merge_same_text,
    quick_check,
)
from rk.live.report import (
    DEFAULT,
    QUICK,
    CheckReport,
    CheckRequest,
    Cost,
    Explanation,
    IdeaSummary,
    Paper,
    Snippet,
    Stages,
)
from rk.live.reranker import LISTWISE, TWO_LISTS, TWO_LISTS_SCHEMA_SHA256
from rk.live.vocab import vocabulary

EXPLAINING = "explaining"  # the stage `progress` is told before the explainer call
from rk.llm import ChatModel, load_prompt
from rk.retrieval import Encoder
from rk.targets import Record

REPORT_FILE = "report.json"
RUN_PREFIX = "check-"


def save_report(report: CheckReport, runs_dir: Path) -> Path:
    """Write the report to runs_dir/check-<timestamp>/report.json (a new directory)."""
    stamp = RUN_PREFIX + report.created_at.strftime("%Y%m%d-%H%M%S")
    name, n = stamp, 1
    while (runs_dir / name).exists():
        n += 1
        name = f"{stamp}-{n}"
    out_dir = runs_dir / name
    out_dir.mkdir(parents=True)
    path = out_dir / REPORT_FILE
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class CheckRun:
    """A check's report, with what it was made from."""

    report: CheckReport
    result: LiveSearchResult  # the retrieval: the candidates, the agent's trace, the pool
    explain: ExplainStep | None  # None in quick mode
    keywords_cached: bool | None  # whether the keyword lookup came from the cache
    merged: list[tuple[str, str]]  # (kept id, merged id): versions merged in the final list


def _position(candidates: list[Candidate], record: Record) -> int | None:
    """Where the same work is in the list (a shared OpenAlex id or alias), if it is."""
    ids = {record["id"], *record["aliases"]}
    for i, c in enumerate(candidates):
        if ids & {c.record["id"], *c.record["aliases"]}:
            return i
    return None


def methods_section(
    top: list[Candidate], methods: list[Candidate], k: int
) -> tuple[list[Candidate], list[int]]:
    """Section B, and the list A ranks of the `top` papers that list B ranks among it.

    Section B is the first k papers of list B (`methods`) that aren't among `top`
    (section A's explained papers). The ranks are those of the `top` papers list B
    ranks before section B is complete, in rank order: down to section B's last paper,
    every paper of list B is in section B or among them (all of `top` that list B
    has, if it runs out first).
    """
    section: list[Candidate] = []
    also: list[int] = []
    for c in methods:
        if len(section) == k:
            break
        i = _position(top, c.record)
        if i is None:
            section.append(c)
        else:
            also.append(top[i].rank)
    return section, sorted(also)


def paper(
    candidate: Candidate,
    snippet: Snippet | None,
    explanation: Explanation | None,
    venue: str | None = None,
) -> Paper:
    record = candidate.record
    return Paper(
        rank=candidate.rank,
        id=record["id"],
        aliases=list(record["aliases"]),
        url=f"https://openalex.org/{record['id']}",
        doi=record.get("doi"),
        title=record["title"],
        publication_date=record["publication_date"],
        type=record.get("type"),
        venue=venue,
        authors=[a["name"] for a in record.get("authors") or [] if a.get("name")],
        snippet=snippet,
        explanation=explanation,
    )


def run_check(
    request: CheckRequest,
    cfg: Config,
    *,
    search: LiveSearchClient,
    encoder: Encoder,
    agent: ChatModel | None = None,
    reranker: ChatModel | None = None,
    explainer: ChatModel | None = None,
    strict: bool = False,
    progress: Callable[[str], None] | None = None,
) -> CheckRun:
    """Check one idea: the default check (agent, reranker, explainer) or the quick one.

    `progress` is told each stage: SEARCHING, "agent query n of m", RERANKING,
    EXPLAINING. With `strict` (evaluation), a failed agent search, rerank call or
    explainer call raises instead of leaving a warning.
    """
    live, ex = cfg.live, cfg.explain
    if live is None or ex is None:
        raise ValueError("a check needs the config's live: and explain: sections")
    capped = replace(cfg, live=replace(live, cost_cap_per_check=ex.cost_cap_per_check))
    start = time.perf_counter()
    if request.quick:
        result = quick_check(
            request.text,
            capped,
            search=search,
            encoder=encoder,
            before=request.before,
            progress=progress,
        )
    else:
        if agent is None or reranker is None or explainer is None or cfg.rerank is None:
            raise ValueError(
                "the default check needs the agent, reranker and explainer clients and the "
                "config's rerank: section"
            )
        result = default_check(
            request.text,
            capped,
            search=search,
            encoder=encoder,
            agent=agent,
            reranker=reranker,
            before=request.before,
            strict=strict,
            progress=progress,
        )
    result, merged = merge_same_text(result)  # before the top papers are chosen
    warnings = list(result.warnings)
    top = result.candidates[: live.explain_top_k]
    top_methods: list[Candidate] = []  # two lists: section B
    also: list[int] | None = None  # two lists: section B's "Also close in methods" line
    if result.methods is not None:
        top = result.candidates[: ex.contribution_top_k]
        top_methods, also = methods_section(top, result.methods, ex.methods_top_k)
    explained = [*top, *top_methods]  # the papers to explain, each once
    records = [c.record for c in explained]

    mark = time.perf_counter()
    found = snippets(request.text, records, encoder) if records else []
    snippets_s = time.perf_counter() - mark

    mark = time.perf_counter()
    ledger = replace(result.ledger)
    keywords: dict[str, list[str]] = {}
    venues: dict[str, str] = {}
    keywords_cached: bool | None = None
    if result.candidates:
        listed = [c.record["id"] for c in [*result.candidates, *top_methods]]
        try:
            ledger.require(LIST_CALL_USD, "the keyword lookup")
            lookup = search.details(list(dict.fromkeys(listed)))
            ledger.add_lookup(lookup)
            keywords, venues, keywords_cached = lookup.keywords, lookup.venues, lookup.cached
        except (CheckCostCapExceeded, OpenAlexError) as exc:
            warnings.append(f"No OpenAlex keywords for the vocabulary map, or venues ({exc}).")
    keywords_s = time.perf_counter() - mark

    step: ExplainStep | None = None
    explanations: list[Explanation | None] = [None] * len(records)
    if not request.quick:
        assert explainer is not None
        if progress is not None:
            progress(EXPLAINING)
        step = explain(
            request.text,
            records,
            explainer,
            ex.model,
            ledger,
            abstract_chars=ex.abstract_chars,
            strict=strict,
            ranks=[
                str(c.rank) if i < len(top) else f"{c.rank} of the methods list"
                for i, c in enumerate(explained)
            ],
        )
        ledger, explanations = step.ledger, step.explanations
        if step.warning:
            warnings.append(step.warning)

    mark = time.perf_counter()
    vocab = vocabulary(
        request.text,
        result.trace.queries[1:],  # the agent's; the first query is the idea itself
        records,
        keywords,
        max_terms=ex.vocabulary_terms,
    )
    vocabulary_s = time.perf_counter() - mark

    rerank_s = result.rerank.latency_s if result.rerank else 0.0
    before = f" published before {request.before.isoformat()}" if request.before else ""
    models = {"encoder": encoder.name}
    prompts: dict[str, str] = {}
    if not request.quick:
        assert cfg.rerank is not None
        models.update(agent=cfg.agent.name, rerank=cfg.rerank.model.name, explain=ex.model.name)
        prompts.update(result.trace.prompt_hashes)
        if result.methods is None:
            prompts[LISTWISE] = load_prompt(LISTWISE).sha256
        else:
            prompts[TWO_LISTS] = load_prompt(TWO_LISTS).sha256
            prompts["rerank_schema"] = TWO_LISTS_SCHEMA_SHA256
        prompts[EXPLAIN_PROMPT] = load_prompt(EXPLAIN_PROMPT).sha256
        prompts["explain_schema"] = EXPLAIN_SCHEMA_SHA256

    def as_paper(c: Candidate, i: int | None) -> Paper:
        """The paper, with the snippet and explanation of `explained[i]` if i isn't None."""
        snippet = found[i] if i is not None and i < len(found) else None
        explanation = explanations[i] if i is not None and i < len(explanations) else None
        return paper(c, snippet, explanation, venues.get(c.record["id"]))

    papers = [as_paper(c, i if i < len(top) else None) for i, c in enumerate(result.candidates)]
    methods = None
    idea = None
    if result.methods is not None:
        methods = [as_paper(c, len(top) + i) for i, c in enumerate(top_methods)]
        r = result.rerank
        if r is not None and (r.contribution_line or r.methods_line):
            idea = IdeaSummary(contribution=r.contribution_line, methods=r.methods_line)
    report = CheckReport(
        coverage=(
            f"Searched {result.pool_size} candidate papers from OpenAlex{before} "
            f"({len(result.searches)} search{'' if len(result.searches) == 1 else 'es'})."
        ),
        mode=QUICK if request.quick else DEFAULT,
        request=request,
        papers=papers,
        methods=methods,
        also_in_methods=also,
        idea=idea,
        vocabulary=vocab,
        warnings=warnings,
        queries=result.trace.queries[1:],
        n_candidates=result.pool_size,
        n_searches=len(result.searches),
        rerank=result.rerank.status if result.rerank else None,
        explainer=step.status if step else None,
        cost=Cost(
            method_cost=ledger.method_cost,
            billed_cost=ledger.billed_cost,
            openalex_cost=ledger.openalex_cost,
            llm_method_cost=ledger.llm.method_cost,
            cap=ledger.cap,
        ),
        stages=Stages(
            search_s=result.latency_s - rerank_s,
            rerank_s=rerank_s,
            snippets_s=snippets_s,
            keywords_s=keywords_s,
            explain_s=step.latency_s if step else 0.0,
            vocabulary_s=vocabulary_s,
            total_s=time.perf_counter() - start,
        ),
        models=models,
        prompt_hashes=prompts,
        encoder=result.encoder,
        encoder_device=result.encoder_device,
        created_at=datetime.now(timezone.utc),
    )
    return CheckRun(report, result, step, keywords_cached, merged)
