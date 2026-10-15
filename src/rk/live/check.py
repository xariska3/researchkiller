"""One whole check, as the tool runs it (docs/phase2_spec.md, Pipeline; Step 5).

`run_check` runs the default check (L5r, rk.live.pipeline.default_check) or the quick
check (L1, rk.live.pipeline.quick_check, no LLM). Then, for the top live.explain_top_k
papers:
1. a snippet each (rk.live.explain.snippets);
2. their OpenAlex keywords, for the vocabulary map (one list call);
3. the explainer call (default check only; rk.live.explain.explain);
4. the vocabulary map (rk.live.vocab).

The whole check runs under explain.cost_cap_per_check. Nothing after the retrieval can
fail it: a keyword lookup or an explainer call that fails or could cross the cap leaves
the papers as they are, with a warning. With `strict` (evaluation), a failed explainer
call raises instead.
"""

from __future__ import annotations

import time
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
    quick_check,
)
from rk.live.report import (
    DEFAULT,
    QUICK,
    CheckReport,
    CheckRequest,
    Cost,
    Explanation,
    Paper,
    Snippet,
    Stages,
)
from rk.live.reranker import LISTWISE
from rk.live.vocab import vocabulary
from rk.llm import ChatModel, load_prompt
from rk.retrieval import Encoder

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


def paper(candidate: Candidate, snippet: Snippet | None, explanation: Explanation | None) -> Paper:
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
) -> CheckRun:
    """Check one idea: the default check (agent, reranker, explainer) or the quick one."""
    live, ex = cfg.live, cfg.explain
    if live is None or ex is None:
        raise ValueError("a check needs the config's live: and explain: sections")
    capped = replace(cfg, live=replace(live, cost_cap_per_check=ex.cost_cap_per_check))
    start = time.perf_counter()
    if request.quick:
        result = quick_check(
            request.text, capped, search=search, encoder=encoder, before=request.before
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
        )
    warnings = list(result.warnings)
    top = result.candidates[: live.explain_top_k]
    records = [c.record for c in top]

    mark = time.perf_counter()
    found = snippets(request.text, records, encoder) if records else []
    snippets_s = time.perf_counter() - mark

    mark = time.perf_counter()
    ledger = replace(result.ledger)
    keywords: dict[str, list[str]] = {}
    keywords_cached: bool | None = None
    if records:
        try:
            ledger.require(LIST_CALL_USD, "the keyword lookup")
            lookup = search.keywords([r["id"] for r in records])
            ledger.add_lookup(lookup)
            keywords, keywords_cached = lookup.keywords, lookup.cached
        except (CheckCostCapExceeded, OpenAlexError) as exc:
            warnings.append(f"No OpenAlex keywords for the vocabulary map ({exc}).")
    keywords_s = time.perf_counter() - mark

    step: ExplainStep | None = None
    explanations: list[Explanation | None] = [None] * len(records)
    if not request.quick:
        assert explainer is not None
        step = explain(
            request.text,
            records,
            explainer,
            ex.model,
            ledger,
            abstract_chars=ex.abstract_chars,
            strict=strict,
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
        prompts[LISTWISE] = load_prompt(LISTWISE).sha256
        prompts[EXPLAIN_PROMPT] = load_prompt(EXPLAIN_PROMPT).sha256
        prompts["explain_schema"] = EXPLAIN_SCHEMA_SHA256
    papers = [
        paper(
            c,
            found[i] if i < len(found) else None,
            explanations[i] if i < len(explanations) else None,
        )
        for i, c in enumerate(result.candidates)
    ]
    report = CheckReport(
        coverage=(
            f"Searched {result.pool_size} candidate papers from OpenAlex{before} "
            f"({len(result.searches)} search{'' if len(result.searches) == 1 else 'es'})."
        ),
        mode=QUICK if request.quick else DEFAULT,
        request=request,
        papers=papers,
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
    return CheckRun(report, result, step, keywords_cached)
