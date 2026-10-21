"""One live check, steps 2-4 of docs/phase2_spec.md: B5 query planning on live search, rerank.

Phase 1's B5 agent loop (rk.agent.AgentRetriever, the same prompt and the same
caps: retrieval.max_queries and max_llm_calls_agent) runs unchanged; only its
search tool is live. Each query, the user's text first, goes to OpenAlex
(keyword search unless it has more than live.keyword_max_words words, plus
semantic search); the results join the candidate pool, and the agent sees the
top-10 titles of the pool ranked for that query. When the loop ends, the final
pool is ranked as B5 ranks the frozen corpus, RRF over every query's hybrid
ranking, and the top live.top_k are kept. Another final fusion
(rk.live.fusion) can rank the same pool instead (`fusion=`, or `with_fusion`
on a finished check).

Every check has its own ledger with a hard cap on its method cost (OpenAlex
plus LLM, cache hits included, so a replayed check stops where the original
would): before each search it reserves the documented price, and before each
agent call a pessimistic bound (rk.llm.pessimistic_cost). A call that could
cross live.cost_cap_per_check raises CheckCostCapExceeded, and the check fails.

The default check is L5r (default_check; docs/phase2_spec.md, Product ideas): the
check with its final pool ranked by L5w's fusion (DEFAULT_FUSION), then reranked by
one listwise LLM call (with_rerank, rk.live.reranker). The rerank call is charged to
the check's ledger under the same cap. If it could cross the cap, fails, or its
reply has no readable ranking, the check keeps L5w's order and carries a warning;
in evaluation (strict), a failed call raises instead. With rerank.two_lists (issue
#22), the same call returns two lists: the check's candidates are list A (closest in
contribution) and its `methods` list B (closest in methods). The quick check
(quick_check) is L1: one semantic search, reranked locally, with no LLM.

rk.live.check runs a whole check: one of these, then what it shows for its top papers.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any

from rk.agent import AgentRetriever
from rk.config import Config, LiveConfig, ModelConfig
from rk.corpus import OpenAlexError
from rk.live.fusion import (
    B5,
    L1_L5,
    PROPOSAL_EQUAL,
    PROPOSAL_ONLY,
    final_ranking,
    l1_ranking,
)
from rk.live.openalex_search import (
    KEYWORD,
    SEARCH_COST_USD,
    SEMANTIC,
    LiveSearchClient,
    SearchResponse,
    WorkDetails,
    search_text,
)
from rk.live.pool import CandidatePool, Hit, same_text
from rk.live.reranker import (
    LISTWISE,
    TWO_LISTS,
    TWO_LISTS_RESPONSE_FORMAT,
    TwoLists,
    candidates,
    listwise_ranking,
    parse_order,
    parse_two_lists,
    render_prompt,
)
from rk.llm import (
    ChatModel,
    Completion,
    LLMError,
    Message,
    Usage,
    load_prompt,
    pessimistic_cost,
)
from rk.retrieval import Encoder, Trace
from rk.targets import Record, id_number

AGENT_PROMPT = "agent_step"  # Phase 1's B5 prompt
# L5w's final fusion, chosen on the dev set (rk.live.evaluate.L5W_FUSION): the default
# check ranks its pool this way before the rerank call.
DEFAULT_FUSION = PROPOSAL_EQUAL
# What the rerank step did (RerankStep.status).
RERANKED = "reranked"  # the reply's order
FALLBACK = "fallback"  # L5w's order, with a warning
SKIPPED = "skipped"  # no candidates, so no call
# Stages a check reports to its `progress` callback (the web app shows them).
SEARCHING = "searching"  # the user's own search; then "agent query n of m"
RERANKING = "reranking"


class CheckCostCapExceeded(RuntimeError):
    """A call could take one check's cost over live.cost_cap_per_check."""

    def __init__(self, message: str, ledger: CheckLedger | None = None) -> None:
        super().__init__(message)
        self.ledger = ledger  # what the check had cost when it stopped


@dataclass
class CheckLedger:
    """What one check has cost so far (USD), with a hard cap on its method cost."""

    cap: float
    openalex_cost: float = 0.0  # every search, cache hits included
    openalex_spent: float = 0.0  # searches actually sent
    llm: Usage = field(default_factory=Usage)
    searches: int = 0
    searches_sent: int = 0

    @property
    def method_cost(self) -> float:
        return self.openalex_cost + self.llm.method_cost

    @property
    def billed_cost(self) -> float:
        """Money actually spent: searches sent, and LLM calls as the API billed them."""
        return self.openalex_spent + self.llm.billed_cost

    def require(self, amount: float, what: str) -> None:
        if self.method_cost + amount > self.cap:
            raise CheckCostCapExceeded(
                f"{what} could cost ${amount:.4f}, and the check has already cost "
                f"${self.method_cost:.4f} of its ${self.cap:.2f} cap",
                self,
            )

    def add_search(self, response: SearchResponse) -> None:
        self.openalex_cost += response.cost_usd
        self.openalex_spent += response.money_spent
        self.searches += 1
        self.searches_sent += not response.cached

    def add_lookup(self, lookup: WorkDetails) -> None:
        """Another OpenAlex call that isn't a search (the papers' keywords and venues)."""
        self.openalex_cost += lookup.cost_usd
        self.openalex_spent += lookup.money_spent


class CappedChatModel:
    """A ChatModel that refuses calls whose pessimistic cost could cross the check's cap."""

    def __init__(
        self,
        inner: ChatModel,
        model: ModelConfig,
        ledger: CheckLedger,
        *,
        what: str = "an agent call",
    ) -> None:
        self.inner = inner
        self.model = model
        self.ledger = ledger
        self.what = what  # names the call in the refusal

    def complete(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> Completion:
        self.ledger.require(pessimistic_cost(self.model, messages), self.what)
        reply = self.inner.complete(messages, seed=seed, response_format=response_format)
        self.ledger.llm += reply.usage
        return reply


class LiveRanker:
    """The agent's search tool (an rk.retrieval Ranker): search OpenAlex, then rank the pool.

    With `skip_failed` (the tool, not evaluation), a failed agent search is skipped with
    a warning (`warnings`) instead of failing the check. The user's own search (the
    first query) is skipped the same way only if OpenAlex rejects it (HTTP 400), and
    `own_search_error` keeps the error; any other failure of it fails the check.
    `progress` is told the stage of each query.
    """

    def __init__(
        self,
        search: LiveSearchClient,
        pool: CandidatePool,
        ledger: CheckLedger,
        live: LiveConfig,
        before: date | None = None,
        *,
        skip_failed: bool = False,
        progress: Callable[[str], None] | None = None,
        max_queries: int = 0,
    ) -> None:
        self.search = search
        self.pool = pool
        self.ledger = ledger
        self.live = live
        self.before = before
        self.skip_failed = skip_failed
        self.progress = progress
        self.max_queries = max_queries  # for the progress line only
        self.queries: list[str] = []
        self.searches: list[SearchResponse] = []
        self.warnings: list[str] = []
        self.own_search_error: OpenAlexError | None = None  # the user's search, rejected

    def kinds(self, query: str) -> list[str]:
        """The searches a query gets: keyword unless it's long, semantic if enabled."""
        kinds = [KEYWORD] if len(query.split()) <= self.live.keyword_max_words else []
        return kinds + [SEMANTIC] if self.live.semantic_search else kinds

    def rank(self, query: str) -> list[str]:
        query_index = len(self.queries)
        self.queries.append(query)
        if self.progress is not None:
            agent = f"agent query {query_index} of {max(self.max_queries - 1, query_index)}"
            self.progress(SEARCHING if query_index == 0 else agent)
        for kind in self.kinds(query):
            self.ledger.require(SEARCH_COST_USD, f"a {kind} search")
            try:
                response = self.search.search(query, kind, before=self.before)
            except OpenAlexError as exc:
                rejected = exc.status == 400  # the request itself: sending it again can't help
                if not self.skip_failed or (query_index == 0 and not rejected):
                    raise
                if query_index == 0:
                    self.own_search_error = self.own_search_error or exc
                    self.warnings.append(
                        f"OpenAlex rejected the search of your idea, so it was skipped ({kind} "
                        f"search: {str(exc)[:200]}); the results come from the agent's searches."
                    )
                else:
                    self.warnings.append(
                        f"An agent search failed and was skipped ({kind} search for "
                        f"{query[:80]!r}: {str(exc)[:200]}); the results come from the others."
                    )
                continue
            self.ledger.add_search(response)
            self.searches.append(response)
            self.pool.add(response.records, query_index=query_index, kind=kind)
        return self.pool.rank(query)


@dataclass(frozen=True)
class Candidate:
    rank: int  # from 1, in the final reranked list
    record: Record
    hits: list[Hit]  # where the live searches found it
    versions: list[Record]  # the search results merged into it, one per OpenAlex id


@dataclass(frozen=True)
class RerankStep:
    """What the rerank step did to a check (with_rerank)."""

    status: str  # RERANKED, FALLBACK (L5w's order kept) or SKIPPED (no candidates)
    n_candidates: int
    latency_s: float  # the step's time, the call included
    usage: Usage  # the call's, if it was answered
    finish_reason: str | None = None
    warning: str | None = None  # why the order is L5w's (FALLBACK only)
    # Two lists only (None otherwise): list B's status (RERANKED, FALLBACK: empty, or
    # SKIPPED), and the reply's one line each on the idea's contribution and methods.
    methods_status: str | None = None
    contribution_line: str | None = None
    methods_line: str | None = None


@dataclass(frozen=True)
class LiveSearchResult:
    """One check's retrieval: the top candidates and what it took to find them."""

    text: str
    before: date | None
    candidates: list[Candidate]  # the top live.top_k, best first
    trace: Trace  # the agent loop: queries run, steps, stop reason, LLM usage, errors
    searches: list[SearchResponse]
    pool_size: int
    pool_dropped: int
    ledger: CheckLedger
    latency_s: float
    encoder: str
    encoder_device: str
    pool: CandidatePool = field(repr=False, compare=False)  # the final pool
    fusion: str = B5  # how the final pool was ranked (rk.live.fusion)
    rerank: RerankStep | None = None  # set by with_rerank
    warnings: tuple[str, ...] = ()  # for the user, e.g. that the rerank step fell back
    # Two lists (with_rerank's two_lists): list B, the top live.top_k closest in methods
    # (empty if it couldn't be read). None for a one-list check.
    methods: list[Candidate] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "before": self.before.isoformat() if self.before else None,
            "fusion": self.fusion,
            "rerank": asdict(self.rerank) if self.rerank else None,
            "warnings": list(self.warnings),
            "queries": self.trace.queries,
            "steps": self.trace.steps,
            "stop_reason": self.trace.stop_reason,
            "errors": self.trace.errors,
            "prompt_hashes": self.trace.prompt_hashes,
            "searches": [
                {
                    "query": s.query,
                    "kind": s.kind,
                    "n_results": s.n_results,
                    "n_usable": len(s.records),
                    "cost_usd": s.cost_usd,
                    "cached": s.cached,
                    "fetched_at": s.fetched_at.isoformat(),
                }
                for s in self.searches
            ],
            "pool_size": self.pool_size,
            "pool_dropped": self.pool_dropped,
            "candidates": [_candidate_dict(c) for c in self.candidates],
            "methods": (
                None if self.methods is None else [_candidate_dict(c) for c in self.methods]
            ),
            "cost": {
                "method_cost": self.ledger.method_cost,
                "billed_cost": self.ledger.billed_cost,
                "openalex_cost": self.ledger.openalex_cost,
                "openalex_spent": self.ledger.openalex_spent,
                "llm": asdict(self.ledger.llm),
                "cap": self.ledger.cap,
            },
            "latency_s": self.latency_s,
            "encoder": self.encoder,
            "encoder_device": self.encoder_device,
        }


def _candidate_dict(c: Candidate) -> dict[str, Any]:
    return {
        "rank": c.rank,
        "id": c.record["id"],
        "aliases": c.record["aliases"],
        "doi": c.record["doi"],
        "title": c.record["title"],
        "type": c.record.get("type"),
        "publication_date": c.record["publication_date"],
        "found_by": [asdict(h) for h in c.hits],
    }


def long_text_warning(text: str, live: LiveConfig) -> str | None:
    """The warning for an idea OpenAlex's search limit cuts (search_text), if it does."""
    sent = search_text(text, live.max_search_chars)
    if sent == text:
        return None
    return (
        f"Your idea is longer than OpenAlex's {live.max_search_chars:,}-character search "
        f"limit, so the first search used its first {len(sent):,} characters; the rest of "
        "the check used the full text."
    )


def proposal_records(text: str, searches: Sequence[SearchResponse]) -> list[Record]:
    """The results of the semantic search on the user's text (L1's input), or none."""
    return next((s.records for s in searches if s.query == text and s.kind == SEMANTIC), [])


def final_candidates(
    pool: CandidatePool,
    queries: Sequence[str],
    fusion: str,
    top_k: int,
    *,
    l1: Sequence[str] = (),
) -> list[Candidate]:
    """The top_k of the final pool ranked by `fusion`, with their hits and versions."""
    return ranked_candidates(pool, final_ranking(pool, queries, fusion, l1=l1), top_k)


def ranked_candidates(pool: CandidatePool, ranking: Sequence[str], top_k: int) -> list[Candidate]:
    """The top_k of a ranking of the pool's works, with their hits and versions."""
    by_id = {r["id"]: r for r in pool.records}
    return [
        Candidate(n, by_id[d], pool.hits(by_id[d]), pool.versions(by_id[d]))
        for n, d in enumerate(ranking[:top_k], start=1)
    ]


def _l1(
    text: str, searches: Sequence[SearchResponse], encoder: Encoder, live: LiveConfig
) -> list[str]:
    records = proposal_records(text, searches)
    return l1_ranking(records, text, encoder, max_size=live.max_pool_size)


def with_fusion(
    result: LiveSearchResult, fusion: str, encoder: Encoder, live: LiveConfig
) -> LiveSearchResult:
    """The same check ranked by another final fusion: same agent loop, searches and pool.

    Its latency is the check's plus the time the new ranking took.
    """
    start = time.perf_counter()
    l1 = _l1(result.text, result.searches, encoder, live) if fusion == L1_L5 else []
    candidates = final_candidates(result.pool, result.trace.queries, fusion, live.top_k, l1=l1)
    latency = result.latency_s + time.perf_counter() - start
    return replace(result, candidates=candidates, fusion=fusion, latency_s=latency)


def with_rerank(
    result: LiveSearchResult,
    cfg: Config,
    reranker: ChatModel,
    encoder: Encoder,
    *,
    strict: bool = False,
    progress: Callable[[str], None] | None = None,
    two_lists: bool = False,
) -> LiveSearchResult:
    """L5r: the check's final pool reranked by one listwise LLM call (rk.live.reranker).

    `result` is the check as L5w ranks it (its fusion). The candidates are the union
    of that ranking's top rerank.depth and B5's, and L5w's other works follow the
    reranked ones. The call is charged to a copy of the check's ledger, under the
    check's cost cap, and the step's time is added to the check's latency.

    If the call could cross the cap, fails (LLMError), or its reply has no readable
    ranking, the result keeps L5w's order and gets a warning. With `strict`, for
    evaluation, a failed call raises instead: an API failure is an error, not L5w.

    With `two_lists` (issue #22), the call is the two-list one: the result's candidates
    are its list A (closest in contribution, then L5w's other works) and its `methods`
    are list B (closest in methods, then the other candidates in presented order).
    Where L5r would keep L5w's order (list A unreadable), list B is empty too, under
    one warning; a readable list A without a readable list B gets a warning of its own.
    """
    rr, live = cfg.rerank, cfg.live
    if rr is None or live is None:
        raise ValueError("the rerank step needs the config's live: and rerank: sections")
    start = time.perf_counter()
    queries = result.trace.queries
    l1 = _l1(result.text, result.searches, encoder, live) if result.fusion == L1_L5 else []
    l5w = final_ranking(result.pool, queries, result.fusion, l1=l1)
    cands = candidates(l5w, final_ranking(result.pool, queries, B5), rr.depth)
    ledger = replace(result.ledger)  # the check's own ledger stays as L5w had it
    if not cands:
        seconds = time.perf_counter() - start
        methods_skipped = SKIPPED if two_lists else None
        step = RerankStep(SKIPPED, 0, seconds, Usage(), methods_status=methods_skipped)
        return replace(
            result,
            ledger=ledger,
            latency_s=result.latency_s + seconds,
            rerank=step,
            methods=[] if two_lists else None,
        )
    if progress is not None:
        progress(RERANKING)
    by_id = {r["id"]: r for r in result.pool.records}
    prompt = load_prompt(TWO_LISTS if two_lists else LISTWISE)
    text = render_prompt(prompt, result.text, [by_id[d] for d in cands], rr.abstract_chars)
    capped = CappedChatModel(reranker, rr.model, ledger, what="the rerank call")
    completion: Completion | None = None
    reply: TwoLists | None = None
    order: list[int] | None = None
    problem: str | None = None
    try:
        completion = capped.complete(
            [{"role": "user", "content": text}],
            response_format=TWO_LISTS_RESPONSE_FORMAT if two_lists else None,
        )
    except CheckCostCapExceeded as exc:  # "the rerank call could cost $x, and the check ..."
        problem = str(exc)
    except LLMError as exc:
        if strict:
            raise
        problem = f"the rerank call failed: {exc}"
    if completion is not None:
        if two_lists:
            reply = parse_two_lists(completion.text, len(cands))
            order = reply.contribution_order if reply else None
        else:
            order = parse_order(completion.text, len(cands))
        if order is None:
            which = " contribution" if two_lists else ""
            problem = f"the rerank reply had no readable{which} ranking"
    warnings = list(result.warnings)
    warning: str | None = None
    if order is None:
        ranking = l5w
        kept = (
            "the contribution list keeps the check's order, and there is no methods list"
            if two_lists
            else "the results keep the check's order"
        )
        warning = f"Not reranked ({problem}); {kept}."
        warnings.append(warning)
    else:
        ranking = list(dict.fromkeys([*listwise_ranking(cands, order), *l5w]))
    methods: list[Candidate] | None = None
    methods_status: str | None = None
    if two_lists:
        methods_order = reply.methods_order if reply and order is not None else None
        methods_status = FALLBACK if methods_order is None else RERANKED
        if methods_order is None:
            methods = []
            if order is not None:
                warnings.append(
                    "No methods list (the rerank reply had no readable methods ranking)."
                )
        else:
            by_methods = listwise_ranking(cands, methods_order)
            methods = ranked_candidates(result.pool, by_methods, live.top_k)
    seconds = time.perf_counter() - start
    step = RerankStep(
        status=FALLBACK if warning else RERANKED,
        n_candidates=len(cands),
        latency_s=seconds,
        usage=completion.usage if completion else Usage(),
        finish_reason=completion.finish_reason if completion else None,
        warning=warning,
        methods_status=methods_status,
        contribution_line=reply.contribution if reply else None,
        methods_line=reply.methods if reply else None,
    )
    return replace(
        result,
        candidates=ranked_candidates(result.pool, ranking, live.top_k),
        ledger=ledger,
        latency_s=result.latency_s + seconds,
        rerank=step,
        warnings=tuple(warnings),
        methods=methods,
    )


def live_search(
    text: str,
    cfg: Config,
    *,
    search: LiveSearchClient,
    encoder: Encoder,
    agent: ChatModel,
    before: date | None = None,
    fusion: str = B5,
    skip_failed_searches: bool = False,
    progress: Callable[[str], None] | None = None,
) -> LiveSearchResult:
    """Run the B5 loop on live search for `text` and rerank the pool (no explainer yet).

    `before` restricts the search to works published before that date; `fusion`
    is how the final pool is ranked (rk.live.fusion; B5's by default). With
    `skip_failed_searches` (the tool), a failed agent search, or a search of `text`
    that OpenAlex rejects (HTTP 400), is skipped with a warning (LiveRanker); the check
    still fails if `text`'s search was rejected and no other search found anything.
    Evaluation leaves it off, so the check fails. Every search sends at most
    live.max_search_chars (search_text); a `text` that is cut gets a warning, and
    everything else uses it in full. `progress` gets each stage (SEARCHING, then
    "agent query n of m").
    """
    live = cfg.live
    if live is None:
        raise ValueError("the config has no live: section")
    start = time.perf_counter()
    ledger = CheckLedger(live.cost_cap_per_check)
    pool = CandidatePool(encoder, max_size=live.max_pool_size)
    ranker = LiveRanker(
        search,
        pool,
        ledger,
        live,
        before,
        skip_failed=skip_failed_searches,
        progress=progress,
        max_queries=cfg.retrieval.max_queries,
    )
    b5 = AgentRetriever(
        ranker,
        CappedChatModel(agent, cfg.agent, ledger),
        titles=pool.titles,  # updated in place as the pool grows
        max_queries=cfg.retrieval.max_queries,
        max_llm_calls=cfg.retrieval.max_llm_calls_agent,
        prompt=load_prompt(AGENT_PROMPT),
        name="B5-live",
    )
    trace = b5.retrieve(text).trace
    rejected = ranker.own_search_error
    if rejected is not None and not len(pool):
        raise OpenAlexError(
            f"OpenAlex rejected the search of your idea, and the agent's searches found "
            f"nothing: {rejected}",
            status=rejected.status,
        ) from rejected
    l1 = _l1(text, ranker.searches, encoder, live) if fusion == L1_L5 else []
    candidates = final_candidates(pool, trace.queries, fusion, live.top_k, l1=l1)
    cut = long_text_warning(text, live)
    return LiveSearchResult(
        text=text,
        before=before,
        candidates=candidates,
        trace=trace,
        searches=ranker.searches,
        pool_size=len(pool),
        pool_dropped=pool.dropped,
        ledger=ledger,
        latency_s=time.perf_counter() - start,
        encoder=encoder.name,
        encoder_device=encoder.device,
        pool=pool,
        fusion=fusion,
        warnings=(*([cut] if cut else []), *ranker.warnings),
    )


def merge_versions(
    candidates: Sequence[Candidate],
) -> tuple[list[Candidate], list[tuple[str, str]]]:
    """A ranked list with versions of one paper merged by their text (pool.same_text).

    Going down the list, a work that is the same text as an earlier entry joins that
    entry, which keeps its place. The entry shows the earliest-published version (on a
    tie, the one earlier in the list); the others' ids and aliases become its aliases,
    and all their versions and hits are kept. The rest move up, so the list can get
    shorter. Returns the list, renumbered, and the (kept id, merged id) pairs.
    """
    kept: list[Candidate] = []
    merged: list[tuple[str, str]] = []
    for candidate in candidates:
        same = next((i for i, k in enumerate(kept) if same_text(k.record, candidate.record)), None)
        if same is None:
            kept.append(candidate)
            continue
        first = kept[same]
        main, other = first.record, candidate.record
        if candidate.record["publication_date"] < first.record["publication_date"]:
            main, other = other, main  # the later entry was published first
        ids = {*first.record["aliases"], other["id"], *candidate.record["aliases"]}
        kept[same] = Candidate(
            first.rank,
            {**main, "aliases": sorted(ids - {main["id"]}, key=id_number)},
            sorted([*first.hits, *candidate.hits], key=lambda h: (h.query_index, h.kind, h.rank)),
            [*first.versions, *candidate.versions],
        )
        merged.append((main["id"], other["id"]))
    return [replace(c, rank=n) for n, c in enumerate(kept, start=1)], merged


def merge_same_text(result: LiveSearchResult) -> tuple[LiveSearchResult, list[tuple[str, str]]]:
    """The check's final list with versions of one paper merged (merge_versions).

    A two-list check's methods list is merged the same way. The pool and its ranking
    are unchanged; live-eval doesn't merge.
    """
    candidates, merged = merge_versions(result.candidates)
    methods = result.methods
    if methods is not None:
        methods, merged_methods = merge_versions(methods)
        merged = list(dict.fromkeys([*merged, *merged_methods]))
    if not merged:
        return result, []
    return replace(result, candidates=candidates, methods=methods), merged


def quick_check(
    text: str,
    cfg: Config,
    *,
    search: LiveSearchClient,
    encoder: Encoder,
    before: date | None = None,
    progress: Callable[[str], None] | None = None,
) -> LiveSearchResult:
    """The quick check, L1: one OpenAlex semantic search of `text`, reranked locally. No LLM.

    The results go into a pool of their own, ranked by its hybrid ranker for `text`
    (the final pool ranked by the proposal alone), as live-eval's L1 ranks them. The
    search sends at most live.max_search_chars of `text` (with a warning if it is cut);
    the ranking uses all of it. A failed search fails the check: there is no other.
    """
    live = cfg.live
    if live is None:
        raise ValueError("the config has no live: section")
    if progress is not None:
        progress(SEARCHING)
    start = time.perf_counter()
    ledger = CheckLedger(live.cost_cap_per_check)
    pool = CandidatePool(encoder, max_size=live.max_pool_size)
    ledger.require(SEARCH_COST_USD, "a semantic search")
    response = search.search(text, SEMANTIC, before=before)
    ledger.add_search(response)
    pool.add(response.records, query_index=0, kind=SEMANTIC)
    cut = long_text_warning(text, live)
    return LiveSearchResult(
        text=text,
        before=before,
        candidates=ranked_candidates(pool, pool.rank(text), live.top_k),
        trace=Trace(queries=[text], budget_used=1),
        searches=[response],
        pool_size=len(pool),
        pool_dropped=pool.dropped,
        ledger=ledger,
        latency_s=time.perf_counter() - start,
        encoder=encoder.name,
        encoder_device=encoder.device,
        pool=pool,
        fusion=PROPOSAL_ONLY,
        warnings=(cut,) if cut else (),
    )


def default_check(
    text: str,
    cfg: Config,
    *,
    search: LiveSearchClient,
    encoder: Encoder,
    agent: ChatModel,
    reranker: ChatModel,
    before: date | None = None,
    strict: bool = False,
    progress: Callable[[str], None] | None = None,
) -> LiveSearchResult:
    """The tool's default check, L5r: live_search ranked by DEFAULT_FUSION, then with_rerank.

    A failed agent search, or a rejected search of the idea (HTTP 400) when the agent's
    searches still find something, is skipped, and a rerank call that fails or can't be
    read leaves L5w's order, each with a warning in the result's `warnings`. With `strict`
    (evaluation), a failed agent search or rerank call raises instead. With
    rerank.two_lists, the rerank call returns two lists (with_rerank's `two_lists`).
    """
    if cfg.rerank is None:
        raise ValueError("the default check needs the config's rerank: section")
    result = live_search(
        text,
        cfg,
        search=search,
        encoder=encoder,
        agent=agent,
        before=before,
        fusion=DEFAULT_FUSION,
        skip_failed_searches=not strict,
        progress=progress,
    )
    return with_rerank(
        result,
        cfg,
        reranker,
        encoder,
        strict=strict,
        progress=progress,
        two_lists=cfg.rerank.two_lists,
    )
