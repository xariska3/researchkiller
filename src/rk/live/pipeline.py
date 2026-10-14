"""One live check, steps 2-4 of docs/phase2_spec.md: B5 query planning on live search, rerank.

Phase 1's B5 agent loop (rk.agent.AgentRetriever, the same prompt and the same
caps: retrieval.max_queries and max_llm_calls_agent) runs unchanged; only its
search tool is live. Each query, the user's text first, goes to OpenAlex
(keyword search unless it has more than live.keyword_max_words words, plus
semantic search); the results join the candidate pool, and the agent sees the
top-10 titles of the pool ranked for that query. When the loop ends, the final
pool is ranked as B5 ranks the frozen corpus, RRF over every query's hybrid
ranking, and the top live.top_k are kept.

Every check has its own ledger with a hard cap on its method cost (OpenAlex
plus LLM, cache hits included, so a replayed check stops where the original
would): before each search it reserves the documented price, and before each
agent call a pessimistic bound (rk.llm.pessimistic_cost). A call that could
cross live.cost_cap_per_check raises CheckCostCapExceeded, and the check fails.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from rk.agent import AgentRetriever
from rk.config import Config, LiveConfig, ModelConfig
from rk.live.openalex_search import (
    KEYWORD,
    SEARCH_COST_USD,
    SEMANTIC,
    LiveSearchClient,
    SearchResponse,
)
from rk.live.pool import CandidatePool, Hit
from rk.llm import ChatModel, Completion, Message, Usage, load_prompt, pessimistic_cost
from rk.retrieval import Encoder, Trace
from rk.targets import Record

AGENT_PROMPT = "agent_step"  # Phase 1's B5 prompt


class CheckCostCapExceeded(RuntimeError):
    """A call could take one check's cost over live.cost_cap_per_check."""


@dataclass
class CheckLedger:
    """What one check has cost so far (USD), with a hard cap on its method cost."""

    cap: float
    openalex_cost: float = 0.0  # every search, cache hits included
    openalex_spent: float = 0.0  # searches actually sent
    llm: Usage = field(default_factory=Usage)

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
                f"${self.method_cost:.4f} of its ${self.cap:.2f} cap"
            )

    def add_search(self, response: SearchResponse) -> None:
        self.openalex_cost += response.cost_usd
        self.openalex_spent += response.money_spent


class CappedChatModel:
    """A ChatModel that refuses calls whose pessimistic cost could cross the check's cap."""

    def __init__(self, inner: ChatModel, model: ModelConfig, ledger: CheckLedger) -> None:
        self.inner = inner
        self.model = model
        self.ledger = ledger

    def complete(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> Completion:
        self.ledger.require(pessimistic_cost(self.model, messages), "an agent call")
        reply = self.inner.complete(messages, seed=seed, response_format=response_format)
        self.ledger.llm += reply.usage
        return reply


class LiveRanker:
    """The agent's search tool (an rk.retrieval Ranker): search OpenAlex, then rank the pool."""

    def __init__(
        self,
        search: LiveSearchClient,
        pool: CandidatePool,
        ledger: CheckLedger,
        live: LiveConfig,
        before: date | None = None,
    ) -> None:
        self.search = search
        self.pool = pool
        self.ledger = ledger
        self.live = live
        self.before = before
        self.queries: list[str] = []
        self.searches: list[SearchResponse] = []

    def kinds(self, query: str) -> list[str]:
        """The searches a query gets: keyword unless it's long, semantic if enabled."""
        kinds = [KEYWORD] if len(query.split()) <= self.live.keyword_max_words else []
        return kinds + [SEMANTIC] if self.live.semantic_search else kinds

    def rank(self, query: str) -> list[str]:
        query_index = len(self.queries)
        self.queries.append(query)
        for kind in self.kinds(query):
            self.ledger.require(SEARCH_COST_USD, f"a {kind} search")
            response = self.search.search(query, kind, before=self.before)
            self.ledger.add_search(response)
            self.searches.append(response)
            self.pool.add(response.records, query_index=query_index, kind=kind)
        return self.pool.rank(query)


@dataclass(frozen=True)
class Candidate:
    rank: int  # from 1, in the final reranked list
    record: Record
    hits: list[Hit]  # where the live searches found it


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "before": self.before.isoformat() if self.before else None,
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
                }
                for s in self.searches
            ],
            "pool_size": self.pool_size,
            "pool_dropped": self.pool_dropped,
            "candidates": [
                {
                    "rank": c.rank,
                    "id": c.record["id"],
                    "aliases": c.record["aliases"],
                    "doi": c.record["doi"],
                    "title": c.record["title"],
                    "type": c.record.get("type"),
                    "publication_date": c.record["publication_date"],
                    "found_by": [asdict(h) for h in c.hits],
                }
                for c in self.candidates
            ],
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


def live_search(
    text: str,
    cfg: Config,
    *,
    search: LiveSearchClient,
    encoder: Encoder,
    agent: ChatModel,
    before: date | None = None,
) -> LiveSearchResult:
    """Run the B5 loop on live search for `text` and rerank the pool (no explainer yet).

    `before` restricts the search to works published before that date.
    """
    live = cfg.live
    if live is None:
        raise ValueError("the config has no live: section")
    start = time.perf_counter()
    ledger = CheckLedger(live.cost_cap_per_check)
    pool = CandidatePool(encoder, max_size=live.max_pool_size)
    ranker = LiveRanker(search, pool, ledger, live, before)
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
    by_id = {r["id"]: r for r in pool.records}
    top = pool.fused(trace.queries)[: live.top_k]
    candidates = [Candidate(n, by_id[d], pool.hits(by_id[d])) for n, d in enumerate(top, start=1)]
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
    )
