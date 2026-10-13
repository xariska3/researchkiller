"""B5: an agent that searches one query at a time (spec: Retrieval configurations).

The proposal itself runs first, with no LLM call. Each agent step then sees the
queries so far with the top-10 titles of each (titles only, to bound tokens)
and either searches again or stops. Hard limits: max_queries queries in total
and max_llm_calls agent calls. A repeated query counts against the budget but
isn't run again; an unreadable step ends the loop. The final ranking uses the
same fusion function as B4 (retrieval.fuse_runs). B5-U is the same agent with
the larger, unconstrained budgets. Searches run through B3; in the secondary
B5-D they run through B2, so the titles the agent sees come from B2 too.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from rk import retrieval
from rk.config import Config
from rk.llm import ChatModel, Prompt, load_prompt, parse_json
from rk.retrieval import (
    BM25Retriever,
    DenseRetriever,
    Encoder,
    HybridRetriever,
    Index,
    Ranker,
    Result,
    Retriever,
    RewriteRetriever,
    Trace,
    normalize_query,
)

TITLES_PER_QUERY = 10


def parse_action(text: str) -> dict[str, str]:
    """{"action": "search", "query": ...} or {"action": "stop", "reason": ...}.

    Raises ValueError (no JSON, unknown action, no query) or TypeError (not an object).
    """
    data = parse_json(text)
    if not isinstance(data, dict):
        raise TypeError("expected a JSON object")
    action = str(data.get("action", "")).strip().lower()
    if action == "search":
        query = data.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search action without a query")
        return {"action": "search", "query": " ".join(query.split())}
    if action == "stop":
        reason = data.get("reason")
        return {"action": "stop", "reason": reason if isinstance(reason, str) else ""}
    raise ValueError(f"unknown action {data.get('action')!r}")


def render_history(
    queries: Sequence[str], runs: Sequence[Sequence[str]], titles: Mapping[str, str]
) -> str:
    """Each query so far with the titles of its top results."""
    blocks = []
    for n, (query, run) in enumerate(zip(queries, runs), start=1):
        lines = [f'{n}. "{query}"']
        shown = [f"   - {titles.get(d, d)}" for d in run[:TITLES_PER_QUERY]]
        blocks.append("\n".join(lines + (shown or ["   (no results)"])))
    return "\n".join(blocks)


class AgentRetriever:
    """B5 (and B5-U, B5-D): search, look at titles, decide the next query or stop."""

    def __init__(
        self,
        base: Ranker,
        agent: ChatModel,
        *,
        titles: Mapping[str, str],
        max_queries: int,
        max_llm_calls: int,
        prompt: Prompt,
        name: str = "B5",
    ) -> None:
        self.name = name
        self.base = base
        self.agent = agent
        self.titles = titles
        self.max_queries = max_queries
        self.max_llm_calls = max_llm_calls
        self.prompt = prompt

    def retrieve(self, query: str) -> Result:
        start = time.perf_counter()
        trace = Trace(prompt_hashes={self.prompt.name: self.prompt.sha256})
        queries, runs = [query], [self.base.rank(query)]  # the proposal itself, no LLM call
        seen = {normalize_query(query)}
        budget_used, calls = 1, 0
        while True:
            if budget_used >= self.max_queries:
                trace.stop_reason = "max_queries"
                break
            if calls >= self.max_llm_calls:
                trace.stop_reason = "max_llm_calls"
                break
            text = self.prompt.render(
                proposal=query,
                history=render_history(queries, runs, self.titles),
                queries_left=str(self.max_queries - budget_used),
            )
            reply = self.agent.complete([{"role": "user", "content": text}])
            calls += 1
            trace.add_call(reply)
            try:
                action = parse_action(reply.text)
            except (ValueError, TypeError) as exc:
                trace.errors.append(f"agent_parse_error: {exc}; reply: {reply.text[:200]!r}")
                trace.stop_reason = "parse_error"
                break
            if action["action"] == "stop":
                trace.steps.append(action)
                trace.stop_reason = "agent_stop"
                break
            budget_used += 1  # repeats count too
            repeat = normalize_query(action["query"]) in seen
            trace.steps.append({**action, "repeat": repeat})
            if not repeat:
                seen.add(normalize_query(action["query"]))
                queries.append(action["query"])
                runs.append(self.base.rank(action["query"]))
        trace.queries = queries
        trace.budget_used = budget_used
        ranking = retrieval.fuse_runs(runs)  # exactly B4's fusion
        trace.latency_s = time.perf_counter() - start
        return Result(ranking, trace)


def make_retrievers(
    cfg: Config, index: Index, encoder: Encoder, agent: ChatModel
) -> dict[str, Retriever]:
    """All eight configurations (B1-B5, B5-U, and B4-D, B5-D on B2), budgets from the config."""
    bm25 = BM25Retriever(index)
    dense = DenseRetriever(index, encoder)
    hybrid = HybridRetriever(bm25, dense)
    rewrite, step = load_prompt("rewrite"), load_prompt("agent_step")
    r = cfg.retrieval

    def agent_on(base: Ranker, name: str, max_queries: int, max_llm_calls: int) -> AgentRetriever:
        return AgentRetriever(
            base,
            agent,
            titles=index.titles,
            max_queries=max_queries,
            max_llm_calls=max_llm_calls,
            prompt=step,
            name=name,
        )

    return {
        "B1": bm25,
        "B2": dense,
        "B3": hybrid,
        "B4": RewriteRetriever(hybrid, agent, max_queries=r.max_queries, prompt=rewrite),
        "B5": agent_on(hybrid, "B5", r.max_queries, r.max_llm_calls_agent),
        "B5-U": agent_on(
            hybrid, "B5-U", r.max_queries_unconstrained, r.max_llm_calls_agent_unconstrained
        ),
        "B4-D": RewriteRetriever(
            dense, agent, max_queries=r.max_queries, prompt=rewrite, name="B4-D"
        ),
        "B5-D": agent_on(dense, "B5-D", r.max_queries, r.max_llm_calls_agent),
    }
