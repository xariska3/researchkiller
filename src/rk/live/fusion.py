"""Final-fusion variants of the live pipeline (docs/phase2_spec.md, Step 2 decisions).

L5 ranks its final pool with B5's fusion: RRF over every query's hybrid
ranking of the pool, so once the agent has run 4 or 5 queries the proposal's
own ranking is one vote among them. The pre-declared variants give the
proposal more say (n is the number of queries run, the proposal first):

- b5: B5's fusion, every query weight 1 (L5 itself).
- proposal-equal (F1): weighted RRF; the proposal weighs n - 1, as much as all
  agent queries combined (1 if the agent ran none), each agent query 1.
- proposal-2x (F2): weighted RRF; the proposal weighs 2, each agent query 1.
- l1-l5 (F3): RRF of L1's list (the proposal's own search reranked on its
  own, mapped onto the final pool's merged works) and the b5 list.
- proposal-only (F4): the final pool ranked by the proposal alone.

Every list is at most rk.retrieval.TOP_K long, as B5's is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rk.live.openalex_search import SEMANTIC
from rk.live.pool import CandidatePool
from rk.retrieval import RRF_K, TOP_K, Encoder, fuse_runs, rrf
from rk.targets import Record, id_number

B5 = "b5"
PROPOSAL_EQUAL = "proposal-equal"
PROPOSAL_2X = "proposal-2x"
L1_L5 = "l1-l5"
PROPOSAL_ONLY = "proposal-only"
VARIANTS = (PROPOSAL_EQUAL, PROPOSAL_2X, L1_L5, PROPOSAL_ONLY)  # F1-F4, in declared order
FUSIONS = (B5, *VARIANTS)


def weighted_rrf(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float],
    *,
    k: int = RRF_K,
    top_k: int = TOP_K,
) -> list[str]:
    """RRF with a weight per ranking: score = sum of w / (k + rank); ties by doc id.

    With every weight 1 it is rk.retrieval.rrf exactly.
    """
    if len(rankings) != len(weights):
        raise ValueError(f"{len(rankings)} rankings but {len(weights)} weights")
    terms: dict[str, list[float]] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            terms.setdefault(doc_id, []).append(weight / (k + rank))
    scores = {doc_id: math.fsum(t) for doc_id, t in terms.items()}
    return sorted(scores, key=lambda d: (-scores[d], id_number(d)))[:top_k]


def proposal_weights(n_queries: int, fusion: str) -> list[float]:
    """The weights of F1 and F2: the proposal's first, then 1 for each agent query."""
    agent = n_queries - 1
    if fusion == PROPOSAL_EQUAL:
        first = float(max(1, agent))
    elif fusion == PROPOSAL_2X:
        first = 2.0
    else:
        raise ValueError(f"{fusion!r} is not a weighted fusion")
    return [first] + [1.0] * agent


def onto_pool(ranking: Sequence[str], pool: CandidatePool) -> list[str]:
    """Another pool's ranking as the final pool's works (each id or alias to its work's id)."""
    owner = {w: r["id"] for r in pool.records for w in (r["id"], *r["aliases"])}
    return list(dict.fromkeys(owner[d] for d in ranking if d in owner))


def l1_ranking(
    records: Sequence[Record], proposal: str, encoder: Encoder, *, max_size: int
) -> list[str]:
    """L1's ranking: the proposal's own search results, pooled and reranked on their own."""
    pool = CandidatePool(encoder, max_size=max_size)
    pool.add(records, query_index=0, kind=SEMANTIC)
    return pool.rank(proposal)


def final_ranking(
    pool: CandidatePool,
    queries: Sequence[str],
    fusion: str,
    *,
    l1: Sequence[str] = (),
) -> list[str]:
    """The final pool ranked by `fusion`; `queries` start with the proposal.

    `l1` is L1's ranking, which only l1-l5 uses.
    """
    runs = [pool.rank(q) for q in queries]
    if fusion == B5:
        return fuse_runs(runs)
    if fusion in (PROPOSAL_EQUAL, PROPOSAL_2X):
        return weighted_rrf(runs, proposal_weights(len(runs), fusion))
    if fusion == L1_L5:
        return rrf([onto_pool(l1, pool), fuse_runs(runs)])
    if fusion == PROPOSAL_ONLY:
        return runs[0]
    raise ValueError(f"unknown fusion {fusion!r}; choose from {list(FUSIONS)}")
