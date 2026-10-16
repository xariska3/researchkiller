"""Two lists (issue #22): one rerank call, list A closest in contribution, list B closest
in methods; the explainer on both tops; the report's two sections.

The checks run on FakeLiveOpenAlex with test_live_eval's FakeLLM agent and FakeEncoder;
the reranker is test_rerank's two-list FakeLLM and the explainer test_explain's.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from bench_data import GOOD, T1
from conftest import FakeEncoder, FakeLLM
from live_fakes import FakeLiveOpenAlex
from test_explain import config_dict, explainer_reply, works
from test_l5r import down, ids, l5w_check
from test_live_eval import _search
from test_live_eval import agent as fake_agent
from test_rerank import CONTRIBUTION_LINE, METHODS_LINE, original_first, two_lists_reply

from rk.config import Config, ConfigError, config_from_dict
from rk.live.check import CheckRun, run_check
from rk.live.explain import render_prompt as explain_prompt
from rk.live.fusion import B5, final_ranking
from rk.live.pipeline import (
    DEFAULT_FUSION,
    FALLBACK,
    RERANKED,
    SKIPPED,
    default_check,
    with_rerank,
)
from rk.live.report import (
    ALSO_IN_METHODS,
    CONTRIBUTION_HEADING,
    CONTRIBUTION_NOTE,
    METHODS_HEADING,
    METHODS_NOTE,
    NO_METHODS,
    CheckReport,
    CheckRequest,
    IdeaSummary,
    render_markdown,
    render_text,
)
from rk.live.reranker import (
    LISTWISE,
    TWO_LISTS,
    TWO_LISTS_RESPONSE_FORMAT,
    TwoLists,
    candidates,
    parse_two_lists,
    render_prompt,
)
from rk.llm import LLMError, load_prompt

TOP_A, TOP_B = 3, 2  # the test config explains list A's top 3 and list B's top 2


def two_lists_dict(tmp_path: Path, **explain: Any) -> dict[str, Any]:
    data = config_dict(tmp_path, **{"contribution_top_k": TOP_A, "methods_top_k": TOP_B, **explain})
    data["rerank"]["two_lists"] = True
    return data


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return config_from_dict(two_lists_dict(tmp_path))


def check(cfg: Config, *, reranker: Any = two_lists_reply, **kwargs: Any) -> CheckRun:
    explainer = kwargs.pop("explainer", None) or FakeLLM(explainer_reply)
    return run_check(
        CheckRequest(text=GOOD[T1], **kwargs),
        cfg,
        search=_search(cfg, FakeLiveOpenAlex(works())),
        encoder=FakeEncoder(),
        agent=fake_agent(),
        reranker=FakeLLM(reranker),
        explainer=explainer,
    )


# --- the reply ---------------------------------------------------------------------------------


def test_a_two_list_reply_gives_two_lines_and_two_orders() -> None:
    reply = {
        "contribution": " Same\n problem,  same claim. ",
        "methods": "",
        "contribution_ranking": [2, "1", 2, 9, 0],
        "methods_ranking": [3, 1],
    }
    assert parse_two_lists(json.dumps(reply), 3) == TwoLists(
        contribution="Same problem, same claim.",
        methods=None,  # empty: no line
        contribution_order=[2, 1],  # repeats and unknown numbers dropped, as R1's
        methods_order=[3, 1],
    )
    partial = parse_two_lists('{"contribution_ranking": [1], "methods_ranking": []}', 3)
    assert partial == TwoLists(None, None, [1], None)
    assert parse_two_lists("no JSON here", 3) is None
    assert parse_two_lists("[1, 2]", 3) is None  # not an object


def test_the_prompt_states_the_idea_then_ranks_twice_in_a_strict_schema() -> None:
    schema = TWO_LISTS_RESPONSE_FORMAT["json_schema"]
    assert schema["strict"] is True
    properties = schema["schema"]["properties"]
    assert list(properties) == [
        "contribution", "methods", "contribution_ranking", "methods_ranking"
    ]  # the lines first, so the model states the idea before it ranks
    assert schema["schema"]["required"] == list(properties)
    assert schema["schema"]["additionalProperties"] is False
    text = load_prompt(TWO_LISTS).render(n="2", proposal="P", candidates="[1] Title: T")
    assert "rank all 2 candidates twice" in text
    assert "shares only tools, datasets or software with the proposal" in text
    assert text.index("First, state in one line each") < text.index("contribution_ranking:")
    assert text.index("contribution_ranking:") < text.index("methods_ranking:")


# --- the rerank step -------------------------------------------------------------------------


def test_one_call_gives_list_a_and_list_b(cfg: Config) -> None:
    assert cfg.rerank is not None and cfg.live is not None
    l5w = l5w_check(cfg)
    reranker = FakeLLM(two_lists_reply)
    two = with_rerank(l5w, cfg, reranker, FakeEncoder(), two_lists=True)
    # The two-list prompt over R1's candidates, with its schema.
    pool, queries = l5w.pool, l5w.trace.queries
    l5w_order = final_ranking(pool, queries, DEFAULT_FUSION)
    cands = candidates(l5w_order, final_ranking(pool, queries, B5), cfg.rerank.depth)
    by_id = {r["id"]: r for r in pool.records}
    (call,) = reranker.calls
    records = [by_id[d] for d in cands]
    prompt = render_prompt(load_prompt(TWO_LISTS), GOOD[T1], records, cfg.rerank.abstract_chars)
    assert call == [{"role": "user", "content": prompt}]
    assert reranker.response_formats == [TWO_LISTS_RESPONSE_FORMAT]
    # List A: the reply's first order, L5w's other works after it; list B: the second.
    reply = json.loads(two_lists_reply(call))
    list_a = [cands[n - 1] for n in reply["contribution_ranking"]]
    list_b = [cands[n - 1] for n in reply["methods_ranking"]]
    top_k = cfg.live.top_k
    assert ids(two) == list(dict.fromkeys([*list_a, *l5w_order]))[:top_k]
    assert two.methods is not None and len(cands) > 2
    assert [c.record["id"] for c in two.methods] == list_b[:top_k]
    assert [c.rank for c in two.methods] == list(range(1, len(two.methods) + 1))
    assert list_b[0] == list_a[1]  # the fake's list B starts with list A's second
    step = two.rerank
    assert step is not None and (step.status, step.methods_status) == (RERANKED, RERANKED)
    assert (step.contribution_line, step.methods_line) == (CONTRIBUTION_LINE, METHODS_LINE)
    assert two.warnings == () and step.warning is None
    assert two.ledger.llm.calls == l5w.ledger.llm.calls + 1  # one call for both lists
    assert two.to_dict()["methods"][0]["id"] == list_b[0]


def test_one_list_has_no_methods(cfg: Config) -> None:
    one = with_rerank(l5w_check(cfg), cfg, FakeLLM(two_lists_reply), FakeEncoder())
    assert one.methods is None and one.to_dict()["methods"] is None
    assert one.rerank is not None and one.rerank.methods_status is None


def test_an_unreadable_list_a_keeps_the_check_order_and_no_methods_list(cfg: Config) -> None:
    l5w = l5w_check(cfg)

    def methods_only(messages: list[dict[str, str]]) -> str:
        reply = json.loads(two_lists_reply(messages))
        reply["contribution_ranking"] = []
        return json.dumps(reply)

    for reranker in (FakeLLM(["Sorry, I can't rank these."]), FakeLLM(methods_only)):
        two = with_rerank(l5w, cfg, reranker, FakeEncoder(), two_lists=True)
        assert ids(two) == ids(l5w) and two.methods == []
        assert two.warnings == (
            (
                "Not reranked (the rerank reply had no readable contribution ranking); the "
                "contribution list keeps the check's order, and there is no methods list."
            ),
        )
        assert two.rerank is not None
        assert (two.rerank.status, two.rerank.methods_status) == (FALLBACK, FALLBACK)


def test_a_readable_list_a_without_list_b_keeps_list_a(cfg: Config) -> None:
    def contribution_only(messages: list[dict[str, str]]) -> str:
        reply = json.loads(two_lists_reply(messages))
        del reply["methods_ranking"]
        return json.dumps(reply)

    l5w = l5w_check(cfg)
    two = with_rerank(l5w, cfg, FakeLLM(contribution_only), FakeEncoder(), two_lists=True)
    full = with_rerank(l5w, cfg, FakeLLM(two_lists_reply), FakeEncoder(), two_lists=True)
    assert ids(two) == ids(full) and two.methods == []
    assert two.warnings == ("No methods list (the rerank reply had no readable methods ranking).",)
    assert two.rerank is not None
    assert (two.rerank.status, two.rerank.methods_status) == (RERANKED, FALLBACK)


def test_a_failed_or_capped_call_leaves_no_methods_list(cfg: Config) -> None:
    l5w = l5w_check(cfg)
    failed = with_rerank(l5w, cfg, FakeLLM(down), FakeEncoder(), two_lists=True)
    assert ids(failed) == ids(l5w) and failed.methods == []
    assert "the rerank call failed: upstream 503" in failed.warnings[0]
    assert failed.warnings[0].endswith("and there is no methods list.")
    with pytest.raises(LLMError, match="upstream 503"):
        with_rerank(l5w, cfg, FakeLLM(down), FakeEncoder(), strict=True, two_lists=True)
    full = replace(l5w, ledger=replace(l5w.ledger, cap=l5w.ledger.method_cost))  # no room
    never = FakeLLM([])
    capped = with_rerank(full, cfg, never, FakeEncoder(), strict=True, two_lists=True)
    assert never.calls == [] and capped.methods == []
    assert "the rerank call could cost $" in capped.warnings[0]


def test_a_check_without_candidates_has_an_empty_methods_list(cfg: Config) -> None:
    never = FakeLLM([])
    empty = with_rerank(
        l5w_check(cfg, FakeLiveOpenAlex([])), cfg, never, FakeEncoder(), two_lists=True
    )
    assert never.calls == [] and empty.methods == [] and empty.warnings == ()
    assert empty.rerank is not None
    assert (empty.rerank.status, empty.rerank.methods_status) == (SKIPPED, SKIPPED)


def test_the_default_check_has_two_lists_when_the_config_says_so(cfg: Config) -> None:
    assert cfg.rerank is not None

    def run(config: Config) -> Any:
        return default_check(
            GOOD[T1],
            config,
            search=_search(config, FakeLiveOpenAlex()),
            encoder=FakeEncoder(),
            agent=fake_agent(),
            reranker=FakeLLM(two_lists_reply),
        )

    assert run(cfg).methods is not None
    one_list = replace(cfg, rerank=replace(cfg.rerank, two_lists=False))
    assert run(one_list).methods is None  # L5r, whose reply this fake can't give: fallback


# --- the whole check ---------------------------------------------------------------------------


def test_a_two_list_check_explains_both_tops_in_one_call(cfg: Config) -> None:
    assert cfg.explain is not None
    explainer = FakeLLM(explainer_reply)
    run = check(cfg, explainer=explainer)
    report, list_a, list_b = run.report, run.result.candidates, run.result.methods
    assert list_b is not None
    # One call: list A's top 3, then list B's top 2 not already among them (its first is
    # list A's second), each once.
    (call,) = explainer.calls
    a_ids = {c.record["id"] for c in list_a[:TOP_A]}
    explained = [*list_a[:TOP_A], *(c for c in list_b[:TOP_B] if c.record["id"] not in a_ids)]
    assert len(explained) == TOP_A + 1
    records = [c.record for c in explained]
    assert call[0]["content"] == explain_prompt(GOOD[T1], records, cfg.explain.abstract_chars)
    # Section A: list A, its top explained; section B: list B's top, in its order.
    assert [p.id for p in report.papers] == [c.record["id"] for c in list_a]
    assert all(p.explanation is not None and p.snippet is not None for p in report.papers[:TOP_A])
    assert all(p.explanation is None and p.snippet is None for p in report.papers[TOP_A:])
    assert report.methods is not None
    assert [p.id for p in report.methods] == [c.record["id"] for c in list_b[:TOP_B]]
    assert [p.rank for p in report.methods] == [1, 2]
    shared, only_b = report.methods
    assert shared.id == report.papers[1].id and shared.rank == 1 and report.papers[1].rank == 2
    assert (shared.explanation, shared.snippet) == (
        report.papers[1].explanation,
        report.papers[1].snippet,
    )
    assert only_b.explanation is not None and not report.in_methods(report.papers[0])
    assert report.in_methods(report.papers[1]) and len(report.explained()) == TOP_A + 1
    assert report.idea == IdeaSummary(contribution=CONTRIBUTION_LINE, methods=METHODS_LINE)
    assert TWO_LISTS in report.prompt_hashes and "rerank_schema" in report.prompt_hashes
    assert LISTWISE not in report.prompt_hashes and report.warnings == []
    assert CheckReport.model_validate_json(report.model_dump_json()) == report
    # The terminal: section A, its "also found", then section B; one paper in both.
    text = render_text(report)
    order = [CONTRIBUTION_HEADING, CONTRIBUTION_NOTE, "Also found", METHODS_HEADING, METHODS_NOTE]
    assert [text.index(part) for part in order] == sorted(text.index(part) for part in order)
    assert text.count(f"[{ALSO_IN_METHODS}]") == 1
    # The Markdown export: the same order, the sections second-level, the papers third.
    md = render_markdown(report)
    assert f"## {CONTRIBUTION_HEADING}\n\n{CONTRIBUTION_NOTE}\n" in md
    assert f"## {METHODS_HEADING}\n\n{METHODS_NOTE}\n" in md
    assert md.index(CONTRIBUTION_HEADING) < md.index("### Also found") < md.index(METHODS_HEADING)
    assert md.count(f"*{ALSO_IN_METHODS}*") == 1 and "### 1. [" in md
    assert "—" not in md and "—" not in text


def test_a_missing_explanation_of_a_list_b_paper_names_its_list(cfg: Config) -> None:
    def all_but_the_last(messages: list[dict[str, str]]) -> str:
        papers = json.loads(explainer_reply(messages))["papers"]
        return json.dumps({"papers": papers[:-1]})

    report = check(cfg, explainer=FakeLLM(all_but_the_last)).report
    assert report.methods is not None
    only_b = report.methods[1]
    assert only_b.explanation is None and only_b.snippet is not None
    assert report.warnings == [
        (
            f"No explanation for the paper(s) at rank {only_b.rank} of the methods list; "
            f"they show their snippets only."
        )
    ]


def test_without_list_b_the_report_says_so(cfg: Config) -> None:
    def contribution_only(messages: list[dict[str, str]]) -> str:
        reply = json.loads(two_lists_reply(messages))
        reply["methods_ranking"] = "none"
        return json.dumps(reply)

    report = check(cfg, reranker=contribution_only).report
    assert report.methods == [] and report.papers
    assert f"{METHODS_HEADING}\n{METHODS_NOTE}\n\n{NO_METHODS}" in render_text(report)
    assert f"## {METHODS_HEADING}\n\n{METHODS_NOTE}\n\n{NO_METHODS}" in render_markdown(report)


def test_the_quick_check_and_one_list_checks_are_unchanged(cfg: Config, tmp_path: Path) -> None:
    quick = check(cfg, quick=True).report
    assert quick.methods is None and quick.idea is None
    assert CONTRIBUTION_HEADING not in render_text(quick)
    one_list = config_from_dict(config_dict(tmp_path))  # rerank.two_lists defaults to false
    assert one_list.rerank is not None and one_list.rerank.two_lists is False
    report = check(one_list, reranker=original_first).report
    assert report.methods is None and report.idea is None and report.rerank == RERANKED
    assert LISTWISE in report.prompt_hashes and TWO_LISTS not in report.prompt_hashes
    for rendered in (render_text(report), render_markdown(report)):
        assert CONTRIBUTION_HEADING not in rendered and METHODS_HEADING not in rendered


def test_the_explain_counts_are_checked(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="contribution_top_k"):
        config_from_dict(two_lists_dict(tmp_path, contribution_top_k=0))
    with pytest.raises(ConfigError, match="methods_top_k"):
        config_from_dict(two_lists_dict(tmp_path, methods_top_k=0))
    data = two_lists_dict(tmp_path)
    data["explain"]["methods_top_k"] = 11  # the test config's live.top_k is 10
    with pytest.raises(ConfigError, match=r"must be <= live.top_k \(10\)"):
        config_from_dict(data)
