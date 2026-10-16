"""Phase 2 Step 4, the LLM rerank variants (python -m rk rerank): candidates, parsing,
the variants' rankings, the selection rule, and whole runs on a replayed live-eval run.

The live-eval run is one on the fixture benchmark (FakeLiveOpenAlex, FakeLLM agent,
FakeEncoder), as in test_live_eval; the rerankers are FakeLLMs, directly or behind a
real LLMClient through httpx.MockTransport.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from bench_data import GOOD, T1, T2, T3, TITLES, corpus_records, judge_reply, target_rows
from conftest import TEST_API_KEY_ENV, FakeEncoder, FakeLLM
from live_fakes import FakeLiveOpenAlex
from test_live_eval import B1_RANKS, B5_RANKS, _config_dict, _paraphraser, _phase1_run, agent

from rk import cli
from rk.bench import make_bench
from rk.config import Config, ConfigError, config_from_dict, load_config
from rk.corpus import freeze
from rk.evaluate import RUN_FILE, SUMMARY_JSON, SUMMARY_MD, EvalError
from rk.live import evaluate as live_evaluate
from rk.live import rerank as live_rerank
from rk.live.evaluate import run_live_eval
from rk.live.fusion import PROPOSAL_EQUAL
from rk.live.openalex_search import LiveSearchClient, OpenAlexCache
from rk.live.rerank import (
    CALLS,
    L5,
    L5W,
    PRODUCT_MODEL,
    R1,
    R2,
    R3,
    R4,
    RESULTS_FILE,
    RS,
    TWO_LISTS_MIN_HIT_AT_5,
    UPPER_MODEL,
    fused_ranking,
    graded_ranking,
    parse_scores,
    run_rerank,
    select_rerank,
)
from rk.live.reranker import (
    TWO_LISTS_RESPONSE_FORMAT,
    candidates,
    listwise_ranking,
    parse_order,
    render_candidates,
)
from rk.llm import BudgetExceeded, LLMError, load_prompt
from rk.retrieval import rrf

CANDIDATE_RE = re.compile(r"^\[(\d+)\] Title: (.*)$", re.MULTILINE)
GRADED_MARK = '{"scores":'  # only the graded prompt's answer format has it
CHOSEN = live_rerank.RERANK_CHOICE  # as committed, before any test patches it


@pytest.fixture(autouse=True)
def _no_choice_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs here have no L5r unless a test chooses one itself."""
    monkeypatch.setattr(live_rerank, "RERANK_CHOICE", None)


def test_l5r_is_the_variant_the_dev_set_chose() -> None:
    assert CHOSEN == R1  # docs/phase2_spec.md, runs/rerank-20261015-132644


# --- candidates and the prompt ------------------------------------------------------------


def test_candidates_take_l5w_and_l5_in_turn_without_repeats() -> None:
    l5w = ["W1", "W2", "W3", "W4"]
    l5 = ["W2", "W5", "W1", "W6"]
    assert candidates(l5w, l5, 3) == ["W1", "W2", "W5", "W3"]  # W2 and W1 only once
    assert candidates(l5w, l5, 10) == ["W1", "W2", "W5", "W3", "W4", "W6"]
    assert candidates(["W1"], [], 30) == ["W1"]  # a small pool gives fewer


def test_the_prompt_shows_numbers_titles_and_cut_abstracts_only() -> None:
    records = [
        {"id": "W1", "title": "First", "abstract": "one two three four", "publication_date": "2026-02-01"},
        {"id": "W2", "title": "Second", "abstract": "short", "publication_date": "2025-01-01"},
    ]
    text = render_candidates(records, abstract_chars=9)
    assert text == "[1] Title: First\nAbstract: one two …\n\n[2] Title: Second\nAbstract: short"
    prompt = load_prompt("rerank_listwise").render(n="2", proposal="P", candidates=text)
    assert "all 2 candidates" in prompt and "2026" not in prompt and "W1" not in prompt
    graded = load_prompt("rerank_graded").render(n="2", proposal="P", candidates=text)
    assert "from 1 to 2" in graded


# --- replies -------------------------------------------------------------------------------


def test_parse_order_keeps_known_numbers_once_in_order() -> None:
    assert parse_order('{"ranking": [3, 1, "2", 3, 9, 0, true]}', 3) == [3, 1, 2]
    assert parse_order("Here you go:\n```json\n[2, 1]\n```", 3) == [2, 1]
    assert parse_order('{"ranking": [2.0, 1]}', 3) == [2, 1]
    assert parse_order("no idea", 3) is None
    assert parse_order('{"ranking": []}', 3) is None  # nothing usable
    assert parse_order('{"order": [1, 2]}', 3) is None


def test_parse_scores_keeps_valid_scores_by_number() -> None:
    assert parse_scores('{"scores": {"1": 3, "2": 0, "3": 7, "4": 1, "x": 2}}', 3) == {1: 3, 2: 0}
    assert parse_scores('{"2": "3", "1": 1}', 3) == {2: 3, 1: 1}
    assert parse_scores("[3, 2]", 3) is None and parse_scores("nothing", 3) is None


def test_the_variants_rank_the_candidates() -> None:
    cands = ["W1", "W2", "W3", "W4"]
    # R1: the reply's order; numbers it left out follow in presented order.
    assert listwise_ranking(cands, [3, 1]) == ["W3", "W1", "W2", "W4"]
    assert listwise_ranking(cands, None) == cands  # unreadable: the presented order
    # R2: RRF of R1's order and the presented order.
    assert fused_ranking(cands, [4, 3, 2, 1]) == rrf([["W4", "W3", "W2", "W1"], cands])
    # R3: by score, ties and missing scores (0) in presented order.
    assert graded_ranking(cands, {2: 3, 4: 3, 1: 1}) == ["W2", "W4", "W1", "W3"]
    assert graded_ranking(cands, None) == cands


def test_select_rerank_needs_a_variant_above_l5w() -> None:
    def q(hit5: float, mrr: float) -> dict[str, float]:
        return {"hit_at_5": hit5, "mrr": mrr}

    quality = {L5W: q(0.70, 0.60), R1: q(0.75, 0.61), R2: q(0.75, 0.63), R3: q(0.72, 0.70)}
    assert select_rerank(quality) == R2  # ties on Hit@5 go to MRR
    quality[R2] = q(0.75, 0.61)
    assert select_rerank(quality) == R1  # then to the declared order
    assert select_rerank({**quality, L5W: q(0.75, 0.50)}) is None  # none is above L5w
    assert select_rerank({**quality, RS: q(0.99, 0.99)}) == R1  # RS is never a candidate


# --- the config ------------------------------------------------------------------------------


def rerank_dict(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    model = {
        "name": "fake/reranker",
        "family": "fakefam",
        "base_url": "http://llm.test/v1",
        "api_key_env": TEST_API_KEY_ENV,
        "price_in_per_m": 1.0,
        "price_out_per_m": 2.0,
        "temperature": 0.0,
        "max_tokens": 200,
        "seed": 42,
        "training_cutoff": "2024-01-01",
    }
    upper = {**model, "name": "fake/upper", "temperature": 1.0, "training_cutoff": "2026-06-30"}
    return {
        "model": model,
        "upper_model": upper,
        "upper_cache_path": str(tmp_path / "upper.sqlite"),
        "upper_cost_cap": 1.0,
        **overrides,
    }


def test_the_product_reranker_must_predate_the_originals(tmp_path: Path) -> None:
    data = _config_dict(tmp_path)
    cfg = config_from_dict({**data, "rerank": rerank_dict(tmp_path)})
    assert cfg.rerank is not None and cfg.rerank.depth == 30
    late = rerank_dict(tmp_path)
    late["model"] = {**late["model"], "training_cutoff": "2026-06-30"}
    with pytest.raises(ConfigError, match="rerank"):
        config_from_dict({**data, "rerank": late})
    with pytest.raises(ConfigError, match="upper_cost_cap"):
        config_from_dict({**data, "rerank": rerank_dict(tmp_path, upper_cost_cap=0)})


def test_the_repo_config_declares_the_spec_variants() -> None:
    cfg = load_config("configs/phase1.yaml")
    assert cfg.rerank is not None
    rr = cfg.rerank
    assert (rr.depth, rr.abstract_chars) == (30, 2000)
    # The product reranker is the agent model on the agent's endpoint.
    assert (rr.model.name, rr.model.provider, rr.model.temperature, rr.model.seed) == (
        cfg.agent.name,
        cfg.agent.provider,
        0.0,
        42,
    )
    assert (rr.upper_model.name, rr.upper_model.base_url) == (
        "claude-sonnet-5-5",
        "https://api.anthropic.com/v1",
    )
    assert rr.upper_model.temperature == 1.0  # Claude 5 models refuse a lower one
    assert rr.upper_model.training_cutoff == date(2026, 6, 30)
    assert (rr.upper_cost_cap, str(rr.upper_cache_path)) == (
        20.0,
        str(Path("data/cache/rerank_sonnet.sqlite")),
    )
    assert rr.upper_cache_path != cfg.run.cache_path  # its own ledger


# --- whole runs ------------------------------------------------------------------------------


PUBLISHED = {T1: "2026-03-01", T2: "2026-08-01", T3: "2026-05-01"}  # T2 after July 2026


def dated_corpus() -> list[dict[str, Any]]:
    """The fixture corpus with publication dates, as the real frozen corpus has them."""
    return [
        {**r, "publication_date": PUBLISHED.get(r["id"], "2025-01-01")} for r in corpus_records()
    ]


@pytest.fixture
def live_run(tmp_path: Path) -> tuple[Config, Path]:
    """The fixture benchmark, a Phase 1 run, and a live-eval run with L1, L5 and L5w."""
    data = _config_dict(tmp_path)
    config = config_from_dict({**data, "rerank": rerank_dict(tmp_path)})
    freeze(config.corpus.out_dir, dated_corpus(), target_rows(), {})
    stats = make_bench(config, _paraphraser(), FakeLLM(lambda messages: judge_reply()))
    assert stats["accepted"] == 3
    _phase1_run(config, B1_RANKS, B5_RANKS)
    assert live_evaluate.L5W_FUSION == PROPOSAL_EQUAL and config.live is not None
    search = LiveSearchClient(
        config.live,
        OpenAlexCache(config.live.cache_path),
        max_age=None,
        transport=FakeLiveOpenAlex().transport,
        sleep=lambda seconds: None,
        clock=lambda: 0.0,
    )
    done = run_live_eval(
        config, agent(), FakeEncoder(), search, configs=["L1", "L5", "L5w"], log_mlflow=False
    )
    return config, done.run_dir


def numbered(prompt: str) -> dict[str, int]:
    """Candidate title -> its number in a rerank prompt."""
    return {title: int(n) for n, title in CANDIDATE_RE.findall(prompt)}


def original_first(messages: list[dict[str, str]]) -> str:
    """A reply that puts the proposal's original (by its title) first, the rest in order."""
    prompt = messages[-1]["content"]
    target = next(t for t in TITLES if GOOD[t] in prompt)
    numbers = numbered(prompt)
    first = [numbers[TITLES[target]]] if TITLES[target] in numbers else []
    rest = [n for n in sorted(numbers.values()) if n not in first]
    if GRADED_MARK in prompt:  # the graded prompt: the original 3, the others 1
        return json.dumps({"scores": {str(n): 3 if n in first else 1 for n in numbers.values()}})
    return json.dumps({"ranking": first + rest})


TWO_LISTS_MARK = '"contribution_ranking":'  # only the two-list prompt's answer format has it
CONTRIBUTION_LINE = "Sparse routing for long inputs."
METHODS_LINE = "Clustered attention on arXiv and PubMed."


def two_lists_reply(messages: list[dict[str, str]]) -> str:
    """A two-list reply. List A is original_first's order; list B starts with list A's
    second paper, then the others from the last, and the original last."""
    order = json.loads(original_first(messages))["ranking"]
    return json.dumps(
        {
            "contribution": " Sparse  routing\nfor long inputs. ",  # read as CONTRIBUTION_LINE
            "methods": METHODS_LINE,
            "contribution_ranking": order,
            "methods_ranking": order[1:2] + order[:1:-1] + order[:1],
        }
    )


def any_reranker(messages: list[dict[str, str]]) -> str:
    """The two-list reply to the two-list prompt, original_first's to the others."""
    if TWO_LISTS_MARK in messages[-1]["content"]:
        return two_lists_reply(messages)
    return original_first(messages)


def run(cfg: Config, run_dir: Path, clients: dict[str, Any], **kwargs: Any) -> Any:
    return run_rerank(cfg, run_dir, encoder=FakeEncoder(), clients=clients, **kwargs)


def results(done: Any) -> list[dict[str, Any]]:
    lines = (done.run_dir / RESULTS_FILE).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_a_run_reranks_every_replayed_check(live_run: tuple[Config, Path]) -> None:
    cfg, run_dir = live_run
    product, upper = FakeLLM(original_first), FakeLLM(original_first)
    done = run(cfg, run_dir, {PRODUCT_MODEL: product, UPPER_MODEL: upper})
    summary = done.summary
    assert summary["n_queries"] == 3
    assert summary["reproduced"] == {L5: 3, L5W: 3, "L5w_rows": 3}
    # R1 and R2 read one listwise reply: two product calls per check, one upper call.
    assert len(product.calls) == 6 and len(upper.calls) == 3
    prompts = [c[-1]["content"] for c in product.calls]
    assert sum(GRADED_MARK in p for p in prompts) == 3
    rows = results(done)
    assert len(rows) == 3 * 6 and {r["status"] for r in rows} == {"ok"}
    by = {(r["config"], r["target_id"]): r for r in rows}
    assert any(by[(L5W, t)]["original_in_candidates"] for t in (T1, T2, T3))
    for target in (T1, T2, T3):
        l5w = by[(L5W, target)]
        if l5w["original_in_candidates"]:
            for variant in (R1, R3, RS):
                assert by[(variant, target)]["target_rank"] == 1
            assert by[(R2, target)]["target_rank"] is not None
        for variant in (R1, R2, R3, RS):
            row = by[(variant, target)]
            assert row["readable"] and row["cost"] > 0
            assert row["total_cost"] == pytest.approx(row["check_cost"] + row["cost"])
            assert row["published"] == l5w["published"]
    meta = json.loads((done.run_dir / RUN_FILE).read_text(encoding="utf-8"))
    assert meta["kind"] == "rerank" and meta["live_run"]["dir"] == str(run_dir)
    assert set(meta["prompts"]) == {"rerank_listwise", "rerank_graded"}
    assert json.loads((done.run_dir / SUMMARY_JSON).read_text(encoding="utf-8"))["variants"] == [
        R1,
        R2,
        R3,
        RS,
    ]
    assert summary["selected"] == select_rerank(summary["quality"])
    assert [p["a"] for p in summary["vs_l5w"]] == [R1, R2, R3, RS]
    assert summary["decides"] is None  # no L5r chosen yet
    parts = summary["by_publication"]
    assert [parts[p][RS]["n"] for p in ("before 2026-07-01", "from 2026-07-01")] == [2, 1]
    markdown = (done.run_dir / SUMMARY_MD).read_text(encoding="utf-8")
    assert "L5 reproduced in 3 of 3, L5w in 3 of 3 recorded" in markdown


def test_unreadable_replies_keep_the_presented_order(live_run: tuple[Config, Path]) -> None:
    cfg, run_dir = live_run
    done = run(cfg, run_dir, {PRODUCT_MODEL: FakeLLM(lambda m: "sorry")}, variants=[R1, R3])
    rows = results(done)
    by = {(r["config"], r["target_id"]): r for r in rows}
    for target in (T1, T2, T3):
        r1, r3 = by[(R1, target)], by[(R3, target)]
        assert not r1["readable"] and not r3["readable"]
        assert r1["doc_ids"] == r3["doc_ids"]  # both the presented order
    assert done.summary["quality"][R1]["unreadable"] == 3
    assert done.summary["selected"] is None  # R2 wasn't run, so there is no selection


def test_a_capped_or_failing_model_gives_rows_left_out(live_run: tuple[Config, Path]) -> None:
    cfg, run_dir = live_run
    asked: list[str] = []

    class Refusing:
        def complete(self, messages: Any, **kwargs: Any) -> Any:
            asked.append(messages[-1]["content"])
            raise BudgetExceeded("over the cap")

    def flaky(messages: list[dict[str, str]]) -> str:
        if GRADED_MARK in messages[-1]["content"]:
            raise LLMError("upstream 500")
        return original_first(messages)

    clients = {PRODUCT_MODEL: FakeLLM(flaky), UPPER_MODEL: Refusing()}
    done = run(cfg, run_dir, clients)
    assert len(asked) == 1  # after one refusal the upper model is not asked again
    quality = done.summary["quality"]
    assert (quality[RS]["n"], quality[RS]["n_capped"]) == (0, 3)
    assert (quality[R3]["n"], quality[R3]["n_error"]) == (0, 3)
    assert quality[R1]["n"] == quality[R2]["n"] == 3
    assert done.summary["selected"] is None  # R3 has no checks


def test_a_chosen_variant_is_compared_with_l5w(
    live_run: tuple[Config, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, run_dir = live_run
    monkeypatch.setattr(live_rerank, "RERANK_CHOICE", R2)
    done = run(cfg, run_dir, {PRODUCT_MODEL: FakeLLM(original_first)}, variants=[R1, R2])
    decides = done.summary["decides"]
    assert (decides["a"], decides["b"], decides["n"]) == (R2, L5W, 3)
    assert "R2 vs L5w (L5r: decides)" in (done.run_dir / SUMMARY_MD).read_text(encoding="utf-8")


def test_run_rerank_checks_its_inputs(
    live_run: tuple[Config, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, run_dir = live_run
    with pytest.raises(EvalError, match="no client"):
        run(cfg, run_dir, {PRODUCT_MODEL: FakeLLM(original_first)}, variants=[RS])
    with pytest.raises(EvalError, match="unknown variants"):
        run(cfg, run_dir, {}, variants=["R9"])
    monkeypatch.setattr(live_evaluate, "L5W_FUSION", None)
    with pytest.raises(EvalError, match="L5w has no final fusion"):
        run(cfg, run_dir, {PRODUCT_MODEL: FakeLLM(original_first)})


def test_limit_takes_the_first_checks_in_qid_order(live_run: tuple[Config, Path]) -> None:
    cfg, run_dir = live_run
    done = run(cfg, run_dir, {PRODUCT_MODEL: FakeLLM(original_first)}, variants=[R1], limit=2)
    meta = json.loads((done.run_dir / RUN_FILE).read_text(encoding="utf-8"))
    assert len(meta["qids"]) == 2 and meta["qids"] == sorted(meta["qids"])
    assert done.summary["n_queries"] == 2


# --- the command -------------------------------------------------------------------------------


def test_the_command_caches_and_charges_each_model_to_its_own_ledger(
    live_run: tuple[Config, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    api_key: str,
) -> None:
    cfg, run_dir = live_run
    config_path = tmp_path / "config.yaml"
    data = _config_dict(tmp_path)
    data["rerank"] = rerank_dict(tmp_path)
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    fakes = {"fake/reranker": FakeLLM(original_first), "fake/upper": FakeLLM(original_first)}
    real_client: Callable[..., Any] = cli.LLMClient

    def client(model: Any, cache: Any, ledger: Any, **kwargs: Any) -> Any:
        return real_client(model, cache, ledger, transport=fakes[model.name].transport, **kwargs)

    monkeypatch.setattr(cli, "LLMClient", client)
    monkeypatch.setattr(cli, "make_encoder", lambda settings: FakeEncoder())
    argv = ["rerank", "--config", str(config_path), str(run_dir)]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "rk rerank: 3 checks; Hit@5: L5" in out
    assert "model ledger $" in out and "upper_model ledger $" in out
    assert len(fakes["fake/reranker"].requests) == 6 and len(fakes["fake/upper"].requests) == 3
    # Each model's spend is in its own file; a second run is served from the caches.
    assert cfg.rerank is not None
    assert (tmp_path / "upper.sqlite").exists() and cfg.run.cache_path.exists()
    assert cli.main([*argv, "--variants", "R1,RS"]) == 0
    assert len(fakes["fake/reranker"].requests) == 6 and len(fakes["fake/upper"].requests) == 3
    with pytest.raises(SystemExit):
        cli.main([*argv, "--variants", "R7"])


def test_each_variant_reads_the_declared_call() -> None:
    assert CALLS == {
        R1: ("rerank_listwise", PRODUCT_MODEL),
        R2: ("rerank_listwise", PRODUCT_MODEL),
        R3: ("rerank_graded", PRODUCT_MODEL),
        R4: ("rerank_two_lists", PRODUCT_MODEL),
        RS: ("rerank_listwise", UPPER_MODEL),
    }
    assert live_rerank.STEP4 == (R1, R2, R3, RS)  # a run without --variants has no R4


# --- R4, two lists (issue #22) -------------------------------------------------------------


def test_r4_is_the_two_list_call_scored_on_list_a(
    live_run: tuple[Config, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, run_dir = live_run
    monkeypatch.setattr(live_rerank, "RERANK_CHOICE", R1)
    product = FakeLLM(any_reranker)
    done = run(cfg, run_dir, {PRODUCT_MODEL: product}, variants=[R4, R1])
    assert done.summary["variants"] == [R1, R4]
    # One two-list call per check, with its strict schema; R1's stays plain text.
    two_list_calls = [i for i, c in enumerate(product.calls) if TWO_LISTS_MARK in c[-1]["content"]]
    assert len(product.calls) == 6 and len(two_list_calls) == 3
    assert [product.response_formats[i] for i in two_list_calls] == [TWO_LISTS_RESPONSE_FORMAT] * 3
    assert product.response_formats.count(None) == 3
    by = {(r["config"], r["target_id"]): r for r in results(done)}
    for target in (T1, T2, T3):
        r4 = by[(R4, target)]
        assert r4["status"] == "ok" and r4["readable"] and r4["methods_readable"]
        assert r4["doc_ids"] == by[(R1, target)]["doc_ids"]  # list A is original_first's order
    two = done.summary["two_lists"]
    hit5 = done.summary["quality"][R4]["hit_at_5"]
    assert two["min_hit_at_5"] == TWO_LISTS_MIN_HIT_AT_5 == 0.84
    assert two["passes"] is (hit5 >= 0.84) and two["methods_unreadable"] == 0
    assert (two["vs_choice"]["a"], two["vs_choice"]["b"], two["vs_choice"]["n"]) == (R4, R1, 3)
    meta = json.loads((done.run_dir / RUN_FILE).read_text(encoding="utf-8"))
    assert set(meta["prompts"]) == {"rerank_listwise", "rerank_two_lists", "two_lists_schema"}
    markdown = (done.run_dir / SUMMARY_MD).read_text(encoding="utf-8")
    assert "## Two lists (issue #22): list A (R4) against L5r" in markdown
    assert f"it **{'passes' if two['passes'] else 'fails'}**" in markdown


def test_r4_reads_list_a_alone_and_counts_unreadable_list_b(
    live_run: tuple[Config, Path],
) -> None:
    cfg, run_dir = live_run

    def original_last(messages: list[dict[str, str]]) -> str:
        reply = json.loads(two_lists_reply(messages))
        order = reply["contribution_ranking"]
        reply["contribution_ranking"] = order[1:] + order[:1]
        reply["methods_ranking"] = "none"
        return json.dumps(reply)

    done = run(cfg, run_dir, {PRODUCT_MODEL: FakeLLM(original_last)}, variants=[R4])
    quality, two = done.summary["quality"][R4], done.summary["two_lists"]
    assert quality["hit_at_1"] == 0 and quality["unreadable"] == 0
    assert two["passes"] is (quality["hit_at_5"] >= TWO_LISTS_MIN_HIT_AT_5)
    assert two["methods_unreadable"] == 3 and two["vs_choice"] is None  # no L5r in the run
