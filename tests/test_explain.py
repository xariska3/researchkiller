"""Step 5: what a check shows for its top papers (snippets, explainer, vocabulary, report).

The checks run on FakeLiveOpenAlex with test_live_eval's FakeLLM agent, test_rerank's
FakeLLM reranker and FakeEncoder; the explainer is a FakeLLM that rates the proposal's
original high and every other paper low, quoting each abstract's first sentence.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from bench_data import ABSTRACTS, GOOD, T1, T2, T3, TITLES, corpus_records, judge_reply, target_rows
from conftest import TEST_API_KEY_ENV, FakeEncoder, FakeLLM
from live_fakes import LIST_COST, FakeLiveOpenAlex, oa_work
from pydantic import ValidationError
from test_live_eval import B1_RANKS, B5_RANKS, _config_dict, _paraphraser, _phase1_run, _search
from test_live_eval import agent as fake_agent
from test_rerank import dated_corpus, original_first, rerank_dict

from rk import cli
from rk.bench import FACETS as JUDGE_FACETS
from rk.bench import make_bench
from rk.config import Config, ConfigError, config_from_dict
from rk.corpus import IDS_PER_CALL, freeze
from rk.evaluate import EvalError
from rk.live.check import REPORT_FILE, run_check
from rk.live.evaluate import run_l1, run_live_eval
from rk.live.explain import (
    EXPLAIN_PROMPT,
    EXPLAIN_RESPONSE_FORMAT,
    EXPLAIN_SCHEMA_SHA256,
    EXPLAINED,
    FALLBACK,
    SKIPPED,
    explain,
    parse_explanations,
    render_prompt,
    snippets,
    verify_quote,
)
from rk.live.explain_eval import REVIEW_COLUMNS, recorded_times, run_explain_eval
from rk.live.openalex_search import KEYWORDS_SELECT
from rk.live.pipeline import AGENT_PROMPT, RERANKED, CheckLedger
from rk.live.report import (
    FACETS,
    HEADER,
    NOT_EXPLAINED,
    UNVERIFIED,
    VERIFIED,
    CheckReport,
    CheckRequest,
    render_text,
)
from rk.live.rerank import PRODUCT_MODEL, R1, run_rerank
from rk.live.reranker import LISTWISE
from rk.live.vocab import KEYWORDS, QUERIES, vocabulary
from rk.live.vocab import TITLES as TITLE_GROUP
from rk.llm import LLMError
from rk.text import content_stems, sentences

PAPER_RE = re.compile(r"\[(\d+)\] Title: (.*)\nAbstract: (.*)")
KEYWORDS_OF_T1 = ["Automatic Summarization", "Sparse attention routing"]


def explain_dict(**overrides: Any) -> dict[str, Any]:
    model = {
        "name": "fake/explainer",
        "family": "fakeexp",
        "base_url": "http://llm.test/v1",
        "api_key_env": TEST_API_KEY_ENV,
        "price_in_per_m": 1.0,
        "price_out_per_m": 2.0,
        "temperature": 0.0,
        "max_tokens": 500,
        "seed": 42,
        "training_cutoff": "2024-01-01",
    }
    return {"model": model, **overrides}


def config_dict(tmp_path: Path, **explain: Any) -> dict[str, Any]:
    return {
        **_config_dict(tmp_path),
        "rerank": rerank_dict(tmp_path),
        "explain": explain_dict(**explain),
    }


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return config_from_dict(config_dict(tmp_path))


def works() -> list[dict[str, Any]]:
    """The fixture corpus as OpenAlex returns it, T1 with OpenAlex keywords."""
    return [
        oa_work(r, keywords=[{"display_name": k} for k in KEYWORDS_OF_T1])
        if r["id"] == T1
        else oa_work(r)
        for r in corpus_records()
    ]


def entry(n: int, abstract: str, high: bool, **changes: Any) -> dict[str, Any]:
    return {
        "paper": n,
        "facets": {
            f: {"verdict": "same" if high else "different", "reason": f"{f} reason"}
            for f in FACETS
        },
        "overlap": "high" if high else "low",
        "evidence": sentences(abstract)[0],
        "differs": "nothing important" if high else "another problem",
        **changes,
    }


def explainer_reply(messages: list[dict[str, str]]) -> str:
    """Every paper: the proposal's original high, the others low; the first sentence quoted."""
    prompt = messages[-1]["content"]
    target = next(t for t in TITLES if GOOD[t] in prompt)
    papers = [
        entry(int(n), abstract, title == TITLES[target])
        for n, title, abstract in PAPER_RE.findall(prompt)
    ]
    return json.dumps({"papers": papers})


def check(cfg: Config, text: str = GOOD[T1], **kwargs: Any) -> Any:
    fake = kwargs.pop("fake", None) or FakeLiveOpenAlex(works())
    explainer = kwargs.pop("explainer", None) or FakeLLM(explainer_reply)
    return run_check(
        CheckRequest(text=text, **kwargs),
        cfg,
        search=_search(cfg, fake),
        encoder=FakeEncoder(),
        agent=fake_agent(),
        reranker=FakeLLM(original_first),
        explainer=explainer,
    )


# --- sentences, snippets and the quote check ------------------------------------------------


def test_sentences_are_substrings_split_after_full_stops_only() -> None:
    text = (
        "We extend BERT (Devlin et al. 2019) to long inputs. It helps, e.g. on QA.  Scores "
        "rise by 3.5 points! J. Smith (Fig. 2) agrees. A last sentence"
    )
    found = sentences(text)
    assert found == [
        "We extend BERT (Devlin et al. 2019) to long inputs.",
        "It helps, e.g. on QA.",
        "Scores rise by 3.5 points!",
        "J. Smith (Fig. 2) agrees.",
        "A last sentence",
    ]
    assert all(s in text for s in found)
    assert sentences("") == [] and sentences("No full stop") == ["No full stop"]


def test_the_snippet_is_the_abstract_sentence_closest_to_the_idea() -> None:
    records = [
        {"id": "W1", "abstract": ABSTRACTS[T3]},
        {"id": "W2", "abstract": ""},
        {"id": "W3", "abstract": ABSTRACTS[T2]},
    ]
    idea = "a graph network trained with random sensor dropout and a learned imputation head"
    first, empty, other = snippets(idea, records, FakeEncoder())
    assert first is not None and first.sentence.startswith("We train a graph network")
    assert first.sentence in ABSTRACTS[T3] and 0 < first.similarity <= 1
    assert empty is None
    assert other is not None and other.sentence in sentences(ABSTRACTS[T2])


def test_a_quote_must_be_in_the_abstract_verbatim_up_to_spaces_and_quote_marks() -> None:
    abstract = 'We call it "routing" here.  The router keeps   quality within half a point.'
    assert verify_quote("The router keeps quality within half a point.", abstract)
    assert verify_quote("The router keeps\nquality within half a point", abstract)  # a prefix
    assert verify_quote("“The router keeps quality within half a point.”", abstract)
    assert verify_quote("We call it “routing” here.", abstract)
    assert not verify_quote("The router keeps quality within one point.", abstract)
    assert not verify_quote("the router keeps quality within half a point.", abstract)  # case
    assert not verify_quote("keeps quality within", abstract)  # under MIN_QUOTE_WORDS
    assert not verify_quote("", abstract)


# --- the explainer call -----------------------------------------------------------------------


def records() -> list[dict[str, Any]]:
    return [{"id": t, "title": TITLES[t], "abstract": ABSTRACTS[t]} for t in (T1, T2, T3)]


def test_the_explainer_sees_every_paper_and_answers_in_its_schema(cfg: Config) -> None:
    assert cfg.explain is not None
    explainer = FakeLLM(explainer_reply)
    ledger = CheckLedger(cap=1.0)
    step = explain(
        GOOD[T1], records(), explainer, cfg.explain.model, ledger, abstract_chars=2000
    )
    (call,) = explainer.calls
    assert call == [{"role": "user", "content": render_prompt(GOOD[T1], records(), 2000)}]
    assert "[2] Title: " + TITLES[T2] in call[0]["content"]
    assert explainer.response_formats == [EXPLAIN_RESPONSE_FORMAT]
    assert step.status == EXPLAINED and step.warning is None
    first, second, third = step.explanations
    assert first is not None and first.overlap == "high" and first.verified
    assert first.evidence == sentences(ABSTRACTS[T1])[0] == first.quote
    assert first.facets.method.verdict == "same" and first.differs == "nothing important"
    assert second is not None and second.overlap == "low" and third is not None
    # The call is charged to a copy of the check's ledger.
    assert step.ledger.llm.calls == 1 and ledger.llm.calls == 0
    assert step.ledger.method_cost == pytest.approx(step.usage.method_cost)


def test_each_entry_is_read_on_its_own() -> None:
    abstracts = [r["abstract"] for r in records()]
    reply = {
        "papers": [
            entry(1, abstracts[0], True, evidence="A sentence the abstract doesn't have at all."),
            entry(2, abstracts[1], False, overlap="very high"),  # not a level
            entry(1, abstracts[0], False),  # a repeat
            entry(9, abstracts[0], False),  # out of range
            "not an entry",
        ]
    }
    first, second, third = parse_explanations(json.dumps(reply), records()) or []
    assert first is not None and first.overlap == "high"
    assert not first.verified and first.evidence is None  # the quote is dropped
    assert first.quote == "A sentence the abstract doesn't have at all."
    assert second is None and third is None
    assert parse_explanations("no JSON here", records()) is None
    assert parse_explanations('{"papers": "none"}', records()) is None


def test_missing_entries_get_a_warning(cfg: Config) -> None:
    assert cfg.explain is not None

    def two_of_three(messages: list[dict[str, str]]) -> str:
        papers = json.loads(explainer_reply(messages))["papers"]
        return json.dumps({"papers": [papers[0], papers[2]]})

    step = explain(
        GOOD[T1], records(), FakeLLM(two_of_three), cfg.explain.model, CheckLedger(1.0),
        abstract_chars=2000,
    )
    assert step.status == EXPLAINED and step.explanations[1] is None
    assert step.warning == (
        "No explanation for the paper(s) at rank 2; they show their snippets only."
    )


def test_an_unreadable_failed_or_capped_call_leaves_snippets_only(cfg: Config) -> None:
    assert cfg.explain is not None
    model = cfg.explain.model

    def down(messages: list[dict[str, str]]) -> str:
        raise LLMError("upstream 503")

    unread = explain(GOOD[T1], records(), FakeLLM(["Sorry."]), model, CheckLedger(1.0),
                     abstract_chars=2000)
    assert unread.status == FALLBACK and unread.explanations == [None, None, None]
    assert unread.warning == (
        "Not explained (the explainer's reply could not be read); the papers show their "
        "snippets only."
    )
    assert unread.ledger.llm.calls == 1  # answered, so charged
    failed = explain(GOOD[T1], records(), FakeLLM(down), model, CheckLedger(1.0),
                     abstract_chars=2000)
    assert failed.status == FALLBACK and "the explainer call failed: upstream 503" in (
        failed.warning or ""
    )
    with pytest.raises(LLMError, match="upstream 503"):
        explain(GOOD[T1], records(), FakeLLM(down), model, CheckLedger(1.0),
                abstract_chars=2000, strict=True)
    full = CheckLedger(cap=0.001, openalex_cost=0.001)
    never = FakeLLM([])
    capped = explain(GOOD[T1], records(), never, model, full, abstract_chars=2000, strict=True)
    assert never.calls == [] and capped.status == FALLBACK
    assert "the explainer call could cost $" in (capped.warning or "")
    nothing = explain(GOOD[T1], [], never, model, CheckLedger(1.0), abstract_chars=2000)
    assert nothing.status == SKIPPED and nothing.warning is None and never.calls == []


# --- the vocabulary map ---------------------------------------------------------------------


def test_the_vocabulary_map_lists_supported_terms_the_text_lacks() -> None:
    text = "We will condense very lengthy books by letting each word attend to a few groups."
    papers = [
        {"id": "W1", "title": "Sparse Attention Routing for Long Document Summarisation",
         "abstract": "Sparse attention routing sends each token to clusters for summarisation."},
        {"id": "W2", "title": "Clustered attention in BERT encoders",
         "abstract": "We cluster tokens so attention is sparse in BERT encoders."},
    ]
    keywords = {"W1": ["Automatic Summarization", "Sparse attention routing"], "W2": ["Books"]}
    queries = ["sparse attention token clustering", "condense books"]
    groups = vocabulary(text, queries, papers, keywords, max_terms=10)
    assert [(g.source, g.terms) for g in groups] == [
        # "Books" is in the text; the groups take turns, so the keywords' second term
        # comes after the titles' and the queries' first.
        (KEYWORDS, ["automatic summarization", "sparse attention routing"]),
        # Both papers use "clustered attention", so it leads; the title's "Sparse
        # Attention Routing" has the same stems as a keyword already taken; "BERT" keeps
        # its capitals.
        (TITLE_GROUP, ["clustered attention", "long document summarisation", "BERT encoders"]),
        # "condense books" has nothing the text lacks.
        (QUERIES, ["sparse attention token clustering"]),
    ]
    known = set(content_stems(text))
    for group in groups:
        assert group.label and len(group.terms) == len(set(group.terms))
        for term in group.terms:
            assert not set(content_stems(term, drop_generic=True)) <= known
    capped = vocabulary(text, queries, papers, keywords, max_terms=2)
    assert [(g.source, len(g.terms)) for g in capped] == [(KEYWORDS, 1), (TITLE_GROUP, 1)]


def test_a_term_no_top_paper_uses_is_dropped() -> None:
    papers = [{"id": "W1", "title": "Graph forecasting", "abstract": "Graph forecasting."}]
    groups = vocabulary("We predict speeds.", ["quantum annealing schedules"], papers, {},
                        max_terms=10)
    assert [g.source for g in groups] == [TITLE_GROUP]  # the agent's terms had no support


# --- the keyword lookup ---------------------------------------------------------------------


def test_keywords_come_from_one_cached_list_call(cfg: Config) -> None:
    fake = FakeLiveOpenAlex(works())
    search = _search(cfg, fake)
    lookup = search.keywords([T1, T2])
    assert lookup.keywords == {T1: KEYWORDS_OF_T1, T2: []}
    assert (lookup.cost_usd, lookup.cached, lookup.money_spent) == (LIST_COST, False, LIST_COST)
    (request,) = fake.requests
    assert request["filter"] == f"ids.openalex:{T1}|{T2}" and request["select"] == KEYWORDS_SELECT
    again = search.keywords([T1, T2])
    assert again.cached and again.money_spent == 0.0 and len(fake.requests) == 1
    assert search.keywords([]).keywords == {}
    with pytest.raises(ValueError, match="at most"):
        search.keywords([f"W{n}" for n in range(IDS_PER_CALL + 1)])


# --- the whole check ------------------------------------------------------------------------


def test_the_default_check_explains_its_top_papers(cfg: Config) -> None:
    assert cfg.live is not None and cfg.explain is not None and cfg.rerank is not None
    run = check(cfg)
    report = run.report
    assert report.header == HEADER and report.mode == "default"
    assert report.coverage.startswith(f"Searched {run.result.pool_size} candidate papers")
    assert [p.rank for p in report.papers] == list(range(1, len(report.papers) + 1))
    assert len(report.papers) == cfg.live.top_k
    top, rest = report.papers[: cfg.live.explain_top_k], report.papers[cfg.live.explain_top_k :]
    assert all(p.snippet is not None and p.explanation is not None for p in top)
    assert all(p.snippet is None and p.explanation is None for p in rest)
    assert report.papers[0].id == T1 and report.papers[0].status == VERIFIED
    assert report.papers[0].explanation is not None
    assert report.papers[0].explanation.overlap == "high"
    assert (report.rerank, report.explainer) == (RERANKED, EXPLAINED)
    assert report.warnings == []
    assert report.queries == run.result.trace.queries[1:] and report.queries
    assert {g.source for g in report.vocabulary} >= {KEYWORDS, TITLE_GROUP}
    # One ledger and one cap for the whole check: retrieval, lookup and explainer.
    assert run.explain is not None
    assert report.cost.cap == cfg.explain.cost_cap_per_check == run.result.ledger.cap
    assert report.cost.method_cost == pytest.approx(
        run.result.ledger.method_cost + LIST_COST + run.explain.usage.method_cost
    )
    assert report.cost.openalex_cost == pytest.approx(run.result.ledger.openalex_cost + LIST_COST)
    assert report.models == {
        "encoder": "fake-encoder",
        "agent": cfg.agent.name,
        "rerank": cfg.rerank.model.name,
        "explain": cfg.explain.model.name,
    }
    assert set(report.prompt_hashes) == {AGENT_PROMPT, LISTWISE, EXPLAIN_PROMPT, "explain_schema"}
    assert report.prompt_hashes["explain_schema"] == EXPLAIN_SCHEMA_SHA256
    assert report.stages.total_s >= report.stages.explain_s > 0
    assert CheckReport.model_validate_json(report.model_dump_json()) == report
    text = render_text(report)
    assert text.startswith(HEADER + "\n" + report.coverage)
    assert f"#1 [high overlap] {TITLES[T1]}" in text and "Evidence: " in text
    assert "Words the literature uses that your text doesn't:" in text


def test_an_unverified_quote_shows_the_snippet(cfg: Config) -> None:
    def paraphrased(messages: list[dict[str, str]]) -> str:
        reply = json.loads(explainer_reply(messages))
        reply["papers"][0]["evidence"] = "Long books need sparse attention to fit in memory."
        return json.dumps(reply)

    report = check(cfg, explainer=FakeLLM(paraphrased)).report
    first = report.papers[0]
    assert first.status == UNVERIFIED and first.explanation is not None
    assert first.explanation.evidence is None and first.snippet is not None
    assert f'Snippet, quote unverified: "{first.snippet.sentence}"' in render_text(report)


def test_nothing_after_the_search_fails_the_check(cfg: Config) -> None:
    def down(messages: list[dict[str, str]]) -> str:
        raise LLMError("upstream 503")

    fake = FakeLiveOpenAlex(works(), fail_lookups=True)
    run = check(cfg, fake=fake, explainer=FakeLLM(down))
    report = run.report
    assert report.explainer == FALLBACK and len(report.warnings) == 2
    assert report.warnings[0].startswith("No OpenAlex keywords for the vocabulary map (")
    assert report.warnings[1].startswith("Not explained (the explainer call failed: upstream")
    top = report.papers[:3]
    assert all(p.snippet is not None and p.status == NOT_EXPLAINED for p in top)
    assert report.vocabulary  # titles and queries still give terms
    assert "Warning: Not explained" in render_text(report)


def test_the_quick_check_is_l1_with_snippets_and_no_llm(cfg: Config) -> None:
    assert cfg.live is not None
    fake = FakeLiveOpenAlex(works())
    run = run_check(
        CheckRequest(text=GOOD[T1], quick=True), cfg, search=_search(cfg, fake),
        encoder=FakeEncoder(),
    )
    report = run.report
    l1 = run_l1(GOOD[T1], _search(cfg, fake), cfg.live, FakeEncoder())
    assert [p.id for p in report.papers] == l1.doc_ids[: cfg.live.top_k]
    assert report.mode == "quick" and (report.rerank, report.explainer) == (None, None)
    assert all(p.snippet is not None for p in report.papers[:3])
    assert all(p.explanation is None for p in report.papers)
    assert QUERIES not in {g.source for g in report.vocabulary}
    assert report.n_searches == 1 and report.models == {"encoder": "fake-encoder"}
    assert report.cost.llm_method_cost == 0
    assert "Quick check (no LLM)" in render_text(report)


def test_the_idea_must_be_40_to_500_words() -> None:
    words = "word " * 39
    with pytest.raises(ValidationError, match="40-500 words long; it has 39"):
        CheckRequest(text=words)
    assert CheckRequest(text=f"  {words}word ").text == (words + "word").strip()
    with pytest.raises(ValidationError):
        CheckRequest(text="word " * 501)
    assert CheckReport.model_fields["header"].default == HEADER
    assert FACETS == JUDGE_FACETS


def test_a_check_needs_the_explain_section(tmp_path: Path) -> None:
    plain = config_from_dict({**_config_dict(tmp_path), "rerank": rerank_dict(tmp_path)})
    with pytest.raises(ValueError, match="explain"):
        check(plain)


def test_the_explainer_must_predate_the_originals(tmp_path: Path) -> None:
    data = config_dict(tmp_path)
    data["explain"]["model"]["training_cutoff"] = "2026-06-30"
    with pytest.raises(ConfigError, match="explain"):
        config_from_dict(data)


# --- the commands ---------------------------------------------------------------------------


def patch_cli(
    monkeypatch: pytest.MonkeyPatch, cfg: Config, fake: FakeLiveOpenAlex, *, evaluation: bool
) -> list[str]:
    """The CLI's clients as fakes; returns the model names it asked for, in order."""

    class Search:
        @staticmethod
        def from_config(config: Config, cache: Any, **kwargs: Any) -> Any:
            assert kwargs.get("evaluation", False) is evaluation
            return _search(config, fake)

    models: list[str] = []
    fakes = {
        cfg.agent.name: fake_agent,
        "fake/reranker": lambda: FakeLLM(original_first),
        "fake/explainer": lambda: FakeLLM(explainer_reply),
    }

    def client(model: Any, cache: Any, ledger: Any) -> FakeLLM:
        models.append(model.name)
        return fakes[model.name]()

    monkeypatch.setattr(cli, "LiveSearchClient", Search)
    monkeypatch.setattr(cli, "LLMClient", client)
    monkeypatch.setattr(cli, "make_encoder", lambda retrieval_cfg: FakeEncoder())
    return models


def test_the_check_command_prints_and_saves_the_report(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict(tmp_path)), encoding="utf-8")
    idea = tmp_path / "idea.txt"
    idea.write_text(GOOD[T1], encoding="utf-8")
    models = patch_cli(monkeypatch, cfg, FakeLiveOpenAlex(works()), evaluation=False)
    assert cli.main(["check", "--config", str(config_path), "--file", str(idea)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(HEADER) and f"[high overlap] {TITLES[T1]}" in out
    assert models == [cfg.agent.name, "fake/reranker", "fake/explainer"]
    (saved,) = (cfg.run.out_dir).glob(f"check-*/{REPORT_FILE}")
    assert f"rk check: report saved to {saved}" in out
    report = CheckReport.model_validate_json(saved.read_text(encoding="utf-8"))
    assert report.request.text == GOOD[T1] and report.mode == "default"
    # --quick asks for no LLM at all, and --before reaches the search.
    models.clear()
    argv = ["check", "--config", str(config_path), "--file", str(idea), "--quick"]
    assert cli.main([*argv, "--before", "2025-06-01"]) == 0
    assert models == [] and "Quick check (no LLM)" in capsys.readouterr().out
    idea.write_text("Too short to check.", encoding="utf-8")
    assert cli.main(argv) == 1
    assert "the idea must be 40-500 words long; it has 4" in capsys.readouterr().err


# --- the evaluation -------------------------------------------------------------------------


@pytest.fixture
def bench(tmp_path: Path) -> Config:
    """The fixture benchmark (dated corpus), a Phase 1 run, and the explain section."""
    config = config_from_dict(config_dict(tmp_path))
    freeze(config.corpus.out_dir, dated_corpus(), target_rows(), {})
    stats = make_bench(config, _paraphraser(), FakeLLM(lambda messages: judge_reply()))
    assert stats["accepted"] == 3
    _phase1_run(config, B1_RANKS, B5_RANKS)
    return config


def evaluate(cfg: Config, tmp_path: Path, **kwargs: Any) -> Any:
    kwargs.setdefault("review_out", tmp_path / "review.csv")
    return run_explain_eval(
        cfg,
        search=_search(cfg, FakeLiveOpenAlex(works())),
        encoder=FakeEncoder(),
        agent=fake_agent(),
        reranker=FakeLLM(original_first),
        explainer=FakeLLM(explainer_reply),
        from_qid="A0000",
        **kwargs,
    )


def test_explain_eval_measures_quotes_overlap_cost_and_time(bench: Config, tmp_path: Path) -> None:
    assert bench.live is not None and bench.explain is not None
    # Step 2's live run and Step 4's rerank run on the same checks, for the timing.
    live = run_live_eval(
        bench, fake_agent(), FakeEncoder(), _search(bench, FakeLiveOpenAlex(works())),
        configs=["L5", "L5w"], log_mlflow=False,
    )
    rerank = run_rerank(
        bench, live.run_dir, encoder=FakeEncoder(),
        clients={PRODUCT_MODEL: FakeLLM(original_first)}, variants=[R1],
    )
    times = recorded_times(rerank.run_dir, bench.run.results_db)
    assert len(times) == 3 and all(t["check_s"] and t["rerank_s"] for t in times.values())
    done = evaluate(bench, tmp_path, limit=3, review_n=2, timing_run=rerank.run_dir)
    s = done.summary
    assert (s["n_checks"], s["n_ok"], s["over_cap"]) == (3, 3, 0)
    assert s["quotes"] == {"papers": 9, "explained": 9, "verified": 9, "rate": 1.0}
    original = s["original"]
    assert original["in_top"] + original["not_in_top"] == 3 and original["in_top"] >= 1
    assert original["levels"]["high"] == original["in_top"] and original["high_rate"] == 1.0
    assert s["high_not_original"] == 0
    assert s["steps"]["explainer"] == {"explained": 3, "fallback": 0, "skipped": 0}
    assert s["cost"]["check"]["max"] <= bench.explain.cost_cap_per_check
    assert s["time"]["estimated_live"]["n"] == 3
    lines = (done.run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    for row in rows:
        stage = row["explainer_stage_s"]
        times_q = times[row["qid"]]
        assert row["estimated_live_s"] == pytest.approx(
            times_q["check_s"] + times_q["rerank_s"] + stage
        )
    assert len(list((done.run_dir / "reports").glob("*.json"))) == 3
    assert "## Measures" in (done.run_dir / "summary.md").read_text(encoding="utf-8")
    # The review CSV: the first 2 checks, their top 5 papers, the review columns empty.
    with (tmp_path / "review.csv").open(encoding="utf-8", newline="") as fh:
        review = list(csv.DictReader(fh))
    assert tuple(review[0]) == REVIEW_COLUMNS and done.review_rows == len(review) == 10
    assert {r["qid"] for r in review} == {rows[0]["qid"], rows[1]["qid"]}
    assert all(r["correct?"] == r["useful?"] == r["false alarm?"] == "" for r in review)
    # The test config explains the top 3 (explain_top_k), so ranks 4 and 5 have neither
    # an explanation nor a snippet.
    explained = [r for r in review if r["overlap"] != NOT_EXPLAINED]
    assert [r["rank"] for r in explained] == ["1", "2", "3"] * 2
    assert all(r["verified"] == "yes" and r["evidence"] for r in explained)
    assert all(r["evidence"] == r["verified"] == "" for r in review if r not in explained)
    with pytest.raises(EvalError, match="never overwritten"):
        evaluate(bench, tmp_path, limit=1)


def test_the_explain_eval_command(
    bench: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict(tmp_path)), encoding="utf-8")
    models = patch_cli(monkeypatch, bench, FakeLiveOpenAlex(works()), evaluation=True)
    out_csv = tmp_path / "review.csv"
    argv = ["explain-eval", "--config", str(config_path), "--from-qid", "A0000", "--limit", "2",
            "--review", "1", "--review-out", str(out_csv)]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "rk explain-eval: 2 of 2 checks; quotes verified 100.0%" in out
    assert models == [bench.agent.name, "fake/reranker", "fake/explainer"]
    assert out_csv.exists() and "(5 rows)" in out
