"""Step 5: what a check shows for its top papers (snippets, explainer, vocabulary, report).

The checks run on FakeLiveOpenAlex with test_live_eval's FakeLLM agent, test_rerank's
FakeLLM reranker and FakeEncoder; the explainer is a FakeLLM that rates the proposal's
original high and every other paper low, quoting each abstract's first sentence.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from bench_data import ABSTRACTS, GOOD, T1, T2, T3, TITLES, corpus_records, judge_reply, target_rows
from conftest import TEST_API_KEY_ENV, FakeEncoder, FakeLLM
from live_fakes import LIST_COST, SEARCH_MAX_BYTES, FakeLiveOpenAlex, oa_work
from pydantic import ValidationError
from test_live_eval import (
    AGENT_QUERY,
    B1_RANKS,
    B5_RANKS,
    _config_dict,
    _paraphraser,
    _phase1_run,
    _rows,
    _search,
)
from test_live_eval import agent as fake_agent
from test_rerank import dated_corpus, original_first, rerank_dict, two_lists_reply

from rk import cli
from rk.bench import FACETS as JUDGE_FACETS
from rk.bench import make_bench
from rk.config import Config, ConfigError, config_from_dict
from rk.corpus import IDS_PER_CALL, OpenAlexError, freeze
from rk.evaluate import EvalError
from rk.live.check import REPORT_FILE, run_check
from rk.live.evaluate import run_l1, run_live_eval
from rk.live.explain import (
    EXPLAIN_PROMPT,
    EXPLAIN_RESPONSE_FORMAT,
    EXPLAIN_SCHEMA_SHA256,
    EXPLAINED,
    FALLBACK,
    NO_DIFFERENCE,
    SKIPPED,
    SNIPPET_CHARS,
    explain,
    locate_quote,
    parse_explanations,
    render_prompt,
    snippets,
)
from rk.live.explain_eval import REVIEW_COLUMNS, recorded_times, run_explain_eval
from rk.live.openalex_search import DETAILS_SELECT, search_text
from rk.live.pipeline import AGENT_PROMPT, RERANKED, Candidate, CheckLedger, merge_same_text
from rk.live.pool import same_text, shared_sentences
from rk.live.report import (
    EXACT,
    FACETS,
    HEADER,
    MATCHED,
    NO_MATCH,
    NOT_EXPLAINED,
    UNVERIFIED,
    VERIFIED_EXACT,
    VERIFIED_MATCHED,
    CheckReport,
    CheckRequest,
    Paper,
    render_text,
)
from rk.live.rerank import PRODUCT_MODEL, R1, R4, run_rerank
from rk.live.reranker import LISTWISE, TWO_LISTS
from rk.live.vocab import KEYWORDS, QUERIES, vocabulary
from rk.live.vocab import TITLES as TITLE_GROUP
from rk.llm import LLMError
from rk.text import content_stems, sentence_spans, sentences

PAPER_RE = re.compile(r"\[(\d+)\] Title: (.*)\nAbstract: (.*)")
KEYWORDS_OF_T1 = ["Automatic Summarization", "Sparse attention routing"]
VENUE_OF_T1 = "Transactions of the Association for Computational Linguistics"


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
    """The fixture corpus as OpenAlex returns it, T1 with OpenAlex keywords and a venue."""
    return [
        oa_work(
            r,
            keywords=[{"display_name": k} for k in KEYWORDS_OF_T1],
            primary_location={"source": {"display_name": VENUE_OF_T1}},
        )
        if r["id"] == T1
        else oa_work(r, primary_location={"source": None})
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
    # A full stop whose space was lost still ends a sentence, except after an abbreviation.
    glued = "Large labelled datasets.We propose a method, e.g.The one in v2.Then it ends."
    assert sentences(glued) == [
        "Large labelled datasets.",
        "We propose a method, e.g.The one in v2.",
        "Then it ends.",
    ]
    assert [glued[b:e] for b, e in sentence_spans(glued)] == sentences(glued)


def test_a_full_stop_followed_directly_by_a_capital_ends_a_sentence() -> None:
    text = "We use a model.NER labels come next.It works (see Fig.A). Done."
    assert sentences(text) == [
        "We use a model.",
        "NER labels come next.",
        "It works (see Fig.A).",
        "Done.",
    ]


def test_a_long_snippet_is_cut_at_a_word_with_an_ellipsis() -> None:
    long_sentence = "We train " + "a very large model on noisy sensor data " * 12 + "today."
    assert len(long_sentence) > SNIPPET_CHARS
    records = [{"id": "W1", "abstract": long_sentence + " Short second sentence here."}]
    (snippet,) = snippets("large model trained on noisy sensor data", records, FakeEncoder())
    assert snippet is not None and snippet.sentence.endswith(" …")
    kept = snippet.sentence[: -len(" …")]
    assert len(kept) <= SNIPPET_CHARS and long_sentence.startswith(kept)
    assert long_sentence[len(kept)] == " "  # cut at a word


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


ROUTER_ABSTRACT = (
    'We call it "routing" here.  To save memory, the router keeps   quality within half a '
    "point.Ablations follow."
)
ROUTER_SENTENCE = "To save memory, the router keeps   quality within half a point."


def test_an_exact_quote_shows_the_abstract_sentence_it_is_in() -> None:
    for quote in (
        "the router keeps quality within half a point.",
        "the router keeps\nquality within half a point",  # whitespace, and a part
        "“the router keeps quality within half a point.”",  # wrapped in curly marks
        "To save memory, the router keeps quality within half a point.",
    ):
        assert locate_quote(quote, ROUTER_ABSTRACT) == (EXACT, ROUTER_SENTENCE)
    assert locate_quote("We call it “routing” here.", ROUTER_ABSTRACT) == (
        EXACT,
        'We call it "routing" here.',  # the abstract's own marks, never the quote's
    )
    spanning = "here. To save memory, the router keeps"  # across two sentences
    assert locate_quote(spanning, ROUTER_ABSTRACT) == (
        EXACT,
        'We call it "routing" here.  ' + ROUTER_SENTENCE,
    )


def test_an_edited_quote_shows_the_sentence_with_90_percent_of_its_words() -> None:
    # A lead-in dropped and the start capitalised: every word of the quote is there.
    dropped = "The router keeps quality within half a point."
    assert locate_quote(dropped, ROUTER_ABSTRACT) == (MATCHED, ROUTER_SENTENCE)
    # 10 of the quote's 11 words are in the sentence: 91%.
    edited = "To save memory the router keeps accuracy within half a point."
    assert locate_quote(edited, ROUTER_ABSTRACT) == (MATCHED, ROUTER_SENTENCE)
    # 9 of 11 are not enough, nor is a quote under 5 words, even a verbatim one.
    assert locate_quote("To save space the router keeps accuracy within half a point.",
                        ROUTER_ABSTRACT) == (NO_MATCH, None)
    assert locate_quote("Ablations follow.", ROUTER_ABSTRACT) == (NO_MATCH, None)
    assert locate_quote("", ROUTER_ABSTRACT) == (NO_MATCH, None)


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
    assert first is not None and first.overlap == "high" and first.match == EXACT
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
    assert first.match == NO_MATCH and first.evidence is None  # the quote is dropped
    assert first.quote == "A sentence the abstract doesn't have at all."
    assert second is None and third is None
    assert parse_explanations("no JSON here", records()) is None
    assert parse_explanations('{"papers": "none"}', records()) is None


def test_differs_is_a_substantive_difference_or_the_fixed_line() -> None:
    prompt = render_prompt(GOOD[T1], records(), 2000)
    assert f'If there is none, write exactly "{NO_DIFFERENCE}"' in prompt
    assert "Never give as a difference that the paper is published or finished" in prompt
    assert "or that the proposal is a plan" in prompt
    abstracts = [r["abstract"] for r in records()]
    reply = {
        "papers": [
            entry(1, abstracts[0], True, differs="no substantive difference found"),
            entry(2, abstracts[1], False, differs="  Rewards,  not   summaries. "),
            entry(3, abstracts[2], False, differs=None),  # not a string: no explanation
        ]
    }
    first, second, third = parse_explanations(json.dumps(reply), records()) or []
    assert first is not None and first.differs == NO_DIFFERENCE
    assert second is not None and second.differs == "Rewards, not summaries."
    assert third is None


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


def test_function_words_end_a_phrase() -> None:
    text = "We will make fill-in-the-blank exercises for language learners."
    papers = [
        {"id": "W1", "title": "Cloze questions using templates via graph-based ranking",
         "abstract": "Cloze questions from templates; ranking is graph-based."},
    ]
    keywords = {"W1": ["Graph-based", "Templates for cloze questions"]}
    queries = ["cloze questions using templates", "templates with ranking"]
    groups = vocabulary(text, queries, papers, keywords, max_terms=10)
    terms = [t for g in groups for t in g.terms]
    assert [(g.source, g.terms) for g in groups] == [
        # "Graph-based" ends in a function word, so it is dropped; the other keyword is
        # cut at "for" into "Templates" and "cloze questions".
        (KEYWORDS, ["templates"]),
        # The title is cut at "using" and "via"; the groups take turns, so the titles
        # take "cloze questions" before the keywords' second turn.
        (TITLE_GROUP, ["cloze questions", "graph-based ranking"]),
        # The queries are cut at "using" and "with"; only "ranking" is left untaken.
        (QUERIES, ["ranking"]),
    ]
    assert not any(t.split()[-1] in ("using", "based", "via", "with", "for") for t in terms)


def test_a_term_no_top_paper_uses_is_dropped() -> None:
    papers = [{"id": "W1", "title": "Graph forecasting", "abstract": "Graph forecasting."}]
    groups = vocabulary("We predict speeds.", ["quantum annealing schedules"], papers, {},
                        max_terms=10)
    assert [g.source for g in groups] == [TITLE_GROUP]  # the agent's terms had no support


# --- the details lookup: keywords and venues ------------------------------------------------


def test_keywords_and_venues_come_from_one_cached_list_call(cfg: Config) -> None:
    fake = FakeLiveOpenAlex(works())
    search = _search(cfg, fake)
    lookup = search.details([T1, T2])
    assert lookup.keywords == {T1: KEYWORDS_OF_T1, T2: []}
    assert lookup.venues == {T1: VENUE_OF_T1}  # T2 has no source
    assert (lookup.cost_usd, lookup.cached, lookup.money_spent) == (LIST_COST, False, LIST_COST)
    (request,) = fake.requests
    assert request["filter"] == f"ids.openalex:{T1}|{T2}" and request["select"] == DETAILS_SELECT
    again = search.details([T1, T2])
    assert again.cached and again.money_spent == 0.0 and len(fake.requests) == 1
    assert search.details([]).keywords == {}
    with pytest.raises(ValueError, match="at most"):
        search.details([f"W{n}" for n in range(IDS_PER_CALL + 1)])


@pytest.mark.parametrize(
    ("venue", "shown"),
    [("83", None), (" 2019 ", None), (VENUE_OF_T1, VENUE_OF_T1), ("3D Research", "3D Research")],
)
def test_a_bare_number_is_not_shown_as_a_venue(venue: str, shown: str | None) -> None:
    paper = Paper(
        rank=1, id=T1, aliases=[], url=f"https://openalex.org/{T1}", doi=None, title="A title",
        publication_date="2020-01-01", type="article", authors=[], venue=venue,
    )
    assert paper.venue == shown
    assert Paper.model_validate_json(paper.model_dump_json()).venue == shown


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
    assert report.papers[0].id == T1 and report.papers[0].status == VERIFIED_EXACT
    assert report.papers[0].explanation is not None
    assert report.papers[0].explanation.overlap == "high"
    assert report.papers[0].venue == VENUE_OF_T1
    assert all(p.venue is None for p in report.papers[1:])
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


def quoting(evidence: str) -> FakeLLM:
    """The explainer, with this evidence for the first paper."""

    def reply(messages: list[dict[str, str]]) -> str:
        answer = json.loads(explainer_reply(messages))
        answer["papers"][0]["evidence"] = evidence
        return json.dumps(answer)

    return FakeLLM(reply)


def test_a_matched_quote_shows_the_abstract_sentence_not_the_quote(cfg: Config) -> None:
    sentence = sentences(ABSTRACTS[T1])[1]
    assert sentence.startswith("We introduce a routing layer")
    edited = sentence.replace("We introduce a routing layer that sends", "A routing layer sends")
    report = check(cfg, explainer=quoting(edited)).report
    first = report.papers[0]
    assert first.status == VERIFIED_MATCHED and first.explanation is not None
    assert first.explanation.evidence == sentence and first.explanation.quote == edited
    text = render_text(report)
    assert f'Evidence (matched): "{sentence}"' in text and edited not in text


def test_an_unverified_quote_shows_the_snippet(cfg: Config) -> None:
    quote = "Long books need sparse attention to fit in memory."
    report = check(cfg, explainer=quoting(quote)).report
    first = report.papers[0]
    assert first.status == UNVERIFIED and first.explanation is not None
    assert first.explanation.evidence is None and first.snippet is not None
    assert f'Snippet, quote unverified: "{first.snippet.sentence}"' in render_text(report)


# A second version of T1 with a reworded title and abstract: no sentence is identical.
T1_VERSION = {
    "id": "W301",
    "aliases": [],
    "title": "Sparse attention routing for summarising long documents",
    "abstract": (
        "Transformer summarisers struggle with very long inputs because full self-attention "
        "grows quadratically with the input length. We propose a routing layer that sends "
        "every token to a few learned clusters, so attention is computed only within "
        "clusters. Trained end to end with the summarisation objective, the router keeps "
        "ROUGE within half a point of dense baselines on arXiv and PubMed while reducing "
        "memory eightfold. Ablations show that the number of clusters matters more than "
        "cluster size."
    ),
}
OTHER_ROUTING = {  # a similar title, another paper
    "id": "W302",
    "aliases": [],
    "title": "Sparse attention routing for long document translation",
    "abstract": ABSTRACTS[T2],
}


def t1_record() -> dict[str, Any]:
    return {"id": T1, "aliases": [], "title": TITLES[T1], "abstract": ABSTRACTS[T1]}


def test_versions_of_one_paper_are_known_by_their_text() -> None:
    assert not set(sentences(T1_VERSION["abstract"])) & set(sentences(ABSTRACTS[T1]))
    assert shared_sentences(ABSTRACTS[T1], T1_VERSION["abstract"]) > 0.5
    assert same_text(t1_record(), T1_VERSION)  # with no shared author
    assert not same_text(t1_record(), OTHER_ROUTING)  # similar title, other abstract
    unlike = {**T1_VERSION, "title": "Clustered transformers for book summaries"}
    assert not same_text(t1_record(), unlike)  # same abstract, other title


def test_the_final_list_merges_versions_into_the_earliest_published(cfg: Config) -> None:
    result = check(cfg).result
    first, second, third = result.candidates[:3]

    def listed(t1_date: str, version_date: str) -> Any:
        t1 = {**first.record, **t1_record(), "publication_date": t1_date}
        version = {**second.record, **T1_VERSION, "publication_date": version_date}
        return replace(
            result,
            candidates=[
                Candidate(1, t1, first.hits, first.versions),
                Candidate(2, version, second.hits, [version]),
                third,
            ],
        )

    # The earlier entry is the earlier version: it stays as it is, the other its alias.
    merged, pairs = merge_same_text(listed("2025-01-01", "2025-06-01"))
    assert pairs == [(T1, "W301")]
    assert [c.record["id"] for c in merged.candidates] == [T1, third.record["id"]]
    assert [c.rank for c in merged.candidates] == [1, 2]
    kept = merged.candidates[0]
    assert kept.record["aliases"] == ["W301"] and kept.record["title"] == TITLES[T1]
    assert [v["id"] for v in kept.versions] == [*(v["id"] for v in first.versions), "W301"]
    # The later entry was published first: the entry keeps its place, shows that
    # version, and the earlier entry's id becomes its alias. A tie keeps the earlier.
    merged, pairs = merge_same_text(listed("2025-06-01", "2025-01-01"))
    assert pairs == [("W301", T1)]
    kept = merged.candidates[0]
    assert (kept.rank, kept.record["id"], kept.record["aliases"]) == (1, "W301", [T1])
    assert kept.record["title"] == T1_VERSION["title"]
    assert merge_same_text(listed("2025-03-01", "2025-03-01"))[1] == [(T1, "W301")]
    unmerged = replace(result, candidates=[listed("2025-01-01", "2025-01-01").candidates[0]])
    assert merge_same_text(unmerged) == (unmerged, [])


def test_a_check_shows_one_entry_for_two_versions(cfg: Config) -> None:
    version = oa_work(T1_VERSION)
    fake = FakeLiveOpenAlex([*works(), version])
    run = run_check(
        CheckRequest(text=ABSTRACTS[T1], quick=True), cfg, search=_search(cfg, fake),
        encoder=FakeEncoder(),
    )
    ids = [p.id for p in run.report.papers]
    assert run.merged == [(T1, "W301")] and T1 in ids and "W301" not in ids
    (t1,) = [p for p in run.report.papers if p.id == T1]
    assert t1.aliases == ["W301"]


def test_a_check_reports_each_stage(cfg: Config) -> None:
    stages: list[str] = []
    run_check(
        CheckRequest(text=GOOD[T1]),
        cfg,
        search=_search(cfg, FakeLiveOpenAlex(works())),
        encoder=FakeEncoder(),
        agent=fake_agent(),
        reranker=FakeLLM(original_first),
        explainer=FakeLLM(explainer_reply),
        progress=stages.append,
    )
    agent_queries = cfg.retrieval.max_queries - 1
    assert stages == ["searching", f"agent query 1 of {agent_queries}", "reranking", "explaining"]
    quick: list[str] = []
    run_check(
        CheckRequest(text=GOOD[T1], quick=True),
        cfg,
        search=_search(cfg, FakeLiveOpenAlex(works())),
        encoder=FakeEncoder(),
        progress=quick.append,
    )
    assert quick == ["searching"]


class AgentQueryTimesOut(FakeLiveOpenAlex):
    """OpenAlex timing out (HTTP 504) on the agent's query, every time."""

    status, error = 504, "query_timeout"

    def handle(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if AGENT_QUERY in (params.get("search"), params.get("search.semantic")):
            return httpx.Response(self.status, json={"error": self.error})
        return super().handle(request)


class AgentQueryRejected(AgentQueryTimesOut):
    """OpenAlex rejecting the agent's query (HTTP 400), as it rejects an invalid search."""

    status, error = 400, "Invalid query"


class IdeaSearchRejected(FakeLiveOpenAlex):
    """OpenAlex rejecting every search but the agent's (HTTP 400): the idea's own search.

    Those requests are recorded too.
    """

    status = 400

    def handle(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        query = params.get("search", params.get("search.semantic"))
        if query is not None and query != AGENT_QUERY:
            self.requests.append(params)
            return httpx.Response(self.status, json={"error": "simulated"})
        return super().handle(request)


class IdeaSearchTimesOut(IdeaSearchRejected):
    """OpenAlex timing out (HTTP 504) on every search but the agent's."""

    status = 504


def idea_attempts(fake: FakeLiveOpenAlex) -> int:
    """How many times the fake was sent the search of GOOD[T1]."""
    return sum(r.get("search.semantic") == GOOD[T1] for r in fake.requests)


LONG_TAIL = "Finally, we will release the code, the data and every prompt under an open licence."


def long_idea() -> str:
    """GOOD[T1] made longer than OpenAlex's search limit; it ends with LONG_TAIL."""
    filler = " We will also compare the method with strong baselines on several more datasets."
    text = GOOD[T1]
    while len(text) + len(LONG_TAIL) < SEARCH_MAX_BYTES + 400:
        text += filler
    return f"{text} {LONG_TAIL}"


LONG_IDEA_WARNING = (
    "Your idea is longer than OpenAlex's search limit of 1,500 bytes (about 1,500 characters "
    "of English text), so the first search used its first {n:,} characters; the rest of the "
    "check used the full text."
)


def sent_queries(fake: FakeLiveOpenAlex) -> list[str]:
    """The query of every search the fake was sent (not the keyword lookups by id)."""
    found = [r.get("search", r.get("search.semantic")) for r in fake.requests]
    return [q for q in found if q is not None]


def test_the_tool_skips_a_failed_agent_search_and_evaluation_fails(cfg: Config) -> None:
    run = check(cfg, fake=AgentQueryTimesOut(works()))
    (keyword, semantic) = [w for w in run.report.warnings if "agent search failed" in w]
    assert keyword.startswith("An agent search failed and was skipped (keyword search for")
    assert "semantic search" in semantic and "HTTP 504" in semantic
    assert run.report.papers and run.report.explainer == EXPLAINED
    assert all(s.query != AGENT_QUERY for s in run.result.searches)
    with pytest.raises(OpenAlexError, match="HTTP 504"):
        run_check(
            CheckRequest(text=GOOD[T1]),
            cfg,
            search=_search(cfg, AgentQueryTimesOut(works())),
            encoder=FakeEncoder(),
            agent=fake_agent(),
            reranker=FakeLLM(original_first),
            explainer=FakeLLM(explainer_reply),
            strict=True,
        )


def test_the_tool_skips_a_rejected_agent_search(cfg: Config) -> None:
    run = check(cfg, fake=AgentQueryRejected(works()))
    keyword, semantic = [w for w in run.report.warnings if "agent search failed" in w]
    assert "keyword search" in keyword and "semantic search" in semantic
    assert all("HTTP 400" in w for w in (keyword, semantic))
    assert run.report.papers and run.report.explainer == EXPLAINED
    assert all(s.query != AGENT_QUERY for s in run.result.searches)


def test_a_rejected_search_of_the_idea_is_skipped_unless_nothing_else_is_found(
    cfg: Config, tmp_path: Path
) -> None:
    assert cfg.live is not None
    nothing = replace(cfg, live=replace(cfg.live, cache_path=tmp_path / "empty.sqlite"))
    run = check(cfg, fake=IdeaSearchRejected(works()))
    (rejected,) = [w for w in run.report.warnings if "search of your idea" in w]
    assert rejected.startswith(
        "OpenAlex rejected the search of your idea, so it was skipped (semantic search: HTTP 400"
    )
    assert rejected.endswith("the results come from the agent's searches.")
    assert run.report.papers and {s.query for s in run.result.searches} == {AGENT_QUERY}
    # Nothing else found (a cache of its own, without the agent's results): the check fails.
    with pytest.raises(OpenAlexError, match="the agent's searches found nothing: HTTP 400"):
        check(nothing, fake=IdeaSearchRejected([]))
    # Another failure of the idea's own search still fails the check (503: retried, then
    # given up), and evaluation fails on a rejected one.
    with pytest.raises(OpenAlexError, match="HTTP 503"):
        check(cfg, fake=FakeLiveOpenAlex(works(), fail_with=[503] * 6))
    with pytest.raises(OpenAlexError, match="HTTP 400"):
        run_check(
            CheckRequest(text=GOOD[T1]),
            cfg,
            search=_search(cfg, IdeaSearchRejected(works())),
            encoder=FakeEncoder(),
            agent=fake_agent(),
            reranker=FakeLLM(original_first),
            explainer=FakeLLM(explainer_reply),
            strict=True,
        )


def test_a_timed_out_search_of_the_idea_is_retried_once_then_skipped(
    cfg: Config, tmp_path: Path
) -> None:
    assert cfg.live is not None
    nothing = replace(cfg, live=replace(cfg.live, cache_path=tmp_path / "empty.sqlite"))
    fake = IdeaSearchTimesOut(works())
    run = check(cfg, fake=fake)
    (skipped,) = [w for w in run.report.warnings if "search of your idea" in w]
    assert skipped.startswith(
        "OpenAlex timed out on the search of your idea, so it was skipped (semantic search: "
        "giving up after 2 attempts"
    )
    assert idea_attempts(fake) == 2  # one retry
    assert run.report.papers and {s.query for s in run.result.searches} == {AGENT_QUERY}
    # Evaluation keeps the client's usual 5 retries, then fails.
    fake = IdeaSearchTimesOut(works())
    with pytest.raises(OpenAlexError, match="giving up after 6 attempts"):
        run_check(
            CheckRequest(text=GOOD[T1]),
            cfg,
            search=_search(cfg, fake),
            encoder=FakeEncoder(),
            agent=fake_agent(),
            reranker=FakeLLM(original_first),
            explainer=FakeLLM(explainer_reply),
            strict=True,
        )
    assert idea_attempts(fake) == 6
    # Nothing else found (a cache of its own): the check fails.
    with pytest.raises(OpenAlexError, match="timed out on the search of your idea, and the"):
        check(nothing, fake=IdeaSearchTimesOut([]))
    # One timeout, then an answer: the retry finds it, and nothing is skipped.
    fake = FakeLiveOpenAlex(works(), fail_with=[504])
    retried = check(cfg, fake=fake)
    assert idea_attempts(fake) == 2 and not retried.report.warnings
    assert GOOD[T1] in {s.query for s in retried.result.searches}


def test_an_idea_over_the_search_limit_runs_and_warns(cfg: Config) -> None:
    assert cfg.live is not None and cfg.live.max_search_bytes == SEARCH_MAX_BYTES
    text = long_idea()
    sent = search_text(text, SEARCH_MAX_BYTES)
    assert len(text) > SEARCH_MAX_BYTES and text.startswith(sent) and sent.endswith(".")
    fake = FakeLiveOpenAlex(works())  # rejects a search over the limit, as OpenAlex does
    agent, reranker, explainer = fake_agent(), FakeLLM(original_first), FakeLLM(explainer_reply)
    run = run_check(
        CheckRequest(text=text),
        cfg,
        search=_search(cfg, fake),
        encoder=FakeEncoder(),
        agent=agent,
        reranker=reranker,
        explainer=explainer,
    )
    warning = LONG_IDEA_WARNING.format(n=len(sent))
    assert run.report.warnings == [warning]
    assert run.report.papers and run.report.explainer == EXPLAINED
    assert sent_queries(fake)[0] == sent and LONG_TAIL not in sent
    # The full text everywhere else: the agent, the rerank and explainer calls, the report.
    for llm in (agent, reranker, explainer):
        assert llm.calls and all(LONG_TAIL in c[-1]["content"] for c in llm.calls)
    assert run.result.trace.queries[0] == run.report.request.text == text
    # The quick check warns the same way.
    quick = run_check(
        CheckRequest(text=text, quick=True),
        cfg,
        search=_search(cfg, FakeLiveOpenAlex(works())),
        encoder=FakeEncoder(),
    )
    assert quick.report.warnings == [warning] and quick.report.papers


def test_no_query_is_sent_longer_than_the_search_limit(cfg: Config) -> None:
    long_query = " ".join([AGENT_QUERY] * 60)  # 2,039 characters, no sentence end

    def agent_reply(messages: list[dict[str, str]]) -> str:
        if '\n2. "' in messages[-1]["content"]:
            return json.dumps({"action": "stop", "reason": "covered"})
        return json.dumps({"action": "search", "query": long_query})

    fake = FakeLiveOpenAlex(works())
    run = run_check(
        CheckRequest(text=long_idea()),
        cfg,
        search=_search(cfg, fake),
        encoder=FakeEncoder(),
        agent=FakeLLM(agent_reply),
        reranker=FakeLLM(original_first),
        explainer=FakeLLM(explainer_reply),
    )
    queries = sent_queries(fake)
    assert len(queries) == 2
    assert all(len(q.encode("utf-8")) <= SEARCH_MAX_BYTES for q in queries)
    assert queries[1] == search_text(long_query, SEARCH_MAX_BYTES)  # cut, not skipped
    assert [s.query for s in run.result.searches] == [long_idea(), long_query]
    assert len(run.report.warnings) == 1  # the idea's: an agent query is cut silently


def test_nothing_after_the_search_fails_the_check(cfg: Config) -> None:
    def down(messages: list[dict[str, str]]) -> str:
        raise LLMError("upstream 503")

    fake = FakeLiveOpenAlex(works(), fail_lookups=True)
    run = check(cfg, fake=fake, explainer=FakeLLM(down))
    report = run.report
    assert report.explainer == FALLBACK and len(report.warnings) == 2
    assert report.warnings[0].startswith("No OpenAlex keywords for the vocabulary map, or")
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
    kwargs.setdefault("explainer", FakeLLM(explainer_reply))
    kwargs.setdefault("reranker", FakeLLM(original_first))
    return run_explain_eval(
        cfg,
        search=_search(cfg, FakeLiveOpenAlex(works())),
        encoder=FakeEncoder(),
        agent=fake_agent(),
        from_qid="A0000",
        **kwargs,
    )


def test_explain_eval_measures_quotes_overlap_cost_and_time(bench: Config, tmp_path: Path) -> None:
    assert bench.live is not None and bench.explain is not None
    # Step 2's live run (L5 rows only, as the dev run has) and Step 4's rerank run on the
    # same checks, for the timing.
    live = run_live_eval(
        bench, fake_agent(), FakeEncoder(), _search(bench, FakeLiveOpenAlex(works())),
        configs=["L5"], log_mlflow=False,
    )
    rerank = run_rerank(
        bench, live.run_dir, encoder=FakeEncoder(),
        clients={PRODUCT_MODEL: FakeLLM(original_first)}, variants=[R1],
    )
    times = recorded_times(rerank.run_dir, bench.run.results_db)
    rows_l5 = _rows(bench, live.run_dir).values()
    l5 = {row["qid"]: row["latency_s"] for row in rows_l5 if row["config"] == "L5"}
    assert {q: t["check_s"] for q, t in times.items()} == l5
    assert len(times) == 3 and all(t["rerank_s"] for t in times.values())
    def mixed(messages: list[dict[str, str]]) -> str:
        """Paper 1's quote exact, paper 2's edited (its first word dropped), 3's invented."""
        answer = json.loads(explainer_reply(messages))
        second = answer["papers"][1]["evidence"].split(" ", 1)[1]
        answer["papers"][1]["evidence"] = second[0].upper() + second[1:]
        answer["papers"][2]["evidence"] = "A sentence that no abstract of these papers has."
        return json.dumps(answer)

    done = evaluate(
        bench, tmp_path, limit=3, review_n=2, timing_run=rerank.run_dir,
        explainer=FakeLLM(mixed),
    )
    s = done.summary
    assert (s["n_checks"], s["n_ok"], s["over_cap"]) == (3, 3, 0)
    third = pytest.approx(1 / 3)
    assert s["quotes"] == {
        "papers": 9,
        "explained": 9,
        "exact": 3,
        "matched": 3,
        "unverified": 3,
        "rates": {"exact": third, "matched": third, "unverified": third},
    }
    original = s["original"]
    assert original["in_top"] + original["not_in_top"] == 3 and original["in_top"] >= 1
    assert original["levels"]["high"] == original["in_top"] and original["high_rate"] == 1.0
    assert original["quotes"] == {  # the original is first, with the exact quote
        VERIFIED_EXACT: original["in_top"], VERIFIED_MATCHED: 0, UNVERIFIED: 0
    }
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
    assert [r["verified"] for r in explained] == ["exact", "matched", "no"] * 2
    assert all(r["evidence"] and not r["evidence"].startswith("[") for r in explained[:2])
    assert explained[2]["evidence"].startswith("[snippet] ")
    assert all(r["evidence"] == r["verified"] == "" for r in review if r not in explained)
    with pytest.raises(EvalError, match="never overwritten"):
        evaluate(bench, tmp_path, limit=1)


def test_explain_eval_counts_two_list_checks_and_times_r4(tmp_path: Path) -> None:
    data = config_dict(tmp_path, contribution_top_k=3, methods_top_k=2)
    data["rerank"]["two_lists"] = True
    bench = config_from_dict(data)
    freeze(bench.corpus.out_dir, dated_corpus(), target_rows(), {})
    assert make_bench(bench, _paraphraser(), FakeLLM(lambda m: judge_reply()))["accepted"] == 3
    _phase1_run(bench, B1_RANKS, B5_RANKS)
    live = run_live_eval(
        bench, fake_agent(), FakeEncoder(), _search(bench, FakeLiveOpenAlex(works())),
        configs=["L5"], log_mlflow=False,
    )
    # The dev rerank run with R4 gives the rerank call's time; it has no R1 rows.
    rerank = run_rerank(
        bench, live.run_dir, encoder=FakeEncoder(),
        clients={PRODUCT_MODEL: FakeLLM(two_lists_reply)}, variants=[R4],
    )
    times = recorded_times(rerank.run_dir, bench.run.results_db, R4)
    assert len(times) == 3 and all(t["rerank_s"] for t in times.values())
    r1_times = recorded_times(rerank.run_dir, bench.run.results_db).values()
    assert all(t["rerank_s"] is None for t in r1_times)
    done = evaluate(
        bench, tmp_path, limit=3, review_n=1, timing_run=rerank.run_dir,
        reranker=FakeLLM(two_lists_reply),
    )
    s = done.summary
    assert (s["n_checks"], s["n_ok"]) == (3, 3)
    two = s["two_lists"]
    assert two["checks"] == 3
    assert two["methods_list"] == {"reranked": 3, "fallback": 0, "skipped": 0}
    lines = (done.run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    for row in rows:
        # List A's top 3, and section B: list B's first 2 not among them, after list A's
        # second (on the "Also close in methods" line). The original is list B's last.
        assert row["lists"] == 2 and row["n_methods"] == 2 and row["n_also_close"] == 1
        assert row["n_top"] == 5 == row["n_explained"]
        assert row["original_in_top"] and row["original_also_close"] is False
        q = times[row["qid"]]
        assert row["estimated_live_s"] == pytest.approx(
            q["check_s"] + q["rerank_s"] + row["explainer_stage_s"]
        )
    assert s["quotes"]["papers"] == 15 and two["explained"]["max"] == 5
    assert two["section_b"]["max"] == 2 and two["also_close"]["max"] == 1
    assert (two["original_in_methods"], two["original_also_close"]) == (0, 0)
    meta = json.loads((done.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["two_lists"] is True and meta["timing_config"] == R4
    assert TWO_LISTS in meta["prompts"] and "rerank_schema" in meta["prompts"]
    assert "## Two lists (issue #22)" in (done.run_dir / "summary.md").read_text(encoding="utf-8")


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
    assert "rk explain-eval: 2 of 2 checks; quotes exact 100.0%, matched 0.0%" in out
    assert models == [bench.agent.name, "fake/reranker", "fake/explainer"]
    assert out_csv.exists() and "(5 rows)" in out
