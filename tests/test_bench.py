from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from bench_data import (
    ABSTRACTS,
    GOOD,
    LEAKY,
    T1,
    T2,
    T3,
    TITLES,
    corpus_records,
    judge_reply,
    target_rows,
)
from conftest import FakeLLM, corpus_config_dict

from rk import cli
from rk.bench import (
    ACCEPTED_FILE,
    CORRUPTIONS,
    HUMAN_CHECK_COLUMNS,
    JUDGE_RESPONSE_FORMAT,
    JUDGE_SCHEMA_SHA256,
    REASONS,
    REJECTIONS_FILE,
    STATS_FILE,
    BenchError,
    LeakFilter,
    _visit_order,
    check_length,
    export_human_check,
    keep_names_line,
    load_prompts,
    make_bench,
    make_qid,
    not_research_contribution,
    parse_judgment,
    validate_judge,
)
from rk.config import Config, config_from_dict
from rk.corpus import CORPUS_FILE, CorpusError, freeze
from rk.llm import load_prompt
from rk.text import stem

YES = judge_reply()
NO = judge_reply({"method": "the method differs"})
PROMPTS = load_prompts()
CAPTURES_FIRST_LINE = PROMPTS["judge_captures"].template.template.splitlines()[0]
SEED = 7  # make_model_config's seed, so paraphraser.seed in these configs


# --- helpers -----------------------------------------------------------------


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    path = tmp_path / "corpus"
    freeze(path, corpus_records(), target_rows(), {})
    return path


def _config_dict(tmp_path: Path, **bench: Any) -> dict[str, Any]:
    data = corpus_config_dict(tmp_path / "corpus")
    data["bench"].update({"n_targets": 3, "out_dir": str(tmp_path / "bench"), **bench})
    return data


def _config(tmp_path: Path, **bench: Any) -> Config:
    return config_from_dict(_config_dict(tmp_path, **bench))


def _target_in(prompt: str, label: str) -> str:
    return next(t for t, title in TITLES.items() if f"{label}: {title}" in prompt)


def paraphraser(**scripts: list[str | Exception]) -> FakeLLM:
    """Replies per target id (default: its GOOD proposal); an Exception entry is raised."""
    queues = {t: list(scripts.get(t, [])) for t in TITLES}

    def reply(messages: list[dict[str, str]]) -> str:
        target = _target_in(messages[-1]["content"], "Title")
        item = queues[target].pop(0) if queues[target] else GOOD[target]
        if isinstance(item, Exception):
            raise item
        return item

    return FakeLLM(reply)


def judge(replies: dict[tuple[str, str], list[str]] | None = None) -> FakeLLM:
    """Replies per (target, direction), then YES."""
    queues = {key: list(value) for key, value in (replies or {}).items()}

    def reply(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        direction = "captures" if prompt.startswith(CAPTURES_FIRST_LINE) else "adds_nothing"
        queue = queues.get((_target_in(prompt, "Paper title"), direction), [])
        return queue.pop(0) if queue else YES

    return FakeLLM(reply)


def _rows(cfg: Config, name: str) -> list[dict[str, Any]]:
    path = cfg.bench.out_dir / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _stats(cfg: Config) -> dict[str, Any]:
    return json.loads((cfg.bench.out_dir / STATS_FILE).read_text(encoding="utf-8"))


def _target_calls(fake: FakeLLM) -> list[tuple[str, int | None]]:
    """(target, seed) for each paraphraser call, in order."""
    return [(_target_in(c[-1]["content"], "Title"), s) for c, s in zip(fake.calls, fake.seeds)]


ORDER = [t["id"] for t in _visit_order(target_rows(), 7)]


# --- the happy path ----------------------------------------------------------


def test_accepts_clean_proposals_and_writes_full_rows(tmp_path: Path, corpus_dir: Path) -> None:
    cfg = _config(tmp_path)
    para, jdg = paraphraser(), judge()

    stats = make_bench(cfg, para, jdg)

    rows = _rows(cfg, ACCEPTED_FILE)
    assert [r["target_id"] for r in rows] == ORDER  # seeded visiting order
    hashes = {name: p.sha256 for name, p in PROMPTS.items()}
    hashes["judge_schema"] = JUDGE_SCHEMA_SHA256
    for row in rows:
        assert row["qid"] == make_qid(ORDER.index(row["target_id"]), row["target_id"])
        assert row["query"] == GOOD[row["target_id"]]
        assert row["target_aliases"] == []
        assert row["attempt"] == 0
        assert row["prompt_hashes"] == hashes
        assert {d: v["verdict"] for d, v in row["judge_verdicts"].items()} == {
            "captures": "yes",
            "adds_nothing": "yes",
        }
        assert row["leak_stats"]["content_jaccard"] < 0.15
    assert _rows(cfg, REJECTIONS_FILE) == []
    assert stats["accepted"] == 3 and stats["rejected"] == 0
    assert stats["prompt_hashes"] == hashes
    assert len(para.calls) == 3 and len(jdg.calls) == 6  # two separate judge calls each
    assert stats == _stats(cfg)
    # Sorting qids reproduces the seeded visiting order, so a qid prefix is a random sample.
    assert [r["target_id"] for r in sorted(rows, key=lambda r: r["qid"])] == ORDER
    assert ORDER != sorted(ORDER)  # (the seeded order isn't just id order here)


def test_rows_and_stats_record_who_served_each_call(tmp_path: Path, corpus_dir: Path) -> None:
    cfg = _config(tmp_path)
    para, jdg = paraphraser(**{ORDER[0]: [LEAKY["length"]]}), judge()
    para.provider, jdg.provider = "Novita", "OpenAI"

    stats = make_bench(cfg, para, jdg)

    (rejected,) = _rows(cfg, REJECTIONS_FILE)
    assert rejected["providers"] == {"paraphraser": ["Novita"], "judge": []}  # never judged
    for row in _rows(cfg, ACCEPTED_FILE):
        assert row["providers"] == {"paraphraser": ["Novita"], "judge": ["OpenAI", "OpenAI"]}
    assert stats["llm_providers"] == {"paraphraser": {"Novita": 4}, "judge": {"OpenAI": 6}}


# --- targets that aren't research contributions --------------------------------

# The pilot's 4 surveys/reviews/tutorials and its 5 papers merely *about* reviews:
# titles and every abstract sentence mentioning survey/review/tutorial/PRISMA.
SURVEY_CASES = json.loads(
    (Path(__file__).parent / "survey_rule_cases.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", SURVEY_CASES, ids=lambda case: case["qid"])
def test_self_description_rule_on_the_pilot_targets(case: dict[str, Any]) -> None:
    phrase = not_research_contribution(case["title"], case["abstract_excerpt"])
    assert (phrase is not None) == case["skip"], phrase


@pytest.mark.parametrize(
    ("title", "abstract", "phrase"),
    [
        ("Hallucinations in LLMs: a lifecycle-based survey", "We study causes.", "survey"),
        ("Methods for X", "Our review covers 40 papers.", "Our review"),
        ("Methods for X", "In this paper we survey recent work.", "we survey"),
        ("Methods for X", "Following PRISMA guidelines, we searched four databases.", "PRISMA"),
        ("Methods for X", "The reviewer assigned this reviewer score.", None),  # not "review"
        ("Methods for X", "We collect systematic reviews and peer reviews of papers.", None),
        ("Methods for X", "Prisma is a prism-shaped toy.", None),  # case matters for PRISMA
    ],
)
def test_self_description_patterns(title: str, abstract: str, phrase: str | None) -> None:
    assert not_research_contribution(title, abstract) == phrase


def _survey_config(tmp_path: Path, survey: str) -> Config:
    records = corpus_records()
    for record in records:
        if record["id"] == survey:
            record["abstract"] = "This survey compares earlier work. " + record["abstract"]
    freeze(tmp_path / "corpus", records, target_rows(), {})
    return _config(tmp_path)


def test_surveys_are_skipped_before_paraphrasing_logged_once_and_counted(tmp_path: Path) -> None:
    survey = ORDER[0]  # visited first
    cfg = _survey_config(tmp_path, survey)
    para = paraphraser()

    stats = make_bench(cfg, para, judge(), limit=2)  # a skipped target doesn't use the limit

    assert {r["target_id"] for r in _rows(cfg, ACCEPTED_FILE)} == set(ORDER[1:])
    assert survey not in {_target_in(c[-1]["content"], "Title") for c in para.calls}
    (skip,) = _rows(cfg, REJECTIONS_FILE)
    assert (skip["target_id"], skip["reason"]) == (survey, "not_research_contribution")
    assert "'This survey'" in skip["detail"]
    assert stats["rejected_by_reason"]["not_research_contribution"] == stats["skipped"] == 1
    assert (stats["accepted"], stats["rejected"], stats["attempts"], stats["exhausted"]) == (
        2,
        0,
        2,
        0,
    )
    assert survey not in stats["attempts_per_target"]

    again = make_bench(cfg, paraphraser(), judge())  # resuming neither re-checks nor re-logs it
    assert len(_rows(cfg, REJECTIONS_FILE)) == 1 and again["skipped"] == 1


# --- rejections --------------------------------------------------------------


@pytest.mark.parametrize("reason", list(LEAKY))
def test_each_length_and_leak_reason_is_recorded(
    reason: str, tmp_path: Path, corpus_dir: Path
) -> None:
    cfg = _config(tmp_path)
    jdg = judge()

    make_bench(cfg, paraphraser(**{T1: [LEAKY[reason]]}), jdg)

    (rejection,) = _rows(cfg, REJECTIONS_FILE)
    assert (rejection["target_id"], rejection["attempt"], rejection["reason"]) == (T1, 0, reason)
    assert rejection["query"] == LEAKY[reason]
    assert rejection["judge_verdicts"] == {}  # the judge never saw it
    accepted = {r["target_id"]: r["attempt"] for r in _rows(cfg, ACCEPTED_FILE)}
    assert accepted == {T1: 1, T2: 0, T3: 0}
    assert len(jdg.calls) == 6


@pytest.mark.parametrize("direction", ["captures", "adds_nothing"])
def test_judge_no_in_either_direction_rejects(
    direction: str, tmp_path: Path, corpus_dir: Path
) -> None:
    cfg = _config(tmp_path)
    jdg = judge({(T2, direction): [NO]})

    make_bench(cfg, paraphraser(), jdg)

    (rejection,) = _rows(cfg, REJECTIONS_FILE)
    assert (rejection["target_id"], rejection["reason"]) == (T2, f"judge_{direction}")
    assert rejection["detail"].endswith("paper's method (method: the method differs)")
    assert rejection["judge_verdicts"][direction]["verdict"] == "no"
    assert rejection["judge_verdicts"][direction]["failed"] == ["method"]
    if direction == "captures":  # the second question isn't asked after a "no"
        assert "adds_nothing" not in rejection["judge_verdicts"]
    assert {r["target_id"] for r in _rows(cfg, ACCEPTED_FILE)} == {T1, T2, T3}


@pytest.mark.parametrize("direction", ["captures", "adds_nothing"])
@pytest.mark.parametrize("reply", ["Yes, it captures it.", '{"verdict": "maybe"}', "[1, 2]"])
def test_unreadable_judge_reply_is_a_rejection_never_a_pass(
    direction: str, reply: str, tmp_path: Path, corpus_dir: Path
) -> None:
    cfg = _config(tmp_path)

    make_bench(cfg, paraphraser(), judge({(T3, direction): [reply]}))

    (rejection,) = _rows(cfg, REJECTIONS_FILE)
    assert (rejection["target_id"], rejection["reason"]) == (T3, "judge_parse_error")
    assert rejection["judge_verdicts"][direction] == {"verdict": None, "raw": reply}


def test_retry_uses_the_next_seed_and_quotes_the_rejection(
    tmp_path: Path, corpus_dir: Path
) -> None:
    cfg = _config(tmp_path)
    para = paraphraser(**{T1: [LEAKY["title_stem"]]})

    make_bench(cfg, para, judge())

    t1_calls = [(i, s) for i, (t, s) in enumerate(_target_calls(para)) if t == T1]
    assert [s for _, s in t1_calls] == [SEED, SEED + 1]
    first, retry = (para.calls[i][-1]["content"] for i, _ in t1_calls)
    assert "previous proposal was rejected" not in first
    because = "rejected because it used rare or distinctive words from the title"
    assert f"previous proposal was {because}" in retry
    assert "(documents, long)" in retry


def test_target_is_exhausted_after_max_attempts(tmp_path: Path, corpus_dir: Path) -> None:
    cfg = _config(tmp_path, max_attempts_per_target=3)

    stats = make_bench(cfg, paraphraser(**{T1: [LEAKY["length"]] * 3}), judge())

    rejections = _rows(cfg, REJECTIONS_FILE)
    assert [(r["target_id"], r["attempt"]) for r in rejections] == [(T1, 0), (T1, 1), (T1, 2)]
    assert {r["target_id"] for r in _rows(cfg, ACCEPTED_FILE)} == {T2, T3}
    assert stats["exhausted"] == 1
    assert stats["attempts_per_target"][T1] == 3


def test_stops_once_n_targets_are_accepted(tmp_path: Path, corpus_dir: Path) -> None:
    cfg = _config(tmp_path, n_targets=2)
    para = paraphraser()

    stats = make_bench(cfg, para, judge())

    assert [r["target_id"] for r in _rows(cfg, ACCEPTED_FILE)] == ORDER[:2]
    assert [t for t, _ in _target_calls(para)] == ORDER[:2]  # the third is never attempted
    assert stats["accepted"] == 2


# --- resuming ----------------------------------------------------------------


def test_resume_skips_accepted_and_exhausted_targets(tmp_path: Path, corpus_dir: Path) -> None:
    first, second, third = ORDER
    cfg = _config(tmp_path, max_attempts_per_target=2)
    make_bench(cfg, paraphraser(), judge(), limit=1)  # accepts `first`
    make_bench(cfg, paraphraser(**{second: [LEAKY["length"]] * 2}), judge(), limit=1)

    later = paraphraser()
    stats = make_bench(cfg, later, judge())

    assert [t for t, _ in _target_calls(later)] == [third]  # first accepted, second exhausted
    assert [r["target_id"] for r in _rows(cfg, ACCEPTED_FILE)] == [first, third]
    assert (stats["accepted"], stats["exhausted"], stats["attempts"]) == (2, 1, 4)


def test_resume_continues_an_interrupted_target(tmp_path: Path, corpus_dir: Path) -> None:
    first = ORDER[0]
    cfg = _config(tmp_path)
    crashing = paraphraser(**{first: [LEAKY["length"], RuntimeError("connection lost")]})
    with pytest.raises(RuntimeError, match="connection lost"):
        make_bench(cfg, crashing, judge())
    assert _stats(cfg)["rejected"] == 1  # stats are written even when the run dies

    later = paraphraser()
    make_bench(cfg, later, judge())

    target, seed = _target_calls(later)[0]
    assert (target, seed) == (first, SEED + 1)
    assert "it had 1 sentences and 12 words" in later.calls[0][-1]["content"]
    accepted = {r["target_id"]: r["attempt"] for r in _rows(cfg, ACCEPTED_FILE)}
    assert accepted[first] == 1


def test_refuses_to_resume_with_other_prompts(tmp_path: Path, corpus_dir: Path) -> None:
    cfg = _config(tmp_path)
    make_bench(cfg, paraphraser(), judge(), limit=1)
    stats_path = cfg.bench.out_dir / STATS_FILE
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["prompt_hashes"]["paraphrase"] = "0" * 64
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(BenchError, match="different prompt hashes"):
        make_bench(cfg, paraphraser(), judge())


def test_needs_an_intact_frozen_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with pytest.raises(CorpusError, match="run `python -m rk build-corpus` first"):
        make_bench(cfg, paraphraser(), judge())

    freeze(cfg.corpus.out_dir, corpus_records(), target_rows(), {})
    with (cfg.corpus.out_dir / CORPUS_FILE).open("a", encoding="utf-8") as fh:
        fh.write("{}\n")
    with pytest.raises(CorpusError, match="doesn't match the sha256"):
        make_bench(cfg, paraphraser(), judge())


# --- stats -------------------------------------------------------------------


def test_stats_add_up(tmp_path: Path, corpus_dir: Path) -> None:
    cfg = _config(tmp_path, max_attempts_per_target=3)
    para = paraphraser(
        **{T1: [LEAKY["length"], LEAKY["title_stem"]], T3: [LEAKY["length"]] * 3}
    )
    jdg = judge({(T2, "adds_nothing"): [NO, "garbled"]})

    stats = make_bench(cfg, para, jdg)

    # T1 passes on attempt 2, T2 too (after a "no" and an unreadable reply), T3 runs out.
    rows = _rows(cfg, ACCEPTED_FILE) + _rows(cfg, REJECTIONS_FILE)
    assert (stats["accepted"], stats["rejected"], stats["exhausted"]) == (2, 7, 1)
    assert stats["accepted"] + stats["rejected"] == stats["attempts"] == len(rows) == 9
    assert sum(stats["rejected_by_reason"].values()) == stats["rejected"]
    assert list(stats["rejected_by_reason"]) == list(REASONS)
    assert stats["rejected_by_reason"]["length"] == 4
    assert stats["rejected_by_reason"]["title_stem"] == 1
    assert stats["rejected_by_reason"]["judge_adds_nothing"] == 1
    assert stats["rejected_by_reason"]["judge_parse_error"] == 1
    assert stats["attempts_per_target"] == {T1: 3, T2: 3, T3: 3}
    assert sum(stats["attempts_per_target"].values()) == stats["attempts"]
    usage = stats["llm_usage"]
    assert usage["paraphraser"]["calls"] == len(para.calls) == stats["attempts"]
    assert usage["judge"]["calls"] == len(jdg.calls)
    total = sum(r["usage"][role]["method_cost"] for r in rows for role in ("paraphraser", "judge"))
    assert stats["method_cost"] == pytest.approx(total) and total > 0


# --- units -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "verdict", "failed"),
    [
        (YES, "yes", []),
        ("```json\n" + YES + "\n```", "yes", []),  # fenced
        (judge_reply({"domain": "English became a foreign tongue"}), "no", ["domain"]),
        # A facet marked "no" rejects even when the judge's own verdict says yes ...
        (judge_reply({"claim": "reversed"}, verdict="yes"), "no", ["claim"]),
        # ... and a "no" verdict stands even if every facet matched.
        (judge_reply(verdict="no", reason="too vague overall"), "no", []),
        (judge_reply({"problem": "a", "method": "b"}), "no", ["problem", "method"]),
    ],
)
def test_parse_judgment(text: str, verdict: str, failed: list[str]) -> None:
    judgment = parse_judgment(text)
    assert judgment is not None
    assert (judgment["verdict"], judgment["failed"]) == (verdict, failed)
    assert set(judgment["match"]) == {"problem", "method", "domain", "claim"}


def _without(facet: str) -> str:
    data = json.loads(YES)
    del data["match"][facet]
    return json.dumps(data)


@pytest.mark.parametrize(
    "text",
    [
        '{"verdict": "yes", "reason": "the old one-line format"}',
        _without("domain"),  # every facet needs a yes or no
        YES.replace('"verdict": "yes"', '"verdict": "maybe"'),
        YES.replace('"match": "yes"', '"match": "partly"', 1),
        "yes",
        "[1, 2]",
    ],
)
def test_unreadable_judgments(text: str) -> None:
    assert parse_judgment(text) is None


def test_judge_rejection_quotes_each_failed_facet_for_the_retry(
    tmp_path: Path, corpus_dir: Path
) -> None:
    cfg = _config(tmp_path)
    failed = {"domain": "a foreign tongue is not English", "claim": "the claim is reversed"}
    para = paraphraser()

    make_bench(cfg, para, judge({(T2, "captures"): [judge_reply(failed)]}))

    (rejection,) = _rows(cfg, REJECTIONS_FILE)
    assert rejection["detail"] == (
        "a reviewer found that it does not keep the paper's domain or population, main claim "
        "(domain or population: a foreign tongue is not English; main claim: the claim is reversed)"
    )
    retry = [c[-1]["content"] for c in para.calls if TITLES[T2] in c[-1]["content"]][-1]
    assert f"rejected because {rejection['detail']}." in retry
    stats = _stats(cfg)
    assert stats["judge_failed_facets"]["captures"] == {
        "problem": 0, "method": 0, "domain": 1, "claim": 1,
    }


@pytest.mark.parametrize(
    ("n_sentences", "n_words", "ok"),
    [
        (2, 40, True),
        (4, 120, True),
        (1, 60, False),
        (5, 60, False),
        (3, 39, False),
        (3, 121, False),
    ],
)
def test_length_bounds_are_inclusive(n_sentences: int, n_words: int, ok: bool) -> None:
    words = ["word"] * n_words
    per = n_words // n_sentences
    chunks = [words[i * per : (i + 1) * per] for i in range(n_sentences - 1)]
    chunks.append(words[(n_sentences - 1) * per :])
    text = " ".join(" ".join(chunk) + "." for chunk in chunks)
    rejection, stats = check_length(text)
    assert (stats["n_sentences"], stats["n_words"]) == (n_sentences, n_words)
    assert (rejection is None) is ok


def test_source_acronyms_that_spell_stopwords_are_not_matched() -> None:
    leak = LeakFilter(corpus_records(), max_overlap=0.15, max_jaccard=1.0)
    abstract = ABSTRACTS[T1] + " We deploy it in US hospitals."
    rejection, stats = leak.check(GOOD[T1] + " It helps us.", TITLES[T1], abstract)
    assert stats["source_acronyms_found"] == []
    assert rejection is None


def test_content_jaccard_ignores_generic_science_words() -> None:
    leak = LeakFilter(corpus_records(), max_overlap=1.0, max_jaccard=1.0)
    abstract = "We propose a novel method and show results on data."  # all generic
    _, stats = leak.check(
        "We present an approach and evaluate the method on a dataset.", "Generic words", abstract
    )
    assert stats["content_jaccard"] == 0.0


def test_leak_filter_lets_a_proposal_keep_the_titles_language_and_country() -> None:
    leak = LeakFilter(corpus_records(), max_overlap=1.0, max_jaccard=1.0)
    title = "Dense retrieval for Turkish legal documents in the UK"
    abstract = "We study Turkish court rulings filed in the UK and in Turkey."
    rejection, stats = leak.check(
        "We plan to search Turkish court rulings from the UK with learned vectors.", title, abstract
    )
    assert rejection is None
    assert stats["acronyms_in_paraphrase"] == stats["title_words_found"] == []
    assert stem("turkish") not in stats["idf_top_terms"]  # never a leaked term either


@pytest.mark.parametrize(
    ("title", "proposal", "reason", "found"),
    [  # organisation and model names stay banned
        ("Classifying Cochrane summaries by conclusiveness",
         "We will sort Cochrane overviews by how firm their conclusions are.",
         "title_stem", "cochrane"),
        ("TurkColBERT for Turkish retrieval",
         "We adapt TurkColBERT to search Turkish text.",
         "acronym_in_paraphrase", "TurkColBERT"),
    ],
)
def test_leak_filter_still_bans_organisation_and_model_names(
    title: str, proposal: str, reason: str, found: str
) -> None:
    leak = LeakFilter(corpus_records(), max_overlap=1.0, max_jaccard=1.0)
    rejection, _ = leak.check(proposal, title, "An abstract.")
    assert rejection is not None and rejection.reason == reason and found in rejection.message


def test_paraphrase_prompt_tells_the_model_which_title_names_to_keep(
    tmp_path: Path,
) -> None:
    assert keep_names_line("Calibrated reward models") == ""
    assert keep_names_line("Turkish retrieval for maritime reports in the Middle East") == (
        "\nThe proposal must keep these words from the title: Turkish, maritime, Middle East.\n"
    )
    records = corpus_records()
    for record in records:
        if record["id"] == T2:
            record["title"] = TITLES[T2] + " in Kazakh clinical notes"
    freeze(tmp_path / "corpus", records, target_rows(), {})
    para = paraphraser()

    make_bench(_config(tmp_path), para, judge())

    prompts = {_target_in(c[-1]["content"], "Title"): c[-1]["content"] for c in para.calls}
    assert "The proposal must keep these words from the title: Kazakh, clinical." in prompts[T2]
    assert "must keep these words" not in prompts[T1]
    avoid_rare = (
        "Avoid the rare or distinctive words of the title, in any form; common field and task "
        "terms (such as translation, question answering or knowledge graph) are fine."
    )
    keep_terms = (
        "If the title names a language, nationality, country or region, or a research field or "
        "application domain (such as physics, maritime or clinical), the proposal must keep"
    )
    assert avoid_rare in prompts[T1] and keep_terms in prompts[T1]


def _translation_corpus() -> list[dict[str, Any]]:
    """The fixture corpus plus 20 documents about translation: a common task word here."""
    extra = [
        {"id": f"W{700 + i}", "title": f"Translation study {i}",
         "abstract": f"We improve translation quality for setting {i}."}
        for i in range(20)
    ]
    return [*corpus_records(), *extra]


COMMON_ONLY = "We will improve translation by inferring the likeliest explanation."


@pytest.mark.parametrize(
    ("max_df", "proposal", "banned"),
    [
        (0.2, COMMON_ONLY, []),
        (0.2, "We will apply abduction to improve how sentences are rendered.", ["abduction"]),
        (1.0, COMMON_ONLY, ["translation"]),  # 1.0: every title stem is banned, as before
    ],
)
def test_title_stems_are_banned_only_when_rare(
    max_df: float, proposal: str, banned: list[str]
) -> None:
    corpus = _translation_corpus()
    leak = LeakFilter(corpus, max_overlap=1.0, max_jaccard=1.0, title_stem_max_df=max_df)
    assert leak.df_fraction(stem("translation")) >= 20 / len(corpus) > 0.2  # common here
    assert leak.df_fraction(stem("abduction")) == 0.0  # in no document: rare
    rejection, stats = leak.check(proposal, "Abduction for translation", "An abstract.")
    assert stats["title_words_found"] == banned
    assert (rejection is None) == (not banned)
    if not banned:
        assert stats["common_title_words_used"] == ["translation"]


def test_make_bench_uses_the_configured_title_stem_threshold(tmp_path: Path) -> None:
    # The fixture's title_stem proposal reuses the title words "documents" and "long".
    strict = _config(tmp_path / "strict")
    freeze(strict.corpus.out_dir, corpus_records(), target_rows(), {})
    make_bench(strict, paraphraser(**{T1: [LEAKY["title_stem"]]}), judge())
    assert [r["reason"] for r in _rows(strict, REJECTIONS_FILE)] == ["title_stem"]

    lenient = _config(tmp_path / "lenient", title_stem_max_df=0.0001)  # nothing that rare
    freeze(lenient.corpus.out_dir, corpus_records(), target_rows(), {})
    make_bench(lenient, paraphraser(**{T1: [LEAKY["title_stem"]]}), judge())
    assert _rows(lenient, REJECTIONS_FILE) == []
    accepted = {r["target_id"]: r for r in _rows(lenient, ACCEPTED_FILE)}
    assert accepted[T1]["query"] == LEAKY["title_stem"]
    assert accepted[T1]["leak_stats"]["common_title_words_used"] == ["documents", "long"]


# The pilot target W7212273718: its abstract spells out the STEP acronym, so
# "STandard" was extracted as an acronym and banned the ordinary word "standard".
STEP_ABSTRACT = (
    "We build an evaluation set directly from product and assembly files compliant with the "
    "STandard for Exchange of Product Model Data (STEP). We fine-tune BERT and LoRA adapters."
)


def _common_words_filter() -> LeakFilter:
    """The fixture corpus plus abstracts that use "standard" and "step" as ordinary words."""
    extra = [
        {"id": f"W{800 + i}", "title": "Procedures",
         "abstract": "We follow a standard procedure, one step at a time."}
        for i in range(3)
    ]
    return LeakFilter([*corpus_records(), *extra], max_overlap=1.0, max_jaccard=1.0)


@pytest.mark.parametrize(
    ("proposal", "found"),
    [  # ordinary words that an extracted acronym spells are fine ...
        ("We will build a test set, step by step, from design files that follow a standard.", []),
        # ... a mixed-case form must match as written ("lora" isn't "LoRA") ...
        ("We will tune small lora adapters on design files.", []),
        # ... and real acronyms still match in any case.
        ("We will adapt bert to design files.", ["BERT"]),
    ],
)
def test_source_acronyms_skip_common_words_and_match_mixed_case_exactly(
    proposal: str, found: list[str]
) -> None:
    leak = _common_words_filter()
    assert leak.is_common_word("standard") and leak.is_common_word("step")
    assert not leak.is_common_word("bert")
    rejection, stats = leak.check(proposal, "Ontology assessment for product data", STEP_ABSTRACT)
    assert stats["source_acronyms_found"] == found
    assert (rejection is None) == (not found)


def test_leak_filter_lets_a_proposal_keep_the_titles_field() -> None:
    leak = LeakFilter(corpus_records(), max_overlap=1.0, max_jaccard=1.0)
    title = "Global trends in physics information-seeking"
    rejection, stats = leak.check(
        "We will track how students look up physics facts on the web over several years.",
        title,
        "We study how students search for physics material.",
    )
    assert rejection is None and stats["title_words_found"] == []


def _bilibili_filter() -> LeakFilter:
    """The fixture corpus plus abstracts that write Bilibili capitalised, and one that
    capitalises a common word inside a title-cased phrase."""
    extra = [
        {"id": "W901", "title": "Danmaku", "abstract":
         "We study danmaku comments posted on Bilibili, a video site. Bilibili users comment."},
        {"id": "W902", "title": "Video", "abstract": "Comments on Bilibili scroll across videos."},
        {"id": "W903", "title": "Search", "abstract":
         "We extend Retrieval-Augmented Generation with better retrieval and generation."},
    ]
    return LeakFilter([*corpus_records(), *extra], max_overlap=1.0, max_jaccard=1.0)


def test_source_name_rejects_the_pilots_bilibili_proposal() -> None:
    title = "Trust the messenger, then the message: the cross-route logic of danmaku engagement"
    abstract = "We analyse danmaku comments from Bilibili and how viewers weigh source cues."
    proposal = (  # A0024 as accepted in the Step 6 pilot
        "We intend to explore how viewers interact with floating annotations on videos from "
        "Bilibili. By analyzing user behavior and text patterns, we will investigate if people "
        "first judge the author's reputation before evaluating the actual information and other "
        "users' reactions."
    )
    rejection, stats = _bilibili_filter().check(proposal, title, abstract)
    assert rejection is not None and rejection.reason == "source_name"
    assert rejection.message == "it used names from the paper (Bilibili)"
    assert stats["source_names_found"] == ["Bilibili"]


@pytest.mark.parametrize(
    "proposal",
    [
        "Bilibili users will be studied. We look at comments.",  # a sentence start isn't a name use
        "We study how Generation of comments works.",  # mostly lower case in the corpus: common
        "We study Arabic comments from the Middle East.",  # exempt languages and regions
    ],
)
def test_source_name_ignores_sentence_starts_common_words_and_exempt_terms(proposal: str) -> None:
    abstract = "Bilibili comments. We extend Retrieval-Augmented Generation to Arabic posts."
    _, stats = _bilibili_filter().check(proposal, "A title", abstract)
    assert stats["source_names_found"] == []


def test_prompts_render_every_placeholder() -> None:
    values = {
        "title": "T", "abstract": "A", "proposal": "P", "feedback": "", "keep": "", "change": "C",
    }
    for prompt in [*PROMPTS.values(), load_prompt("corrupt")]:
        assert "$" not in prompt.render(**values)
        assert len(prompt.sha256) == 64


@pytest.mark.parametrize("name", ["judge_captures", "judge_adds_nothing"])
def test_judge_prompts_compare_four_facets(name: str) -> None:
    text = PROMPTS[name].template.template
    for facet in ("problem", "method", "domain", "claim"):
        assert f"- {facet}: " in text and f'"{facet}": {{"match": "yes"' in text
    assert "including any language, region" in text
    assert 'The verdict is "yes" only if all four facets are "yes".' in text
    # Removed names never count against a proposal, in either direction.
    assert "are removed on purpose: their absence never makes a facet \"no\"" in text
    assert "a described stand-in" in text
    # The "too general" test, with its one example, is in both directions: with it
    # only in "captures", generalised corruptions were caught 7/30 instead of 24/30.
    for phrase in (
        "no longer tells this work apart from close sibling work",
        "drops the specific domain, population, data type or language",
        "Example of a too-general proposal",
    ):
        assert phrase in text


def _strict_everywhere(schema: dict[str, Any]) -> bool:
    if schema.get("type") != "object":
        return True
    props = schema["properties"]
    return (
        schema["additionalProperties"] is False
        and schema["required"] == list(props)
        and all(_strict_everywhere(p) for p in props.values())
    )


def test_judge_answers_with_a_strict_json_schema(tmp_path: Path, corpus_dir: Path) -> None:
    fmt = JUDGE_RESPONSE_FORMAT
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert _strict_everywhere(schema)
    assert list(schema["properties"]) == ["facets", "match", "verdict", "reason"]  # think first
    facets = ["problem", "method", "domain", "claim"]
    assert list(schema["properties"]["match"]["properties"]) == facets
    para, jdg = paraphraser(), judge()

    make_bench(_config(tmp_path), para, jdg)

    assert jdg.response_formats == [JUDGE_RESPONSE_FORMAT] * len(jdg.calls)
    assert para.response_formats == [None] * len(para.calls)
    assert _stats(_config(tmp_path))["prompt_hashes"]["judge_schema"] == JUDGE_SCHEMA_SHA256


def test_claim_corruption_changes_the_expected_outcome_not_the_question() -> None:
    claim = CORRUPTIONS["claim"]
    assert "what the work will show or deliver" in claim
    assert "Do not merely negate a question" in claim


# --- judge validation -----------------------------------------------------------


def _corrupting_paraphraser() -> FakeLLM:
    """Writes 'CORRUPTED-<kind> ...' for each corruption instruction it is given."""

    def reply(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        kind = next(k for k, change in CORRUPTIONS.items() if f"Change: {change}" in prompt)
        return f"CORRUPTED-{kind} proposal for {_target_in(prompt, 'Title')}."

    return FakeLLM(reply)


def _validation_judge() -> FakeLLM:
    """Rejects the domain and method corruptions, can't be read on the claim ones, and
    lets the generalised ones through."""

    def reply(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        if "CORRUPTED-domain" in prompt:
            return judge_reply({"domain": "another language"})
        if "CORRUPTED-method" in prompt:
            return judge_reply({"method": "another method", "claim": "follows from it"})
        if "CORRUPTED-claim" in prompt:
            return "garbled"
        return YES

    return FakeLLM(reply)


def test_validate_judge_reports_acceptance_and_rejection_rates(
    tmp_path: Path, corpus_dir: Path
) -> None:
    cfg = _config(tmp_path)
    make_bench(cfg, paraphraser(), judge())  # three accepted pairs
    para, jdg = _corrupting_paraphraser(), _validation_judge()
    out = tmp_path / "judge_validation"

    summary = validate_judge(cfg, para, jdg, n=2, out_dir=out)

    first_two = sorted(_rows(cfg, ACCEPTED_FILE), key=lambda r: r["qid"])[:2]
    assert summary["n_pairs"] == 2
    assert len(para.calls) == 2 * 4  # one call per corruption
    assert len(jdg.calls) == 2 * 5 * 2  # both directions for the original and each corruption
    original_in_prompt = first_two[0]["query"]
    assert original_in_prompt in para.calls[0][-1]["content"]
    v = summary["variants"]
    assert v["original"]["acceptance_rate"] == 1.0
    rates = {kind: v[kind]["rejection_rate"] for kind in CORRUPTIONS}
    assert rates == {"domain": 1.0, "method": 1.0, "claim": 1.0, "generalised": 0.0}
    assert v["domain"]["rejected_by_direction"] == {"captures": 2, "adds_nothing": 2}
    assert v["domain"]["flagged_facets"] == {"problem": 0, "method": 0, "domain": 2, "claim": 0}
    assert v["method"]["flagged_facets"]["claim"] == 2
    assert v["claim"]["unreadable"] == 2  # unreadable counts as rejected, as in make-bench
    assert summary["money_spent"] > 0 and "corrupt" in summary["prompt_hashes"]

    pairs = [json.loads(x) for x in (out / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [p["qid"] for p in pairs] == [r["qid"] for r in first_two]
    corruptions = (out / "corruptions.jsonl").read_text(encoding="utf-8").splitlines()
    verdicts = [
        json.loads(x) for x in (out / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(corruptions) == 8 and len(verdicts) == 10
    assert all(set(r["judgments"]) == {"captures", "adds_nothing"} for r in verdicts)
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary


def test_validate_judge_needs_accepted_pairs(tmp_path: Path, corpus_dir: Path) -> None:
    with pytest.raises(BenchError, match="no accepted queries"):
        validate_judge(_config(tmp_path), paraphraser(), judge(), n=5, out_dir=tmp_path / "v")


def test_cli_validate_judge(
    tmp_path: Path,
    corpus_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_dict(tmp_path)), encoding="utf-8")
    make_bench(_config(tmp_path), paraphraser(), judge())
    fakes = {"para-model": _corrupting_paraphraser(), "judge-model": _validation_judge()}
    monkeypatch.setattr(cli, "LLMClient", lambda model, cache, ledger: fakes[model.name])
    out = tmp_path / "jv"

    argv = ["validate-judge", "--config", str(config_path), "--n", "3", "--out-dir", str(out)]
    assert cli.main(argv) == 0

    printed = capsys.readouterr().out
    assert "rk validate-judge: 3 pairs" in printed
    assert "original     accepted 3/3 (100%)" in printed
    assert "generalised  rejected 0/3 (0%)" in printed
    assert (out / "summary.json").exists()

    # --set swaps the judge for one run, e.g. to compare a stronger model.
    fakes["bigger-judge"] = _validation_judge()
    other = tmp_path / "jv-bigger"
    overrides = ["--set", "judge.name=bigger-judge", "--set", "judge.price_in_per_m=2.0"]
    assert cli.main([*argv[:-1], str(other), *overrides]) == 0
    summary = json.loads((other / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"]["judge"] == "bigger-judge"
    assert fakes["bigger-judge"].calls and summary["n_pairs"] == 3


# --- human check export ----------------------------------------------------------


def test_export_human_check_writes_a_seeded_sample_and_never_overwrites(
    tmp_path: Path, corpus_dir: Path
) -> None:
    cfg = _config(tmp_path)
    make_bench(cfg, paraphraser(), judge())
    out = tmp_path / "check" / "human_check.csv"

    assert export_human_check(cfg, n=2, out_path=out) == (2, 0)

    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM, so spreadsheets read UTF-8
    rows = _csv_rows(out)
    assert tuple(rows[0]) == HUMAN_CHECK_COLUMNS
    body = rows[1:]
    assert len(body) == 2 and [r[1] for r in body] == sorted(r[1] for r in body)
    accepted = {r["qid"]: r for r in _rows(cfg, ACCEPTED_FILE)}
    for part, qid, title, abstract, proposal, reason, verdict, note in body:
        target = accepted[qid]["target_id"]
        assert part == "A" and (title, abstract) == (TITLES[target], ABSTRACTS[target])
        assert proposal == accepted[qid]["query"] and verdict == note == ""
        assert reason.startswith("captures: ") and " | adds_nothing: " in reason

    with pytest.raises(BenchError, match="refusing to overwrite"):
        export_human_check(cfg, n=2, out_path=out)
    again = tmp_path / "again.csv"
    export_human_check(cfg, n=2, out_path=again)
    assert again.read_bytes() == raw  # the same seeded sample


def _csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def _write_validation(path: Path, judge_name: str) -> None:
    """A validate-judge output with two rejected originals, one accepted, one corruption."""
    failed = {
        "verdict": "no",
        "failed": ["claim"],
        "match": {"claim": {"match": "no", "reason": "states an aim, not the findings"}},
        "reason": "the claim differs",
    }
    passed = {"verdict": "yes", "failed": [], "match": {}, "reason": "all facets match"}
    unreadable = {"verdict": None, "raw": "{"}
    rows = [
        ("A0002-x", T2, "original", False, {"captures": failed, "adds_nothing": unreadable}),
        ("A0001-x", T1, "original", False, {"captures": passed, "adds_nothing": failed}),
        ("A0003-x", T3, "original", True, {"captures": passed, "adds_nothing": passed}),
        ("A0001-x", T1, "method", False, {"captures": failed, "adds_nothing": failed}),
    ]
    path.mkdir(parents=True)
    summary = {"models": {"judge": judge_name}}
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    verdicts = [
        {
            "qid": qid,
            "target_id": target,
            "variant": variant,
            "proposal": f"{variant} proposal for {qid}",
            "accepted": accepted,
            "judgments": judgments,
        }
        for qid, target, variant, accepted, judgments in rows
    ]
    text = "".join(json.dumps(v) + "\n" for v in verdicts)
    (path / "verdicts.jsonl").write_text(text, encoding="utf-8")


def test_export_human_check_part_b_holds_the_originals_the_judge_rejected(
    tmp_path: Path, corpus_dir: Path
) -> None:
    cfg = _config(tmp_path)
    make_bench(cfg, paraphraser(), judge())
    validation = tmp_path / "jv"
    _write_validation(validation, cfg.judge.name)
    out = tmp_path / "hc.csv"

    assert export_human_check(cfg, n=1, out_path=out, validation_dir=validation) == (1, 2)

    part_b = [r for r in _csv_rows(out)[1:] if r[0] == "B"]
    assert [r[:6] for r in part_b] == [
        [
            "B",
            "A0001-x",
            TITLES[T1],
            ABSTRACTS[T1],
            "original proposal for A0001-x",
            "captures: all facets match | adds_nothing/claim: states an aim, not the findings",
        ],
        [
            "B",
            "A0002-x",
            TITLES[T2],
            ABSTRACTS[T2],
            "original proposal for A0002-x",
            "captures/claim: states an aim, not the findings | adds_nothing: unreadable answer",
        ],
    ]
    assert all(r[6] == r[7] == "" for r in part_b)

    # The reasons must come from the judge the config names.
    other = tmp_path / "jv-other"
    _write_validation(other, "another-judge")
    with pytest.raises(BenchError, match="judged by another-judge"):
        export_human_check(cfg, n=1, out_path=tmp_path / "x.csv", validation_dir=other)


def test_cli_export_human_check(
    tmp_path: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_dict(tmp_path)), encoding="utf-8")
    cfg = _config(tmp_path)
    make_bench(cfg, paraphraser(), judge())
    out = tmp_path / "hc.csv"

    argv = ["export-human-check", "--config", str(config_path), "--n", "20", "--out", str(out)]
    assert cli.main(argv) == 0
    assert "wrote 3 accepted pairs" in capsys.readouterr().out  # all there are
    assert cli.main(argv) == 1  # never overwritten
    assert "refusing to overwrite" in capsys.readouterr().err

    validation = tmp_path / "jv"
    _write_validation(validation, cfg.judge.name)
    both = tmp_path / "both.csv"
    assert cli.main([*argv[:-1], str(both), "--validation-dir", str(validation)]) == 0
    assert "and 2 rejected originals from" in capsys.readouterr().out
    assert [r[0] for r in _csv_rows(both)[1:]] == ["A", "A", "A", "B", "B"]


# --- CLI ---------------------------------------------------------------------


def test_cli_make_bench_with_limit_and_show(
    tmp_path: Path,
    corpus_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_dict(tmp_path)), encoding="utf-8")
    fakes = {"para-model": paraphraser(), "judge-model": judge()}

    def make_client(model: Any, cache: Any, ledger: Any) -> FakeLLM:
        return fakes[model.name]

    monkeypatch.setattr(cli, "LLMClient", make_client)

    assert cli.main(["make-bench", "--config", str(config_path), "--limit", "1", "--show"]) == 0

    out = capsys.readouterr().out
    first = ORDER[0]
    assert "1 of 3 queries accepted" in out
    assert f"{make_qid(0, first)}  {TITLES[first]}" in out
    opening = ". ".join(ABSTRACTS[first].split(". ")[:2]) + "."
    assert f"abstract: {opening}" in out
    assert f"proposal: {GOOD[first]}" in out
