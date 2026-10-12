from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from bench_data import ABSTRACTS, GOOD, LEAKY, T1, T2, T3, TITLES, corpus_records, target_rows
from conftest import FakeLLM, corpus_config_dict

from rk import cli
from rk.bench import (
    ACCEPTED_FILE,
    REASONS,
    REJECTIONS_FILE,
    STATS_FILE,
    BenchError,
    LeakFilter,
    _visit_order,
    check_length,
    load_prompts,
    make_bench,
    make_qid,
    parse_verdict,
)
from rk.config import Config, config_from_dict
from rk.corpus import CORPUS_FILE, CorpusError, freeze

YES = '{"verdict": "yes", "reason": "same problem, method and claim"}'
NO = '{"verdict": "no", "reason": "the method differs"}'
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
    assert "the method differs" in rejection["detail"]
    assert rejection["judge_verdicts"][direction]["verdict"] == "no"
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
    assert "previous proposal was rejected because it used words from the title" in retry
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
    ("text", "verdict"),
    [
        (YES, {"verdict": "yes", "reason": "same problem, method and claim"}),
        (
            '```json\n{"verdict": "No", "reason": "adds a dataset"}\n```',
            {"verdict": "no", "reason": "adds a dataset"},
        ),
        ('{"verdict": "yes"}', {"verdict": "yes", "reason": ""}),
        ('{"verdict": "maybe", "reason": "unsure"}', None),
        ('{"answer": "yes"}', None),
        ("yes", None),
    ],
)
def test_parse_verdict(text: str, verdict: dict[str, str] | None) -> None:
    assert parse_verdict(text) == verdict


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


def test_prompts_render_every_placeholder() -> None:
    values = {"title": "T", "abstract": "A", "proposal": "P", "feedback": ""}
    for prompt in PROMPTS.values():
        assert "$" not in prompt.render(**values)
        assert len(prompt.sha256) == 64


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
