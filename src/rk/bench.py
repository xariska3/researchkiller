"""Test A benchmark generation (spec: Benchmark generation).

For each target the paraphraser writes a terminology-shifted research proposal
from the title and abstract. A length check and a deterministic lexical-leak
filter run first; then the judge (another model family) checks the two
directions of equivalence in two separate calls, facet by facet (problem,
method, domain or population, main claim). The first failing check is the
rejection reason, and the next attempt's prompt quotes it. Every attempt is
appended to test_a.jsonl (accepted) or rejections.jsonl (rejected) as it
happens, which is also what makes a rerun resume where the last one stopped.
validate_judge measures the judge on accepted pairs and deliberate corruptions.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rk.config import Config
from rk.corpus import CORPUS_FILE, load_frozen_corpus
from rk.llm import ChatModel, Completion, Prompt, Usage, load_prompt, parse_json
from rk.targets import Record, id_number
from rk.text import (
    STOPWORDS,
    capitalised_words,
    content_stems,
    content_tokens,
    extract_acronyms,
    find_domain_terms,
    is_generic,
    lowercase_words,
    mask_domain_terms,
    stem,
)

PROMPT_NAMES = ("paraphrase", "judge_captures", "judge_adds_nothing")
MIN_SENTENCES, MAX_SENTENCES = 2, 4
MIN_WORDS, MAX_WORDS = 40, 120
IDF_TOP_TERMS = 20
JUDGE_DIRECTIONS = ("captures", "adds_nothing")
FACETS = ("problem", "method", "domain", "claim")  # the judge compares these one by one
FACET_LABELS = {
    "problem": "problem",
    "method": "method",
    "domain": "domain or population",
    "claim": "main claim",
}


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    """A JSON-schema object in OpenAI strict mode: every property required, no others."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_YES_NO = {"type": "string", "enum": ["yes", "no"]}
# The judge's answer, enforced by the API (response_format json_schema, strict), so its
# format can't slip; the order makes the judge describe the facets before deciding.
JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "facet_judgment",
        "strict": True,
        "schema": _strict_object(
            {
                "facets": _strict_object(
                    {f: _strict_object({"paper": _STRING, "proposal": _STRING}) for f in FACETS}
                ),
                "match": _strict_object(
                    {f: _strict_object({"match": _YES_NO, "reason": _STRING}) for f in FACETS}
                ),
                "verdict": _YES_NO,
                "reason": _STRING,
            }
        ),
    },
}
JUDGE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(JUDGE_RESPONSE_FORMAT, sort_keys=True).encode("utf-8")
).hexdigest()

ACCEPTED_FILE = "test_a.jsonl"
REJECTIONS_FILE = "rejections.jsonl"
STATS_FILE = "stats.json"

NOT_RESEARCH = "not_research_contribution"

# Every rejection reason, in the order the checks run; the first failure wins.
REASONS = (
    NOT_RESEARCH,  # checked once per target, before any paraphrase
    "length",
    "acronym_in_paraphrase",
    "source_acronym",
    "source_name",
    "title_stem",
    "idf_overlap",
    "content_jaccard",
    "judge_parse_error",
    "judge_captures",
    "judge_adds_nothing",
)


class BenchError(RuntimeError):
    """The benchmark can't be generated as asked (missing corpus, incompatible outputs)."""


def load_prompts() -> dict[str, Prompt]:
    """The benchmark's prompt templates, each with the sha256 of its text."""
    return {name: load_prompt(name) for name in PROMPT_NAMES}


def prompt_hashes(prompts: dict[str, Prompt]) -> dict[str, str]:
    """sha256 of every prompt, plus the judge's answer schema (it shapes the answers too)."""
    return {**{name: p.sha256 for name, p in prompts.items()}, "judge_schema": JUDGE_SCHEMA_SHA256}


@dataclass(frozen=True)
class Rejection:
    reason: str  # one of REASONS
    message: str  # human-readable; quoted in the retry prompt


# A paper that calls itself a survey, review or tutorial isn't a research contribution:
# its proposal is generic and matches many papers equally well, so the target isn't
# the one right answer. Self-description only, so papers *about* reviews (Cochrane
# reviews, peer review, literature-review tooling) are kept.
_SURVEY_TITLE = re.compile(r"\bsurveys?\b", re.IGNORECASE)
_SELF_DESCRIPTION = (
    re.compile(r"\b(?:this|our|the present) (?:survey|review|tutorial)\b", re.IGNORECASE),
    re.compile(r"\bwe (?:survey|review)\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are) reviewed\b", re.IGNORECASE),
    re.compile(r"\bPRISMA\b"),  # the systematic-review reporting guideline
)


def not_research_contribution(title: str, abstract: str) -> str | None:
    """The phrase by which a paper describes itself as a survey, review or tutorial, else None."""
    if match := _SURVEY_TITLE.search(title):
        return match.group(0)
    text = f"{title}. {abstract}"
    for pattern in _SELF_DESCRIPTION:
        if match := pattern.search(text):
            return match.group(0)
    return None


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*")


def sentences(text: str) -> list[str]:
    """Split prose into sentences at ., ! or ? followed by whitespace."""
    return [s for s in _SENTENCE_BREAK.split(text.strip()) if s]


def clean_proposal(text: str) -> str:
    """Collapse whitespace and drop quotation marks wrapped around the whole reply."""
    text = " ".join(text.split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def check_length(proposal: str) -> tuple[Rejection | None, dict[str, Any]]:
    n_words, n_sentences = len(proposal.split()), len(sentences(proposal))
    stats = {"n_words": n_words, "n_sentences": n_sentences}
    if MIN_SENTENCES <= n_sentences <= MAX_SENTENCES and MIN_WORDS <= n_words <= MAX_WORDS:
        return None, stats
    message = (
        f"it had {n_sentences} sentences and {n_words} words, but must have "
        f"{MIN_SENTENCES} to {MAX_SENTENCES} sentences and {MIN_WORDS} to {MAX_WORDS} words"
    )
    return Rejection("length", message), stats


def _match_key(text: str) -> str:
    """Case- and punctuation-insensitive key: 'GPT-4' -> 'gpt4'."""
    return re.sub(r"[\W_]+", "", text.lower())


def _word_keys(text: str) -> set[str]:
    """Match keys of every word in `text`, hyphenated compounds whole and by part."""
    keys = set()
    for match in _WORD_RE.finditer(text):
        word = match.group()
        keys.add(_match_key(word))
        keys.update(_match_key(part) for part in word.split("-"))
    return keys


class LeakFilter:
    """The deterministic lexical-leak checks, with IDF computed over the frozen corpus.

    Checks run in this order and the first failure is the reason:
    1. any acronym or named artefact in the proposal at all;
    2. any acronym or named artefact from the title or abstract (case-insensitive;
       ones that spell a stopword, like "IT" or "US", are skipped);
    3. any capitalised name from the abstract ("Bilibili") used mid-sentence in
       the proposal; a word counts as a name if the corpus abstracts capitalise it
       mid-sentence more often than they write it in lower case, so common words
       capitalised by a title-cased phrase ("Retrieval-Augmented Generation") don't;
    4. any distinctive title stem (title content stems minus generic words);
    5. more than a fraction max_overlap of the abstract's top-20 IDF stems;
    6. content-stem Jaccard with title + abstract above max_jaccard, generic
       science words excluded on both sides (text.is_generic).
    Named languages, nationalities, countries, regions and fields (rk/data/)
    are blanked out of proposal, title and abstract first: a proposal may keep
    them, and they never count as leaked terms.
    """

    def __init__(self, corpus: Iterable[Record], *, max_overlap: float, max_jaccard: float):
        df: Counter[str] = Counter()
        self._capitalised: Counter[str] = Counter()  # docs capitalising a word mid-sentence
        self._lowercase: Counter[str] = Counter()  # docs writing it in lower case
        n_docs = 0
        for record in corpus:
            n_docs += 1
            df.update(set(content_stems(f"{record['title']} {record['abstract']}")))
            capitalised = capitalised_words(record["abstract"], mid_sentence=True)
            self._capitalised.update({w.lower() for w in capitalised})
            self._lowercase.update(lowercase_words(record["abstract"]))
        self._idf = {s: math.log(n_docs / n) for s, n in df.items()}
        self._unseen_idf = math.log(max(n_docs, 1))  # as rare as a term can be
        self.max_overlap = max_overlap
        self.max_jaccard = max_jaccard

    def top_terms(self, abstract: str) -> list[str]:
        """The abstract's IDF_TOP_TERMS highest-IDF content stems.

        Ties (common among rare terms) go to stems used more often in the
        abstract, then alphabetical order.
        """
        counts = Counter(content_stems(abstract))

        def rank(s: str) -> tuple[float, int, str]:
            return (-self._idf.get(s, self._unseen_idf), -counts[s], s)

        return sorted(counts, key=rank)[:IDF_TOP_TERMS]

    def is_name(self, word: str) -> bool:
        """True if the corpus abstracts capitalise `word` mid-sentence more than they don't."""
        key = word.lower()
        if key in STOPWORDS or is_generic(key):
            return False
        return self._capitalised[key] > self._lowercase[key]

    def check(
        self, proposal: str, title: str, abstract: str
    ) -> tuple[Rejection | None, dict[str, Any]]:
        proposal, title, abstract = map(mask_domain_terms, (proposal, title, abstract))
        own_acronyms = extract_acronyms(proposal)

        source = set(extract_acronyms(title)) | set(extract_acronyms(abstract))
        source_by_key = {_match_key(a): a for a in sorted(source)}
        proposal_keys = _word_keys(proposal)
        source_found = sorted(
            a for key, a in source_by_key.items() if key in proposal_keys and key not in STOPWORDS
        )
        source_names = sorted(
            w
            for w in capitalised_words(proposal, mid_sentence=True)
            & capitalised_words(abstract, mid_sentence=False)
            if self.is_name(w)
        )

        proposal_tokens = content_tokens(proposal)
        title_stems = set(content_stems(title, drop_generic=True))
        title_words = sorted({t for t in proposal_tokens if stem(t) in title_stems})

        top = self.top_terms(abstract)
        proposal_stems = {stem(t) for t in proposal_tokens}
        overlap = [s for s in top if s in proposal_stems]
        allowed = math.floor(self.max_overlap * len(top) + 1e-9)
        overlap_words = sorted({t for t in proposal_tokens if stem(t) in overlap})

        specific = set(content_stems(proposal, drop_generic=True))
        doc_stems = set(content_stems(f"{title} {abstract}", drop_generic=True))
        union = specific | doc_stems
        jaccard = len(specific & doc_stems) / len(union) if union else 0.0

        stats = {
            "acronyms_in_paraphrase": own_acronyms,
            "source_acronyms_found": source_found,
            "source_names_found": source_names,
            "title_words_found": title_words,
            "idf_top_terms": top,
            "idf_overlap": overlap,
            "idf_overlap_fraction": len(overlap) / len(top) if top else 0.0,
            "content_jaccard": jaccard,
        }
        checks = [
            (
                "acronym_in_paraphrase",
                bool(own_acronyms),
                f"it used acronyms or names ({', '.join(own_acronyms)}), and must use none",
            ),
            (
                "source_acronym",
                bool(source_found),
                f"it used names or acronyms from the paper ({', '.join(source_found)})",
            ),
            (
                "source_name",
                bool(source_names),
                f"it used names from the paper ({', '.join(source_names)})",
            ),
            (
                "title_stem",
                bool(title_words),
                f"it used words from the title ({', '.join(title_words)})",
            ),
            (
                "idf_overlap",
                len(overlap) > allowed,
                (
                    f"it reused {len(overlap)} of the paper's {len(top)} most distinctive "
                    f"terms ({', '.join(overlap_words)}), but at most {allowed} are allowed"
                ),
            ),
            (
                "content_jaccard",
                jaccard > self.max_jaccard,
                (
                    f"its wording overlaps too much with the paper (content-word Jaccard "
                    f"{jaccard:.2f}, limit {self.max_jaccard:.2f})"
                ),
            ),
        ]
        for reason, failed, message in checks:
            if failed:
                return Rejection(reason, message), stats
        return None, stats


def _yes_no(value: Any) -> str | None:
    if isinstance(value, str) and value.strip().lower() in ("yes", "no"):
        return value.strip().lower()
    return None


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def parse_judgment(text: str) -> dict[str, Any] | None:
    """A facet-by-facet judge reply, or None if it can't be read.

    Returns {"verdict", "failed", "match", "facets", "judge_verdict", "reason"}.
    `verdict` is "yes" only if the judge answered yes AND every facet in FACETS
    matched: a facet marked "no" rejects, whatever the judge concluded. A reply
    without a yes/no for every facet and for the verdict is unreadable.
    """
    try:
        data = parse_json(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("match"), dict):
        return None
    match: dict[str, dict[str, str]] = {}
    for facet in FACETS:
        entry = data["match"].get(facet)
        value = _yes_no(entry.get("match")) if isinstance(entry, dict) else None
        if value is None:
            return None
        match[facet] = {"match": value, "reason": _text(entry.get("reason"))}
    judge_verdict = _yes_no(data.get("verdict"))
    if judge_verdict is None:
        return None
    failed = [facet for facet in FACETS if match[facet]["match"] == "no"]
    facets = data.get("facets")
    return {
        "verdict": "yes" if judge_verdict == "yes" and not failed else "no",
        "failed": failed,
        "match": match,
        "facets": facets if isinstance(facets, dict) else None,
        "judge_verdict": judge_verdict,
        "reason": _text(data.get("reason")),
    }


def judge_proposal(
    judge: ChatModel, prompt: Prompt, title: str, abstract: str, proposal: str
) -> tuple[dict[str, Any] | None, Completion]:
    """One directional judge call: the parsed judgment (None if unreadable) and the reply."""
    text = prompt.render(title=title, abstract=abstract, proposal=proposal)
    reply = judge.complete(
        [{"role": "user", "content": text}], response_format=JUDGE_RESPONSE_FORMAT
    )
    return parse_judgment(reply.text), reply


@dataclass
class Attempt:
    query: str
    rejection: Rejection | None
    leak_stats: dict[str, Any]
    judge_verdicts: dict[str, Any]
    usage: dict[str, Usage]
    providers: dict[str, list[str | None]]  # per role, who served each call, in order


_JUDGE_NO = {
    "captures": "a reviewer found that it does not keep the paper's",
    "adds_nothing": "a reviewer found that it changes or adds to the paper's",
}


def judge_rejection(direction: str, judgment: dict[str, Any]) -> Rejection:
    """The rejection for a "no", naming each failed facet with the judge's reason."""
    if judgment["failed"]:
        names = ", ".join(FACET_LABELS[f] for f in judgment["failed"])
        reasons = "; ".join(
            f"{FACET_LABELS[f]}: {judgment['match'][f]['reason']}" for f in judgment["failed"]
        )
        message = f"{_JUDGE_NO[direction]} {names} ({reasons})"
    else:  # every facet matched, but the judge still said no
        message = f"{_JUDGE_NO[direction]} research idea ({judgment['reason']})"
    return Rejection(f"judge_{direction}", message)


def run_attempt(
    record: Record,
    attempt: int,
    previous: str | None,
    *,
    prompts: dict[str, Prompt],
    paraphraser: ChatModel,
    judge: ChatModel,
    leak: LeakFilter,
    base_seed: int,
) -> Attempt:
    """One paraphrase attempt through the length check, leak filter and both judge calls."""
    title, abstract = record["title"], record["abstract"]
    feedback = ""
    if previous:
        feedback = (
            f"\nYour previous proposal was rejected because {previous}. "
            "Write a new proposal that fixes this and follows all the rules.\n"
        )
    prompt = prompts["paraphrase"].render(
        title=title, abstract=abstract, keep=keep_names_line(title), feedback=feedback
    )
    completion = paraphraser.complete(
        [{"role": "user", "content": prompt}], seed=base_seed + attempt
    )
    usage = {"paraphraser": completion.usage, "judge": Usage()}
    providers: dict[str, list[str | None]] = {"paraphraser": [completion.provider], "judge": []}
    query = clean_proposal(completion.text)

    length_rejection, length_stats = check_length(query)
    leak_rejection, leak_stats = leak.check(query, title, abstract)
    rejection = length_rejection or leak_rejection
    verdicts: dict[str, Any] = {}
    if rejection is None:
        for direction in JUDGE_DIRECTIONS:  # separate calls: one answer can't anchor the other
            judgment, reply = judge_proposal(
                judge, prompts[f"judge_{direction}"], title, abstract, query
            )
            usage["judge"] = usage["judge"] + reply.usage
            providers["judge"].append(reply.provider)
            if judgment is None:
                verdicts[direction] = {"verdict": None, "raw": reply.text[:500]}
                rejection = Rejection(
                    "judge_parse_error", "the reviewer's answer about it could not be read"
                )
                break
            verdicts[direction] = judgment
            if judgment["verdict"] == "no":
                rejection = judge_rejection(direction, judgment)
                break
    return Attempt(query, rejection, {**length_stats, **leak_stats}, verdicts, usage, providers)


def keep_names_line(title: str) -> str:
    """The paraphrase prompt's line naming the title's languages, places and fields, or ""."""
    names = find_domain_terms(title)
    if not names:
        return ""
    return f"\nThe proposal must keep these words from the title: {', '.join(names)}.\n"


def make_bench(
    cfg: Config, paraphraser: ChatModel, judge: ChatModel, *, limit: int | None = None
) -> dict[str, Any]:
    """Generate (or resume) Test A in bench.out_dir; return the stats.

    Targets are visited in a seeded random order. A target that describes itself
    as a survey, review or tutorial is skipped before any LLM call and logged
    once in rejections.jsonl. Targets already accepted, exhausted or skipped are
    passed over, a target interrupted mid-way continues with its next attempt,
    and the run stops at n_targets accepted. `limit` caps how many targets this
    run paraphrases (skipped ones don't count).
    """
    corpus, targets, manifest = load_frozen_corpus(cfg.corpus.out_dir)
    prompts = load_prompts()
    meta = {
        "prompt_hashes": prompt_hashes(prompts),
        "models": {"paraphraser": cfg.paraphraser.name, "judge": cfg.judge.name},
        "corpus_sha256": manifest["files"][CORPUS_FILE]["sha256"],
    }
    out_dir = cfg.bench.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_rows = _read_jsonl(out_dir / ACCEPTED_FILE)
    rejected_rows = _read_jsonl(out_dir / REJECTIONS_FILE)
    _check_resumable(out_dir, meta, [*accepted_rows, *rejected_rows])

    accepted = {r["target_id"] for r in accepted_rows}
    skipped = {r["target_id"] for r in rejected_rows if r["reason"] == NOT_RESEARCH}
    attempts: Counter[str] = Counter(r["target_id"] for r in rejected_rows)
    last_reason = {r["target_id"]: r["detail"] for r in sorted(rejected_rows, key=_by_attempt)}
    leak = LeakFilter(
        corpus,
        max_overlap=cfg.bench.max_distinctive_overlap,
        max_jaccard=cfg.bench.max_content_jaccard,
    )
    by_id = {r["id"]: r for r in corpus}
    max_attempts = cfg.bench.max_attempts_per_target
    width = max(4, len(str(len(targets))))
    worked_on = 0
    try:
        for position, target in enumerate(_visit_order(targets, cfg.bench.seed)):
            target_id = target["id"]
            qid = make_qid(position, target_id, width)
            if len(accepted) >= cfg.bench.n_targets:
                break
            if target_id in accepted or target_id in skipped or attempts[target_id] >= max_attempts:
                continue
            if limit is not None and worked_on >= limit:
                break
            record = by_id[target_id]
            phrase = not_research_contribution(record["title"], record["abstract"])
            if phrase is not None:
                row = _skip_row(target, phrase, meta["prompt_hashes"])
                _append(out_dir / REJECTIONS_FILE, row)
                rejected_rows.append(row)
                skipped.add(target_id)
                continue
            worked_on += 1
            for attempt in range(attempts[target_id], max_attempts):
                result = run_attempt(
                    record,
                    attempt,
                    last_reason.get(target_id),
                    prompts=prompts,
                    paraphraser=paraphraser,
                    judge=judge,
                    leak=leak,
                    base_seed=cfg.paraphraser.seed,
                )
                row = _row(qid, target, attempt, result, meta["prompt_hashes"])
                if result.rejection is None:
                    _append(out_dir / ACCEPTED_FILE, row)
                    accepted_rows.append(row)
                    accepted.add(target_id)
                    break
                _append(out_dir / REJECTIONS_FILE, row)
                rejected_rows.append(row)
                attempts[target_id] += 1
                last_reason[target_id] = result.rejection.message
    finally:  # stats always describe everything on disk, even after a crash
        stats = bench_stats(accepted_rows, rejected_rows, cfg, meta, n_available=len(targets))
        (out_dir / STATS_FILE).write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return stats


def _visit_order(targets: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    order = sorted(targets, key=lambda t: id_number(t["id"]))
    random.Random(seed).shuffle(order)
    return order


def make_qid(position: int, target_id: str, width: int = 4) -> str:
    """'A0007-W123': the target's position in the seeded visiting order, then its id.

    Sorting qids therefore gives the seeded order, so the first N queries (as in
    `run --limit N`) are a seeded random sample, not the lowest ids.
    """
    return f"A{position:0{width}d}-{target_id}"


def _by_attempt(row: dict[str, Any]) -> int:
    return int(row["attempt"])


def _row(
    qid: str,
    target: dict[str, Any],
    attempt: int,
    result: Attempt,
    prompt_hashes: dict[str, str],
) -> dict[str, Any]:
    row: dict[str, Any] = {"target_id": target["id"], "attempt": attempt}
    if result.rejection is None:
        row = {"qid": qid, **row, "target_aliases": target["aliases"]}
    else:
        row.update(reason=result.rejection.reason, detail=result.rejection.message)
    row.update(
        query=result.query,
        leak_stats=result.leak_stats,
        judge_verdicts=result.judge_verdicts,
        prompt_hashes=prompt_hashes,
        usage={role: dataclasses.asdict(u) for role, u in result.usage.items()},
        providers=result.providers,
    )
    return row


def _skip_row(target: dict[str, Any], phrase: str, prompt_hashes: dict[str, str]) -> dict[str, Any]:
    """A rejections.jsonl row for a target skipped before any LLM call."""
    return {
        "target_id": target["id"],
        "attempt": 0,
        "reason": NOT_RESEARCH,
        "detail": f"it describes itself as a survey, review or tutorial ({phrase!r})",
        "query": None,
        "leak_stats": {},
        "judge_verdicts": {},
        "prompt_hashes": prompt_hashes,
        "usage": {role: dataclasses.asdict(Usage()) for role in ("paraphraser", "judge")},
        "providers": {"paraphraser": [], "judge": []},
    }


def bench_stats(
    accepted_rows: Sequence[dict[str, Any]],
    rejected_rows: Sequence[dict[str, Any]],
    cfg: Config,
    meta: dict[str, Any],
    *,
    n_available: int,
) -> dict[str, Any]:
    """Counts, attempts per target and LLM usage over every row written so far.

    Skipped targets (not research contributions) are counted under their reason
    and in `skipped`, but they had no attempt, so they aren't in the attempt counts.
    """
    skip_rows = [r for r in rejected_rows if r["reason"] == NOT_RESEARCH]
    rejected_rows = [r for r in rejected_rows if r["reason"] != NOT_RESEARCH]
    rows = [*accepted_rows, *rejected_rows]
    per_target = Counter(r["target_id"] for r in rows)
    accepted_ids = {r["target_id"] for r in accepted_rows}
    exhausted = [
        t
        for t, n in per_target.items()
        if t not in accepted_ids and n >= cfg.bench.max_attempts_per_target
    ]
    reasons = Counter(r["reason"] for r in [*rejected_rows, *skip_rows])
    failed_facets = {d: Counter[str]() for d in JUDGE_DIRECTIONS}
    for r in rejected_rows:
        if r["reason"] in (f"judge_{d}" for d in JUDGE_DIRECTIONS):
            direction = r["reason"].removeprefix("judge_")
            failed_facets[direction].update(r["judge_verdicts"][direction].get("failed", []))
    usage = {
        role: sum((Usage(**r["usage"][role]) for r in rows), Usage())
        for role in ("paraphraser", "judge")
    }
    total = usage["paraphraser"] + usage["judge"]
    providers = {
        role: dict(Counter(str(p) for r in rows for p in r.get("providers", {}).get(role, [])))
        for role in ("paraphraser", "judge")
    }
    return {
        "n_targets_goal": cfg.bench.n_targets,
        "targets_available": n_available,
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),  # rejected attempts
        "rejected_by_reason": {reason: reasons[reason] for reason in REASONS},
        "attempts": len(rows),
        "exhausted": len(exhausted),
        "judge_failed_facets": {  # per direction: judge rejections naming each facet
            d: {f: failed_facets[d][f] for f in FACETS} for d in JUDGE_DIRECTIONS
        },
        "skipped": len(skip_rows),  # targets that aren't research contributions
        "attempts_per_target": dict(sorted(per_target.items(), key=lambda kv: id_number(kv[0]))),
        "llm_usage": {role: dataclasses.asdict(u) for role, u in usage.items()},
        "llm_providers": providers,  # calls per reported provider; "None" if not reported
        "method_cost": total.method_cost,
        "money_spent": total.money_spent,
        **meta,
    }


def _check_resumable(out_dir: Path, meta: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    """Refuse to mix outputs made with other prompts, models or another corpus."""
    stats_path = out_dir / STATS_FILE
    if stats_path.exists():
        previous = json.loads(stats_path.read_text(encoding="utf-8"))
        for key, value in meta.items():
            if previous.get(key) != value:
                raise BenchError(
                    f"{out_dir} holds a benchmark made with different {key.replace('_', ' ')}; "
                    "use another bench.out_dir to start a new one"
                )
    if any(r["prompt_hashes"] != meta["prompt_hashes"] for r in rows):
        raise BenchError(f"{out_dir} holds rows made with different prompts")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise BenchError(f"{path}:{n} is not valid JSON ({exc})") from exc
    return rows


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def format_accepted(cfg: Config) -> str:
    """Title, first two abstract sentences and proposal of each accepted target, for --show."""
    corpus, _, _ = load_frozen_corpus(cfg.corpus.out_dir)
    by_id = {r["id"]: r for r in corpus}
    blocks = []
    for row in _read_jsonl(cfg.bench.out_dir / ACCEPTED_FILE):
        record = by_id[row["target_id"]]
        opening = " ".join(sentences(record["abstract"])[:2])
        blocks.append(
            f"{row['qid']}  {record['title']}\n"
            f"  abstract: {opening}\n"
            f"  proposal: {row['query']}\n"
        )
    return "\n".join(blocks) if blocks else "(no accepted queries yet)\n"


# --- judge validation ----------------------------------------------------------

JUDGE_VALIDATION_DIR = Path("data/judge_validation")

# One deliberate change each; the paraphraser applies them (prompts/corrupt.txt).
CORRUPTIONS = {
    "domain": (
        "Move the work to a different domain or population, for example another language, "
        "region, application field, data source or user group. Keep the problem, the method "
        "and the main claim."
    ),
    "method": (
        "Replace the method with a different but plausible method for the same problem. Keep "
        "the problem, the domain or population and the main claim."
    ),
    "claim": (
        "Change the expected outcome or contribution, that is, what the work will show or "
        "deliver: a clearly different finding it expects, or a different result, resource "
        "or artefact it will produce. Do not merely negate a question (turning \"whether X\" "
        "into \"whether not X\" asks the same question). Keep the problem, the method and the "
        "domain or population."
    ),
    "generalised": (
        "Make the whole proposal so general that it no longer identifies this work: describe "
        "the problem, method, domain and claim only in broad terms that would fit many papers "
        "in the same field."
    ),
}
VARIANTS = ("original", *CORRUPTIONS)


def validate_judge(
    cfg: Config,
    paraphraser: ChatModel,
    judge: ChatModel,
    *,
    n: int,
    out_dir: Path = JUDGE_VALIDATION_DIR,
) -> dict[str, Any]:
    """Measure the judge on accepted pairs and on deliberate corruptions of them.

    Takes the first `n` accepted queries of bench.out_dir in qid order. For each,
    the paraphraser rewrites the proposal once per CORRUPTIONS entry, with exactly
    that change; the judge then answers both directions (always both, to see which
    one catches what) for the original and every corruption. A variant counts as
    accepted only if both say yes. Writes pairs.jsonl, corruptions.jsonl,
    verdicts.jsonl and summary.json to out_dir, replacing earlier ones (the LLM
    cache makes a rerun free).

    The originals passed this judge in make-bench, so with unchanged prompts and
    models they are accepted by construction; the informative numbers are the
    corruptions' rejection rates.
    """
    corpus, _, _ = load_frozen_corpus(cfg.corpus.out_dir)
    by_id = {r["id"]: r for r in corpus}
    accepted = sorted(_read_jsonl(cfg.bench.out_dir / ACCEPTED_FILE), key=lambda r: r["qid"])
    if not accepted:
        raise BenchError(
            f"no accepted queries in {cfg.bench.out_dir}: run `python -m rk make-bench` first"
        )
    prompts = load_prompts()
    corrupt = load_prompt("corrupt")
    usage = {"paraphraser": Usage(), "judge": Usage()}
    pairs: list[dict[str, Any]] = []
    corruptions: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    for row in accepted[:n]:
        record = by_id[row["target_id"]]
        title, abstract, original = record["title"], record["abstract"], row["query"]
        ids = {"qid": row["qid"], "target_id": row["target_id"]}
        pairs.append({**ids, "title": title, "proposal": original})
        variants = {"original": original}
        for kind, change in CORRUPTIONS.items():
            text = corrupt.render(title=title, abstract=abstract, proposal=original, change=change)
            reply = paraphraser.complete([{"role": "user", "content": text}])
            usage["paraphraser"] += reply.usage
            variants[kind] = clean_proposal(reply.text)
            unchanged = variants[kind] == original  # the paraphraser ignored the instruction
            corruptions.append(
                {**ids, "kind": kind, "proposal": variants[kind], "unchanged": unchanged}
            )
        for variant, proposal in variants.items():
            judgments: dict[str, Any] = {}
            for direction in JUDGE_DIRECTIONS:
                judgment, reply = judge_proposal(
                    judge, prompts[f"judge_{direction}"], title, abstract, proposal
                )
                usage["judge"] += reply.usage
                judgments[direction] = judgment or {"verdict": None, "raw": reply.text[:500]}
            verdicts.append(
                {
                    **ids,
                    "variant": variant,
                    "proposal": proposal,
                    "accepted": all(j["verdict"] == "yes" for j in judgments.values()),
                    "judgments": judgments,
                }
            )

    total = usage["paraphraser"] + usage["judge"]
    by_variant = {v: [r for r in verdicts if r["variant"] == v] for v in VARIANTS}
    summary = {
        "bench_dir": str(cfg.bench.out_dir),
        "n_pairs": len(pairs),
        "variants": {v: _variant_stats(rows) for v, rows in by_variant.items()},
        "unchanged_corruptions": dict(Counter(c["kind"] for c in corruptions if c["unchanged"])),
        "corruption_instructions": CORRUPTIONS,
        "prompt_hashes": {**prompt_hashes(prompts), "corrupt": corrupt.sha256},
        "models": {"paraphraser": cfg.paraphraser.name, "judge": cfg.judge.name},
        "llm_usage": {role: dataclasses.asdict(u) for role, u in usage.items()},
        "method_cost": total.method_cost,
        "money_spent": total.money_spent,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("pairs.jsonl", pairs),
        ("corruptions.jsonl", corruptions),
        ("verdicts.jsonl", verdicts),
    ):
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


HUMAN_CHECK_FILE = Path("data/human_check.csv")
HUMAN_CHECK_COLUMNS = ("qid", "title", "abstract", "proposal", "same_paper", "note")


def export_human_check(cfg: Config, *, n: int, out_path: Path = HUMAN_CHECK_FILE) -> int:
    """Write a seeded random sample of accepted pairs to a CSV for checking by hand.

    The sample is drawn with bench.seed and listed in qid order; same_paper and
    note are left empty for the reader. The file is UTF-8 with a byte-order mark
    so spreadsheet programs read it correctly. Refuses to overwrite a file, which
    may already hold someone's answers. Returns the number of rows written.
    """
    if out_path.exists():
        raise BenchError(f"{out_path} already exists: refusing to overwrite it (it may hold answers)")
    accepted = sorted(_read_jsonl(cfg.bench.out_dir / ACCEPTED_FILE), key=lambda r: r["qid"])
    if not accepted:
        raise BenchError(
            f"no accepted queries in {cfg.bench.out_dir}: run `python -m rk make-bench` first"
        )
    sample = random.Random(cfg.bench.seed).sample(accepted, min(n, len(accepted)))
    corpus, _, _ = load_frozen_corpus(cfg.corpus.out_dir)
    by_id = {r["id"]: r for r in corpus}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HUMAN_CHECK_COLUMNS)
        for row in sorted(sample, key=lambda r: r["qid"]):
            record = by_id[row["target_id"]]
            writer.writerow([row["qid"], record["title"], record["abstract"], row["query"], "", ""])
    return len(sample)


def _variant_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Acceptance and rejection counts of one variant, with what the judge flagged."""
    n = len(rows)
    accepted = sum(r["accepted"] for r in rows)
    flagged: Counter[str] = Counter()
    for r in rows:  # each variant counts once per facet, whichever direction flagged it
        flagged.update({f for j in r["judgments"].values() for f in j.get("failed", [])})
    return {
        "n": n,
        "accepted": accepted,
        "rejected": n - accepted,
        "acceptance_rate": accepted / n if n else None,
        "rejection_rate": (n - accepted) / n if n else None,
        "rejected_by_direction": {
            d: sum(r["judgments"][d]["verdict"] != "yes" for r in rows) for d in JUDGE_DIRECTIONS
        },
        "unreadable": sum(
            any(j["verdict"] is None for j in r["judgments"].values()) for r in rows
        ),
        "flagged_facets": {f: flagged[f] for f in FACETS},
    }
