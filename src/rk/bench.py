"""Test A benchmark generation (spec: Benchmark generation).

For each target the paraphraser writes a terminology-shifted research proposal
from the title and abstract. A length check and a deterministic lexical-leak
filter run first; then the judge (another model family) checks the two
directions of equivalence in two separate calls. The first failing check is
the rejection reason, and the next attempt's prompt quotes it. Every attempt is
appended to test_a.jsonl (accepted) or rejections.jsonl (rejected) as it
happens, which is also what makes a rerun resume where the last one stopped.
"""

from __future__ import annotations

import dataclasses
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
from rk.text import STOPWORDS, content_stems, content_tokens, extract_acronyms, stem

PROMPT_NAMES = ("paraphrase", "judge_captures", "judge_adds_nothing")
MIN_SENTENCES, MAX_SENTENCES = 2, 4
MIN_WORDS, MAX_WORDS = 40, 120
IDF_TOP_TERMS = 20
JUDGE_DIRECTIONS = ("captures", "adds_nothing")

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
    3. any distinctive title stem (title content stems minus generic words);
    4. more than a fraction max_overlap of the abstract's top-20 IDF stems;
    5. content-stem Jaccard with title + abstract above max_jaccard, generic
       science words excluded on both sides (text.is_generic).
    """

    def __init__(self, corpus: Iterable[Record], *, max_overlap: float, max_jaccard: float):
        df: Counter[str] = Counter()
        n_docs = 0
        for record in corpus:
            n_docs += 1
            df.update(set(content_stems(f"{record['title']} {record['abstract']}")))
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

    def check(
        self, proposal: str, title: str, abstract: str
    ) -> tuple[Rejection | None, dict[str, Any]]:
        own_acronyms = extract_acronyms(proposal)

        source = set(extract_acronyms(title)) | set(extract_acronyms(abstract))
        source_by_key = {_match_key(a): a for a in sorted(source)}
        proposal_keys = _word_keys(proposal)
        source_found = sorted(
            a for key, a in source_by_key.items() if key in proposal_keys and key not in STOPWORDS
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


def parse_verdict(text: str) -> dict[str, str] | None:
    """{"verdict": "yes"|"no", "reason": ...} from a judge reply, or None if unreadable."""
    try:
        data = parse_json(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().lower() not in ("yes", "no"):
        return None
    reason = data.get("reason")
    return {"verdict": verdict.strip().lower(), "reason": reason if isinstance(reason, str) else ""}


@dataclass
class Attempt:
    query: str
    rejection: Rejection | None
    leak_stats: dict[str, Any]
    judge_verdicts: dict[str, Any]
    usage: dict[str, Usage]
    providers: dict[str, list[str | None]]  # per role, who served each call, in order


_JUDGE_NO = {
    "captures": "a reviewer found that it does not capture the paper's problem, method and claim",
    "adds_nothing": "a reviewer found that it adds something the paper does not support",
}


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
    prompt = prompts["paraphrase"].render(title=title, abstract=abstract, feedback=feedback)
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
            judge_prompt = prompts[f"judge_{direction}"].render(
                title=title, abstract=abstract, proposal=query
            )
            reply: Completion = judge.complete([{"role": "user", "content": judge_prompt}])
            usage["judge"] = usage["judge"] + reply.usage
            providers["judge"].append(reply.provider)
            verdict = parse_verdict(reply.text)
            if verdict is None:
                verdicts[direction] = {"verdict": None, "raw": reply.text[:500]}
                rejection = Rejection(
                    "judge_parse_error", "the reviewer's answer about it could not be read"
                )
                break
            verdicts[direction] = verdict
            if verdict["verdict"] == "no":
                rejection = Rejection(
                    f"judge_{direction}", f"{_JUDGE_NO[direction]} ({verdict['reason']})"
                )
                break
    return Attempt(query, rejection, {**length_stats, **leak_stats}, verdicts, usage, providers)


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
        "prompt_hashes": {name: p.sha256 for name, p in prompts.items()},
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
