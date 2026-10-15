"""Text utilities: tokenising, stopwords, generic-science words, a crude stemmer,
acronym extraction, named domain entities, title normalisation and OpenAlex
abstract reconstruction.

Everything here is deterministic and dependency-free so the lexical-leak filter
and the corpus builder give identical results on every run.
"""

from __future__ import annotations

import functools
import html
import itertools
import re
import unicodedata
from collections.abc import Mapping, Sequence
from importlib import resources

STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "across", "after", "again", "against", "ain", "al", "all", "also",
    "although", "am", "among", "amongst", "an", "and", "any", "are", "aren", "as", "at", "be",
    "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could",
    "couldn", "d", "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each",
    "eg", "either", "et", "etc", "few", "for", "from", "further", "had", "hadn", "has", "hasn",
    "have", "haven", "having", "he", "hence", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "however", "i", "ie", "if", "in", "into", "is", "isn", "it", "its", "itself",
    "just", "ll", "m", "ma", "may", "me", "might", "mightn", "more", "moreover", "most", "must",
    "mustn", "my", "myself", "needn", "neither", "no", "nor", "not", "now", "o", "of", "off", "on",
    "once", "one", "only", "onto", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "per", "re", "s", "same", "shall", "shan", "she", "should", "shouldn", "so", "some", "such",
    "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "therefore", "these", "they", "this", "those", "thus", "through", "to", "too", "toward",
    "towards", "under", "until", "up", "upon", "us", "ve", "very", "via", "vs", "was", "wasn", "we",
    "were", "weren", "what", "when", "where", "whether", "which", "while", "who", "whom", "why",
    "will", "with", "within", "without", "won", "would", "wouldn", "y", "yet", "you", "your",
    "yours", "yourself", "yourselves",
})

# Words that appear in almost any (AI/ML) abstract and so say nothing about a
# specific paper. The leak filter ignores them when looking for distinctive
# title terms (see is_generic for how inflections match). Tuned for the Phase 1
# subfield sweeps; extend as the pilot shows.
GENERIC_SCIENCE_WORDS: frozenset[str] = frozenset({
    "ability", "able", "accuracy", "accurate", "achieve", "achieves", "achieved", "address",
    "addresses", "advance", "advances", "algorithm", "algorithms", "analysis", "analyse", "analyze",
    "application", "applications", "apply", "applied", "approach", "approaches", "art", "aspect",
    "aspects", "available", "base", "based", "baseline", "baselines", "benchmark", "benchmarks",
    "better", "build", "case", "cases", "challenge", "challenges", "challenging", "compare",
    "compared", "comparison", "component", "components", "comprehensive", "conduct", "consider",
    "contribution", "contributions", "current", "data", "dataset", "datasets", "deep",
    "demonstrate", "demonstrates", "describe", "design", "designed", "develop", "developed",
    "development", "different", "effect", "effective", "effectiveness", "efficiency", "efficient",
    "empirical", "empirically", "enable", "enables", "enhance", "enhanced", "evaluate", "evaluated",
    "evaluation", "evidence", "example", "examples", "existing", "experiment", "experimental",
    "experiments", "explore", "extensive", "feature", "features", "field", "find", "finding",
    "findings", "first", "focus", "framework", "frameworks", "further", "furthermore", "future",
    "gain", "gains", "general", "given", "good", "high", "higher", "however", "identify", "impact",
    "implement", "implementation", "important", "improve", "improved", "improvement",
    "improvements", "improves", "include", "includes", "including", "increase", "information",
    "insight", "insights", "introduce", "introduces", "investigate", "issue", "issues", "key",
    "large", "learn", "learning", "level", "leverage", "limitation", "limitations", "limited",
    "low", "lower", "main", "make", "many", "measure", "method", "methodology", "methods", "metric",
    "metrics", "model", "models", "multiple", "need", "network", "networks", "neural", "new",
    "novel", "number", "obtain", "offer", "outperform", "outperforms", "overall", "paper",
    "particular", "perform", "performance", "potential", "present", "presents", "previous",
    "problem", "problems", "process", "propose", "proposed", "proposes", "provide", "provides",
    "quality", "range", "real", "recent", "recently", "reduce", "related", "report", "research",
    "result", "results", "reveal", "scale", "scenario", "scenarios", "set", "several", "show",
    "shows", "significant", "significantly", "similar", "simple", "small", "solution", "solutions",
    "specific", "state", "strategy", "strong", "study", "superior", "system", "systems", "task",
    "tasks", "technique", "techniques", "test", "tool", "tools", "train", "trained", "training",
    "type", "understand", "use", "used", "useful", "uses", "using", "validate", "value", "variety",
    "various", "way", "well", "wide", "widely", "work", "works", "world",
})

_TOKEN_RE = re.compile(r"[^\W_]+")
_NON_ALNUM_RE = re.compile(r"[\W_]+")
_TAG_RE = re.compile(r"<[^>]+>")


def _fold(text: str) -> str:
    """Strip accents: 'naïve' -> 'naive'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Lowercase, accent-folded alphanumeric tokens; punctuation and hyphens split."""
    return _TOKEN_RE.findall(_fold(text).lower())


# Checked longest first; one suffix per round, up to _STEM_ROUNDS rounds.
_SUFFIXES: tuple[str, ...] = (
    "ization", "isation",
    "ation", "ition", "ative", "izing", "ising", "ating",
    "ment", "ness", "ized", "ised", "ated", "izer", "iser",
    "ity", "ive", "ing", "ion", "ize", "ise", "ate",
    "ed", "er", "ly", "al",
)
_MIN_STEM = 3
_MIN_STEM_FOR = {"al": 5}  # "retrieval" -> "retriev", but "neural", "local" stay whole
_STEM_ROUNDS = 3


def stem(word: str) -> str:
    """Crude suffix-stripping stemmer.

    Not linguistically correct, only consistent: inflected and derived forms of
    a word usually collapse to one key ("segment", "segmenting", "segmentation"
    -> "seg"). Unrelated words can collide ("gene", "generation" -> "gen"); for
    the leak filter that errs on the strict side. Short words and tokens
    containing digits are returned unchanged.
    """
    w = word.lower()
    if len(w) <= _MIN_STEM or not w.isalpha():
        return w
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith(("ies", "ied")) and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for _ in range(_STEM_ROUNDS):
        for suffix in _SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= _MIN_STEM_FOR.get(suffix, _MIN_STEM):
                w = w[: -len(suffix)]
                if len(w) > _MIN_STEM and w[-1] == w[-2] and w[-1] not in "aeious":
                    w = w[:-1]  # "embedd" -> "embed", "mapp" -> "map"
                break
        else:
            break
    if len(w) > _MIN_STEM and w.endswith("e"):
        w = w[:-1]
    return w


def content_tokens(text: str) -> list[str]:
    """Tokens minus stopwords, bare numbers and single characters, in order."""
    return [t for t in tokenize(text) if len(t) > 1 and not t.isdigit() and t not in STOPWORDS]


# Stems of listed words, kept only when at least 5 characters long: short stems
# collide too easily ("general" and "generation" both stem to "gen").
_GENERIC_STEMS: frozenset[str] = frozenset(
    s for s in map(stem, GENERIC_SCIENCE_WORDS) if len(s) >= 5
)


def is_generic(token: str) -> bool:
    """True if `token` is a listed generic-science word or an inflection of one.

    Inflections are matched by stem, but only through stems of 5+ characters:
    "proposing" is generic (stem "propos"), "generation" is not (stem "gen").
    """
    word = token.lower()
    return word in GENERIC_SCIENCE_WORDS or stem(word) in _GENERIC_STEMS


def content_stems(text: str, *, drop_generic: bool = False) -> list[str]:
    """Stems of content tokens, in order; optionally minus generic-science words."""
    tokens = content_tokens(text)
    if drop_generic:
        tokens = [t for t in tokens if not is_generic(t)]
    return [stem(t) for t in tokens]


_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*")  # hyphenated compounds stay whole
_LETTER_RUN_RE = re.compile(r"[^\W\d_]{2,}")
_ROMAN_RE = re.compile(r"[IVXLC]+")
_ORDINAL_RE = re.compile(r"\d+(?:st|nd|rd|th)", re.IGNORECASE)


def _acronym_form(part: str, allow_caps: bool) -> str | None:
    """The acronym/named-artefact form of one hyphen-free word part, or None."""
    has_alpha = any(c.isalpha() for c in part)
    if not has_alpha:
        return None
    if any(c.isdigit() for c in part):  # BM25, T5, word2vec, YOLOv8
        return None if _ORDINAL_RE.fullmatch(part) else part
    if any(a.islower() and b.isupper() for a, b in itertools.pairwise(part)):  # ResNet, LoRA
        return part
    if allow_caps:
        core = part[:-1] if part.endswith("s") and part[:-1].isupper() else part  # CNNs
        if sum(c.isupper() for c in core) >= 2 and not _ROMAN_RE.fullmatch(core):
            return core
    return None


def extract_acronyms(text: str) -> list[str]:
    """Acronyms and named artefacts (model, dataset, method names) in `text`.

    Matches words with two or more capitals (BERT, CNNs -> CNN), letter-digit
    mixes (BM25, word2vec), internal capitals (ImageNet, LoRA) and capitalised
    names with a version number (Llama-2). A hyphenated name is returned whole
    and by its acronym parts ("GPT-4" -> "GPT-4", "GPT"). Roman numerals and
    ordinals are skipped. If most words are all-caps (a shouting title), the
    two-capitals rule is switched off. Returns sorted surface forms; matching
    them against other text (case sensitivity etc.) is up to the caller.
    """
    letter_runs = _LETTER_RUN_RE.findall(text)
    shouting = len(letter_runs) >= 4 and sum(w.isupper() for w in letter_runs) > len(letter_runs) / 2
    found: set[str] = set()
    for match in _WORD_RE.finditer(text):
        token = match.group()
        parts = token.split("-")
        forms = [_acronym_form(p, allow_caps=not shouting) for p in parts]
        found.update(f for f in forms if f)
        if len(parts) > 1:
            whole = all(
                f is not None or p.isdigit() or (i == 0 and p[0].isupper())
                for i, (p, f) in enumerate(zip(parts, forms))
            )
            if whole and (any(forms) or any(p.isdigit() for p in parts[1:])):
                found.add(token)
    return sorted(found)


def _data_list(name: str) -> tuple[str, ...]:
    """Non-comment lines of rk/data/<name>; a leading "=" (kept) marks exact-case matching."""
    text = resources.files("rk").joinpath("data").joinpath(name).read_text(encoding="utf-8")
    lines = (line.split("#", 1)[0].strip() for line in text.splitlines())
    return tuple(line for line in lines if line)


@functools.lru_cache(maxsize=1)
def domain_entities() -> tuple[str, ...]:
    """Languages, nationalities, countries and regions (rk/data/domain_entities.txt)."""
    return _data_list("domain_entities.txt")


@functools.lru_cache(maxsize=1)
def domain_fields() -> tuple[str, ...]:
    """Research fields and application domains (rk/data/domain_fields.txt)."""
    return _data_list("domain_fields.txt")


@functools.lru_cache(maxsize=1)
def _domain_term_re() -> re.Pattern[str]:
    alternatives = []
    entries = sorted({*domain_entities(), *domain_fields()}, key=len, reverse=True)
    for entry in entries:  # longest first: "South Africa" before "Africa"
        name = entry.removeprefix("=")
        body = r"\s+".join(re.escape(word) for word in name.split())
        exact = entry.startswith("=") or name.isupper()
        alternatives.append(body if exact else f"(?i:{body})")
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")s?(?!\w)")


def find_domain_terms(text: str) -> list[str]:
    """Languages, nationalities, countries, regions and fields named in `text`, in order.

    Whole words only ("TurkColBERT" names a model, not Turkish); repeats that
    differ only in case or a plural "s" are listed once.
    """
    found: dict[str, str] = {}
    for match in _domain_term_re().finditer(text):
        name = " ".join(match.group().split())
        found.setdefault(name.lower().removesuffix("s"), name)
    return list(found.values())


def mask_domain_terms(text: str) -> str:
    """`text` with every named language, nationality, country, region and field blanked out.

    The lexical-leak filter runs on masked text, so a proposal may keep the
    language, region or field its paper names.
    """
    return _domain_term_re().sub(" ", text)


_CAPITALISED_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")  # "Bilibili"; acronyms are handled elsewhere
_LOWER_WORD_RE = re.compile(r"\b[a-z]{3,}\b")
_SENTENCE_END = ".!?:"


def capitalised_words(text: str, *, mid_sentence: bool) -> set[str]:
    """Words written with one leading capital ("Bilibili"), optionally only mid-sentence.

    A word is mid-sentence if some text precedes it and the last non-space
    character before it doesn't end a sentence (.!?:).
    """
    found = set()
    for match in _CAPITALISED_RE.finditer(text):
        i = match.start() - 1
        while i >= 0 and text[i].isspace():
            i -= 1
        if mid_sentence and (i < 0 or text[i] in _SENTENCE_END):
            continue
        found.add(match.group())
    return found


def lowercase_words(text: str) -> set[str]:
    """Words of three or more letters written all in lower case."""
    return set(_LOWER_WORD_RE.findall(text))


def normalize_title(title: str | None) -> str:
    """Dedupe key for titles: no markup, accents, case, punctuation or spaces.

    "Pre-Training <i>BERT</i>: A Study." and "pretraining BERT - a study" give
    the same key. Not meant for display.
    """
    if not title:
        return ""
    text = _TAG_RE.sub("", html.unescape(title))
    return _NON_ALNUM_RE.sub("", _fold(text).lower())


def opening(text: str, chars: int) -> str:
    """The first `chars` characters of a text, cut at a word, with an ellipsis if cut."""
    if len(text) <= chars:
        return text
    return text[:chars].rsplit(" ", 1)[0] + " …"


def reconstruct_abstract(inverted_index: Mapping[str, Sequence[int]] | None) -> str:
    """Rebuild an abstract from OpenAlex's `abstract_inverted_index` (word -> positions)."""
    if not inverted_index:
        return ""
    words_at: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for position in positions:
            words_at.setdefault(position, word)
    return " ".join(words_at[p] for p in sorted(words_at))
