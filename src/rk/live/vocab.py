"""The vocabulary map of a check (docs/phase2_spec.md, Pipeline 6; Step 5): deterministic.

Terms the literature uses for the idea that the user's text doesn't, from three
sources, each a group of the map:
- **keywords**: the top papers' OpenAlex keywords, each one term;
- **titles**: phrases of the top papers' titles;
- **queries**: phrases of the agent's own queries (not the user's text).

A phrase is a run of words between stopwords and punctuation, without leading generic
science words ("novel", "efficient"); a run longer than MAX_TERM_WORDS is cut at the
words the text already has. A term is new if one of its content words, by stem and
other than generic science words, isn't in the text. Its support is how many top papers
use all those stems (in their title, abstract or keywords); a term no top paper uses
is dropped. Each group is ranked by support, then by first appearance, and the map
takes the best remaining term of each group in turn until it has `max_terms`. A term
already taken (the same stems) isn't repeated in a later group.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rk.live.report import VocabularyGroup
from rk.targets import Record
from rk.text import STOPWORDS, content_stems, is_generic

KEYWORDS, TITLES, QUERIES = "keywords", "titles", "queries"
GROUPS = (KEYWORDS, TITLES, QUERIES)  # in the order the map takes turns
LABELS = {
    KEYWORDS: "Keywords of the closest papers (OpenAlex)",
    TITLES: "In the closest papers' titles",
    QUERIES: "Searched for by the agent",
}
MAX_TERM_WORDS = 4

# Words (hyphenated compounds whole) and single punctuation marks, which end a phrase.
_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*|[^\w\s]")


def _is_word(token: str) -> bool:
    return token[0].isalnum() and token.lower() not in STOPWORDS and not token.isdigit()


def _runs(text: str) -> list[list[str]]:
    """Runs of words between stopwords, bare numbers and punctuation, as written."""
    runs: list[list[str]] = []
    current: list[str] = []
    for token in _TOKEN_RE.findall(text):
        if _is_word(token):
            current.append(token)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _stems(words: Sequence[str]) -> tuple[str, ...]:
    """The term's content stems, minus generic science words (its identity)."""
    return tuple(content_stems(" ".join(words), drop_generic=True))


def _cut(run: list[str], known: set[str]) -> list[list[str]]:
    """A run as phrases: whole if short, else cut at words the text has, then in pieces."""
    if len(run) <= MAX_TERM_WORDS:
        return [run]
    pieces: list[list[str]] = []
    current: list[str] = []
    for word in run:
        if all(s in known for s in content_stems(word)):
            if current:
                pieces.append(current)
            current = []
        else:
            current.append(word)
    if current:
        pieces.append(current)
    return [p[i : i + MAX_TERM_WORDS] for p in pieces for i in range(0, len(p), MAX_TERM_WORDS)]


def _phrases(text: str, known: set[str]) -> list[list[str]]:
    out = []
    for run in _runs(text):
        for piece in _cut(run, known):
            while piece and is_generic(piece[0]):
                piece = piece[1:]
            out.append(piece)
    return out


def _display(words: Sequence[str]) -> str:
    """Lower case, except words with two or more capitals (acronyms, names like 'BERT')."""
    return " ".join(w if sum(c.isupper() for c in w) >= 2 else w.lower() for w in words)


def vocabulary(
    text: str,
    queries: Sequence[str],
    papers: Sequence[Record],
    keywords: Mapping[str, Sequence[str]],
    *,
    max_terms: int,
) -> list[VocabularyGroup]:
    """The vocabulary map: up to `max_terms` new terms, grouped by source (GROUPS order)."""
    known = set(content_stems(text))
    paper_stems = [
        set(content_stems(" ".join([p["title"], p["abstract"], *keywords.get(p["id"], [])])))
        for p in papers
    ]
    sources: dict[str, list[list[str]]] = {
        KEYWORDS: [
            _TOKEN_RE.findall(k) for p in papers for k in keywords.get(p["id"], [])
        ],
        TITLES: [w for p in papers for w in _phrases(p["title"], known)],
        QUERIES: [w for q in queries for w in _phrases(q, known)],
    }
    ranked: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for group, phrases in sources.items():
        seen: dict[tuple[str, ...], tuple[int, int, str]] = {}  # stems -> support, order, form
        for order, phrase in enumerate(phrases):
            words = [w for w in phrase if w[0].isalnum()]
            stems = _stems(words)
            if not stems or stems in seen or all(s in known for s in stems):
                continue
            support = sum(set(stems) <= used for used in paper_stems)
            if support:
                seen[stems] = (support, order, _display(words))
        best = sorted(seen.items(), key=lambda item: (-item[1][0], item[1][1]))
        ranked[group] = [(stems, form) for stems, (_, _, form) in best]
    taken: dict[str, list[str]] = {g: [] for g in GROUPS}
    used: set[tuple[str, ...]] = set()
    position = {g: 0 for g in GROUPS}
    while sum(len(t) for t in taken.values()) < max_terms:
        added = False
        for group in GROUPS:
            terms = ranked[group]
            while position[group] < len(terms) and terms[position[group]][0] in used:
                position[group] += 1
            if position[group] == len(terms):
                continue
            stems, form = terms[position[group]]
            used.add(stems)
            taken[group].append(form)
            added = True
            if sum(len(t) for t in taken.values()) == max_terms:
                break
        if not added:
            break
    return [
        VocabularyGroup(source=g, label=LABELS[g], terms=taken[g]) for g in GROUPS if taken[g]
    ]
