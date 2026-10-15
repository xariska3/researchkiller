"""Does a listed paper exist? Checked through free routes only.

- OpenAlex singleton lookups (https://help.openalex.org/access/example-costs/,
  checked 2026-10-14): one work by its id or DOI (/works/W123,
  /works/doi:10.x/y) is free and doesn't count against the daily budget. OpenAlex
  searches cost $1 per 1,000 and are never used here.
- Crossref title search (https://api.crossref.org/swagger-ui/index.html; limits from
  https://www.crossref.org/documentation/retrieve-metadata/rest-api/, "Access and
  authentication"): /works?query.bibliographic=<title> ranks works by bibliographic
  match. Free; the public pool allows a rate of 5 with 1 request at a time, and a
  `mailto` gives the polite pool's 10 with 3.

A listed paper is checked in this order, stopping at the first work whose title agrees
with the listed title (or at the first work found, if no title was listed):

1. an OpenAlex work id in its link (openalex.org/W...): OpenAlex by id;
2. its DOI, or one in its link (a doi.org link or any URL holding 10.NNNN/...):
   OpenAlex by DOI;
3. an arXiv id in its link (arxiv.org/abs/ or /pdf/): OpenAlex by arXiv's DataCite DOI,
   10.48550/arXiv.<id>;
4. its title: Crossref's top 5 bibliographic matches; a match's DOI is then looked up
   in OpenAlex for its work id.

An id or DOI that leads to a work with another title is a DOI mismatch (it points at
the wrong paper), and the next route is tried. A paper no route verifies is
unresolved: it may not exist, or exist only where these routes can't see it (an arXiv
preprint listed without a link, for example, since Crossref doesn't index arXiv).

Titles agree if their normalised forms are equal, their content stems have Jaccard
similarity >= 0.8, or the stems of one (at least 4) are all in the other (a dropped
subtitle). Every lookup, a miss included, is cached in the compete cache file.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sqlite3
import time
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from rk.compete.systems import Http, Item, find_doi
from rk.config import Config
from rk.corpus import OPENALEX_URL, normalize_doi, short_id
from rk.live.pool import title_jaccard
from rk.text import content_stems, normalize_title

CROSSREF_URL = "https://api.crossref.org"
OPENALEX_RATE = 10.0  # requests per second, the keyless limit
CROSSREF_RATE = 5.0  # requests per second, the public pool's limit
CROSSREF_ROWS = 5
TITLE_MIN_JACCARD = 0.8  # as near-duplicate merging in the live pool
TITLE_MIN_SUBSET_STEMS = 4
OPENALEX_SELECT = "id,doi,title,publication_year"
VERIFIED, UNRESOLVED = "verified", "unresolved"

_OPENALEX_ID_RE = re.compile(r"openalex\.org/(?:works/)?(W\d+)", re.IGNORECASE)
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS compete_lookups (
    key TEXT PRIMARY KEY,
    response TEXT,
    fetched_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class Resolution:
    status: str  # VERIFIED or UNRESOLVED
    route: str | None = None  # openalex-id, openalex-doi, openalex-arxiv or crossref-title
    openalex_id: str | None = None
    doi: str | None = None
    title: str | None = None  # as the source has it
    year: int | None = None
    doi_mismatch: bool = False  # a listed id or DOI led to a work with another title

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def titles_agree(a: str, b: str) -> bool:
    """The same title: normalised equal, stem Jaccard >= 0.8, or one's 4+ stems in the other."""
    if normalize_title(a) and normalize_title(a) == normalize_title(b):
        return True
    if title_jaccard(a, b) >= TITLE_MIN_JACCARD:
        return True
    shorter, longer = sorted((set(content_stems(a)), set(content_stems(b))), key=len)
    return len(shorter) >= TITLE_MIN_SUBSET_STEMS and shorter <= longer


def _ident(value: str) -> str:
    """An OpenAlex singleton path segment, with the characters a URL can't hold escaped."""
    return urllib.parse.quote(value, safe="/:;()._-~,")


class Resolver:
    """Checks listed papers against OpenAlex (by id or DOI) and Crossref (by title)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        openalex_key: str | None = None,
        crossref_mailto: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._conn = conn
        conn.execute(_SCHEMA)
        conn.commit()
        self._openalex_headers = {"Authorization": f"Bearer {openalex_key}"} if openalex_key else {}
        self._mailto = crossref_mailto

        def http(rate: float) -> Http:
            return Http(
                requests_per_second=rate,
                timeout=60.0,
                transport=transport,
                sleep=sleep,
                clock=clock,
            )

        self._openalex = http(OPENALEX_RATE)
        self._crossref = http(CROSSREF_RATE)

    @classmethod
    def from_config(cls, cfg: Config, conn: sqlite3.Connection, **kwargs: Any) -> Resolver:
        """With the OpenAlex key (corpus.api_key_env) and Crossref mailto the config names."""
        key_env = cfg.corpus.api_key_env
        mail_env = cfg.compete.crossref_mailto_env if cfg.compete else None
        return cls(
            conn,
            openalex_key=(os.environ.get(key_env) or None) if key_env else None,
            crossref_mailto=(os.environ.get(mail_env) or None) if mail_env else None,
            **kwargs,
        )

    @property
    def requests_sent(self) -> int:
        return self._openalex.calls + self._crossref.calls

    def close(self) -> None:
        self._openalex.close()
        self._crossref.close()

    # --- cached lookups ---------------------------------------------------------------

    def _cached(self, key: str, fetch: Callable[[], Any]) -> Any:
        row = self._conn.execute(
            "SELECT response FROM compete_lookups WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            return None if row[0] is None else json.loads(row[0])
        data = fetch()
        self._conn.execute(
            "INSERT OR REPLACE INTO compete_lookups (key, response, fetched_at) VALUES (?, ?, ?)",
            (
                key,
                None if data is None else json.dumps(data, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return data

    def openalex_work(self, ident: str) -> dict[str, Any] | None:
        """One OpenAlex work by id (W123) or DOI (doi:10.x/y), or None if there is none."""

        def fetch() -> Any:
            data, _ = self._openalex.call(
                "GET",
                f"{OPENALEX_URL}/works/{_ident(ident)}",
                headers=self._openalex_headers,
                params={"select": OPENALEX_SELECT},
                missing_ok=True,
            )
            return data

        data = self._cached(f"openalex:{ident}", fetch)
        if not isinstance(data, dict) or not data.get("id"):
            return None
        year = data.get("publication_year")
        return {
            "id": short_id(data["id"]),
            "doi": normalize_doi(data.get("doi")),
            "title": data.get("title") or "",
            "year": year if isinstance(year, int) else None,
        }

    def crossref_matches(self, title: str) -> list[dict[str, Any]]:
        """Crossref's top bibliographic matches for a title: DOI, title and year."""

        def fetch() -> Any:
            params = {
                "query.bibliographic": title,
                "rows": str(CROSSREF_ROWS),
                "select": "DOI,title,issued",
            }
            if self._mailto:
                params["mailto"] = self._mailto
            data, _ = self._crossref.call("GET", f"{CROSSREF_URL}/works", params=params)
            return data

        data = self._cached(f"crossref:{' '.join(title.split())}", fetch)
        message = data.get("message") if isinstance(data, dict) else None
        matches = []
        for work in (message or {}).get("items") or []:
            titles = work.get("title") or []
            parts = ((work.get("issued") or {}).get("date-parts") or [[None]])[0] or [None]
            matches.append(
                {
                    "doi": find_doi(work.get("DOI")),
                    "title": titles[0] if titles and isinstance(titles[0], str) else "",
                    "year": parts[0] if isinstance(parts[0], int) else None,
                }
            )
        return matches

    # --- resolving ----------------------------------------------------------------------

    @staticmethod
    def identifiers(item: Item) -> Iterator[tuple[str, str]]:
        """(route, OpenAlex singleton id) pairs to try for an item, in order, without repeats."""
        seen: set[str] = set()
        link = item.link or ""
        candidates = []
        match = _OPENALEX_ID_RE.search(link)
        if match:
            candidates.append(("openalex-id", match.group(1).upper()))
        doi = item.doi or find_doi(urllib.parse.unquote(link))
        if doi:
            candidates.append(("openalex-doi", f"doi:{doi}"))
        match = _ARXIV_RE.search(link)
        if match:
            candidates.append(("openalex-arxiv", f"doi:10.48550/arxiv.{match.group(1)}"))
        for route, ident in candidates:
            if ident not in seen:
                seen.add(ident)
                yield route, ident

    def resolve(self, item: Item) -> Resolution:
        mismatch = False
        for route, ident in self.identifiers(item):
            work = self.openalex_work(ident)
            if work is None:
                continue
            if item.title is None or titles_agree(item.title, work["title"]):
                return Resolution(
                    VERIFIED, route, work["id"], work["doi"], work["title"], work["year"], mismatch
                )
            mismatch = True
        if item.title:
            for match in self.crossref_matches(item.title):
                if not titles_agree(item.title, match["title"]):
                    continue
                work = self.openalex_work(f"doi:{match['doi']}") if match["doi"] else None
                return Resolution(
                    VERIFIED,
                    "crossref-title",
                    work["id"] if work else None,
                    match["doi"],
                    match["title"],
                    match["year"],
                    mismatch,
                )
        return Resolution(UNRESOLVED, doi_mismatch=mismatch)
