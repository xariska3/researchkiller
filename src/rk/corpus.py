"""Corpus build (spec: Corpus): OpenAlex sweeps, dedupe, targets, citation expansion, freeze.

OpenAlex API facts this module relies on, checked 2026-10-12 against the docs
and with small live probes:

- Sorting (https://help.openalex.org/api/sorting/): the documented sort fields
  are display_name, cited_by_count, works_count, publication_date and
  relevance_score; several fields are comma-separated, ":desc" sorts
  descending. `id` is not sortable (sort=publication_date:desc,id:asc returned
  HTTP 400), so ties on publication_date are ordered locally (fetch_newest).
- Paging (https://help.openalex.org/api/paging/): per_page is at most 100 (200
  is deprecated legacy, so PER_PAGE is 100, not the spec's 200). Cursor paging
  starts at cursor=*, follows meta.next_cursor and ends when next_cursor is
  null and results are empty. The docs don't cover sort with cursor paging;
  probes showed sort=publication_date:desc&cursor=* works.
- Filters (https://help.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists):
  comma is AND, | is OR within one filter, at most 100 OR values (references
  go IDS_PER_CALL=50 per call). Probes confirmed ids.openalex and cites.
- Auth and limits (https://help.openalex.org/api/authentication/,
  https://help.openalex.org/api/deprecations.md,
  https://help.openalex.org/access/example-costs/): the key goes in an api_key
  query parameter or an "Authorization: Bearer" header. We use the header, so
  the key never appears in URLs or error messages. Keys are optional; a free
  key gives 10x the keyless daily budget ($1/day vs $0.10/day; list calls cost
  $0.0001). The mailto "polite pool" was replaced by API keys in February 2026
  and mailto is now ignored; it is still sent when configured. More than 100
  requests per second gets HTTP 429.
- Errors (https://help.openalex.org/api/errors/): retry 429, 5xx and timeouts
  with exponential backoff (1 s, 2 s, 4 s, ...) and don't retry other 4xx. No
  Retry-After header is documented; one is honoured if sent.
- Daily budget (https://help.openalex.org/how-to-use-the-api/rate-limits-and-authentication,
  checked 2026-10-14): a spent daily budget and a per-second overrun both
  return HTTP 429, with no documented difference in the body. Every response
  has X-RateLimit-Remaining (credits left today; 1 credit = $0.0001, as
  /rate-limit reports: credits_limit 10,000 for daily_budget_usd 1) and
  X-RateLimit-Reset (seconds until midnight UTC). So a 429 whose remaining
  credits can't pay for the call is a spent budget (OpenAlexBudgetExhausted,
  not retried); any other 429 is retried. GET /rate-limit is free (a probe
  cost 0 credits) and reports daily_remaining_usd, prepaid_remaining_usd and
  resets_in_seconds.
- Data: newest-first results start with bogus future dates (a probe returned
  2031-01-01 and 2030-01-01 first), so sweep and citer queries add
  to_publication_date:<build date>.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from rk.config import Config, CorpusConfig
from rk.llm import retry_after_seconds
from rk.targets import Record, TargetSelectionError, id_number, select_targets, target_problems
from rk.text import content_tokens, normalize_title, reconstruct_abstract

OPENALEX_URL = "https://api.openalex.org"
PER_PAGE = 100  # documented maximum
MAX_REQUESTS_PER_SECOND = 100.0  # documented limit
CREDIT_USD = 0.0001  # one OpenAlex credit
LIST_CALL_USD = 0.0001  # documented price of one list+filter call
IDS_PER_CALL = 50
SELECT = (
    "id,doi,title,publication_date,abstract_inverted_index,referenced_works,"
    "primary_topic,cited_by_count,language"
)
NEWEST_FIRST = "publication_date:desc"
ORIGINS = ("sweep", "reference", "citer")  # precedence when versions merge
TITLE_MERGE_MIN_CONTENT_WORDS = 5
TITLE_MERGE_MAX_YEAR_GAP = 2

CORPUS_FILE = "corpus.jsonl"
TARGETS_FILE = "targets.jsonl"
MANIFEST_FILE = "manifest.json"


class OpenAlexError(RuntimeError):
    """An OpenAlex request failed: rejected, malformed, or retries exhausted."""


class OpenAlexBudgetExhausted(OpenAlexError):
    """HTTP 429 because today's budget can't pay for the call; it resets at midnight UTC."""

    def __init__(self, message: str, *, resets_in_s: float | None = None) -> None:
        super().__init__(message)
        self.resets_in_s = resets_in_s


class CorpusError(RuntimeError):
    """The corpus can't be built or frozen as asked."""


def _header_number(response: httpx.Response, name: str) -> float | None:
    try:
        return float(response.headers[name])
    except (KeyError, ValueError):
        return None


class OpenAlexClient:
    """Throttled, retrying client for OpenAlex /works list requests.

    `call_cost_usd` is what one call costs: a 429 whose X-RateLimit-Remaining
    can't pay for it means today's budget is spent, so it raises
    OpenAlexBudgetExhausted at once instead of being retried.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mailto: str | None = None,
        transport: httpx.BaseTransport | None = None,
        base_url: str = OPENALEX_URL,
        max_requests_per_second: float = MAX_REQUESTS_PER_SECOND,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        timeout: float = 60.0,
        call_cost_usd: float = LIST_CALL_USD,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.authenticated = bool(api_key)
        self.calls = 0  # HTTP requests sent, retries included
        self.call_credits = round(call_cost_usd / CREDIT_USD)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._mailto = mailto
        self._interval = 1.0 / max_requests_per_second
        self._next_slot = 0.0
        self._sleep = sleep
        self._clock = clock
        self._http = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    @classmethod
    def from_config(cls, corpus: CorpusConfig, **kwargs: Any) -> OpenAlexClient:
        """Client with the API key and mailto read from the env vars the config names."""
        api_key = os.environ.get(corpus.api_key_env) if corpus.api_key_env else None
        mailto = os.environ.get(corpus.mailto_env) if corpus.mailto_env else None
        return cls(api_key=api_key or None, mailto=mailto or None, **kwargs)

    def get_works(self, params: dict[str, str]) -> dict[str, Any]:
        """One GET /works page, throttled, retrying 429, 5xx and transport errors."""
        if self._mailto:
            params = {**params, "mailto": self._mailto}
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.calls += 1
            retry_after: float | None = None
            try:
                response = self._http.get("/works", params=params)
            except httpx.TransportError as exc:  # timeouts are transport errors in httpx
                problem = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status == 429 and self._budget_spent(response):
                    raise OpenAlexBudgetExhausted(
                        f"OpenAlex's daily budget is spent: "
                        f"{response.headers['x-ratelimit-remaining']} credits left, and a call "
                        f"costs {self.call_credits}",
                        resets_in_s=_header_number(response, "x-ratelimit-reset"),
                    )
                if status == 429 or status >= 500:
                    problem = f"HTTP {status}: {response.text[:300]}"
                    retry_after = retry_after_seconds(response)
                elif response.is_error:
                    raise OpenAlexError(
                        f"HTTP {status} for filter {params.get('filter')!r}: {response.text[:500]}"
                    )
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise OpenAlexError("response is not JSON") from exc
                    if not isinstance(data, dict):
                        raise OpenAlexError(f"unexpected response: {str(data)[:300]}")
                    return data
            if attempt == self.max_retries:
                raise OpenAlexError(
                    f"giving up after {attempt + 1} attempts for filter "
                    f"{params.get('filter')!r}; last error: {problem}"
                )
            delay = self.backoff_base * 2**attempt
            if retry_after is not None:
                delay = max(delay, retry_after)
            self._sleep(min(delay, self.backoff_max))
        raise AssertionError("unreachable")

    def iter_works(
        self, filters: str, *, sort: str | None = None, per_page: int = PER_PAGE
    ) -> Iterator[dict[str, Any]]:
        """Every work matching `filters`, fetched lazily with cursor paging."""
        if not 1 <= per_page <= PER_PAGE:
            raise ValueError(f"per_page must be in [1, {PER_PAGE}]")
        cursor: str | None = "*"
        while cursor:
            params = {
                "filter": filters,
                "select": SELECT,
                "per_page": str(per_page),
                "cursor": cursor,
            }
            if sort:
                params["sort"] = sort
            page = self.get_works(params)
            results = page.get("results") or []
            yield from results
            if not results:
                return
            cursor = (page.get("meta") or {}).get("next_cursor")

    def rate_limit(self) -> dict[str, Any]:
        """Today's budget for the API key, from GET /rate-limit (free; not retried)."""
        self._throttle()
        self.calls += 1
        try:
            response = self._http.get("/rate-limit")
        except httpx.TransportError as exc:
            raise OpenAlexError(f"/rate-limit: {type(exc).__name__}: {exc}") from exc
        if response.is_error:
            raise OpenAlexError(f"/rate-limit: HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise OpenAlexError("/rate-limit: response is not JSON") from exc
        budget = data.get("rate_limit") if isinstance(data, dict) else None
        if not isinstance(budget, dict):
            raise OpenAlexError(f"/rate-limit: unexpected response: {str(data)[:300]}")
        return budget

    def close(self) -> None:
        self._http.close()

    def _budget_spent(self, response: httpx.Response) -> bool:
        remaining = _header_number(response, "x-ratelimit-remaining")
        return remaining is not None and remaining < self.call_credits

    def _throttle(self) -> None:
        now = self._clock()
        if now < self._next_slot:
            self._sleep(self._next_slot - now)
            now = self._next_slot
        self._next_slot = now + self._interval


@dataclass(frozen=True)
class FetchStats:
    filter: str
    truncated: bool
    oldest_date: str | None
    newest_date: str | None
    n_fetched: int
    n_kept: int


def fetch_newest(
    client: OpenAlexClient, filters: str, cap: int, *, per_page: int = PER_PAGE
) -> tuple[list[dict[str, Any]], FetchStats]:
    """The `cap` newest works matching `filters`, cut deterministically.

    OpenAlex can't sort by id, so works sharing the publication date at the cut
    come back in no guaranteed order. Past the cap we keep reading while that
    date lasts, then order everything by (date desc, id asc) and cut locally.
    """
    fetched: list[dict[str, Any]] = []
    more = False
    if cap > 0:
        for work in client.iter_works(filters, sort=NEWEST_FIRST, per_page=per_page):
            if len(fetched) >= cap and work.get("publication_date") != _cut_date(fetched, cap):
                more = True
                break
            fetched.append(work)
    kept = sorted(fetched, key=_newest_first)[:cap]
    stats = FetchStats(
        filter=filters,
        truncated=more or len(fetched) > cap,
        oldest_date=kept[-1].get("publication_date") if kept else None,
        newest_date=kept[0].get("publication_date") if kept else None,
        n_fetched=len(fetched),
        n_kept=len(kept),
    )
    return kept, stats


def _cut_date(fetched: Sequence[dict[str, Any]], cap: int) -> str | None:
    return fetched[cap - 1].get("publication_date")


def _newest_first(work: dict[str, Any]) -> tuple[int, int]:
    published = date.fromisoformat(work.get("publication_date") or "0001-01-01")
    return (-published.toordinal(), id_number(short_id(work["id"])))


def sweep_filter(base: str, build_date: date) -> str:
    """The sweep filter as sent: `base` plus has_abstract:true and a to_publication_date cap."""
    clauses = [c for c in base.split(",") if c]
    keys = {c.split(":", 1)[0] for c in clauses}
    if "has_abstract" not in keys:
        clauses.append("has_abstract:true")
    if "to_publication_date" not in keys:
        clauses.append(f"to_publication_date:{build_date.isoformat()}")
    return ",".join(clauses)


def short_id(openalex_id: str) -> str:
    """'https://openalex.org/W123' -> 'W123'."""
    return openalex_id.rsplit("/", 1)[-1]


def normalize_doi(doi: str | None) -> str | None:
    """'https://doi.org/10.1/ABC' -> '10.1/abc' (DOIs are case-insensitive)."""
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        doi = doi.removeprefix(prefix)
    return doi or None


def to_record(work: dict[str, Any], origin: str) -> Record:
    """An OpenAlex work as a corpus record, with its abstract rebuilt."""
    return {
        "id": short_id(work["id"]),
        "aliases": [],
        "doi": normalize_doi(work.get("doi")),
        "title": work.get("title") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "publication_date": work.get("publication_date"),
        "origin": origin,
        "referenced_works": sorted(
            {short_id(w) for w in work.get("referenced_works") or []}, key=id_number
        ),
        "primary_topic": _compact_topic(work.get("primary_topic")),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "language": work.get("language"),
    }


def _compact_topic(topic: dict[str, Any] | None) -> dict[str, Any] | None:
    if not topic:
        return None
    subfield = topic.get("subfield") or {}
    return {
        "id": short_id(topic.get("id") or ""),
        "display_name": topic.get("display_name"),
        "subfield_id": short_id(subfield.get("id") or "") or None,
        "subfield": subfield.get("display_name"),
    }


def _accept(
    works: Iterable[dict[str, Any]], origin: str, corpus: CorpusConfig, dropped: Counter[str]
) -> list[Record]:
    """Records for the works that have an abstract and pass the language and length filters."""
    accepted = []
    for work in works:
        if not work.get("publication_date"):
            dropped["no_publication_date"] += 1
            continue
        record = to_record(work, origin)
        if not record["abstract"]:
            dropped["no_abstract"] += 1
        elif record["language"] != corpus.language:
            dropped["language"] += 1
        elif len(record["abstract"].split()) < corpus.min_abstract_words:
            dropped["short_abstract"] += 1
        else:
            accepted.append(record)
    return accepted


def dedupe(records: Iterable[Record]) -> list[Record]:
    """Merge versions of the same work.

    Two records merge if they share an id or alias, share a DOI, or have the
    same normalised title with at least 5 content words and publication years
    at most 2 apart (title merges chain through intermediate versions). The
    earliest version keeps its id; the other ids become aliases. A merged
    record keeps the earliest version's fields, except: references are the
    union over all versions, the DOI is the first one found, and the origin is
    the strongest (sweep > reference > citer).
    """
    recs = sorted(records, key=lambda r: (r["publication_date"], id_number(r["id"])))
    parent = list(range(len(recs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)  # the root is always the earliest version

    by_key: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        for work_id in (r["id"], *r["aliases"]):
            by_key.setdefault(f"id:{work_id}", []).append(i)
        if r["doi"]:
            by_key.setdefault(f"doi:{r['doi']}", []).append(i)
    for group in by_key.values():
        for j in group[1:]:
            union(group[0], j)

    by_title: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        if len(content_tokens(r["title"])) >= TITLE_MERGE_MIN_CONTENT_WORDS:
            by_title.setdefault(normalize_title(r["title"]), []).append(i)
    for group in by_title.values():  # each group is in date order
        for a, b in itertools.pairwise(group):
            if _year(recs[b]) - _year(recs[a]) <= TITLE_MERGE_MAX_YEAR_GAP:
                union(a, b)

    members: dict[int, list[Record]] = {}
    for i, r in enumerate(recs):
        members.setdefault(find(i), []).append(r)
    merged = [_merge(group) for group in members.values()]
    return sorted(merged, key=lambda r: id_number(r["id"]))


def _year(record: Record) -> int:
    return int(record["publication_date"][:4])


def _merge(versions: list[Record]) -> Record:
    canonical = dict(versions[0])
    ids = {work_id for r in versions for work_id in (r["id"], *r["aliases"])}
    canonical["aliases"] = sorted(ids - {canonical["id"]}, key=id_number)
    refs = {w for r in versions for w in r["referenced_works"]}
    canonical["referenced_works"] = sorted(refs, key=id_number)
    canonical["doi"] = next((r["doi"] for r in versions if r["doi"]), None)
    canonical["origin"] = min((r["origin"] for r in versions), key=ORIGINS.index)
    return canonical


def expand_citations(
    client: OpenAlexClient,
    corpus: Sequence[Record],
    targets: Sequence[Record],
    cfg: CorpusConfig,
    build_date: date,
    dropped: dict[str, Counter[str]],
    *,
    per_page: int = PER_PAGE,
) -> tuple[list[Record], list[FetchStats]]:
    """Fetch the targets' references (batched by id) and up to N newest citers each.

    References are fetched without has_abstract:true so that the ones without
    an abstract are counted (as dropped) rather than silently missing; ids that
    OpenAlex doesn't return are counted as not_found.
    """
    known = {work_id for r in corpus for work_id in (r["id"], *r["aliases"])}
    ref_ids = sorted({w for t in targets for w in t["referenced_works"]} - known, key=id_number)
    found: list[Record] = []
    for start in range(0, len(ref_ids), IDS_PER_CALL):
        batch = ref_ids[start : start + IDS_PER_CALL]
        page = client.get_works(
            {
                "filter": f"ids.openalex:{'|'.join(batch)}",
                "select": SELECT,
                "per_page": str(IDS_PER_CALL),
            }
        )
        results = page.get("results") or []
        dropped["reference"]["not_found"] += len(batch) - len(results)
        found += _accept(results, "reference", cfg, dropped["reference"])

    citer_stats = []
    for target in targets:
        cited = "|".join([target["id"], *target["aliases"]][:100])  # OR limit
        filters = f"cites:{cited},has_abstract:true,to_publication_date:{build_date.isoformat()}"
        works, stats = fetch_newest(client, filters, cfg.max_citers_per_target, per_page=per_page)
        citer_stats.append(stats)
        found += _accept(works, "citer", cfg, dropped["citer"])
    return found, citer_stats


def remap_targets(
    selected: Sequence[Record], corpus: Sequence[Record], target_from_date: date
) -> tuple[list[dict[str, Any]], int]:
    """Map sampled targets to their canonical records after the final dedupe.

    A target that merged with an earlier version takes that version's id and
    is checked against the target rules again (the earlier version may predate
    target_from_date). Returns the targets and how many were dropped.
    """
    canonical = {work_id: r for r in corpus for work_id in (r["id"], *r["aliases"])}
    targets: dict[str, dict[str, Any]] = {}
    dropped = 0
    for chosen in selected:
        record = canonical[chosen["id"]]
        if target_problems(record, target_from_date):
            dropped += 1
            continue
        targets.setdefault(
            record["id"],
            {
                "id": record["id"],
                "aliases": record["aliases"],
                "remapped_from": chosen["id"] if chosen["id"] != record["id"] else None,
            },
        )
    if not targets:
        raise TargetSelectionError(
            f"all {len(selected)} sampled targets failed the target rules after the final dedupe"
        )
    return sorted(targets.values(), key=lambda t: id_number(t["id"])), dropped


def build_corpus(
    cfg: Config,
    client: OpenAlexClient,
    *,
    build_date: date | None = None,
    per_page: int = PER_PAGE,
) -> dict[str, Any]:
    """Build and freeze the corpus and targets (spec: Corpus, steps 1-5); return the manifest."""
    out_dir = cfg.corpus.out_dir
    _refuse_overwrite(out_dir / MANIFEST_FILE)
    build_date = build_date or date.today()
    calls_before = client.calls
    dropped: dict[str, Counter[str]] = {origin: Counter() for origin in ORIGINS}

    sweeps: list[FetchStats] = []
    records: list[Record] = []
    for base in cfg.corpus.filters:
        works, stats = fetch_newest(
            client,
            sweep_filter(base, build_date),
            cfg.corpus.max_works_per_sweep,
            per_page=per_page,
        )
        sweeps.append(stats)
        records += _accept(works, "sweep", cfg.corpus, dropped["sweep"])
    corpus = dedupe(records)

    selection = select_targets(corpus, cfg.bench)
    expanded, citers = expand_citations(
        client, corpus, selection.targets, cfg.corpus, build_date, dropped, per_page=per_page
    )
    corpus = dedupe([*corpus, *expanded])
    targets, n_dropped = remap_targets(selection.targets, corpus, cfg.bench.target_from_date)

    origins = Counter(r["origin"] for r in corpus)
    manifest: dict[str, Any] = {
        "build_date": build_date.isoformat(),
        "openalex": {"authenticated": client.authenticated, "per_page": per_page},
        "config": {
            "filters": list(cfg.corpus.filters),
            "language": cfg.corpus.language,
            "min_abstract_words": cfg.corpus.min_abstract_words,
            "max_works_per_sweep": cfg.corpus.max_works_per_sweep,
            "max_citers_per_target": cfg.corpus.max_citers_per_target,
            "target_from_date": cfg.bench.target_from_date.isoformat(),
            "n_targets": cfg.bench.n_targets,
            "seed": cfg.bench.seed,
        },
        "sweeps": [dataclasses.asdict(s) for s in sweeps],
        "citers": {
            "queries": len(citers),
            "truncated": sum(s.truncated for s in citers),
            "n_kept": sum(s.n_kept for s in citers),
        },
        "api_calls": client.calls - calls_before,
        "n_docs": len(corpus),
        "n_aliases": sum(len(r["aliases"]) for r in corpus),
        "counts_by_origin": {origin: origins[origin] for origin in ORIGINS},
        "dropped": {origin: dict(sorted(dropped[origin].items())) for origin in ORIGINS},
        "targets": {
            "n_candidates": selection.n_candidates,
            "n_sampled": len(selection.targets),
            "n_remapped": sum(t["remapped_from"] is not None for t in targets),
            "n_dropped_after_final_dedupe": n_dropped,
            "n_final": len(targets),
        },
    }
    return freeze(out_dir, corpus, targets, manifest)


def freeze(
    out_dir: Path, corpus: Sequence[Record], targets: Sequence[dict[str, Any]], info: dict[str, Any]
) -> dict[str, Any]:
    """Write corpus.jsonl, targets.jsonl and, last, manifest.json; never overwrite a manifest."""
    manifest_path = out_dir / MANIFEST_FILE
    _refuse_overwrite(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in ((CORPUS_FILE, corpus), (TARGETS_FILE, targets)):
        data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        encoded = data.encode("utf-8")
        (out_dir / name).write_bytes(encoded)  # bytes, so line endings never vary by OS
        files[name] = {"sha256": hashlib.sha256(encoded).hexdigest(), "lines": len(rows)}
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {**info, "files": files, "created_at": created}
    try:
        with manifest_path.open("x", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
    except FileExistsError:  # appeared while we were building
        raise _overwrite_error(manifest_path) from None
    return manifest


def load_frozen_corpus(
    corpus_dir: Path,
) -> tuple[list[Record], list[dict[str, Any]], dict[str, Any]]:
    """The frozen corpus, its targets and manifest; the corpus must match its sha256."""
    manifest_path = corpus_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise CorpusError(
            f"no frozen corpus in {corpus_dir}: run `python -m rk build-corpus` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_bytes = (corpus_dir / CORPUS_FILE).read_bytes()
    if hashlib.sha256(corpus_bytes).hexdigest() != manifest["files"][CORPUS_FILE]["sha256"]:
        raise CorpusError(f"{corpus_dir / CORPUS_FILE} doesn't match the sha256 in its manifest")
    corpus = [json.loads(line) for line in corpus_bytes.decode("utf-8").splitlines()]
    targets_text = (corpus_dir / TARGETS_FILE).read_text(encoding="utf-8")
    targets = [json.loads(line) for line in targets_text.splitlines()]
    return corpus, targets, manifest


def _refuse_overwrite(manifest_path: Path) -> None:
    if manifest_path.exists():
        raise _overwrite_error(manifest_path)


def _overwrite_error(manifest_path: Path) -> CorpusError:
    return CorpusError(
        f"{manifest_path} already exists: refusing to overwrite a frozen corpus "
        "(choose another corpus.out_dir or delete it deliberately)"
    )
