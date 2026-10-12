from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import (
    FIXTURE_BUILD_DATE,
    FIXTURE_WORKS,
    SWEEP_FILTER,
    FakeOpenAlex,
    corpus_config,
    corpus_config_dict,
)

from rk.cli import main
from rk.config import config_from_dict
from rk.corpus import (
    CORPUS_FILE,
    MANIFEST_FILE,
    NEWEST_FIRST,
    TARGETS_FILE,
    CorpusError,
    OpenAlexClient,
    OpenAlexError,
    build_corpus,
    dedupe,
    fetch_newest,
    sweep_filter,
    to_record,
)
from rk.targets import TargetSelectionError

EFFECTIVE_SWEEP = f"{SWEEP_FILTER},has_abstract:true,to_publication_date:2026-01-15"
WORKS = {w["id"].rsplit("/", 1)[-1]: w for w in FIXTURE_WORKS}


def _records(*ids: str, origin: str = "sweep") -> list[dict[str, Any]]:
    return [to_record(WORKS[i], origin) for i in ids]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build(out_dir: Path, fake: FakeOpenAlex, **corpus_overrides: Any) -> dict[str, Any]:
    cfg = corpus_config(out_dir, **corpus_overrides)
    return build_corpus(cfg, fake.client(), build_date=FIXTURE_BUILD_DATE)


# --- client ------------------------------------------------------------------


def test_cursor_paging_follows_next_cursor_until_an_empty_page() -> None:
    fake = FakeOpenAlex()
    works = list(fake.client().iter_works(EFFECTIVE_SWEEP, sort=NEWEST_FIRST, per_page=4))

    assert len(works) == 23  # 25 sweep works minus no-abstract W1013 and future-dated W1014
    assert len({w["id"] for w in works}) == 23
    assert [r["cursor"] for r in fake.requests] == ["*"] + [f"offset:{4 * i}" for i in range(1, 7)]
    assert all(r["per_page"] == "4" and r["sort"] == NEWEST_FIRST for r in fake.requests)


def test_429_then_success_is_retried_with_backoff() -> None:
    fake = FakeOpenAlex(fail_with=[429])
    sleeps: list[float] = []
    client = fake.client(sleep=sleeps.append)

    page = client.get_works({"filter": "ids.openalex:W1001", "per_page": "5"})

    assert [w["id"] for w in page["results"]] == ["https://openalex.org/W1001"]
    assert sleeps == [1.0]
    assert client.calls == len(fake.requests) == 2


def test_other_client_errors_are_not_retried() -> None:
    fake = FakeOpenAlex()
    client = fake.client()
    with pytest.raises(OpenAlexError, match="HTTP 400"):
        client.get_works({"filter": "cites:W1001", "sort": "publication_date:desc,id:asc"})
    assert len(fake.requests) == 1


def test_requests_are_throttled() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = FakeOpenAlex().client(max_requests_per_second=10, sleep=sleep, clock=lambda: now[0])
    for _ in range(3):
        client.get_works({"filter": "ids.openalex:W1001"})
    assert sleeps == pytest.approx([0.1, 0.1])


def test_api_key_goes_in_a_header_and_never_in_the_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RK_TEST_OA_KEY", "secret-key")
    monkeypatch.setenv("RK_TEST_OA_MAILTO", "me@example.org")
    cfg = corpus_config(tmp_path, api_key_env="RK_TEST_OA_KEY", mailto_env="RK_TEST_OA_MAILTO")
    fake = FakeOpenAlex()
    client = OpenAlexClient.from_config(
        cfg.corpus, transport=fake.transport, max_requests_per_second=float("inf")
    )

    client.get_works({"filter": "ids.openalex:W1001"})

    assert client.authenticated
    assert fake.auth_headers == ["Bearer secret-key"]
    assert fake.requests[0]["mailto"] == "me@example.org"
    assert "secret-key" not in json.dumps(fake.requests)


def test_sweep_filter_adds_abstract_and_date_caps_once() -> None:
    assert sweep_filter(SWEEP_FILTER, FIXTURE_BUILD_DATE) == EFFECTIVE_SWEEP
    assert sweep_filter(EFFECTIVE_SWEEP, date(2027, 1, 1)) == EFFECTIVE_SWEEP


# --- truncation --------------------------------------------------------------


@pytest.mark.parametrize("tie_order", ["asc", "desc"])
def test_fetch_newest_cuts_ties_by_id_whatever_the_api_order(tie_order: str) -> None:
    # Three works share 2024-05-05, the newest date under this filter; keep two.
    client = FakeOpenAlex(tie_order=tie_order).client()
    filters = "primary_topic.subfield.id:1702,to_publication_date:2024-05-05"

    works, stats = fetch_newest(client, filters, 2, per_page=2)

    assert [w["id"].rsplit("/", 1)[-1] for w in works] == ["W1018", "W1019"]
    assert stats.truncated
    assert (stats.n_fetched, stats.n_kept) == (3, 2)  # read through the tied date
    assert stats.oldest_date == stats.newest_date == "2024-05-05"


def test_truncation_is_recorded_in_the_manifest(tmp_path: Path) -> None:
    manifest = _build(tmp_path, FakeOpenAlex(), max_works_per_sweep=5)

    assert manifest["sweeps"] == [
        {
            "filter": EFFECTIVE_SWEEP,
            "truncated": True,
            "oldest_date": "2025-10-01",
            "newest_date": "2026-01-10",
            "n_fetched": 5,
            "n_kept": 5,
        }
    ]
    # Newest first, so the newest target (W1003) survived the cut.
    assert [t["id"] for t in _read_jsonl(tmp_path / TARGETS_FILE)] == ["W1003"]


# --- dedupe ------------------------------------------------------------------


def test_dedupe_merges_versions_sharing_a_doi() -> None:
    (merged,) = dedupe(_records("W1007", "W1006"))
    assert merged["id"] == "W1006"  # earliest version keeps its id
    assert merged["aliases"] == ["W1007"]
    assert merged["doi"] == "10.1000/pair"  # "10.1000/PAIR" normalised
    assert merged["referenced_works"] == [f"W{n}" for n in range(2001, 2012)]  # union


def test_dedupe_merges_long_identical_titles_at_most_two_years_apart() -> None:
    records = dedupe(_records("W1010", "W1011") + _records("W2013", origin="reference"))
    assert [(r["id"], r["aliases"]) for r in records] == [("W1010", ["W1011"]), ("W2013", [])]


@pytest.mark.parametrize(("year", "merged"), [(2026, True), (2027, False)])
def test_title_rule_year_gap_boundary(year: int, merged: bool) -> None:
    late = {**_records("W1011")[0], "publication_date": f"{year}-03-01"}
    records = dedupe([*_records("W1010"), late])  # W1010 is from 2024
    assert (len(records) == 1) is merged


def test_dedupe_never_merges_short_titles() -> None:
    records = dedupe(_records("W1008", "W1009"))  # both just "Editorial"
    assert [(r["id"], r["aliases"]) for r in records] == [("W1008", []), ("W1009", [])]


def test_dedupe_collapses_the_same_work_found_twice() -> None:
    records = dedupe(_records("W1001") + _records("W1001", origin="citer"))
    assert [(r["id"], r["aliases"], r["origin"]) for r in records] == [("W1001", [], "sweep")]


# --- full build --------------------------------------------------------------


def test_build_freezes_corpus_targets_and_manifest(tmp_path: Path) -> None:
    fake = FakeOpenAlex()
    manifest = _build(tmp_path, fake)
    corpus = _read_jsonl(tmp_path / CORPUS_FILE)
    ids = {r["id"] for r in corpus} | {a for r in corpus for a in r["aliases"]}

    assert manifest["sweeps"][0] == {
        "filter": EFFECTIVE_SWEEP,
        "truncated": False,
        "oldest_date": "2023-03-14",
        "newest_date": "2026-01-10",
        "n_fetched": 23,
        "n_kept": 23,
    }
    assert manifest["n_docs"] == len(corpus) == 36
    assert manifest["counts_by_origin"] == {"sweep": 20, "reference": 13, "citer": 3}
    assert manifest["dropped"] == {"language": 1}
    assert {"W1012", "W1013", "W1014", "W2999"}.isdisjoint(ids)  # German, no abstract, future, missing
    assert manifest["citers"] == {"queries": 3, "truncated": 1, "n_kept": 3}
    assert {"W4001", "W4002", "W4004"} <= ids and "W4003" not in ids  # cap of 2 for W1001
    assert manifest["api_calls"] == len(fake.requests)
    for name in (CORPUS_FILE, TARGETS_FILE):
        digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert manifest["files"][name]["sha256"] == digest
    on_disk = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_build_applies_both_dedupe_rules_but_not_to_editorials(tmp_path: Path) -> None:
    _build(tmp_path, FakeOpenAlex())
    by_id = {r["id"]: r for r in _read_jsonl(tmp_path / CORPUS_FILE)}

    assert by_id["W1006"]["aliases"] == ["W1007"]  # DOI rule
    assert by_id["W1010"]["aliases"] == ["W1011"]  # title rule
    assert by_id["W2013"]["aliases"] == []  # same title, 4 years earlier
    assert by_id["W1008"]["aliases"] == by_id["W1009"]["aliases"] == []  # two "Editorial"s


def test_target_is_remapped_to_an_earlier_version_found_by_expansion(tmp_path: Path) -> None:
    manifest = _build(tmp_path, FakeOpenAlex())
    by_id = {r["id"]: r for r in _read_jsonl(tmp_path / CORPUS_FILE)}

    # W1002 was sampled; its earlier version W3001 (same DOI) only arrived as a
    # reference of W1001, so the final dedupe made W3001 canonical.
    assert _read_jsonl(tmp_path / TARGETS_FILE) == [
        {"id": "W1001", "aliases": [], "remapped_from": None},
        {"id": "W1003", "aliases": [], "remapped_from": None},
        {"id": "W3001", "aliases": ["W1002"], "remapped_from": "W1002"},
    ]
    assert by_id["W3001"]["origin"] == "sweep"
    assert manifest["targets"] == {
        "n_candidates": 3,
        "n_sampled": 3,
        "n_remapped": 1,
        "n_dropped_after_final_dedupe": 0,
        "n_final": 3,
    }


def test_refuses_to_overwrite_an_existing_manifest(tmp_path: Path) -> None:
    _build(tmp_path, FakeOpenAlex())
    before = (tmp_path / CORPUS_FILE).read_bytes()
    second = FakeOpenAlex()

    with pytest.raises(CorpusError, match="refusing to overwrite"):
        _build(tmp_path, second)

    assert second.requests == []  # failed before any API call
    assert (tmp_path / CORPUS_FILE).read_bytes() == before


def test_output_is_deterministic_across_runs(tmp_path: Path) -> None:
    # The cap of 17 cuts through three works sharing a date, and the two fake
    # servers return those ties in opposite orders.
    manifests = [
        build_corpus(
            corpus_config(tmp_path / order, max_works_per_sweep=17),
            FakeOpenAlex(tie_order=order).client(),
            build_date=FIXTURE_BUILD_DATE,
            per_page=5,
        )
        for order in ("asc", "desc")
    ]

    assert manifests[0]["sweeps"][0]["truncated"]
    for name in (CORPUS_FILE, TARGETS_FILE):
        assert (tmp_path / "asc" / name).read_bytes() == (tmp_path / "desc" / name).read_bytes()
    first, second = ({k: v for k, v in m.items() if k != "created_at"} for m in manifests)
    assert first == second


def test_fails_loudly_without_target_candidates(tmp_path: Path) -> None:
    data = corpus_config_dict(tmp_path)
    data["bench"]["target_from_date"] = date(2026, 1, 1)  # after every fixture target
    with pytest.raises(TargetSelectionError, match="no target candidates"):
        build_corpus(config_from_dict(data), FakeOpenAlex().client(), build_date=FIXTURE_BUILD_DATE)
    assert not (tmp_path / MANIFEST_FILE).exists()


# --- CLI ---------------------------------------------------------------------


def test_cli_build_corpus_with_max_works_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "corpus"
    config_path.write_text(yaml.safe_dump(corpus_config_dict(out_dir)), encoding="utf-8")
    fake = FakeOpenAlex()
    monkeypatch.setattr(
        OpenAlexClient, "from_config", classmethod(lambda cls, corpus, **kw: fake.client())
    )

    assert main(["build-corpus", "--config", str(config_path), "--max-works", "5"]) == 0
    assert "froze" in capsys.readouterr().out
    manifest = json.loads((out_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["config"]["max_works_per_sweep"] == 5
    assert manifest["sweeps"][0]["n_kept"] == 5

    assert main(["build-corpus", "--config", str(config_path)]) == 1  # frozen already
    assert "refusing to overwrite" in capsys.readouterr().err
