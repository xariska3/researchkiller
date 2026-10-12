from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from rk.config import Config, ConfigError, config_from_dict, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase1.yaml"


def _model(name: str, family: str) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "base_url": "http://llm.test/v1",
        "api_key_env": "RK_TEST_API_KEY",
        "price_in_per_m": 1.0,
        "price_out_per_m": 2,
        "temperature": 0,
        "max_tokens": 64,
        "seed": 7,
    }


def _valid() -> dict[str, Any]:
    return {
        "corpus": {
            "filters": ["primary_topic.subfield.id:1702"],
            "language": "en",
            "min_abstract_words": 50,
            "max_citers_per_target": 20,
            "out_dir": "data/corpus",
        },
        "bench": {
            "n_targets": 10,
            "target_from_date": date(2025, 6, 1),
            "max_attempts_per_target": 3,
            "max_distinctive_overlap": 3,
            "max_content_jaccard": 0.15,
            "seed": 1,
            "out_dir": "data/bench",
        },
        "retrieval": {
            "query_prefix": "query: ",
            "max_queries": 5,
            "max_llm_calls_agent": 6,
            "max_queries_unconstrained": 15,
            "max_llm_calls_agent_unconstrained": 20,
            "index_dir": "data/index",
        },
        "run": {"out_dir": "runs", "cache_path": "data/cache/llm.sqlite", "seed": 1},
        "paraphraser": _model("para-model", "family-a"),
        "judge": _model("judge-model", "family-b"),
        "agent": _model("agent-model", "family-c"),
    }


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_repo_config_loads() -> None:
    cfg = load_config(REPO_CONFIG)
    assert isinstance(cfg, Config)
    assert cfg.run.budget_cap == 20.0
    assert cfg.retrieval.encoder == "BAAI/bge-small-en-v1.5"
    assert isinstance(cfg.bench.target_from_date, date)
    assert isinstance(cfg.corpus.filters, tuple) and cfg.corpus.filters


def test_load_coerces_types_and_applies_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _valid()))
    assert cfg.paraphraser.price_out_per_m == 2.0
    assert isinstance(cfg.paraphraser.temperature, float)
    assert cfg.bench.target_from_date == date(2025, 6, 1)
    assert cfg.run.out_dir == Path("runs")
    assert cfg.run.budget_cap == 20.0  # spec default when omitted
    assert cfg.run.mlflow is False
    assert cfg.retrieval.encoder == "BAAI/bge-small-en-v1.5"
    assert cfg.corpus.mailto_env is None


def test_date_may_be_a_quoted_string() -> None:
    data = _valid()
    data["bench"]["target_from_date"] = "2025-06-01"
    assert config_from_dict(data).bench.target_from_date == date(2025, 6, 1)


@pytest.mark.parametrize(
    "section", [None, "corpus", "bench", "retrieval", "run", "paraphraser", "judge", "agent"]
)
def test_unknown_keys_raise(section: str | None) -> None:
    data = _valid()
    (data if section is None else data[section])["surprise"] = 1
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['surprise'\]"):
        config_from_dict(data)


def test_missing_key_raises() -> None:
    data = _valid()
    del data["bench"]["seed"]
    with pytest.raises(ConfigError, match=r"bench\.seed: missing required key"):
        config_from_dict(data)


def test_missing_section_raises() -> None:
    data = _valid()
    del data["judge"]
    with pytest.raises(ConfigError, match="judge: missing required key"):
        config_from_dict(data)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("bench", "n_targets", "many"),
        ("bench", "n_targets", True),  # bools are not ints
        ("bench", "n_targets", 2.5),
        ("bench", "target_from_date", "June 2025"),
        ("run", "mlflow", "yes"),
        ("corpus", "filters", "primary_topic.subfield.id:1702"),
        ("corpus", "filters", [1702]),
        ("paraphraser", "temperature", None),
    ],
)
def test_wrong_types_raise(section: str, key: str, value: Any) -> None:
    data = _valid()
    data[section][key] = value
    with pytest.raises(ConfigError, match=rf"{section}\.{key}"):
        config_from_dict(data)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("bench", "max_content_jaccard", 1.5),
        ("bench", "max_distinctive_overlap", 21),
        ("run", "budget_cap", 0),
        ("retrieval", "max_queries_unconstrained", 4),
        ("judge", "max_tokens", 0),
    ],
)
def test_out_of_range_values_raise(section: str, key: str, value: Any) -> None:
    data = _valid()
    data[section][key] = value
    with pytest.raises(ConfigError, match=section):
        config_from_dict(data)


def test_duplicate_keys_raise(tmp_path: Path) -> None:
    text = REPO_CONFIG.read_text(encoding="utf-8").replace(
        "  budget_cap: 20.0", "  budget_cap: 20.0\n  budget_cap: 500.0"
    )
    path = tmp_path / "dup.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key 'budget_cap'"):
        load_config(path)


@pytest.mark.parametrize("role", ["judge", "agent"])
def test_independence_check_rejects_shared_family(role: str) -> None:
    data = _valid()
    data[role]["family"] = " FAMILY-A "  # same family as the paraphraser, modulo case/space
    with pytest.raises(ConfigError, match=rf"model-independence check failed.*{role}"):
        config_from_dict(data)


def test_independence_check_reports_both_clashes() -> None:
    data = _valid()
    data["judge"]["family"] = data["agent"]["family"] = "family-a"
    with pytest.raises(ConfigError, match="judge and agent"):
        config_from_dict(data)


def test_independence_check_catches_same_model_under_another_family_label() -> None:
    data = _valid()
    data["agent"]["name"] = data["paraphraser"]["name"]
    with pytest.raises(ConfigError, match="model-independence check failed"):
        config_from_dict(data)


def test_judge_and_agent_may_share_a_family() -> None:
    data = _valid()
    data["judge"]["family"] = data["agent"]["family"]
    cfg = config_from_dict(data)
    assert cfg.judge.family == cfg.agent.family
