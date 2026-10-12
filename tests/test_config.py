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
        "training_cutoff": date(2024, 1, 1),
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
            "max_distinctive_overlap": 0.15,
            "max_content_jaccard": 0.15,
            "seed": 1,
            "out_dir": "data/bench",
        },
        "retrieval": {
            "query_prefix": "query: ",
            "encoder_revision": "0123456789abcdef0123456789abcdef01234567",
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
    assert cfg.run.budget_cap == 12.0
    assert (cfg.bench.n_targets, cfg.bench.target_from_date) == (500, date(2026, 1, 1))
    assert cfg.corpus.filters and all("language:en" in f for f in cfg.corpus.filters)
    assert len({m.family for m in cfg.models().values()}) == 3
    assert all(isinstance(m.training_cutoff, date) for m in cfg.models().values())
    assert all(m.provider for m in cfg.models().values())  # every role is pinned


def test_repo_config_uses_qwen3_embedding_as_its_model_card_says() -> None:
    r = load_config(REPO_CONFIG).retrieval
    assert (r.encoder, r.encoder_dtype, r.encoder_max_seq_length) == (
        "Qwen/Qwen3-Embedding-0.6B",
        "float16",
        512,
    )
    # "Instruct: {task_description}\nQuery:{query}" with a one-sentence task.
    task = r.query_prefix.removeprefix("Instruct: ").removesuffix("\nQuery:")
    assert r.query_prefix == f"Instruct: {task}\nQuery:"
    assert task and "\n" not in task and not task.endswith(".")


def test_load_coerces_types_and_applies_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _valid()))
    assert cfg.paraphraser.price_out_per_m == 2.0
    assert isinstance(cfg.paraphraser.temperature, float)
    assert cfg.bench.target_from_date == date(2025, 6, 1)
    assert cfg.agent.training_cutoff == date(2024, 1, 1)
    assert cfg.run.out_dir == Path("runs")
    assert cfg.run.budget_cap == 20.0  # spec default when omitted
    assert cfg.run.results_db == Path("data/results.sqlite")
    assert cfg.run.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert cfg.retrieval.encoder == "BAAI/bge-small-en-v1.5"
    assert (cfg.retrieval.encoder_dtype, cfg.retrieval.encoder_max_seq_length) == ("float32", None)
    assert cfg.corpus.max_works_per_sweep == 25000
    assert cfg.corpus.mailto_env is None
    assert cfg.agent.provider is None  # OpenRouter routes freely unless pinned


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
        ("run", "mlflow_experiment", 5),
        ("corpus", "filters", "primary_topic.subfield.id:1702"),
        ("corpus", "filters", [1702]),
        ("paraphraser", "temperature", None),
        ("judge", "training_cutoff", "late 2024"),
        ("retrieval", "encoder_max_seq_length", "512"),
        ("agent", "provider", ["openai"]),
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
        ("bench", "max_distinctive_overlap", 3),  # a count, not a fraction
        ("bench", "max_distinctive_overlap", -0.1),
        ("corpus", "max_works_per_sweep", 0),
        ("run", "budget_cap", 0),
        ("retrieval", "max_queries_unconstrained", 4),
        ("judge", "max_tokens", 0),
        ("retrieval", "encoder_revision", "main"),
        ("retrieval", "encoder_revision", "5C38EC7C405EC4B44B94CC5A9BB96E735B38267A"),
        ("retrieval", "encoder_dtype", "fp16"),  # torch names only
        ("retrieval", "encoder_dtype", "int8"),
        ("retrieval", "encoder_max_seq_length", 0),
        ("agent", "provider", " "),
    ],
)
def test_out_of_range_values_raise(section: str, key: str, value: Any) -> None:
    data = _valid()
    data[section][key] = value
    with pytest.raises(ConfigError, match=section):
        config_from_dict(data)


def test_duplicate_keys_raise(tmp_path: Path) -> None:
    text = REPO_CONFIG.read_text(encoding="utf-8").replace(
        "  budget_cap: 12.0", "  budget_cap: 12.0\n  budget_cap: 500.0"
    )
    path = tmp_path / "dup.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key 'budget_cap'"):
        load_config(path)


@pytest.mark.parametrize(
    ("role_a", "role_b"), [("paraphraser", "judge"), ("paraphraser", "agent"), ("judge", "agent")]
)
def test_independence_check_rejects_any_shared_family(role_a: str, role_b: str) -> None:
    data = _valid()
    data[role_b]["family"] = f" {data[role_a]['family'].upper()} "  # modulo case and spaces
    with pytest.raises(
        ConfigError, match=rf"model-independence check failed.*{role_a} \(.*\) and {role_b} \("
    ):
        config_from_dict(data)


def test_independence_check_reports_every_clash() -> None:
    data = _valid()
    data["judge"]["family"] = data["agent"]["family"] = "family-a"
    with pytest.raises(ConfigError) as exc_info:
        config_from_dict(data)
    message = str(exc_info.value)
    for pair in ("paraphraser (", "judge (", "agent ("):
        assert pair in message
    assert message.count(") and ") == 3


def test_independence_check_catches_same_model_under_another_family_label() -> None:
    data = _valid()
    data["agent"]["name"] = data["judge"]["name"]
    with pytest.raises(ConfigError, match="model-independence check failed"):
        config_from_dict(data)


@pytest.mark.parametrize("role", ["paraphraser", "judge", "agent"])
@pytest.mark.parametrize("cutoff", [date(2025, 6, 1), date(2025, 9, 1)])  # equal, later
def test_training_cutoff_check_rejects_targets_not_after_a_cutoff(role: str, cutoff: date) -> None:
    data = _valid()  # target_from_date is 2025-06-01
    data[role]["training_cutoff"] = cutoff
    with pytest.raises(ConfigError, match=rf"target_from_date .*not later than: {role} "):
        config_from_dict(data)


def test_training_cutoff_check_accepts_targets_after_every_cutoff() -> None:
    data = _valid()
    for role in ("paraphraser", "judge", "agent"):
        data[role]["training_cutoff"] = date(2025, 5, 31)
    assert config_from_dict(data).bench.target_from_date == date(2025, 6, 1)
