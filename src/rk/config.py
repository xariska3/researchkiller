"""Load configs/phase1.yaml into frozen dataclasses.

Loading is strict: unknown, duplicate or missing keys and wrong types raise
ConfigError, naming the offending key. Constructing a Config always runs the
model-family independence check.
"""

from __future__ import annotations

import contextlib
import dataclasses
import types
import typing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """The configuration is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class CorpusConfig:
    filters: tuple[str, ...]  # OpenAlex /works filter strings, one sweep each
    language: str
    min_abstract_words: int
    max_citers_per_target: int
    out_dir: Path
    mailto_env: str | None = None  # env var holding the OpenAlex polite-pool email
    api_key_env: str | None = None  # env var holding an OpenAlex API key

    def __post_init__(self) -> None:
        _require(len(self.filters) > 0, "filters must list at least one OpenAlex filter")
        _require(self.min_abstract_words >= 0, "min_abstract_words must be >= 0")
        _require(self.max_citers_per_target >= 0, "max_citers_per_target must be >= 0")


@dataclass(frozen=True)
class BenchConfig:
    n_targets: int
    target_from_date: date
    max_attempts_per_target: int
    max_distinctive_overlap: int  # count of the abstract's top-20 IDF terms
    max_content_jaccard: float
    seed: int
    out_dir: Path

    def __post_init__(self) -> None:
        _require(self.n_targets >= 1, "n_targets must be >= 1")
        _require(self.max_attempts_per_target >= 1, "max_attempts_per_target must be >= 1")
        _require(
            0 <= self.max_distinctive_overlap <= 20,
            "max_distinctive_overlap counts top-20 IDF terms, so must be in [0, 20]",
        )
        _require(0.0 <= self.max_content_jaccard <= 1.0, "max_content_jaccard must be in [0, 1]")


@dataclass(frozen=True)
class RetrievalConfig:
    query_prefix: str
    max_queries: int  # B4 and B5
    max_llm_calls_agent: int  # B5
    max_queries_unconstrained: int  # B5-U
    max_llm_calls_agent_unconstrained: int  # B5-U
    index_dir: Path
    encoder: str = "BAAI/bge-small-en-v1.5"

    def __post_init__(self) -> None:
        _require(self.max_queries >= 1, "max_queries must be >= 1")
        _require(self.max_llm_calls_agent >= 1, "max_llm_calls_agent must be >= 1")
        _require(
            self.max_queries_unconstrained >= self.max_queries,
            "max_queries_unconstrained must be >= max_queries",
        )
        _require(
            self.max_llm_calls_agent_unconstrained >= self.max_llm_calls_agent,
            "max_llm_calls_agent_unconstrained must be >= max_llm_calls_agent",
        )


@dataclass(frozen=True)
class RunConfig:
    out_dir: Path
    cache_path: Path  # SQLite LLM cache shared by every model role
    seed: int
    budget_cap: float = 20.0  # USD, hard cap on real LLM spend
    mlflow: bool = False

    def __post_init__(self) -> None:
        _require(self.budget_cap > 0, "budget_cap must be > 0")


@dataclass(frozen=True)
class ModelConfig:
    """One model role behind an OpenAI-compatible /chat/completions endpoint."""

    name: str
    family: str
    base_url: str
    api_key_env: str
    price_in_per_m: float  # USD per million prompt tokens
    price_out_per_m: float  # USD per million completion tokens
    temperature: float
    max_tokens: int
    seed: int

    def __post_init__(self) -> None:
        _require(bool(self.name.strip()), "name must not be empty")
        _require(bool(self.family.strip()), "family must not be empty")
        _require(self.price_in_per_m >= 0 and self.price_out_per_m >= 0, "prices must be >= 0")
        _require(self.temperature >= 0, "temperature must be >= 0")
        _require(self.max_tokens >= 1, "max_tokens must be >= 1")


@dataclass(frozen=True)
class Config:
    corpus: CorpusConfig
    bench: BenchConfig
    retrieval: RetrievalConfig
    run: RunConfig
    paraphraser: ModelConfig
    judge: ModelConfig
    agent: ModelConfig

    def __post_init__(self) -> None:
        check_model_independence(self.paraphraser, self.judge, self.agent)


def check_model_independence(
    paraphraser: ModelConfig, judge: ModelConfig, agent: ModelConfig
) -> None:
    """Fail loudly unless the paraphraser's family differs from the judge's and the agent's.

    A same-family agent can "undo" its sibling's word choices, and a same-family
    judge rubber-stamps them. Families compare case-insensitively. An identical
    model name also counts as a clash, so a mistyped family label can't hide one.
    """

    def same(a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()

    clashes = [
        role
        for role, other in (("judge", judge), ("agent", agent))
        if same(other.family, paraphraser.family) or same(other.name, paraphraser.name)
    ]
    if clashes:
        raise ConfigError(
            f"model-independence check failed: the paraphraser ({paraphraser.name}, "
            f"family {paraphraser.family!r}) shares a model family with the "
            f"{' and '.join(clashes)}; the paraphraser must come from a different family"
        )


def load_config(path: str | Path) -> Config:
    """Read a YAML config file into a validated Config."""
    text = Path(path).read_text(encoding="utf-8")
    return config_from_dict(yaml.load(text, Loader=_UniqueKeyLoader))


def config_from_dict(data: Any) -> Config:
    """Validate an already-parsed mapping into a Config."""
    return _build(Config, data, "")


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise ConfigError(f"duplicate key {key!r} on line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _build(cls: type[T], data: Any, where: str) -> T:
    label = where or "config"
    if not isinstance(data, dict):
        raise ConfigError(f"{label}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(str(k) for k in data if k not in fields)
    if unknown:
        raise ConfigError(f"{label}: unknown key(s) {unknown}; allowed: {sorted(fields)}")
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, field in fields.items():
        key = f"{where}.{name}" if where else name
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], key)
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{key}: missing required key")
    try:
        return cls(**kwargs)
    except ConfigError as exc:  # value checks in __post_init__
        raise ConfigError(f"{label}: {exc}") from None


def _coerce(tp: Any, value: Any, key: str) -> Any:
    """Check `value` against the field type `tp`, converting where YAML can't express it."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:  # only Optional[X] is used
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        (inner,) = (a for a in args if a is not type(None))
        return _coerce(inner, value, key)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, key)
    if origin is tuple:  # tuple[X, ...] from a YAML list
        if isinstance(value, list):
            item_tp = typing.get_args(tp)[0]
            return tuple(_coerce(item_tp, v, f"{key}[{i}]") for i, v in enumerate(value))
        expected = "a list"
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        expected = "a number"
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        expected = "an integer"
    elif tp is bool:
        if isinstance(value, bool):
            return value
        expected = "true or false"
    elif tp is str:
        if isinstance(value, str):
            return value
        expected = "a string"
    elif tp is Path:
        if isinstance(value, str) and value:
            return Path(value)
        expected = "a non-empty path string"
    elif tp is date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                return date.fromisoformat(value)
        expected = "a date (YYYY-MM-DD)"
    else:
        raise TypeError(f"{key}: unsupported config field type {tp!r}")
    raise ConfigError(f"{key}: expected {expected}, got {value!r}")
