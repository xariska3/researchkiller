"""Load configs/phase1.yaml into frozen dataclasses.

Loading is strict: unknown, duplicate or missing keys and wrong types raise
ConfigError, naming the offending key. Constructing a Config always runs the
model-family independence check.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import re
import types
import typing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

ENCODER_DTYPES = ("float32", "float16", "bfloat16")  # torch dtype names


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
    max_works_per_sweep: int = 25000  # per filter string, newest first; deterministic cut
    mailto_env: str | None = None  # env var with an email; OpenAlex ignores mailto since 2026-02
    api_key_env: str | None = None  # env var holding an OpenAlex API key (optional, 10x budget)

    def __post_init__(self) -> None:
        _require(len(self.filters) > 0, "filters must list at least one OpenAlex filter")
        _require(self.min_abstract_words >= 0, "min_abstract_words must be >= 0")
        _require(self.max_citers_per_target >= 0, "max_citers_per_target must be >= 0")
        _require(self.max_works_per_sweep >= 1, "max_works_per_sweep must be >= 1")


@dataclass(frozen=True)
class BenchConfig:
    n_targets: int
    target_from_date: date
    max_attempts_per_target: int
    max_distinctive_overlap: float  # fraction of the abstract's top-20 IDF terms (0.15 = 3)
    max_content_jaccard: float
    seed: int
    out_dir: Path

    def __post_init__(self) -> None:
        _require(self.n_targets >= 1, "n_targets must be >= 1")
        _require(self.max_attempts_per_target >= 1, "max_attempts_per_target must be >= 1")
        _require(
            0.0 <= self.max_distinctive_overlap <= 1.0,
            "max_distinctive_overlap is a fraction of the top-20 IDF terms, so must be in [0, 1]",
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
    encoder_revision: str  # Hugging Face commit hash of the encoder, for reproducibility
    encoder: str = "BAAI/bge-small-en-v1.5"
    encoder_dtype: str = "float32"  # weights and activations; one of ENCODER_DTYPES
    encoder_max_seq_length: int | None = None  # longer inputs are truncated; None: model default

    def __post_init__(self) -> None:
        _require(
            re.fullmatch(r"[0-9a-f]{40}", self.encoder_revision) is not None,
            "encoder_revision must be a 40-character Hugging Face commit hash",
        )
        _require(
            self.encoder_dtype in ENCODER_DTYPES,
            f"encoder_dtype must be one of {list(ENCODER_DTYPES)}, got {self.encoder_dtype!r}",
        )
        _require(
            self.encoder_max_seq_length is None or self.encoder_max_seq_length >= 1,
            "encoder_max_seq_length must be >= 1 (or null for the model's default)",
        )
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
    cache_path: Path  # SQLite database: LLM cache and the cumulative spend table
    seed: int  # bootstrap resampling
    budget_cap: float = 20.0  # USD, hard cap on real LLM spend across all commands
    results_db: Path = Path("data/results.sqlite")  # SQL results of every evaluation run
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"  # the documented local default backend
    mlflow_experiment: str = "researchkiller-phase1"

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
    training_cutoff: date  # targets must be published after this
    # OpenRouter endpoint slug to pin (e.g. "openai", "novita/fp8"), sent with fallbacks
    # disabled; None lets OpenRouter route.
    provider: str | None = None

    def __post_init__(self) -> None:
        _require(bool(self.name.strip()), "name must not be empty")
        _require(bool(self.family.strip()), "family must not be empty")
        _require(
            self.provider is None or bool(self.provider.strip()),
            "provider must not be empty (leave it out to let OpenRouter route)",
        )
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

    def models(self) -> dict[str, ModelConfig]:
        return {"paraphraser": self.paraphraser, "judge": self.judge, "agent": self.agent}

    def __post_init__(self) -> None:
        check_model_independence(self.models())
        check_training_cutoffs(self.bench.target_from_date, self.models())


def check_model_independence(models: dict[str, ModelConfig]) -> None:
    """Fail loudly unless every model role comes from a different model family.

    A same-family agent can "undo" the paraphraser's word choices, a same-family
    judge rubber-stamps them, and a judge sharing the agent's family selects
    paraphrases the agent finds natural. Families compare case-insensitively. An
    identical model name also counts as a clash, so a mistyped family label
    can't hide one.
    """

    def same(a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()

    clashes = [
        f"{role_a} ({a.name}, family {a.family!r}) and {role_b} ({b.name}, family {b.family!r})"
        for (role_a, a), (role_b, b) in itertools.combinations(models.items(), 2)
        if same(a.family, b.family) or same(a.name, b.name)
    ]
    if clashes:
        raise ConfigError(
            "model-independence check failed: paraphraser, judge and agent must come from "
            f"three different model families, but these share one: {'; '.join(clashes)}"
        )


def check_training_cutoffs(target_from_date: date, models: dict[str, ModelConfig]) -> None:
    """Fail loudly unless targets start strictly after every model's training cutoff."""
    stale = [
        f"{role} ({model.name}, cutoff {model.training_cutoff})"
        for role, model in models.items()
        if target_from_date <= model.training_cutoff
    ]
    if stale:
        raise ConfigError(
            f"bench.target_from_date ({target_from_date}) must be later than every model's "
            f"training_cutoff, but is not later than: {'; '.join(stale)}"
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
