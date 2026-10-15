from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEncoder, FakeLLM
from dotenv import dotenv_values

from rk.cli import load_env_file, main
from rk.config import load_config


def test_fake_llm_reports_usage_and_records_prompts() -> None:
    llm = FakeLLM(["first reply", "second"])
    prompt = [{"role": "user", "content": "three word prompt"}]
    out = llm.complete(prompt)
    assert out.text == "first reply"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens) == (3, 2)
    assert out.usage.method_cost == out.usage.money_spent > 0
    assert llm.calls == [prompt]
    llm.complete(prompt)
    with pytest.raises(AssertionError, match="exhausted"):
        llm.complete(prompt)


def test_fake_encoder_is_deterministic_normalised_and_lexically_sensible() -> None:
    texts = [
        "contrastive pretraining of image encoders",
        "contrastive pretraining for image encoding",
        "gradient boosting on tabular records",
        "",
    ]
    vectors = FakeEncoder(dim=64).encode_docs(texts)
    assert vectors.shape == (4, 64) and vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors, FakeEncoder(dim=64).encode_queries(texts))
    np.testing.assert_allclose(np.linalg.norm(vectors[:3], axis=1), 1.0, rtol=1e-6)
    assert not vectors[3].any()
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "--config", "c.yaml", "--configs", "B1,B9"], "unknown configs ['B9']"),
        (["run", "--config", "c.yaml", "--retry-errors"], "--retry-errors needs --run-dir"),
        (["run", "--config", "c.yaml", "--limit", "0"], "must be >= 1"),
    ],
)
def test_cli_rejects_bad_run_arguments(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_python_dash_m_rk_loads_dotenv_then_runs_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # .env is read from the working directory, never the repo's here
    (tmp_path / ".env").write_text("RK_TEST_FROM_FILE=loaded\n", encoding="utf-8")
    monkeypatch.delenv("RK_TEST_FROM_FILE", raising=False)  # unset again after the test
    monkeypatch.setattr(sys, "argv", ["rk", "report", str(tmp_path / "no-such-run")])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("rk", run_name="__main__")
    assert exit_info.value.code == 1
    assert "is not a run directory" in capsys.readouterr().err
    assert os.environ["RK_TEST_FROM_FILE"] == "loaded"


def test_env_file_never_overrides_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("RK_TEST_FROM_FILE=file\nRK_TEST_ALREADY_SET=file\n", encoding="utf-8")
    monkeypatch.delenv("RK_TEST_FROM_FILE", raising=False)
    monkeypatch.setenv("RK_TEST_ALREADY_SET", "environment")

    assert load_env_file(env)
    assert os.environ["RK_TEST_FROM_FILE"] == "file"
    assert os.environ["RK_TEST_ALREADY_SET"] == "environment"
    assert not load_env_file(tmp_path / "missing.env")


def test_env_example_lists_every_variable_the_config_reads_with_empty_values() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "phase1.yaml")
    read = {m.api_key_env for m in cfg.models().values()}
    read |= {cfg.corpus.api_key_env, cfg.corpus.mailto_env}
    assert cfg.compete is not None
    read |= {s.api_key_env for s in cfg.compete.systems} | {cfg.compete.crossref_mailto_env}
    example = dotenv_values(root / ".env.example")
    assert read == set(example)
    assert not any(example.values())
