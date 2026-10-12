from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEncoder, FakeLLM

from rk.cli import main


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


def test_python_dash_m_rk_runs_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["rk", "report", str(tmp_path / "no-such-run")])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("rk", run_name="__main__")
    assert exit_info.value.code == 1
    assert "is not a run directory" in capsys.readouterr().err
