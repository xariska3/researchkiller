from __future__ import annotations

import runpy
import sys

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
    vectors = FakeEncoder(dim=64).encode(texts)
    assert vectors.shape == (4, 64) and vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors, FakeEncoder(dim=64).encode(texts))
    np.testing.assert_allclose(np.linalg.norm(vectors[:3], axis=1), 1.0, rtol=1e-6)
    assert not vectors[3].any()
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


@pytest.mark.parametrize(
    "argv",
    [
        ["build-index", "--config", "configs/phase1.yaml"],
        ["make-bench", "--config", "configs/phase1.yaml"],
        ["run", "--config", "configs/phase1.yaml", "--configs", "B1,B4,B5", "--limit", "5"],
        ["report", "runs/20260101-000000"],
    ],
)
def test_cli_subcommands_are_stubs(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 1
    assert capsys.readouterr().out.strip() == f"rk {argv[0]}: not implemented"


def test_python_dash_m_rk_runs_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["rk", "report", "runs/20260101-000000"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("rk", run_name="__main__")
    assert exit_info.value.code == 1
    assert capsys.readouterr().out.strip() == "rk report: not implemented"
