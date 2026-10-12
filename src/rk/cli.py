"""The `rk` command. Step 1: argument parsing only; every subcommand is a stub."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rk", description="ResearchKiller Phase 1: terminology-shift retrieval benchmark."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    for name, help_text in (
        ("build-corpus", "build and freeze the OpenAlex corpus and targets"),
        ("build-index", "build the BM25 and dense indexes for the frozen corpus"),
        ("make-bench", "generate and filter the Test A paraphrase benchmark"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--config", type=Path, required=True)

    run = sub.add_parser("run", help="evaluate retrieval configurations on the benchmark")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--configs", help="comma-separated configs to run, e.g. B1,B4,B5")
    run.add_argument("--limit", type=int, help="evaluate only the first N queries")

    report = sub.add_parser("report", help="summarise a finished run")
    report.add_argument("run_dir", type=Path, help="runs/<timestamp>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"rk {args.command}: not implemented")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
