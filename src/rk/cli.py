"""The `rk` command (run as `python -m rk`). Only build-corpus is implemented so far."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from rk.config import ConfigError, load_config
from rk.corpus import CorpusError, OpenAlexClient, OpenAlexError, build_corpus
from rk.targets import TargetSelectionError


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rk", description="ResearchKiller Phase 1: terminology-shift retrieval benchmark."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    corpus = sub.add_parser("build-corpus", help="build and freeze the OpenAlex corpus and targets")
    corpus.add_argument("--config", type=Path, required=True)
    corpus.add_argument(
        "--max-works",
        type=_positive_int,
        help="override corpus.max_works_per_sweep, e.g. for a small pilot",
    )

    for name, help_text in (
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


def build_corpus_command(config_path: Path, max_works: int | None) -> int:
    cfg = load_config(config_path)
    if max_works is not None:
        corpus = dataclasses.replace(cfg.corpus, max_works_per_sweep=max_works)
        cfg = dataclasses.replace(cfg, corpus=corpus)
    with closing(OpenAlexClient.from_config(cfg.corpus)) as client:
        manifest = build_corpus(cfg, client)
    origins = ", ".join(f"{n} {origin}" for origin, n in manifest["counts_by_origin"].items())
    truncated = sum(s["truncated"] for s in manifest["sweeps"])
    print(
        f"rk build-corpus: froze {manifest['n_docs']} documents ({origins}) and "
        f"{manifest['targets']['n_final']} targets in {cfg.corpus.out_dir}; "
        f"{manifest['api_calls']} OpenAlex calls; "
        f"{truncated} of {len(manifest['sweeps'])} sweeps truncated"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-corpus":
        try:
            return build_corpus_command(args.config, args.max_works)
        except (ConfigError, CorpusError, OpenAlexError, TargetSelectionError) as exc:
            print(f"rk build-corpus: {exc}", file=sys.stderr)
            return 1
    print(f"rk {args.command}: not implemented")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
