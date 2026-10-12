"""The `rk` command (run as `python -m rk`). build-corpus and make-bench are implemented."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from contextlib import ExitStack, closing
from pathlib import Path

from rk.bench import BenchError, format_accepted, make_bench
from rk.config import ConfigError, load_config
from rk.corpus import CorpusError, OpenAlexClient, OpenAlexError, build_corpus
from rk.llm import BudgetExceeded, Ledger, LLMCache, LLMClient, LLMError
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

    index = sub.add_parser("build-index", help="build the BM25 and dense indexes")
    index.add_argument("--config", type=Path, required=True)

    bench = sub.add_parser("make-bench", help="generate (or resume) the Test A benchmark")
    bench.add_argument("--config", type=Path, required=True)
    bench.add_argument(
        "--limit", type=_positive_int, help="work on at most N targets in this run"
    )
    bench.add_argument(
        "--show",
        action="store_true",
        help="print title, abstract opening and proposal for each accepted target",
    )

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


def make_bench_command(config_path: Path, limit: int | None, show: bool) -> int:
    cfg = load_config(config_path)
    with ExitStack() as stack:
        cache = stack.enter_context(LLMCache(cfg.run.cache_path))
        ledger = stack.enter_context(
            Ledger(cfg.run.cache_path, cfg.run.budget_cap, command="make-bench")
        )
        paraphraser = stack.enter_context(closing(LLMClient(cfg.paraphraser, cache, ledger)))
        judge = stack.enter_context(closing(LLMClient(cfg.judge, cache, ledger)))
        stats = make_bench(cfg, paraphraser, judge, limit=limit)
    print(
        f"rk make-bench: {stats['accepted']} of {stats['n_targets_goal']} queries accepted "
        f"({stats['rejected']} rejected attempts, {stats['exhausted']} targets exhausted); "
        f"method cost ${stats['method_cost']:.4f}, money spent ${stats['money_spent']:.4f}; "
        f"outputs in {cfg.bench.out_dir}"
    )
    if show:
        print(format_accepted(cfg), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-corpus":
            return build_corpus_command(args.config, args.max_works)
        if args.command == "make-bench":
            return make_bench_command(args.config, args.limit, args.show)
    except (
        ConfigError,
        CorpusError,
        OpenAlexError,
        TargetSelectionError,
        BenchError,
        LLMError,
        BudgetExceeded,
    ) as exc:
        print(f"rk {args.command}: {exc}", file=sys.stderr)
        return 1
    print(f"rk {args.command}: not implemented")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
