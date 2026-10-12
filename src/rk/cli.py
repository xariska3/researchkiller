"""The `rk` command (run as `python -m rk`)."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from contextlib import ExitStack, closing
from pathlib import Path

from dotenv import load_dotenv

from rk.bench import (
    HUMAN_CHECK_FILE,
    JUDGE_VALIDATION_DIR,
    BenchError,
    export_human_check,
    format_accepted,
    make_bench,
    validate_judge,
)
from rk.config import ConfigError, load_config
from rk.corpus import CorpusError, OpenAlexClient, OpenAlexError, build_corpus
from rk.evaluate import CONFIG_ORDER, SUMMARY_MD, EvalError, report, run_eval
from rk.llm import BudgetExceeded, Ledger, LLMCache, LLMClient, LLMError
from rk.retrieval import RetrievalError, build_index, make_encoder
from rk.targets import TargetSelectionError

ENV_FILE = Path(".env")  # API keys; relative to the repo root, like the config's paths


def load_env_file(path: Path = ENV_FILE) -> bool:
    """Load KEY=value lines from `path` into the environment, if the file exists.

    Variables already set in the environment win over the file. Returns
    whether a file was found.
    """
    if not path.is_file():
        return False
    load_dotenv(path, override=False)
    return True


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _config_list(text: str) -> list[str]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIG_ORDER]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown configs {unknown}; choose from {','.join(CONFIG_ORDER)}"
        )
    return names


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value (repeatable), e.g. --set judge.name=openai/gpt-4.1",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rk", description="ResearchKiller Phase 1: terminology-shift retrieval benchmark."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    corpus = sub.add_parser("build-corpus", help="build and freeze the OpenAlex corpus and targets")
    _add_config_arguments(corpus)
    corpus.add_argument(
        "--max-works",
        type=_positive_int,
        help="override corpus.max_works_per_sweep, e.g. for a small pilot",
    )

    index = sub.add_parser("build-index", help="build the BM25 and dense indexes")
    _add_config_arguments(index)

    bench = sub.add_parser("make-bench", help="generate (or resume) the Test A benchmark")
    _add_config_arguments(bench)
    bench.add_argument(
        "--limit", type=_positive_int, help="work on at most N targets in this run"
    )
    bench.add_argument(
        "--show",
        action="store_true",
        help="print title, abstract opening and proposal for each accepted target",
    )

    validate = sub.add_parser(
        "validate-judge",
        help="measure the judge on accepted pairs and on deliberately corrupted proposals",
    )
    _add_config_arguments(validate)
    validate.add_argument(
        "--n", type=_positive_int, default=30, help="use the first N accepted pairs (default 30)"
    )
    validate.add_argument(
        "--out-dir",
        type=Path,
        default=JUDGE_VALIDATION_DIR,
        help=f"where to write the results (default {JUDGE_VALIDATION_DIR})",
    )

    check = sub.add_parser(
        "export-human-check", help="write a random sample of accepted pairs to a CSV to check by hand"
    )
    _add_config_arguments(check)
    check.add_argument(
        "--n", type=_positive_int, default=20, help="how many pairs to sample (default 20)"
    )
    check.add_argument(
        "--out",
        type=Path,
        default=HUMAN_CHECK_FILE,
        help=f"CSV to write; never overwritten (default {HUMAN_CHECK_FILE})",
    )

    run = sub.add_parser("run", help="evaluate retrieval configurations on the benchmark")
    _add_config_arguments(run)
    run.add_argument(
        "--configs",
        type=_config_list,
        help=f"comma-separated configs to run (default: all of {','.join(CONFIG_ORDER)})",
    )
    run.add_argument("--limit", type=_positive_int, help="evaluate only the first N queries")
    run.add_argument(
        "--run-dir", type=Path, help="resume this run directory instead of starting a new run"
    )
    run.add_argument(
        "--retry-errors",
        action="store_true",
        help="with --run-dir: redo the rows that ended in an LLM error",
    )

    report_cmd = sub.add_parser("report", help="regenerate a run's summary from the database")
    report_cmd.add_argument("run_dir", type=Path, help="runs/<timestamp>")
    return parser


def build_corpus_command(
    config_path: Path, max_works: int | None, overrides: Sequence[str] = ()
) -> int:
    cfg = load_config(config_path, overrides)
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


def build_index_command(config_path: Path, overrides: Sequence[str] = ()) -> int:
    cfg = load_config(config_path, overrides)
    encoder = make_encoder(cfg.retrieval)
    manifest = build_index(cfg, encoder)
    print(
        f"rk build-index: indexed {manifest['n_docs']} documents in "
        f"{manifest['build_seconds']:.1f} s with {manifest['encoder']} on {manifest['device']}; "
        f"saved to {cfg.retrieval.index_dir}"
    )
    return 0


def make_bench_command(
    config_path: Path, limit: int | None, show: bool, overrides: Sequence[str] = ()
) -> int:
    cfg = load_config(config_path, overrides)
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
        f"({stats['rejected']} rejected attempts, {stats['exhausted']} targets exhausted, "
        f"{stats['skipped']} skipped as not research contributions); "
        f"method cost ${stats['method_cost']:.4f}, money spent ${stats['money_spent']:.4f}; "
        f"outputs in {cfg.bench.out_dir}"
    )
    if show:
        print(format_accepted(cfg), end="")
    return 0


def validate_judge_command(
    config_path: Path, n: int, out_dir: Path, overrides: Sequence[str] = ()
) -> int:
    cfg = load_config(config_path, overrides)
    with ExitStack() as stack:
        cache = stack.enter_context(LLMCache(cfg.run.cache_path))
        ledger = stack.enter_context(
            Ledger(cfg.run.cache_path, cfg.run.budget_cap, command="validate-judge")
        )
        paraphraser = stack.enter_context(closing(LLMClient(cfg.paraphraser, cache, ledger)))
        judge = stack.enter_context(closing(LLMClient(cfg.judge, cache, ledger)))
        summary = validate_judge(cfg, paraphraser, judge, n=n, out_dir=out_dir)
    lines = [f"rk validate-judge: {summary['n_pairs']} pairs from {summary['bench_dir']}"]
    for variant, s in summary["variants"].items():
        verb, count, rate = (
            ("accepted", s["accepted"], s["acceptance_rate"])
            if variant == "original"
            else ("rejected", s["rejected"], s["rejection_rate"])
        )
        lines.append(f"  {variant:12} {verb} {count}/{s['n']} ({rate:.0%})")
    lines.append(
        f"method cost ${summary['method_cost']:.4f}, money spent ${summary['money_spent']:.4f}; "
        f"outputs in {out_dir}"
    )
    print("\n".join(lines))
    return 0


def export_human_check_command(
    config_path: Path, n: int, out: Path, overrides: Sequence[str] = ()
) -> int:
    cfg = load_config(config_path, overrides)
    written = export_human_check(cfg, n=n, out_path=out)
    print(f"rk export-human-check: wrote {written} pairs from {cfg.bench.out_dir} to {out}")
    return 0


def run_command(
    config_path: Path,
    configs: list[str] | None,
    limit: int | None,
    run_dir: Path | None,
    retry_errors: bool,
    overrides: Sequence[str] = (),
) -> int:
    cfg = load_config(config_path, overrides)
    with ExitStack() as stack:
        cache = stack.enter_context(LLMCache(cfg.run.cache_path))
        ledger = stack.enter_context(Ledger(cfg.run.cache_path, cfg.run.budget_cap, command="run"))
        agent = stack.enter_context(closing(LLMClient(cfg.agent, cache, ledger)))
        out = run_eval(
            cfg,
            agent,
            make_encoder(cfg.retrieval),
            configs=configs,
            limit=limit,
            retry_errors=retry_errors,
            run_dir=run_dir,
        )
    print(f"rk run: results in {cfg.run.results_db}; summary in {out / SUMMARY_MD}")
    return 0


def report_command(run_dir: Path) -> int:
    summary = report(run_dir)
    errors = sum(summary["errors"].values())
    print(
        f"rk report: rewrote {run_dir / SUMMARY_MD} and summary.json "
        f"({len(summary['configs'])} configs, {errors} error rows)"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.retry_errors and args.run_dir is None:
        parser.error("--retry-errors needs --run-dir (the run whose errors to retry)")
    try:
        if args.command == "build-corpus":
            return build_corpus_command(args.config, args.max_works, args.overrides)
        if args.command == "build-index":
            return build_index_command(args.config, args.overrides)
        if args.command == "make-bench":
            return make_bench_command(args.config, args.limit, args.show, args.overrides)
        if args.command == "validate-judge":
            return validate_judge_command(args.config, args.n, args.out_dir, args.overrides)
        if args.command == "export-human-check":
            return export_human_check_command(args.config, args.n, args.out, args.overrides)
        if args.command == "run":
            return run_command(
                args.config,
                args.configs,
                args.limit,
                args.run_dir,
                args.retry_errors,
                args.overrides,
            )
        return report_command(args.run_dir)
    except (
        ConfigError,
        CorpusError,
        OpenAlexError,
        TargetSelectionError,
        BenchError,
        LLMError,
        BudgetExceeded,
        RetrievalError,
        EvalError,
    ) as exc:
        print(f"rk {args.command}: {exc}", file=sys.stderr)
        return 1


def entry() -> int:
    """Process entry point (`python -m rk` and the `rk` script): load .env, then run main()."""
    load_env_file()
    return main()


if __name__ == "__main__":
    raise SystemExit(entry())
