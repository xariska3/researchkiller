"""The `rk` command (run as `python -m rk`)."""

from __future__ import annotations

import argparse
import dataclasses
import re
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
from rk.compete import evaluate as compete_eval
from rk.compete.resolve import Resolver
from rk.compete.systems import CompeteError, connect_store, make_system
from rk.config import ConfigError, load_config
from rk.corpus import CorpusError, OpenAlexClient, OpenAlexError, build_corpus
from rk.evaluate import CONFIG_ORDER, SUMMARY_MD, EvalError, _f, report, run_eval
from rk.live import evaluate as live_eval
from rk.live import replay as live_replay
from rk.live.fusion import FUSIONS
from rk.live.openalex_search import LiveSearchClient, OpenAlexCache
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


def _live_config_list(text: str) -> list[str]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in live_eval.LIVE_CONFIGS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown live configs {unknown}; choose from {','.join(live_eval.LIVE_CONFIGS)}"
        )
    return names


def _name_list(text: str) -> list[str]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("expected comma-separated names")
    return names


def _qid_start(text: str) -> str:
    if re.fullmatch(r"A\d{4,}", text) is None:
        raise argparse.ArgumentTypeError(f"expected a qid prefix like A0045, got {text!r}")
    return text


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
    check.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="also add the originals the judge rejected in this validate-judge run (part B)",
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

    live = sub.add_parser(
        "live-eval", help="evaluate live retrieval over OpenAlex (L0, L1, L5) on the benchmark"
    )
    _add_config_arguments(live)
    live.add_argument(
        "--configs",
        type=_live_config_list,
        help=f"comma-separated configs (default: all of {','.join(live_eval.LIVE_CONFIGS)})",
    )
    live.add_argument(
        "--from-qid",
        type=_qid_start,
        help="start at this qid in seeded order, e.g. A0045 to leave out Phase 1's tuning targets",
    )
    live.add_argument(
        "--limit", type=_positive_int, help="evaluate only the first N queries (from --from-qid)"
    )
    live.add_argument(
        "--run-dir",
        type=Path,
        help="resume this run directory, e.g. after OpenAlex's daily budget ran out",
    )
    live.add_argument(
        "--retry-errors",
        action="store_true",
        help="with --run-dir: redo the rows that ended in an error",
    )

    fusion = sub.add_parser(
        "live-fusion",
        help="compare the final-fusion variants on a live run, replayed offline from the cache",
    )
    fusion.add_argument("run_dir", type=Path, help="runs/live-<timestamp>")

    misses = sub.add_parser(
        "live-misses",
        help="write a seeded sample of a live run's L5 misses (not in the top 5) to a CSV",
    )
    misses.add_argument("run_dir", type=Path, help="runs/live-<timestamp>")
    misses.add_argument(
        "--n", type=_positive_int, default=10, help="how many misses to sample (default 10)"
    )
    misses.add_argument(
        "--out",
        type=Path,
        default=live_replay.MISSES_FILE,
        help=f"CSV to write; never overwritten (default {live_replay.MISSES_FILE})",
    )

    compete = sub.add_parser(
        "compete",
        help="compare the live tool with other systems on a stratified sample of live-eval queries",
    )
    _add_config_arguments(compete)
    compete.add_argument(
        "--limit", type=_positive_int, help="ask only the first N sampled queries (seeded order)"
    )
    compete.add_argument(
        "--systems",
        type=_name_list,
        help="comma-separated systems (default: ours and every system in compete.systems)",
    )
    compete.add_argument(
        "--dry-run",
        action="store_true",
        help="show the sample and what the uncached calls could cost; send nothing",
    )
    compete.add_argument(
        "--save-sample",
        type=Path,
        metavar="CSV",
        help=(
            f"also save the sample as a CSV, never overwritten (e.g. {compete_eval.SAMPLE_FILE})"
        ),
    )

    report_cmd = sub.add_parser("report", help="regenerate a run's summary from the database")
    report_cmd.add_argument("run_dir", type=Path, help="runs/<timestamp> or runs/live-<timestamp>")
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
    config_path: Path,
    n: int,
    out: Path,
    validation_dir: Path | None = None,
    overrides: Sequence[str] = (),
) -> int:
    cfg = load_config(config_path, overrides)
    part_a, part_b = export_human_check(cfg, n=n, out_path=out, validation_dir=validation_dir)
    line = f"rk export-human-check: wrote {part_a} accepted pairs from {cfg.bench.out_dir}"
    if validation_dir is not None:
        line += f" and {part_b} rejected originals from {validation_dir}"
    print(f"{line} to {out}")
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


def resume_hint(
    config_path: Path,
    run_dir: Path,
    configs: Sequence[str] | None,
    from_qid: str | None,
    limit: int | None,
    overrides: Sequence[str],
) -> str:
    """The live-eval command that resumes `run_dir` with the same selection."""
    parts = ["python -m rk live-eval", f"--config {config_path}", f"--run-dir {run_dir}"]
    if configs:
        parts.append(f"--configs {','.join(configs)}")
    if from_qid:
        parts.append(f"--from-qid {from_qid}")
    if limit:
        parts.append(f"--limit {limit}")
    parts += [f'--set "{o}"' for o in overrides]
    return " ".join(parts)


def live_eval_command(
    config_path: Path,
    configs: list[str] | None,
    from_qid: str | None,
    limit: int | None,
    run_dir: Path | None,
    retry_errors: bool,
    overrides: Sequence[str] = (),
) -> int:
    cfg = load_config(config_path, overrides)
    if cfg.live is None:
        raise ConfigError(f"{config_path} has no live: section")
    with ExitStack() as stack:
        cache = stack.enter_context(LLMCache(cfg.run.cache_path))
        ledger = stack.enter_context(
            Ledger(cfg.run.cache_path, cfg.run.budget_cap, command="live-eval")
        )
        agent = stack.enter_context(closing(LLMClient(cfg.agent, cache, ledger)))
        responses = stack.enter_context(closing(OpenAlexCache(cfg.live.cache_path)))
        search = stack.enter_context(
            closing(LiveSearchClient.from_config(cfg, responses, evaluation=True))
        )
        print(f"rk live-eval: {live_eval.describe_budget(search)}", flush=True)
        done = live_eval.run_live_eval(
            cfg,
            agent,
            make_encoder(cfg.retrieval),
            search,
            configs=configs,
            from_qid=from_qid,
            limit=limit,
            retry_errors=retry_errors,
            run_dir=run_dir,
        )
    progress = f"{done.n_done} of {done.n_queries} queries done"
    summary = done.run_dir / SUMMARY_MD
    if done.budget_stop is None:
        print(f"rk live-eval: {progress}; results in {cfg.run.results_db}; summary in {summary}")
        return 0
    hint = resume_hint(config_path, done.run_dir, configs, from_qid, limit, overrides)
    print(
        f"rk live-eval: stopped: {done.budget_stop}. {progress}; the unfinished query has no "
        f"row. The budget resets in {live_eval.duration(done.budget_stop.resets_in_s)} "
        f"(midnight UTC). Summary so far in {summary}. Resume with:\n  {hint}"
    )
    return 0


def live_fusion_command(run_dir: Path) -> int:
    _, cfg, _ = live_replay.load_run(run_dir)
    summary = live_replay.evaluate_fusions(run_dir, make_encoder(cfg.retrieval))
    rep, n = summary["reproduced"], summary["n_queries"]
    hit5 = ", ".join(f"{f} {summary['quality'][f]['hit_at_5']:.3f}" for f in FUSIONS)
    selected = summary["selected"]
    choice = f"L5w = {selected}" if selected else "no variant beats L5, so there is no L5w"
    print(
        f"rk live-fusion: replayed {n} checks offline (L5 reproduced {rep['L5']}/{n}, "
        f"L1 {rep['L1']}/{rep['L1_rows']}); Hit@5: {hit5}; {choice}; "
        f"table in {run_dir / live_replay.FUSION_MD}"
    )
    return 0


def live_misses_command(run_dir: Path, n: int, out: Path) -> int:
    _, cfg, _ = live_replay.load_run(run_dir)
    rows = live_replay.export_misses(run_dir, make_encoder(cfg.retrieval), n=n, out_path=out)
    print(f"rk live-misses: wrote {rows} of {run_dir}'s L5 misses to {out}")
    return 0


def compete_command(
    config_path: Path,
    limit: int | None,
    systems: list[str] | None,
    dry_run: bool,
    save_sample: Path | None = None,
    overrides: Sequence[str] = (),
) -> int:
    cfg = load_config(config_path, overrides)
    if cfg.compete is None:
        raise ConfigError(f"{config_path} has no compete: section")
    plan = compete_eval.make_plan(cfg, systems=systems, limit=limit)
    if save_sample is not None:
        compete_eval.save_sample(plan, save_sample)
        print(f"rk compete: wrote the {len(plan.sample)} sampled queries to {save_sample}")
    by_name = {s.name: s for s in cfg.compete.systems}
    with ExitStack() as stack:
        conn = stack.enter_context(closing(connect_store(cfg.compete.cache_path)))
        clients = {
            name: stack.enter_context(closing(make_system(by_name[name])))
            for name in plan.systems
            if name != compete_eval.OURS
        }
        if dry_run:
            ledger = compete_eval.make_ledger(cfg.compete, conn)
            print(compete_eval.describe_plan(plan, clients, conn, ledger))
            return 0
        resolver = stack.enter_context(closing(Resolver.from_config(cfg, conn)))
        ours = compete_eval.OURS in plan.systems
        done = compete_eval.run_compete(
            cfg,
            plan,
            clients=clients,
            resolver=resolver,
            conn=conn,
            encoder=make_encoder(cfg.retrieval) if ours else None,
            progress=lambda line: print(f"rk compete: {line}", flush=True),
        )
    hit5 = ", ".join(f"{s} {_f(q['hit_at_5'])}" for s, q in done.summary["quality"].items())
    print(
        f"rk compete: {len(plan.sample)} queries; Hit@5: {hit5}; "
        f"summary in {done.run_dir / SUMMARY_MD}"
    )
    return 0


def report_command(run_dir: Path) -> int:
    if live_eval.is_live_run(run_dir):
        summary = live_eval.report(run_dir)
    else:
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
    retry = args.command in ("run", "live-eval") and args.retry_errors
    if retry and args.run_dir is None:
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
            return export_human_check_command(
                args.config, args.n, args.out, args.validation_dir, args.overrides
            )
        if args.command == "run":
            return run_command(
                args.config,
                args.configs,
                args.limit,
                args.run_dir,
                args.retry_errors,
                args.overrides,
            )
        if args.command == "live-eval":
            return live_eval_command(
                args.config,
                args.configs,
                args.from_qid,
                args.limit,
                args.run_dir,
                args.retry_errors,
                args.overrides,
            )
        if args.command == "live-fusion":
            return live_fusion_command(args.run_dir)
        if args.command == "live-misses":
            return live_misses_command(args.run_dir, args.n, args.out)
        if args.command == "compete":
            return compete_command(
                args.config,
                args.limit,
                args.systems,
                args.dry_run,
                args.save_sample,
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
        CompeteError,
    ) as exc:
        print(f"rk {args.command}: {exc}", file=sys.stderr)
        return 1


def entry() -> int:
    """Process entry point (`python -m rk` and the `rk` script): load .env, then run main()."""
    load_env_file()
    return main()


if __name__ == "__main__":
    raise SystemExit(entry())
