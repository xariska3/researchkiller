"""Run an rk command; once cumulative spend passes a threshold, stop it after the current row.

Usage (from the repo root):
    python scripts/spend_guard.py <log file> <threshold USD> <bench|run> -- <command ...>

Spend is the ledger total in data/cache/llm.sqlite (billed cost, or computed cost
where none was billed), as the Ledger counts it. A row is a line appended to
data/bench/test_a.jsonl or rejections.jsonl (bench), or a row in
data/results.sqlite query_results (run). Both are read read-only.

One-off script of Phase 1: the watcher behind the $10 stop rule of the full bench
(docs/reports/full-bench.md, Notes), kept as it was run on 2026-10-13.
"""

import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LEDGER = Path("data/cache/llm.sqlite")
RESULTS = Path("data/results.sqlite")
BENCH_FILES = [Path("data/bench/test_a.jsonl"), Path("data/bench/rejections.jsonl")]


def read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)


def spend() -> float:
    conn = read_only(LEDGER)
    try:
        sql = "SELECT COALESCE(SUM(COALESCE(billed_cost, cost)), 0) FROM spend"
        return float(conn.execute(sql).fetchone()[0])
    finally:
        conn.close()


def rows(kind: str) -> int:
    if kind == "bench":  # complete lines only
        return sum(p.read_bytes().count(b"\n") for p in BENCH_FILES if p.exists())
    conn = read_only(RESULTS)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM query_results").fetchone()[0])
    finally:
        conn.close()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    log_path, threshold, kind = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    cmd = sys.argv[sys.argv.index("--") + 1 :]
    print(f"{now()} start: spend ${spend():.4f}, rows {rows(kind)}; {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        over_at = None
        last_note = time.monotonic()
        while proc.poll() is None:
            time.sleep(1 if over_at is not None else 5)
            try:
                total, n = spend(), rows(kind)
            except sqlite3.OperationalError as exc:
                print(f"{now()} poll failed: {exc}", flush=True)
                continue
            if over_at is None and total > threshold:
                over_at = n
                print(
                    f"{now()} spend ${total:.4f} passed ${threshold:.2f} at {n} rows; "
                    "stopping after the current row",
                    flush=True,
                )
            elif over_at is not None and n > over_at:
                proc.terminate()
                proc.wait()
                print(f"{now()} stopped after row {n}; spend ${total:.4f}", flush=True)
                break
            if time.monotonic() - last_note > 600:
                print(f"{now()} spend ${total:.4f}, rows {n}", flush=True)
                last_note = time.monotonic()
    print(f"{now()} exit code {proc.returncode}; spend ${spend():.4f}, rows {rows(kind)}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
