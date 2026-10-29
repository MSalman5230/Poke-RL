"""Measure training throughput sequentially, with fresh processes and weights.

Run with this project's Python. Reports include both training-loop throughput
and whole-process wall time; this is a speed benchmark, not a strength test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument('--learning-rate',type=float,default=3e-5)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument("--reference-workers", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-serial", action="store_true")
    args = parser.parse_args()
    counts = [int(s) for s in args.workers.split(",")]
    if min(args.steps, args.envs, args.horizon, args.repeats, args.batch_size, *counts) < 1:
        parser.error("Sizes and repeat counts must be positive")
    if max(counts) > args.envs or args.reference_workers < 0:
        parser.error("Worker count must fit envs; reference-workers must be nonnegative")
    args.initial = args.initial.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [*sorted((ROOT / "champions_rl").glob("*.py")), *sorted((ROOT/'simulator').glob('*.cjs'))]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch": version("torch"),
        "checkpoint": str(args.initial),
        "checkpoint_sha256": digest(args.initial),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "notes": [
            "Runs execute sequentially and start from the same weights; seeds vary by repeat.",
            "Parallel decisions exclude waiting phases and include both seats of current-policy self-play.",
            "Serial steps record one learner trajectory; parallel self-play records both current-policy seats.",
            "Parallel and serial opponent mixtures differ; compare systems, not identical trajectories.",
            "Whole-process time includes imports, startup, checkpoints, and shutdown.",
            "Parallel opponent fractions are recorded in each run's settings; reference replay is separate.",
            "Each process starts with a cold native-damage cache; collection and optimization times are reported separately.",
            "The serial baseline uses no reference games and 30% snapshots.",
            "Throughput alone does not demonstrate better playing strength.",
        ],
        "runs": [],
    }
    try:
        report["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        report["gpu"] = None

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    configurations = [("parallel", n) for n in counts]
    if args.include_serial:
        configurations.insert(0, ("serial", 1))
    save()
    for repeat in range(args.repeats):
        for mode, workers in configurations:
            name = f"{mode}_w{workers}_repeat{repeat}"
            output = args.output / name
            command = [sys.executable, "-u", "-m", "champions_rl",
                       "train-fast" if mode == "parallel" else "train",
                       "--initial", str(args.initial), "--output", str(output),
                       "--steps", str(args.steps), "--envs", str(args.envs), "--horizon", str(args.horizon),
                       "--device", args.device, "--seed", str(1701 + repeat), "--learning-rate", str(args.learning_rate),
                       "--stop-file", str(args.output / "STOP")]
            if mode == "parallel":
                command += ["--workers", str(workers), "--reference-workers", str(args.reference_workers),
                            "--batch-size", str(args.batch_size),'--epochs',str(args.epochs),'--validation-every','0']
            print(json.dumps({"event": "benchmark_started", "run": name}), flush=True)
            started = time.monotonic()
            with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            wall = time.monotonic() - started
            if completed.returncode:
                raise RuntimeError(f"Benchmark {name} failed; inspect its log (exit {completed.returncode})")
            metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            last = metrics[-1]
            elapsed = last["seconds"]
            row = {"mode": mode, "workers": workers, "repeat": repeat, "steps": last["steps"],
                   "training_seconds": elapsed, "decisions_per_second": last["steps"] / elapsed,
                   "process_seconds": wall, "process_decisions_per_second": last["steps"] / wall,
                   "games": last.get("games", last.get("episodes")), "command": command}
            if mode == "parallel":
                summary = json.loads((output / "summary.json").read_text())
                row.update(training_seconds=summary["elapsed_training_seconds"],
                           decisions_per_second=last["steps"] / summary["elapsed_training_seconds"],
                           simulation_steps=sum(m["simulation_steps"] for m in metrics),
                           reference_games=last["reference_games"], reference_errors=last["reference_errors"],
                           mean_current_batch=sum(m["current_positions"] for m in metrics) /
                                              sum(m["current_batches"] for m in metrics),
                           rollout_seconds=sum(m["rollout_seconds"] for m in metrics),
                           optimization_seconds=sum(m["optimization_seconds"] for m in metrics))
            report["runs"].append(row)
            report["medians"] = {
                f"{kind}_w{n}": statistics.median(r["decisions_per_second"] for r in report["runs"]
                                                 if r["mode"] == kind and r["workers"] == n)
                for kind, n in configurations
                if any(r["mode"] == kind and r["workers"] == n for r in report["runs"])
            }
            save()
            print(json.dumps({"event": "benchmark_finished", **row}), flush=True)


if __name__ == "__main__":
    main()
