import csv
import os
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psutil
import time

try:
    import resource  # Unix-only (Linux/macOS)

    HAS_RESOURCE = True
except ImportError:
    resource = None
    HAS_RESOURCE = False

from ga_fp import run_ga


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def rusage_cpu_seconds() -> Tuple[float, float, float, float]:
    """Return (self_user, self_sys, child_user, child_sys) in seconds.
    On Windows, 'resource' is unavailable, so we approximate using psutil for self
    and set children to 0.0.
    """
    if HAS_RESOURCE:
        self_ru = resource.getrusage(resource.RUSAGE_SELF)
        child_ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        return self_ru.ru_utime, self_ru.ru_stime, child_ru.ru_utime, child_ru.ru_stime

    # Windows fallback
    ct = psutil.Process().cpu_times()
    return float(ct.user), float(ct.system), 0.0, 0.0


def run_one(
    *,
    mode: str,
    dataset: str,
    generations: int,
    seed: int,
    workers: Optional[int],
    chunks: Optional[int],
    mutation_rate: float,
    mutation_sigma: float,
    elite_fraction: float,
) -> Dict[str, float]:
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1024**2
    cpu0 = rusage_cpu_seconds()

    res = run_ga(
        mode=mode,
        dataset_path=dataset,
        generations=generations,
        seed=seed,
        workers=workers,
        chunks=chunks,
        mutation_rate=mutation_rate,
        mutation_sigma=mutation_sigma,
        elite_fraction=elite_fraction,
    )

    cpu1 = rusage_cpu_seconds()
    mem_after = proc.memory_info().rss / 1024**2

    self_cpu = (cpu1[0] - cpu0[0]) + (cpu1[1] - cpu0[1])
    child_cpu = (cpu1[2] - cpu0[2]) + (cpu1[3] - cpu0[3])

    return {
        "time_sec": float(res["time_sec"]),
        "best_fitness_final": float(res["best_fitness_final"]),
        "mean_fitness_final": float(res["mean_fitness_final"]),
        "mem_before_mb": float(mem_before),
        "mem_after_mb": float(mem_after),
        "mem_delta_mb": float(mem_after - mem_before),
        "cpu_self_sec": float(self_cpu),
        "cpu_children_sec": float(child_cpu),
    }


def summarize(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Benchmark GA (seq vs parallel) with repeats and worker scaling"
    )
    p.add_argument("--dataset", type=str, default="data/population.npy")
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--modes",
        type=str,
        default="seq,par_thread,par_proc",
        help="Comma-separated: seq, par_thread, par_proc",
    )
    p.add_argument(
        "--workers",
        type=str,
        default="1,2,4,8",
        help="Comma-separated worker counts (ignored for seq).",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--mutation-rate", type=float, default=0.01)
    p.add_argument("--mutation-sigma", type=float, default=0.1)
    p.add_argument("--elite-fraction", type=float, default=0.5)
    p.add_argument("--out", type=str, default="benchmark_results.csv")
    args = p.parse_args()

    modes = parse_csv_strs(args.modes)
    worker_list = parse_csv_ints(args.workers)

    rows: List[Dict[str, object]] = []

    # We use the first seq result as baseline for speedup.
    baseline_time: Optional[float] = None

    for mode in modes:
        if mode == "seq":
            configs = [(mode, 1)]
        else:
            configs = [(mode, w) for w in worker_list]

        for m, w in configs:
            times: List[float] = []
            bests: List[float] = []
            means: List[float] = []
            mem_deltas: List[float] = []
            cpu_self: List[float] = []
            cpu_children: List[float] = []

            for i in range(args.repeats):
                seed_i = args.seed + i
                workers = None if m == "seq" else w

                # Sensible default chunking per backend.
                if args.chunks is not None:
                    chunks = args.chunks
                else:
                    if m == "par_proc":
                        chunks = w * 4
                    elif m == "par_thread":
                        chunks = w * 8
                    else:
                        chunks = None

                out = run_one(
                    mode=m,
                    dataset=args.dataset,
                    generations=args.gens,
                    seed=seed_i,
                    workers=workers,
                    chunks=chunks,
                    mutation_rate=args.mutation_rate,
                    mutation_sigma=args.mutation_sigma,
                    elite_fraction=args.elite_fraction,
                )

                times.append(out["time_sec"])
                bests.append(out["best_fitness_final"])
                means.append(out["mean_fitness_final"])
                mem_deltas.append(out["mem_delta_mb"])
                cpu_self.append(out["cpu_self_sec"])
                cpu_children.append(out["cpu_children_sec"])

            t_mean, t_std = summarize(times)
            b_mean, b_std = summarize(bests)
            m_mean, m_std = summarize(means)
            md_mean, md_std = summarize(mem_deltas)
            cs_mean, cs_std = summarize(cpu_self)
            cc_mean, cc_std = summarize(cpu_children)

            if mode == "seq" and baseline_time is None:
                baseline_time = t_mean

            speedup = (baseline_time / t_mean) if baseline_time else 1.0

            row: Dict[str, object] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": m,
                "dataset": args.dataset,
                "gens": args.gens,
                "workers": w,
                "chunks": (
                    args.chunks
                    if args.chunks is not None
                    else (
                        w * 8
                        if m == "par_thread"
                        else (w * 4 if m == "par_proc" else "")
                    )
                ),
                "repeats": args.repeats,
                "time_mean_sec": t_mean,
                "time_std_sec": t_std,
                "speedup_vs_seq": speedup,
                "best_mean": b_mean,
                "best_std": b_std,
                "meanfitness_mean": m_mean,
                "meanfitness_std": m_std,
                "mem_delta_mean_mb": md_mean,
                "mem_delta_std_mb": md_std,
                "cpu_self_mean_sec": cs_mean,
                "cpu_self_std_sec": cs_std,
                "cpu_children_mean_sec": cc_mean,
                "cpu_children_std_sec": cc_std,
            }
            rows.append(row)

            print(
                f"[{m:10s} w={w:2d}] time={t_mean:.3f}±{t_std:.3f}s  speedup={speedup:.2f}x  best={b_mean:.3f}±{b_std:.3f}"
            )

    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
