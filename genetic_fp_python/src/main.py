import argparse
import os
import psutil

from ga_fp import run_ga


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["seq", "par", "par_thread", "par_proc"], default="seq")
    p.add_argument("--dataset", type=str, default="data/population.npy")
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--mutation-rate", type=float, default=0.01)
    p.add_argument("--mutation-sigma", type=float, default=0.1)
    p.add_argument("--elite-fraction", type=float, default=0.5)
    args = p.parse_args()

    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1024**2

    res = run_ga(
        mode=args.mode,
        dataset_path=args.dataset,
        generations=args.gens,
        seed=args.seed,
        workers=args.workers if args.mode != "seq" else None,
        chunks=args.chunks if args.mode != "seq" else None,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        elite_fraction=args.elite_fraction,
    )

    mem_after = proc.memory_info().rss / 1024**2

    print("\n=== RESULT ===")
    print("Mode:", res["mode"])
    print("Dataset:", res["dataset"])
    print("Shape:", res["shape"])
    print("Generations:", res["generations"])
    print("Time (sec):", round(res["time_sec"], 4))
    print("Best fitness final:", round(res["best_fitness_final"], 6))
    print("Mean fitness final:", round(res["mean_fitness_final"], 6))
    print("Workers:", res["workers"])
    print("Seed start:", res["seed_start"])
    print("Seed end  :", res["seed_end"])
    print("Mem before (MB):", round(mem_before, 2))
    print("Mem after  (MB):", round(mem_after, 2))
    print("Δmem (MB):", round(mem_after - mem_before, 2))


if __name__ == "__main__":
    main()
