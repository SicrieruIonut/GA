from ga_fp import run_ga


def main():
    dataset = "data/population.npy"
    gens = 10
    seed = 12345

    r1 = run_ga(mode="seq", dataset_path=dataset, generations=gens, seed=seed)
    r2 = run_ga(mode="seq", dataset_path=dataset, generations=gens, seed=seed)

    same_best = r1["best_fitness_final"] == r2["best_fitness_final"]
    same_mean = r1["mean_fitness_final"] == r2["mean_fitness_final"]
    same_seed_end = r1["seed_end"] == r2["seed_end"]

    print(
        "best identical    :",
        same_best,
        r1["best_fitness_final"],
        r2["best_fitness_final"],
    )
    print(
        "mean identical    :",
        same_mean,
        r1["mean_fitness_final"],
        r2["mean_fitness_final"],
    )
    print("seed_end identical:", same_seed_end, r1["seed_end"], r2["seed_end"])

    assert (
        same_best and same_mean and same_seed_end
    ), "Not deterministic: purity check failed!"
    print("OK: Deterministic for same input (dataset, gens, seed).")


if __name__ == "__main__":
    main()
