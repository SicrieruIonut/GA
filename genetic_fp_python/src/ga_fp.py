from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Tuple, Optional, Dict, Any

import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import shared_memory

Array = np.ndarray
FitnessEval = Callable[[Array], Array]


def next_seed(seed: int) -> int:
    return (1103515245 * seed + 12345) & 0x7FFFFFFF


def fitness_rastrigin(pop: Array) -> Array:
    d = pop.shape[1]
    ras = 10.0 * d + np.sum(pop * pop - 10.0 * np.cos(2.0 * math.pi * pop), axis=1)
    return -ras


def evaluate_seq(pop: Array) -> Array:
    return fitness_rastrigin(pop)


def _fitness_chunk(pop: Array, start: int, end: int) -> Tuple[int, Array]:
    """Evaluate fitness on a slice. Used by thread-based parallel evaluation."""
    return start, fitness_rastrigin(pop[start:end])


def evaluate_parallel_threads(
    pop: Array,
    workers: Optional[int] = None,
    chunks: Optional[int] = None,
) -> Array:
    """Parallel fitness evaluation using threads.

    Why threads here?
    - NumPy ufuncs (sum/cos/etc.) release the GIL, so threads can run concurrently.
    - Avoids process + shared-memory overhead (copy/IPC/pickling).

    NOTE: For fair benchmarking, it's a good idea to limit internal NumPy threads
    via environment variables (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, etc.).
    """
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    n = pop.shape[0]
    if chunks is None:
        chunks = workers * 8

    boundaries = np.linspace(0, n, num=chunks + 1, dtype=int)
    ranges = [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(chunks)]
    ranges = [(a, b) for (a, b) in ranges if b > a]

    fits = np.empty((n,), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start, part in ex.map(lambda ab: _fitness_chunk(pop, ab[0], ab[1]), ranges):
            fits[start : start + part.shape[0]] = part
    return fits


def _worker_fitness_chunk(
    args: Tuple[str, Tuple[int, int], str, int, int],
) -> Tuple[int, Array]:
    shm_name, shape, dtype_str, start, end = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        pop = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        chunk = pop[start:end]
        fits = fitness_rastrigin(chunk)
        return start, fits
    finally:
        shm.close()


def evaluate_parallel_sharedmem(
    pop: Array,
    workers: Optional[int] = None,
    chunks: Optional[int] = None,
) -> Array:
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    n = pop.shape[0]
    if chunks is None:
        chunks = workers * 4

    boundaries = np.linspace(0, n, num=chunks + 1, dtype=int)
    ranges = [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(chunks)]
    ranges = [(a, b) for (a, b) in ranges if b > a]

    shm = shared_memory.SharedMemory(create=True, size=pop.nbytes)
    try:
        shm_arr = np.ndarray(pop.shape, dtype=pop.dtype, buffer=shm.buf)
        shm_arr[:] = pop

        tasks = [(shm.name, pop.shape, str(pop.dtype), a, b) for (a, b) in ranges]
        fits = np.empty((n,), dtype=np.float64)

        with ProcessPoolExecutor(max_workers=workers) as ex:
            for start, part in ex.map(_worker_fitness_chunk, tasks):
                fits[start : start + part.shape[0]] = part

        return fits
    finally:
        shm.close()
        shm.unlink()


def load_population(dataset_path: str) -> Array:
    pop = np.load(dataset_path, mmap_mode="r")
    if pop.ndim != 2:
        raise ValueError("Dataset must be 2D (N, D).")
    return pop


def select_top_k(pop: Array, fits: Array, k: int) -> Array:
    idx = np.argpartition(-fits, kth=k - 1)[:k]
    idx_sorted = idx[np.argsort(-fits[idx])]
    return pop[idx_sorted]


def crossover_pairwise(parents: Array) -> Array:
    k, d = parents.shape
    if k % 2 == 1:
        raise ValueError("Number of parents must be even.")
    p1 = parents[0::2]
    p2 = parents[1::2]
    point = d // 2
    c1 = np.concatenate([p1[:, :point], p2[:, point:]], axis=1)
    c2 = np.concatenate([p2[:, :point], p1[:, point:]], axis=1)
    children = np.stack([c1, c2], axis=1).reshape(k, d)
    return children


def mutate(
    pop: Array, seed: int, rate: float = 0.01, sigma: float = 0.1
) -> Tuple[Array, int]:
    rng = np.random.default_rng(seed)
    mask = rng.random(pop.shape) < rate
    noise = rng.normal(loc=0.0, scale=sigma, size=pop.shape)
    delta = np.where(mask, noise, 0.0)
    mutated = np.clip(pop + delta, -5.12, 5.12)
    return mutated, next_seed(seed)


@dataclass(frozen=True)
class GAConfig:
    generations: int = 50
    mutation_rate: float = 0.01
    mutation_sigma: float = 0.1
    elite_fraction: float = 0.5


@dataclass(frozen=True)
class GAState:
    pop: Array
    gen: int
    seed: int


@dataclass(frozen=True)
class GAStats:
    gen: int
    best_fitness: float
    mean_fitness: float


def next_generation(
    pop: Array,
    seed: int,
    fitness_eval: FitnessEval,
    mutation_rate: float,
    mutation_sigma: float,
    elite_fraction: float,
) -> Tuple[Array, int, GAStats]:
    fits = fitness_eval(pop)
    best = float(np.max(fits))
    mean = float(np.mean(fits))

    n = pop.shape[0]
    # How many parents survive (elitism). Kept as a parameter for experiments.
    # Must be >= 2 and even (pairwise crossover).
    k = int(max(2, min(n, round(n * elite_fraction))))
    if k % 2 == 1:
        k -= 1

    parents = select_top_k(pop, fits, k)
    children = crossover_pairwise(parents)
    children, seed2 = mutate(children, seed, rate=mutation_rate, sigma=mutation_sigma)

    needed = n - parents.shape[0]
    children = children[:needed]
    new_pop = np.concatenate([parents, children], axis=0)

    return new_pop, seed2, GAStats(gen=-1, best_fitness=best, mean_fitness=mean)


def evolve_recursive(
    state: GAState,
    config: GAConfig,
    fitness_eval: FitnessEval,
) -> Tuple[GAState, Tuple[GAStats, ...]]:
    if state.gen >= config.generations:
        return state, tuple()

    new_pop, new_seed, stats = next_generation(
        state.pop,
        state.seed,
        fitness_eval,
        mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma,
        elite_fraction=config.elite_fraction,
    )

    stats = GAStats(
        gen=state.gen, best_fitness=stats.best_fitness, mean_fitness=stats.mean_fitness
    )
    new_state = GAState(pop=new_pop, gen=state.gen + 1, seed=new_seed)

    final_state, rest = evolve_recursive(new_state, config, fitness_eval)
    return final_state, (stats,) + rest


def generations_lazy(
    state: GAState,
    config: GAConfig,
    fitness_eval: FitnessEval,
) -> Iterator[Tuple[GAState, GAStats]]:
    fits = fitness_eval(state.pop)
    yield state, GAStats(
        gen=state.gen,
        best_fitness=float(np.max(fits)),
        mean_fitness=float(np.mean(fits)),
    )

    if state.gen >= config.generations:
        return

    new_pop, new_seed, _ = next_generation(
        state.pop,
        state.seed,
        fitness_eval,
        mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma,
        elite_fraction=config.elite_fraction,
    )

    next_state = GAState(pop=new_pop, gen=state.gen + 1, seed=new_seed)
    yield from generations_lazy(next_state, config, fitness_eval)


def run_ga(
    mode: str,
    dataset_path: str,
    generations: int,
    seed: int,
    workers: Optional[int] = None,
    chunks: Optional[int] = None,
    mutation_rate: float = 0.01,
    mutation_sigma: float = 0.1,
    elite_fraction: float = 0.5,
) -> Dict[str, Any]:
    pop0 = load_population(dataset_path)
    config = GAConfig(
        generations=generations,
        mutation_rate=mutation_rate,
        mutation_sigma=mutation_sigma,
        elite_fraction=elite_fraction,
    )
    state0 = GAState(pop=pop0, gen=0, seed=seed)

    if mode == "seq":
        fitness_eval = evaluate_seq
    elif mode in {"par", "par_proc"}:
        fitness_eval = lambda p: evaluate_parallel_sharedmem(
            p, workers=workers, chunks=chunks
        )
    elif mode in {"par_thread", "thread"}:
        fitness_eval = lambda p: evaluate_parallel_threads(p, workers=workers, chunks=chunks)
    else:
        raise ValueError("mode must be one of: 'seq', 'par'/'par_proc', 'par_thread'")

    t0 = time.perf_counter()
    final_state, stats = evolve_recursive(state0, config, fitness_eval)
    t1 = time.perf_counter()

    final_fits = fitness_eval(final_state.pop)

    return {
        "mode": mode,
        "dataset": dataset_path,
        "shape": tuple(pop0.shape),
        "generations": generations,
        "time_sec": (t1 - t0),
        "best_fitness_final": float(np.max(final_fits)),
        "mean_fitness_final": float(np.mean(final_fits)),
        "workers": workers if mode != "seq" else 1,
        "chunks": chunks if mode != "seq" else None,
        "seed_start": seed,
        "seed_end": final_state.seed,
        "stats_per_gen": stats,
    }
