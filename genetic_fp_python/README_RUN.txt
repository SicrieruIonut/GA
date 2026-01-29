# Genetic Algorithm in functional style (Python)

Acest mini-proiect implementeaza un algoritm genetic (selectie + crossover + mutatie) pentru maximizarea unei functii (Rastrigin inversat: maximizam -Rastrigin).

Scop (tema):
- stil functional: fara stare globala; starea se transmite ca parametru; functii pure pe cat posibil
- recursivitate (evolutie generatii) si co-recursivitate (lazy sequences)
- paralelizare a evaluarii fitnessului
- masurare/comparatie (timp, memorie, calitatea solutiei)

## Structura
- src/ga_fp.py         : implementarea GA (functional-core) + backends fitness (seq / threads / proc)
- src/main.py          : rulare un singur experiment (CLI)
- src/benchmark.py     : benchmark cu repeats + scalare workers (tabel pentru raport)
- src/report_table.py  : transforma CSV-ul de benchmark intr-un tabel Markdown
- src/purity_check.py  : verificare determinism (referential transparency)
- data/population.npy  : dataset mare (1_000_000 x 10) folosit drept populatie initiala

## Instalare
Recomandat:

python -m venv .venv
.venv\Scripts\activate   (Windows)
source .venv/bin/activate (Linux/Mac)

pip install -r requirements.txt

## Rulare (GA)
Din folderul genetic_fp_python:

python src/main.py --mode seq --gens 30
python src/main.py --mode par_thread --workers 8 --gens 30
python src/main.py --mode par_proc --workers 8 --gens 30

Parametri utili pentru experimente:
- --mutation-rate 0.01
- --mutation-sigma 0.1
- --elite-fraction 0.5

## Co-recursivitate (lazy generations)
In ga_fp.py exista generations_lazy(...) care produce un iterator de (state, stats) pe generatii.

## Determinism (puritate)
python src/purity_check.py

Daca rulezi de doua ori cu aceleasi intrari (dataset, gens, seed), obtii exact acelasi rezultat.

## Benchmark + tabel pentru raport
Benchmark cu repeats + workers:

python src/benchmark.py --dataset data/population.npy --gens 30 --repeats 3 --modes seq,par_thread,par_proc --workers 1,2,4,8 --out benchmark_results.csv

Genereaza tabel Markdown:

python src/report_table.py --in benchmark_results.csv --out benchmark_table.md

### Nota importanta pentru benchmarking
NumPy poate folosi propriul paralelism intern (OpenMP/MKL) in unele build-uri. Ca sa nu „dublezi” paralelismul (oversubscription), e bine sa limitezi thread-urile interne cand compari modurile paralele:

Windows PowerShell:
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"

Linux/Mac:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python src/benchmark.py ...

## Ce pui in raport (minim, dar „cercetare”)
1) Comparatie seq vs paralel (threads vs proc) pe timp + memorie + calitate (best_fitness_final).
2) Scalare: workers = 1,2,4,8,... si speedup.
3) Parametri GA: 2-3 valori pentru mutation-rate / sigma / elite-fraction si efectul asupra calitatii vs timp.

