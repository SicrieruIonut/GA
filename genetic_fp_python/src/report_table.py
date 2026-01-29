import argparse

import pandas as pd


def fmt_pm(mean: float, std: float, ndigits: int = 3) -> str:
    return f"{mean:.{ndigits}f}±{std:.{ndigits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Markdown table from benchmark_results.csv (aggregated results)."
    )
    parser.add_argument("--in", dest="inp", type=str, default="benchmark_results.csv")
    parser.add_argument("--out", dest="out", type=str, default="benchmark_table.md")
    args = parser.parse_args()

    df = pd.read_csv(args.inp)

    # Keep only the columns we want to show in the report.
    view = df[[
        "mode",
        "workers",
        "repeats",
        "time_mean_sec",
        "time_std_sec",
        "speedup_vs_seq",
        "best_mean",
        "best_std",
        "mem_delta_mean_mb",
        "mem_delta_std_mb",
        "cpu_self_mean_sec",
        "cpu_children_mean_sec",
    ]].copy()

    view["time"] = view.apply(lambda r: fmt_pm(r.time_mean_sec, r.time_std_sec, 3), axis=1)
    view["best"] = view.apply(lambda r: fmt_pm(r.best_mean, r.best_std, 3), axis=1)
    view["mem_delta_mb"] = view.apply(lambda r: fmt_pm(r.mem_delta_mean_mb, r.mem_delta_std_mb, 1), axis=1)
    view["speedup"] = view.speedup_vs_seq.map(lambda x: f"{x:.2f}x")
    view["cpu_total_sec"] = (view.cpu_self_mean_sec + view.cpu_children_mean_sec).map(lambda x: f"{x:.2f}")

    out = view[[
        "mode",
        "workers",
        "repeats",
        "time",
        "speedup",
        "best",
        "mem_delta_mb",
        "cpu_total_sec",
    ]].sort_values(["mode", "workers"])

    md = out.to_markdown(index=False)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
