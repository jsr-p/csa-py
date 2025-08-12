"""
Benchmark script for CSA estimation.
Mirrors R benchmarking with att_gt + aggte.

Usage:
    python scripts/benchmark/bench.py --N 1000 --e 45 --delta 3
"""

import argparse
import time
import os
import polars as pl
from csa import estimate, agg_te


def benchmark(data: pl.DataFrame):
    res = estimate(
        data=data,
        group="g",
        time="t",
        outcome="y",
        unit="id",
        covariates=["X"],
        balanced=False,
        control="notyet",
        method="dr",
        verbose=False,
    )
    agg_res = agg_te(res, method="dynamic", boot=False)
    return agg_res


def run_benchmark(file_path: str, reps: int = 5):
    times = []
    for _ in range(reps):
        data = pl.read_csv(file_path)
        start = time.perf_counter()
        benchmark(data)
        end = time.perf_counter()
        times.append(end - start)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--e", type=int, required=True)
    parser.add_argument("--delta", type=int, required=True)
    args = parser.parse_args()

    filename = f"data/testing/sim_ub_{args.N}_{args.e}_{args.delta}.csv"
    print(f"Profiling CSA Python with N={args.N}, e={args.e}, delta={args.delta}")

    times = run_benchmark(filename)
    summary = [
        {"time": t, "N": args.N, "e": args.e, "delta": args.delta, "file": filename}
        for t in times
    ]
    os.makedirs("data/benchmark", exist_ok=True)
    pl.DataFrame(summary).write_csv(
        file := f"data/benchmark/py_{args.N}_{args.e}_{args.delta}.json"
    )

    print(f"Saved benchmark summary to {file}")


if __name__ == "__main__":
    main()
