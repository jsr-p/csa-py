"""
Benchmark script for CSA estimation vs fastdid R package.

See also:
- https://tsailintung.github.io/fastdid/articles/misc.html
"""

import time
import polars as pl
from csa import estimate, agg_te


def benchmark(data: pl.DataFrame):
    res = estimate(
        data=data,
        group="g",
        time="t",
        outcome="y",
        unit="id",
        balanced=True,
        control="notyet",
        method="dr",
        verbose=False,
    )
    agg_res = agg_te(res, method="dynamic", boot=False)
    return agg_res


def run_benchmark(data: pl.DataFrame, reps: int = 5):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        benchmark(data)
        end = time.perf_counter()
        times.append(end - start)
    return times


def run_all(
    orders: list[int],
    reps: int = 25,
) -> list[dict]:
    total = []
    for i in orders:
        filename = f"data/fastdid/sim{i}.csv"

        data = (
            pl.read_csv(
                filename,
                ignore_errors=True,
                infer_schema_length=50_000,
                columns=["G", "time", "y", "unit"],
                schema_overrides={
                    "G": pl.String,
                    "time": pl.Int64,
                    "y": pl.Float64,
                    "unit": pl.String,
                },
            )
            .rename(
                {"G": "g", "unit": "id", "time": "t"},
            )
            .select("g", "t", "y", "id")
            .with_columns(pl.col("g").str.replace("Inf", "0").cast(pl.Int64))
        )

        times = run_benchmark(data, reps=reps)
        total.extend(
            [
                {
                    "file": filename,
                    "time": t,
                    "order": i,
                }
                for t in times
            ]
        )
        print(f"Benchmark for {filename} completed")
    return total


def main():
    pl.DataFrame(run_all([2, 3, 4, 5], reps=25)).write_csv(
        file := "data/fastdid/py_benchmark.csv"
    )
    print(f"Saved to {file}")
    print("Estimating...")
    pl.DataFrame(run_all([6], reps=10)).write_csv(
        file := "data/fastdid/py_benchmark_6.csv"
    )
    print(f"Saved to {file}")


if __name__ == "__main__":
    main()
