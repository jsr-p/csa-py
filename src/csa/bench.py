import time
from collections.abc import Callable
import polars as pl


def run_benchmark(benchmark: Callable, reps: int = 5) -> list[float]:
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        benchmark()
        end = time.perf_counter()
        times.append(end - start)
    return times


def read_fastdid(filename: str) -> pl.DataFrame:
    return (
        pl.read_csv(
            filename,
            ignore_errors=True,
            infer_schema_length=50_000,
            columns=["G", "time", "y", "unit", "x", "x2", "xvar"],
            schema_overrides={
                "G": pl.String,
                "time": pl.Int64,
                "y": pl.Float64,
                "unit": pl.String,
                "x": pl.Float64,
                "x2": pl.Float64,
                "xvar": pl.Float64,
            },
        )
        .rename({"G": "g", "unit": "id", "time": "t", "xvar": "constant"})
        .with_columns(pl.col("g").str.replace("Inf", "0").cast(pl.Int64))
    )
