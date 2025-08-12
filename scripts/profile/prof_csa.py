import time
import click
import polars as pl
from line_profiler import LineProfiler

from csa import estimate, agg_te
from csa import csa_ as csa_mod
from csa import drdid


@click.group()
def cli():
    pass


def benchmark(file_path: str):
    data = pl.read_csv(file_path)
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
    )
    print(res)
    print(res.atts[(30, 28)])  # (g, t)
    agg_res = agg_te(res, method="dynamic", boot=False)
    print(res.estimates)
    print(agg_res.estimates)


def benchmark_fastdid(file: str):
    data = (
        pl.read_csv(
            file,
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
    res = estimate(
        data=data,
        group="g",
        time="t",
        outcome="y",
        unit="id",
        balanced=True,
        control="notyet",
        method="dr",
        verbose=True,
    )
    agg_res = agg_te(res, method="dynamic", boot=False)
    print(agg_res.estimates)


@cli.command()
@click.option(
    "--file",
    "-f",
    type=str,
    default="data/testing/sim_ub_10000_45_9.csv",
)
@click.option("--fn", type=str, default="csa")
def line(file: str, fn: str):
    match fn:
        case "csa":
            bmark_fn = benchmark
        case "fastdid":
            bmark_fn = benchmark_fastdid
        case _:
            raise ValueError(f"Unknown function name: {fn}")
    lp = LineProfiler()
    lp.add_function(estimate)
    lp.add_function(csa_mod.att_gt_all)
    lp.add_function(csa_mod.subset_data)
    lp.add_function(csa_mod.get_base)
    lp.add_function(csa_mod.att_gt_panel)
    lp.add_function(drdid.att_dr_panel)
    lp.enable_by_count()
    start = time.perf_counter()
    bmark_fn(f"{file}")
    end = time.perf_counter()
    lp.print_stats()
    print(f"\nTotal time: {end - start:.4f} seconds")


if __name__ == "__main__":
    cli()
