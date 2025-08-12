"""
Simulate some test data for the CSA package.
"""

import itertools as it
import click
import numpy as np
import polars as pl

from csa.sim import SimParams, build_sim_csa, plot_sim
from csa import testing


@click.group()
def cli():
    pass


@cli.command()
def simplot():
    sp = SimParams(
        start=1,
        end=6,
        groups=[0] + list(range(2, 5)),
        N=100,
        xvar=True,
        balanced=True,
        delta=3,
    )
    np.random.seed(42)
    data = build_sim_csa(sp)
    plot = plot_sim(data, sp)
    plot.save("figs/sim_data.png")


@cli.command()
def simub():
    np.random.seed(42)
    sp = SimParams(
        start=1,
        end=(e := 12),
        delta=(delta := 3),
        groups=list(range(4, e - delta + 1)),
        N=1_000,
        xvar=True,
        balanced=False,
    )
    data = build_sim_csa(sp)
    data.write_csv(testing.FP_DATA / "sim_ub.csv")


@cli.command()
def simub_plot():
    sp = SimParams(
        start=1,
        end=(e := 12),
        delta=(delta := 3),
        groups=list(range(4, e - delta + 1)),
        N=1_000,
        xvar=True,
        balanced=False,
    )
    data = pl.read_csv(testing.FP_DATA / "sim_ub.csv")
    plot = plot_sim(data, sp)
    plot.save("figs/sim_data_ub.png", width=8, height=8)

    for c in ["t", "K", "g"]:
        print(data.group_by(c).agg(pl.len()).sort(c))

    gp = data.group_by("g").agg(pl.col("t").unique().sort()).sort("g")
    for g, t in zip(gp["g"], gp["t"]):
        print(f"{g=}: {t.to_list()=}")


@cli.command()
def datasets():
    for N, end, delta in it.product(
        [1_000, 2_500, 5_000, 10_000, 50_000],
        [15, 30, 45],
        [3, 6, 9],
    ):
        print(f"Generating dataset with N={N}")
        np.random.seed(42)
        sp = SimParams(
            start=1,
            end=end,
            delta=delta,
            groups=list(range(4, end - delta + 1)),
            N=N,
            xvar=True,
            balanced=False,
        )
        data = build_sim_csa(sp)
        data.write_csv(testing.FP_DATA / f"sim_ub_{N}_{end}_{delta}.csv")
        print(f"Dataset with {N=}; {end=}; {delta=} saved.")


@cli.command()
def simub_large():
    np.random.seed(42)
    sp = SimParams(
        start=1,
        end=(e := 40),
        delta=(delta := 6),
        groups=list(range(4, e - delta + 1)),
        N=20_000,
        xvar=True,
        balanced=False,
    )
    data = build_sim_csa(sp)

    plot = plot_sim(data, sp)
    plot.save("figs/sim_data_ub_large.png", width=8, height=8)

    for c in ["t", "K", "g"]:
        print(data.group_by(c).agg(pl.len()).sort(c))

    data.write_csv(testing.FP_DATA / "sim_ub_large.csv")


if __name__ == "__main__":
    cli()
