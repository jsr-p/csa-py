import numpy as np
from dataclasses import dataclass

import polars as pl
import numpy.random as npr
import plotnine as pn


@dataclass
class SimParams:
    start: int
    end: int
    groups: list[int]
    N: int
    balanced: bool = False
    xvar: bool = True
    delta: int = 3
    probs: list[float] | None = None

    """
    Parameters for simulation of a CSA dataset.

    Attributes:
        delta: num periods on each side of treatment
    """


def build_sim_csa(sp: SimParams):
    groups = sp.groups
    N = sp.N
    delta = sp.delta
    if not sp.probs:
        probs = np.ones_like(groups) / len(groups)
    else:
        probs = sp.probs
    groups = npr.choice(groups, p=probs, size=N)
    base = pl.DataFrame(
        {
            "g": groups,
            "id": np.arange(N),
            "eta": npr.normal(loc=groups, scale=1),
            "X": npr.normal(loc=groups > 0, scale=1),
        }
    )
    if sp.balanced:
        t_range = np.arange(sp.start, sp.end + 1)
        base = base.with_columns(t=t_range.tolist())
    else:
        base = (
            base.with_columns(
                start_panel=pl.col("g") - delta,
                end_panel=pl.col("g") + delta + 1,  # Inclusive
            )
            .with_columns(t=pl.int_ranges("start_panel", "end_panel"))
            .drop("start_panel", "end_panel")
        )

    def outcomes(df: pl.DataFrame):
        if sp.xvar:
            df = df.with_columns(
                y0=pl.col("eta") + pl.col("X") * pl.col("beta") + pl.col("v"),
            )
        else:
            df = df.with_columns(
                y0=pl.col("eta") + pl.col("v"),
            )
        return (
            df.with_columns(
                # post treat period (for treated)
                pt=pl.col("K").ge(0) * pl.col("treat")
            )
            .with_columns(
                # treatment effect
                yg=pl.col("y0")
                + pl.col("pt") * (pl.col("K") + 1)
                + pl.col("u")
                - pl.col("v"),
            )
            .with_columns(
                y=pl.col("pt") * pl.col("yg") + (1 - pl.col("pt")) * pl.col("y0")
            )
        )

    return (
        base.explode("t")
        .with_columns(
            K=pl.col("t") - pl.col("g"),
            treat=pl.col("g").gt(0).cast(pl.Int8),
        )
        .filter(
            # unbalanced case ensure periods are within time range.
            pl.col("t").is_between(sp.start, sp.end)
        )
        .pipe(
            lambda df: df.with_columns(
                theta=pl.col("t"),
                beta=pl.col("t"),
                v=pl.lit(npr.normal(size=df.shape[0])),
                u=pl.lit(npr.normal(size=df.shape[0])),
            )
        )
        .pipe(outcomes)
        .select("id", "pt", "g", "t", "K", "X", "y0", "y")
    )


def plot_sim(data: pl.DataFrame, sp: SimParams):
    gp = (
        data.group_by("g", "t")
        .agg(
            pl.col("y").mean(),
            pl.col("y0").mean(),
        )
        .sort("g", "t")
    )
    plot = (
        pn.ggplot(
            gp.cast({"g": str}).to_pandas(),
            pn.aes(x="t", y="y", group="g", color="g"),
        )
        + pn.geom_line()
        + pn.geom_point()
        + pn.geom_line(pn.aes(y="y0", group="g", color="g"), linetype="dashed")
        + pn.theme_classic()
        + pn.theme(text=pn.element_text(size=14), plot_caption=pn.element_text(size=12))
        + pn.ggtitle("Evolution of outcomes $Y_{it}$")
        + pn.labs(
            caption=(
                "Dashed lines show $Y_{it}(0)$"
                + "\nError terms differ between $Y_{it}(0)$ and $Y_{it}(g)$"
            )
        )
        + pn.scale_color_discrete(name="Group")
        + pn.scale_x_continuous(breaks=range(sp.start, sp.end + 1))
    )
    return plot
