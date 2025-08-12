"""
Module to compute multiplier bootstrap.
"""

from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
from csa.utils import summarize_fields

import numpy as np
import polars as pl
import scipy as sp
import tqdm


__all__ = ["boot", "confidence_intervals"]

KAPPA = (np.sqrt(5) + 1) / 2  # Mammen


def qnorm(p: float):
    """Quantile function for normal distribution."""
    return sp.stats.norm.ppf(p)


def to_arr(val):
    return pl.DataFrame(val).to_numpy()


def rademacher(n: int):
    return np.random.choice(
        [KAPPA, 1 - KAPPA],
        size=n,
        p=[1 - KAPPA / np.sqrt(5), KAPPA / np.sqrt(5)],
    )


@dataclass
class MBootRes:
    """
    samples: draws from the bootstrap procedure.
    """

    samples: np.ndarray
    se: np.ndarray
    c: float
    V: np.ndarray

    def __repr__(self):
        return summarize_fields(self)


def mult_boot(IF: np.ndarray, B: int, verbose: bool = False):
    r"""Multiplier bootstrap ala. CSA.

    Returns matrix of bootstrap draws drawn by perturbing the influence
    function, see the paper for details.
    """
    n, k = IF.shape
    bres_sim = np.zeros(shape=(B, k))
    with (
        tqdm.tqdm(desc="Bootstraping...", total=B)
        if verbose
        else nullcontext() as pbar,
    ):
        for b in range(B):
            if pbar:
                pbar.set_postfix_str(f"b={b}")
            V = rademacher(n)
            bres_sim[b] = V @ IF  # Inner product each col of IF with V
            if pbar:
                pbar.update(1)
    bres_sim = bres_sim / n  # E_n [V * IF]
    return np.sqrt(n) * bres_sim


def compute_b_sigma(bres: np.ndarray):
    r"""Computes \hat{\Sigma}^{1/2}(g, t)"""
    qdiff = np.quantile(bres, q=0.75, axis=0) - np.quantile(bres, q=0.25, axis=0)
    b_sigma = qdiff / (qnorm(0.75) - qnorm(0.25))
    return b_sigma


def boot(
    IF: np.ndarray,
    B: int = 1000,
    alpha: float = 0.05,
    verbose: bool = False,
) -> MBootRes:
    n = IF.shape[0]
    bres = mult_boot(IF, B, verbose=verbose)
    V = np.cov(bres, rowvar=False)
    b_sigma = compute_b_sigma(bres)
    b_t = np.abs(bres / b_sigma[None, :]).max(axis=1)  # Step 5
    crit_val = np.quantile(b_t, q=1 - alpha).item()
    se = b_sigma / np.sqrt(n)
    if crit_val >= 7:
        print(
            "Simultaneous critical value is arguably `too large' to be"
            "realible. This usually happens when number of observations per"
            "group is small and/or there is no much variation in outcomes."
        )

    return MBootRes(
        samples=bres,
        se=se,
        V=V,
        c=crit_val,
    )


def confidence_intervals(mb: MBootRes, atts: np.ndarray):
    return (
        pl.DataFrame({"att": atts})
        .with_columns(se=pl.Series(mb.se), c=mb.c)
        .with_columns(
            lower=pl.col("att") - pl.col("se") * pl.col("c"),
            upper=pl.col("att") + pl.col("se") * pl.col("c"),
        )
        .drop("c")
    )
