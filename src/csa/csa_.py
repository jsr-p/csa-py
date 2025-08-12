"""
Partial implementation of the Callaway-Sant'Anna (CSA) estimator for
Difference-in-Differences (DiD) analysis.

Mainly coded to increase my own understanding of the method.
Lots of inspiration from the original R implementation:
https://github.com/bcallaway11/did/

author: jsr-p
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
import itertools as it
from copy import deepcopy
from functools import partial
from typing import Any, Literal, overload, TypeAlias, Union
from contextlib import nullcontext
from scipy.stats import norm
import numpy as np
import tabulate
import polars as pl
from tqdm import tqdm

from csa import drdid, multiplier
from csa.drdid import PanelParams, att_oreg_panel, att_dr_panel
from csa.multiplier import MBootRes
from csa.utils import pull, sorted_unqs
from csa.utils import summarize_fields

__all__ = [
    "estimate",
    "agg_te",
    "agg_te_custom_group",
    "ATT",
    "FailedATT",
    "get_controls",
]

ATTs: TypeAlias = dict[tuple[int, int], Union["ATT", "FailedATT"]]


def se_if(IF: np.ndarray):
    n = IF.shape[0]
    return np.sqrt((IF * IF / n).sum() / n)


def se_ifs(IFs: np.ndarray):
    nc = IFs.shape[1]
    return np.array([se_if(IFs[:, i]).item() for i in range(nc)])


def compute_se_if(IF: np.ndarray):
    n = IF.shape[0]
    cov = IF.T @ IF / n
    se = np.sqrt(np.diag(cov) / n)
    return se


def test_rel_probs(gp, g_col: str = "g"):
    p = pull(gp, "pg")
    K = pull(gp, "K")

    for k in np.unique(K):
        we = K == k
        pge = p[we] / p[we].sum()
        pge2 = pull(gp.filter(pl.col("K").eq(k)).sort(g_col), "pge")
        assert np.allclose(pge, pge2), f"failed {pge}; {pge2}"


@dataclass(repr=False)
class Periods:
    gs: np.ndarray
    ts: np.ndarray
    t_min: int
    t_max: int
    gr: np.ndarray
    tr: np.ndarray
    gts: list[tuple[int, list[int]]]
    g_max: int
    gts_map: dict[int, list[int]]
    any_never_treated: bool

    def csa_pairs(self):
        gts = list(it.product(self.gr.tolist(), self.tr.tolist()))
        return gts

    def total_gts(self):
        return sum(len(ts) for _, ts in self.gts)

    def __repr__(self):
        return summarize_fields(self)


def group_probs(
    df: pl.DataFrame,
    g_col: str = "g",
    i_col: str = "id",
):
    return (
        df.group_by(g_col)
        .agg(pl.col(i_col).n_unique().alias("count"))
        .with_columns(pg=pl.col("count") / pl.col("count").sum())
    )


def gt_count(
    df: pl.DataFrame,
    g_col: str = "g",
    t_col: str = "t",
):
    return df.group_by(g_col, t_col).agg(pl.len().alias("gt_count"))


def _cprobs_m(
    data: pl.DataFrame,
    pg: pl.DataFrame,
    t_col: str = "t",
    g_col: str = "g",
):
    treat = pl.col("treat")
    vt = treat & pl.col(g_col).le(pl.col(t_col))
    return (
        data.select(g_col, t_col)
        .unique()
        .sort(g_col, t_col)
        .join(pg, how="left", on=g_col)
        .with_columns(
            K=pl.col(t_col) - pl.col(g_col),
            treat=pl.col(g_col).gt(0),
        )
        .with_columns(
            pge=(pl.col("pg") / pl.col("pg").sum().over("K")) * pl.col("treat"),
            pgt=(pl.col("pg").mul(vt) / pl.col("pg").mul(vt).sum().over(t_col)),
        )
    )


def wif_csa(
    g_col: np.ndarray,
    g: np.ndarray,
    p: np.ndarray,
    which: np.ndarray,
):
    """IF for weights as in CSA."""
    ps = p[which].reshape(1, -1)
    gs = g[which].reshape(1, -1)
    fact = 1 / ps.sum().item()
    i_g = 1 * (g_col == gs)
    if1 = fact * (i_g - ps)
    if2 = (fact**2) * ps * (i_g - ps).sum(axis=1, keepdims=True)
    return if1 - if2


def wif_cp(
    g_col: np.ndarray,
    g: np.ndarray,
    p: np.ndarray,
    cp: np.ndarray,
    which: np.ndarray,
    i_c: np.ndarray,
):
    """IF for weights with conditional probability as estimand.

    Args:
        i_c: indicator for conditioning event of conditional prob.
    """
    ps = p[which].reshape(1, -1)
    gs = g[which].reshape(1, -1)
    fact = 1 / ps.sum().item()
    i_g = 1 * (g_col == gs)
    IF = i_c * fact * (i_g - cp[which])
    return IF


@dataclass(repr=False)
class CsaParams:
    att: np.ndarray
    gp: pl.DataFrame
    df: pl.DataFrame
    n: int
    if_att: np.ndarray
    g_col: str
    t_col: str
    y_col: str
    i_col: str
    control: Control = "never"
    method: Method = "reg"

    def __repr__(self):
        return summarize_fields(self)


def simple_csa(csap: CsaParams):
    """Compute the overall aggregate effect.

    Returns:
    """
    pg = pull(csap.gp, "pg")
    g = pull(csap.gp, csap.g_col)
    t = pull(csap.gp, csap.t_col)
    gcol = pull(csap.df, csap.g_col, keep_dims=True)
    att = csap.att
    IF_att = csap.if_att

    we = g <= t
    w = pg[we] / pg[we].sum()
    est = (att[we] * w).sum()
    IF_w = wif_csa(gcol, g, pg, which=we)
    IF_s = IF_att[:, we] @ w + IF_w @ att[we]
    se = se_if(IF_s)
    return AggTeResult(
        estimates=pl.DataFrame(
            {"effect": "overall", "att": est, "se": se}
        ).with_columns(pl_ci("att")),
        overall_att=est,
        overall_se=se,
        IF=IF_s,
        IF_overall=IF_s,
        params=csap,
        method="simple",
        IFs_weight=dict(single=IF_w),
    )


def dynamic_csa(csap: CsaParams):
    """Compute the dynamic aggregate effects.

    Returns:
    """
    k_vals = sorted_unqs(pull(csap.gp.select(pl.col("K")), "K")).tolist()
    K = pull(csap.gp, "K")
    pge = pull(csap.gp, "pge")
    p = pull(csap.gp, "pg")
    g = pull(csap.gp, csap.g_col)
    gcol = pull(csap.df, csap.g_col, keep_dims=True)
    att = csap.att
    IF_att = csap.if_att

    res = []
    IF_e = np.zeros(shape=(csap.n, len(k_vals)))
    IFs_w = dict()
    for i, k in enumerate(k_vals):
        we = K == k
        w = pge[we]
        i_c = np.max(g[we] == gcol, axis=1, keepdims=True)
        IF_w = wif_cp(gcol, g, p, pge, we, i_c=i_c)
        # IF_w2 = wif_csa(gcol, g, p, we)
        # assert np.allclose(IF_w2, IF_w)
        IF = IF_att[:, we]
        IF_k = IF @ w + IF_w @ att[we]
        se = se_if(IF_k).item()
        est = (att[we] @ pge[we]).item()
        res.append((k, est, se))
        IF_e[:, i] = IF_k
        IFs_w[k] = IF_w
    res = pl.DataFrame(res, schema=["k", "att", "se"], orient="row").with_columns(
        pl_ci("att")
    )
    ke = pull(res, "k") >= 0
    attk = pull(res, "att")[ke]
    est = attk.mean().item()
    se = se_if(IF_s := IF_e[:, ke].mean(axis=1)).item()
    return AggTeResult(
        estimates=res,
        overall_att=est,
        overall_se=se,
        IF=IF_e,
        IF_overall=IF_s,
        params=csap,
        method="dynamic",
        IFs_weight=IFs_w,
    )


def calendar_csa(csap: CsaParams):
    """Compute the calendar aggregate effects.

    Returns:
    """
    cp = pull(csap.gp, "pgt")  # conditonal t
    pg = pull(csap.gp, "pg")
    g = pull(csap.gp, csap.g_col)
    t = pull(csap.gp, csap.t_col)
    gcol = pull(csap.df, csap.g_col, keep_dims=True)
    att = csap.att
    IF_att = csap.if_att
    n = csap.n

    t_vals = sorted_unqs(t[t >= g.min()])

    res = []
    IF_e = np.zeros(shape=(n, len(t_vals)))
    IFs_w = dict()
    for i, tv in enumerate(t_vals):
        we = (t == tv) & (g <= tv)
        w = cp[we]
        # wip_cp fails in unbalanced case
        # i_c = (gcol <= tv) & (gcol != 0)
        # IF_w = wif_cp(gcol, g, pg, cp, which=we, i_c=i_c)
        IF_w = wif_csa(gcol, g, pg, which=we)
        IF = IF_att[:, we]
        IF_k = IF @ w + IF_w @ att[we]
        se = se_if(IF_k).item()
        est = (att[we] @ w).item()
        res.append((tv, est, se))
        IF_e[:, i] = IF_k
        IFs_w[tv] = IF_w
    res = pl.DataFrame(
        res, schema=[csap.t_col, "att", "se"], orient="row"
    ).with_columns(pl_ci("att"))
    attk = pull(res, "att")
    est = attk.mean().item()
    se = se_if(IF_s := IF_e.mean(axis=1)).item()
    return AggTeResult(
        estimates=res,
        overall_att=est,
        overall_se=se,
        IF=IF_e,
        IF_overall=IF_s,
        params=csap,
        method="calendar",
        IFs_weight=IFs_w,
    )


def group_csa(csap: CsaParams):
    """Compute the group aggregate effects.

    Returns:
    """
    g_col = csap.g_col
    g_vals = pull(csap.gp.select(pl.col(g_col).unique().sort()), g_col).tolist()
    cp = pull(csap.gp, "pgt")  # conditonal t
    t = pull(csap.gp, csap.t_col)
    att = csap.att
    IF_att = csap.if_att
    n = csap.n
    g = pull(csap.gp, g_col)
    gcol = pull(csap.df, g_col, keep_dims=True)

    res = []
    IF_e = np.zeros(shape=(n, len(g_vals)))
    for i, gv in enumerate(g_vals):
        we = (g == gv) & (t >= g)  # IF(g, t) for g = gv and t >= g.
        cp = np.ones_like(g) / sum(we)  # uniform over postperiods
        w = cp[we]
        # No weights estimator for this
        IF = IF_att[:, we]
        IF_k = IF @ w
        se = se_if(IF_k).item()
        est = att[we].mean().item()  # no weights
        res.append((gv, est, se))
        IF_e[:, i] = IF_k

    res = pl.DataFrame(
        res,
        schema=[g_col, "att", "se"],
        orient="row",
    ).with_columns(pl_ci("att"))
    pgg = pull(csap.gp.select(g_col, "pg").unique().sort(g_col), "pg")
    gg = pull(csap.gp.select(g_col).unique().sort(g_col), g_col)

    w_f = pgg / pgg.sum()  # rescale group probs to only include treated groups
    attk = pull(res, "att")
    est = (w_f * attk).sum()
    IF_w = wif_csa(gcol, gg, pgg, which=np.repeat(True, repeats=gg.shape[0]))
    IF_s = IF_e @ w_f + IF_w @ attk
    se = se_if(IF_s)
    return AggTeResult(
        estimates=res,
        overall_att=est,
        overall_se=se,
        IF=IF_e,
        IF_overall=IF_s,
        params=csap,
        method="group",
    )


Control = Literal["never", "notyet"]
Method = Literal["reg", "dr", "dr_imp"]
AggMethod = Literal["simple", "group", "calendar", "dynamic"]


@dataclass(repr=False)
class DidParams:
    data: pl.DataFrame
    g_col: str
    t_col: str
    y_col: str
    i_col: str
    n: int
    data_orig: pl.DataFrame  # For aggregation later
    units_map: dict[Any, int]
    covariates: list[str] | None = None
    balanced: bool = True
    control: Control = "never"
    method: Method = "reg"
    verbose: bool = True
    prelim_check: bool = False

    """Parameters for Callaway-Sant'Anna estimator."""

    def __repr__(self):
        return summarize_fields(self)


@dataclass
class GroupTimePair:
    g: int
    t: int
    pt: int


@dataclass(repr=False)
class ATT:
    att: float
    IF: np.ndarray
    IF2: np.ndarray
    g: int
    t: int
    ids: pl.DataFrame

    def __repr__(self):
        return summarize_fields(self, ignore=["IF2"])


@dataclass
class FailedATT:
    att: float
    g: int
    t: int
    reason: str = ""

    def __repr__(self):
        return summarize_fields(self)


def repr_res_atts(atts: ATTs | None = None):
    if not atts:
        return "ResAtts(0)"

    pairs = atts.values()

    num_failed = sum(isinstance(p, FailedATT) for p in pairs)
    num_attgt = sum(isinstance(p, ATT) for p in pairs)
    return f"ResAtts(#att={num_attgt}, #failed={num_failed})"


def nyt_controls(g_col: pl.Expr, pr: GroupTimePair):
    """
    Controls in the nyt case are:
    - Not-treated  (G inf.)
    - treated at some time greater than t (and observed at t)
      where t equals the post-treatment period.
    """
    t, g = pr.t, pr.g
    return g_col.eq(0) | (g_col.gt(t) & g_col.ne(g))


@dataclass
class CsaData:
    base: pl.DataFrame
    subset: pl.DataFrame


def get_base(dp: DidParams, pr: GroupTimePair):
    t_col = dp.t_col
    data = dp.data
    g_col = dp.g_col
    match dp.control:
        case "never":
            C = pl.col(g_col).eq(0)
        case "notyet":
            C = nyt_controls(pl.col(g_col), pr)
    base = data.filter(
        pl.col(t_col).eq(pr.pt).or_(pl.col(t_col).eq(pr.t))
    ).with_columns(
        C=C.cast(pl.Int8),
        G=pl.col(g_col).eq(pr.g).cast(pl.Int8),
    )
    return base


def subset_data(dp: DidParams, pr: GroupTimePair):
    """Subset data for a given group-time pair.

    We assume data is sorted over `i_col`, `t_col` in order for the `diff`
    operation to be correct.
    """
    base = get_base(dp, pr)
    if dp.balanced:
        y_col = dp.y_col
        t_col = dp.t_col
        base = (
            # much faster than
            base.with_columns(
                # y1 - y0
                dy=pl.col(y_col) - pl.col(y_col).shift(1)
            )
            # NOTE: this filter has to be done after the difference
            .filter(pl.col(t_col).eq(pr.t))
        )
        subset = base.filter(pl.col("C").eq(1).or_(pl.col("G").eq(1)))
        return CsaData(base=base, subset=subset)
    # unbalanced is treated as repeated cross-section;
    # thus `subset` actually redundant
    base = base.filter(pl.col("C").eq(1).or_(pl.col("G").eq(1)))
    return CsaData(base=base, subset=base)


@dataclass
class BaseCheck:
    gt: int
    gpt: int
    ct: int
    cpt: int
    controls: dict[str, list[int]]
    pr: GroupTimePair


@dataclass
class NytCheck:
    ok: bool
    bc: BaseCheck
    msg: str = ""


def base_check(base: pl.DataFrame, pr: GroupTimePair, dp: DidParams):
    t_col = dp.t_col
    G = pull(base, "G")
    C = pull(base, "C")
    t = pull(base, t_col)
    gt = ((G == 1) & (t == pr.t)).sum().item()
    gpt = ((G == 1) & (t == pr.pt)).sum().item()
    ct = ((C == 1) & (t == pr.t)).sum().item()
    cpt = ((C == 1) & (t == pr.pt)).sum().item()

    groups = pull(base, dp.g_col)
    controls = {
        "pt": sorted_unqs(groups[(C == 1) & (t == pr.pt)]).tolist(),
        "t": sorted_unqs(groups[(C == 1) & (t == pr.t)]).tolist(),
    }
    return BaseCheck(
        gt=gt,
        gpt=gpt,
        ct=ct,
        cpt=cpt,
        controls=controls,
        pr=pr,
    )


def check_nyt(base: pl.DataFrame, pr: GroupTimePair, dp: DidParams):
    """Computes check for not yet treated case."""
    #  TODO: Refactor this
    cases = np.array(["gt", "gpt", "ct", "cpt"], dtype=str)
    bc = base_check(base, pr, dp)
    checks = np.array([bc.gt, bc.gpt, bc.ct, bc.cpt], dtype=int)
    failed = cases[checks == 0]
    if failed.size == 0:
        return NytCheck(ok=True, msg="", bc=bc)
    msg = ""
    for c in failed:
        match c:
            case "gt":
                msg = f"no treated units at t for  (pt, t)={(pr.pt, pr.t)} for g={pr.g}"

            case "gpt":
                msg = f"no treated units at pt for (pt, t)={(pr.pt, pr.t)} for g={pr.g}"

            case "ct":
                msg = f"no control units at t for  (pt, t)={(pr.pt, pr.t)} for g={pr.g}"

            case "cpt":
                msg = f"no control units at pt for (pt, t)={(pr.pt, pr.t)} for g={pr.g}"
            case _:
                raise ValueError(f"Invalid case {c}")
    return NytCheck(ok=False, msg=msg, bc=bc)


def att_gt_panel(
    dp: DidParams,
    pr: GroupTimePair,
    method: Literal["reg", "dr"],
):
    """outcome regression panel"""
    i_col = dp.i_col
    covariates = dp.covariates
    g = pr.g
    t = pr.t

    match method:
        case "reg":
            drdid_fn = att_oreg_panel
        case "dr":
            drdid_fn = att_dr_panel
        case _:
            raise ValueError(f"Invalid method {method}")

    data = subset_data(dp, pr)
    base = data.base
    subset = data.subset

    n = base.shape[0]
    n1 = subset.shape[0]
    G = pull(subset, "G", keep_dims=True)
    X = drdid.get_xmat(subset, covariates=covariates)
    if dp.prelim_check:
        pc = drdid.PrelimCheck(subset=subset, G=G.ravel(), X=X, g=g, t=t)
        if check_failed := drdid.prelim_check(pc):
            print(f"Preliminary check failed: {check_failed}")

    res = drdid_fn(
        PanelParams(
            subset=subset,
            X=X,
            G=G,
            dy=pull(subset, "dy", keep_dims=True),
        )
    )
    IF = (n / n1) * res.IF
    return ATT(att=res.att, IF=IF, IF2=res.IF, g=g, t=t, ids=subset.select(i_col, "G"))


def att_gt_or_panel(dp: DidParams, pr: GroupTimePair):
    return att_gt_panel(dp, pr, method="reg")


def att_gt_dr_panel(dp: DidParams, pr: GroupTimePair):
    return att_gt_panel(dp, pr, method="dr")


def att_gt_or_rc(dp: DidParams, pr: GroupTimePair):
    """Computes te for OR in repeated cross-section case.

    In the rc case the IF is aggregated over units i.
    """
    data = subset_data(dp, pr)
    subset = data.subset
    T = pull(subset, "t", keep_dims=True)
    post = T == pr.t
    G = pull(subset, "G", keep_dims=True)
    Y = pull(subset, dp.y_col, keep_dims=True)
    X = drdid.get_xmat(subset, covariates=dp.covariates)
    orp = drdid.ORegParamsRc(subset=subset, X=X, Y=Y, post=post, G=G)
    ores = drdid.att_oreg_rc(orp)
    IF = (dp.n / subset.shape[0]) * ores.IF  # (n / n_1) * IF
    gp = (
        subset.select(dp.i_col, "G")
        .with_columns(IF=pl.lit(IF.ravel()))
        .group_by(dp.i_col, "G")
        .agg(pl.col("IF").sum())
        .sort(dp.i_col)
    )
    IF = pull(gp, "IF", keep_dims=True)
    return ATT(
        att=ores.att, IF=IF, IF2=IF, g=pr.g, t=pr.t, ids=gp.select(dp.i_col, "G")
    )


def att_gt_dr_rc(
    dp: DidParams,
    pr: GroupTimePair,
    method: Method = "dr",
):
    match method:
        case "dr":
            drdid_fn = drdid.att_dr_rc
        case "dr_imp":
            drdid_fn = drdid.att_dr_rc_imp
        case _:
            raise ValueError("Invalid method")
    data = subset_data(dp, pr)
    subset = data.subset
    T = pull(subset, dp.t_col, keep_dims=True)
    G = pull(subset, "G", keep_dims=True)
    Y = pull(subset, dp.y_col, keep_dims=True)
    X = drdid.get_xmat(subset, covariates=dp.covariates)
    drp = drdid.DrParamsRc(subset=subset, X=X, Y=Y, T=(T == pr.t), D=G)
    dres = drdid_fn(drp)
    IF = (dp.n / subset.shape[0]) * dres.IF  # (n / n_1) * IF
    gp = (
        subset.select(dp.i_col, "G")
        .with_columns(IF=pl.lit(IF.ravel()))
        .group_by(dp.i_col, "G")
        .agg(pl.col("IF").sum())
        .sort(dp.i_col)
    )
    IF = pull(gp, "IF", keep_dims=True)
    return ATT(
        att=dres.att, IF=IF, IF2=IF, g=pr.g, t=pr.t, ids=gp.select(dp.i_col, "G")
    )


def concat_ifs(dp: DidParams, res: ATTs):
    i_col = dp.i_col
    IFs = np.zeros(shape=(dp.n, len(res)))
    for j, r in enumerate(res.values()):
        if isinstance(r, FailedATT):
            IFs[:, j] = np.nan
            continue
        IFs[pull(r.ids, i_col), j] = r.IF.ravel()
    return IFs


def pl_ci(est_col: str = "att", c: float = norm.ppf(0.975).item()):
    return (
        (pl.col(est_col) - pl.col("se") * c).alias("lower"),
        (pl.col(est_col) + pl.col("se") * c).alias("upper"),
    )


def get_res_ov(
    res: ATTs,
    se: np.ndarray,
    g_col: str = "g",
    t_col: str = "t",
):
    return pl.DataFrame(
        [(r.g, r.t, r.att, se_i) for r, se_i in zip(res.values(), se)],
        schema=[g_col, t_col, "att", "se"],
        orient="row",
    ).with_columns(pl_ci("att"))


def get_gts_map(gr: np.ndarray, tr: np.ndarray, dp: DidParams):
    """Constructs map of each group g to relevant time periods t."""
    gp = (
        dp.data.filter(
            pl.col(dp.g_col).is_in(gr),
            pl.col(dp.t_col).is_in(tr),
        )
        .group_by(dp.g_col)
        .agg(pl.col(dp.t_col).unique())
        .sort(dp.g_col)
    )
    gts_map = OrderedDict(
        {g: sorted(ts.to_list()) for g, ts in zip(gp[dp.g_col], gp[dp.t_col])}
    )
    return gts_map


def prep_periods(dp: DidParams):
    gs = sorted_unqs(pull(dp.data, dp.g_col))
    ts = sorted_unqs(pull(dp.data, dp.t_col))
    t_min = ts.min()
    t_max = ts.max()
    # relevant groups; treated at some date that is not the first period
    gr = gs[(gs > 0) & (gs > t_min)]
    tr = ts[ts > t_min]  # Y_{t} - Y_{t-1} only defined for t > t_min.
    g_max = gr.max()
    any_never_treated = (gs == 0).any()
    if dp.control == "never" and not any_never_treated:
        raise ValueError(
            "No never-treated group found in data; "
            "set control='notyet' to allow for never-treated groups. "
            "We assume `g_col` equals 0 for the never-treated group"
        )
    if dp.control == "notyet" and not any_never_treated:
        # Don't compute ATT(g, t) for last treated group (is control group)
        # Also drop time periods with t >= max(G) as we have no c-group there
        gr = gr[gr < g_max]
        tr = tr[tr < g_max]
    gts_map = get_gts_map(gr, tr, dp)
    gts = sorted([(g, ts) for g, ts in gts_map.items()], key=lambda x: x[0])
    return Periods(
        gs=gs,
        ts=ts,
        t_min=t_min,
        t_max=t_max,
        gr=gr,
        tr=tr,
        gts=gts,
        gts_map=gts_map,
        g_max=g_max,
        any_never_treated=any_never_treated,
    )


def prep_data(dp: DidParams, p: Periods):
    # map units to integers for easy indexing
    dp.data = dp.data.with_columns(
        pl.col(dp.i_col).replace_strict(dp.units_map).alias(dp.i_col)
    )
    if dp.control == "notyet":
        if not p.any_never_treated:
            # filter out periods with t >= max(G) if no never-treated group
            dp.data = dp.data.filter(pl.col(dp.t_col).lt(p.g_max))
        dp.data = dp.data.with_columns(
            # groups with G larger than max(t) are considered never-treated.
            pl.when(pl.col(dp.g_col) > p.t_max)
            .then(pl.lit(0))
            .otherwise(pl.col(dp.g_col))
            .alias(dp.g_col)
        )
    # Sort once here; e.g. `get_base` assumes sorted data.
    dp.data = dp.data.sort(dp.i_col, dp.t_col)
    return dp


@dataclass
class CsaDidResult:
    """
    Results of estimating the Callaway-Sant'Anna estimator.

    Attributes:
        estimates: DataFrame with the estimated ATT(g, t) for each group-time pair.
        IF: Influence function matrix for the estimates.
        n: Number of units in the dataset.
        periods: Periods object containing information about the periods.
        dp: DidParams object containing the parameters used for estimation.
        units: DataFrame with the unique units in the dataset.
        atts: OrderedDict with the ATT(g, t) results for each group-time pair.
        fail_msg: Message indicating any failures in the estimation process.
    """

    estimates: pl.DataFrame
    IF: np.ndarray
    n: int
    periods: Periods
    dp: DidParams
    units: pl.DataFrame
    atts: ATTs
    fail_msg: str = ""

    def __repr__(self):
        return summarize_fields(
            self,
            ignore=["fail_msg", "dp", "periods", "units"],
            custom_repr_fns={
                # "atts": lambda x: f"atts={repr_res_atts(x)}",
                # "periods": lambda x: f"periods=Periods(g_max={x.g_max}, total_gts={x.total_gts()})",
            },
        )

    def summary(self):
        summarize_gt(self)


def cprobs_csa(res: CsaDidResult):
    """Compute probabilities for weighting ala. CSA.

    NOTE:
        - the group probabilities are computed with the full data
          (i.e. also with the possibly excluded groups)
    """
    dp = res.dp
    data = res.dp.data_orig  # need the original data before filtering
    data_p = data.join(
        res.estimates.select(dp.g_col, dp.t_col), how="inner", on=[dp.g_col, dp.t_col]
    )
    pg = group_probs(data, g_col=dp.g_col, i_col=dp.i_col)
    cprobs = _cprobs_m(data_p, pg, g_col=dp.g_col, t_col=dp.t_col)
    return cprobs


def compute_att_gt(dp: DidParams, pr: GroupTimePair) -> ATT | FailedATT:
    """Computes the ATT(g, t) for the given group-time pair.

    Selects the correct data and method based on the parameters.
    """
    match dp.control, dp.method, dp.balanced:
        case _, method, True:
            match method:
                case "reg":
                    return att_gt_or_panel(dp, pr)
                case "dr":
                    return att_gt_dr_panel(dp, pr)
                case _:
                    raise ValueError("Invalid method")
        case "notyet", method, False:
            # Unbalanced case is repeated cross-section
            match method:
                case "reg":
                    fn = att_gt_or_rc
                case "dr" | "dr_imp":
                    fn = partial(att_gt_dr_rc, method=method)
                case _:
                    raise ValueError("Invalid method")
            dpair = subset_data(dp, pr)
            base = dpair.base
            check = check_nyt(base, pr, dp)
            if check.ok:
                att_gt = fn(dp, pr)
                return att_gt
            return FailedATT(att=np.nan, g=pr.g, t=pr.t, reason=check.msg)
        case _:
            raise ValueError(
                "Invalid control or method; "
                f"{dp.control=}, {dp.method=}, balanced={dp.balanced=}"
            )


def get_failed_msg(res: ATTs):
    failed = [r for r in res if isinstance(r, FailedATT)]
    if len(failed) == 0:
        return ""
    msg = "\n".join([f"ATT(g={r.g}, t={r.t}): {r.reason}" for r in failed])
    return "Failed ATTs:\n" + msg


def get_pt_period(g: int, t: int) -> int:
    if g <= t:  # post-treatment period
        return g - 1
    return t - 1


def compute_all(dp: DidParams, periods: Periods):
    res: ATTs = OrderedDict()
    gts = [(g, ts) for g, ts in periods.gts_map.items()]
    with (
        tqdm(desc="Computing ATT(g, t)", total=sum(len(ts) for _, ts in gts))
        if dp.verbose
        else nullcontext() as pbar
    ):
        for g, ts in gts:
            for t in ts:
                if pbar:
                    pbar.set_postfix_str(f"g={g}, t={t}")
                att_gt = compute_att_gt(
                    dp,
                    GroupTimePair(g=g, t=t, pt=get_pt_period(g, t)),
                )
                res[(g, t)] = att_gt
                if pbar:
                    pbar.update(1)
    return res


def att_gt_all(dp: DidParams):
    periods = prep_periods(dp)
    dp = prep_data(dp, periods)
    res = compute_all(dp, periods)
    IF = concat_ifs(dp, res)
    att_res = get_res_ov(res, compute_se_if(IF), dp.g_col, dp.t_col)
    return CsaDidResult(
        estimates=att_res,
        IF=IF,
        n=dp.n,
        periods=periods,
        dp=dp,
        units=dp.data.select(dp.i_col, dp.g_col).unique().sort(dp.i_col),
        atts=res,
        fail_msg=get_failed_msg(res),
    )


def units_map(data: pl.DataFrame, i_col: str) -> dict[Any, int]:
    units = data.select(pl.col(i_col).unique()).with_row_index(name="id_int")
    mapping = dict(zip(units[i_col], units["id_int"]))
    return mapping


def estimate(
    data: pl.DataFrame,
    outcome: str,
    group: str,
    time: str,
    unit: str,
    covariates: list[str] | None = None,
    balanced: bool = True,
    control: Literal["never", "notyet"] = "never",
    method: Method = "reg",
    verbose: bool = True,
) -> CsaDidResult:
    """Estimate Callaway-Sant'Anna Difference-in-Differences estimator.

    Args:
        data: DataFrame with the data.
        outcome: Name of the outcome variable.
        group: Name of the group variable (treatment indicator).
        time: Name of the time variable.
        unit: Name of the unit identifier.
        covariates: List of covariates to include in the model.
        balanced: Whether the data is balanced (default: True).
        control: Type of control group to use ("never" or "notyet").
        method: Method to use for estimation ("reg", "dr", "dr_imp").
            - The `dr_imp` is an improved doubly-robust estimator; the default
            in the R package `did` is to *not* use this but just the vanilla
            `dr` estimator.
        verbose: Whether to print progress messages.

    Returns:
        CsaDidResult: The result of the estimation containing the ATT estimates,
        influence functions, and other relevant information.
    """
    return att_gt_all(
        DidParams(
            data=data,
            g_col=group,
            t_col=time,
            y_col=outcome,
            i_col=unit,
            covariates=covariates,
            n=data.select(pl.col(unit).n_unique()).item(),
            balanced=balanced,
            control=control,
            method=method,
            data_orig=data.select(group, time, outcome, unit).clone(),
            verbose=verbose,
            units_map=units_map(data, unit),
        )
    )


def inspect_controls(res: CsaDidResult):
    dp = res.dp
    periods = res.periods
    gts = periods.gts
    data = []
    for g, ts in gts:
        for t in ts:
            if g <= t:  # post-treatment period
                pt = g - 1
            else:
                pt = t - 1
            pr = GroupTimePair(g=g, t=t, pt=pt)
            dpair = subset_data(dp, pr)
            base = dpair.base
            check = check_nyt(base, pr, dp)
            data.append((g, t, check.bc.controls["pt"], check.bc.controls["t"]))
    return pl.DataFrame(data, schema=["g", "t", "c_pt", "c_t"], orient="row")


@dataclass
class BootRes:
    estimates: pl.DataFrame
    estimates_overall: pl.DataFrame
    aggregate: MBootRes
    overall: MBootRes

    def __repr__(self):
        return summarize_fields(self)


@dataclass
class AggTeResult:
    """Aggregate Treatment Effect Results.

    estimates: DataFrame with aggregated treatment effects.
    overall_att: Overall average treatment effect.
    overall_se: Standard error of the overall average treatment effect.
    IF: Influence function for the aggregated treatment effects.
    IF_overall: Influence function for the overall average treatment effect.
    params: CsaParams with parameters used for the aggregation.
    method: Aggregation method used (e.g., "simple", "group", "calendar", "dynamic").
    IFs_weight: IF of the weight of each aggregate effect.
    """

    estimates: pl.DataFrame
    overall_att: float
    overall_se: float
    IF: np.ndarray
    IF_overall: np.ndarray
    method: AggMethod
    IFs_weight: dict[Any, np.ndarray] = field(default_factory=dict, kw_only=True)
    params: CsaParams

    def __repr__(self):
        return summarize_fields(
            self,
            ignore=["fail_msg", "params"],
        )

    def summary(self):
        summarize_agg_te(self)


@dataclass
class AggTeResultBoot(AggTeResult):
    """Aggregate Treatment Effect Results Bootstrapped."""

    boot: BootRes

    def __repr__(self):
        return summarize_fields(self)


def get_col(method: str, csap: CsaParams):
    match method:
        case "group":
            return csap.g_col
        case "calendar":
            return csap.t_col
        case "dynamic":
            # relative time is denoted by `k`
            return "k"
        case _:
            raise ValueError(f"Invalid method: {method}")


def boot_agg_te(
    agg_res: AggTeResult,
    B: int = 1000,
    verbose: bool = False,
):
    """Computes bootstrapped aggregate results.

    The IF of the summary
    """
    # Bootstrap overall aggregate effect
    boot_res_overall = multiplier.boot(
        agg_res.IF_overall.reshape(agg_res.params.n, -1),
        B=B,
        verbose=verbose,
    )
    res_overall = multiplier.confidence_intervals(
        boot_res_overall, np.array([agg_res.overall_att])
    )
    if agg_res.method == "simple":
        return BootRes(
            # overall aggregate and aggregate effects the sample for
            # `simple` case
            estimates=res_overall,
            estimates_overall=res_overall,
            aggregate=boot_res_overall,
            overall=boot_res_overall,
        )

    # Bootstrap aggregate effects
    boot_res = multiplier.boot(
        agg_res.IF,
        B=B,
        verbose=verbose,
    )
    res = (
        multiplier.confidence_intervals(boot_res, pull(agg_res.estimates, "att"))
        .with_columns(
            # get name column
            pl.Series(
                pull(
                    agg_res.estimates,
                    col := get_col(agg_res.method, agg_res.params),
                )
            ).alias(col),
        )
        .select(col, "att", "se", "lower", "upper")
    )
    return BootRes(
        estimates=res,
        estimates_overall=res_overall,
        aggregate=boot_res,
        overall=boot_res_overall,
    )


@overload
def agg_te(
    res: CsaDidResult,
    method: AggMethod = "simple",
    boot: Literal[False] = False,
    B: int = 1000,
    verbose: bool = False,
) -> AggTeResult: ...


@overload
def agg_te(
    res: CsaDidResult,
    method: AggMethod,
    boot: Literal[True],
    B: int = 1000,
    verbose: bool = False,
) -> AggTeResultBoot: ...


def agg_te(
    res: CsaDidResult,
    method: AggMethod = "simple",
    boot: bool = False,
    B: int = 1000,
    verbose: bool = False,
) -> AggTeResult | AggTeResultBoot:
    """Aggregate treatment effects into summary measure.

    Args:
        res: CSA results object.
        method: Aggregation method to use.
            - "simple": Simple average of the treatment effects.
            - "group": Aggregate treatment effects by group.
            - "calendar": Aggregate treatment effects by calendar time.
            - "dynamic": Aggregate treatment effects by dynamic time
                         (relative to treatment).
        boot: Whether to bootstrap the aggregated treatment effects.
        B: Number of bootstrap samples to use if boot is True.
        verbose: Whether to print progress messages.

    Returns:
        AggTeResult or AggTeResultBoot: The aggregated treatment effect
        results.
    """
    atts = res.estimates
    IF = res.IF
    dp = res.dp
    notna = atts.select(pl.col("att").is_not_nan()).to_numpy().ravel()
    atts = atts.filter(notna).with_columns(K=pl.col(dp.t_col) - pl.col(dp.g_col))
    res = deepcopy(res)  # hacky fix
    res.estimates = atts
    gp = cprobs_csa(res)
    csap = CsaParams(
        att=pull(atts, "att"),
        gp=gp,
        df=res.units,
        n=res.n,
        if_att=IF[:, notna],
        i_col=dp.i_col,
        g_col=dp.g_col,
        t_col=dp.t_col,
        y_col=dp.y_col,
        control=dp.control,
        method=dp.method,
    )

    match method:
        case "simple":
            agg_res = simple_csa(csap)
        case "group":
            agg_res = group_csa(csap)
        case "calendar":
            agg_res = calendar_csa(csap)
        case "dynamic":
            agg_res = dynamic_csa(csap)

    if boot:
        return AggTeResultBoot(
            **{k: v for k, v in asdict(agg_res).items() if k != "params"},
            boot=boot_agg_te(agg_res, B=B, verbose=verbose),
            params=agg_res.params,
        )
    return agg_res


def custom_group_csa(csap: CsaParams, cgroup: str):
    """
    Custom group aggregation for CSA.
    """
    cp = pull(csap.gp, "pgt")
    t = pull(csap.gp, csap.t_col)
    att = csap.att
    IF_att = csap.if_att
    n = csap.n

    cg = pull(csap.gp, cgroup)
    g_vals = sorted_unqs(pull(csap.gp, cgroup)).tolist()

    g = pull(csap.gp, csap.g_col)
    gcol = pull(csap.df, csap.g_col, keep_dims=True)

    res = []
    IF_e = np.zeros(shape=(n, len(g_vals)))
    # pbar = tqdm(desc="Aggregating...", total=len(g_vals))
    for i, gv in tqdm(enumerate(g_vals)):
        # pbar.set_postfix_str(f"g={gv}")
        pt = t >= g  # post treat
        we = (cg == gv) & pt  # IF(g, t) for group g = gv in post-period
        cp = np.ones_like(g) / sum(we)  # uniform over postperiods
        w = cp[we]
        # No weights estimator for this on this
        IF = IF_att[:, we]
        IF_k = IF @ w
        se = se_if(IF_k).item()
        est = att[we].mean().item()  # no weights
        res.append((gv, est, se))
        IF_e[:, i] = IF_k
    #     pbar.update(1)
    # pbar.close()

    res = pl.DataFrame(
        res, schema=[csap.g_col, "att", "se"], orient="row"
    ).with_columns(pl_ci("att"))
    pgg = pull(
        # aggregate over custom groups e.g. union of several cohorts.
        csap.gp.select(cgroup, "pg").unique().group_by(cgroup).sum(),
        "pg",
    )
    gg = pull(csap.gp.select(cgroup).unique().sort(cgroup), cgroup)

    w_f = pgg / pgg.sum()  # rescale group probs to only include treated groups
    attk = pull(res, "att")
    est = (w_f * attk).sum()
    IF_w = wif_csa(gcol, gg, pgg, which=np.repeat(True, repeats=gg.shape[0]))
    IF_s = IF_e @ w_f + IF_w @ attk
    se = se_if(IF_s)
    return AggTeResult(
        estimates=res,
        overall_att=est,
        overall_se=se,
        IF=IF_e,
        IF_overall=IF_s,
        params=csap,
        method="group",
    )


@overload
def agg_te_custom_group(
    res: CsaDidResult,
    custom_group_map: dict[int, str],
    boot: Literal[True] = True,
    B: int = 1000,
    verbose: bool = False,
) -> AggTeResultBoot: ...


@overload
def agg_te_custom_group(
    res: CsaDidResult,
    custom_group_map: dict[int, str],
    boot: Literal[False] = False,
    B: int = 1000,
    verbose: bool = False,
) -> AggTeResult: ...


CGROUP = "custom_group"


def agg_te_custom_group(
    res: CsaDidResult,
    custom_group_map: dict[int, str],
    boot: bool = False,
    B: int = 1000,
    verbose: bool = False,
) -> AggTeResult | AggTeResultBoot:
    """Custom group effect aggregation.

    Args:
        custom_group_map: full mapping from unit ids to custom group names.
    See `agg_te` for the other parameters.

    Returns:
        AggTeResult or AggTeResultBoot: The aggregated treatment effect
        results for the custom groups.
    """
    atts = res.estimates
    IF = res.IF
    dp = res.dp
    notna = atts.select(pl.col("att").is_not_nan()).to_numpy().ravel()
    atts = atts.filter(notna).with_columns(K=pl.col(dp.t_col) - pl.col(dp.g_col))
    res = deepcopy(res)  # hacky fix if already estimated object
    res.estimates = atts

    def add_cgroups(df: pl.DataFrame):
        """Map cohorts to the custom groups before aggregating."""
        return df.with_columns(
            pl.col(dp.g_col).replace_strict(custom_group_map).alias(CGROUP)
        )

    gp = cprobs_csa(res).pipe(add_cgroups)
    csap = CsaParams(
        att=pull(atts, "att"),
        gp=gp,
        df=res.units.pipe(add_cgroups),
        n=res.n,
        if_att=IF[:, notna],
        i_col=dp.i_col,
        g_col=dp.g_col,
        t_col=dp.t_col,
        y_col=dp.y_col,
        control=dp.control,
        method=dp.method,
    )
    agg_res = custom_group_csa(csap, cgroup=CGROUP)
    if boot:
        return AggTeResultBoot(
            **{k: v for k, v in asdict(agg_res).items() if k != "params"},
            boot=boot_agg_te(agg_res, B=B, verbose=verbose),
            params=agg_res.params,
        )
    return agg_res


def get_controls(att: ATT, res: CsaDidResult):
    """Selects controls for a given ATT and CsaDidResult.

    This helper remaps the unit ids back to the original values
    and returns the controls for the given ATT.
    The ids are remapped internally to allow for easy indexing into the
    influence functions.
    """
    return (
        res.dp.data_orig.select(res.dp.i_col, res.dp.g_col)
        .unique()
        .join(
            # pick out controls and remap ids to original ids
            att.ids.filter(pl.col("G").eq(0)).select(
                pl.col(res.dp.i_col).replace_strict(
                    {v: k for k, v in res.dp.units_map.items()}
                )
            ),
            how="inner",
            on=res.dp.i_col,
        )
    )


### --- Summary stuff --- ###


def add_stars(df: pl.DataFrame):
    return df.with_columns(
        stars=pl.when(pl.col("lower").is_nan() | pl.col("upper").is_nan())
        .then(pl.lit(""))
        .when(
            pl.col("lower").gt(0) & pl.col("upper").gt(0)
            | pl.col("lower").lt(0) & pl.col("upper").lt(0)
        )
        .then(pl.lit("*"))
        .otherwise(pl.lit(""))
    )


method_titles = {
    "dynamic": "Dynamic",
    "group": "Group",
    "calendar": "Time",
}
method_col_titles = {
    "dynamic": "Event time",
    "group": "Group",
    "calendar": "Time",
}
control_map = {
    "never": "Never treated",
    "notyet": "Not yet treated",
}
type_agg_map = {
    "dynamic": "event-study/dynamic",
    "group": "group/cohort",
    "calendar": "calendar time",
}
est_method_title = {
    "dr": "Doubly Robust",
    "dr_imp": "Doubly Robust (improved)",
    "reg": "Outcome Regression",
}


def summarize_gt(res: CsaDidResult):
    print("Group-Time Average Treatment Effects:")
    table_str_gt = tabulate.tabulate(
        res.estimates.pipe(add_stars).rows(),
        headers=[
            "Group",
            "Time",
            "ATT(g, t)",
            "Std. Error",
            "[95% Pointwise.",
            "Conf. Band]",
            "",
        ],
        floatfmt=(".0f", ".0f", ".4f", ".4f", ".4f", ".4f", "s"),
        tablefmt="plain",
    )
    print(table_str_gt)
    print("---")
    print("Signif. codes: `*' confidence band does not cover 0")
    print(f"Control group: {control_map.get(res.dp.control, '')}")
    print(f"Estimation Method: {est_method_title.get(res.dp.method, '')}")


def overall_table(overall_data: list[tuple[Any, ...]]):
    df_oa = pl.DataFrame(
        overall_data,
        schema=["att", "se", "lower", "upper"],
        orient="row",
    ).pipe(add_stars)
    return tabulate.tabulate(
        df_oa.rows(),
        headers=[
            "ATT",
            "Std. Error",
            "[95%  ",
            "Conf. Band]",
            "",
        ],
        floatfmt=(".4f", ".4f", ".4f", ".4f", "s"),
        tablefmt="plain",
    )


def summarize_agg_te(agg_te: AggTeResult | AggTeResultBoot):
    df = agg_te.estimates.pipe(add_stars)
    method = agg_te.method
    method_title = method_titles.get(method, method.capitalize())
    method_col_title = method_col_titles.get(method, method.capitalize())
    controls = control_map.get(agg_te.params.control, "")
    type_ = type_agg_map.get(method, method.capitalize())
    cint_left_title = (
        "[95% Simult." if isinstance(agg_te, AggTeResultBoot) else "[95% Pointwise"
    )
    if isinstance(agg_te, AggTeResultBoot):
        overall_data = agg_te.boot.estimates_overall.select(
            "att", "se", "lower", "upper"
        ).rows()
    else:
        overall_data = [
            (
                est := agg_te.overall_att,
                se := agg_te.overall_se,
                est - 1.96 * se,
                est + 1.96 * se,
            )
        ]
    table_str_oa = overall_table(overall_data)
    table_str = tabulate.tabulate(
        df.rows(),
        headers=[
            method_col_title,
            "Estimate",
            "Std. Error",
            cint_left_title,
            "Conf. Band]",
            "",
        ],
        floatfmt=(".0f", ".4f", ".4f", ".4f", ".4f", "s"),
        tablefmt="plain",
    )
    if method == "simple":
        print(table_str_oa)
    else:
        print(f"Overall summary of ATT's based on {type_} aggregation:")
        print(table_str_oa)
        print()
        print(f"{method_title} effects:")
        print(table_str)
        print("---")
        print("Signif. codes: `*' confidence band does not cover 0")
        print(f"Control group: {controls}")
        print(f"Estimation Method: {est_method_title.get(agg_te.params.method, '')}")
