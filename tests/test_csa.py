from pathlib import Path

import numpy as np
import polars as pl
import pytest

import csa
from csa import utils, CsaDidResult
from csa.csa_ import CsaParams, pull
from csa.testing import helper_test_csa, assert_close, assert_cprobs


def test_aggregate_own():
    mpd = (
        pl.read_csv("data/mpdta.csv")
        .rename({"year": "t", "first.treat": "g"})
        .with_columns(K=pl.col("t") - pl.col("g"))
    )
    res = csa.estimate(
        data=mpd,
        group="g",
        time="t",
        outcome="lemp",
        unit="countyreal",
        covariates=None,
        balanced=True,
        control="never",
    )
    gp = csa.cprobs_csa(res)
    att_res = res.estimates
    IF = res.IF
    df = mpd.filter(pl.col("t").eq(2003))  # csa filters by t == 2
    csap = CsaParams(
        att=pull(att_res, "att"),
        gp=gp,
        df=df,
        n=res.n,
        if_att=IF,
        g_col="g",
        t_col="t",
        y_col="lemp",
        i_col="countyreal",
    )

    agg_data = utils.load_json(
        # Without xvariables; constant only.
        Path("data/testing/resoreg.json")
    )
    helper_test_csa(csap, agg_data)
    print("All ok")


def test_aggregate_own_othername():
    """rename to random columns"""
    time = "random_time"
    group = "random_group"
    unit = "random_unit"
    outcome = "random_outcome"
    mpd = (
        pl.read_csv("data/mpdta.csv")
        .rename(
            {
                "year": time,
                "first.treat": group,
                "countyreal": unit,
                "lemp": outcome,
            }
        )
        .with_columns(K=pl.col(time) - pl.col(group))
    )
    res = csa.estimate(
        data=mpd,
        group=group,
        time=time,
        outcome=outcome,
        unit=unit,
        covariates=None,
        balanced=True,
        control="never",
    )
    gp = csa.cprobs_csa(res)
    att_res = res.estimates
    IF = res.IF
    df = mpd.filter(pl.col(time).eq(2003))  # csa filters by t == 2
    csap = CsaParams(
        att=pull(att_res, "att"),
        gp=gp,
        df=df,
        n=res.n,
        if_att=IF,
        g_col=group,
        t_col=time,
        y_col=outcome,
        i_col=unit,
    )

    agg_data = utils.load_json(
        # Without xvariables; constant only.
        Path("data/testing/resoreg.json")
    )
    helper_test_csa(csap, agg_data)
    print("All ok")


def test_aggregate_own_xvars():
    mpd = (
        pl.read_csv("data/mpdta.csv")
        .rename({"year": "t", "first.treat": "g"})
        .with_columns(K=pl.col("t") - pl.col("g"))
    )
    df = mpd.filter(pl.col("t").eq(2003))  # csa filters by t == 2
    res = csa.estimate(
        data=mpd,
        group="g",
        time="t",
        outcome="lemp",
        unit="countyreal",
        covariates=["lpop"],
        balanced=True,
        control="never",
    )
    gp = csa.cprobs_csa(res)
    att_res = res.estimates
    IF = res.IF
    csap = CsaParams(
        att=pull(att_res, "att"),
        gp=gp,
        df=df,
        n=res.n,
        if_att=IF,
        g_col="g",
        t_col="t",
        y_col="lemp",
        i_col="countyreal",
    )

    agg_data = utils.load_json(
        # With xvariables
        Path("data/testing/resoregxvars.json")
    )
    helper_test_csa(csap, agg_data)
    print("All ok")


def test_balanced_mpdta():
    res = csa.estimate(
        data=pl.read_csv("data/mpdta.csv"),
        group="first.treat",
        time="year",
        outcome="lemp",
        unit="countyreal",
        covariates=None,
        balanced=True,
        control="never",
    )
    att_res = res.estimates
    IF = res.IF
    # test atts
    test_d = utils.load_json(
        Path("data/testing/att_oreg.json")
    )
    atts = np.array(test_d["att"])
    se_t = np.array(test_d["se"])

    ifcsaoreg = pl.read_csv(
        "data/testing/if_oreg.csv"
    ).to_numpy()
    assert np.allclose(ifcsaoreg, IF)

    assert np.allclose(pull(att_res, "att"), atts)
    assert np.allclose(pull(att_res, "se"), se_t)


@pytest.fixture
def unbalanced_oreg_result():
    np.random.seed(42)
    data = csa.build_sim_csa(
        csa.SimParams(
            start=1,
            end=(e := 12),
            delta=(delta := 3),
            groups=list(range(4, e - delta + 1)),
            N=1_000,
            xvar=True,
            balanced=False,
        )
    )
    csa_res = csa.estimate(
        data=data,
        group="g",
        time="t",
        outcome="y",
        unit="id",
        covariates=["X"],
        balanced=False,
        control="notyet",
    )
    return csa_res


def load_reference():
    resoreg = pl.read_csv("data/testing/resoregub.csv")
    resoreg = resoreg.with_columns(
        pl.all().replace({"NA": np.nan}).cast(pl.Float64),
    )
    return resoreg.drop_nans()


def test_att_estimates_match(unbalanced_oreg_result: CsaDidResult):
    csa_res = unbalanced_oreg_result
    res_ov = csa_res.estimates.drop_nans()
    resoreg = load_reference()
    assert res_ov.shape[0] == resoreg.shape[0]
    assert np.allclose(pull(res_ov, "att"), pull(resoreg, "out.att"))


def test_standard_errors_match(unbalanced_oreg_result: CsaDidResult):
    csa_res = unbalanced_oreg_result
    res_ov = csa_res.estimates.drop_nans()
    resoreg = load_reference()
    assert res_ov.shape[0] == resoreg.shape[0]
    e1 = pull(res_ov, "se")
    e2 = pull(resoreg, "out.se")
    assert np.allclose(e1, e2), f"{e1} != {e2}"


def test_if_match(unbalanced_oreg_result):
    csa_res = unbalanced_oreg_result
    test = pl.read_csv(
        "data/testing/if_oreg_all_new.csv"
    ).with_columns(
        pl.selectors.string().replace({"NA": None}).cast(pl.Float64),
    )
    IF = csa_res.IF
    atts = csa_res.estimates
    notna = pull(atts.select(pl.col("att").is_not_nan()), "att")
    atts = atts.filter(notna)
    IF = IF[:, notna]

    # Select only relevant (g, t) pairs; CSA code loops over G x T while we
    # only loop over observed (g, t)
    tups = atts.select("g", "t").rows()
    barr = [pair in tups for pair in csa_res.periods.csa_pairs()]
    IF_test = test.to_numpy()[:, barr]

    assert np.allclose(IF, IF_test)


def test_unbalanced_weighting():
    data = pl.read_csv("data/testing/sim_ub.csv")
    csa_res = csa.estimate(
        data=data,
        group="g",
        time="t",
        outcome="y",
        unit="id",
        covariates=["X"],
        balanced=False,
        control="notyet",
    )
    IF = csa_res.IF
    res_ov = csa_res.estimates
    resoreg = pl.read_csv("data/testing/resoregub.csv")
    resoreg = resoreg.with_columns(
        pl.all().replace({"NA": np.nan}).cast(pl.Float64),
    )
    resoreg = resoreg.drop_nans()
    res_ov = res_ov.drop_nans()
    assert resoreg.shape[0] == 18
    assert res_ov.shape[0] == 18
    assert np.allclose(pull(res_ov, "att"), pull(resoreg, "out.att"))
    assert np.allclose(
        e1 := pull(res_ov, "se"),
        e2 := pull(resoreg, "out.se"),
    )
    print("Not yet treated panel test passed.")

    # weighting
    dp = csa_res.dp
    pr = csa_res.periods
    atts = csa_res.estimates
    notna = atts.select(pl.col("att").is_not_nan()).to_numpy().ravel()
    atts = atts.filter(notna).with_columns(K=pl.col("t") - pl.col("g"))

    csa_res.estimates = atts
    gp = csa.cprobs_csa(csa_res)

    assert_cprobs(gp)

    # influence function are equal to R
    if_oreg = pl.read_csv("data/testing/if_oreg_all_new.csv")
    tups = atts.select("g", "t").rows()
    barr = [pair in tups for pair in csa_res.periods.csa_pairs()]
    assert_close(if_oreg[:, barr].to_numpy(), IF[:, notna])

    csap = CsaParams(
        # NOTE: should only have the estimated ATTs and corresponding IF
        att=pull(atts, "att"),
        gp=gp,
        # should be full df, no? no! should match length of IF
        df=csa_res.units,
        n=csa_res.n,
        if_att=IF[:, notna],
        i_col=dp.i_col,
        g_col=dp.g_col,
        t_col=dp.t_col,
        y_col=dp.y_col,
    )

    # NOTE: Test IFs equal
    agg_data = utils.load_json(
        Path("data/testing/resoregunbalanced.json")
    )
    dyn = csa.dynamic_csa(csap)
    cal = csa.calendar_csa(csap)
    gro = csa.group_csa(csap)
    sim = csa.simple_csa(csap)

    ifs = utils.load_json(
        Path("data/testing/resoreguballifs.json")
    )

    simple = np.column_stack([v for v in ifs["simple"].values()]).ravel()
    gsel_agg = np.column_stack([v for v in ifs["group"]["selective"].values()]).ravel()
    gsel = np.column_stack([v for v in ifs["group"]["select"].values()])
    call = np.column_stack([v for v in ifs["calendar"]["calendar"].values()]).ravel()
    cal_t = np.column_stack([v for v in ifs["calendar"]["calendar_t"].values()])
    dynn = np.column_stack([v for v in ifs["dynamic"]["dynamic"].values()]).ravel()
    dyn_e = np.column_stack([v for v in ifs["dynamic"]["dynamic_e"].values()])

    np.allclose(dyn.IF, dyn_e)
    np.allclose(dyn.IF_overall, dynn)
    np.allclose(cal.IF_overall, call)
    np.allclose(cal.IF, cal_t)
    np.allclose(gro.IF, gsel)
    np.allclose(gro.IF_overall, gsel_agg)
    np.allclose(sim.IF, simple)

    helper_test_csa(csap, agg_data)


def test_group_probs():
    data = (
        pl.read_csv("data/mpdta.csv")
        .rename({"year": "t", "first.treat": "g"})
        .with_columns(K=pl.col("t") - pl.col("g"))
    )
    res = csa.estimate(
        data=data,
        group="g",
        time="t",
        outcome="lemp",
        unit="countyreal",
        covariates=["lpop"],
        balanced=True,
        control="never",
    )

    gp = csa.cprobs_csa(res)
    att_res = res.estimates
    IF = res.IF
    df = data.filter(pl.col("t").eq(2003))  # csa filters by t == 2

    agg_data = utils.load_json(
        # With xvariables
        Path("data/testing/resoregxvars.json")
    )

    helper_test_csa(
        CsaParams(
            att=pull(att_res, "att"),
            gp=gp,
            df=df,
            n=res.n,
            if_att=IF,
            g_col="g",
            t_col="t",
            y_col="lemp",
            i_col="countyreal",
        ),
        agg_data,
    )


def tt():
    data = pl.read_csv("data/testing/sim_ub.csv")

    res = csa.estimate(
        data=data,
        group="g",
        time="t",
        outcome="y",
        unit="id",
        covariates=["X"],
        balanced=False,
        control="notyet",
        method="dr_imp",
    )
    csao = csa.agg_te(res, method="dynamic")

    res = csa.estimate(
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
    csao = csa.agg_te(res, method="dynamic")


def test_fastdid_equal():
    """test estimates from fastdid equals csa equals csa-py"""
    respy = pl.read_csv("data/testing/fastdidpy.csv")
    resrfd = pl.read_csv("data/testing/fastdid-fd.csv")
    resrcsa = pl.read_csv("data/testing/fastdid-csa.csv")

    assert np.allclose(pull(resrcsa, "g"), pull(resrfd, "g"))
    assert np.allclose(pull(resrcsa, "t"), pull(resrfd, "t"))

    assert np.allclose(pull(resrcsa, "att"), pull(resrfd, "att"))
    assert np.allclose(pull(respy, "att"), pull(resrfd, "att"))
    assert np.allclose(pull(resrcsa, "se"), pull(resrfd, "se"))
    assert np.allclose(pull(respy, "se"), pull(resrfd, "se"))
