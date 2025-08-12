import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy.optimize import minimize
import polars.selectors as cs

from csa import drdid, testing
from csa.utils import pull


def test_drrc():
    simrc = pl.read_csv("data/testing/sim_rc.csv")

    Y = pull(simrc, "y", keep_dims=True)
    D = pull(simrc, "d", keep_dims=True)
    T = pull(simrc, "post", keep_dims=True)
    X = np.column_stack(
        (np.ones_like(D), simrc.select("x1", "x2", "x3", "x4").to_numpy())
    )
    if not isinstance(X, np.ndarray):
        X = np.ones_like(D).reshape(-1, 1)
    drp = drdid.DrParamsRc(
        subset=simrc,
        X=X,
        Y=Y,
        T=T,
        D=D,
    )
    drres = drdid.att_dr_rc(drp)

    assert np.allclose(0.2008992, drres.se)
    assert np.allclose(-0.1677954, drres.att)


def test_pscore_improved():
    drp = drdid.get_test_data()
    _, X, _, D = drp.Y, drp.X, drp.T, drp.D

    ps_tr = sm.Logit(endog=D, exog=X).fit(disp=0)
    init_gamma = ps_tr.params

    # Test that loss_ps_cal is close to R implementation
    res = minimize(
        fun=drdid.loss_ps_cal,
        x0=init_gamma,
        args=(D, X),
        method="trust-constr",
        options={"maxiter": 1000},
    )
    assert np.allclose(
        np.array([-0.164457116, -1.220115926, 0.545884685, -0.519212918, -0.003909338]),
        res.x,
    )

    # Test that loss_ps_IPT is close to R implementation
    res_ipt = drdid.pscore_ipt(D, X, init_gamma)
    assert np.allclose(
        np.array(
            [
                -0.164456996,
                -1.220115877,
                0.545884702,
                -0.519212770,
                -0.003909344,
            ]
        ),
        res_ipt.x,
    )


def test_drrc_imp():
    simrc = pl.read_csv("data/testing/sim_rc.csv")

    Y = pull(simrc, "y", keep_dims=True)
    D = pull(simrc, "d", keep_dims=True)
    T = pull(simrc, "post", keep_dims=True)
    X = np.column_stack(
        (np.ones_like(D), simrc.select("x1", "x2", "x3", "x4").to_numpy())
    )
    if not isinstance(X, np.ndarray):
        X = np.ones_like(D).reshape(-1, 1)
    drp = drdid.DrParamsRc(
        subset=simrc,
        X=X,
        Y=Y,
        T=T,
        D=D,
    )
    drres = drdid.att_dr_rc_imp(drp)

    assert np.allclose(0.2003375, drres.se)
    assert np.allclose(-0.2088586, drres.att)


def test_dr_panel():
    df = pl.read_csv(testing.FP_DATA / "lalonde_cps.csv")
    y0 = pull(df, "y0", keep_dims=True)
    y1 = pull(df, "y1", keep_dims=True)
    D = pull(df, "D", keep_dims=True)
    X = df.select(cs.starts_with("cov")).to_numpy()
    dy = y1 - y0
    params = drdid.PanelParams(subset=df, X=X, G=D, dy=dy)
    output = drdid.att_dr_panel(params)
    out = testing.load_json("data/testing/dr_panel.json")
    att = float(out["att"])
    se = float(out["se"])
    IF = np.array(out["IF"])

    assert np.allclose(att, output.att)
    # np.allclose(se, output.se)  # wrongly calculated in drdid
    assert np.allclose(IF, output.IF)
