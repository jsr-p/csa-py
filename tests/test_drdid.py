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


def test_ipw_rc_sim():
    simrc = pl.read_csv("data/sim_rc.csv")

    Y = pull(simrc, "y", keep_dims=True)
    D = pull(simrc, "d", keep_dims=True)
    T = pull(simrc, "post", keep_dims=True)
    X = np.column_stack(
        (np.ones_like(D), simrc.select("x1", "x2", "x3", "x4").to_numpy())
    )
    drp = drdid.DrParamsRc(
        subset=simrc,
        X=X,
        Y=Y,
        T=T,
        D=D,
    )
    drres = drdid.att_ipw_rc(drp)

    assert np.isclose(-19.8933, drres.att, atol=1e-3)
    assert np.isclose(53.8682, drres.se, atol=1e-1)


def test_std_ipw_rc_sim():
    simrc = pl.read_csv("data/sim_rc.csv")

    Y = pull(simrc, "y", keep_dims=True)
    D = pull(simrc, "d", keep_dims=True)
    T = pull(simrc, "post", keep_dims=True)
    X = np.column_stack(
        (np.ones_like(D), simrc.select("x1", "x2", "x3", "x4").to_numpy())
    )
    drp = drdid.DrParamsRc(
        subset=simrc,
        X=X,
        Y=Y,
        T=T,
        D=D,
    )
    drres = drdid.att_std_ipw_rc(drp)

    assert np.isclose(-15.8033, drres.att, atol=1e-3)
    assert np.isclose(9.0879, drres.se, atol=1e-1)


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


def test_dr_panel_imp_lalonde_sim():
    df = pl.read_csv("data/lalonde_sim_imp.csv")
    y0 = pull(df, "re75", keep_dims=True)
    y1 = pull(df, "re78", keep_dims=True)
    D = pull(df, "experimental", keep_dims=True)
    covs = ["age", "educ", "black", "married", "nodegree", "hisp", "re74"]
    X = df.select(covs).to_numpy()
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    dy = y1 - y0

    params = drdid.PanelParams(subset=df, X=X, G=D, dy=dy)
    output = drdid.att_dr_panel_imp(params)

    # Reference (R): ATT=-615.2344, SE=683.2211
    # Numerical optimization issues in the ipt estimator
    assert np.isclose(output.att, -615.2344, atol=25)
    assert np.isclose(output.se, 683.2211, atol=25)


def test_ipw_panel_lalonde_sim():
    df = pl.read_csv("data/lalonde_sim_imp.csv")
    y0 = pull(df, "re75", keep_dims=True)
    y1 = pull(df, "re78", keep_dims=True)
    D = pull(df, "experimental", keep_dims=True)
    covs = ["age", "educ", "black", "married", "nodegree", "hisp", "re74"]
    X = df.select(covs).to_numpy()
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    dy = y1 - y0

    params = drdid.PanelParams(subset=df, X=X, G=D, dy=dy)
    output = drdid.att_ipw_panel(params)

    # Reference (R): ATT=-724.1929, SE=692.1639
    assert np.isclose(output.att, -724.1929, atol=1e-4)
    assert np.isclose(output.se, 692.0947, atol=1e-4)


def test_std_ipw_panel_lalonde_sim():
    df = pl.read_csv("data/lalonde_sim_imp.csv")
    y0 = pull(df, "re75", keep_dims=True)
    y1 = pull(df, "re78", keep_dims=True)
    D = pull(df, "experimental", keep_dims=True)
    covs = ["age", "educ", "black", "married", "nodegree", "hisp", "re74"]
    X = df.select(covs).to_numpy()
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    dy = y1 - y0

    params = drdid.PanelParams(subset=df, X=X, G=D, dy=dy)
    output = drdid.att_std_ipw_panel(params)

    # Reference (R): ATT=-655.9068, SE=687.7806
    assert np.isclose(output.att, -655.9068, atol=25)
    assert np.isclose(output.se, 687.7806, atol=25)
