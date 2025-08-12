from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import scipy.linalg as spl
import statsmodels.api as sm

from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize
from scipy.special import expit

from csa.utils import pull

import fastlr

__all__ = [
    "att_oreg_panel",
    "att_oreg_rc",
    "att_dr_rc",
    "att_dr_rc_imp",
    "att_dr_panel",
]


def ols(endog: np.ndarray, exog: np.ndarray):
    """Fit OLS model and return params."""
    model = sm.OLS(endog=endog, exog=exog)
    results = model.fit(disp=0)
    return results.params.reshape(-1, 1)


def logit(endog: np.ndarray, exog: np.ndarray):
    """Fit OLS model and return params."""
    model = sm.Logit(endog=endog, exog=exog).fit(disp=0)
    return model


@dataclass(repr=False)
class FastLogitWrap:
    beta: np.ndarray
    X: np.ndarray

    def predict(self):
        return expit(self.X @ self.beta)


def fast_logit(endog: np.ndarray, exog: np.ndarray):
    out = fastlr.fastlr(X=exog, y=endog)
    return FastLogitWrap(out.coefficients, exog)


def se_if(IF: np.ndarray):
    n = IF.shape[0]
    return np.sqrt((IF * IF / n).sum() / n)


def prelim_logit(y: np.ndarray, X: np.ndarray):
    mod = sm.Logit(endog=y, exog=X).fit(disp=0)
    preds = mod.predict()
    if max(preds) >= 0.999:
        return True
    return False


def check_reg(X: np.ndarray):
    if np.linalg.cond(X.T @ X) < np.finfo(float).eps:
        return True
    return False


@dataclass
class PrelimCheck:
    subset: pl.DataFrame
    G: np.ndarray
    X: np.ndarray
    g: int
    t: int


def prelim_check(pc: PrelimCheck) -> bool:
    g = pc.g
    t = pc.t
    if overlap_v := prelim_logit(y=pc.G, X=pc.X):
        print(f"Overlap condition violated for {g=} in {t}")
    X = pc.X[pc.G == 0, :]
    if reg_v := check_reg(X.T @ X):
        print(
            f"Not enough control units for group for {g=} in {t}"
            " to run specified regression"
        )
    return overlap_v or reg_v


def if_ols(eps: np.ndarray, X: np.ndarray, n: int):
    XtX = X.T @ X / n
    iXtX = spl.solve(XtX, np.eye(XtX.shape[0]))  # (X^T X)^{-1}
    if check_reg(XtX):
        raise ValueError("XtX is singular")
    if_ols_pre = X * eps @ iXtX
    return if_ols_pre


@dataclass
class ORegRes:
    att: float
    IF: np.ndarray


def get_xmat(subset: pl.DataFrame, covariates: list[str] | None = None):
    if covariates:
        cols = ["constant"] + covariates
    else:
        cols = ["constant"]
    X = subset.with_columns(constant=pl.lit(1)).select(cols).to_numpy()
    return X


@dataclass
class PanelParams:
    subset: pl.DataFrame
    X: np.ndarray
    G: np.ndarray
    dy: np.ndarray


@dataclass
class ORegAtt:
    att: float
    bhat: np.ndarray
    dy_c: np.ndarray
    dy: np.ndarray
    eta_t: float
    eta_c: float
    att_t: np.ndarray
    att_c: np.ndarray
    w_t: np.ndarray
    w_c: np.ndarray


def att_oreg_panel(orp: PanelParams):
    """ATT outcome regression estimator for panel data."""
    subset = orp.subset
    G = orp.G
    X = orp.X
    dy = pull(subset, "dy", keep_dims=True)

    barr = (G == 0).ravel()
    mr = sm.OLS(endog=dy[barr], exog=X[barr]).fit(disp=0)
    coef = mr.params.reshape(-1, 1)
    dy_c = X @ coef  # predicted dy i.e. Y_1 - Y_0
    w_t = G
    w_c = G
    att_t = w_t * dy
    att_c = w_c * dy_c
    eta_t = att_t.mean() / w_t.mean()
    eta_c = att_c.mean() / w_c.mean()  # average across n_treat
    att = eta_t - eta_c

    n = subset.shape[0]
    w_ols = 1 - G
    wols_x = w_ols * X  # pick out control rows; ols estimated on G == 0
    eps = dy - dy_c
    wols_ex = w_ols * eps * X
    xpx = wols_x.T @ X / n  # divide by full data not just n_treat?
    i_xpx = spl.solve(xpx, np.eye(xpx.shape[0]))
    if_ols = wols_ex @ i_xpx
    if_t = (att_t - w_t * eta_t) / w_t.mean()
    if_c1 = att_c - w_c * eta_c
    m1 = np.mean(w_c * X, axis=0).reshape(-1, 1)
    if_c2 = if_ols @ m1
    if_c = (if_c1 + if_c2) / w_c.mean()
    if_reg = if_t - if_c
    return DrDidRes(att=att, se=se_if(if_reg), IF=if_reg)


@dataclass
class ORegAttRc:
    att: float
    yh_pre: np.ndarray
    yh_post: np.ndarray
    eta_t_pre: float
    eta_t_post: float
    eta_c: float
    reg_att_treat_pre: np.ndarray
    reg_att_treat_post: np.ndarray
    reg_att_cont: np.ndarray
    w_t_pre: np.ndarray
    w_t_post: np.ndarray
    w_cont: np.ndarray


@dataclass
class ORegParamsRc:
    subset: pl.DataFrame
    X: np.ndarray
    Y: np.ndarray
    post: np.ndarray
    G: np.ndarray


def att_oreg_rc(orp: ORegParamsRc):
    """computes ATT for outcome regression estimator"""
    G = orp.G
    X = orp.X
    Y = orp.Y
    post = orp.post

    b = ((post == 0) & (G == 0)).ravel()
    coef_pre = sm.OLS(endog=Y[b], exog=X[b, :]).fit(disp=0).params.reshape(-1, 1)
    yh_pre = X @ coef_pre

    b = ((post == 1) & (G == 0)).ravel()
    coef_post = sm.OLS(endog=Y[b], exog=X[b, :]).fit(disp=0).params.reshape(-1, 1)
    yh_post = X @ coef_post

    w_t_pre = G * (1 - post)
    w_t_post = G * post
    w_cont = G

    reg_att_treat_pre = w_t_pre * Y
    reg_att_treat_post = w_t_post * Y
    reg_att_cont = w_cont * (yh_post - yh_pre)

    eta_t_pre = reg_att_treat_pre.mean() / w_t_pre.mean()
    eta_t_post = reg_att_treat_post.mean() / w_t_post.mean()
    eta_c = reg_att_cont.mean() / w_cont.mean()

    reg_att = (eta_t_post - eta_c) - eta_t_pre

    # IF

    n = orp.subset.shape[0]
    w_ols_pre = (1 - G) * (1 - post)
    w_ols_post = (1 - G) * post
    if_ols_pre = if_ols(eps=Y - yh_pre, X=w_ols_pre * X, n=n)
    if_ols_post = if_ols(eps=Y - yh_post, X=w_ols_post * X, n=n)

    # treat
    if_t_pre = (reg_att_treat_pre - w_t_pre * eta_t_pre) / w_t_pre.mean()
    it_t_post = (reg_att_treat_post - w_t_post * eta_t_post) / w_t_post.mean()
    if_treat = it_t_post - if_t_pre

    # control component
    M1 = np.mean(w_cont * X, axis=0).reshape(-1, 1)
    if_c1 = reg_att_cont - w_cont * eta_c
    if_c2_post = if_ols_post @ M1
    if_c2_pre = if_ols_pre @ M1
    if_control = (if_c1 + if_c2_post - if_c2_pre) / w_cont.mean()
    if_reg = if_treat - if_control

    return ORegRes(att=reg_att, IF=if_reg)


@dataclass
class DrDidRes:
    att: float
    se: float
    IF: np.ndarray


@dataclass
class ORegDrRc:
    m: Any
    params: np.ndarray
    pred: np.ndarray


@dataclass
class DrParamsRc:
    subset: pl.DataFrame
    X: np.ndarray
    Y: np.ndarray
    T: np.ndarray
    D: np.ndarray


def oreg(
    X: np.ndarray,
    Y: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray | None = None,
):
    if b.dtype != np.bool_:
        raise ValueError("Input array must be of boolean dtype.")
    if weights is not None:
        m = sm.WLS(endog=Y[b], exog=X[b, :], weights=weights[b], hasconst=True).fit(
            disp=0
        )
    else:
        m = sm.OLS(endog=Y[b], exog=X[b, :]).fit(disp=0)
    params = m.params.reshape(-1, 1)
    pred = m.predict(X).reshape(-1, 1)
    if np.isnan(params).any():
        raise ValueError(f"NaN in params {params}")
    return ORegDrRc(m=m, params=params, pred=pred)


def att_dr_rc(drp: DrParamsRc) -> DrDidRes:
    """Doubly-Robust repeated cross-section.

    Notes:
        I take everything as column vectors besides boolean indices!
    """

    Y, X, T, D = drp.Y, drp.X, drp.T, drp.D

    n = Y.shape[0]
    ps_tr = sm.Logit(endog=D, exog=X).fit(disp=0)
    ps_fit = np.minimum(ps_tr.predict(), 1 - 1e-6).reshape(-1, 1)
    W = ps_fit * (1 - ps_fit)

    # Control group pre-treat
    b = (D == 0) & (T == 0)
    oreg_c_pre = oreg(X, Y, b.ravel())
    # Control group post-treat
    b = (D == 0) & (T == 1)
    oreg_c_post = oreg(X, Y, b.ravel())
    # Treat group pre-treat
    b = (D == 1) & (T == 0)
    oreg_t_pre = oreg(X, Y, b.ravel())
    # Treat group post-treat
    b = (D == 1) & (T == 1)
    oreg_t_post = oreg(X, Y, b.ravel())

    # \mu_{0, Y}^{rc}
    mu_0y = T * oreg_c_post.pred + (1 - T) * oreg_c_pre.pred

    # weight numerators
    # w_{1, t}^{rc}
    w_t_pre = D * (1 - T)
    w_t_post = D * T
    # w_{0, t}^{rc}
    w_c_pre = ps_fit * (1 - D) * (1 - T) / (1 - ps_fit)
    w_c_post = ps_fit * (1 - D) * T / (1 - ps_fit)

    # components of \hat{\tau}^{dr, rc}_{1}
    res_tau1 = Y - mu_0y  # residual
    eta_t_pre = w_t_pre * res_tau1 / w_t_pre.mean()
    eta_t_post = w_t_post * res_tau1 / w_t_post.mean()
    eta_c_pre = w_c_pre * res_tau1 / w_c_pre.mean()
    eta_c_post = w_c_post * res_tau1 / w_c_post.mean()

    res_tau2_post = oreg_t_post.pred - oreg_c_post.pred
    eta_d_post = (D / D.mean()) * res_tau2_post
    eta_dt1_post = (D * T / (D * T).mean()) * res_tau2_post

    res_tau2_pre = oreg_t_pre.pred - oreg_c_pre.pred
    eta_d_pre = (D / D.mean()) * res_tau2_pre
    eta_dt0_pre = (D * (1 - T) / (D * (1 - T)).mean()) * res_tau2_pre

    # Estimator consists of 4 terms
    att_t_pre = eta_t_pre.mean()
    att_t_post = eta_t_post.mean()
    att_c_pre = eta_c_pre.mean()
    att_c_post = eta_c_post.mean()
    att_d_pre = eta_d_pre.mean()
    att_d_post = eta_d_post.mean()
    att_dt0_pre = eta_dt0_pre.mean()
    att_dt1_post = eta_dt1_post.mean()

    tau1 = (att_t_post - att_t_pre) - (att_c_post - att_c_pre)
    tau2_1 = att_d_post - att_dt1_post
    tau2_2 = att_d_pre - att_dt0_pre
    tau2 = tau1 + (tau2_1 - tau2_2)  # ATT estimator

    # *IF*

    # IF OLS k-dimensional matrix
    if_ols_c_pre = if_ols(eps=Y - oreg_c_pre.pred, X=(1 - D) * (1 - T) * X, n=n)
    if_ols_c_post = if_ols(eps=Y - oreg_c_post.pred, X=(1 - D) * T * X, n=n)
    if_ols_t_pre = if_ols(eps=Y - oreg_t_pre.pred, X=w_t_pre * X, n=n)
    if_ols_t_post = if_ols(eps=Y - oreg_t_post.pred, X=w_t_post * X, n=n)

    # IF of logreg
    score_ps = (D - ps_fit) * X
    L = cholesky(X.T @ (W * X))  # W = ps_fit * (1 - ps_fit)
    hess_ps = cho_solve((L, False), np.eye(X.shape[1])) * n
    if_ps = score_ps @ hess_ps  # IF of logreg; dim (n, k)

    # IF of treat compoment; \eta_{1}^{rc, 1}
    if_t_pre = eta_t_pre - w_t_pre * att_t_pre / w_t_pre.mean()
    if_t_post = eta_t_post - w_t_post * att_t_post / w_t_post.mean()

    # estimation effect beta
    m1_post = -1 * np.mean(w_t_post * T * X, axis=0) / w_t_post.mean()
    m1_pre = -1 * np.mean(w_t_pre * (1 - T) * X, axis=0) / w_t_pre.mean()
    m1_post = m1_post.reshape(-1, 1)
    m1_pre = m1_pre.reshape(-1, 1)

    # IF related to or
    if_t_or_post = if_ols_c_post @ m1_post
    if_t_or_pre = if_ols_c_pre @ m1_pre

    # IF of control compoment; \eta_{0}^{rc, 1}
    if_c_pre = eta_c_pre - w_c_pre * att_c_pre / w_c_pre.mean()
    if_c_post = eta_c_post - w_c_post * att_c_post / w_c_post.mean()

    # Estimation effect gamma hat; pscore
    m2_pre = np.mean(w_c_pre * (Y - mu_0y - att_c_pre) * X, axis=0) / w_c_pre.mean()
    m2_post = np.mean(w_c_post * (Y - mu_0y - att_c_post) * X, axis=0) / w_c_post.mean()
    if_c_ps = if_ps @ (m2_post - m2_pre).reshape(-1, 1)

    # estimation effect beta pre and post periods
    m3_post = -1 * np.mean(w_c_post * T * X / w_c_post.mean(), axis=0)
    m3_pre = -1 * np.mean(w_c_pre * (1 - T) * X / w_c_pre.mean(), axis=0)

    if_c_or_post = if_ols_c_post @ m3_post.reshape(-1, 1)
    if_c_or_pre = if_ols_c_pre @ m3_pre.reshape(-1, 1)

    # IF of adjustment terms
    if_e1 = eta_d_post - D * att_d_post / D.mean()
    if_e2 = eta_dt1_post - D * T * att_dt1_post / (D * T).mean()
    if_e3 = eta_d_pre - D * att_d_pre / D.mean()
    if_e4 = eta_dt0_pre - D * (1 - T) * att_dt0_pre / (D * (1 - T)).mean()
    if_e = (if_e1 - if_e2) - (if_e3 - if_e4)

    # estimation effect OR coefs.
    mom_post = np.mean((D / D.mean() - D * T / (D * T).mean()) * X, axis=0)
    mom_pre = np.mean(
        (D / D.mean() - D * (1 - T) / (D * (1 - T)).mean()) * X,
        axis=0,
    )
    if_or_post = (if_ols_t_post - if_ols_c_post) @ mom_post.reshape(-1, 1)
    if_or_pre = (if_ols_t_pre - if_ols_c_pre) @ mom_pre.reshape(-1, 1)

    # if estimation effect beta
    if_t_or = if_t_or_post + if_t_or_pre
    if_c_or = if_c_or_post + if_c_or_pre

    # if estimation effect OR coefs
    if_or = if_or_post - if_or_pre

    # IF treat component
    if_t = if_t_post - if_t_pre + if_t_or
    # IF control component
    if_c = if_c_post - if_c_pre + if_c_ps + if_c_or

    # IF tau1
    if_tau1 = if_t - if_c

    # IF tau2
    if_tau2 = if_tau1 + if_e + if_or

    # se_if(if_tau2)
    se_tau2 = if_tau2.std(ddof=1) / np.sqrt(n)  # how csa computes it

    return DrDidRes(
        att=tau2,
        se=se_tau2,
        IF=if_tau2,
    )


def get_test_data():
    simrc = pl.read_csv("data/testing/sim_rc.csv")

    Y = pull(simrc, "y", keep_dims=True)
    D = pull(simrc, "d", keep_dims=True)
    T = pull(simrc, "post", keep_dims=True)
    X = np.column_stack(
        (np.ones_like(D), simrc.select("x1", "x2", "x3", "x4").to_numpy())
    )
    if not isinstance(X, np.ndarray):
        X = np.ones_like(D).reshape(-1, 1)
    drp = DrParamsRc(
        subset=simrc,
        X=X,
        Y=Y,
        T=T,
        D=D,
    )
    return drp


def loss_ps_cal(
    gam: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
):
    """
    Loss function for estimation of the calibrated PS, using trust Based on Tan
    (2019).
    """
    gam = gam.reshape(-1, 1)
    ps_in = X @ gam
    exp_ps_in = np.exp(ps_in)
    val = -np.mean(np.where(D, ps_in, -exp_ps_in))
    return val


def loss_ps_IPT(gamma1: np.ndarray, n: int, D: np.ndarray, X: np.ndarray):
    """
    Loss function for estimation of the Bias reduced PS, based on Graham,
    Pinton and Egel (2012, 2016)
    """
    gamma1 = gamma1.reshape(-1, 1)
    cn = -(n - 1)
    bn = -n + (n - 1) * np.log(n - 1)
    an = -(n - 1) * (1 - np.log(n - 1) + 0.5 * (np.log(n - 1)) ** 2)
    vstar = np.log(n - 1)

    v = X @ gamma1
    phi = np.where(v < vstar, -v - np.exp(v), an + bn * v + 0.5 * cn * v**2)
    phi1 = np.where(v < vstar, -1 - np.exp(v), bn + cn * v)

    res = -np.sum((1 - D) * phi + v)

    grad = -X.T @ (((1 - D) * phi1 + 1).reshape(-1))

    return res, grad


@dataclass
class PScoreRes:
    pscore: np.ndarray
    flag: int


def pscore_ipt(
    D: np.ndarray,
    X: np.ndarray,
    init_gamma: np.ndarray | None = None,
):
    n = D.shape[0]
    if init_gamma is None:
        init_gamma = sm.Logit(endog=D, exog=X).fit(disp=0).params
    pscore_ipt = minimize(
        fun=lambda x: loss_ps_IPT(x, n, D, X)[0],
        x0=init_gamma,
        jac=lambda x: loss_ps_IPT(x, n, D, X)[1],
        method="trust-constr",
        options={"maxiter": 10000, "gtol": 1e-6},
    )
    return pscore_ipt


def pscore_cal(D: np.ndarray, X: np.ndarray):
    """
    Compute pscore using the methods in the Zhao & SantAnna R impelementation.
    """
    # Initial gamma estimate
    ps_tr = sm.Logit(endog=D, exog=X).fit(disp=0)
    init_gamma = ps_tr.params

    res = minimize(
        fun=loss_ps_cal,
        x0=init_gamma,
        args=(D, X),
        method="trust-constr",
        options={"maxiter": 1000},
    )

    if res.success:
        gamma_cal = res.x
        flag = 0
    else:
        res_ipt = pscore_ipt(D, X, init_gamma)
        if res_ipt.success:
            gamma_cal = res_ipt.x
            flag = 1
        else:
            # Neither worked; use initial gamma from simple logit
            flag = 2
            gamma_cal = init_gamma
    pscore_index = X @ gamma_cal
    pscore = expit(pscore_index)
    return PScoreRes(pscore=pscore, flag=flag)


def att_dr_rc_imp(drp: DrParamsRc) -> DrDidRes:
    """Doubly-Robust repeated cross-section estimator improved.

    Notes:
        I take everything as column vectors besides boolean indices!
    """

    Y, X, T, D = drp.Y, drp.X, drp.T, drp.D

    n = Y.shape[0]

    ps_ipt = pscore_cal(D, X)
    ps_fit = ps_ipt.pscore
    ps_fit = np.minimum(ps_fit, 1 - 1e-6).reshape(-1, 1)

    # Compute the Outcome regression for the control group

    w_ps = (ps_fit / (1 - ps_fit)).ravel()
    b = (D == 0) & (T == 0)
    oreg_c_pre = oreg(X, Y, b=b.ravel(), weights=w_ps)
    b = ((D == 0) & (T == 1)).ravel()
    oreg_c_post = oreg(X, Y, b=b, weights=w_ps)
    mu_0y = T * oreg_c_post.pred + (1 - T) * oreg_c_pre.pred

    # oreg treated in pre and post
    b = (D == 1) & (T == 0)
    oreg_t_pre = oreg(X, Y, b=b.ravel())
    b = (D == 1) & (T == 1)
    oreg_t_post = oreg(X, Y, b=b.ravel())

    # \mu_{0, Y}^{rc}
    mu_0y = T * oreg_c_post.pred + (1 - T) * oreg_c_pre.pred

    # weight numerators
    # w_{1, t}^{rc}
    w_t_pre = D * (1 - T)
    w_t_post = D * T
    # w_{0, t}^{rc}
    w_c_pre = ps_fit * (1 - D) * (1 - T) / (1 - ps_fit)
    w_c_post = ps_fit * (1 - D) * T / (1 - ps_fit)

    # components of \hat{\tau}^{dr, rc}_{1}
    res_tau1 = Y - mu_0y  # residual
    eta_t_pre = w_t_pre * res_tau1 / w_t_pre.mean()
    eta_t_post = w_t_post * res_tau1 / w_t_post.mean()
    eta_c_pre = w_c_pre * res_tau1 / w_c_pre.mean()
    eta_c_post = w_c_post * res_tau1 / w_c_post.mean()

    res_tau2_post = oreg_t_post.pred - oreg_c_post.pred
    eta_d_post = (D / D.mean()) * res_tau2_post
    eta_dt1_post = (D * T / (D * T).mean()) * res_tau2_post

    res_tau2_pre = oreg_t_pre.pred - oreg_c_pre.pred
    eta_d_pre = (D / D.mean()) * res_tau2_pre
    eta_dt0_pre = (D * (1 - T) / (D * (1 - T)).mean()) * res_tau2_pre

    # Estimator consists of 4 terms
    att_t_pre = eta_t_pre.mean()
    att_t_post = eta_t_post.mean()
    att_c_pre = eta_c_pre.mean()
    att_c_post = eta_c_post.mean()
    att_d_pre = eta_d_pre.mean()
    att_d_post = eta_d_post.mean()
    att_dt0_pre = eta_dt0_pre.mean()
    att_dt1_post = eta_dt1_post.mean()

    tau1 = (att_t_post - att_t_pre) - (att_c_post - att_c_pre)
    tau2_1 = att_d_post - att_dt1_post
    tau2_2 = att_d_pre - att_dt0_pre
    tau2 = tau1 + (tau2_1 - tau2_2)  # ATT estimator

    # *IF*

    # IF of treat compoment; \eta_{1}^{rc, 1}
    if_t_pre = eta_t_pre - w_t_pre * att_t_pre / w_t_pre.mean()
    if_t_post = eta_t_post - w_t_post * att_t_post / w_t_post.mean()

    if_t = if_t_post - if_t_pre

    # IF of control compoment; \eta_{0}^{rc, 1}
    if_c_pre = eta_c_pre - w_c_pre * att_c_pre / w_c_pre.mean()
    if_c_post = eta_c_post - w_c_post * att_c_post / w_c_post.mean()

    if_c = if_c_post - if_c_pre

    if_tau1 = if_t - if_c

    # IF of adjustment terms
    if_e1 = eta_d_post - D * att_d_post / D.mean()
    if_e2 = eta_dt1_post - D * T * att_dt1_post / (D * T).mean()
    if_e3 = eta_d_pre - D * att_d_pre / D.mean()
    if_e4 = eta_dt0_pre - D * (1 - T) * att_dt0_pre / (D * (1 - T)).mean()
    if_e = (if_e1 - if_e2) - (if_e3 - if_e4)

    # IF tau2
    if_tau2 = if_tau1 + if_e

    # se_if(if_tau2)
    se_tau2 = if_tau2.std(ddof=1) / np.sqrt(n)  # how csa computes it

    return DrDidRes(
        att=tau2,
        se=se_tau2,
        IF=if_tau2,
    )


def att_dr_panel(orp: PanelParams):
    """Doubly-Robust panel estimator.

    https://psantanna.com/DRDID/reference/drdid_panel.html
    """
    G = orp.G
    X = orp.X
    dy = orp.dy
    n = dy.shape[0]

    ps_tr = fast_logit(endog=G.ravel(), exog=X)
    ps_fit = np.minimum(ps_tr.predict(), 1 - 1e-6).reshape(-1, 1)
    W = ps_fit * (1 - ps_fit)

    barr = (G == 0).ravel()
    coef = ols(endog=dy[barr], exog=X[barr])
    dy_hat = X @ coef  # predicted dy i.e. Y_1 - Y_0

    w_t = G.reshape(-1, 1)
    w_c = ps_fit * (1 - G) / (1 - ps_fit)

    dr_att_t = w_t * (dy - dy_hat)
    dr_att_c = w_c * (dy - dy_hat)

    # w_1(dY - \mu_{0, delta}) and w_0(dY - \mu_{0, delta})
    eta_t = dr_att_t.mean() / w_t.mean()
    eta_c = dr_att_c.mean() / w_c.mean()

    dr_att = eta_t - eta_c

    w_ols = 1 - G
    if_ols_pre = if_ols(eps=dy - dy_hat, X=w_ols * X, n=n)

    # IF of logreg
    score_ps = (G - ps_fit) * X
    L = cholesky(X.T @ (W * X))  # W = ps_fit * (1 - ps_fit)
    hess_ps = cho_solve((L, False), np.eye(X.shape[1])) * n
    if_ps = score_ps @ hess_ps  # IF of logreg; dim (n, k)

    # if_ols_pre: (n, k); X is just the \dot{\mu} while OLS
    if_t_1 = dr_att_t - w_t * eta_t
    m1 = (w_t * X).mean(axis=0, keepdims=True).T  # (k, 1)
    if_t_2 = if_ols_pre @ m1
    if_t = (if_t_1 - if_t_2) / w_t.mean()

    # if control component
    if_c_1 = dr_att_c - w_c * eta_c
    # estimation effect pscore
    m2 = (w_c * (dy - dy_hat - eta_c) * X).mean(axis=0, keepdims=True).T
    if_c_2 = if_ps @ m2

    # estimation effect ols
    m3 = (w_c * X).mean(axis=0, keepdims=True).T
    if_c_3 = if_ols_pre @ m3

    if_t = (if_t_1 - if_t_2) / w_t.mean()
    if_c = (if_c_1 + if_c_2 - if_c_3) / w_c.mean()
    if_dr_att = if_t - if_c

    return DrDidRes(
        att=dr_att,
        se=se_if(if_dr_att),
        IF=if_dr_att,
    )
