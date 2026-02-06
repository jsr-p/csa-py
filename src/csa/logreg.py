"""
Module with a basic logistic regression and iteratively reweighted least
squares implementation.

author: jsr-p

Taken from https://github.com/jsr-p/fastlr
"""

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from scipy.linalg import solve


def s(X: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sigmoid function"""
    return expit(X @ beta)


def neg_loglik(
    X: NDArray[np.float64], y: NDArray[np.float64], beta: NDArray[np.float64]
) -> float:
    """negative log likehood for logistic regression"""
    linear = X @ beta
    return -np.sum(y * linear - np.log1p(np.exp(linear)))


def grad(
    X: NDArray[np.float64], y: NDArray[np.float64], beta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gradient (wrt. params) for logistic regression"""
    p = s(X, beta)
    return -X.T @ (y - p)


def hess(X: NDArray[np.float64], beta: NDArray[np.float64]):
    """Hessian (wrt. params) for logistic regression"""
    p = s(X, beta)
    W = p * (1 - p)
    return X.T @ (X * W[:, np.newaxis])


def generate_data_simple(
    N: int, beta: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    X = np.column_stack(
        (np.ones((N, 1)), np.random.normal(size=(N, beta.shape[0] - 1)))
    )
    y = np.random.binomial(1, expit(X @ beta))
    return X, y


@dataclass
class IRLSResult:
    coefficients: NDArray[np.float64]
    iterations: int
    converged: bool = False
    time: float = 0.0


def irls(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    tol: float = 1e-8,
    max_iter: int = 100,
):
    """Iteratively reweighted least squares solver."""
    beta_t = np.zeros(shape=(X.shape[1], 1))
    changed = np.inf
    iters = 0
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    start_time = time.perf_counter()
    Xt = X.T
    while changed and iters < max_iter:
        eta = X @ beta_t
        mu_t = expit(eta)
        W_t = mu_t * (1 - mu_t)
        z_t = eta + (y - mu_t) / W_t
        beta_t_next = solve(Xt @ (X * W_t), Xt @ (W_t * z_t))
        changed = np.any(np.abs(beta_t_next - beta_t) > tol)
        beta_t = beta_t_next
        iters += 1
    elapsed = time.perf_counter() - start_time
    return IRLSResult(
        coefficients=beta_t.ravel(),
        iterations=iters,
        time=elapsed,
        converged=not bool(changed),
    )
