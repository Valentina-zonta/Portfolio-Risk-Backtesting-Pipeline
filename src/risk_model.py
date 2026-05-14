"""Risk model: covariance estimation."""

import logging

import numpy as np
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def estimate_cov(returns_window, method: str, annualization: int = 252, ewma_lambda: float = 0.94):
    """Estimate annualized covariance matrix.

    Parameters
    ----------
    returns_window : DataFrame of daily returns (estimation window)
    method : 'ledoit_wolf' or 'ewma'
    annualization : trading days per year
    ewma_lambda : decay factor for EWMA

    Returns
    -------
    np.ndarray – annualized covariance matrix (n x n)
    """
    # Drop rows with any NaN
    X = returns_window.dropna()

    if len(X) < 20:
        raise ValueError(f"Only {len(X)} clean observations – not enough to estimate covariance.")

    if method == "ledoit_wolf":
        cov_daily = LedoitWolf().fit(X.values).covariance_
    elif method == "ewma":
        cov_daily = _ewma_cov(X.values, lam=ewma_lambda)
    else:
        raise ValueError(f"Unknown cov method '{method}'.")

    # Annualize
    cov_ann = cov_daily * annualization

    # Ensure symmetry (numerical safety)
    cov_ann = 0.5 * (cov_ann + cov_ann.T)

    return cov_ann


def _ewma_cov(X: np.ndarray, lam: float) -> np.ndarray:
    """Exponentially Weighted Moving Average covariance.

    Parameters
    ----------
    X : (T, n) array of returns
    lam : decay parameter (0 < lam < 1)

    Returns
    -------
    (n, n) covariance matrix (daily scale)
    """
    T, n = X.shape
    # Demean
    mu = X.mean(axis=0)
    X_dm = X - mu

    # Initialize with sample covariance of first min(60, T//2) observations
    init_len = min(60, T // 2)
    S = np.cov(X_dm[:init_len], rowvar=False)

    for t in range(init_len, T):
        r = X_dm[t].reshape(-1, 1)
        S = lam * S + (1 - lam) * (r @ r.T)

    return S
