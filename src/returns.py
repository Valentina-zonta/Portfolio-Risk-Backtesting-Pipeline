"""Compute returns from price data."""

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns from prices.

    Parameters
    ----------
    prices : DataFrame of cleaned prices
    method : 'log' for log returns, 'simple' for arithmetic returns

    Returns
    -------
    DataFrame of returns (first row dropped)
    """
    if method == "log":
        rets = np.log(prices).diff()
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log' or 'simple'.")

    # Drop the first row (NaN from diff)
    rets = rets.iloc[1:]

    # Winsorize extreme returns at 0.1% / 99.9% per column for robustness
    lower = rets.quantile(0.001)
    upper = rets.quantile(0.999)
    rets = rets.clip(lower=lower, upper=upper, axis=1)

    return rets
