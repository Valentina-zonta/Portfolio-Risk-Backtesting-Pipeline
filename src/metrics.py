"""Performance metrics for backtest results."""

import numpy as np
import pandas as pd


def compute_drawdown(nav: pd.Series) -> pd.Series:
    """Compute drawdown series from NAV."""
    running_max = nav.cummax()
    dd = nav / running_max - 1
    return dd


def compute_performance_stats(
    returns: pd.Series,
    annualization: int = 252,
    rf: float = 0.0,
) -> dict:
    """Compute key performance statistics.

    Parameters
    ----------
    returns : daily return series
    annualization : trading days per year
    rf : annualized risk-free rate

    Returns
    -------
    dict with CAGR, vol, Sharpe, max_drawdown, etc.
    """
    rets = returns.dropna()
    n_days = len(rets)
    if n_days < 2:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}

    # NAV
    nav = (1 + rets).cumprod()
    total_return = nav.iloc[-1] / 1.0  # start = 1
    years = n_days / annualization

    # CAGR
    cagr = total_return ** (1 / years) - 1

    # Annualized volatility
    vol = rets.std() * np.sqrt(annualization)

    # Sharpe
    excess = rets.mean() - rf / annualization
    sharpe = (excess / rets.std()) * np.sqrt(annualization) if rets.std() > 0 else np.nan

    # Max drawdown
    dd = compute_drawdown(nav)
    max_dd = dd.min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "CAGR": round(cagr * 100, 2),
        "Vol (ann.)": round(vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Max Drawdown": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3),
        "Total Days": n_days,
        "Years": round(years, 1),
    }
