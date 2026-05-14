"""Walk-forward backtest engine with monthly rebalancing and transaction costs."""

import logging

import numpy as np
import pandas as pd

from .config import Config
from .risk_model import estimate_cov
from .optimizer import solve_gmv

logger = logging.getLogger(__name__)


def run_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: Config,
) -> dict:
    """Run a walk-forward backtest.

    Parameters
    ----------
    prices : cleaned price DataFrame
    returns : daily return DataFrame
    config : Config object

    Returns
    -------
    dict with keys: nav, returns, weights, turnover, benchmark_nav
    """
    # ------------------------------------------------------------------
    # 1) Determine rebalance dates (end-of-month dates present in index)
    # ------------------------------------------------------------------
    reb_candidates = returns.resample(config.rebalance_freq).last().index
    reb_dates = reb_candidates[reb_candidates.isin(returns.index)]

    logger.info("Backtest: %d rebalance dates from %s to %s",
                len(reb_dates), reb_dates[0].date(), reb_dates[-1].date())

    all_tickers = returns.columns.tolist()
    port_rets_list = []           # (date, return)
    weights_records = []          # (reb_date, {ticker: weight})
    turnover_records = []         # (reb_date, turnover)
    w_prev = np.zeros(len(all_tickers))
    prev_reb = None

    for i, t in enumerate(reb_dates):
        # ---------------------------------------------------------------
        # 2) Estimation window: last N trading days up to and including t
        # ---------------------------------------------------------------
        window = returns.loc[:t].tail(config.estimation_window_days)
        if len(window) < config.estimation_window_days * 0.5:
            logger.debug("Skipping %s – not enough history (%d rows).", t.date(), len(window))
            prev_reb = t
            continue

        # Only use columns that have data in the window
        valid_cols = window.dropna(axis=1, how="all").columns
        window_clean = window[valid_cols].dropna()
        if window_clean.shape[1] < 2 or len(window_clean) < 30:
            logger.debug("Skipping %s – too few clean assets/observations.", t.date())
            prev_reb = t
            continue

        # ---------------------------------------------------------------
        # 3) Estimate covariance
        # ---------------------------------------------------------------
        try:
            cov = estimate_cov(
                window_clean,
                method=config.cov_method,
                annualization=config.annualization,
                ewma_lambda=config.ewma_lambda,
            )
        except Exception as exc:
            logger.warning("Cov estimation failed at %s: %s", t.date(), exc)
            prev_reb = t
            continue

        # ---------------------------------------------------------------
        # 4) Optimize
        # ---------------------------------------------------------------
        w_opt = solve_gmv(cov, w_max=config.w_max, long_only=config.long_only)

        # Map weights to full ticker universe (0 for excluded assets)
        w_full = pd.Series(0.0, index=all_tickers)
        for j, col in enumerate(valid_cols):
            if col in w_full.index:
                w_full[col] = w_opt[j] if j < len(w_opt) else 0.0
        # Re-normalise (in case some tickers are missing)
        if w_full.sum() > 0:
            w_full /= w_full.sum()
        w_arr = w_full.values

        # ---------------------------------------------------------------
        # 5) Record weights & turnover
        # ---------------------------------------------------------------
        turnover = np.abs(w_arr - w_prev).sum()
        turnover_records.append((t, turnover))
        weights_records.append((t, w_full.to_dict()))

        # ---------------------------------------------------------------
        # 6) Compute portfolio returns in the NEXT period (t, t_next]
        # ---------------------------------------------------------------
        if i + 1 < len(reb_dates):
            t_next = reb_dates[i + 1]
        else:
            t_next = returns.index[-1]

        period_rets = returns.loc[t:t_next].iloc[1:]  # exclude t itself
        if period_rets.empty:
            w_prev = w_arr
            prev_reb = t
            continue

        # Daily portfolio return = sum(w * r), handling NaN by filling 0
        daily_port = (period_rets.fillna(0) * w_arr).sum(axis=1)

        # Subtract transaction cost on first day of the period
        cost = (config.tc_bps / 10_000) * turnover
        if len(daily_port) > 0:
            daily_port.iloc[0] -= cost

        port_rets_list.append(daily_port)

        w_prev = w_arr
        prev_reb = t

    # ------------------------------------------------------------------
    # 7) Assemble results
    # ------------------------------------------------------------------
    if not port_rets_list:
        raise RuntimeError("Backtest produced zero periods – check data coverage.")

    port_returns = pd.concat(port_rets_list).sort_index()
    port_returns.name = "portfolio"
    nav = (1 + port_returns).cumprod()
    nav.name = "portfolio"

    weights_df = pd.DataFrame.from_records(
        [(d, w) for d, w in weights_records],
        columns=["date", "weights"],
    )
    weights_df = pd.DataFrame(weights_df["weights"].tolist(), index=weights_df["date"])

    turnover_s = pd.Series(
        [v for _, v in turnover_records],
        index=[d for d, _ in turnover_records],
        name="turnover",
    )

    # ------------------------------------------------------------------
    # 8) Benchmark (buy-and-hold SPY or user-specified)
    # ------------------------------------------------------------------
    benchmark_nav = None
    bench = config.benchmark_ticker
    if bench and bench in prices.columns:
        bench_px = prices[bench].dropna()
        # Align to portfolio nav dates
        common = bench_px.index.intersection(nav.index)
        if len(common) > 10:
            bench_px = bench_px.loc[common]
            benchmark_nav = bench_px / bench_px.iloc[0]
            benchmark_nav.name = "benchmark"

    results = {
        "nav": nav,
        "returns": port_returns,
        "weights": weights_df,
        "turnover": turnover_s,
        "benchmark_nav": benchmark_nav,
    }
    return results
