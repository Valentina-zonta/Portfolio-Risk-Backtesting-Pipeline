"""Clean and filter price data."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_prices(prices: pd.DataFrame, min_history_ratio: float = 0.85) -> pd.DataFrame:
    """Clean raw prices: remove bad values, filter by coverage, forward-fill gaps.

    Parameters
    ----------
    prices : DataFrame of raw prices (index=date, cols=tickers)
    min_history_ratio : minimum fraction of non-NaN observations required

    Returns
    -------
    DataFrame of cleaned prices
    """
    df = prices.copy()

    # 1) Replace non-positive prices with NaN
    df[df <= 0] = np.nan

    # 2) Coverage filter
    total = len(df)
    coverage = df.notna().sum() / total
    keep = coverage[coverage >= min_history_ratio].index.tolist()
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        logger.warning(
            "Dropped %d assets below %.0f%% coverage: %s",
            len(dropped),
            min_history_ratio * 100,
            dropped,
        )
    df = df[keep]

    # 3) Forward-fill with limit, then leave residual NaN
    df = df.ffill(limit=5)

    # 4) Sort index
    df = df.sort_index()

    logger.info("Clean prices: %d assets, %d rows", df.shape[1], df.shape[0])
    return df
