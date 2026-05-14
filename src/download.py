"""Download price data from Yahoo Finance."""

import logging
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_prices(
    tickers: List[str],
    start: str,
    end: Optional[str],
    field: str = "Adj Close",
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
    start, end : date strings
    field : price field to extract (default 'Adj Close')

    Returns
    -------
    pd.DataFrame  – index: datetime, columns: tickers, values: float prices
    """
    logger.info("Downloading %d tickers from Yahoo Finance …", len(tickers))

    px = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=True,
    )

    if px.empty:
        raise RuntimeError("Yahoo Finance returned an empty DataFrame.")

    # Handle MultiIndex columns (typical when >1 ticker)
    if isinstance(px.columns, pd.MultiIndex):
        if field in px.columns.get_level_values(0):
            px_field = px[field].copy()
        else:
            # Newer yfinance versions may swap levels
            available = px.columns.get_level_values(0).unique().tolist()
            logger.warning("Field '%s' not at level-0. Available: %s", field, available)
            # Try level-1
            if field in px.columns.get_level_values(1).unique():
                px_field = px.xs(field, axis=1, level=1).copy()
            else:
                raise KeyError(f"'{field}' not found in downloaded columns: {available}")
    else:
        # Single ticker → simple columns
        if field in px.columns:
            px_field = px[[field]].copy()
            px_field.columns = tickers[:1]
        else:
            px_field = px.copy()

    # Ensure column names are plain strings
    px_field.columns = [str(c) for c in px_field.columns]

    # Drop columns that are entirely NaN
    px_field = px_field.dropna(axis=1, how="all")

    # Ensure datetime index
    px_field.index = pd.to_datetime(px_field.index)
    px_field = px_field.sort_index()

    dropped = set(tickers) - set(px_field.columns)
    if dropped:
        logger.warning("Tickers dropped (no data): %s", dropped)

    logger.info(
        "Downloaded %d tickers, %d rows (%s → %s)",
        len(px_field.columns),
        len(px_field),
        px_field.index[0].date(),
        px_field.index[-1].date(),
    )
    return px_field
