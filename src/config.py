"""Configuration dataclass for the portfolio pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Config:
    tickers: List[str] = field(default_factory=list)
    start_date: str = "2005-01-01"
    end_date: Optional[str] = "2025-01-01"
    price_field: str = "Adj Close"
    min_history_ratio: float = 0.85
    estimation_window_days: int = 252
    rebalance_freq: str = "ME"
    cov_method: str = "ledoit_wolf"          # "ledoit_wolf" or "ewma"
    ewma_lambda: float = 0.94
    portfolio_type: str = "gmv"              # global minimum variance
    long_only: bool = True
    w_max: float = 0.15                      # max weight per asset
    tc_bps: float = 10.0                     # transaction cost in basis points
    annualization: int = 252
    risk_free_rate: float = 0.0
    benchmark_ticker: str = "SPY"
