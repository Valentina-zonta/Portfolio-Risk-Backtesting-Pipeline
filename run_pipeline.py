#!/usr/bin/env python3
"""
run_pipeline.py – Portfolio Construction + Risk Model + Backtest + Report

Usage:
    python run_pipeline.py [--force-download]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.download import download_prices
from src.clean import clean_prices
from src.returns import compute_returns
from src.backtest import run_backtest
from src.metrics import compute_performance_stats
from src.report import build_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def main():
    parser = argparse.ArgumentParser(description="Portfolio risk pipeline")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download prices even if cache exists")
    args = parser.parse_args()

    # -----------------------------------------------------------
    # 1) Read tickers
    # -----------------------------------------------------------
    tickers_file = PROJECT_ROOT / "data" / "tickers.txt"
    tickers = [
        line.strip()
        for line in tickers_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    logger.info("Tickers: %s", tickers)

    # -----------------------------------------------------------
    # 2) Create Config
    # -----------------------------------------------------------
    config = Config(
        tickers=tickers,
        start_date="2005-01-01",
        end_date="2025-01-01",
        price_field="Adj Close",
        min_history_ratio=0.85,
        estimation_window_days=252,
        rebalance_freq="ME",
        cov_method="ledoit_wolf",
        portfolio_type="gmv",
        long_only=True,
        w_max=0.15,
        tc_bps=10.0,
        benchmark_ticker="SPY",
    )

    # -----------------------------------------------------------
    # 3) Download prices (with caching)
    # -----------------------------------------------------------
    cache_path = PROJECT_ROOT / "data" / "raw_prices.parquet"
    if cache_path.exists() and not args.force_download:
        logger.info("Loading cached prices from %s", cache_path)
        prices_raw = pd.read_parquet(cache_path)
    else:
        # Make sure benchmark is included in download
        dl_tickers = list(set(tickers + [config.benchmark_ticker]))
        prices_raw = download_prices(
            dl_tickers,
            start=config.start_date,
            end=config.end_date,
            field=config.price_field,
        )
        os.makedirs(cache_path.parent, exist_ok=True)
        prices_raw.to_parquet(cache_path)
        logger.info("Saved raw prices → %s", cache_path)

    # -----------------------------------------------------------
    # 4) Clean prices
    # -----------------------------------------------------------
    prices_clean = clean_prices(prices_raw, config.min_history_ratio)

    # -----------------------------------------------------------
    # 5) Compute returns
    # -----------------------------------------------------------
    returns = compute_returns(prices_clean, method="log")
    logger.info("Returns shape: %s", returns.shape)

    # -----------------------------------------------------------
    # 6) Run backtest
    # -----------------------------------------------------------
    results = run_backtest(prices_clean, returns, config)
    logger.info("Backtest complete – NAV from %.4f to %.4f",
                results["nav"].iloc[0], results["nav"].iloc[-1])

    # -----------------------------------------------------------
    # 7) Compute metrics
    # -----------------------------------------------------------
    stats = compute_performance_stats(
        results["returns"], config.annualization, config.risk_free_rate
    )
    logger.info("=" * 50)
    logger.info("PORTFOLIO PERFORMANCE")
    for k, v in stats.items():
        logger.info("  %-18s %s", k, v)
    logger.info("=" * 50)

    # -----------------------------------------------------------
    # 8) Generate report
    # -----------------------------------------------------------
    report_dir = str(PROJECT_ROOT / "outputs" / "reports")
    report_path = build_report(results, config, output_dir=report_dir)

    logger.info("✅  Pipeline complete.  Report → %s", report_path)
    return report_path


if __name__ == "__main__":
    main()
