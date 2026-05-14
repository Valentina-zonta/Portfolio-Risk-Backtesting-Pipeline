# Portfolio Risk & Backtesting Pipeline

End-to-end **Python** pipeline that builds a long-only equity portfolio, estimates its risk model, runs a **walk-forward backtest** with realistic transaction costs and produces a self-contained **HTML report**.

The project was developed as a personal exercise to practice the full risk / portfolio-construction workflow on real market data, mirroring the skeleton of a quantitative risk-team production pipeline.

## What the pipeline does

```
Yahoo Finance ──► Clean prices ──► Log-returns ──► Risk model ──► GMV optimiser ──► Walk-forward backtest ──► HTML report
                                  (winsorised)    (Ledoit–Wolf  (cvxpy, long-only,  (monthly rebal,
                                                   or EWMA)       per-asset cap)    transaction costs)
```

| Step | Module | Highlights |
|---|---|---|
| 1. Download | `src/download.py` | Adjusted-close prices via `yfinance`, parquet caching to avoid re-downloads. |
| 2. Clean | `src/clean.py` | Coverage filter (`min_history_ratio`), forward-fill on isolated gaps, removal of stale series. |
| 3. Returns | `src/returns.py` | Daily **log-returns**, optional winsorization to dampen outliers without dropping observations. |
| 4. Risk model | `src/risk_model.py` | Annualised covariance via either **Ledoit–Wolf shrinkage** (`scikit-learn`) or **EWMA** (RiskMetrics-style, configurable λ); explicit symmetrisation for numerical safety. |
| 5. Optimiser | `src/optimizer.py` | **Global Minimum Variance** portfolio solved with `cvxpy` (OSQP, fallback to SCS); long-only and per-asset cap constraints. |
| 6. Backtest | `src/backtest.py` | **Walk-forward** rebalancing (`M`/`ME` end-of-month), 252-day rolling estimation window, **transaction costs** in basis points applied on portfolio turnover. |
| 7. Metrics | `src/metrics.py` | CAGR, annualised volatility, Sharpe, **maximum drawdown**, hit-ratio. |
| 8. Report | `src/report.py` | Interactive **Plotly** charts (NAV vs benchmark, drawdown, weights heatmap, turnover) embedded in a Jinja2 HTML template. |

## Quick start

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Run the pipeline
python run_pipeline.py

# Optional: re-download prices, ignoring the parquet cache
python run_pipeline.py --force-download
```

The HTML report is saved to `outputs/reports/report_<timestamp>.html`.

## Configuration

All knobs live in the `Config` dataclass instantiated in `run_pipeline.py` and defined in `src/config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `tickers` | `data/tickers.txt` | Universe of tradable assets, one ticker per line. |
| `start_date` / `end_date` | 2005-01-01 / 2025-01-01 | Sample window. |
| `estimation_window_days` | 252 | Length of the rolling window used to estimate the covariance. |
| `cov_method` | `ledoit_wolf` | `ledoit_wolf` (shrinkage) or `ewma` (with `ewma_lambda`, default 0.94). |
| `portfolio_type` | `gmv` | Currently Global Minimum Variance; the optimiser API is generic. |
| `long_only` | `True` | Enforce non-negativity of weights. |
| `w_max` | 0.15 | Maximum weight per single asset (concentration cap). |
| `tc_bps` | 10.0 | Per-trade transaction cost in basis points, applied to turnover. |
| `rebalance_freq` | `ME` | Pandas offset alias (monthly end). |
| `benchmark_ticker` | `SPY` | Benchmark used in the report. |

## Project structure

```
portfolio_risk_project/
├── data/
│   ├── tickers.txt          # universe of ETFs / stocks
│   └── raw_prices.parquet   # cached download
├── src/
│   ├── config.py            # Config dataclass
│   ├── download.py          # yfinance wrapper + caching
│   ├── clean.py             # data quality filters
│   ├── returns.py           # log-returns + winsorization
│   ├── risk_model.py        # Ledoit-Wolf / EWMA covariance
│   ├── optimizer.py         # GMV solver via cvxpy
│   ├── backtest.py          # walk-forward backtest engine
│   ├── metrics.py           # performance & risk metrics
│   └── report.py            # Plotly + Jinja2 HTML report
├── templates/
│   └── report_template.html
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── run_pipeline.py          # entry point
└── README.md
```

## Why the design choices

- **Ledoit–Wolf shrinkage** is the standard remedy for the ill-conditioned sample covariance when the number of assets approaches the number of observations – the typical setting in production risk models.
- **EWMA(λ=0.94)** reproduces the **RiskMetrics** convention used by countless market-risk implementations.
- **GMV with per-asset cap** is a deliberate baseline: a fully unconstrained mean-variance problem is dominated by estimation error in expected returns, and a capped GMV portfolio is the cleanest stress-test of how much of the realised performance comes from the risk model alone.
- **Walk-forward backtest with transaction costs** avoids look-ahead bias and gives realistic, net-of-cost performance numbers – the only kind that matter for validation.

## Possible extensions

- Add **risk-parity** and **maximum diversification** portfolios alongside GMV.
- Plug in a **factor risk model** (Barra-style) instead of the asset-level covariance.
- Replace the historical covariance with a **GARCH/DCC**-implied conditional covariance for tail-risk-aware rebalancing.
- Add **VaR / Expected Shortfall** backtesting on the realised portfolio P&L.

## Author

**Valentina Zonta** – MSc candidate, Mathematical Engineering (Financial Track), University of Padova.
GitHub: [Valentina-zonta](https://github.com/Valentina-zonta)
