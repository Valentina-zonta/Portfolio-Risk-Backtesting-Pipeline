# Portfolio Risk Report

Pipeline Python per costruzione portafoglio, risk model e report automatico.

## Pipeline

1. **Download** – Scarica prezzi storici da Yahoo Finance (`yfinance`)
2. **Clean** – Rimuove dati sporchi, filtra per copertura, forward-fill
3. **Returns** – Calcola log-returns giornalieri con winsorization
4. **Risk Model** – Stima matrice di covarianza (Ledoit-Wolf o EWMA)
5. **Optimizer** – Portafoglio Global Minimum Variance (long-only, cap per asset) via `cvxpy`
6. **Backtest** – Walk-forward mensile con costi di transazione
7. **Report** – HTML con grafici Plotly interattivi e tabelle metriche

## Quick Start

```bash
pip install pandas numpy yfinance scikit-learn cvxpy plotly jinja2 pyarrow
python run_pipeline.py
```

Il report viene generato in `outputs/reports/report_*.html`.

## Configurazione

Modifica i parametri in `run_pipeline.py` o `src/config.py`:

| Parametro | Default | Descrizione |
|---|---|---|
| `estimation_window_days` | 252 | Finestra stima covarianza |
| `cov_method` | `ledoit_wolf` | Metodo covarianza (`ledoit_wolf` / `ewma`) |
| `w_max` | 0.15 | Cap massimo per singolo asset |
| `tc_bps` | 10 | Costi di transazione (basis points) |
| `rebalance_freq` | `M` | Frequenza ribilanciamento |

## Struttura

```
portfolio_risk_report/
  data/tickers.txt          # universo ETF
  src/                      # moduli pipeline
  templates/                # template Jinja2
  outputs/reports/          # report HTML generati
  run_pipeline.py           # entry point
```
