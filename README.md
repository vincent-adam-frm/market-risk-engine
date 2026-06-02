# Market Risk Engine

A portfolio market risk system built in Python. Reads daily positions, prices and FX rates from CSV files, computes VaR, CVaR and risk attribution, backtests the model, and produces a formatted Excel report.

---

## Project layout

```
03_market_risk/
├── portfolio_demo.py        Entry point — runs the full pipeline and generates the report
├── build_report.py          Excel report builder (called automatically by portfolio_demo.py)
│
├── risk_engine/             Core risk library (importable as a package)
│   ├── __init__.py
│   ├── volatility.py        EWMA conditional volatility
│   ├── risk_models.py       Historical VaR, FHS-VaR, CVaR, covariance attribution
│   ├── rolling_var.py       Rolling out-of-sample forecast series
│   └── modelvalidation.py   Backtesting — Binomial, Christoffersen, CVaR exceedance, Basel Traffic Light
│
└── data/
    └── inbound/
        ├── positions.csv    Snapshot holdings (one date, one row per security)
        ├── prices.csv       Historical daily prices
        └── fx_rates.csv     Historical daily FX rates to report currency
```

---

## Quick start

```bash
# Install dependencies
pip install requirements.txt

# Generate sample data and run the full report
python portfolio_demo.py --sample

# Run on your own CSV files (default: data/inbound/)
python portfolio_demo.py

# Print terminal report only, skip Excel output
python portfolio_demo.py --no-excel
```

The script produces:
- A terminal risk report
- `risk_report.xlsx` in the project root — 7-tab Excel workbook ready to share

---

## Input file format

| File | Key columns |
|---|---|
| `positions.csv` | Date, Desk, Book, Strategy, Security, Quantity, Price_Currency |
| `prices.csv` | Date, Security, Price, Price_Currency |
| `fx_rates.csv` | Date, Currency, FX_to_Report_Ccy |

`positions.csv` must contain a **single snapshot date**. USD should appear in `fx_rates.csv` with `FX_to_Report_Ccy = 1.0`. Any currency not in the file is assumed to already be in report currency.

---

## What the report covers

| Section | Content |
|---|---|
| Volatility | EWMA current and average, annualised |
| Risk measures | Historical VaR, FHS-VaR, CVaR, FHS-CVaR at 1-day 99% |
| Attribution | Component VaR by security, book and desk |
| Backtest | Binomial, Christoffersen, CVaR exceedance, Basel Traffic Light |
| FX decomposition | Total / equity / FX split, FX add-on |

For methodology detail and references see [`risk_engine/README.md`](risk_engine/README.md).
