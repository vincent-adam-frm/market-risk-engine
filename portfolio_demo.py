"""
portfolio_demo.py
=================
Daily portfolio risk report from three CSV files.

Inputs (all in data/inbound/)
------------------------------
positions.csv  : Date, Desk, Book, Strategy, Security, Quantity, Price_Currency
                 One row per security, snapshot of current holdings.
                 Date is the snapshot date, used to fix FX rates for
                 the equity-only return decomposition.

prices.csv     : Date, Security, Price, Price_Currency
                 Historical daily prices.

fx_rates.csv   : Date, Currency, FX_to_Report_Ccy
                 Historical daily FX rates.

Pipeline
--------
positions + prices + fx  ->  MV_i(t) = Qty_i * Price_i(t) * FX_i(t)
                         ->  NAV(t)   = sum_i MV_i(t)
                         ->  r(t)     = log(NAV(t) / NAV(t-1))
                         ->  risk_engine
Run
---
python portfolio_demo.py --sample          # generate sample data and run
python portfolio_demo.py                   # run on data/inbound/ files
python portfolio_demo.py --no-excel        # print report only, skip Excel output
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import risk_engine as re

# ── DIR ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "inbound")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "outbound")

# ── Config ────────────────────────────────────────────────────────────────

confidence_level = 0.99

# ── Loaders ────────────────────────────────────────────────────────────────

REQUIRED_POSITION_COLS = {"Date", "Desk", "Book", "Strategy", "Security", "Quantity", "Price_Currency"}

def load_positions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = REQUIRED_POSITION_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"positions.csv is missing columns: {sorted(missing)}\n"
            f"Run:  python portfolio_demo.py --sample   to regenerate sample data."
        )
    dates = df["Date"].unique()
    if len(dates) > 1:
        raise ValueError(
            f"positions.csv must contain a single snapshot date, found {len(dates)} dates."
        )
    print(f"[positions]  {len(df)} securities | "
          f"{df['Desk'].nunique()} desks | "
          f"{df['Book'].nunique()} books | "
          f"snapshot {df['Date'].iloc[0].date()}")
    return df


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"[prices]     {df['Security'].nunique()} securities | "
          f"{df['Date'].nunique()} dates")
    return df


def load_fx(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"[fx_rates]   {df['Currency'].nunique()} currencies | "
          f"{df['Date'].nunique()} dates")
    return df


# ── Build NAV ──────────────────────────────────────────────────────────────

def build_nav(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    fx: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build daily NAV, per-security MV matrix, and hierarchy mapping.

    MV_i(t) = Quantity_i * Price_i(t) * FX_i(t)

    Both equity and FX moves contribute to NAV(t), so all downstream
    risk measures capture both risks in a single return series.

    Returns
    -------
    nav       : daily NAV series
    nav_equity: daily NAV with FX frozen at first date (for FX decomposition)
    mv_wide   : (Date x Security) MV matrix — asset-level returns
    hierarchy : Security -> Desk/Book/Strategy
    """
    ## FX ffill fix (for missing dates and bank holidays)
    # 1. Create a complete grid of all dates present in prices and all currencies
    all_dates = pd.Series(prices["Date"].unique(), name="Date")
    all_currencies = pd.Series(fx["Currency"].unique(), name="Currency")
    full_grid = all_dates.to_frame().merge(all_currencies.to_frame(), how="cross")

    # 2. Merge with actual FX data, sort, and forward-fill missing dates per currency
    fx = full_grid.merge(
        fx[["Date", "Currency", "FX_to_Report_Ccy"]],
        on=["Date", "Currency"],
        how="left"
    )
    fx = fx.sort_values(by=["Currency", "Date"])
    fx["FX_to_Report_Ccy"] = fx.groupby("Currency")["FX_to_Report_Ccy"].ffill().fillna(1.0)

    # snapshot positions across all historical dates
    df = prices[["Date", "Security", "Price"]].merge(
        positions[["Security", "Quantity", "Price_Currency",
                   "Desk", "Book", "Strategy"]],
        on="Security", how="inner",
    )

    # Join historical FX on both date and currency dimensions
    df = df.merge(
        fx[["Date", "Currency", "FX_to_Report_Ccy"]],
        left_on=["Date", "Price_Currency"],
        right_on=["Date", "Currency"],
        how="left",
    )
    df["FX_to_Report_Ccy"] = df["FX_to_Report_Ccy"].fillna(1.0)
    df["MV"] = df["Quantity"] * df["Price"] * df["FX_to_Report_Ccy"]

    nav = df.groupby("Date")["MV"].sum().sort_index()

    # Freeze FX at the positions snapshot date for equity-only NAV.
    # This answers: "what would returns look like if I had been FX-hedged
    # at the rates prevailing on the day my positions were recorded?"
    snapshot_date = positions["Date"].iloc[0]
    fx_base = (
        fx[fx["Date"] == snapshot_date]
        .set_index("Currency")["FX_to_Report_Ccy"]
        .rename("FX_base")
    )
    df = df.merge(fx_base.reset_index(), left_on="Price_Currency",
                  right_on="Currency", how="left")
    df["FX_base"] = df["FX_base"].fillna(1.0)
    df["MV_equity"] = df["Quantity"] * df["Price"] * df["FX_base"]
    nav_equity = df.groupby("Date")["MV_equity"].sum().sort_index()

    mv_wide = df.pivot_table(index="Date", columns="Security",
                              values="MV", aggfunc="sum").sort_index()

    hierarchy = (
        positions[["Security", "Desk", "Book", "Strategy"]]
        .set_index("Security")
        .loc[mv_wide.columns]
    )

    print(f"\n[NAV] {len(nav)} days | {nav.index[0].strftime('%Y-%m-%d')}:{nav.iloc[0]:>12,.0f} | {nav.index[-1].strftime('%Y-%m-%d')}:{nav.iloc[-1]:>12,.0f}  (report ccy)\n")
    return nav, nav_equity, mv_wide, hierarchy


def _log_returns(series):
    return np.log(series / series.shift(1)).dropna().values


# ── Weights from positions snapshot ────────────────────────────────────────

def snapshot_weights(positions, prices, fx, columns):
    """
    Compute portfolio weights from the positions snapshot.
    """

    snap_date = positions["Date"].iloc[0]
    snap_price  = (prices[prices["Date"] == snap_date]
                   .set_index("Security")["Price"])
    snap_fx = (fx[fx["Date"] == snap_date]
                   .set_index("Currency")["FX_to_Report_Ccy"])

    pos = positions.copy()
    pos["MV"] = (
        pos["Quantity"]
        * pos["Security"].map(snap_price)
        * pos["Price_Currency"].map(snap_fx).fillna(1.0)
    )
    total = pos["MV"].sum()
    return (pos.set_index("Security")["MV"].loc[columns] / total).values


# ── Risk report ────────────────────────────────────────────────────────────

def run_risk_report(
    returns: np.ndarray,
    returns_equity: np.ndarray,
    asset_returns: np.ndarray,
    weights: np.ndarray,
    tickers: list,
    hierarchy: pd.DataFrame,
    snapshot_date=None,
    return_dates=None,
) -> dict:
    """
    Compute all risk measures, print the terminal report, and return a
    data dict that can be passed directly to build_excel_report().
    """
    T = len(returns)
    print(f"{'='*60}")
    print(f" PORTFOLIO RISK REPORT ({T} daily observations)")
    print(f"{'='*60}")

    # ── 1. Volatility ──────────────────────────────────────────────────────
    # EWMA conditional volatility, used in FHS-VaR/CVaR below.
    sigma = re.ewma_volatility(returns)
    print(f"\n[Volatility - EWMA lambda=0.94]")
    print(f" Current : {sigma[-1]*100:.2f}%/day  "
          f"({sigma[-1]*np.sqrt(252)*100:.1f}% ann.)")
    print(f" Average : {sigma.mean()*100:.2f}%/day  "
          f"({sigma.mean()*np.sqrt(252)*100:.1f}% ann.)")

    # ── 2. Risk measures ───────────────────────────────────────────────────
    # Each pair is internally consistent: Historical uses raw returns,
    # FHS uses EWMA-standardised residuals rescaled by current vol.
    # Both columns answer the same question — VaR and CVaR — but under
    # different volatility assumptions. Compare them to see how much the
    # current vol regime matters relative to the historical average.
    var_hist = re.var_historical(returns, confidence_level)
    var_fhs_ = re.var_fhs(returns, sigma, confidence_level)
    cvar_hist = re.cvar(returns, confidence_level)
    cvar_fhs_ = re.cvar_fhs(returns, sigma, confidence_level)

    regime = "elevated" if var_fhs_ > var_hist else "subdued"
    print(f"\n[1-day 99% Risk Measures - equity + FX]")
    print(f" {'':20}  {'VaR':>8}  {'CVaR':>8}")
    print(f" {'-'*40}")
    print(f" {'Historical':20}  {var_hist*100:>7.3f}%  {cvar_hist*100:>7.3f}%")
    print(f" {'FHS (EWMA)':20}  {var_fhs_*100:>7.3f}%  {cvar_fhs_*100:>7.3f}%  "
          f"<- current vol {regime}")
    print(f" {'-'*40}")
    print(f" Historical: full return history, no vol adjustment")
    print(f" FHS: standardised by EWMA vol, rescaled by today's sigma")

    # ── 3. Attribution ─────────────────────────────────────────────────────
    # The Normal assumption here is intentional.
    # We are using the covariance model purely for attribution:
    # it first computes Marginal VaR (mVaR), the sensitivity of total portfolio VaR to a small change in each security's position,
    # then multiplies it by the current position size to obtain VaR Contributions (VaRC, also called component VaR).
    # These VaRC values add up exactly to the portfolio VaR, giving clean additive attribution by desk and book with no unexplained residual.
    # This is standard practice: distribution-free measures for the primary risk numbers, a covariance model for the attribution.

    cov = re.var_parametric_cov(asset_returns, weights, confidence=confidence_level)
    comp_var = pd.Series(cov["component_var"], index=tickers)
    total_var = cov["var"]

    print(f"\n[Risk Attribution - Covariance Model (Normal, for decomposition only)]")
    print(f" Ledoit-Wolf shrinkage delta : {cov['shrinkage']:.2f}  (0=none, 1=full identity)")
    print(f" Portfolio VaR (covariance)  : {total_var*100:.3f}%"
          f" cf. Historical VaR: {var_hist*100:.3f}%"
          f" <- use Historical for risk decisions")

    print(f"\n By Security:")
    for sec, cv in comp_var.items():
        desk = hierarchy.loc[sec, "Desk"]
        book = hierarchy.loc[sec, "Book"]
        print(f"  {sec:<20}  {cv*100:+.3f}%  ({cv/total_var*100:.1f}%)  "
              f"[{desk} / {book}]")

    print(f"\n By Book:")
    for book, cv in comp_var.groupby(hierarchy["Book"]).sum().sort_values().items():
        print(f"  {book:<20}  {cv*100:+.3f}%  ({cv/total_var*100:.1f}%)")

    print(f"\n By Desk:")
    desk_var = comp_var.groupby(hierarchy["Desk"]).sum().sort_values()
    for desk, cv in desk_var.items():
        print(f"  {desk:<20}  {cv*100:+.3f}%  ({cv/total_var*100:.1f}%)")

    assert abs(desk_var.sum() - total_var) < 1e-10, "Desk rollup error"

    # ── 4. Backtest ────────────────────────────────────────────────────────
    # Rolling out-of-sample forecasts — each uses only data up to t-1.
    # Binomial: is breach occurrence correct?
    # Christoffersen: are breaches independent (e.g. not crisis-clustered)?
    # CVaR exceedance: when we breach, does the model size the loss correctly?
    window = min(250, T // 2)
    var_series = re.rolling_var(returns, window=window, confidence=confidence_level, method="historical")
    var_series_fhs = re.rolling_var(returns, window=window, confidence=confidence_level, method="fhs")
    cvar_series = re.rolling_cvar(returns, window=window, confidence=confidence_level)
    valid = ~np.isnan(var_series)

    print(f"\n[Backtest — rolling {window}-day window @ {confidence_level}]")
    print(re.backtest_summary(returns[valid], var_series[valid], cvar_forecasts=cvar_series[valid]))

    # ── 5. FX decomposition ───────────────────────────────────────────────
    # r_total = r_equity + r_fx (exact log decomposition).
    # FX add-on: how much does being unhedged cost (or save) in VaR terms?
    # Diversification: are equity and FX moves offsetting each other?
    r_fx = returns - returns_equity

    var_total  = re.var_historical(returns, confidence_level)
    var_equity = re.var_historical(returns_equity, confidence_level)

    cvar_total  = re.cvar(returns, confidence_level)
    cvar_equity = re.cvar(returns_equity, confidence_level)
    var_fx_only  = re.var_historical(r_fx, confidence_level)
    cvar_fx_only = re.cvar(r_fx, confidence_level)

    print(f"\n[FX Risk Decomposition @ 99%]")
    print(f" {'':30}  {'VaR':>7}  {'CVaR':>7}")
    print(f" {'-'*47}")
    for label, r in [
        ("Total (equity + FX)", returns),
        ("Equity (FX frozen)", returns_equity),
        ("FX (equity frozen)", r_fx),
    ]:
        print(f" {label:<30}  "
              f"{re.var_historical(r, confidence_level)*100:>6.3f}%  "
              f"{re.cvar(r, confidence_level)*100:>6.3f}%")
    print(f" {'-'*47}")
    print(f" FX add-on  (total - equity)    "
          f"{(var_total - var_equity)*100:>+6.3f}%")

    print(f"\n{'='*60}\n")

    # Return everything the Excel builder needs
    return dict(
        snapshot_date  = snapshot_date,
        return_dates   = return_dates,
        returns        = returns,
        returns_equity = returns_equity,
        r_fx           = r_fx,
        sigma          = sigma,
        var_hist       = var_hist,
        var_fhs        = var_fhs_,
        cvar_hist      = cvar_hist,
        cvar_fhs       = cvar_fhs_,
        cov            = cov,
        tickers        = tickers,
        weights        = weights,
        hierarchy      = hierarchy,
        bts            = re.backtest_summary(returns[valid], var_series[valid],
        cvar_forecasts = cvar_series[valid]),
        window         = window,
        var_series     = var_series,
        var_series_fhs = var_series_fhs,
        cvar_series    = cvar_series,
        var_total      = var_total,
        var_equity     = var_equity,
        cvar_total     = cvar_total,
        cvar_equity    = cvar_equity,
        var_fx_only    = var_fx_only,
        cvar_fx_only   = cvar_fx_only,
        T              = T,
    )


# ── Sample data generator ──────────────────────────────────────────────────

def make_sample_csvs(output_dir):
    rng = np.random.default_rng(42)

    # (Security, Currency, Quantity, StartPrice, DailyVol, Desk, Book, Strategy)
    securities = [
        ("AAPL", "USD", 1000, 150.0, 0.018, "Americas", "US_Tech", "Momentum"),
        ("MSFT", "USD",  500, 280.0, 0.016, "Americas", "US_Tech", "Value"),
        ("ASML.AS", "EUR",  200, 600.0, 0.017, "EMEA", "EU_Semis", "Growth"),
        ("7203.T", "JPY", 5000, 2000.0,0.015, "APAC", "JP_Auto", "Value" ),
        ("BNP.PA", "EUR", 700, 83.0, 0.015, "EMEA", "EU_Banks", "Value"),
        ("SHELL.AS", "EUR", 500, 41.0, 0.015, "EMEA", "EU_Energy", "Momentum"),
    ]
    start_fx = {"USD": 1.0, "EUR": 1.08, "JPY": 0.0067}
    fx_vols = {"USD": 0.0, "EUR": 0.005, "JPY": 0.004}

    dates  = pd.bdate_range("2021-04-01", "2025-12-08")
    prices = {s: p for s, _, _, p, _, _, _, _ in securities}
    fx = dict(start_fx)

    snapshot_date = dates[-1].date()   # last business date = "today"
    pos_rows = [
        {"Date": snapshot_date, "Desk": desk, "Book": book, "Strategy": strat,
         "Security": sec, "Quantity": qty, "Price_Currency": ccy}
        for sec, ccy, qty, _, _, desk, book, strat in securities
    ]
    price_rows, fx_rows = [], []

    for d in dates:
        for sec, ccy, qty, _, vol, *_ in securities:
            prices[sec] *= np.exp(rng.normal(0.0003, vol))
        for ccy in ["EUR", "JPY"]:
            fx[ccy] *= np.exp(rng.normal(0.0, fx_vols[ccy]))

        for sec, ccy, *_ in securities:
            price_rows.append({"Date": d.date(), "Security": sec,
                                "Price": round(prices[sec], 4),
                                "Price_Currency": ccy})
        for ccy, rate in fx.items():
            fx_rows.append({"Date": d.date(), "Currency": ccy,
                             "FX_to_Report_Ccy": round(rate, 6)})

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(pos_rows).to_csv(os.path.join(output_dir, "positions.csv"), index=False)
    pd.DataFrame(price_rows).to_csv(os.path.join(output_dir, "prices.csv"), index=False)
    pd.DataFrame(fx_rows).to_csv(os.path.join(output_dir, "fx_rates.csv"), index=False)
    print(f"Sample data written to {output_dir}/\n")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", default=os.path.join(DATA_DIR, "positions.csv"))
    parser.add_argument("--prices", default=os.path.join(DATA_DIR, "prices.csv"))
    parser.add_argument("--fx", default=os.path.join(DATA_DIR, "fx_rates.csv"))
    parser.add_argument("--sample", action="store_true", help="Generate sample CSVs and run")
    parser.add_argument("--no-excel", action="store_true", help="Skip Excel report generation")
    args = parser.parse_args()

    if args.sample:
        make_sample_csvs(DATA_DIR)

    positions = load_positions(args.positions)
    prices = load_prices(args.prices)
    fx = load_fx(args.fx)

    nav, nav_equity, mv_wide, hierarchy = build_nav(positions, prices, fx)

    returns = _log_returns(nav)
    returns_equity = _log_returns(nav_equity)
    asset_returns = np.log(mv_wide / mv_wide.shift(1)).dropna().values
    weights = snapshot_weights(positions, prices, fx, mv_wide.columns.tolist())

    print(f"[Weights from snapshot]")
    for sec, w in zip(mv_wide.columns, weights):
        print(f" {sec:<20}  {w*100:.1f}%")

    data = run_risk_report(
        returns, returns_equity, asset_returns,
        weights, mv_wide.columns.tolist(), hierarchy,
        snapshot_date=positions["Date"].iloc[0],
        return_dates=nav.index[1:],
    )

    if not args.no_excel:
        from build_report import build_excel_report
        out_path = os.path.join(OUT_DIR, f"{positions["Date"].iloc[0].strftime('%Y%m%d')}_risk_report.xlsx")
        build_excel_report(data, out_path)
        print(f"\nExcel report saved to {out_path}")