#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trading_script.py — CLI-friendly portfolio updater for the Nathan-Smith-style experiment.

What this script does
---------------------
- Loads the most recent portfolio "snapshot" and cash from the CSV you pass with -f / --file.
- Optionally asks for manual trades if run in interactive mode (default when run in a terminal).
- Pulls latest prices for current holdings using yfinance and computes PnL and equity.
- Appends today's rows to the CSV and writes/updates a trade log alongside it.
- Prints one concise summary for copy/paste into the LLM.

It is designed to be resilient on CI:
- Use `--non-interactive` to skip all prompts.
- Price downloads are best-effort: tickers that fail are skipped, benchmark is optional.

CSV expectations
----------------
This script is compatible with the "Start Your Own" folder format from the repo:
Columns used when **writing**:
  Date, Ticker, Shares, Buy Price, Cost Basis, Stop Loss, Current Price, Total Value, PnL, Action, Cash Balance, Total Equity

When **reading**, we only require these to recover holdings from the **last date block**:
  Date, Ticker, Shares, Buy Price, Cost Basis, Stop Loss, Cash Balance, Total Equity
Rows having a Ticker value form the holdings; a single row with a Cash Balance/Total Equity holds summary.

Author: ChatGPT helper
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional, cast

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # yfinance not available; we will degrade gracefully

# ------------------------------ Globals ---------------------------------------

DATA_DIR: Path = Path.cwd()

def set_data_dir(dirpath: Optional[Path]) -> None:
    global DATA_DIR
    if dirpath:
        DATA_DIR = dirpath
    else:
        DATA_DIR = Path.cwd()

# ----------------------------- Small helpers ----------------------------------

def today_ymd() -> pd.Timestamp:
    # We store dates naive in CSV (no tz). Use current UTC date.
    return pd.Timestamp.utcnow().normalize()

def to_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def robust_history(symbol: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Fetch daily OHLC history for a symbol in a resilient way.
    Returns an empty DataFrame if nothing can be fetched.
    """
    if yf is None:
        return pd.DataFrame()

    # yfinance sometimes 429/JSON errors on CI; try both Ticker.history and download.
    try:
        t = yf.Ticker(symbol)
        df = t.history(start=start, end=end, period="1d", auto_adjust=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass

    try:
        df2 = yf.download(symbol, start=start, end=end, period="1d", progress=False, auto_adjust=False, threads=False)
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            return df2
    except Exception:
        pass

    return pd.DataFrame()

@dataclass
class Holding:
    ticker: str
    shares: float
    buy_price: float
    cost_basis: float
    stop_loss: Optional[float] = None

def _normalize_ticker(x: str) -> str:
    return (x or "").strip().upper()

def _is_holding_row(row: pd.Series) -> bool:
    t = str(row.get("Ticker", "")).strip()
    sh = row.get("Shares", None)
    return bool(t) and (not pd.isna(sh))

def _extract_last_block(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    From a whole CSV, find the most recent 'Date' value and return:
      - holdings DataFrame (ticker rows for that date)
      - cash balance (from the summary row)
    """
    if df.empty:
        return pd.DataFrame(columns=["Ticker","Shares","Buy Price","Cost Basis","Stop Loss"]), 0.0

    # Ensure Date is parsed
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        last_dt = cast(pd.Timestamp, df["Date"].dropna().max())
        block = df[df["Date"] == last_dt].copy()
    else:
        # No dates? take last contiguous block of non-NaN Ticker rows at the end
        block = df.tail(1000).copy()

    holdings = block[block.apply(_is_holding_row, axis=1)].copy()

    # Cash balance from any row that has it on that date
    cash_series = pd.to_numeric(block.get("Cash Balance", pd.Series(dtype=float)), errors="coerce")
    cash_val = float(cash_series.dropna().iloc[-1]) if not cash_series.dropna().empty else 0.0

    # Standardize column names
    for col in ["Shares", "Buy Price", "Cost Basis", "Stop Loss"]:
        if col not in holdings.columns:
            holdings[col] = np.nan

    holdings = holdings[["Ticker","Shares","Buy Price","Cost Basis","Stop Loss"]].copy()
    holdings["Ticker"] = holdings["Ticker"].astype(str).map(_normalize_ticker)
    holdings["Shares"] = pd.to_numeric(holdings["Shares"], errors="coerce").fillna(0.0)
    holdings["Buy Price"] = pd.to_numeric(holdings["Buy Price"], errors="coerce").fillna(0.0)
    holdings["Cost Basis"] = pd.to_numeric(holdings["Cost Basis"], errors="coerce").fillna(0.0)
    if "Stop Loss" in holdings.columns:
        holdings["Stop Loss"] = pd.to_numeric(holdings["Stop Loss"], errors="coerce")

    return holdings, cash_val

# ----------------------------- I/O functions ----------------------------------

def load_latest_portfolio_state(file: str) -> Tuple[List[Holding], float]:
    """
    Reads the CSV and returns the last day's portfolio (list of Holding) and cash.
    If the file doesn't exist or is empty, returns empty list and cash=0.
    """
    p = Path(file)
    if not p.exists() or p.stat().st_size == 0:
        return [], 0.0

    df = pd.read_csv(p)
    holdings_df, cash = _extract_last_block(df)

    holdings: List[Holding] = []
    for _, r in holdings_df.iterrows():
        holdings.append(
            Holding(
                ticker=str(r["Ticker"]).upper(),
                shares=float(r["Shares"]),
                buy_price=float(r["Buy Price"]),
                cost_basis=float(r["Cost Basis"]),
                stop_loss=to_float(r.get("Stop Loss")),
            )
        )
    # Show what we loaded (useful for debugging CI)
    if not holdings:
        print("No holdings found in CSV; continuing with empty portfolio.")
    else:
        print(holdings_df.to_string(index=False))
    return holdings, float(cash)

def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        # Append preserving columns
        existing = pd.read_csv(path)
        missing_cols = [c for c in existing.columns if c not in df.columns]
        for c in missing_cols:
            df[c] = np.nan
        df = df[existing.columns]
        out = pd.concat([existing, df], ignore_index=True)
    else:
        out = df
    out.to_csv(path, index=False)

def _append_trade_log(ticker: str, side: str, qty: float, price: float) -> None:
    log_path = DATA_DIR / "chatgpt_trade_log.csv"
    row = {
        "Date": today_ymd().date().isoformat(),
        "Ticker": ticker,
        "Side": side,
        "Qty": qty,
        "Price": price,
    }
    if log_path.exists():
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(log_path, index=False)

# ----------------------------- Core logic -------------------------------------

def process_portfolio(portfolio: Iterable[Holding] | pd.DataFrame | List[Dict[str, object]], cash: float, *, interactive: bool=True) -> Tuple[List[Holding], float]:
    """
    Applies any automatic rules (stop-loss) and optional manual prompts.
    Returns the possibly-updated portfolio and cash balance.
    """
    # Normalize to list[Holding]
    if isinstance(portfolio, pd.DataFrame):
        holdings: List[Holding] = [
            Holding(
                ticker=str(r["Ticker"]).upper(),
                shares=float(r["Shares"]),
                buy_price=float(r["Buy Price"]),
                cost_basis=float(r["Cost Basis"]),
                stop_loss=to_float(r.get("Stop Loss")),
            )
            for _, r in portfolio.iterrows()
        ]
    elif isinstance(portfolio, list) and portfolio and isinstance(portfolio[0], dict):
        holdings = [
            Holding(
                ticker=str(r.get("ticker","")).upper(),
                shares=float(r.get("shares",0)),
                buy_price=float(r.get("buy_price",0)),
                cost_basis=float(r.get("cost_basis",0)),
                stop_loss=to_float(r.get("stop_loss")),
            )
            for r in cast(List[Dict[str, object]], portfolio)
        ]
    else:
        holdings = list(cast(Iterable[Holding], portfolio))

    # Weekend warning (only when interactive)
    if interactive:
        weekday = today_ymd().weekday()  # Mon=0, Sun=6
        if weekday in (5, 6):  # Sat/Sun
            ans = input(
                "⚠️ Marchés fermés ce week-end. On utilisera le dernier cours de clôture disponible.\n"
                "Continuer ? [Entrée] ou taper 1 pour quitter: "
            )
            if ans.strip() == "1":
                raise SystemExit("Exit by user.")

    # For this quick script we don't auto-execute buys/sells except stop-loss on current price check.
    # Fetch last close for each ticker and apply stop-loss if triggered.
    updated: List[Holding] = []
    for h in holdings:
        price = None
        df = robust_history(h.ticker)
        if not df.empty and "Close" in df.columns:
            try:
                price = float(df["Close"].iloc[-1])
            except Exception:
                price = None

        if h.stop_loss is not None and price is not None and price <= h.stop_loss and h.shares > 0:
            # Stop triggered: sell all
            cash += float(price * h.shares)
            _append_trade_log(h.ticker, "SELL", h.shares, price)
            # Do not keep in holdings
        else:
            updated.append(h)

    # Optional manual trades (skipped on CI)
    if interactive:
        # Very simple prompt loop
        while True:
            cmd = input("Souhaitez-vous enregistrer un trade manuel ? (buy/sell/none): ").strip().lower()
            if cmd in ("n", "no", "non", "none", ""):
                break
            if cmd not in ("buy", "sell"):
                print("Entrez 'buy', 'sell' ou 'none'.")
                continue
            ticker = input("Ticker: ").strip().upper()
            qty = float(input("Quantité: ").strip())
            px = float(input("Prix: ").strip())
            if cmd == "buy":
                # Add or increase position
                found = next((x for x in updated if x.ticker == ticker), None)
                if found:
                    total_shares = found.shares + qty
                    new_cb = (found.cost_basis * found.shares + px * qty) / max(total_shares, 1e-9)
                    found.shares = total_shares
                    found.buy_price = px
                    found.cost_basis = new_cb
                else:
                    updated.append(Holding(ticker=ticker, shares=qty, buy_price=px, cost_basis=px))
                cash -= qty * px
                _append_trade_log(ticker, "BUY", qty, px)
            else:
                # SELL
                found = next((x for x in updated if x.ticker == ticker), None)
                if not found:
                    print("Position introuvable.")
                else:
                    sell_qty = min(qty, found.shares)
                    found.shares -= sell_qty
                    cash += sell_qty * px
                    _append_trade_log(ticker, "SELL", sell_qty, px)
                    if found.shares <= 1e-9:
                        updated = [x for x in updated if x.ticker != ticker]

    return updated, float(cash)

def _compute_equity(holdings: List[Holding], cash: float) -> Tuple[float, pd.DataFrame]:
    """
    Get last prices for holdings and compute current value & PnL rows for CSV write.
    """
    rows: List[Dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0

    date_str = today_ymd().date().isoformat()

    for h in holdings:
        price = None
        data = robust_history(h.ticker)
        if not data.empty and "Close" in data.columns:
            try:
                price = float(data["Close"].iloc[-1])
            except Exception:
                price = None

        if price is None:
            print(f"No data for {h.ticker}")
            price = np.nan
            value = 0.0
            pnl = 0.0
            action = "DATA_MISSING"
        else:
            value = float(price * h.shares)
            pnl = float((price - h.cost_basis) * h.shares)
            action = "HOLD"

        total_value += value
        total_pnl += pnl

        rows.append({
            "Date": date_str,
            "Ticker": h.ticker,
            "Shares": h.shares,
            "Buy Price": h.buy_price,
            "Cost Basis": h.cost_basis,
            "Stop Loss": h.stop_loss if h.stop_loss is not None else "",
            "Current Price": price,
            "Total Value": round(value, 2),
            "PnL": round(pnl, 2),
            "Action": action,
            "Cash Balance": "",
            "Total Equity": "",
        })

    equity = float(total_value + cash)
    # Add one summary row (no ticker) with cash and equity
    rows.append({
        "Date": date_str,
        "Ticker": "",
        "Shares": "",
        "Buy Price": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": round(total_value, 2),
        "PnL": round(total_pnl, 2),
        "Action": "TOTAL",
        "Cash Balance": round(cash, 2),
        "Total Equity": round(equity, 2),
    })

    return equity, pd.DataFrame(rows)

def _maybe_benchmark_print(file_csv: Path) -> None:
    """
    Attempt to print an S&P 500 baseline value for $100 invested at the first date present in CSV.
    Safe to call even if Yahoo blocks us; we'll just print a notice.
    """
    if yf is None:
        print("S&P 500 baseline skipped (no yfinance installed).")
        return

    # Prefer ^GSPC, then SPY
    for sym in ("^GSPC", "SPY", "^SPX"):
        df = robust_history(sym)
        if not df.empty and "Close" in df.columns:
            # Try to compute since the earliest date in CSV
            start_dt: Optional[pd.Timestamp] = None
            try:
                if file_csv.exists():
                    dfp = pd.read_csv(file_csv)
                    if "Date" in dfp.columns:
                        dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")
                        start_dt = cast(pd.Timestamp, dfp["Date"].dropna().min())
            except Exception:
                start_dt = None

            if start_dt is not None:
                end_dt = today_ymd() + pd.Timedelta(days=1)
                hist = robust_history(sym, start=start_dt, end=end_dt)
            else:
                hist = df

            if not hist.empty and "Close" in hist.columns and len(hist) >= 2:
                initial = float(hist["Close"].iloc[0])
                latest = float(hist["Close"].iloc[-1])
                if initial > 0:
                    value = 100.0 * latest / initial
                    print(f"$100 Invested in the S&P 500: ${value:.2f}")
                    return
        # next symbol fallback
    print("S&P 500 data unavailable; skipping benchmark.")

def daily_results(holdings: List[Holding], cash: float, csv_path: Path) -> None:
    """
    Compute today's results, write CSVs, and print a concise summary.
    """
    equity, rows_df = _compute_equity(holdings, cash)

    # Persist
    _write_rows(csv_path, rows_df.to_dict(orient="records"))

    # Print summary for the operator / LLM
    print(f"prices and updates for {today_ymd().date().isoformat()}")
    _maybe_benchmark_print(csv_path)
    print(f"Latest ChatGPT Equity: ${equity:.2f}")
    print("today's portfolio:")
    printable = rows_df[rows_df["Action"] != "TOTAL"][["Ticker","Shares","Buy Price","Cost Basis","Stop Loss"]]
    print(printable.to_string(index=False))
    # Cash line
    print(f"cash balance: {cash:.2f}")
    # Instruction blurb (kept from the original style)
    print(
        "Here are is your update for today. You can make any changes you see fit (if necessary),\n"
        "but you may not use deep research. You do have to ask premissons for any changes, as you have full control.\n"
        "You can however use the Internet and check current prices for potenial buys.*"
    )

# ----------------------------- CLI / main -------------------------------------

def main(file: str, data_dir: Optional[Path] = None, interactive: Optional[bool] = None) -> None:
    """
    Run the trading script end-to-end.
    """
    if data_dir is not None:
        set_data_dir(data_dir)

    # Decide interactivity
    if interactive is None:
        interactive = not (
            os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("CI") == "true"
            or not sys.stdin.isatty()
        )

    holdings, cash = load_latest_portfolio_state(file)
    holdings, cash = process_portfolio(holdings, cash, interactive=interactive)

    # Where to write
    csv_path = Path(file)
    daily_results(holdings, cash, csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maintain/compute daily results for the ChatGPT micro-cap portfolio.")
    parser.add_argument("-f", "--file", required=True,
                        help="Path to the portfolio CSV containing historical records.")
    parser.add_argument("-d", "--data-dir", default=None,
                        help="Directory to write chatgpt_trade_log.csv (defaults to current dir)")
    ig = parser.add_mutually_exclusive_group()
    ig.add_argument("--interactive", dest="interactive", action="store_true",
                    help="Force prompts for manual trades.")
    ig.add_argument("--non-interactive", dest="interactive", action="store_false",
                    help="Disable prompts (useful for CI).")
    parser.set_defaults(interactive=None)
    args = parser.parse_args()

    main(args.file, Path(args.data_dir) if args.data_dir else None, interactive=args.interactive)
