# trading_script.py
# (version corrigée: CLI + mode non interactif + fix prompt week-end)

import os
import sys
import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


DATA_DIR = Path.cwd()
PORTFOLIO_FILE_NAME = "chatgpt_portfolio_update.csv"
TRADE_LOG_FILE_NAME = "chatgpt_trade_log.csv"


def set_data_dir(p: Path) -> None:
    global DATA_DIR
    DATA_DIR = p


def portfolio_csv_path() -> Path:
    return DATA_DIR / PORTFOLIO_FILE_NAME


def trade_log_csv_path() -> Path:
    return DATA_DIR / TRADE_LOG_FILE_NAME


def load_latest_portfolio_state(file: str) -> Tuple[List[Dict], float]:
    """
    Reads the latest row from the portfolio CSV, returning (holdings_list, cash).
    The CSV expected columns: Date, Cash, Equity, SPX_100, Holdings (json list).
    """
    p = Path(file)
    if not p.exists() or p.stat().st_size == 0:
        # nothing there yet; start blank
        return [], 0.0

    df = pd.read_csv(p)
    if df.empty:
        return [], 0.0

    last = df.iloc[-1]
    cash = float(last.get("Cash", 0.0))
    holdings_raw = last.get("Holdings", "[]")
    try:
        holdings = json.loads(holdings_raw) if isinstance(holdings_raw, str) else holdings_raw
    except Exception:
        holdings = []
    return holdings, cash


def save_portfolio_row(date: datetime, cash: float, equity: float, spx_100: Optional[float], holdings: List[Dict]) -> None:
    out = portfolio_csv_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "Date": date.strftime("%Y-%m-%d"),
        "Cash": round(float(cash), 2),
        "Equity": round(float(equity), 2) if equity == equity else "",  # keep empty if NaN
        "SPX_100": round(float(spx_100), 2) if spx_100 is not None and spx_100 == spx_100 else "",
        "Holdings": json.dumps(holdings, separators=(",", ":")),
    }
    if out.exists() and out.stat().st_size > 0:
        df = pd.read_csv(out)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out, index=False)


def append_trade_log(trade: Dict) -> None:
    out = trade_log_csv_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {**trade}
    if out.exists() and out.stat().st_size > 0:
        df = pd.read_csv(out)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out, index=False)


def today_utc_date() -> datetime:
    # Use UTC for CI determinism
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def fetch_price(ticker: str) -> Optional[float]:
    """
    Get latest close price for a ticker using yfinance.
    Returns None on failure.
    """
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1d")
        if hist is None or hist.empty:
            print(f"No data for {ticker}")
            return None
        price = float(hist["Close"].iloc[-1])
        return price
    except Exception as e:
        # Mirror the logs for visibility
        print(f"Failed to get ticker '{ticker}' reason: {e}")
        return None


def benchmark_spx100(start_value: float = 100.0) -> Optional[float]:
    """
    Compute the value of $start_value invested in S&P500.
    Try ^GSPC -> SPY -> ^SPX; return None if all fail.
    """
    tickers = ["^GSPC", "SPY", "^SPX"]
    for t in tickers:
        try:
            data = yf.Ticker(t).history(period="1d")
            if data is None or data.empty:
                print(f"Data for {t} was empty or incomplete.")
                continue
            last = float(data["Close"].iloc[-1])
            # Normalize to start_value using the first close we could find today (here 1:1)
            return start_value * (last / last)
        except Exception as e:
            print(f"Failed to get ticker '{t}' reason: {e}")
            print(f"\n1 Failed download:\n['{t}']: {type(e).__name__}('{e}')")
    print("S&P 500 data unavailable; skipping benchmark.")
    return None


def compute_equity(holdings: List[Dict], cash: float) -> float:
    total = float(cash)
    for h in holdings:
        t = h.get("ticker")
        shares = float(h.get("shares", 0.0))
        px = fetch_price(t) if t else None
        if px is None:
            print(f"Data for {t} was empty or incomplete.")
            continue
        total += shares * px
    return total


def pretty_print_holdings(holdings: List[Dict]) -> None:
    if not holdings:
        print("No holdings.")
        return
    print("Ticker Shares Buy Price Cost Basis Stop Loss")
    for h in holdings:
        print(f"  {h.get('ticker'):4s} {float(h.get('shares',0)):4.1f} "
              f"{float(h.get('buy_price',0)):.2f} {float(h.get('cost_basis',0)):.2f} "
              f"{float(h.get('stop_loss',0)):.2f}")


def process_portfolio(holdings: List[Dict], cash: float, interactive: bool = True) -> Tuple[List[Dict], float]:
    """
    Core portfolio step:
    - Non-interactif: aucun input(), on ne fait que valoriser les positions existantes.
    - Interactif: prompts pour ajouter/enlever des lignes (manuel).
    """
    # Week-end info/prompt — corriger la priorité des opérateurs
    day = today_utc_date().weekday()  # 0=Mon ... 5=Sat, 6=Sun
    if (day == 6 or day == 5) and interactive:
        try:
            _ = input("C'est le week-end. Appuie sur Entrée pour continuer (les prix d'aujourd'hui seront peut-être faux)... ")
        except Exception:
            pass

    if interactive:
        try:
            choice = input("Ajouter une transaction manuelle ? (buy/sell/skip): ").strip().lower()
        except Exception:
            choice = "skip"
        while choice in ("buy", "sell"):
            try:
                ticker = input("Ticker: ").strip().upper()
                shares = float(input("Shares (nombre): ").strip())
                price = fetch_price(ticker) or float(input("Prix à utiliser (fallback): ").strip())
            except Exception:
                print("Entrée invalide, on sort des prompts...")
                break

            if choice == "buy":
                cost = shares * price
                if cost > cash:
                    print("Pas assez de cash.")
                else:
                    # append holding (regroupe simple si même ticker)
                    found = False
                    for h in holdings:
                        if h.get("ticker") == ticker:
                            h["shares"] = float(h.get("shares", 0)) + shares
                            h["cost_basis"] = float(h.get("cost_basis", 0)) + cost
                            found = True
                            break
                    if not found:
                        holdings.append({
                            "ticker": ticker,
                            "shares": shares,
                            "buy_price": price,
                            "cost_basis": cost,
                            "stop_loss": round(price * 0.85, 2),
                        })
                    cash -= cost
                    append_trade_log({
                        "Date": today_utc_date().strftime("%Y-%m-%d"),
                        "Type": "BUY",
                        "Ticker": ticker,
                        "Shares": shares,
                        "Price": round(price, 4),
                        "Cash_After": round(cash, 2),
                    })
            else:  # sell
                idx = next((i for i, h in enumerate(holdings) if h.get("ticker") == ticker), None)
                if idx is None:
                    print("Position introuvable.")
                else:
                    lot = holdings[idx]
                    sell_qty = min(shares, float(lot.get("shares", 0)))
                    proceeds = sell_qty * price
                    lot["shares"] = float(lot.get("shares", 0)) - sell_qty
                    if lot["shares"] <= 0:
                        holdings.pop(idx)
                    cash += proceeds
                    append_trade_log({
                        "Date": today_utc_date().strftime("%Y-%m-%d"),
                        "Type": "SELL",
                        "Ticker": ticker,
                        "Shares": sell_qty,
                        "Price": round(price, 4),
                        "Cash_After": round(cash, 2),
                    })

            pretty_print_holdings(holdings)
            try:
                choice = input("Autre transaction ? (buy/sell/skip): ").strip().lower()
            except Exception:
                choice = "skip"

    return holdings, cash


def daily_results(holdings: List[Dict], cash: float) -> None:
    date = today_utc_date()
    spx_val = benchmark_spx100(100.0)  # peut être None si indispo
    equity = compute_equity(holdings, cash)
    pretty_print_holdings(holdings)
    print(f"cash balance: {cash:.2f}")
    print("Here are is your update for today. You can make any changes you see fit (if necessary),")
    print("but you may not use deep research. You do have to ask premissons for any changes, as you have full control.")
    print("You can however use the Internet and check current prices for potenial buys.*")
    save_portfolio_row(date, cash, equity, spx_val, holdings)


def compute_metrics_safe() -> None:
    """
    Try to compute Sharpe/Sortino from the CSV; if missing Equity, skip quietly.
    """
    try:
        df = pd.read_csv(portfolio_csv_path(), usecols=["Date", "Equity"])
        if df.empty or df["Equity"].isna().all():
            raise ValueError("No Equity values")
        series = df["Equity"].pct_change().dropna()
        if series.empty:
            raise ValueError("Not enough data")
        sharpe = (series.mean() / (series.std() + 1e-12)) * math.sqrt(252)
        neg = series[series < 0]
        sortino = (series.mean() / (neg.std() + 1e-12)) * math.sqrt(252)
        print(f"Total Sharpe Ratio over {len(series)} days: {sharpe/ math.sqrt(252):.4f}")
        print(f"Total Sortino Ratio over {len(series)} days: {sortino/ math.sqrt(252):.4f}")
        print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
        print(f"Annualized Sortino Ratio: {sortino:.4f}")
    except Exception as e:
        print(f"Could not compute Sharpe/Sortino: {e}")


def main(file: str, data_dir: Path | None = None, interactive: bool | None = None) -> None:
    """Run the trading script."""
    chatgpt_portfolio, cash = load_latest_portfolio_state(file)
    if data_dir is not None:
        set_data_dir(data_dir)

    # decide interactivity: explicit flag wins; otherwise CI/non-TTY -> False
    if interactive is None:
        interactive = not (
            os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("CI") == "true"
            or not sys.stdin.isatty()
        )

    chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash, interactive=interactive)
    daily_results(chatgpt_portfolio, cash)
    compute_metrics_safe()


# ---- CLI entry point ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maintain/compute daily results for the ChatGPT micro-cap portfolio.")
    parser.add_argument("-f", "--file", required=True,
                        help="Path to the portfolio CSV containing historical records.")
    parser.add_argument("-d", "--data-dir", default=None,
                        help="Directory to write chatgpt_portfolio_update.csv and chatgpt_trade_log.csv")
    ig = parser.add_mutually_exclusive_group()
    ig.add_argument("--interactive", dest="interactive", action="store_true",
                    help="Force prompts for manual trades.")
    ig.add_argument("--non-interactive", dest="interactive", action="store_false",
                    help="Disable prompts (useful for CI).")
    parser.set_defaults(interactive=None)
    args = parser.parse_args()

    main(args.file, Path(args.data_dir) if args.data_dir else None, interactive=args.interactive)
