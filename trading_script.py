# trading_script.py
# Mode CI non-interactif, chargement CSV robuste, exécution d'ordres du jour
# via "pending_orders.csv" (Date,Type,Ticker,Shares,Price).
# Fallback de prix si yfinance indispo (last_price/buy_price).

import os
import sys
import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path.cwd()
PORTFOLIO_FILE_NAME = "chatgpt_portfolio_update.csv"
TRADE_LOG_FILE_NAME = "chatgpt_trade_log.csv"
PENDING_ORDERS_FILE = "pending_orders.csv"  # dans DATA_DIR

def set_data_dir(p: Path) -> None:
    global DATA_DIR
    DATA_DIR = p

def portfolio_csv_path() -> Path:
    return DATA_DIR / PORTFOLIO_FILE_NAME

def trade_log_csv_path() -> Path:
    return DATA_DIR / TRADE_LOG_FILE_NAME

def pending_orders_path() -> Path:
    return DATA_DIR / PENDING_ORDERS_FILE

def today_utc_date() -> datetime:
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x).strip())
    except Exception:
        return default

def load_latest_portfolio_state(file: str) -> Tuple[List[Dict], float]:
    p = Path(file)
    if not p.exists() or p.stat().st_size == 0:
        return [], 100.0  # défaut sain si rien n'existe

    df = pd.read_csv(p)

    if df.empty:
        return [], 100.0

    # cash: dernière valeur valide sinon 100
    cash = 100.0
    if "Cash" in df.columns:
        cash_series = pd.to_numeric(df["Cash"], errors="coerce")
        valid = cash_series.dropna()
        if not valid.empty:
            cash = float(valid.iloc[-1])

    # holdings: dernier JSON non vide sinon []
    holdings = []
    if "Holdings" in df.columns:
        # On remonte du bas jusqu'au premier champ non-null/non-vide
        for v in reversed(df["Holdings"].tolist()):
            if isinstance(v, str) and v.strip():
                try:
                    holdings = json.loads(v)
                    break
                except Exception:
                    continue

    return holdings or [], cash

def save_portfolio_row(date: datetime, cash: float, equity: float,
                       spx_100: Optional[float], holdings: List[Dict]) -> None:
    out = portfolio_csv_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "Date": date.strftime("%Y-%m-%d"),
        "Cash": round(float(cash), 2),
        "Equity": "" if pd.isna(equity) else round(float(equity), 2),
        "SPX_100": "" if (spx_100 is None or pd.isna(spx_100)) else round(float(spx_100), 2),
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

def fetch_price(ticker: str) -> Optional[float]:
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1d")
        if hist is None or hist.empty:
            print(f"{ticker}: No price data found, symbol may be delisted (period=1d)")
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"Failed to get ticker '{ticker}' reason: {e}")
        return None

def benchmark_spx100(start_value: float = 100.0) -> Optional[float]:
    for t in ["^GSPC", "SPY", "^SPX"]:
        try:
            data = yf.Ticker(t).history(period="1d")
            if data is None or data.empty:
                print(f"Data for {t} was empty or incomplete.")
                continue
            last = float(data["Close"].iloc[-1])
            return start_value * (last / last)  # 1:1 pour consistance
        except Exception as e:
            print(f"Failed to get ticker '{t}' reason: {e}")
            print(f"\n1 Failed download:\n['{t}']: {type(e).__name__}('{e}')")
    print("S&P 500 data unavailable; skipping benchmark.")
    return None

def compute_equity(holdings: List[Dict], cash: float) -> float:
    total = float(cash)
    for h in holdings:
        t = (h.get("ticker") or "").upper()
        shares = _safe_float(h.get("shares"), 0.0)
        px = fetch_price(t) if t else None
        if px is None:
            # Fallback CI: dernier prix connu ou buy_price
            px = _safe_float(h.get("last_price"), None)
            if px is None:
                px = _safe_float(h.get("buy_price"), 0.0)
        else:
            # Mets à jour last_price pour garder une trace
            h["last_price"] = round(px, 6)
        total += shares * px
    return total

def pretty_print_holdings(holdings: List[Dict]) -> None:
    if not holdings:
        print("No holdings.")
        return
    print("Ticker Shares Buy Price Cost Basis Stop Loss")
    for h in holdings:
        print(f"  {h.get('ticker',''):4s} "
              f"{_safe_float(h.get('shares'),0):4.1f} "
              f"{_safe_float(h.get('buy_price'),0):.2f} "
              f"{_safe_float(h.get('cost_basis'),0):.2f} "
              f"{_safe_float(h.get('stop_loss'),0):.2f}")

def _apply_buy(holdings: List[Dict], cash: float, ticker: str, shares: float, price: float) -> float:
    cost = shares * price
    if cost > cash + 1e-9:
        print(f"BUY {ticker}: not enough cash (need {cost:.2f}, have {cash:.2f})")
        return cash
    # aggréger si même ticker
    for h in holdings:
        if (h.get("ticker") or "").upper() == ticker:
            new_shares = _safe_float(h.get("shares"), 0.0) + shares
            new_cb = _safe_float(h.get("cost_basis"), 0.0) + cost
            h["shares"] = new_shares
            h["cost_basis"] = new_cb
            h["buy_price"] = _safe_float(h.get("buy_price"), price)  # garde le premier si déjà
            h["last_price"] = price
            break
    else:
        holdings.append({
            "ticker": ticker,
            "shares": shares,
            "buy_price": price,
            "cost_basis": cost,
            "stop_loss": round(price * 0.85, 4),
            "last_price": price,
        })
    cash -= cost
    append_trade_log({
        "Date": today_utc_date().strftime("%Y-%m-%d"),
        "Type": "BUY",
        "Ticker": ticker,
        "Shares": shares,
        "Price": round(price, 6),
        "Cash_After": round(cash, 2),
    })
    return cash

def _apply_sell(holdings: List[Dict], cash: float, ticker: str, shares: float, price: float) -> float:
    idx = next((i for i, h in enumerate(holdings) if (h.get("ticker") or "").upper() == ticker), None)
    if idx is None:
        print(f"SELL {ticker}: position not found.")
        return cash
    lot = holdings[idx]
    qty = min(shares, _safe_float(lot.get("shares"), 0.0))
    proceeds = qty * price
    lot["shares"] = _safe_float(lot.get("shares"), 0.0) - qty
    lot["last_price"] = price
    if lot["shares"] <= 1e-12:
        holdings.pop(idx)
    cash += proceeds
    append_trade_log({
        "Date": today_utc_date().strftime("%Y-%m-%d"),
        "Type": "SELL",
        "Ticker": ticker,
        "Shares": qty,
        "Price": round(price, 6),
        "Cash_After": round(cash, 2),
    })
    return cash

def apply_pending_orders_if_any(holdings: List[Dict], cash: float,
                                allow_fetch_price: bool = True) -> Tuple[List[Dict], float]:
    """Exécute uniquement les ordres dont Date == aujourd'hui (UTC)."""
    p = pending_orders_path()
    if not p.exists() or p.stat().st_size == 0:
        return holdings, cash

    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"Could not read pending orders: {e}")
        return holdings, cash

    required_cols = {"Date", "Type", "Ticker", "Shares"}
    if not required_cols.issubset(set(df.columns)):
        print(f"pending_orders.csv missing columns. Need: {required_cols}")
        return holdings, cash

    today_str = today_utc_date().strftime("%Y-%m-%d")
    day_orders = df[df["Date"].astype(str).str.strip() == today_str]
    if day_orders.empty:
        return holdings, cash

    for _, row in day_orders.iterrows():
        typ = str(row["Type"]).strip().upper()
        tkr = str(row["Ticker"]).strip().upper()
        sh = _safe_float(row["Shares"], 0.0)
        px = row["Price"] if "Price" in row and not (pd.isna(row["Price"])) else None
        price = _safe_float(px, None)

        if price is None and allow_fetch_price:
            price = fetch_price(tkr)
        if price is None:
            # Fall back: dernier last_price ou buy_price d’une éventuelle ligne existante
            lot = next((h for h in holdings if (h.get("ticker") or "").upper() == tkr), None)
            if lot:
                price = _safe_float(lot.get("last_price"), None)
                if price is None:
                    price = _safe_float(lot.get("buy_price"), None)
        if price is None:
            print(f"Skip {typ} {tkr}: no price available.")
            continue

        if typ == "BUY":
            cash = _apply_buy(holdings, cash, tkr, sh, price)
        elif typ == "SELL":
            cash = _apply_sell(holdings, cash, tkr, sh, price)
        else:
            print(f"Unknown order type '{typ}' for {tkr}, skipping.")

    return holdings, cash

def process_portfolio(holdings: List[Dict], cash: float, interactive: bool = True) -> Tuple[List[Dict], float]:
    # Week-end → prompt seulement si interactif
    day = today_utc_date().weekday()  # 0=Mon ... 6=Sun
    if (day in (5, 6)) and interactive:
        try:
            input("C'est le week-end. Entrée pour continuer...")
        except Exception:
            pass

    if not interactive:
        # Mode CI: exécute les ordres du jour si présents
        holdings, cash = apply_pending_orders_if_any(holdings, cash, allow_fetch_price=True)
    else:
        # Prompts manuels (facultatif)
        try:
            choice = input("Ajouter une transaction manuelle ? (buy/sell/skip): ").strip().lower()
        except Exception:
            choice = "skip"
        while choice in ("buy", "sell"):
            try:
                ticker = input("Ticker: ").strip().upper()
                shares = _safe_float(input("Shares (nombre): ").strip(), 0.0)
                price = fetch_price(ticker) or _safe_float(input("Prix à utiliser (fallback): ").strip(), 0.0)
            except Exception:
                print("Entrée invalide, on sort des prompts...")
                break
            if choice == "buy":
                cash = _apply_buy(holdings, cash, ticker, shares, price)
            else:
                cash = _apply_sell(holdings, cash, ticker, shares, price)

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
    print("Daily update complete.")
    save_portfolio_row(date, cash, equity, spx_val, holdings)

def compute_metrics_safe() -> None:
    try:
        df = pd.read_csv(portfolio_csv_path(), usecols=["Date", "Equity"])
        series = pd.to_numeric(df["Equity"], errors="coerce").pct_change().dropna()
        if series.empty:
            raise ValueError("Not enough data")
        sharpe = (series.mean() / (series.std() + 1e-12)) * math.sqrt(252)
        neg = series[series < 0]
        sortino = (series.mean() / (neg.std() + 1e-12)) * math.sqrt(252)
        print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
        print(f"Annualized Sortino Ratio: {sortino:.4f}")
    except Exception as e:
        print(f"Could not compute Sharpe/Sortino: {e}")

def main(file: str, data_dir: Optional[Path] = None, interactive: Optional[bool] = None) -> None:
    if data_dir is not None:
        set_data_dir(data_dir)

    # Décide interactivité si non spécifié
    if interactive is None:
        interactive = not (
            os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("CI") == "true"
            or not sys.stdin.isatty()
        )

    holdings, cash = load_latest_portfolio_state(file)
    holdings, cash = process_portfolio(holdings, cash, interactive=interactive)
    daily_results(holdings, cash)
    compute_metrics_safe()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI-friendly micro-cap portfolio runner.")
    parser.add_argument("-f", "--file", required=True,
                        help="Path to the portfolio CSV with historical records.")
    parser.add_argument("-d", "--data-dir", default=None,
                        help="Directory for outputs (CSV/logs).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--interactive", dest="interactive", action="store_true")
    group.add_argument("--non-interactive", dest="interactive", action="store_false")
    parser.set_defaults(interactive=None)
    args = parser.parse_args()

    main(args.file, Path(args.data_dir) if args.data_dir else None, interactive=args.interactive)
