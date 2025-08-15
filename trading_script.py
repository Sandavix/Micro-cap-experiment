#!/usr/bin/env python3
import argparse, os, sys, json, math
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import pandas as pd

# --- Market data: yfinance primary, Stooq fallback ---
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

DATA_ERRORS: List[str] = []
HARD_FAIL_ON_DATA_ERRORS = os.getenv("FAIL_ON_DATA_ERRORS", "false").lower() == "true"

def _now_utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def fetch_from_yahoo_last_close(ticker: str) -> Optional[float]:
    try:
        import yfinance as yf
        # try short period
        df = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            # fallback download
            df = yf.download(ticker, period="10d", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        # take last non-na close
        s = df["Close"].dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception as e:
        # yfinance can throw JSONDecodeError when Yahoo hiccups
        return None

def _to_stooq_symbol(ticker: str) -> str:
    t = ticker.strip().upper()
    # Index mappings commonly used
    if t in ("^GSPC", "^SPX"):
        return "^spx"         # Stooq index name
    # ETFs/equities: append .US if no suffix already
    if "." not in t and not t.startswith("^"):
        return f"{t}.US"
    return t

def fetch_from_stooq_last_close(ticker: str) -> Optional[float]:
    try:
        from pandas_datareader import data as pdr
        sym = _to_stooq_symbol(ticker)
        df = pdr.DataReader(sym, "stooq")
        if df is None or df.empty:
            return None
        # Stooq returns most-recent first; ensure latest close
        s = df["Close"].dropna()
        if s.empty:
            return None
        return float(s.iloc[0])
    except Exception:
        return None

def fetch_price(ticker: str) -> Optional[float]:
    px = fetch_from_yahoo_last_close(ticker)
    if px is not None and math.isfinite(px):
        return px
    px = fetch_from_stooq_last_close(ticker)
    if px is not None and math.isfinite(px):
        return px
    DATA_ERRORS.append(f"price:{ticker}")
    print(f"[warn] No data for {ticker}")
    return None

def fetch_spx_benchmark() -> Optional[float]:
    # Try SPY first (usually more available), then ^GSPC/^SPX
    for t in ("SPY", "^GSPC", "^SPX"):
        px = fetch_price(t)
        if px is not None:
            return px
    # As a last resort, explicit Stooq symbol
    px = fetch_from_stooq_last_close("^spx")
    if px is not None:
        return px
    DATA_ERRORS.append("benchmark")
    return None

# --- Portfolio storage helpers ---
def parse_holdings(cell: str) -> List[Dict]:
    if isinstance(cell, list):
        return cell
    if pd.isna(cell) or not str(cell).strip():
        return []
    s = str(cell).strip()
    # tolerate single quotes
    if s.startswith("[") and "'" in s and '"' not in s:
        s = s.replace("'", '"')
    try:
        x = json.loads(s)
        if isinstance(x, list):
            return x
        return []
    except Exception:
        return []

def holdings_to_cell(holdings: List[Dict]) -> str:
    return json.dumps(holdings, separators=(",", ":"))

def load_portfolio(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    # init empty frame with expected columns
    return pd.DataFrame(columns=["Date","Cash","Equity","SPX_100","Holdings"])

def save_row(csv_path: str, row: Dict):
    exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    df = pd.DataFrame([row])
    if exists:
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

def load_trade_log(dir_path: str) -> str:
    path = os.path.join(dir_path, "chatgpt_trade_log.csv")
    if not os.path.exists(path):
        pd.DataFrame(columns=["TimestampUTC","Type","Ticker","Shares","Price","Notional"]).to_csv(path, index=False)
    return path

def append_trade(trade_log_path: str, ttype: str, ticker: str, shares: float, price: float):
    notional = round(shares * price, 2)
    row = {
        "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "Type": ttype,
        "Ticker": ticker.upper(),
        "Shares": float(shares),
        "Price": float(price),
        "Notional": notional,
    }
    pd.DataFrame([row]).to_csv(trade_log_path, mode="a", header=False, index=False)

# --- Order execution ---
def load_orders(orders_path: str) -> pd.DataFrame:
    if not orders_path or not os.path.exists(orders_path) or os.path.getsize(orders_path) == 0:
        return pd.DataFrame(columns=["Date","Type","Ticker","Shares","Price"])
    df = pd.read_csv(orders_path)
    # normalize
    for col in ["Date","Type","Ticker","Shares","Price"]:
        if col not in df.columns:
            df[col] = None
    return df

def clear_orders(orders_path: str):
    pd.DataFrame(columns=["Date","Type","Ticker","Shares","Price"]).to_csv(orders_path, index=False)

def execute_orders(today: str, cash: float, holdings: List[Dict], orders_df: pd.DataFrame, trade_log_path: str) -> Tuple[float, List[Dict]]:
    if orders_df is None or orders_df.empty:
        print("No pending orders.")
        return cash, holdings

    # Only orders dated up to 'today'
    try:
        orders_df["Date"] = pd.to_datetime(orders_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        orders_df["Date"] = today

    to_exec = orders_df[orders_df["Date"] <= today].copy()
    if to_exec.empty:
        print("No eligible orders for today.")
        return cash, holdings

    # index by ticker for faster updates
    idx = {h["Ticker"].upper(): i for i, h in enumerate(holdings) if "Ticker" in h}

    for _, r in to_exec.iterrows():
        ttype = str(r.get("Type","")).strip().upper()
        ticker = str(r.get("Ticker","")).strip().upper()
        shares = _safe_float(r.get("Shares"))
        limit = _safe_float(r.get("Price"))

        if not ticker or shares is None or shares <= 0 or ttype not in ("BUY","SELL"):
            print(f"[skip] Bad order line: {dict(r)}")
            continue

        last = fetch_price(ticker)
        if last is None:
            print(f"[skip] No price for {ticker}")
            continue

        # limit logic
        if limit is not None:
            if ttype == "BUY" and last > limit:
                print(f"[skip] BUY {ticker} limit {limit} not reached (last {last})")
                continue
            if ttype == "SELL" and last < limit:
                print(f"[skip] SELL {ticker} limit {limit} not reached (last {last})")
                continue
        exec_price = last

        if ttype == "BUY":
            cost = shares * exec_price
            if cash + 1e-9 < cost:
                print(f"[skip] Not enough cash for BUY {ticker} x{shares} @ {exec_price:.4f} (need {cost:.2f}, cash {cash:.2f})")
                continue
            cash -= cost
            if ticker in idx:
                i = idx[ticker]
                # simple running average
                prev_sh = float(holdings[i].get("Shares",0))
                prev_px = float(holdings[i].get("BuyPrice", exec_price))
                new_sh = prev_sh + shares
                avg_px = (prev_sh*prev_px + shares*exec_price) / new_sh if new_sh>0 else exec_price
                holdings[i]["Shares"] = round(new_sh, 6)
                holdings[i]["BuyPrice"] = round(avg_px, 6)
            else:
                holdings.append({"Ticker": ticker, "Shares": float(shares), "BuyPrice": float(exec_price)})
                idx[ticker] = len(holdings)-1
            append_trade(trade_log_path, "BUY", ticker, shares, exec_price)
            print(f"BUY  {ticker:6s}  x{shares}  @ {exec_price:.4f}")

        elif ttype == "SELL":
            if ticker not in idx:
                print(f"[skip] No position to SELL for {ticker}")
                continue
            i = idx[ticker]
            pos_sh = float(holdings[i].get("Shares",0))
            if pos_sh <= 0:
                print(f"[skip] No shares left in {ticker}")
                continue
            sell_qty = min(pos_sh, shares)
            proceeds = sell_qty * exec_price
            cash += proceeds
            new_sh = pos_sh - sell_qty
            holdings[i]["Shares"] = round(new_sh, 6)
            if new_sh <= 0:
                # remove empty position
                holdings.pop(i)
                # rebuild index
                idx = {h["Ticker"].upper(): j for j,h in enumerate(holdings)}
            append_trade(trade_log_path, "SELL", ticker, sell_qty, exec_price)
            print(f"SELL {ticker:6s}  x{sell_qty}  @ {exec_price:.4f}")

    return cash, holdings

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="portfolio CSV file")
    parser.add_argument("-d", "--dir", required=True, help="output dir (for trade log)")
    parser.add_argument("--orders", default="", help="pending orders CSV path")
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    trade_log_path = load_trade_log(args.dir)

    df = load_portfolio(args.file)
    today = _now_utc_date_str()

    # determine starting state
    cash = 0.0
    equity = 0.0
    spx_100 = ""
    holdings: List[Dict] = []

    if not df.empty:
        last = df.iloc[-1]
        cash = _safe_float(last.get("Cash")) or 0.0
        equity = _safe_float(last.get("Equity")) or cash
        spx_100 = "" if pd.isna(last.get("SPX_100")) else last.get("SPX_100")
        holdings = parse_holdings(last.get("Holdings"))
    else:
        cash = 100.0
        equity = 100.0
        holdings = []

    # 1) Execute pending orders (if any)
    orders_df = load_orders(args.orders) if args.orders else pd.DataFrame()
    cash, holdings = execute_orders(today, cash, holdings, orders_df, trade_log_path)
    if args.orders:
        clear_orders(args.orders)

    # 2) Price positions for equity calc
    pos_value = 0.0
    if holdings:
        print("Current holdings (post-orders):")
        print("Ticker    Shares      BuyPx       LastPx       Value")
        for h in holdings:
            t = h.get("Ticker")
            sh = float(h.get("Shares",0))
            last = fetch_price(t)
            last = last if last is not None else 0.0
            val = sh * last
            pos_value += val
            print(f"{t:6s}  {sh:8.4f}  {float(h.get('BuyPrice',0)):<10.4f}  {last:<10.4f}  {val:<10.2f}")
    else:
        print("No holdings.")

    equity = round(cash + pos_value, 2)

    # 3) Benchmark (optional)
    spx_px = fetch_spx_benchmark()
    if spx_px is None:
        print("S&P 500 data unavailable; skipping benchmark.")
        spx_cell = ""
    else:
        spx_cell = f"{spx_px:.4f}"

    # 4) Write daily row
    row = {
        "Date": today,
        "Cash": f"{cash:.2f}",
        "Equity": f"{equity:.2f}",
        "SPX_100": spx_cell,
        "Holdings": holdings_to_cell(holdings),
    }
    save_row(args.file, row)

    print(f"cash balance: {cash:.2f}")
    print("Daily update complete.")

    if HARD_FAIL_ON_DATA_ERRORS and DATA_ERRORS:
        print(f"::error title=Data errors detected::{', '.join(DATA_ERRORS)}")
        sys.exit(1)

    # Sharpe/Sortino placeholder (requires longer history)
    try:
        hist = load_portfolio(args.file)
        if hist is None or len(hist) < 5:
            print("Could not compute Sharpe/Sortino: Not enough data")
    except Exception:
        print("Could not compute Sharpe/Sortino: Not enough data")


if __name__ == "__main__":
    main()
