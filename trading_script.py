#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json, datetime as dt
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# ---------- util filesystem ----------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def load_csv(path: str, cols: List[str]) -> pd.DataFrame:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            df = pd.read_csv(path)
            df = df.dropna(how="all")
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            return df
        except Exception as e:
            print(f"WARN: could not read {path}: {e}", file=sys.stderr)
    return pd.DataFrame(columns=cols)

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

# ---------- holdings ----------
def parse_holdings(s) -> List[Dict]:
    if isinstance(s, str) and s.strip():
        try:
            return json.loads(s)
        except Exception:
            pass
    return []

def holdings_to_str(h) -> str:
    return json.dumps(h, separators=(",", ":"))

def last_numeric(df: pd.DataFrame, col: str, default: float) -> float:
    if df.empty or col not in df.columns:
        return default
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else default

def upsert_holding(holds: List[Dict], ticker: str, delta_qty: int, price: float) -> Tuple[List[Dict], float]:
    realized = 0.0
    h = next((x for x in holds if x["ticker"] == ticker), None)
    if h is None:
        if delta_qty > 0:
            holds.append({"ticker": ticker, "qty": int(delta_qty), "avg_cost": float(price)})
        return holds, realized
    if delta_qty > 0:  # BUY
        total_cost = h["avg_cost"] * h["qty"] + price * delta_qty
        new_qty = h["qty"] + delta_qty
        h["qty"] = new_qty
        h["avg_cost"] = total_cost / new_qty
    else:  # SELL
        sell_qty = min(h["qty"], -delta_qty)
        realized = (price - h["avg_cost"]) * sell_qty
        h["qty"] -= sell_qty
        if h["qty"] == 0:
            holds[:] = [x for x in holds if x["ticker"] != ticker]
    return holds, realized

# ---------- pricing ----------
def _price_from_yahoo(ticker: str) -> Optional[float]:
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        p = None
        try:
            p = float(tk.fast_info.get("last_price"))
        except Exception:
            p = None
        if p is None:
            hist = tk.history(period="5d", interval="1d", auto_adjust=False, prepost=False, rounding=False)
            if not hist.empty:
                p = float(hist["Close"].dropna().iloc[-1])
        return p
    except Exception as e:
        print(f"Yahoo fail for {ticker}: {e}")
        return None

def _map_to_stooq(ticker: str) -> Optional[str]:
    if ticker.startswith("^"):
        return None
    if "." in ticker:
        return ticker
    if ticker.replace("-", "").isalpha():
        return f"{ticker}.US"
    return ticker

def _price_from_stooq(ticker: str) -> Optional[float]:
    try:
        import pandas_datareader.data as web
        t2 = _map_to_stooq(ticker.upper())
        if not t2:
            return None
        df = web.DataReader(t2, "stooq")
        if df is not None and not df.empty:
            return float(df["Close"].dropna().iloc[0])
    except Exception as e:
        print(f"Stooq fail for {ticker}: {e}")
    return None

def get_price(ticker: str, manual: Optional[float] = None) -> Optional[float]:
    if manual is not None and manual > 0:
        return float(manual)
    p = _price_from_yahoo(ticker)
    if p is None:
        p = _price_from_stooq(ticker)
    return p

def equity_now(holds: List[Dict], cash: float) -> float:
    total = cash
    for h in holds:
        p = get_price(h["ticker"])
        if p is None:
            p = h["avg_cost"]
        total += h["qty"] * p
    return float(total)

# ---------- Alpaca (paper/live) ----------
def alpaca_enabled() -> bool:
    return bool(os.environ.get("ALPACA_KEY") and os.environ.get("ALPACA_SECRET"))

def alpaca_base_url() -> str:
    # par défaut: paper trading
    return os.environ.get("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets"

def alpaca_place_order(ticker: str, qty: int, side: str) -> Optional[dict]:
    import requests
    try:
        url = alpaca_base_url().rstrip("/") + "/v2/orders"
        headers = {
            "APCA-API-KEY-ID": os.environ["ALPACA_KEY"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"],
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "symbol": ticker,
            "qty": qty,
            "side": side.lower(),  # "buy" ou "sell"
            "type": "market",
            "time_in_force": "day"
        }
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        if r.status_code >= 400:
            print(f"Alpaca error {r.status_code}: {r.text}")
            return None
        return r.json()
    except Exception as e:
        print(f"Alpaca request failed for {ticker}: {e}")
        return None

# ---------- orders ----------
def process_orders(orders_path: str, cash: float, holds: List[Dict], trade_log_path: str, keep_orders: bool = False):
    df = load_csv(orders_path, ["Date", "Type", "Ticker", "Shares", "Price"])
    if df.empty:
        print("No pending orders.")
        return cash, holds, 0

    processed, success = [], 0
    use_alpaca = alpaca_enabled()

    for _, row in df.iterrows():
        typ = str(row.get("Type", "")).strip().upper()
        ticker = str(row.get("Ticker", "")).strip().upper()
        shares = int(pd.to_numeric(row.get("Shares"), errors="coerce") or 0)
        m = pd.to_numeric(row.get("Price"), errors="coerce")
        manual_price = float(m) if not np.isnan(m) else None

        if not ticker or shares <= 0 or typ not in ("BUY", "SELL"):
            continue

        price = get_price(ticker, manual=manual_price)
        if price is None:
            print(f"Price unavailable for {ticker}; skipped.")
            continue

        broker = ""
        broker_order_id = ""

        if typ == "BUY":
            max_afford = int(cash // price)
            qty = min(shares, max_afford) if not use_alpaca else shares  # si Alpaca, on envoie la quantité demandée
            if qty <= 0:
                print(f"Insufficient cash to buy {ticker}; need {shares}×{price:.2f}, cash {cash:.2f}")
                continue
            # Simu ledger local
            cash -= qty * price
            holds, _ = upsert_holding(holds, ticker, qty, price)
            # Envoi broker si activé
            if use_alpaca:
                r = alpaca_place_order(ticker, qty, "buy")
                if r:
                    broker = "alpaca"
                    broker_order_id = r.get("id", "")
        else:
            have = next((x["qty"] for x in holds if x["ticker"] == ticker), 0)
            qty = min(have, shares) if not use_alpaca else shares
            if qty <= 0:
                print(f"No position to sell for {ticker}; have {have}, requested {shares}.")
                continue
            holds, _ = upsert_holding(holds, ticker, -qty, price)
            cash += qty * price
            if use_alpaca:
                r = alpaca_place_order(ticker, qty, "sell")
                if r:
                    broker = "alpaca"
                    broker_order_id = r.get("id", "")

        processed.append({
            "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": typ, "ticker": ticker, "shares": qty,
            "price": round(price, 4), "status": "FILLED",
            "broker": broker, "broker_order_id": broker_order_id
        })
        success += 1

    if processed:
        logdf = load_csv(trade_log_path, ["ts", "type", "ticker", "shares", "price", "status", "broker", "broker_order_id"])
        logdf = pd.concat([logdf, pd.DataFrame(processed)], ignore_index=True)
        save_csv(logdf, trade_log_path)

    if not keep_orders:
        remaining = []
        for _, row in df.iterrows():
            typ = str(row.get("Type", "")).strip().upper()
            ticker = str(row.get("Ticker", "")).strip().upper()
            shares = int(pd.to_numeric(row.get("Shares"), errors="coerce") or 0)
            filled = any(p["ticker"] == ticker and p["type"] == typ and p["shares"] == shares for p in processed)
            if not filled:
                remaining.append(row)
        remdf = pd.DataFrame(remaining, columns=["Date", "Type", "Ticker", "Shares", "Price"])
        save_csv(remdf, orders_path)

    return cash, holds, success

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="portfolio CSV path")
    ap.add_argument("-d", "--dir", required=True, help="output dir")
    ap.add_argument("--orders", default=None, help="pending_orders CSV")
    ap.add_argument("--non-interactive", action="store_true")
    ap.add_argument("--fail-on-data-errors", action="store_true", default=False)
    ap.add_argument("--keep-orders", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.dir)

    portfolio_path = args.file
    trade_log_path = os.path.join(args.dir, "chatgpt_trade_log.csv")
    orders_path = args.orders or os.path.join(args.dir, "pending_orders.csv")

    if not os.path.exists(portfolio_path):
        save_csv(pd.DataFrame(columns=["Date", "Cash", "Equity", "SPX_100", "Holdings"]), portfolio_path)
    if not os.path.exists(trade_log_path):
        save_csv(pd.DataFrame(columns=["ts", "type", "ticker", "shares", "price", "status", "broker", "broker_order_id"]), trade_log_path)
    if not os.path.exists(orders_path):
        save_csv(pd.DataFrame(columns=["Date", "Type", "Ticker", "Shares", "Price"]), orders_path)

    df = load_csv(portfolio_path, ["Date", "Cash", "Equity", "SPX_100", "Holdings"])
    cash = last_numeric(df, "Cash", 100.0)
    equity = last_numeric(df, "Equity", cash)
    holds = []
    if "Holdings" in df.columns and not df.empty:
        s = df["Holdings"].dropna().astype(str)
        s = s[s.str.strip() != ""]
        holds = parse_holdings(s.iloc[-1]) if not s.empty else []

    cash, holds, n = process_orders(orders_path, cash, holds, trade_log_path, args.keep_orders)
    equity = equity_now(holds, cash)

    today = dt.datetime.utcnow().date().isoformat()
    row = pd.DataFrame([{
        "Date": today,
        "Cash": round(cash, 2),
        "Equity": round(equity, 2),
        "SPX_100": "",
        "Holdings": holdings_to_str(holds)
    }])
    df = pd.concat([df, row], ignore_index=True)
    save_csv(df, portfolio_path)

    print(f"Processed orders: {n}")
    print(f"Holdings: {holdings_to_str(holds)}")
    print(f"cash balance: {cash:.2f}")
    if alpaca_enabled():
        print("Alpaca: ENABLED (orders also submitted to broker).")
    else:
        print("Alpaca: DISABLED (CSV simulation only).")
    print("Daily update complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
