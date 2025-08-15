#!/usr/bin/env python3
"""
Trading script — version corrigée
- Point d’entrée __main__ (CLI)
- Mode non-interactif (CI) sans input()
- Téléchargements yfinance robustes avec fallbacks
- 'final_date' correctement défini à partir du CSV (ou dernier jour ouvré)
- Sauvegarde quotidienne dans chatgpt_portfolio_update.csv et chatgpt_trade_log.csv
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
#    Configuration globale
# ----------------------------

# Répertoire par défaut (comme dans le dépôt Nathan)
DEFAULT_DATA_DIR = Path("Scripts and CSV Files")

# Chemins de données (modifiables via set_data_dir)
DATA_DIR = DEFAULT_DATA_DIR
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"

# Colonnes attendues pour les positions
POSITION_COLUMNS = ["ticker", "shares", "buy_price", "cost_basis", "stop_loss"]


def set_data_dir(path: Path) -> None:
    """Change le répertoire de données (CSV) et s’assure qu’il existe."""
    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV
    DATA_DIR = path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
    TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"


# ----------------------------
#    Outils dates & marché
# ----------------------------

def _last_business_day(ts: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    d = (ts or pd.Timestamp.utcnow()).normalize()
    # reculer si samedi/dimanche
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _compute_final_date_from_csv() -> pd.Timestamp:
    """
    Devine la 'final_date' :
    - si le CSV de portefeuille existe et contient une colonne Date valide -> max(Date)
    - sinon -> dernier jour ouvré (UTC)
    """
    try:
        if PORTFOLIO_CSV.exists() and PORTFOLIO_CSV.stat().st_size > 0:
            df = pd.read_csv(PORTFOLIO_CSV, usecols=["Date"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            m = df["Date"].dropna()
            if not m.empty:
                return m.max().normalize()
    except Exception:
        pass
    return _last_business_day()


# ----------------------------
#    Téléchargements robustes
# ----------------------------

def _robust_history(symbols, start, end) -> pd.DataFrame:
    """
    Téléchargement yfinance avec fallbacks et protections.
    Essaie chaque symbole jusqu’à obtenir une DataFrame non vide.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    for sym in symbols:
        try:
            df = yf.download(
                sym,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            print(f"Failed to get ticker {sym!r} reason: {e}")
    return pd.DataFrame()


def _robust_last_close(ticker: str) -> Optional[float]:
    """Dernier cours de clôture (avec gestion des erreurs)."""
    try:
        df = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            return float(df["Close"].dropna().iloc[-1])
    except Exception as e:
        print(f"Failed to get ticker {ticker!r} reason: {e}")
    print(f"{ticker}: No price data found (period=5d)")
    return None


# ----------------------------
#    Chargement état courant
# ----------------------------

@dataclass
class PortfolioState:
    positions: pd.DataFrame
    cash: float


def _empty_positions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=POSITION_COLUMNS)


def _parse_positions_like_list_str(s: str) -> pd.DataFrame:
    """
    Parse une liste de dicts sérialisée (JSON ou repr Python) en DataFrame positions.
    """
    try:
        obj = json.loads(s)
    except Exception:
        # tentative avec repr Python
        import ast
        obj = ast.literal_eval(s)
    if not isinstance(obj, list):
        return _empty_positions_df()
    rows = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        row = {k: item.get(k) for k in POSITION_COLUMNS}
        rows.append(row)
    df = pd.DataFrame(rows, columns=POSITION_COLUMNS)
    return df


def load_latest_portfolio_state(file: str) -> PortfolioState:
    """
    Charge le dernier état depuis chatgpt_portfolio_update.csv (ou fichier passé en CLI).
    - positions (DataFrame)
    - cash (float)
    Si vide/inexistant: positions vides, cash=0.0
    """
    path = Path(file)
    if not path.exists() or path.stat().st_size == 0:
        return PortfolioState(_empty_positions_df(), 0.0)

    df = pd.read_csv(path)
    if df.empty:
        return PortfolioState(_empty_positions_df(), 0.0)

    last = df.tail(1).copy()

    # cash
    cash = 0.0
    for col in ["Cash", "cash", "cash_balance", "Equity_Cash"]:
        if col in last.columns:
            try:
                cash = float(last.iloc[0][col])
                break
            except Exception:
                pass

    # positions
    positions_df = _empty_positions_df()
    # Cas 1: colonne sérialisée "Holdings" / "holdings" / "Positions"
    for col in ["Holdings", "holdings", "Positions", "positions"]:
        if col in last.columns:
            cell = str(last.iloc[0][col])
            if cell and cell != "nan":
                positions_df = _parse_positions_like_list_str(cell)
                break

    # Cas 2: colonnes tabulaires déjà présentes
    if positions_df.empty and all(c in last.columns for c in POSITION_COLUMNS):
        positions_df = last[POSITION_COLUMNS].copy()

    # Nettoyage types
    if not positions_df.empty:
        positions_df["ticker"] = positions_df["ticker"].astype(str)
        for col in ["shares", "buy_price", "cost_basis", "stop_loss"]:
            if col in positions_df.columns:
                positions_df[col] = pd.to_numeric(positions_df[col], errors="coerce")

    # Affichages (comme dans tes logs)
    if not positions_df.empty:
        try:
            # Date d’en-tête de la ligne
            if "Date" in last.columns:
                print(pd.to_datetime(last.iloc[0]["Date"]))
            print(positions_df)
            print(positions_df.to_dict(orient="records"))
            print(positions_df.to_dict(orient="records"))
            print(positions_df.reset_index(drop=True))
        except Exception:
            pass

    return PortfolioState(positions_df, float(cash))


# ----------------------------
#    Exécution/MAJ portefeuille
# ----------------------------

def process_portfolio(positions: pd.DataFrame, cash: float, interactive: bool) -> Tuple[pd.DataFrame, float]:
    """
    Met à jour les prix courants des positions et (si interactive) permet d’enregistrer
    des achats/ventes manuels. En mode non-interactif, ne pose aucune question.
    """
    pos = positions.copy() if not positions.empty else _empty_positions_df()

    # Mise à jour des prix de marché (infos pour l’affichage/logique utilisateur)
    if not pos.empty and "ticker" in pos.columns:
        latest_prices = []
        for t in pos["ticker"]:
            px = _robust_last_close(str(t))
            latest_prices.append(px if px is not None else np.nan)
        pos["last_close"] = latest_prices

    # (Optionnel) Enregistrer des trades si mode interactif
    if interactive:
        try:
            ans = input("Souhaitez-vous enregistrer des achats/ventes manuels ? [y/N]: ").strip().lower()
        except Exception:
            ans = "n"
        if ans == "y":
            _interactive_trade_loop(pos, cash)

    return pos, cash


def _interactive_trade_loop(pos: pd.DataFrame, cash: float) -> None:
    """
    Mini boucle d’édition de trades. N’est appelée que si interactive=True.
    Écrit TRADE_LOG_CSV.
    """
    print("Enregistrement manuel de trades. Laissez 'ticker' vide pour terminer.")
    trades = []
    while True:
        t = input("Ticker (vide pour finir): ").strip().upper()
        if not t:
            break
        side = input("Side [BUY/SELL]: ").strip().upper()
        try:
            qty = float(input("Quantité: ").strip())
        except Exception:
            print("Quantité invalide, abort.")
            continue
        price = _robust_last_close(t) or 0.0
        cost = qty * price * (1 if side == "BUY" else -1)
        trades.append(
            {"Date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
             "Action": side, "Ticker": t, "Shares": qty, "Price": price, "Cost": cost, "Notes": ""}
        )
        # Mise à jour cash (approx simple)
        cash -= cost

    if trades:
        TRADE_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        tl = pd.DataFrame(trades)
        if TRADE_LOG_CSV.exists() and TRADE_LOG_CSV.stat().st_size > 0:
            tl.to_csv(TRADE_LOG_CSV, mode="a", header=False, index=False)
        else:
            tl.to_csv(TRADE_LOG_CSV, index=False)
        print("Trades enregistrés dans", TRADE_LOG_CSV)


# ----------------------------
#    Résultats quotidiens
# ----------------------------

def daily_results(positions: pd.DataFrame, cash: float) -> None:
    """
    Calcule/affiche les résultats du jour et les enregistre dans chatgpt_portfolio_update.csv.
    - Définit final_date correctement
    - Benchmark S&P 500 avec fallbacks
    - Sharpe/Sortino si historique présent
    """
    # 1) Définir la date du jour de travail (final_date)
    final_date = _compute_final_date_from_csv()

    # 2) Valeur du portefeuille (approx)
    market_value = 0.0
    if not positions.empty:
        # Utilise last_close si dispo, sinon buy_price
        px = positions.get("last_close")
        if px is None or px.isna().all():
            px = positions.get("buy_price")
        shares = pd.to_numeric(positions.get("shares"), errors="coerce").fillna(0.0)
        prices = pd.to_numeric(px, errors="coerce").fillna(0.0)
        market_value = float((shares * prices).sum())

    equity = market_value + float(cash)

    # 3) Benchmark S&P 500 robuste
    start_dt = final_date - pd.Timedelta(days=60)
    end_dt = final_date + pd.Timedelta(days=1)
    spx = _robust_history(["^GSPC", "SPY", "^SPX"], start_dt, end_dt)

    spx_value_100 = np.nan
    if not spx.empty and "Close" in spx.columns:
        close = spx["Close"].dropna()
        if not close.empty:
            base = float(close.iloc[0])
            spx_value_100 = 100.0 * (float(close.iloc[-1]) / base)
    else:
        print("S&P 500 data unavailable; skipping benchmark.")

    # 4) Affichages principaux
    print(f"prices and updates for {final_date.date()}")
    print(f"Latest ChatGPT Equity: ${equity:.2f}")
    if isinstance(spx_value_100, float) and not np.isnan(spx_value_100):
        print(f"$100 Invested in the S&P 500: ${spx_value_100:.2f}")
    else:
        print("$100 Invested in the S&P 500: $nan")

    if not positions.empty:
        print("today's portfolio:")
        print(positions[["ticker", "shares", "buy_price", "cost_basis", "stop_loss"]]
              .reset_index(drop=True))
    print(f"cash balance: {equity - market_value:.2f}")

    # 5) Sauvegarde de la ligne du jour
    row = {
        "Date": final_date.strftime("%Y-%m-%d"),
        "Cash": round(float(cash), 2),
        "Equity": round(float(equity), 2),
        "SPX_100": round(float(spx_value_100), 4) if isinstance(spx_value_100, float) else "",
        "Holdings": json.dumps(positions[POSITION_COLUMNS].fillna("").to_dict(orient="records"))
                    if not positions.empty else "[]",
    }
    out = pd.DataFrame([row])
    PORTFOLIO_CSV.parent.mkdir(parents=True, exist_ok=True)
    if PORTFOLIO_CSV.exists() and PORTFOLIO_CSV.stat().st_size > 0:
        out.to_csv(PORTFOLIO_CSV, mode="a", header=False, index=False)
    else:
        out.to_csv(PORTFOLIO_CSV, index=False)

    # 6) Statistiques (Sharpe/Sortino) si historique disponible
    try:
        hist = pd.read_csv(PORTFOLIO_CSV, usecols=["Date", "Equity"])
        hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
        hist = hist.dropna(subset=["Date", "Equity"]).sort_values("Date")
        if len(hist) >= 2:
            ret = hist["Equity"].pct_change().dropna()
            if not ret.empty:
                sharpe_total = ret.mean() / (ret.std(ddof=1) + 1e-12)
                sortino_total = ret.mean() / (ret[ret < 0].std(ddof=1) + 1e-12)
                sharpe_ann = sharpe_total * np.sqrt(252)
                sortino_ann = sortino_total * np.sqrt(252)
                print(f"Total Sharpe Ratio over {len(ret)} days: {sharpe_total:.4f}")
                print(f"Total Sortino Ratio over {len(ret)} days: {sortino_total:.4f}")
                print(f"Annualized Sharpe Ratio: {sharpe_ann:.4f}")
                print(f"Annualized Sortino Ratio: {sortino_ann:.4f}")
    except Exception as e:
        print(f"Could not compute Sharpe/Sortino: {e}")


# ----------------------------
#    Programme principal
# ----------------------------

def main(file: str, data_dir: Path | None = None, interactive: bool | None = None) -> None:
    """Run the trading script."""
    if data_dir is not None:
        set_data_dir(data_dir)

    state = load_latest_portfolio_state(file)

    # Décider l'interactivité : flag explicite prioritaire; sinon CI/non-TTY -> False
    if interactive is None:
        interactive = not (
            os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("CI") == "true"
            or not sys.stdin.isatty()
        )

    positions, cash = process_portfolio(state.positions, state.cash, interactive=interactive)
    daily_results(positions, cash)


# ----------------------------
#    CLI (__main__)
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Maintain/compute daily results for the ChatGPT micro-cap portfolio."
    )
    parser.add_argument(
        "-f", "--file", required=True,
        help="Path to the portfolio CSV containing historical records."
    )
    parser.add_argument(
        "-d", "--data-dir", default=None,
        help="Directory to write chatgpt_portfolio_update.csv and chatgpt_trade_log.csv"
    )
    ig = parser.add_mutually_exclusive_group()
    ig.add_argument(
        "--interactive", dest="interactive", action="store_true",
        help="Force prompts for manual trades."
    )
    ig.add_argument(
        "--non-interactive", dest="interactive", action="store_false",
        help="Disable prompts (useful for CI)."
    )
    parser.set_defaults(interactive=None)
    args = parser.parse_args()

    main(args.file, Path(args.data_dir) if args.data_dir else None, interactive=args.interactive)
