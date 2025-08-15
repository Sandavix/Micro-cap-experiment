from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
import sys
import argparse

# --- rest of your existing code above stays unchanged ---

# NOTE: the following constants are as in your original file; adjust if your folder names differ
DEFAULT_DATA_DIR = Path("Scripts and CSV Files")
PORTFOLIO_CSV = DEFAULT_DATA_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DEFAULT_DATA_DIR / "chatgpt_trade_log.csv"

# -----------------------------------------------------------------------------
# Utilitaires pour structure de fichiers
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def set_data_dir(data_dir: Path) -> None:
    global DEFAULT_DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV
    DEFAULT_DATA_DIR = data_dir
    PORTFOLIO_CSV = DEFAULT_DATA_DIR / "chatgpt_portfolio_update.csv"
    TRADE_LOG_CSV = DEFAULT_DATA_DIR / "chatgpt_trade_log.csv"
    ensure_dir(DEFAULT_DATA_DIR)

# -----------------------------------------------------------------------------
# Fonctions principales (extraits de ton script d’origine)
# -----------------------------------------------------------------------------
def load_latest_portfolio_state(file: str):
    """
    Charge l’état le plus récent (positions + cash) depuis le CSV d’historique.
    Si le fichier n’existe pas, démarre avec portefeuille vide et cash demandé.
    """
    fp = Path(file)
    if not fp.exists() or fp.stat().st_size == 0:
        # démarrage à blanc
        print("Le CSV de portefeuille est vide/inexistant — démarrage avec portefeuille vide.")
        return pd.DataFrame(columns=["ticker", "shares", "buy_price", "cost_basis", "stop_loss"]), 100.0

    df = pd.read_csv(fp)
    # On garde les colonnes essentielles si présentes
    cols = [c for c in ["ticker", "shares", "buy_price", "cost_basis", "stop_loss"] if c in df.columns]
    positions = df[cols].dropna(how="all")
    positions = positions[positions.get("ticker", pd.Series(dtype=str)).notna()]
    # Cash : on tente de lire la dernière ligne TOTAL si présente
    cash = 0.0
    if "Cash Balance" in df.columns and "Ticker" in df.columns and "Date" in df.columns:
        try:
            totals = df[df["Ticker"] == "TOTAL"].copy()
            totals["Date"] = pd.to_datetime(totals["Date"], errors="coerce")
            latest = totals.sort_values("Date").iloc[-1]
            cash = float(latest["Cash Balance"])
        except Exception:
            pass
    return positions.reset_index(drop=True), float(cash)

def process_portfolio(chatgpt_portfolio: pd.DataFrame,
                      cash: float,
                      interactive: bool = True):
    """
    Ta logique existante ici : mise à jour des prix, calculs, et éventuellement
    prompts manuels si `interactive=True`. Je conserve la signature d’origine.
    """
    # ... (toute ta logique telle qu’elle existe déjà dans ton fichier)
    # Pour garder la réponse concise ici, je ne re-déplie pas le bloc entier :
    # remets simplement ton code existant de process_portfolio.
    return chatgpt_portfolio, cash

def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """
    Calcule et affiche les performances quotidiennes + indices de référence.
    Section S&P 500 rendue robuste aux erreurs de téléchargement.
    """
    # --- ta logique existante au-dessus (PNL, ratios, etc.) ---

    # Suppose que tu construis un DataFrame `df` avec au moins les colonnes ["Date", "Ticker", "Cash Balance"]
    # et que tu as besoin de final_date = dernière date de df.
    # Si ton code d’origine diffère, garde ton calcul actuel de final_date.
    # Exemple générique :
    # df = ...  # ton DataFrame agrégé
    # df["Date"] = pd.to_datetime(df["Date"])
    # final_date = df["Date"].max()

    # -------------- Début du bloc robuste S&P 500 --------------
    # Get S&P 500 data (robust with fallbacks)
    start_dt = pd.Timestamp("2025-06-27")
    # ATTENTION: final_date doit exister à ce stade dans ta fonction
    end_dt = final_date + pd.Timedelta(days=1)

    spx_series = None
    for __sym in ["^GSPC", "SPY", "^SPX"]:
        try:
            __df = yf.download(
                __sym,
                start=start_dt,
                end=end_dt,
                progress=False,
                auto_adjust=True,
                threads=False
            )
            if not __df.empty and "Close" in __df.columns:
                __ser = __df["Close"].dropna()
                if not __ser.empty:
                    spx_series = __ser
                    break
        except Exception as __e:
            print(f"Failed to get ticker {__sym!r} reason: {__e}")

    if spx_series is None or spx_series.empty:
        print("S&P 500 data unavailable; skipping benchmark lines.")
        spx_value = float("nan")
    else:
        initial_price = float(spx_series.iloc[0])
        spx_value = 100.0 * (float(spx_series.iloc[-1]) / initial_price)

    # --- ta logique d’affichage / sauvegarde existante continue ici ---
    # -------------- Fin du bloc robuste S&P 500 ----------------

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

# -----------------------------------------------------------------------------
# Point d’entrée CLI
# -----------------------------------------------------------------------------
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
