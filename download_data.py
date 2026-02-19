import os
import time
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest

load_dotenv()

SYMBOLS = ["ANET"]
START_DATE = datetime(2026, 1, 5)
END_DATE = datetime(2026, 2, 5)
RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/nasdaq_trades.csv"
DELAY_SECONDS = 3  #keep for rate limit pause


def get_trading_days(start, end):
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def download_day(client, symbols, date):
    os.makedirs(RAW_DIR, exist_ok=True)
    date_str = date.strftime('%Y-%m-%d')
    symbol_str = '_'.join(symbols)
    filepath = f"{RAW_DIR}/{symbol_str}_{date_str}.csv"

    if os.path.exists(filepath):
        print(f"{date_str} already downloaded")
        return filepath

    try:
        start = datetime(date.year, date.month, date.day, 13, 30)
        end = datetime(date.year, date.month, date.day, 20, 0)

        request = StockTradesRequest(
            symbol_or_symbols=symbols,
            start=start,
            end=end
        )
        trades = client.get_stock_trades(request)
        df = trades.df.reset_index(level='symbol', drop=False)

        if len(df) == 0:
            print(f"{date_str} no trades holiday")
            return None

        df.to_csv(filepath)
        print(f"{date_str} — {len(df):,} trades saved")
        return filepath

    except Exception as e:
        print(f" Fail {date_str} — {e}")
        return None


def merge_raw_files():
    os.makedirs("src/data/processed", exist_ok=True)
    raw_files = sorted([
        f"{RAW_DIR}/{f}" for f in os.listdir(RAW_DIR)
        if f.endswith('.csv')
    ])

    if not raw_files:
        raise FileNotFoundError(f"No files in {RAW_DIR}")

    print(f"\nMerging {len(raw_files)} files")
    dfs = []
    for f in raw_files:
        df = pd.read_csv(f, index_col=0)
        df.index = pd.to_datetime(df.index, format='mixed', utc=True)
        dfs.append(df)

    df_merged = pd.concat(dfs)
    df_merged.index = pd.to_datetime(df_merged.index, utc=True)
    df_merged = df_merged.sort_index()

    df_merged.to_csv(OUTPUT_FILE)
    print(f"  Merged: {OUTPUT_FILE}")
    print(f"  Total trades:  {len(df_merged):,}")
    print(f"  Date range:    {df_merged.index.min()} → {df_merged.index.max()}")
    print(f"  Symbols:       {df_merged['symbol'].unique().tolist()}")

    return df_merged


def main():
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY")
    )

    trading_days = get_trading_days(START_DATE, END_DATE)
    print(f"Downloading {len(trading_days)} trading days for {SYMBOLS}")
    print(f"Range: {START_DATE.date()} - {END_DATE.date()}")


    for i, day in enumerate(trading_days):
        print(f"[{i+1}/{len(trading_days)}]", end=" ")
        download_day(client, SYMBOLS, day)
        if i < len(trading_days) - 1:
            time.sleep(DELAY_SECONDS)

    df = merge_raw_files()
    print("\nDownload complete")
    return df


if __name__ == "__main__":
    main()
