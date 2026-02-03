import os
from dotenv import load_dotenv
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
from datetime import datetime
import pandas as pd

load_dotenv()
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(api_key, secret_key)

nasdaq_top_50 = [
    "NVDA", "MU", "GOOGL","SOFI","INTC","TSLA","AAPL","PLTR","META","AMD"
]

request = StockTradesRequest(symbol_or_symbols=nasdaq_top_50,
    start=datetime(2026, 1, 15),
    end=datetime(2026, 1, 16)
)

trades = client.get_stock_trades(request)
df = trades.df.reset_index(level='symbol', drop=False)

df.to_csv('nasdaq_top50_trades.csv')
print(f"Downloaded {len(df):,} trades for {len(nasdaq_top_50)} stocks")
print("Trades per symbol:")
print(df.groupby('symbol').size().sort_values(ascending=False))
