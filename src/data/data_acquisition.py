import os
from dotenv import load_dotenv
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
from datetime import datetime
import pandas as pd

def download_nasdaq_data(symbols, start_date, end_date, output_file='nasdaq_trades.csv'):
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    client = StockHistoricalDataClient(api_key, secret_key)
    
    request = StockTradesRequest(
        symbol_or_symbols=symbols,
        start=start_date,
        end=end_date
    )
    
    trades = client.get_stock_trades(request)
    df = trades.df.reset_index(level='symbol', drop=False)
    df.to_csv(output_file)
    
    return df
