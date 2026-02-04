import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

from src.data.data_acquisition import download_nasdaq_data
from src.feature_engineering.engineering import create_sliding_windows, prepare_train_test_split

def main():
    config = {
        'symbols': ["ANET"],
        'start_date': datetime(2026, 2, 2),
        'end_date': datetime(2026, 2, 3),
        'output_file': 'nasdaq_top10_trades.csv',
        'window_size': 100,
        'prediction_horizon': 50,
        'train_test_split': 0.8
    }
    
    print("Downloading trade execution data:")
    print(f"Symbols: {', '.join(config['symbols'])}")
    print(f"Date range: {config['start_date'].date()} to {config['end_date'].date()}")
    

    df = download_nasdaq_data(
        symbols=config['symbols'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        output_file=config['output_file']
    )
    
    print("Engineering features and creating sliding windows:")
    print(f"Window size: {config['window_size']} trades")
    print(f"Prediction horizon: {config['prediction_horizon']} trades ahead")
    
    
        # Process each symbol separately
    all_X, all_y, all_symbols = [], [], []
        
    for symbol in df['symbol'].unique():
        print(f"\nProcessing {symbol}:")
        df_symbol = df[df['symbol'] == symbol].copy()
            
        X, y = create_sliding_windows(
            df_symbol,
            window_size=config['window_size'],
            prediction_horizon=config['prediction_horizon']
            )
            
        all_X.append(X)
        all_y.append(y)
        all_symbols.extend([symbol] * len(X))
            
        print(f"{len(X):,} windows")
        print(f"Upward movement: {y.mean()*100:.1f}%")
        
        # Combine all symbols
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        print(f"Total windows: {len(X_combined):,}")
        print(f"Input shape: {X_combined.shape}")
        print(f"Target shape: {y_combined.shape}")
    
    print(f"Train ratio: {config['train_test_split']*100:.0f}%")
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X_combined, 
        y_combined,
        train_ratio=config['train_test_split']
    )
    
    np.savez_compressed(
        'processed_data.npz',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        symbols=all_symbols
    )
    
    print("Complete")
    print(f"Total trades downloaded: {len(df):,}")
    print(f"Training windows: {len(X_train):,}")
    print(f"Test windows: {len(X_test):,}")
    print(f"Features per window: {X_train.shape[2]}")
    print(f"Window size: {X_train.shape[1]} trades")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()