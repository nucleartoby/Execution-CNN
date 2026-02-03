import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_sliding_windows(df, window_size=100, prediction_horizon=50):
    df = df.copy()
    df['price_change'] = df['price'].pct_change()
    df['volume_ma'] = df['size'].rolling(window=20).mean()
    df['price_volatility'] = df['price'].rolling(window=20).std()
    df['trade_intensity'] = df['size'] / df['size'].rolling(window=50).mean()
    
    df = df.dropna()
    
    df['future_price'] = df['price'].shift(-prediction_horizon)
    df['target'] = (df['future_price'] > df['price']).astype(int)
    df = df[:-prediction_horizon]
    
    features = ['price', 'size', 'price_change', 'volume_ma','price_volatility', 'trade_intensity']
    
    X, y = [], []
    
    for i in range(len(df) - window_size):
        window = df[features].iloc[i:i+window_size].values
        X.append(window)
        target = df['target'].iloc[i+window_size]
        y.append(target)
    
    return np.array(X), np.array(y)

def prepare_train_test_split(X, y, train_ratio=0.8):
    split_idx = int(len(X) * train_ratio)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    
    X_test_scaled = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
