import numpy as np

def create_sliding_windows(df, window_size=100, prediction_horizon=50):
    df['price_change'] = df['price'].pct_change()
    df['volume_ma'] = df['size'].rolling(window=20).mean()
    df['price_volatility'] = df['price'].rolling(window=20).std()
    df['trade_intensity'] = df['size'] / df['size'].rolling(window=50).mean()
    
    df = df.dropna()
    
    df['future_price'] = df['price'].shift(-prediction_horizon)
    df['target'] = (df['future_price'] > df['price']).astype(int)
    
    df = df[:-prediction_horizon]
    
    features = ['price', 'size', 'price_change', 'volume_ma', 
                'price_volatility', 'trade_intensity']
    
    X, y = [], []
    
    for i in range(len(df) - window_size):
        window = df[features].iloc[i:i+window_size].values
        X.append(window)
        target = df['target'].iloc[i+window_size]
        y.append(target)
    
    return np.array(X), np.array(y)

X, y = create_sliding_windows(df, window_size=100, prediction_horizon=50)
print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Movement: {y.mean()*100:.1f}%")
