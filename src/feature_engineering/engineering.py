import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


def build_day_windows(df, window_size=150, prediction_horizon=20, min_move_pct=0.002, stride=10):
    ts = pd.to_datetime(df.index, format='mixed', utc=True).tz_convert('America/New_York')
    df = df.assign(day=ts.date, tod=ts.hour * 60 + ts.minute)
    df = df[(df['tod'] >= 9 * 60 + 30) & (df['tod'] < 16 * 60)]

    all_X, all_y = [], []
    for _, day in df.groupby('day', sort=True):
        group = np.arange(len(day)) // stride
        bars = day.groupby(group).agg(
            price=('price', 'last'), size=('size', 'sum'), n=('size', 'count'))
        bars = bars[bars['n'] == stride][['price', 'size']]
        if len(bars) <= window_size + 2 * prediction_horizon + 50:
            continue
        X, y = create_sliding_windows(
            bars, window_size, prediction_horizon, min_move_pct, verbose=False)
        all_X.append(X)
        all_y.append(y)

    X, y = np.concatenate(all_X), np.concatenate(all_y)
    print(f"Sessions: {len(all_X)}  windows: {len(X):,}  shape: {X.shape}")
    print(f"Target distribution - Down: {(y == 0).sum():,}, Up: {(y == 1).sum():,}  (Up ratio: {y.mean()*100:.1f}%)")
    return X, y


def create_sliding_windows(df, window_size=100, prediction_horizon=500, min_move_pct=0.003, verbose=True):
    df = df.copy()

    df['price_change'] = df['price'].pct_change()
    df['volume_ma'] = df['size'].rolling(window=20).mean()
    df['price_volatility'] = df['price'].rolling(window=20).std()
    df['trade_intensity'] = df['size'] / df['size'].rolling(window=20).mean()
    df['price_ma_5'] = df['price'].rolling(window=5).mean()
    df['price_ma_20'] = df['price'].rolling(window=20).mean()
    df['ma_crossover'] = (df['price_ma_5'] > df['price_ma_20']).astype(int)
    df['momentum_10'] = df['price'].pct_change(10)
    df['momentum_50'] = df['price'].pct_change(50)
    df['volume_change'] = df['size'].pct_change()
    df['volume_spike'] = (df['size'] > df['volume_ma'] * 2.0).astype(int)
    df = df.dropna()

    df['future_price'] = df['price'].shift(-prediction_horizon)
    df['future_return'] = (df['future_price'] - df['price']) / df['price']
    df['target'] = (df['future_return'] > min_move_pct).astype(int)
    df = df.dropna()
    df = df[:-prediction_horizon]

    if verbose:
        print(f"Target distribution - Down: {sum(df['target']==0):,}, Up: {sum(df['target']==1):,}")
        print(f"Up ratio: {df['target'].mean()*100:.1f}%")

    features = [
        'price', 'size', 'price_change', 'volume_ma', 'price_volatility',
        'trade_intensity', 'ma_crossover', 'momentum_10', 'momentum_50',
        'volume_change', 'volume_spike']

    data = df[features].values.astype(np.float32)   # (N, F)
    targets = df['target'].values                     # (N,)

    n_samples = len(data) - window_size
    n_features = data.shape[1]

    shape   = (n_samples, window_size, n_features)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).copy()
    y = targets[window_size:window_size + n_samples]

    if verbose:
        print(f"Windows created: {len(X):,}  shape: {X.shape}")
    return X, y


def prepare_train_test_split(X, y, train_ratio=0.8, scaler_path='scaler.pkl', gap=0):
    split_idx = int(len(X) * train_ratio)

    X_train, X_test = X[:split_idx - gap], X[split_idx:]
    y_train, y_test = y[:split_idx - gap], y[split_idx:]

    X_train = np.clip(X_train, -1e9, 1e9)
    X_test = np.clip(X_test, -1e9, 1e9)

    assert not np.isnan(X_train).any(), "NaN found in X_train after clipping"
    assert not np.isinf(X_train).any(), "Inf found in X_train after clipping"
    assert not np.isnan(X_test).any(), "NaN found in X_test after clipping"
    assert not np.isinf(X_test).any(), "Inf found in X_test after clipping"

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

    X_test_scaled = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test
