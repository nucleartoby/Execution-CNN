import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

from src.data.data_acquisition import download_nasdaq_data
from src.feature_engineering.engineering import create_sliding_windows, prepare_train_test_split
from src.model.model import build_cnn_model
from src.utils.visualise import (
    plot_training_history, 
    evaluate_predictions,
    plot_prediction_confidence,
    plot_predictions_over_time,
    plot_performance_curves,
    analyse_feature_importance,
    plot_prediction_heatmap
)


def main():
    print("CNN Trade Execution Prediction")

    config = {
        'symbols': ["ANET"],
        'start_date': datetime(2026, 2, 2),
        'end_date': datetime(2026, 2, 3),
        'output_file': 'nasdaq_top10_trades.csv',
        'window_size': 100,
        'prediction_horizon': 50,
        'train_test_split': 0.8,
        'epochs': 50,
        'batch_size': 64
    }
    
    feature_names = ['price', 'size', 'price_change', 'volume_ma', 
                     'price_volatility', 'trade_intensity']

    print(f"Symbols: {', '.join(config['symbols'])}")
    print(f"Date range: {config['start_date'].date()} to {config['end_date'].date()}")
    
    df = download_nasdaq_data(
        symbols=config['symbols'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        output_file=config['output_file']
    )
    print(f"{len(df):,} trades")
    all_X, all_y, all_symbols = [], [], []
    
    for symbol in df['symbol'].unique():
        print(f"\nProcessing {symbol}...")
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
    
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    
    print(f"\nTotal windows: {len(X_combined):,}")
    print(f"Input shape: {X_combined.shape}")
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X_combined,
        y_combined,
        train_ratio=config['train_test_split']
    )
    print(f"Training: {len(X_train):,} | Test: {len(X_test):,}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)
    
    print(f"Model input shape: {input_shape}")
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001
    )
    
    print(f"Epochs: {config['epochs']}, Batch size: {config['batch_size']}")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    test_loss, test_acc, test_precision, test_recall = model.evaluate(
        X_test, y_test, verbose=0
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
 
    np.savez_compressed(
        'processed_data.npz',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        symbols=all_symbols
    )

    model.save('nasdaq_cnn_model.h5')

    np.save('training_history.npy', history.history)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    plot_training_history(history)
    evaluate_predictions(y_test, y_pred)
    plot_prediction_confidence(y_test, y_pred_proba)
    plot_predictions_over_time(y_test, y_pred, y_pred_proba, sample_size=500)
    plot_performance_curves(y_test, y_pred_proba)
    analyse_feature_importance(model, X_test, feature_names)
    plot_prediction_heatmap(y_test, y_pred, sample_size=1000)

    print(f"Total trades: {len(df):,}")
    print(f"Training windows: {len(X_train):,}")
    print(f"Test windows: {len(X_test):,}")
    print(f"Features: {X_train.shape[2]}")
    print(f"Window size: {X_train.shape[1]} trades")
    print(f"\nModel Performance:")
    print(f"  Accuracy:  {test_acc:.2%}")
    print(f"  Precision: {test_precision:.2%}")
    print(f"  Recall:    {test_recall:.2%}")
 
    return model, history, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()
