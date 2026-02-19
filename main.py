import os
import pandas as pd
from datetime import datetime
import numpy as np

from src.feature_engineering.engineering import create_sliding_windows, prepare_train_test_split
from src.model.model import build_cnn_model
from src.model.train import prepare_training_data, train_model, get_callbacks
from src.model.evaluate import evaluate_model
from src.utils.visualise import (
    plot_training_history,
    evaluate_predictions,
    plot_prediction_confidence,
    plot_predictions_over_time,
    plot_performance_curves,
    analyse_feature_importance,
    plot_prediction_heatmap
)


DATA_FILE = "data/processed/nasdaq_trades.csv"


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\nData file not found: {filepath}\n"
        )
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"{len(df):,} trades loaded")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    print(f"  Symbols:    {df['symbol'].unique().tolist()}")
    return df


def main():
    print("CNN Trade Execution Prediction Pipeline")

    config = {
        'data_file': DATA_FILE,
        'window_size': 100,
        'prediction_horizon': 100,
        'min_move_pct': 0.0,
        'train_test_split': 0.8,
        'epochs': 50,
        'batch_size': 64,
        'use_smote': False,
        'confidence_threshold': 0.55
    }

    feature_names = [
        'price', 'size', 'price_change', 'volume_ma', 'price_volatility',
        'trade_intensity', 'ma_crossover', 'momentum_10', 'momentum_50',
        'volume_change', 'volume_spike'
    ]

    print(f"\nLoading Data")
    df = load_data(config['data_file'])

    print(f"\nFeature Engineering")
    all_X, all_y, all_symbols = [], [], []

    for symbol in df['symbol'].unique():
        df_symbol = df[df['symbol'] == symbol].copy()
        X, y = create_sliding_windows(
            df_symbol,
            window_size=config['window_size'],
            prediction_horizon=config['prediction_horizon'],
            min_move_pct=config['min_move_pct']
        )
        all_X.append(X)
        all_y.append(y)
        all_symbols.extend([symbol] * len(X))

    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    print(f"✓ {len(X_combined):,} windows | shape: {X_combined.shape}")

    print(f"\nTrain/Test Split")
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X_combined, y_combined,
        train_ratio=config['train_test_split']
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    print(f"\nPreparing Training Data")
    X_train, y_train, class_weight_dict = prepare_training_data(
        X_train, y_train, use_smote=config['use_smote']
    )

    print(f"\nBuilding and Training CNN")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)
    model.summary()

    callbacks = get_callbacks(monitor='val_auc', patience=10)

    history = train_model(
        model, X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        class_weight_dict=class_weight_dict,
        validation_split=0.2,
        callbacks=callbacks
    )

    print(f"\nEvaluation")
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

    np.savez_compressed(
        'processed_data.npz',
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        symbols=all_symbols
    )
    model.save('nasdaq_cnn_model.keras')
    np.save('training_history.npy', history.history)

    plot_training_history(history)
    evaluate_predictions(y_test, y_pred)
    plot_prediction_confidence(y_test, y_pred_proba)
    plot_predictions_over_time(y_test, y_pred, y_pred_proba)
    plot_performance_curves(y_test, y_pred_proba)
    analyse_feature_importance(model, X_test, feature_names)
    plot_prediction_heatmap(y_test, y_pred)

    print(f"ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Recall:   {metrics['recall']:.4f}")

    return model, history, metrics


if __name__ == "__main__":
    main()
