import os
import pandas as pd
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
    plot_prediction_heatmap,
)

DATA_FILE = "data/processed/nasdaq_trades.csv"


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"\nData file not found: {filepath}\n")
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"{len(df):,} trades loaded")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    print(f"  Symbols:    {df['symbol'].unique().tolist()}")
    return df


def main():
    print("CNN Trade Execution Prediction Pipeline")

    config = {
        "data_file": DATA_FILE,
        "window_size": 150,
        "prediction_horizon": 20,
        "min_move_pct": 0.002,
        "train_test_split": 0.8,
        "epochs": 50,
        "batch_size": 64,
        "use_smote": False,
        "confidence_threshold": 0.55,
        "l2_reg": 0.001,
        "dropout_rate": 0.4,
        "focal_alpha": 0.60,
    }

    MAX_ROWS = 5_000_000   # max trades per symbol
    STRIDE = 10            # keep every 10th trade

    feature_names = [
        "price", "size", "price_change", "volume_ma", "price_volatility",
        "trade_intensity", "ma_crossover", "momentum_10", "momentum_50",
        "volume_change", "volume_spike",
    ]

    df = load_data(config["data_file"])

    all_X, all_y, all_symbols = [], [], []

    for symbol in df["symbol"].unique():
        df_symbol = df[df["symbol"] == symbol].copy()
        df_symbol = df_symbol.iloc[:MAX_ROWS].copy()
        df_symbol = df_symbol.iloc[::STRIDE].copy()

        X, y = create_sliding_windows(
            df_symbol,
            window_size=config["window_size"],
            prediction_horizon=config["prediction_horizon"],
            min_move_pct=config["min_move_pct"],
        )
        all_X.append(X)
        all_y.append(y)
        all_symbols.extend([symbol] * len(X))

    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    print(f"{len(X_combined):,} windows | shape: {X_combined.shape}")

    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X_combined,
        y_combined,
        train_ratio=config["train_test_split"],
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    X_train, y_train, class_weight_dict = prepare_training_data(
        X_train,
        y_train,
        use_smote=config["use_smote"],
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(
        input_shape,
        l2_reg=config["l2_reg"],
        dropout_rate=config["dropout_rate"],
        focal_alpha=config["focal_alpha"],
    )
    model.summary()

    callbacks = get_callbacks(monitor="val_auc", patience=10)

    history = train_model(
        model,
        X_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight_dict=class_weight_dict,
        validation_split=0.2,
        callbacks=callbacks,
    )

    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

    high_conf_threshold = 0.75
    high_conf_up = y_pred_proba.flatten() > high_conf_threshold
    high_conf_correct = np.mean(y_test[high_conf_up])
    high_conf_count = np.sum(high_conf_up)
    print(f"  Precision: {high_conf_correct:.3f} ({np.sum(y_test[high_conf_up])}/{high_conf_count})")
    print(f"  Hit rate:  {high_conf_count/len(y_test)*100:.1f}% of predictions")

    np.savez_compressed(
        "processed_data.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        symbols=all_symbols,
    )
    model.save("nasdaq_cnn_model.keras")
    np.save("training_history.npy", history.history)

    plot_training_history(history)
    evaluate_predictions(y_test, y_pred)
    plot_prediction_confidence(y_test, y_pred_proba)
    plot_predictions_over_time(y_test, y_pred, y_pred_proba)
    plot_performance_curves(y_test, y_pred_proba)
    analyse_feature_importance(model, X_test, feature_names)
    plot_prediction_heatmap(y_test, y_pred)

    print(f"\nROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Recall:   {metrics['recall']:.4f}")

    return model, history, metrics


if __name__ == "__main__":
    main()
