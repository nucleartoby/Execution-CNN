import os
import sys
from datetime import datetime
import numpy as np

from src.data.data_acquisition import download_nasdaq_data
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

def main():
    print("CNN trade execution prediction")

    config = {
        'symbols': ["ANET"],
        'start_date': datetime(2026, 2, 2),
        'end_date': datetime(2026, 2, 3),
        'output_file': 'nasdaq_trades.csv',
        'window_size': 100,
        'prediction_horizon': 100,
        'train_test_split': 0.8,
        'epochs': 50,
        'batch_size': 32,
        'use_smote': True,
        'confidence_threshold': 0.6
    }
    
    feature_names = ['price', 'size', 'price_change', 'volume_ma', 
                     'price_volatility', 'trade_intensity']
    
    df = download_nasdaq_data(
        symbols=config['symbols'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        output_file=config['output_file']
    )
    print(f"Downloaded {len(df):,} trades")
    
    all_X, all_y, all_symbols = [], [], []
    
    for symbol in df['symbol'].unique():
        df_symbol = df[df['symbol'] == symbol].copy()
        X, y = create_sliding_windows(
            df_symbol,
            window_size=config['window_size'],
            prediction_horizon=config['prediction_horizon']
        )
        all_X.append(X)
        all_y.append(y)
        all_symbols.extend([symbol] * len(X))
    
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    print(f"Total windows: {len(X_combined):,}")
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X_combined, y_combined, train_ratio=config['train_test_split']
    )
    print(f"Training: {len(X_train):,} | Test: {len(X_test):,}")
    
    X_train, y_train, class_weight_dict = prepare_training_data(
        X_train, y_train, use_smote=config['use_smote']
    )
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)
    model.summary()
    
    callbacks = get_callbacks(monitor='val_loss', patience=15)
    history = train_model(
        model, X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        class_weight_dict=class_weight_dict,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    np.savez_compressed('processed_data.npz',
                       X_train=X_train, X_test=X_test,
                       y_train=y_train, y_test=y_test,
                       symbols=all_symbols)
    model.save('nasdaq_cnn_model.keras')
    np.save('training_history.npy', history.history)
    print("Saved all artifacts")

    plot_training_history(history)
    evaluate_predictions(y_test, y_pred)
    plot_prediction_confidence(y_test, y_pred_proba)
    plot_predictions_over_time(y_test, y_pred, y_pred_proba)
    plot_performance_curves(y_test, y_pred_proba)
    analyse_feature_importance(model, X_test, feature_names)
    plot_prediction_heatmap(y_test, y_pred)
    
    print("Pipeline Complete")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    
    return model, history, metrics

if __name__ == "__main__":
    main()
