import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_test, y_test, verbose=True):
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    metrics = {
        'loss': test_results[0],
        'accuracy': test_results[1],
        'precision': test_results[2],
        'recall': test_results[3],
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'predictions': {
            'up': y_pred.sum(),
            'down': len(y_pred) - y_pred.sum(),
            'up_pct': y_pred.sum() / len(y_pred) * 100
        }
    }
    
    if verbose:
        print("Test Set Evaluation")
        print(f"Loss:      {metrics['loss']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nPrediction Distribution:")
        print(f"  Up:   {metrics['predictions']['up']:,} ({metrics['predictions']['up_pct']:.1f}%)")
        print(f"  Down: {metrics['predictions']['down']:,} ({100-metrics['predictions']['up_pct']:.1f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    return metrics, y_pred, y_pred_proba


def calculate_trading_metrics(y_test, y_pred_proba, prices_test, 
                              confidence_threshold=0.6, transaction_cost=0.001):
    high_confidence_mask = (y_pred_proba > confidence_threshold) | (y_pred_proba < (1 - confidence_threshold))
    high_confidence_preds = (y_pred_proba > 0.5).astype(int).flatten()
    
    n_trades = high_confidence_mask.sum()
    
    correct_trades = (high_confidence_preds[high_confidence_mask] == y_test[high_confidence_mask]).sum()
    accuracy_on_trades = correct_trades / n_trades
    
    returns = []
    for i in range(len(y_test)):
        if high_confidence_mask[i]:
            predicted_direction = 1 if y_pred_proba[i] > 0.5 else -1
            actual_direction = 1 if y_test[i] == 1 else -1
            
            if predicted_direction == actual_direction:
                returns.append(abs(prices_test[i+1] - prices_test[i]) / prices_test[i] - transaction_cost)
            else:
                returns.append(-abs(prices_test[i+1] - prices_test[i]) / prices_test[i] - transaction_cost)
    
    returns = np.array(returns)
    
    metrics = {
        'n_trades': n_trades,
        'accuracy_on_trades': accuracy_on_trades,
        'win_rate': (returns > 0).sum() / len(returns),
        'total_return': returns.sum(),
        'avg_return_per_trade': returns.mean(),
        'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
        'max_drawdown': (np.maximum.accumulate(returns.cumsum()) - returns.cumsum()).max()
    }
    
    return metrics
