import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_predictions(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    accuracy = (y_pred == y_test).mean()
    up_predictions = y_pred.sum()
    down_predictions = len(y_pred) - up_predictions
    
    print(f"  Predicted Up:   {up_predictions:,} ({up_predictions/len(y_pred)*100:.1f}%)")
    print(f"  Predicted Down: {down_predictions:,} ({down_predictions/len(y_pred)*100:.1f}%)")


def plot_prediction_confidence(y_test, y_pred_proba):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(y_pred_proba, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax1.set_xlabel('Predicted Probability (Up)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    y_pred_binary = (y_pred_proba > 0.5).astype(int).flatten()
    correct_mask = (y_test == y_pred_binary)
    
    correct = y_pred_proba[correct_mask]
    incorrect = y_pred_proba[~correct_mask]
    
    ax2.hist(correct, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    ax2.hist(incorrect, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence: Correct vs Incorrect Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_confidence.png', dpi=300, bbox_inches='tight')
    print("Saved prediction_confidence.png")
    plt.close()


def plot_predictions_over_time(y_test, y_pred, y_pred_proba, sample_size=500):
    actual_size = min(sample_size, len(y_test))
    subset = slice(0, actual_size)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    x = np.arange(actual_size)
    y_test_sub = y_test[subset]
    y_pred_sub = y_pred[subset]
    y_pred_proba_sub = y_pred_proba[subset].flatten()
    
    up_mask = y_test_sub == 1
    down_mask = y_test_sub == 0
    ax1.scatter(x[up_mask], np.ones(sum(up_mask)), 
                color='green', marker='^', s=50, label='Actual Up', alpha=0.6)
    ax1.scatter(x[down_mask], np.zeros(sum(down_mask)), 
                color='red', marker='v', s=50, label='Actual Down', alpha=0.6)
    
    pred_up_mask = y_pred_sub == 1
    pred_down_mask = y_pred_sub == 0
    ax1.scatter(x[pred_up_mask], np.ones(sum(pred_up_mask)) + 0.1, 
                color='blue', marker='o', s=30, label='Predicted Up', alpha=0.4)
    ax1.scatter(x[pred_down_mask], np.zeros(sum(pred_down_mask)) + 0.1, 
                color='orange', marker='o', s=30, label='Predicted Down', alpha=0.4)
    ax1.set_ylabel('Direction')
    ax1.set_title('Predictions vs Actual Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(y_pred_proba_sub, linewidth=1, alpha=0.8)
    ax2.axhline(0.5, color='red', linestyle='--', label='Decision Threshold')
    ax2.fill_between(x, 0.5, y_pred_proba_sub, 
                     where=y_pred_proba_sub > 0.5, 
                     alpha=0.3, color='green', label='Up Confidence')
    ax2.fill_between(x, y_pred_proba_sub, 0.5, 
                     where=y_pred_proba_sub < 0.5, 
                     alpha=0.3, color='red', label='Down Confidence')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Predicted Probability (Up)')
    ax2.set_title('Model Confidence Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_curves(y_test, y_pred_proba):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    y_pred_proba_flat = y_pred_proba.flatten()
    
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba_flat)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_flat)
    avg_precision = average_precision_score(y_test, y_pred_proba_flat)
    
    ax2.plot(recall, precision, linewidth=2, label=f'PR (AP = {avg_precision:.3f})')
    ax2.axhline(y=y_test.mean(), color='k', linestyle='--', 
                linewidth=1, label=f'Baseline ({y_test.mean():.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyse_feature_importance(model, X_test, feature_names):
    sample_size = min(100, len(X_test))
    X_sample = X_test[:sample_size]
    
    X_sample_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(X_sample_tensor)
        predictions = model(X_sample_tensor, training=False)
    
    gradients = tape.gradient(predictions, X_sample_tensor)
    
    importance = np.abs(gradients.numpy()).mean(axis=(0, 1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(feature_names))
    ax.barh(x_pos, importance, color='steelblue', edgecolor='black')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Average Gradient Magnitude')
    ax.set_title('Feature Importance (Gradient-based)')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved feature_importance.png")


def plot_prediction_heatmap(y_test, y_pred, sample_size=1000):
    batch_size = 50
    actual_size = min(sample_size, len(y_test))
    n_batches = actual_size // batch_size
    
    if n_batches == 0:
        print("Not enough data for heatmap")
        return
    
    correct = (y_test[:n_batches*batch_size] == y_pred[:n_batches*batch_size]).astype(int)
    heatmap_data = correct.reshape(n_batches, batch_size)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xlabel('Prediction Index within Batch')
    ax.set_ylabel('Batch Number')
    ax.set_title('Prediction Correctness Heatmap (Green=Correct, Red=Incorrect)')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correct (1) / Incorrect (0)')
    
    plt.tight_layout()
    plt.savefig('prediction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
