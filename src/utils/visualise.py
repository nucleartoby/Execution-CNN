import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

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
    plt.show()

def evaluate_predictions(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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
    
    correct = y_pred_proba[y_test == ((y_pred_proba > 0.5).astype(int).flatten())]
    incorrect = y_pred_proba[y_test != ((y_pred_proba > 0.5).astype(int).flatten())]
    
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
    plt.show()
