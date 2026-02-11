import numpy as np
from tensorflow import keras
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import TimeSeriesSplit
    

def prepare_training_data(X_train, y_train, use_smote=True):
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
        
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_2d, y_train)
        
    X_train = X_train_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    y_train = y_train_resampled
 
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class weights: {class_weight_dict}")
    
    return X_train, y_train, class_weight_dict


def get_callbacks(monitor='val_loss', patience=15, reduce_lr_patience=5):
    early_stop = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=0.00001,
        verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'best_model_checkpoint.h5',
        monitor=monitor,
        save_best_only=True,
        verbose=1
    )
    
    return [early_stop, reduce_lr, model_checkpoint]


def train_model(model, X_train, y_train, X_val=None, y_val=None,
                epochs=50, batch_size=64, class_weight_dict=None,
                validation_split=0.2, callbacks=None):

    print(f"Training samples: {len(X_train):,}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split if X_val is None else 'Separate validation set'}")
    print(f"Class weights: {class_weight_dict}")
    
    if callbacks is None:
        callbacks = get_callbacks()
    
    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

    return history


def train_with_cross_validation(model_builder, X, y, n_folds=5, epochs=50, batch_size=64):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    histories = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_folds}")
        print(f"{'='*60}")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        X_train_fold, y_train_fold, class_weights = prepare_training_data(
            X_train_fold, y_train_fold, use_smote=True
        )
        
        model = model_builder(input_shape=(X_train_fold.shape[1], X_train_fold.shape[2]))
        
        history = train_model(
            model, X_train_fold, y_train_fold,
            X_val=X_val_fold, y_val=y_val_fold,
            epochs=epochs, batch_size=batch_size,
            class_weight_dict=class_weights
        )
        
        histories.append(history)
        models.append(model)
    
    return histories, models
