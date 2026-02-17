import numpy as np
from tensorflow import keras
from sklearn.utils import class_weight
from sklearn.model_selection import TimeSeriesSplit


def prepare_training_data(X_train, y_train, use_smote=False):
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: weights[0], 1: weights[1]}
    print(f"Class weights: Down={weights[0]:.3f}, Up={weights[1]:.3f}")

    return X_train, y_train, class_weight_dict


def get_callbacks(monitor='val_auc', patience=10, reduce_lr_patience=5):
    early_stop = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'best_model_checkpoint.keras',
        monitor=monitor,
        save_best_only=True,
        mode='max',
        verbose=1
    )

    return [early_stop, reduce_lr, model_checkpoint]


def train_model(model, X_train, y_train, X_val=None, y_val=None,
                epochs=50, batch_size=64, class_weight_dict=None,
                validation_split=0.2, callbacks=None):

    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Epochs: {epochs} | Batch size: {batch_size}")
    print(f"Class weights: {class_weight_dict}")

    if callbacks is None:
        callbacks = get_callbacks()

    fit_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            **fit_kwargs
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_split=validation_split,
            **fit_kwargs
        )

    print("\nTraining complete")
    return history


def train_with_cross_validation(model_builder, X, y, n_folds=5, epochs=50, batch_size=64):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    histories, models = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{n_folds}")
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        X_fold_train, y_fold_train, cw = prepare_training_data(
            X_fold_train, y_fold_train, use_smote=False
        )
        model = model_builder(
            input_shape=(X_fold_train.shape[1], X_fold_train.shape[2])
        )
        history = train_model(
            model, X_fold_train, y_fold_train,
            X_val=X_fold_val, y_val=y_fold_val,
            epochs=epochs, batch_size=batch_size,
            class_weight_dict=cw
        )
        histories.append(history)
        models.append(model)

    return histories, models
