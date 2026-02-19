import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)

        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)

        ce = -tf.math.log(p_t)
        loss = focal_weight * ce
        return tf.reduce_mean(loss)
    return loss_fn


def build_cnn_model(input_shape):
    model = keras.Sequential([
        layers.Conv1D(
            filters=32, kernel_size=5, activation='relu',
            input_shape=input_shape, padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(
            filters=64, kernel_size=3, activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(
            filters=64, kernel_size=3, activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=focal_loss(gamma=2.0, alpha=0.50),
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    return model
