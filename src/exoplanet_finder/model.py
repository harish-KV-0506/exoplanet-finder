"""1D CNN architecture for light-curve classification."""

import tensorflow as tf


def build_model(sequence_length: int) -> tf.keras.Model:
    if sequence_length < 16:
        raise ValueError("A light curve must contain at least 16 flux samples.")

    inputs = tf.keras.Input(shape=(sequence_length, 1), name="light_curve")
    x = tf.keras.layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="candidate_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="kepler_transit_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
