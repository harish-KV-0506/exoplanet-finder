"""Shared model training workflow for the CLI and web application."""

from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from exoplanet_finder.model import build_model


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int = 42,
    callbacks: Sequence[tf.keras.callbacks.Callback] | None = None,
    verbose: int = 0,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train with a stratified validation split and balanced class weights."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if features.ndim != 3 or features.shape[0] != labels.shape[0]:
        raise ValueError("Features and labels must have matching curve counts.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must contain only 0 and 1.")
    if np.bincount(labels.astype(np.int32), minlength=2).min() < 3:
        raise ValueError(
            "At least three examples of each class are required so both "
            "training and validation splits contain both labels."
        )

    x_train, x_validation, y_train, y_validation = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes.tolist(), weights.tolist(), strict=True))

    tf.keras.utils.set_random_seed(seed)
    model = build_model(sequence_length=features.shape[1])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=list(callbacks or []),
        verbose=verbose,
    )
    return model, history
