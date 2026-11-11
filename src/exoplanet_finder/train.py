"""Train and evaluate the Kepler light-curve CNN."""

import argparse
import json
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from exoplanet_finder.data import load_labeled_curves
from exoplanet_finder.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Labeled training CSV.")
    parser.add_argument("--test", type=Path, required=True, help="Labeled test CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")

    tf.keras.utils.set_random_seed(args.seed)
    x, y = load_labeled_curves(args.train)
    x_test, y_test = load_labeled_curves(args.test)
    if x.shape[1:] != x_test.shape[1:]:
        parser.error("Training and test curves must have matching sequence lengths.")

    x_train, x_validation, y_train, y_validation = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    classes = [0, 1]
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights, strict=True))

    model = build_model(sequence_length=x.shape[1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "exoplanet_cnn.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    model = tf.keras.models.load_model(model_path)
    metrics = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
    results = {name: float(value) for name, value in metrics.items()}
    metrics_path = args.output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved test metrics: {metrics_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
