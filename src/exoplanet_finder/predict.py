"""Generate planet-candidate probabilities from unlabeled light curves."""

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from exoplanet_finder.data import load_unlabeled_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Saved Keras model.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with numeric flux columns.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    curves = load_unlabeled_curves(args.input)
    expected_shape = model.input_shape[1:]
    if curves.shape[1:] != expected_shape:
        parser.error(
            f"Input curves have shape {curves.shape[1:]}; model expects {expected_shape}."
        )

    probabilities = model.predict(curves, verbose=0).reshape(-1)
    output = pd.DataFrame(
        {
            "planet_candidate_probability": probabilities,
            "predicted_label": (probabilities >= 0.5).astype(int),
        }
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
