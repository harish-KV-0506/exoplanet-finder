"""Loading and normalization helpers for Kepler light-curve CSV files."""

from pathlib import Path

import numpy as np
import pandas as pd


def _flux_values(frame: pd.DataFrame, path: Path) -> np.ndarray:
    if frame.empty:
        raise ValueError(f"No light curves found in {path}.")

    try:
        values = frame.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Flux columns in {path} must contain only numbers.") from error

    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"Expected at least one flux column in {path}.")
    if not np.isfinite(values).all():
        raise ValueError(f"Flux values in {path} must all be finite.")
    return values


def normalize_light_curves(flux: np.ndarray) -> np.ndarray:
    """Standardize each curve independently and add the CNN channel dimension."""
    values = np.asarray(flux, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Flux data must be a non-empty 2D array.")
    if not np.isfinite(values).all():
        raise ValueError("Flux values must all be finite.")

    means = values.mean(axis=1, keepdims=True)
    deviations = values.std(axis=1, keepdims=True)
    deviations[deviations == 0] = 1.0
    return ((values - means) / deviations)[..., np.newaxis]


def load_labeled_curves(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Kepler CSV with LABEL values 1 (non-candidate) or 2 (candidate)."""
    path = Path(csv_path)
    frame = pd.read_csv(path)
    if "LABEL" not in frame.columns:
        raise ValueError(f"Expected a LABEL column in {path}.")

    labels = frame.pop("LABEL").to_numpy()
    if not np.isin(labels, [1, 2]).all():
        raise ValueError(f"LABEL values in {path} must be 1 or 2.")

    flux = _flux_values(frame, path)
    features = normalize_light_curves(flux)
    targets = (labels == 2).astype(np.int32)
    return features, targets


def load_unlabeled_curves(csv_path: str | Path) -> np.ndarray:
    """Load and normalize a CSV containing only numeric flux columns."""
    path = Path(csv_path)
    frame = pd.read_csv(path)
    if "LABEL" in frame.columns:
        raise ValueError(f"Remove the LABEL column from prediction input {path}.")
    return normalize_light_curves(_flux_values(frame, path))
