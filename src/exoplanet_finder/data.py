"""Loading and normalization helpers for Kepler light-curve CSV files."""

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


CsvSource = str | Path | IO[bytes]


def _source_name(source: CsvSource) -> str:
    name = getattr(source, "name", None)
    if name:
        return str(name)
    if isinstance(source, (str, Path)):
        return str(source)
    return "uploaded CSV"


def _flux_values(frame: pd.DataFrame, source_name: str) -> np.ndarray:
    if frame.empty:
        raise ValueError(f"No light curves found in {source_name}.")

    try:
        values = frame.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Flux columns in {source_name} must contain only numbers."
        ) from error

    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"Expected at least one flux column in {source_name}.")
    if not np.isfinite(values).all():
        raise ValueError(f"Flux values in {source_name} must all be finite.")
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


def load_labeled_curves(csv_path: CsvSource) -> tuple[np.ndarray, np.ndarray]:
    """Load a Kepler CSV with LABEL values 1 (non-candidate) or 2 (candidate)."""
    source_name = _source_name(csv_path)
    frame = pd.read_csv(csv_path)
    if "LABEL" not in frame.columns:
        raise ValueError(f"Expected a LABEL column in {source_name}.")

    labels = frame.pop("LABEL").to_numpy()
    if not np.isin(labels, [1, 2]).all():
        raise ValueError(f"LABEL values in {source_name} must be 1 or 2.")

    flux = _flux_values(frame, source_name)
    features = normalize_light_curves(flux)
    targets = (labels == 2).astype(np.int32)
    return features, targets


def load_unlabeled_curves(csv_path: CsvSource) -> np.ndarray:
    """Load and normalize a CSV containing only numeric flux columns."""
    source_name = _source_name(csv_path)
    frame = pd.read_csv(csv_path)
    if "LABEL" in frame.columns:
        raise ValueError(
            f"Remove the LABEL column from prediction input {source_name}."
        )
    return normalize_light_curves(_flux_values(frame, source_name))
