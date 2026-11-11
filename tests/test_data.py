import numpy as np
import pandas as pd
import pytest

from exoplanet_finder.data import (
    load_labeled_curves,
    load_unlabeled_curves,
    normalize_light_curves,
)


def test_load_labeled_curves_maps_kepler_labels_and_normalizes(tmp_path):
    path = tmp_path / "curves.csv"
    pd.DataFrame(
        {
            "LABEL": [1, 2],
            "flux_1": [10.0, 20.0],
            "flux_2": [12.0, 18.0],
            "flux_3": [14.0, 16.0],
        }
    ).to_csv(path, index=False)

    features, targets = load_labeled_curves(path)

    assert features.shape == (2, 3, 1)
    np.testing.assert_array_equal(targets, [0, 1])
    np.testing.assert_allclose(features.mean(axis=1), 0.0, atol=1e-6)


def test_normalize_light_curves_handles_constant_curve():
    normalized = normalize_light_curves(np.array([[3.0, 3.0, 3.0]], dtype=np.float32))

    np.testing.assert_array_equal(normalized, np.zeros((1, 3, 1), dtype=np.float32))


def test_load_labeled_curves_rejects_unknown_labels(tmp_path):
    path = tmp_path / "bad-label.csv"
    pd.DataFrame({"LABEL": [3], "flux_1": [1.0], "flux_2": [2.0]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="must be 1 or 2"):
        load_labeled_curves(path)


def test_load_unlabeled_curves_rejects_label_column(tmp_path):
    path = tmp_path / "labeled.csv"
    pd.DataFrame({"LABEL": [2], "flux_1": [1.0], "flux_2": [2.0]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="Remove the LABEL column"):
        load_unlabeled_curves(path)
