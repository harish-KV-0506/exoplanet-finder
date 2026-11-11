# Exoplanet Finder

[Open the live web app](https://harish-kv-0506-exoplanet-finder-app-ti1m2d.streamlit.app/)

A compact 1D convolutional neural network (CNN) baseline for classifying
NASA Kepler light curves. It learns patterns in stellar brightness over time
and predicts whether a labeled curve belongs to the planet-candidate class.

## Dataset

This project uses the public **Kepler-labelled time series** CSV dataset:

- [Kepler-labelled time series on Kaggle](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)

Download `exoTrain.csv` and `exoTest.csv` from the dataset page and place them
in a local `data/` directory. The CSVs are intentionally not committed to this
repository.

In this dataset, `LABEL=2` is the planet-candidate class and `LABEL=1` is the
non-candidate class. The latter is a broad comparison class; it should not be
interpreted as a specifically curated eclipsing-binary set. The model is a
learning project, not a validated scientific discovery system.

## Setup

Python 3.10 or newer is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Train and evaluate

```powershell
exoplanet-train --train data/exoTrain.csv --test data/exoTest.csv
```

The command holds out part of the training data for validation, accounts for
class imbalance during fitting, evaluates against `exoTest.csv`, and writes the
trained model and evaluation metrics under `artifacts/`.

Useful options:

```powershell
exoplanet-train --help
```

## Predict

Use the trained model on a CSV containing only numeric flux columns (no
`LABEL` column):

```powershell
exoplanet-predict --model artifacts/exoplanet_cnn.keras --input path\to\light_curves.csv
```

The command prints the probability of the planet-candidate class for each row.
Predictions are model outputs and need independent scientific validation.

## Run the web app

The browser app trains a real CNN from the labeled CSV you upload; it does not
show sample or precomputed predictions. You can optionally upload a separate
labeled CSV for held-out evaluation, then score unlabeled curves and download
the predictions or trained model.

After installing the project as described above, start the app:

```powershell
streamlit run app.py
```

Streamlit opens the app in your browser. Upload your own CSVs using the formats
described in the app. Training happens on the machine running Streamlit.

### Publish the web app

Deploy the GitHub repository to
[Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=harish-KV-0506%2Fexoplanet-finder&branch=main&mainModule=app.py).
Sign in, confirm the `main` branch and `app.py`, then deploy. The service
installs dependencies from `requirements.txt`. No model or sample dataset is
bundled; users supply the labeled CSVs in the app. When hosted, uploaded data
is processed by the hosting service, so only upload data you are authorized to
share with that service.

## Tests

```powershell
python -m pytest
```

## Resume summary

> Built a 1D CNN that detects exoplanet transits in NASA Kepler light curves,
> separating real planets from false positives like eclipsing binaries.

For accuracy, the current training labels distinguish planet candidates from a
general non-candidate class. Treat the resume wording as a project summary, not
as a claim that this baseline independently validated transit discoveries.
