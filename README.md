# Exoplanet Finder

[Open the live web app](https://harish-kv-0506-exoplanet-finder-app-ti1m2d.streamlit.app/)

A compact 1D convolutional neural network (CNN) baseline for classifying
NASA Kepler light curves. The web app supports both Kepler candidate vs.
non-candidate data and an explicit planet-transit vs. eclipsing-binary task.
The latter requires event-level examples whose labels identify those two
classes; the broad non-candidate label in the standard Kepler CSV does not
identify eclipsing binaries.

## Dataset

This project uses the public **Kepler-labelled time series** CSV dataset:

- [Kepler-labelled time series on Kaggle](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)

Download `exoTrain.csv` and `exoTest.csv` from the dataset page and place them
in a local `data/` directory. The CSVs are intentionally not committed to this
repository.

In this dataset, `LABEL=2` is the planet-candidate class and `LABEL=1` is the
non-candidate class. It is useful for the candidate-vs-non-candidate task, but
not for claiming a planet-vs-eclipsing-binary classification. For the latter,
prepare a separately labeled set of equal-length, event-centered transit and
eclipse flux windows from a documented source such as the NASA
[Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and
[MAST Kepler archive](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html).
Keep the class labels tied to the source catalog's dispositions and document
the selection process. The model is a learning project, not a validated
scientific discovery system.

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

The browser app trains a CNN from the labeled CSV you upload; it does not show
sample or precomputed predictions. Select one of two objectives:

- **Kepler candidate vs non-candidate:** accepts `LABEL=1` and `LABEL=2` from
  the Kepler-labelled dataset. This does not specifically identify eclipsing
  binaries.
- **Planet transit vs eclipsing binary:** requires explicit `LABEL` values
  `planet` and `eclipsing_binary`. Use equal-length, event-centered windows,
  rather than treating an unlabeled/broad false-positive class as binaries.

Each row is one curve/window; remaining columns must be numeric flux samples.
The prediction CSV has the same flux columns without `LABEL`. For better
evaluation, upload a separate labeled dataset that includes both classes.
After training, score unlabeled curves and download predictions or the trained
model.

Run locally:

```powershell
streamlit run app.py
```

Streamlit opens the app in your browser. Upload your own CSVs using the formats
described in the app. Training happens on the machine running Streamlit. The
interface uses animated motion effects and honors reduced-motion browser
preferences.

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
