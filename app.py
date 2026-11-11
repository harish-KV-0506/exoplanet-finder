"""Browser interface for training and using the Kepler transit CNN."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from exoplanet_finder.data import load_labeled_curves, load_unlabeled_curves
from exoplanet_finder.training import train_model


st.set_page_config(
    page_title="Exoplanet Finder",
    page_icon="🪐",
    layout="wide",
)

st.title("🪐 Exoplanet Finder")
st.markdown(
    "Train a 1D CNN on labeled Kepler light curves, evaluate it on held-out "
    "labeled data, then score new light curves with the trained model."
)
st.info(
    "This app trains on the data you upload; it does not use a pre-trained "
    "model or sample predictions. In the Kaggle Kepler dataset, LABEL=2 means "
    "planet candidate and LABEL=1 means non-candidate—not a curated eclipsing-binary class."
)

with st.expander("CSV format"):
    st.markdown(
        "- **Training and evaluation files:** one light curve per row, one numeric "
        "flux sample per column, plus a `LABEL` column containing only `1` or `2`.\n"
        "- **Prediction file:** the same numeric flux columns, without `LABEL`.\n"
        "- Use matching flux columns in the same order and the same number of samples "
        "for training, evaluation, and prediction files."
    )

train_tab, predict_tab = st.tabs(["Train and evaluate", "Predict"])

with train_tab:
    st.subheader("Train on your labeled light curves")
    with st.form("training_form"):
        training_file = st.file_uploader(
            "Labeled training CSV",
            type=["csv"],
            key="training_csv",
        )
        evaluation_file = st.file_uploader(
            "Optional held-out evaluation CSV",
            type=["csv"],
            key="evaluation_csv",
            help="Use a separate labeled dataset to measure performance beyond the validation split.",
        )
        epoch_count = st.slider("Maximum epochs", min_value=1, max_value=50, value=15)
        batch_size = st.select_slider(
            "Batch size",
            options=[16, 32, 64, 128],
            value=32,
        )
        train_submitted = st.form_submit_button("Train CNN", type="primary")

    if train_submitted:
        if training_file is None:
            st.error("Upload a labeled training CSV to train the model.")
        else:
            try:
                x_train, y_train = load_labeled_curves(
                    BytesIO(training_file.getvalue())
                )
                if evaluation_file is not None:
                    x_evaluation, y_evaluation = load_labeled_curves(
                        BytesIO(evaluation_file.getvalue())
                    )
                    if x_train.shape[1:] != x_evaluation.shape[1:]:
                        raise ValueError(
                            "Training and evaluation curves must have matching "
                            "sequence lengths."
                        )
                    if np.unique(y_evaluation).size != 2:
                        raise ValueError(
                            "Evaluation data must contain both LABEL=1 and LABEL=2."
                        )

                progress = st.progress(0, text="Preparing training...")
                epoch_status = st.empty()

                class TrainingProgress(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                        values = logs or {}
                        progress.progress(
                            (epoch + 1) / epoch_count,
                            text=f"Completed epoch {epoch + 1} of at most {epoch_count}",
                        )
                        if "val_auc" in values:
                            epoch_status.write(
                                f"Validation AUC: {values['val_auc']:.3f} · "
                                f"Validation loss: {values['val_loss']:.3f}"
                            )

                with st.spinner("Training the CNN on your uploaded data..."):
                    model, history = train_model(
                        x_train,
                        y_train,
                        epochs=epoch_count,
                        batch_size=batch_size,
                        callbacks=[
                            TrainingProgress(),
                            tf.keras.callbacks.EarlyStopping(
                                monitor="val_auc",
                                mode="max",
                                patience=5,
                                restore_best_weights=True,
                            ),
                        ],
                    )

                st.session_state["exoplanet_model"] = model
                st.session_state["training_sequence_shape"] = x_train.shape[1:]
                st.session_state.pop("test_metrics", None)
                st.success("Training complete. The model is ready in the Predict tab.")

                history_frame = pd.DataFrame(history.history)
                chart_columns = [
                    column
                    for column in ("loss", "val_loss", "auc", "val_auc")
                    if column in history_frame
                ]
                if chart_columns:
                    st.line_chart(history_frame[chart_columns])

                validation_auc = history.history.get("val_auc", [])
                if validation_auc:
                    st.metric("Best validation AUC", f"{max(validation_auc):.3f}")

                if evaluation_file is not None:
                    test_metrics = model.evaluate(
                        x_evaluation,
                        y_evaluation,
                        return_dict=True,
                        verbose=0,
                    )
                    st.session_state["test_metrics"] = {
                        name: float(value) for name, value in test_metrics.items()
                    }
                    st.subheader("Held-out evaluation results")
                    metric_columns = st.columns(len(test_metrics))
                    for column, (name, value) in zip(
                        metric_columns,
                        test_metrics.items(),
                        strict=True,
                    ):
                        column.metric(name.replace("_", " ").title(), f"{value:.3f}")

                with TemporaryDirectory() as temp_dir:
                    model_path = Path(temp_dir) / "exoplanet_cnn.keras"
                    model.save(model_path)
                    model_bytes = model_path.read_bytes()
                st.session_state["model_download"] = model_bytes
            except (
                ValueError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ) as error:
                st.error(str(error))

    model_download = st.session_state.get("model_download")
    if model_download is not None:
        st.download_button(
            "Download trained model",
            data=model_download,
            file_name="exoplanet_cnn.keras",
            mime="application/octet-stream",
        )

with predict_tab:
    st.subheader("Score new light curves")
    model = st.session_state.get("exoplanet_model")
    expected_shape = st.session_state.get("training_sequence_shape")
    if model is None:
        st.warning("Train a model in the first tab before making predictions.")
    else:
        st.write(f"Model loaded. Expected samples per curve: **{expected_shape[0]}**.")
        prediction_file = st.file_uploader(
            "Unlabeled light-curve CSV",
            type=["csv"],
            key="prediction_csv",
        )
        if prediction_file is not None and st.button("Predict planet-candidate probabilities"):
            try:
                features = load_unlabeled_curves(BytesIO(prediction_file.getvalue()))
                if features.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Prediction curves have shape {features.shape[1:]}; "
                        f"the trained model expects {expected_shape}."
                    )

                probabilities = model.predict(features, verbose=0).reshape(-1)
                results = pd.DataFrame(
                    {
                        "light_curve_row": np.arange(1, len(probabilities) + 1),
                        "planet_candidate_probability": probabilities,
                        "predicted_label": np.where(probabilities >= 0.5, 2, 1),
                    }
                )
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="exoplanet_predictions.csv",
                    mime="text/csv",
                )
                st.caption(
                    "Probabilities are model estimates, not confirmed discoveries. "
                    "Label 2 means candidate; label 1 means non-candidate."
                )
            except (
                ValueError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ) as error:
                st.error(str(error))

st.divider()
st.caption(
    "A machine-learning learning project—not a substitute for scientific "
    "validation or an official exoplanet confirmation."
)
