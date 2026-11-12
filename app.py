"""Browser interface for training and using the Kepler transit CNN."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from exoplanet_finder.data import load_labeled_curves, load_unlabeled_curves
from exoplanet_finder.training import train_model


st.set_page_config(
    page_title="Exoplanet Finder",
    page_icon="🪐",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --space-bg: #070b17;
        --space-panel: rgba(15, 23, 45, .76);
        --space-border: rgba(161, 174, 231, .16);
        --space-lilac: #a998ff;
        --space-cyan: #7de8f5;
        --space-muted: #9aa8c7;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 9% 2%, rgba(101, 75, 181, .19), transparent 34%),
            radial-gradient(ellipse at 91% 13%, rgba(31, 134, 170, .13), transparent 31%),
            linear-gradient(160deg, #0b1021 0%, var(--space-bg) 62%, #10152a 100%);
        color: #f3f5ff;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: -35%;
        z-index: 0;
        pointer-events: none;
        opacity: .20;
        background-image:
            radial-gradient(1px 1px at 12% 21%, #fff 98%, transparent),
            radial-gradient(1px 1px at 61% 73%, #b8c5ff 98%, transparent),
            radial-gradient(1.5px 1.5px at 82% 32%, #fff 98%, transparent),
            radial-gradient(1px 1px at 37% 88%, #9deafa 98%, transparent),
            radial-gradient(1px 1px at 91% 81%, #fff 98%, transparent);
        background-size: 290px 250px;
        animation: star-drift 100s linear infinite;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] { position: relative; z-index: 1; }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    @keyframes star-drift {
        from { transform: translate3d(0, 0, 0); }
        to { transform: translate3d(120px, 85px, 0); }
    }
    @keyframes arrive {
        from { opacity: 0; transform: translateY(18px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes orbit {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(169, 152, 255, .18); }
        50% { box-shadow: 0 0 38px rgba(125, 232, 245, .30); }
    }
    .hero {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        overflow: hidden;
        padding: clamp(1.6rem, 5vw, 3.6rem);
        margin: .4rem 0 1.4rem;
        border: 1px solid var(--space-border);
        border-radius: 28px;
        background: linear-gradient(115deg, rgba(23, 30, 59, .94), rgba(11, 24, 42, .78));
        box-shadow: 0 28px 90px rgba(0, 0, 0, .28);
        animation: arrive .8s ease-out both;
    }
    .hero-copy { position: relative; z-index: 1; max-width: 730px; }
    .eyebrow {
        color: var(--space-cyan);
        font: 700 .76rem/1.2 sans-serif;
        letter-spacing: .19em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .75rem 0 .65rem;
        color: #f8f6ff;
        font: 750 clamp(2.4rem, 6vw, 4.5rem)/1.02 sans-serif;
        letter-spacing: -.055em;
    }
    .hero h1 span {
        color: var(--space-lilac);
        text-shadow: 0 0 28px rgba(169, 152, 255, .32);
    }
    .hero p {
        max-width: 620px;
        margin: 0;
        color: #bdc8e3;
        font: 400 clamp(.98rem, 2vw, 1.12rem)/1.7 sans-serif;
    }
    .orbit-scene {
        position: relative;
        flex: 0 0 210px;
        width: 210px;
        height: 210px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99, 83, 171, .18), transparent 69%);
    }
    .orbit-ring {
        position: absolute;
        inset: 20px 2px;
        border: 1px solid rgba(176, 184, 255, .40);
        border-radius: 50%;
        transform: rotate(-27deg);
    }
    .orbit-ring.ring-two {
        inset: 42px -4px;
        border-color: rgba(125, 232, 245, .24);
        transform: rotate(46deg);
    }
    .orbit-star {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 72px;
        height: 72px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #fff5cb, #f8b86b 44%, #db718d 75%);
        box-shadow: 0 0 34px rgba(255, 190, 133, .45);
        transform: translate(-50%, -50%);
    }
    .orbit-planet {
        position: absolute;
        top: 10px;
        left: 101px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: linear-gradient(135deg, #b6a7ff, #78dff0);
        box-shadow: 0 0 20px rgba(125, 232, 245, .55);
        transform-origin: 9px 95px;
        animation: orbit 12s linear infinite;
    }
    .signal-card {
        padding: 1.15rem 1.3rem;
        margin-bottom: .9rem;
        border: 1px solid var(--space-border);
        border-radius: 18px;
        background: var(--space-panel);
        animation: arrive .55s ease-out both;
        transition: transform .22s ease, border-color .22s ease, background .22s ease;
    }
    .signal-card:hover {
        transform: translateY(-3px);
        border-color: rgba(169, 152, 255, .48);
        background: rgba(22, 32, 58, .92);
    }
    .signal-label {
        color: var(--space-muted);
        font: 700 .72rem/1.4 sans-serif;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .signal-value {
        margin-top: .35rem;
        color: #f4f3ff;
        font: 650 1.02rem/1.4 sans-serif;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        border: 1px solid rgba(190, 178, 255, .35);
        border-radius: 12px;
        background: linear-gradient(100deg, #6959c7, #267e99);
        color: white;
        font-weight: 700;
        transition: transform .18s ease, filter .18s ease, box-shadow .18s ease;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.13);
        box-shadow: 0 8px 26px rgba(91, 109, 207, .30);
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed rgba(158, 177, 229, .38);
        border-radius: 16px;
        background: rgba(15, 23, 45, .48);
        transition: border-color .2s ease, background .2s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--space-cyan);
        background: rgba(22, 37, 61, .65);
    }
    div[data-testid="stTabs"] button[role="tab"] {
        transition: color .18s ease, background .18s ease;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--space-cyan);
    }
    @media (max-width: 720px) {
        .hero { padding: 1.7rem; }
        .orbit-scene { flex-basis: 110px; width: 110px; height: 110px; transform: scale(.64); }
        .hero { gap: .3rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Kepler signal lab · 1D convolutional neural network</div>
        <h1>Find the <span>transit</span><br>in the starlight.</h1>
        <p>Train on real labeled light curves. Learn the difference between
        a planet-like transit and its stellar look-alikes, then score new signals.</p>
      </div>
      <div class="orbit-scene" aria-hidden="true">
        <div class="orbit-ring"></div>
        <div class="orbit-ring ring-two"></div>
        <div class="orbit-star"></div>
        <div class="orbit-planet"></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

mode_options = {
    "Kepler candidate vs non-candidate": "kepler",
    "Planet transit vs eclipsing binary": "planet_vs_eb",
}
train_tab, predict_tab = st.tabs(["01  Train & evaluate", "02  Predict"])

with train_tab:
    st.subheader("Choose what the network should learn")
    selected_mode = st.radio(
        "Classification task",
        options=list(mode_options),
        horizontal=True,
        label_visibility="collapsed",
        key="classification_task",
    )
    label_mode = mode_options[selected_mode]

    if label_mode == "planet_vs_eb":
        st.markdown(
            """
            <div class="signal-card">
              <div class="signal-label">Event-level classification</div>
              <div class="signal-value">Planet transit ↔ eclipsing binary</div>
              <p>Use equal-length, event-centered flux windows labeled exactly
              <code>planet</code> or <code>eclipsing_binary</code>. A broad
              “non-candidate” label is not an eclipsing-binary label.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="signal-card">
              <div class="signal-label">Kepler-labelled dataset mode</div>
              <div class="signal-value">Planet candidate ↔ general non-candidate</div>
              <p>For the Kaggle Kepler-labelled CSV format: <code>LABEL=2</code>
              is candidate and <code>LABEL=1</code> is non-candidate. This
              comparison class is not specifically eclipsing binaries.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Add labeled observations")
    training_file = st.file_uploader(
        "Training light curves (one curve per row)",
        type=["csv"],
        key="training_csv",
        help=(
            "Include a LABEL column. In planet-vs-binary mode, use exact values "
            "'planet' and 'eclipsing_binary'."
        ),
    )
    evaluation_file = st.file_uploader(
        "Optional independent evaluation CSV",
        type=["csv"],
        key="evaluation_csv",
        help="A separate labeled file gives a more useful held-out score.",
    )

    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    if training_file is not None:
        try:
            x_train, y_train = load_labeled_curves(
                BytesIO(training_file.getvalue()),
                label_mode=label_mode,
            )
            st.markdown(
                f"""
                <div class="signal-card">
                  <div class="signal-label">Training set ready</div>
                  <div class="signal-value">{len(y_train):,} light curves ·
                  {x_train.shape[1]:,} samples per curve</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            curve_index = st.slider(
                "Preview a normalized light curve",
                min_value=1,
                max_value=len(y_train),
                value=1,
                key="curve_preview",
            )
            curve = x_train[curve_index - 1, :, 0]
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=np.arange(len(curve)),
                    y=curve,
                    mode="lines",
                    line={"color": "#8f80f5", "width": 1.5},
                    fill="tozeroy",
                    fillcolor="rgba(143, 128, 245, .10)",
                    hovertemplate="Sample %{x}<br>Normalized flux %{y:.3f}<extra></extra>",
                )
            )
            figure.update_layout(
                title="Normalized stellar flux",
                xaxis_title="Sample index",
                yaxis_title="Relative flux",
                height=300,
                margin={"l": 8, "r": 8, "t": 48, "b": 8},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,12,25,.35)",
                font={"color": "#c5cee5"},
                xaxis={"gridcolor": "rgba(151,166,205,.12)"},
                yaxis={"gridcolor": "rgba(151,166,205,.12)"},
                transition={"duration": 350},
            )
            st.plotly_chart(figure, use_container_width=True)
            class_counts = np.bincount(y_train.astype(np.int32), minlength=2)
            class_a, class_b = st.columns(2)
            if label_mode == "planet_vs_eb":
                class_a.metric("Planet transit examples", f"{class_counts[1]:,}")
                class_b.metric("Eclipsing binary examples", f"{class_counts[0]:,}")
            else:
                class_a.metric("Planet-candidate examples", f"{class_counts[1]:,}")
                class_b.metric("Non-candidate examples", f"{class_counts[0]:,}")
        except (
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as error:
            st.error(str(error))

    controls = st.columns([1, 1, 2])
    epoch_count = controls[0].slider("Max epochs", min_value=1, max_value=50, value=15)
    batch_size = controls[1].select_slider(
        "Batch size",
        options=[16, 32, 64, 128],
        value=32,
    )
    if controls[2].button(
        "Train the CNN",
        type="primary",
        use_container_width=True,
        disabled=training_file is None or x_train is None,
    ):
        try:
            if evaluation_file is not None:
                x_evaluation, y_evaluation = load_labeled_curves(
                    BytesIO(evaluation_file.getvalue()),
                    label_mode=label_mode,
                )
                if x_train is None or x_train.shape[1:] != x_evaluation.shape[1:]:
                    raise ValueError(
                        "Training and evaluation curves must have matching sequence lengths."
                    )
                if np.unique(y_evaluation).size != 2:
                    raise ValueError(
                        "Evaluation data must contain examples from both classes."
                    )

            progress = st.progress(0, text="Preparing the training run...")
            epoch_status = st.empty()

            class TrainingProgress(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                    values = logs or {}
                    progress.progress(
                        min((epoch + 1) / epoch_count, 1.0),
                        text=f"Learning from the signal · epoch {epoch + 1}/{epoch_count}",
                    )
                    if "val_auc" in values:
                        epoch_status.caption(
                            f"Validation AUC {values['val_auc']:.3f} · "
                            f"validation loss {values['val_loss']:.3f}"
                        )

            with st.spinner("The network is learning the light-curve patterns..."):
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
            st.session_state["classification_mode"] = label_mode
            st.session_state.pop("model_download", None)
            st.success("Training complete. Your model is ready in the Predict tab.")

            validation_auc = history.history.get("val_auc", [])
            if validation_auc:
                st.metric("Best validation AUC", f"{max(validation_auc):.3f}")

            chart_data = pd.DataFrame(
                {
                    key: values
                    for key, values in history.history.items()
                    if key in {"loss", "val_loss", "auc", "val_auc"}
                }
            )
            if not chart_data.empty:
                st.line_chart(chart_data)

            if evaluation_file is not None:
                test_metrics = model.evaluate(
                    x_evaluation,
                    y_evaluation,
                    return_dict=True,
                    verbose=0,
                )
                st.markdown("#### Independent evaluation")
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
                st.session_state["model_download"] = model_path.read_bytes()
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
    st.subheader("Score a new light curve")
    model = st.session_state.get("exoplanet_model")
    expected_shape = st.session_state.get("training_sequence_shape")
    prediction_mode = st.session_state.get("classification_mode")
    if model is None:
        st.markdown(
            """
            <div class="signal-card">
              <div class="signal-label">No model in this session yet</div>
              <div class="signal-value">Train your CNN first</div>
              <p>Your trained model stays available in this browser session.
              Start with the Train &amp; evaluate tab.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if prediction_mode == "planet_vs_eb":
            positive_label, negative_label = "Planet transit", "Eclipsing binary"
        else:
            positive_label, negative_label = "Planet candidate", "Non-candidate"

        st.markdown(
            f"""
            <div class="signal-card">
              <div class="signal-label">Model online</div>
              <div class="signal-value">{positive_label} vs {negative_label}</div>
              <p>Upload curves with {expected_shape[0]:,} flux samples each.
              Keep columns in the same order used for training.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prediction_file = st.file_uploader(
            "Unlabeled light-curve CSV",
            type=["csv"],
            key="prediction_csv",
            help="One light curve per row; flux sample columns only, no LABEL column.",
        )
        if prediction_file is not None and st.button(
            "Analyze light curves",
            type="primary",
            key="predict_button",
        ):
            try:
                features = load_unlabeled_curves(BytesIO(prediction_file.getvalue()))
                if features.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Prediction curves have shape {features.shape[1:]}; "
                        f"the trained model expects {expected_shape}."
                    )
                scores = model.predict(features, verbose=0).reshape(-1)
                results = pd.DataFrame(
                    {
                        "light_curve_row": np.arange(1, len(scores) + 1),
                        "planet_score": scores,
                        "predicted_class": np.where(
                            scores >= 0.5,
                            positive_label,
                            negative_label,
                        ),
                    }
                )
                st.markdown("#### Signal analysis")
                metric_columns = st.columns(3)
                metric_columns[0].metric("Curves analyzed", f"{len(scores):,}")
                metric_columns[1].metric(
                    f"{positive_label} scores",
                    f"{int((scores >= 0.5).sum()):,}",
                )
                metric_columns[2].metric(
                    f"{negative_label} scores",
                    f"{int((scores < 0.5).sum()):,}",
                )
                st.dataframe(
                    results.style.format({"planet_score": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                score_figure = go.Figure()
                score_figure.add_trace(
                    go.Histogram(
                        x=scores,
                        nbinsx=20,
                        marker_color="#8f80f5",
                        opacity=.88,
                    )
                )
                score_figure.add_vline(
                    x=.5,
                    line_dash="dash",
                    line_color="#7de8f5",
                    annotation_text="decision threshold",
                )
                score_figure.update_layout(
                    title="Model score distribution",
                    xaxis_title="Planet-like model score",
                    yaxis_title="Number of curves",
                    height=290,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(8,12,25,.35)",
                    font={"color": "#c5cee5"},
                    transition={"duration": 450},
                )
                st.plotly_chart(score_figure, use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="exoplanet_predictions.csv",
                    mime="text/csv",
                )
                st.caption(
                    "Scores are model outputs, not calibrated probabilities or "
                    "confirmed discoveries. Validate scientific findings independently."
                )
            except (
                ValueError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ) as error:
                st.error(str(error))

st.divider()
st.caption(
    "NASA Kepler light curves · Predictions require appropriate labeled training "
    "data and independent scientific validation."
)
