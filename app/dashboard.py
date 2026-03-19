"""
dashboard.py
------------
Streamlit dashboard for NeuroSense EEG Motor-Imagery Classification.

Run with:
    streamlit run app/dashboard.py

Features:
  • Multi-channel EEG signal plot (medical-monitor style, dark theme)
  • Predicted motor-imagery class
  • Confidence score
  • Model accuracy
  • Confusion matrix heatmap
  • Buttons: Upload Dataset, Run Classification
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSense – EEG Motor Imagery",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS (dark theme accents) ──────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; }
    /* Card-like containers */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.75rem;
    }
    /* Accent colours */
    h1, h2, h3 { color: #58a6ff; }
    .stButton > button {
        background-color: #238636;
        color: #fff;
        border-radius: 6px;
        border: none;
        font-size: 0.95rem;
        padding: 0.5rem 1.2rem;
    }
    .stButton > button:hover { background-color: #2ea043; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Matplotlib dark style ─────────────────────────────────────────────────────
plt.style.use("dark_background")

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _eeg_monitor_figure(eeg_data: np.ndarray, sfreq: float,
                         ch_names: list[str], n_channels: int = 8):
    """
    Create a multi-channel EEG time-series plot that resembles a clinical
    EEG monitor.

    Parameters
    ----------
    eeg_data : np.ndarray, shape (n_channels, n_times)
    sfreq : float – sampling frequency in Hz
    ch_names : list[str]
    n_channels : int – number of channels to display

    Returns
    -------
    fig : matplotlib Figure
    """
    n_ch = min(n_channels, eeg_data.shape[0])
    t = np.arange(eeg_data.shape[1]) / sfreq

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, n_ch * 0.9),
                              sharex=True, facecolor="#0d1117")
    fig.subplots_adjust(hspace=0.05, left=0.08, right=0.98,
                        top=0.93, bottom=0.06)
    fig.suptitle("Multi-Channel EEG Signal", color="#58a6ff",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.cool(np.linspace(0.2, 0.9, n_ch))

    for i, ax in enumerate(axes):
        signal = eeg_data[i]
        ax.plot(t, signal, color=colors[i], linewidth=0.8, alpha=0.9)
        ax.set_ylabel(ch_names[i] if i < len(ch_names) else f"Ch{i+1}",
                      color="#8b949e", fontsize=7, rotation=0,
                      labelpad=28, va="center")
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#4a5568", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#21262d")
        ax.axhline(0, color="#21262d", linewidth=0.4, linestyle="--")

    axes[-1].set_xlabel("Time (s)", color="#8b949e", fontsize=9)
    return fig


def _confusion_matrix_figure(cm: np.ndarray, class_names: list[str]):
    """Return a styled confusion-matrix heatmap figure."""
    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#161b22")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#30363d",
    )
    ax.set_facecolor("#161b22")
    ax.set_xlabel("Predicted", color="#c9d1d9", fontsize=10)
    ax.set_ylabel("True", color="#c9d1d9", fontsize=10)
    ax.set_title("Confusion Matrix", color="#58a6ff", fontsize=11)
    ax.tick_params(colors="#c9d1d9")
    fig.patch.set_facecolor("#161b22")
    plt.tight_layout()
    return fig


def _confidence_gauge(confidence: float, label: str):
    """Render a simple progress-bar-style confidence gauge."""
    pct = int(confidence * 100)
    color = "#2ea043" if pct >= 75 else ("#d29922" if pct >= 50 else "#f85149")
    st.markdown(
        f"""
        <div class="metric-card">
            <p style="margin:0; font-size:0.8rem; color:#8b949e;">
                Predicted Class</p>
            <p style="margin:0.2rem 0 0.5rem; font-size:1.4rem;
               font-weight:700; color:#58a6ff;">{label}</p>
            <p style="margin:0; font-size:0.8rem; color:#8b949e;">
                Confidence Score</p>
            <div style="background:#21262d; border-radius:4px;
                        height:18px; width:100%; margin-top:4px;">
                <div style="background:{color}; width:{pct}%;
                            height:18px; border-radius:4px;"></div>
            </div>
            <p style="margin:4px 0 0; font-size:1.1rem;
               font-weight:600; color:{color};">{pct}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "dataset_uploaded": False,
        "epochs": None,
        "pipeline": None,
        "csp": None,
        "accuracy": None,
        "cm": None,
        "prediction": None,
        "subject_id": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _init_state()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 🧠 NeuroSense")
    st.markdown(
        "<p style='color:#8b949e; margin-top:-0.5rem;'>"
        "EEG Motor Imagery Classification Dashboard</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        subject_id = st.number_input(
            "PhysioNet Subject ID",
            min_value=1,
            max_value=109,
            value=st.session_state["subject_id"],
            step=1,
            help="Subject numbers 1–109 from the PhysioNet EEG dataset.",
        )
        st.session_state["subject_id"] = int(subject_id)

        st.markdown("---")
        upload_btn = st.button("📂 Upload / Load Dataset", use_container_width=True)
        run_btn = st.button("▶️ Run Classification", use_container_width=True)
        st.markdown("---")
        st.markdown(
            "<small style='color:#6e7681;'>NeuroSense v1.0 · PhysioNet EEGBCI</small>",
            unsafe_allow_html=True,
        )

    # ── Two-column layout ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1], gap="large")

    # ── Dataset upload / loading ──────────────────────────────────────────────
    if upload_btn:
        with st.spinner("Loading EEG dataset via MNE …"):
            try:
                from app.preprocessing import run_preprocessing
                epochs = run_preprocessing(st.session_state["subject_id"])
                st.session_state["epochs"] = epochs
                st.session_state["dataset_uploaded"] = True
                st.session_state["pipeline"] = None   # reset on new data
                st.session_state["accuracy"] = None
                st.session_state["cm"] = None
                st.session_state["prediction"] = None
                st.success(
                    f"✅ Dataset Uploaded Successfully – "
                    f"{len(epochs)} epochs loaded "
                    f"(subject {st.session_state['subject_id']})."
                )
            except Exception as exc:
                st.error(f"❌ Failed to load dataset: {exc}")

    # ── Classification ────────────────────────────────────────────────────────
    if run_btn:
        if not st.session_state["dataset_uploaded"]:
            st.warning("⚠️ Please load a dataset first.")
        else:
            with st.spinner("Extracting CSP features and training SVM …"):
                try:
                    from app.feature_extraction import extract_features
                    from app.classifier import train_classifier, predict_single

                    epochs = st.session_state["epochs"]
                    X, y, csp = extract_features(epochs)
                    pipeline, accuracy, cm, X_test, y_test, y_pred = \
                        train_classifier(X, y)

                    st.session_state["csp"] = csp
                    st.session_state["pipeline"] = pipeline
                    st.session_state["accuracy"] = accuracy
                    st.session_state["cm"] = cm

                    # Predict the last test epoch as a demo
                    result = predict_single(pipeline, X_test[-1])
                    st.session_state["prediction"] = result

                    st.success("✅ Classification complete!")
                except Exception as exc:
                    st.error(f"❌ Classification failed: {exc}")

    # ── EEG signal plot ───────────────────────────────────────────────────────
    with col_left:
        st.markdown("### 📈 EEG Signal Monitor")
        if st.session_state["dataset_uploaded"] and \
                st.session_state["epochs"] is not None:
            epochs = st.session_state["epochs"]
            # Show the first epoch of whatever condition is available
            all_epochs = epochs.get_data()
            sample_epoch = all_epochs[0]          # shape: (n_ch, n_times)
            ch_names = epochs.ch_names
            sfreq = epochs.info["sfreq"]
            fig = _eeg_monitor_figure(sample_epoch, sfreq, ch_names)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Load a dataset to visualise EEG signals.")

    # ── Results panel ─────────────────────────────────────────────────────────
    with col_right:
        st.markdown("### 🎯 Classification Results")

        if st.session_state["prediction"] is not None:
            pred = st.session_state["prediction"]
            _confidence_gauge(pred["confidence"],
                               f"{pred['class_label']} Movement")

            acc = st.session_state["accuracy"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <p style="margin:0; font-size:0.8rem; color:#8b949e;">
                        Model Accuracy</p>
                    <p style="margin:0.2rem 0 0; font-size:1.6rem;
                       font-weight:700; color:#58a6ff;">
                       {acc * 100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confusion matrix
            st.markdown("#### Confusion Matrix")
            cm_fig = _confusion_matrix_figure(
                st.session_state["cm"],
                class_names=["Left Hand", "Right Hand"],
            )
            st.pyplot(cm_fig, use_container_width=True)
            plt.close(cm_fig)
        else:
            st.info("Run classification to see results here.")

    # ── Status bar ────────────────────────────────────────────────────────────
    st.divider()
    status_cols = st.columns(4)
    status_cols[0].metric(
        "Dataset",
        "Loaded ✅" if st.session_state["dataset_uploaded"] else "Not loaded",
    )
    if st.session_state["epochs"] is not None:
        status_cols[1].metric("Epochs", len(st.session_state["epochs"]))
    if st.session_state["accuracy"] is not None:
        status_cols[2].metric(
            "Accuracy", f"{st.session_state['accuracy'] * 100:.1f}%"
        )
    if st.session_state["prediction"] is not None:
        status_cols[3].metric(
            "Confidence",
            f"{int(st.session_state['prediction']['confidence'] * 100)}%",
        )


if __name__ == "__main__":
    main()
