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
from mne.io import read_raw_edf
from mne.time_frequency import psd_array_welch

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
    .stApp { background-color: #000000; color: #f5f7fb; }
    section[data-testid="stSidebar"] { background-color: #0b0c10; }
    .metric-card {
        background: #0d1117;
        border: 1px solid #1f2833;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    }
    h1, h2, h3, h4 { color: #61dafb; }
    .stButton > button {
        background: linear-gradient(90deg, #0b84ff, #1f6feb);
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #1f6feb;
        font-size: 0.95rem;
        padding: 0.55rem 1.2rem;
    }
    .stButton > button:hover { border-color: #8ab4ff; }
    .eeg-card {
        background: #0d1117;
        border: 1px solid #1f2833;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    }
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

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, n_ch * 0.95),
                              sharex=True, facecolor="#000000")
    fig.subplots_adjust(hspace=0.07, left=0.08, right=0.98,
                        top=0.9, bottom=0.08)
    fig.suptitle("Multi-Channel EEG Signal", color="#61dafb",
                 fontsize=14, fontweight="bold")

    base_palette = ["#8ab4ff", "#66c2ff", "#ffffff", "#4da3ff",
                    "#7fb0ff", "#b3d1ff", "#99c9ff", "#d7e8ff"]

    for i, ax in enumerate(axes):
        signal = eeg_data[i]
        color = base_palette[i % len(base_palette)]
        ax.plot(t, signal, color=color, linewidth=0.9, alpha=0.95)
        ax.set_ylabel(ch_names[i] if i < len(ch_names) else f"Ch{i+1}",
                      color="#cfd8e3", fontsize=7, rotation=0,
                      labelpad=28, va="center")
        ax.set_facecolor("#000000")
        ax.tick_params(colors="#6b7280", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1f2833")
        ax.axhline(0, color="#1f2833", linewidth=0.4, linestyle="--")

    axes[-1].set_xlabel("Time (s)", color="#cfd8e3", fontsize=9)
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


def _spectrogram_figure(signal: np.ndarray, sfreq: float, ch_label: str):
    """Return a spectrogram focused on mu/beta activity."""
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#000000")
    ax.specgram(signal, Fs=sfreq, NFFT=256, noverlap=128, cmap="magma")
    ax.axhspan(8, 30, color="#1f6feb", alpha=0.2, label="8–30 Hz")
    ax.set_ylabel("Frequency (Hz)", color="#cfd8e3")
    ax.set_xlabel("Time (s)", color="#cfd8e3")
    ax.set_title(f"Spectrogram – {ch_label}", color="#61dafb", fontsize=11)
    ax.tick_params(colors="#cfd8e3")
    ax.legend(facecolor="#0d1117", edgecolor="#1f2833")
    fig.patch.set_facecolor("#000000")
    plt.tight_layout()
    return fig


def _bandpower_figure(epoch: np.ndarray, sfreq: float):
    """Return band-power bars for mu (8–13) and beta (13–30) bands."""
    psd, freqs = psd_array_welch(
        epoch,
        sfreq=sfreq,
        fmin=1,
        fmax=40,
        n_per_seg=min(epoch.shape[-1], int(sfreq * 2)),
    )
    mu_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs > 13) & (freqs <= 30)
    mu_power = psd[:, mu_mask].mean()
    beta_power = psd[:, beta_mask].mean()

    fig, ax = plt.subplots(figsize=(4.5, 3.2), facecolor="#000000")
    bars = ax.bar(
        ["Mu (8–13 Hz)", "Beta (13–30 Hz)"],
        [mu_power, beta_power],
        color=["#8ab4ff", "#4da3ff"],
        edgecolor="#1f2833",
    )
    ax.set_ylabel("Normalized Power", color="#cfd8e3")
    ax.set_title("Band Power", color="#61dafb", fontsize=11)
    ax.tick_params(colors="#cfd8e3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1f2833")
    ax.bar_label(bars, fmt="%.2f", label_type="edge", color="#cfd8e3")
    fig.patch.set_facecolor("#000000")
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
        "cv_score": None,
        "cm": None,
        "prediction": None,
        "subject_id": 1,
        "dataset_source": "PhysioNet",
        "uploaded_name": None,
        "uploaded_processed": False,
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
    st.markdown("# 🧠 NeuroSense EEG System")
    st.markdown(
        "<p style='color:#cfd8e3; margin-top:-0.5rem;'>"
        "Clinical-style motor imagery monitoring and classification</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎛️ Controls")
        st.caption("T1 → Left Hand · T2 → Right Hand")
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
        download_btn = st.button("📥 Download & Load Dataset", use_container_width=True)
        uploaded_file = st.file_uploader("📂 Upload EDF File", type=["edf"])
        run_btn = st.button("▶️ Run Classification", use_container_width=True)
        synth_btn = st.button("🧪 Generate Synthetic Data", use_container_width=True)
        st.markdown("---")
        st.markdown("#### Dataset Info")
        if st.session_state["epochs"] is not None:
            epochs = st.session_state["epochs"]
            st.markdown(
                f"- **Source:** {st.session_state['dataset_source']}  \n"
                f"- **Subject:** {st.session_state['subject_id']}  \n"
                f"- **Epochs:** {len(epochs)}  \n"
                f"- **Channels:** {len(epochs.ch_names)}  \n"
                f"- **Sampling Rate:** {epochs.info['sfreq']:.1f} Hz"
            )
        else:
            st.markdown(
                "- Source: pending  \n"
                "- Subject: –  \n"
                "- Epochs: –  \n"
                "- Channels: –  \n"
                "- Sampling Rate: –"
            )
        st.markdown("---")
        st.markdown(
            "<small style='color:#6e7681;'>NeuroSense v1.0 · PhysioNet EEGBCI</small>",
            unsafe_allow_html=True,
        )

    # ── Two-column layout ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1], gap="large")

    # ── Dataset upload / loading ──────────────────────────────────────────────
    if download_btn:
        with st.spinner("Downloading and preprocessing PhysioNet EEGBCI …"):
            try:
                from app.preprocessing import run_preprocessing

                epochs = run_preprocessing(st.session_state["subject_id"])
                st.session_state["epochs"] = epochs
                st.session_state["dataset_uploaded"] = True
                st.session_state["pipeline"] = None
                st.session_state["accuracy"] = None
                st.session_state["cv_score"] = None
                st.session_state["cm"] = None
                st.session_state["prediction"] = None
                st.session_state["dataset_source"] = "PhysioNet"
                st.session_state["uploaded_name"] = None
                st.session_state["uploaded_processed"] = False
                st.success(
                    f"✅ Dataset ready – {len(epochs)} epochs (subject "
                    f"{st.session_state['subject_id']})."
                )
            except Exception as exc:
                st.error(f"❌ Failed to load dataset: {exc}")

    if uploaded_file is not None and (
        not st.session_state["uploaded_processed"]
        or uploaded_file.name != st.session_state["uploaded_name"]
    ):
        with st.spinner("Processing uploaded EDF file …"):
            try:
                from app.preprocessing import preprocess_raw

                raw = read_raw_edf(
                    io.BytesIO(uploaded_file.getvalue()),
                    preload=True,
                    stim_channel="auto",
                )
                epochs = preprocess_raw(raw)
                st.session_state["epochs"] = epochs
                st.session_state["dataset_uploaded"] = True
                st.session_state["pipeline"] = None
                st.session_state["accuracy"] = None
                st.session_state["cv_score"] = None
                st.session_state["cm"] = None
                st.session_state["prediction"] = None
                st.session_state["dataset_source"] = "EDF Upload"
                st.session_state["uploaded_name"] = uploaded_file.name
                st.session_state["uploaded_processed"] = True
                st.success(
                    f"✅ EDF uploaded – {len(epochs)} epochs processed "
                    f"({len(epochs.ch_names)} channels)."
                )
            except Exception as exc:
                st.error(f"❌ Failed to process EDF: {exc}")

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
                    pipeline, accuracy, cm, X_test, y_test, y_pred, cv_score = \
                        train_classifier(X, y)

                    st.session_state["csp"] = csp
                    st.session_state["pipeline"] = pipeline
                    st.session_state["accuracy"] = accuracy
                    st.session_state["cv_score"] = cv_score
                    st.session_state["cm"] = cm

                    # Predict the last test epoch as a demo
                    result = predict_single(pipeline, X_test[-1])
                    st.session_state["prediction"] = result

                    st.success("✅ Classification complete!")
                except Exception as exc:
                    st.error(f"❌ Classification failed: {exc}")

    if synth_btn:
        st.info("Synthetic data generation is a placeholder for future GAN integration.")

    sample_epoch = None
    ch_names = []
    sfreq = None
    if st.session_state["dataset_uploaded"] and st.session_state["epochs"] is not None:
        epochs = st.session_state["epochs"]
        data = epochs.get_data()
        if len(data):
            sample_epoch = data[0]
            ch_names = epochs.ch_names
            sfreq = epochs.info["sfreq"]

    # ── EEG signal plot ───────────────────────────────────────────────────────
    with col_left:
        st.markdown("### 📈 EEG Signal Monitor")
        if sample_epoch is not None:
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
            cv = st.session_state["cv_score"]
            metrics_col1, metrics_col2 = st.columns(2, gap="small")
            with metrics_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <p style="margin:0; font-size:0.8rem; color:#8b949e;">
                            Model Accuracy</p>
                        <p style="margin:0.2rem 0 0; font-size:1.6rem;
                           font-weight:700; color:#61dafb;">
                           {acc * 100:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with metrics_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <p style="margin:0; font-size:0.8rem; color:#8b949e;">
                            Cross-validation</p>
                        <p style="margin:0.2rem 0 0; font-size:1.6rem;
                           font-weight:700; color:#61dafb;">
                           {cv * 100:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Run classification to see results here.")

    # ── Feature visualisations ────────────────────────────────────────────────
    st.markdown("### 🔬 Feature Visualisations")
    if sample_epoch is not None:
        feat_col1, feat_col2, feat_col3 = st.columns([1.2, 1, 1], gap="large")
        with feat_col1:
            ch_label = ch_names[0] if ch_names else "Ch1"
            spec_fig = _spectrogram_figure(sample_epoch[0], sfreq, ch_label)
            st.pyplot(spec_fig, use_container_width=True)
            plt.close(spec_fig)
        with feat_col2:
            bp_fig = _bandpower_figure(sample_epoch, sfreq)
            st.pyplot(bp_fig, use_container_width=True)
            plt.close(bp_fig)
        with feat_col3:
            if st.session_state["cm"] is not None:
                cm_fig = _confusion_matrix_figure(
                    st.session_state["cm"],
                    class_names=["Left Hand", "Right Hand"],
                )
                st.pyplot(cm_fig, use_container_width=True)
                plt.close(cm_fig)
            else:
                st.info("Run classification to view confusion matrix.")
    else:
        st.info("Load a dataset to explore feature visualisations.")

    # ── Final result ──────────────────────────────────────────────────────────
    st.markdown("### 🏁 Final Result")
    if st.session_state["prediction"] is not None:
        pred = st.session_state["prediction"]
        st.markdown(
            f"""
            <div class="eeg-card">
                <p style="margin:0; color:#8b949e; font-size:0.9rem;">
                    Predicted Class</p>
                <p style="margin:0.2rem 0 0.4rem; font-size:1.6rem;
                          font-weight:700; color:#61dafb;">
                    {pred['class_label']} Movement</p>
                <p style="margin:0; color:#8b949e; font-size:0.9rem;">
                    Confidence Score</p>
                <p style="margin:0.2rem 0 0.4rem; font-size:1.4rem;
                          font-weight:600; color:#8ab4ff;">
                    {pred['confidence'] * 100:.1f}%</p>
                <p style="margin:0; color:#8b949e; font-size:0.9rem;">
                    Accuracy · Cross-validation</p>
                <p style="margin:0.2rem 0 0; font-size:1.2rem;
                          font-weight:600; color:#8ab4ff;">
                    {st.session_state['accuracy'] * 100:.1f}% · {st.session_state['cv_score'] * 100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Results will appear here after running classification.")

    # ── Status bar ────────────────────────────────────────────────────────────
    st.divider()
    status_cols = st.columns(5)
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
    if st.session_state["cv_score"] is not None:
        status_cols[3].metric(
            "Cross-val", f"{st.session_state['cv_score'] * 100:.1f}%"
        )
    if st.session_state["prediction"] is not None:
        status_cols[4].metric(
            "Confidence",
            f"{int(st.session_state['prediction']['confidence'] * 100)}%",
        )


if __name__ == "__main__":
    main()
