"""
preprocessing.py
----------------
EEG data loading and preprocessing pipeline for NeuroSense.

Steps:
  1. Load PhysioNet EEG Motor Movement/Imagery data via MNE eegbci helper.
  2. Apply a band-pass filter (8–30 Hz, mu + beta bands).
  3. Remove eye-movement / muscle artefacts with ICA.
  4. Segment the continuous signal into epochs time-locked to motor-imagery cues.
     - Event T1 → Left-hand imagery
     - Event T2 → Right-hand imagery
"""

import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, events_from_annotations
from mne.preprocessing import ICA


# ── PhysioNet run numbers that contain motor-imagery tasks ────────────────────
IMAGERY_RUNS = [6, 10, 14]  # left/right hand imagery (T1/T2)

# ── Band-pass filter limits ───────────────────────────────────────────────────
FREQ_LOW = 8.0   # Hz (lower edge of mu/beta band)
FREQ_HIGH = 30.0 # Hz (upper edge of beta band)

# ── Epoch parameters ─────────────────────────────────────────────────────────
# The window starts 1 s after the cue to skip the early evoked response and
# focus on the sustained motor-imagery oscillations (mu/beta ERD/ERS).
TMIN = 1.0   # seconds after cue onset
TMAX = 2.0   # seconds after cue onset

# ── Class labels ─────────────────────────────────────────────────────────────
EVENT_ID = {"T1": 2, "T2": 3}  # T1 = left hand, T2 = right hand


def load_raw_data(subject: int = 1) -> mne.io.Raw:
    """
    Download (or load from cache) and concatenate PhysioNet EEG runs for one
    subject.

    Parameters
    ----------
    subject : int
        Subject number (1–109 in the PhysioNet dataset).

    Returns
    -------
    raw : mne.io.Raw
        Concatenated raw EEG recording.
    """
    raw_fnames = eegbci.load_data(subject, IMAGERY_RUNS)
    raw_files = [read_raw_edf(f, preload=True, stim_channel="auto") for f in raw_fnames]
    raw = concatenate_raws(raw_files)

    # Standardise channel names (e.g. "Fc5." → "FC5")
    eegbci.standardize(raw)
    mne.set_eeg_reference(raw, "average", projection=True)
    return raw


def apply_bandpass_filter(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply a zero-phase FIR band-pass filter (8–30 Hz).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.

    Returns
    -------
    raw : mne.io.Raw
        Filtered raw data (in-place modification).
    """
    raw.filter(FREQ_LOW, FREQ_HIGH, fir_window="hamming", skip_by_annotation="edge")
    return raw


def remove_artifacts(raw: mne.io.Raw, n_components: int = 20) -> mne.io.Raw:
    """
    Run ICA to identify and remove ocular and muscle artefacts.

    Parameters
    ----------
    raw : mne.io.Raw
        Band-pass filtered raw data.
    n_components : int
        Number of ICA components to decompose into.

    Returns
    -------
    raw_clean : mne.io.Raw
        Artefact-cleaned raw data.
    """
    picks_eeg = pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")

    ica = ICA(
        n_components=n_components,
        method="fastica",
        random_state=42,
        max_iter=200,
    )
    ica.fit(raw, picks=picks_eeg, decim=3)

    # Automatically detect EOG-like components (blink / eye movement)
    eog_indices, _ = ica.find_bads_eog(raw, ch_name="Fpz", threshold=3.0)
    if not eog_indices:
        # Fall back: flag components with very high kurtosis (muscle artefact)
        eog_indices = []

    ica.exclude = eog_indices
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    return raw_clean


def preprocess_raw(raw: mne.io.Raw) -> mne.Epochs:
    """
    Apply the full preprocessing chain to an already-loaded Raw object.
    """
    print("[Preprocessing] Applying band-pass filter (8–30 Hz) …")
    raw = apply_bandpass_filter(raw)

    print("[Preprocessing] Running ICA artefact removal …")
    raw = remove_artifacts(raw)

    print("[Preprocessing] Segmenting into epochs …")
    epochs = create_epochs(raw)

    print(f"[Preprocessing] Done – {len(epochs)} epochs retained "
          f"({epochs['T1'].events.shape[0]} left, "
          f"{epochs['T2'].events.shape[0]} right).")
    return epochs


def create_epochs(raw: mne.io.Raw) -> mne.Epochs:
    """
    Extract motor-imagery epochs from the cleaned raw recording.

    Parameters
    ----------
    raw : mne.io.Raw
        Artefact-cleaned, band-pass filtered raw data.

    Returns
    -------
    epochs : mne.Epochs
        Labelled epochs (T1 = left hand, T2 = right hand).
    """
    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    picks = pick_types(raw.info, eeg=True, stim=False, eog=False, exclude="bads")

    epochs = Epochs(
        raw,
        events,
        EVENT_ID,
        tmin=TMIN,
        tmax=TMAX,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    epochs.drop_bad()
    return epochs


def run_preprocessing(subject: int = 1) -> mne.Epochs:
    """
    Full preprocessing pipeline: load → filter → ICA → epoch.

    Parameters
    ----------
    subject : int
        PhysioNet subject ID.

    Returns
    -------
    epochs : mne.Epochs
        Ready-to-use labelled epochs.
    """
    print(f"[Preprocessing] Loading subject {subject} …")
    raw = load_raw_data(subject)

    return preprocess_raw(raw)


def epochs_to_xy(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert labelled epochs into model-ready inputs.

    Returns
    -------
    X : np.ndarray
        EEG epoch tensor with shape (n_epochs, n_channels, n_times).
    y : np.ndarray
        Binary labels where 0 = Left Hand (T1), 1 = Right Hand (T2).
    """
    X = epochs.get_data()
    inv_event = {v: k for k, v in epochs.event_id.items()}
    label_map = {"T1": 0, "T2": 1}
    y = np.array([label_map[inv_event[code]] for code in epochs.events[:, 2]], dtype=int)
    return X, y


def load_preprocessed_xy(subject: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience API required by the project specification:
    load PhysioNet, preprocess EEG, and return X/y.
    """
    epochs = run_preprocessing(subject)
    return epochs_to_xy(epochs)
