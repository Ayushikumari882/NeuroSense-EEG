"""
feature_extraction.py
---------------------
Common Spatial Pattern (CSP) feature extraction for NeuroSense.

CSP finds spatial filters that maximise variance for one class while
minimising it for the other, making it ideal for mu/beta ERD/ERS-based
motor-imagery classification.
"""

import numpy as np
import mne
from mne.decoding import CSP
from typing import Optional


# ── Default CSP hyper-parameters ─────────────────────────────────────────────
N_COMPONENTS = 4        # number of CSP components to keep
LOG_VARIANCE = True     # use log-variance features (standard for EEG BCI)


def build_csp(n_components: int = N_COMPONENTS) -> CSP:
    """
    Instantiate an MNE CSP transformer.

    Parameters
    ----------
    n_components : int
        Number of spatial filters / log-variance features to extract.

    Returns
    -------
    csp : mne.decoding.CSP
        Unfitted CSP object (call ``csp.fit(X, y)`` before use).
    """
    csp = CSP(
        n_components=n_components,
        reg=None,
        log=LOG_VARIANCE,
        norm_trace=False,
    )
    return csp


def extract_features(epochs: mne.Epochs, csp: Optional[CSP] = None):
    """
    Fit CSP on the epochs and return feature matrix and labels.

    Parameters
    ----------
    epochs : mne.Epochs
        Pre-processed, labelled EEG epochs.
    csp : mne.decoding.CSP or None
        Pre-fitted CSP object.  If *None* a new CSP is built and fitted.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_components)
        CSP log-variance features for every epoch.
    y : np.ndarray, shape (n_epochs,)
        Integer class labels (0 = left hand / T1, 1 = right hand / T2).
    csp : mne.decoding.CSP
        Fitted CSP transformer (store this for inference on new data).
    """
    # Epoch data matrix: shape (n_epochs, n_channels, n_times)
    X = epochs.get_data()

    # Map MNE event codes back to 0 (left/T1) or 1 (right/T2)
    label_map = {"T1": 0, "T2": 1}
    event_id_inv = {v: k for k, v in epochs.event_id.items()}
    y = np.array([label_map[event_id_inv[code]] for code in epochs.events[:, 2]],
                 dtype=int)

    if csp is None:
        csp = build_csp()
        csp.fit(X, y)

    X_feat = csp.transform(X)
    return X_feat, y, csp


def get_csp_patterns(csp: CSP) -> np.ndarray:
    """
    Return the CSP spatial patterns (activation maps) for visualisation.

    Parameters
    ----------
    csp : mne.decoding.CSP
        Fitted CSP object.

    Returns
    -------
    patterns : np.ndarray, shape (n_components, n_channels)
        Spatial activation patterns.
    """
    return csp.patterns_[:csp.n_components]
