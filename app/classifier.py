"""
classifier.py
-------------
SVM-based motor-imagery classifier for NeuroSense.

Pipeline:
  1. Standardise features (zero mean, unit variance).
  2. Train a radial-basis-function SVM.
  3. Wrap with CalibratedClassifierCV for well-calibrated probability scores.
  4. Evaluate with accuracy and confusion matrix.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    StratifiedKFold,
)
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Label names for display ───────────────────────────────────────────────────
CLASS_NAMES = ["Left Hand", "Right Hand"]


def build_classifier() -> Pipeline:
    """
    Build an sklearn Pipeline: StandardScaler → calibrated RBF-SVM.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Unfitted classification pipeline.
    """
    base_svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=False,   # CalibratedClassifierCV adds probabilities
        random_state=42,
    )
    calibrated_svm = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",       # Platt scaling
        cv=StratifiedKFold(n_splits=3),
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", calibrated_svm),
        ]
    )
    return pipeline


def train_classifier(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """
    Split data, train the pipeline, and return results.

    Parameters
    ----------
    X : np.ndarray, shape (n_epochs, n_features)
        Feature matrix from CSP.
    y : np.ndarray, shape (n_epochs,)
        Class labels.
    test_size : float
        Fraction of data reserved for evaluation.

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    accuracy : float  (0–1)
    cm : np.ndarray, shape (2, 2)  – confusion matrix
    X_test : np.ndarray
    y_test : np.ndarray
    y_pred : np.ndarray
    cv_score : float
        Mean cross-validation score on the training split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    cv_scores = cross_val_score(
        build_classifier(),
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    )

    pipeline = build_classifier()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(
        f"[Classifier] Cross-validation score: "
        f"{cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%"
    )
    print(f"[Classifier] Test accuracy: {accuracy * 100:.1f}%")
    print(f"[Classifier] Confusion matrix:\n{cm}")

    return pipeline, accuracy, cm, X_test, y_test, y_pred, float(cv_scores.mean())


def predict_single(pipeline, X_single: np.ndarray) -> dict:
    """
    Run inference on a single feature vector.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_single : np.ndarray, shape (1, n_features) or (n_features,)
        Feature vector for one epoch.

    Returns
    -------
    result : dict
        Keys: ``class_label`` (str), ``class_index`` (int),
        ``confidence`` (float, 0–1), ``probabilities`` (np.ndarray).
    """
    X_single = np.atleast_2d(X_single)
    class_index = int(pipeline.predict(X_single)[0])
    proba = pipeline.predict_proba(X_single)[0]
    confidence = float(proba[class_index])

    return {
        "class_index": class_index,
        "class_label": CLASS_NAMES[class_index],
        "confidence": confidence,
        "probabilities": proba,
    }
