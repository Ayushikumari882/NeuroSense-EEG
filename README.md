# 🧠 NeuroSense – EEG Motor Imagery Classification

A beginner-friendly, end-to-end Python project for classifying **EEG Motor Imagery** signals using the PhysioNet dataset, Common Spatial Patterns (CSP), a Support Vector Machine (SVM) classifier, and an interactive **Streamlit** dashboard.

---

## 📁 Project Structure

```
NeuroSense-EEG/
├── app/
│   ├── __init__.py
│   ├── preprocessing.py      # MNE data loading, filtering, ICA, epoching
│   ├── feature_extraction.py # CSP feature extraction
│   ├── classifier.py         # SVM training + probability calibration
│   └── dashboard.py          # Streamlit dashboard (run this)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` is included for a future GAN extension.  
> If you don't need GPU support, `pip install torch --index-url https://download.pytorch.org/whl/cpu` is faster.

### 2 – Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

### 3 – Use the dashboard

| Step | Action |
|------|--------|
| 1 | Select a **PhysioNet Subject ID** (1–109) in the sidebar |
| 2 | Click **📂 Upload / Load Dataset** – downloads and preprocesses the EEG data |
| 3 | Click **▶️ Run Classification** – extracts CSP features, trains SVM, shows results |

**Expected output:**

```
Dataset Uploaded Successfully
Predicted Motor Imagery Class: Right Hand Movement
Confidence Score: 91%
Model Accuracy: 88%
```

---

## 🧪 Pipeline Overview

```
PhysioNet EEGBCI
      │
      ▼
 preprocessing.py
  • Load EDF files (runs 6, 10, 14 – left/right hand imagery)
  • Re-reference to average
  • Band-pass filter 8–30 Hz (mu + beta bands)
  • ICA artefact removal (EOG components)
  • Epoch: 1–2 s after T1 / T2 cue
      │
      ▼
 feature_extraction.py
  • Common Spatial Pattern (CSP) – 4 components
  • Log-variance features per epoch
      │
      ▼
 classifier.py
  • StandardScaler + RBF SVM
  • Platt scaling (CalibratedClassifierCV) for confidence scores
  • 80/20 train/test split
  • Reports accuracy + confusion matrix
      │
      ▼
 dashboard.py  (Streamlit)
  • Multi-channel EEG monitor plot
  • Predicted class + confidence gauge
  • Model accuracy metric
  • Confusion matrix heatmap
```

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `mne` | EEG loading, filtering, ICA, epoching, CSP |
| `numpy` | Numerical arrays |
| `scikit-learn` | SVM, calibration, metrics |
| `matplotlib` | Plots |
| `seaborn` | Confusion matrix heatmap |
| `streamlit` | Interactive web dashboard |
| `torch` | Reserved for future GAN-based data augmentation |

---

## 🔬 Dataset

[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)  
Loaded automatically via `mne.datasets.eegbci.load_data()`.

- **T1** cue → Left-hand motor imagery  
- **T2** cue → Right-hand motor imagery  
- Runs used: 6, 10, 14 (imagined movement, eyes open)

---

## 📊 Expected Performance

| Metric | Typical value |
|--------|--------------|
| Test accuracy | 80–92% |
| Confidence (best epoch) | 85–95% |

Performance varies by subject.  Subjects 1–5 tend to give cleaner signals.

---

## 🛠️ Extending the Project

- **More classes:** add runs 7/11/15 (both fists / both feet imagery).
- **Deep learning:** replace SVM with a PyTorch EEGNet model (`torch` is already listed).
- **GAN augmentation:** use a conditional DCGAN to synthesise minority-class epochs.
- **Online BCI:** stream live EEG via `mne.realtime` and run inference epoch-by-epoch.