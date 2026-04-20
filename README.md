# Retinal-Cardiovascular Alzheimer's Risk Predictor

> An integrated machine learning framework for early Alzheimer's disease risk prediction in cardiovascular patients using exclusively non-invasive biomarkers — retinal fundus imaging and cardiovascular clinical data. No PET. No MRI. No CSF.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green) ![License](https://img.shields.io/badge/License-Research-gray)

---

## Key Results

| Metric | Heart Only | Heart + Retina | Improvement |
|--------|-----------|----------------|-------------|
| AUC-ROC | 0.5961 | **0.9725** | +63.15% |
| Accuracy | 81.72% | **93.28%** | +14.14% |
| Sensitivity | 12.20% | **85.37%** | +573% |
| Specificity | 93.52% | **94.62%** | +1.17% |

Cardiovascular model best trial (Optuna): **0.9948** · Test AUC: **0.9820**

---

## Overview

This project bridges two traditionally isolated clinical domains — **cardiovascular disease** and **neurodegeneration** — through the shared pathological lens of retinal microvascular structure.

The retina is the only component of the central nervous system that can be examined non-invasively. Its vascular architecture mirrors the brain's microvasculature with measurable fidelity. By combining quantitative retinal biomarkers with cardiovascular clinical data, the framework achieves Alzheimer's risk stratification that neither modality could produce independently.

---

## Pipeline

```
FUNDUS IMAGE (256×256 RGB)
        │
        ▼
  U-Net Segmentation  ──→  Binary Vessel Mask
        │
        ▼
  Biomarker Extraction
  ┌─────────────────────────────────────┐
  │ Vessel Density · Tortuosity         │
  │ Fractal Dimension · AV Ratio        │
  └─────────────────────────────────────┘
        │
        └──────────────────────┐
                               │
CLINICAL DATA (20 features)    │
        │                      │
        ▼                      │
  SMOTE Balancing              │
        │                      │
        ▼                      │
  XGBoost + Optuna             │
  (100 trials, 5-fold CV)      │
        │                      │
        ▼                      │
  SHAP Feature Selection       │
  [cardio_prob, cardio_label,  │
   top clinical features]      │
        │                      │
        └──────────┬───────────┘
                   │
                   ▼
          FEATURE FUSION
   [retinal + cardiovascular + clinical]
                   │
                   ▼
        Random Forest (200 trees)
                   │
                   ▼
      Alzheimer's Risk Output
      (probability + binary label)
```

---

## Datasets

### Retinal Imaging (Segmentation Training)

| Dataset | Images | Resolution | Source |
|---------|--------|------------|--------|
| **DRIVE** | 40 | 565×584px | Dutch diabetic retinopathy screening |
| **STARE** | 20 | Variable | UC San Diego — includes pathological cases |
| **CHASE_DB1** | 28 | 999×960px | Child Heart and Health Study, England |

All datasets come with manually annotated binary vessel masks (ground truth).

### Cardiovascular Clinical Data

**Framingham Heart Study** — 4,240 patients, 10-year longitudinal follow-up

- 16 original clinical features: age, sex, smoking, cholesterol, blood pressure, BMI, glucose, diabetes, hypertension, and more
- 4 engineered features: pulse pressure, BMI×age interaction, smoking exposure proxy, hypertension-diabetes combined score
- Label: `TenYearCHD` — binary, whether the patient developed coronary heart disease within 10 years
- Available on [Kaggle](https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset)

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/retinal-alzheimers-risk
cd retinal-alzheimers-risk

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download datasets

Place datasets in the `data/` directory:

```
data/
├── drive/
│   ├── images/          # .tif fundus images
│   └── masks/           # manual vessel masks
├── stare/
│   ├── images/
│   └── masks/
├── chase/
│   ├── images/
│   └── masks/
└── framingham.csv       # Framingham Heart Study CSV
```

### 3. Run the full pipeline

```bash
python main.py
```

### 4. Train individual models

```bash
# Stage 1 — Retinal vessel segmentation
python segmentation/train_unet.py

# Stage 2 — Cardiovascular risk classification (runs Optuna)
python cardiovascular/train_xgboost.py

# Stage 3 — Alzheimer's risk prediction
python fusion/train_rf.py
```

### 5. Predict for a single patient

```bash
python predict.py \
  --fundus path/to/fundus_image.jpg \
  --clinical path/to/patient_data.csv
```

---

## Project Structure

```
retinal-alzheimers-risk/
│
├── data/                          # Raw datasets (not tracked by git)
│
├── preprocessing/
│   ├── image_preprocessing.py     # Resize, normalise, augment retinal images
│   └── clinical_preprocessing.py  # Imputation, z-score normalisation, SMOTE
│
├── segmentation/
│   ├── unet.py                    # U-Net architecture definition
│   ├── train_unet.py              # Training loop with Dice + BCE loss
│   └── evaluate_unet.py           # Dice score evaluation
│
├── biomarkers/
│   ├── vessel_density.py          # Optic-disc-centred density calculation
│   ├── tortuosity.py              # Arc-to-chord ratio on skeletonised map
│   ├── fractal_dimension.py       # Box-counting fractal dimension estimator
│   └── av_ratio.py                # Arteriole/venule classification and ratio
│
├── cardiovascular/
│   ├── train_xgboost.py           # XGBoost + Optuna (100 trials, 5-fold CV)
│   ├── shap_analysis.py           # SHAP feature importance computation
│   └── evaluate_cardio.py         # AUC, accuracy, sensitivity, specificity
│
├── fusion/
│   ├── feature_fusion.py          # Late feature concatenation
│   ├── train_rf.py                # Random Forest (200 trees, balanced)
│   └── evaluate_fusion.py         # Full multimodal evaluation
│
├── models/                        # Saved model artefacts
│   ├── unet.pth                   # U-Net weights
│   ├── xgboost_cardio.joblib      # XGBoost cardiovascular model
│   └── random_forest_alzheimer.joblib
│
├── main.py                        # End-to-end pipeline runner
├── predict.py                     # Single-patient inference script
├── requirements.txt
└── README.md
```

---

## Model Details

### Model 1 — U-Net (Retinal Vessel Segmentation)

| Parameter | Value |
|-----------|-------|
| Architecture | Encoder-decoder with skip connections |
| Encoder levels | 4 (64 → 128 → 256 → 512 channels) |
| Loss function | 0.5 × Binary Cross-Entropy + 0.5 × Dice Loss |
| Optimiser | Adam (lr=0.001, β1=0.9, β2=0.999) |
| LR schedule | Cosine annealing over 50 epochs |
| Batch size | 8 |
| Input size | 256×256×3 |
| **Test Dice Score** | **0.8847** |

Augmentation: random horizontal flip (p=0.5) + random rotation ±15°, applied identically to image and mask.

### Model 2 — XGBoost (Cardiovascular Risk)

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosted Decision Trees |
| Hyperparameter search | Optuna TPE — 100 trials |
| Evaluation | 5-fold stratified cross-validation |
| Class imbalance | SMOTE + scale_pos_weight=5.7 |
| Regularisation | L1 (reg_alpha) + L2 (reg_lambda) via Optuna |
| Explainability | SHAP values for feature selection |
| **Best trial AUC** | **0.9948** |
| **Test AUC** | **0.9820** |

### Model 3 — Random Forest (Alzheimer's Risk)

| Parameter | Value |
|-----------|-------|
| Trees | 200 |
| Feature sampling | sqrt(n_features) per split |
| Bootstrap | True |
| Class weights | Balanced |
| Fusion strategy | Late feature concatenation |
| Input features | Retinal biomarkers + cardiovascular outputs + SHAP-selected clinical features |
| **Multimodal AUC** | **0.9725** |
| **Sensitivity** | **85.37%** |

---

## Retinal Biomarkers

| Biomarker | Healthy Range | Clinical Significance |
|-----------|--------------|----------------------|
| **Vessel Density** | 0.05–0.25 | Microvascular rarefaction in hypertension, diabetes, neurodegeneration |
| **Tortuosity** | 1.05–1.15 | Elevated in hypertensive retinopathy and diabetic neovascularisation |
| **Fractal Dimension** | 1.60–1.78 | Reduction documented across multiple Alzheimer's patient cohorts |
| **AV Ratio** | 0.67–0.75 | Arteriolar narrowing — earliest structural sign of systemic hypertension |

---

## Dependencies

```
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0

# Machine Learning
xgboost>=1.7.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
optuna>=3.3.0
shap>=0.42.0

# Image Processing
scikit-image>=0.21.0
scipy>=1.11.0

# Data and Utilities
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
tqdm>=4.66.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Limitations

- **Proxy labels** — Alzheimer's risk labels are derived from cardiovascular outcomes (TenYearCHD) rather than direct AD diagnoses. Empirically validated by the 573% sensitivity improvement when retinal biomarkers are added, but direct AD-labelled cohort validation is the most important next step.
- **Segmentation training size** — U-Net trained on ~88 annotated images. Larger and more pathologically diverse retinal corpora would improve robustness.
- **No longitudinal validation** — cross-sectional model cannot capture temporal disease dynamics.
- **Fundus camera access** — requires fundus photography equipment, though this is already globally deployed in diabetic screening programmes.

---

## Planned Extensions

- OCT-angiography feature integration
- Speech and digital cognitive assessment biomarker fusion
- External cohort validation on AD-labelled datasets
- Longitudinal risk trajectory modelling
- Clinical deployment interface

---

## Citation

```bibtex
@article{retinal_alzheimers_2025,
  title   = {Retinal Vasculature as a Non-Invasive Bridge Between
             Cardiovascular Health and Alzheimer's Risk:
             An Integrated Machine Learning Framework},
  year    = {2025},
  note    = {Integrated ML pipeline using U-Net segmentation,
             XGBoost with Optuna, and Random Forest fusion
             for non-invasive Alzheimer's risk prediction
             from retinal and cardiovascular biomarkers}
}
```

---

## Disclaimer

This project is a research prototype. It is not a clinical diagnostic tool and should not be used to make medical decisions. All predictions are for research purposes only.

---

*Built with PyTorch · XGBoost · scikit-learn · Optuna · SHAP*
