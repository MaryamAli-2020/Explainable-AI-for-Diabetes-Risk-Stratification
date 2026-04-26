# Explainable Multi-Task Healthcare Analytics for Diabetes Risk Stratification

This project is a publication-style mini research study built around the Kaggle diabetes dataset "Predict Diabetes Based on Diagnostic Measures". It combines exploratory healthcare analytics, diabetes classification, glucose regression, unsupervised patient clustering, calibration analysis, threshold tuning, and explainable AI in one reproducible workflow.

## Project layout

```
diabetes-mini-study/
  data/raw/diabetes.csv
  notebooks/01_diabetes_healthcare_analytics.ipynb
  src/
  outputs/figures/
  outputs/tables/
  outputs/models/
  report/mini_paper_outline.md
  requirements.txt
```

## Dataset setup

1. Download the Kaggle dataset from:
   https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures
2. Save the CSV file as `data/raw/diabetes.csv`.

The dataset should include the following fields:

- Cholesterol
- Glucose
- HDLChol
- CholHDLRatio
- Age
- Gender
- Height
- Weight
- BMI
- SystolicBP
- DiastolicBP
- Waist
- Hip
- WaistHipRatio
- Outcome

## Environment Setup

This project is designed to be reproducible using a standard Python environment.

### Option 1: Virtual Environment (Recommended)

```
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

### Option 2: Conda Environment

```
conda create -n diabetes-study python=3.10 -y
conda activate diabetes-study
pip install -r requirements.txt
jupyter lab
```

## Running the Notebook

1. Open `notebooks/01_diabetes_healthcare_analytics.ipynb`
2. Run all cells from top to bottom
3. Outputs will be saved automatically in:

```
outputs/figures/
outputs/tables/
outputs/models/
```

## Outputs Produced

- Publication-style figures (300 DPI)
- Classification and regression comparison tables
- Cluster profiles and clinical narratives
- Threshold tuning and calibration plots
- Explainability outputs (SHAP or permutation importance)

## Notes for Interpretation

- Comma-style decimal values (e.g., 22,5) are automatically converted to numeric format
- Target labels are standardized to binary Outcome values
- Suspicious zero values are treated as missing and imputed
- Outliers are retained for clinical relevance
- Model selection prioritizes ROC-AUC; recall is emphasized for screening safety

## Reproducibility Notes

- All dependencies are listed in `requirements.txt`
- The workflow is deterministic when random seeds are fixed
- Compatible with local and cloud-based notebook environments (Jupyter, Kaggle, Colab)
