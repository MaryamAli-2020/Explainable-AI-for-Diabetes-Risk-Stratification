# Explainable Multi-Task Healthcare Analytics for Diabetes Risk Stratification

This project is a publication-style mini research study built around the Kaggle diabetes dataset "Predict Diabetes Based on Diagnostic Measures". It combines exploratory healthcare analytics, diabetes classification, glucose regression, unsupervised patient clustering, calibration analysis, threshold tuning, and explainable AI in one reproducible workflow.

## Project layout

```text
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

1. Download the Kaggle dataset from [Predict Diabetes Based on Diagnostic Measures](https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures).
2. Save the CSV file as `data/raw/diabetes.csv`.

The code assumes the diagnostic-measures schema in your Kaggle CSV:

- `Cholesterol`
- `Glucose`
- `HDLChol`
- `CholHDLRatio`
- `Age`
- `Gender`
- `Height`
- `Weight`
- `BMI`
- `SystolicBP`
- `DiastolicBP`
- `Waist`
- `Hip`
- `WaistHipRatio`
- `Outcome`

## Environment setup

The machine used to scaffold this project did not expose a system-wide `python` or `jupyter` command, so the notebook is designed to run with the bundled Codex Python runtime or a local virtual environment created from it.

### Option 1: create a local virtual environment

```powershell
& "C:\Users\mhral\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

### Option 2: install directly into the bundled runtime

```powershell
& "C:\Users\mhral\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install -r requirements.txt
& "C:\Users\mhral\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m jupyter lab
```

## Running the notebook

1. Open `notebooks/01_diabetes_healthcare_analytics.ipynb`.
2. Run all cells from top to bottom.
3. Generated figures, tables, and model artifacts will be saved automatically under `outputs/`.

## Outputs produced

- Publication-style figures at 300 dpi
- Classification and regression comparison tables
- Cluster profiles and cluster narratives
- Threshold tuning and calibration outputs
- Explainability artifacts with SHAP when installed, or permutation importance fallback

## Notes for interpretation

- The loader automatically converts comma-style decimal strings such as `22,5` and `0,84` into numeric values.
- The target labels `Diabetes` and `No diabetes` are standardized to binary `Outcome` values.
- The notebook audits suspicious zeros across numeric clinical measurements and converts them to missing values before imputation.
- Outliers are documented but not automatically removed because extreme patients may still be clinically meaningful.
- Classification model selection prioritizes ROC-AUC, while deployment discussion emphasizes recall because false negatives are risky in screening workflows.
