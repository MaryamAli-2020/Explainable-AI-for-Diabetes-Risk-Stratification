# Explainable Machine Learning for Diabetes Status Assessment

This project assesses recorded diabetes status in the Kaggle dataset "Predict Diabetes Based on Diagnostic Measures". It combines classification, glucose regression, descriptive patient clustering, calibration, training-only threshold selection, and model explanations. It does not estimate future disease incidence or validate subgroup-specific interventions.

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

## Reviewer revision analyses

Run `.\.venv\Scripts\python.exe run_reviewer_revision.py` from the project root to
reproduce the Glucose ablation, additional clinical classification metrics, SMOTE
sensitivity results, and 2,000-replicate bootstrap confidence intervals. The
primary model is selected using training CV; the test partition is never used to
select a model or threshold. The original corrected training OOF predictions are
reused for the unchanged primary threshold; the ablation regenerates nested OOF
predictions. All ablation models use the same 312 training and 78 test patients.

Read `outputs/tables/reviewer_revision_audit.json` for exact search grids and best
parameters, and the revision confidence-interval CSVs for every reported metric.
`requirements-lock.txt` records exact installed package versions for the revision
runtime; install it in a fresh Windows Python 3.12 environment to reproduce that stack.
Model-selection CV scores are tuning scores, not fully nested generalization
estimates. Test bootstrap intervals condition on the fitted model and fixed class
counts. Run `verify_label_provenance.py` to reproduce record matching to the primary
403-record cohort. All 390 export labels agree with primary glycosylated hemoglobin
greater than 7.0; publisher adjudication and assay timing remain undocumented.
The target is recorded diabetes status. This is exploratory research, not a screening
tool.

After the analyses, run `refresh_revision_figures.py` for annotated ROC/PR figures,
`refresh_shap_direct.py` for verified Random Forest explanations,
`refresh_regression_selection.py` for the training-selected regressor's diagnostic
figure. These scripts use saved fits wherever possible.

For a fresh checkout, first run the notebook top to bottom (or use
`rerun_notebook.py`) to regenerate the corrected primary fits and training OOF
predictions. Then run `verify_threshold_results.py`, `verify_label_provenance.py`,
`run_reviewer_revision.py`, and the refresh scripts above in that order.
Generated results in an older checkout may predate these corrections; regenerate
them before running the additional nested experiment.

## Nested Complete Pipeline Evaluation

Run `.\.venv\Scripts\python.exe run_nested_pipeline.py` for the additional nested
experiment. Five stratified outer folds use only the original 312 training
patients. Each outer-training partition tunes all four families with the same
grids and five-fold inner CV repeated three times, selects the best inner AUC,
then derives its threshold from a separate five-fold training-only cross-fit.
Hyperparameters are retuned within each threshold-training subfold. The selected
family stays fixed during that threshold cross-fit. Only afterward is the locked
pipeline evaluated on the outer-validation patients.

The 78 test patients are excluded; original fitted models, held-out predictions,
and reported thresholds are hash-protected and not replaced. This experiment is
additional evaluation of the selection procedure, not retrospective model choice.
`verify_nested_pipeline.py` independently checks partitions, family selection,
thresholds, outer metrics, means/SD, and preserved test artifacts. The outer-fold
SD is descriptive, not a confidence interval. Computation exceeds the 6,450 fits
needed for family searches because threshold cross-fitting adds further searches.
The completed run gave mean outer ROC-AUC 0.9185 (sample SD 0.0434), AP 0.7671
(SD 0.1171), sensitivity 0.8956 (SD 0.0710), and PPV 0.5410 (SD 0.1497).
Random Forest was selected in four folds and SVM in one. Thresholds ranged from
0.10 to 0.35; these fold-specific choices do not replace the original test cutoff
0.45. Run `test_nested_isolation.py` for a small synthetic regression test that
checks training/validation separation and invariance to held-out test labels.

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
