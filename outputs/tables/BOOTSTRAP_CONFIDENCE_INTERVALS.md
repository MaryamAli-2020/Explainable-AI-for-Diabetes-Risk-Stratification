# Complete bootstrap confidence intervals

These aggregate results accompany the revised CAISAIS 2026 Paper 270 analysis.
They contain no patient-level records or predictions.

- [Classification intervals](revision_classification_confidence_intervals.csv): 143 rows, covering 13 metrics for each of 11 model/operating-point combinations. Includes all four full-predictor classifiers, all four Glucose-ablation classifiers, the primary and ablation forests at their training-derived thresholds, and Random Forest with SMOTE.
- [Regression intervals](revision_regression_confidence_intervals.csv): 12 rows, covering MAE, MSE, RMSE, and R2 for Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.

Classification metrics are Accuracy, PPV, Sensitivity, Specificity, NPV, F1,
ROC-AUC, PR-AUC (average precision), Brier, FN, FP, TN, and TP. `OperatingPoint`
distinguishes estimator-default decisions from the training-selected thresholds
of 0.45 (full predictors) and 0.25 (without Glucose). Discrimination metrics use
probabilities and are independent of the decision threshold.

`Estimate` is the original held-out point estimate; `Lower95` and `Upper95` are
the 2.5th and 97.5th bootstrap percentiles. Every interval uses 2,000 replicates
with random seed 42. Classification resampling preserves the 12 positive and
66 negative test cases. Regression resamples the 78 test patients without class
stratification. Models and thresholds remain fixed; neither is retuned during
bootstrapping. Undefined bootstrap precision/NPV values are omitted from the
corresponding percentile calculation, as implemented by `numpy.nanpercentile`.

These are conditional test-set intervals, not uncertainty intervals for the
entire training and model-selection procedure. Nested-CV fold standard
deviations are a separate descriptive analysis and are not confidence intervals.

The implementation is in [src/reviewer_revision.py](../../src/reviewer_revision.py).
The results were computed previously; publishing these files did not rerun or
change the experiments.
