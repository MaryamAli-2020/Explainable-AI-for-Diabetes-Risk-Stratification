"""Regenerate reviewer figures from saved fitted models, with counts and CI labels."""
import os
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from src.data import load_diabetes_dataset, prepare_dataset
from src.preprocess import split_classification_data
from src.models import evaluate_classifier
from src.plots import (set_publication_style, save_figure, plot_roc_curves, plot_calibration_curves,
    plot_class_distribution, plot_threshold_sweep, plot_regression_diagnostics)

root = Path(__file__).resolve().parent
tables = root / "outputs/tables"
bundle = prepare_dataset(load_diabetes_dataset())
X_train, X_test, y_train, y_test = split_classification_data(bundle.clean)
ci = pd.read_csv(tables / "revision_classification_confidence_intervals.csv")
runs = []
for name in ("Logistic Regression", "Random Forest", "Support Vector Machine", "XGBoost"):
    path = root / "outputs/models" / ("revision_full_predictors_" + name.lower().replace(" ", "_") + ".joblib")
    model = joblib.load(path)
    runs.append({"Model": name, **evaluate_classifier(model, X_test, y_test)})
set_publication_style()
fig = plot_roc_curves(runs)
axis = fig.axes[0]
for line, run in zip(axis.lines[:4], runs):
    row = ci[(ci.Model == run["Model"]) & (ci.PredictorSet == "Full predictors") & (ci.OperatingPoint == "Estimator default") & (ci.Metric == "ROC-AUC")].iloc[0]
    line.set_label(f"{run['Model']}: {row.Estimate:.3f} [{row.Lower95:.3f}, {row.Upper95:.3f}]")
axis.legend(title="ROC-AUC [95% test bootstrap CI]", fontsize=10, title_fontsize=11, loc="lower right")
fig.tight_layout()
save_figure(fig, "classification_roc_comparison.png")
save_figure(plot_calibration_curves(runs), "classification_calibration_curves.png")
fig, axis = plt.subplots(figsize=(8, 6))
for run in runs:
    precision, recall, _ = precision_recall_curve(y_test, run["Probabilities"])
    ap = average_precision_score(y_test, run["Probabilities"])
    axis.step(recall, precision, where="post", label=f"{run['Model']} (AP={ap:.3f})")
axis.axhline(y_test.mean(), color="grey", linestyle="--", label=f"Test prevalence={y_test.mean():.3f}")
axis.set(xlabel="Sensitivity (recall)", ylabel="Positive predictive value (precision)",
    title="Held-out Precision-Recall Curves (n=78; positive=12)", xlim=(0, 1), ylim=(0, 1.03))
axis.legend(fontsize=10)
fig.tight_layout()
save_figure(fig, "classification_pr_comparison.png")
save_figure(plot_class_distribution(bundle.clean), "class_distribution.png")
save_figure(plot_threshold_sweep(pd.read_csv(tables / "classification_threshold_sweep.csv"), .45, sample_count=len(y_train)), "classification_threshold_sweep.png")
print("ROC, PR, calibration, class counts, and training-only threshold figures refreshed.")
