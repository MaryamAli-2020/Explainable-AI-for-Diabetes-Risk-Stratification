"""Run new reviewer experiments without resampling or choosing models on test data."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
from src.reviewer_revision import run_reviewer_revision
from src.plots import plot_roc_curves, plot_calibration_curves, save_figure, set_publication_style
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

if __name__ == "__main__":
    set_publication_style()
    runs = run_reviewer_revision()
    save_figure(plot_roc_curves(runs["Full predictors"]), "classification_roc_comparison.png")
    save_figure(plot_calibration_curves(runs["Full predictors"]), "classification_calibration_curves.png")
    y = pd.read_csv("outputs/tables/revision_test_predictions.csv").Outcome.to_numpy()
    fig, axis = plt.subplots(figsize=(8, 6))
    for run in runs["Full predictors"]:
        precision, recall, _ = precision_recall_curve(y, run["Probabilities"])
        ap = average_precision_score(y, run["Probabilities"])
        axis.step(recall, precision, where="post", label=f"{run['Model']} (AP={ap:.3f})")
    axis.axhline(y.mean(), color="grey", linestyle="--", label=f"Test prevalence={y.mean():.3f}")
    axis.set(xlabel="Sensitivity (recall)", ylabel="Positive predictive value (precision)",
        title=f"Held-out Precision-Recall Curves (n={len(y)}; positive={int(y.sum())})", xlim=(0, 1), ylim=(0, 1.03))
    axis.legend(fontsize=11)
    fig.tight_layout()
    save_figure(fig, "classification_pr_comparison.png")
    print("Reviewer experiments complete.", flush=True)
