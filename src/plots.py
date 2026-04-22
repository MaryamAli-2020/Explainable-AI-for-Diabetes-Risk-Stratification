from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, PAIRPLOT_SAMPLE_SIZE, PUBLICATION_STYLE


def set_publication_style() -> None:
    sns.set_theme(
        style=PUBLICATION_STYLE["style"],
        context=PUBLICATION_STYLE["context"],
        palette=PUBLICATION_STYLE["palette"],
    )
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def save_figure(fig: plt.Figure, filename: str, directory: Path = FIGURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.savefig(path, bbox_inches="tight", dpi=300)
    return path


def plot_zero_audit(audit_frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=audit_frame, x="ZeroCount", y="Feature", palette="mako", ax=ax)
    ax.set_title("Clinically Implausible Zero Values by Feature")
    ax.set_xlabel("Count of zero values")
    ax.set_ylabel("")
    return fig


def plot_class_distribution(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=frame, x="Outcome", palette="viridis", ax=ax)
    ax.set_title("Diabetes Outcome Distribution")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Patients")
    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    return fig


def plot_histograms(frame: pd.DataFrame, columns: list[str]) -> plt.Figure:
    numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(4 * n_rows, 8)))
    axes = axes.flatten()
    for axis, column in zip(axes, numeric_columns, strict=False):
        sns.histplot(data=frame, x=column, hue="Outcome", kde=True, element="step", ax=axis)
        axis.set_title(f"{column} distribution")
    for axis in axes[len(numeric_columns) :]:
        axis.axis("off")
    fig.suptitle("Feature Distributions by Diabetes Status", y=1.02)
    fig.tight_layout()
    return fig


def plot_boxplots(frame: pd.DataFrame, columns: list[str]) -> plt.Figure:
    numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(4 * n_rows, 8)))
    axes = axes.flatten()
    for axis, column in zip(axes, numeric_columns, strict=False):
        sns.boxplot(data=frame, x="Outcome", y=column, ax=axis, palette="Set2")
        axis.set_title(f"{column} by outcome")
        axis.set_xticklabels(["No Diabetes", "Diabetes"])
    for axis in axes[len(numeric_columns) :]:
        axis.axis("off")
    fig.suptitle("Boxplots Highlighting Outliers and Group Separation", y=1.02)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(correlation_frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_frame, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Clinical Features")
    return fig


def plot_scatter_panel(frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=frame, x="Glucose", y="BMI", hue="Outcome", palette="viridis", ax=axes[0])
    axes[0].set_title("Glucose vs BMI")
    sns.scatterplot(data=frame, x="Waist", y="SystolicBP", hue="Outcome", palette="viridis", ax=axes[1])
    axes[1].set_title("Waist vs Systolic BP")
    fig.tight_layout()
    return fig


def plot_pairplot(frame: pd.DataFrame, sample_size: int = PAIRPLOT_SAMPLE_SIZE):
    sample = frame.sample(min(sample_size, len(frame)), random_state=42).copy()
    pairplot_vars = ["Glucose", "BMI", "Age", "WaistHipRatio"]
    plot = sns.pairplot(
        sample,
        vars=[column for column in pairplot_vars if column in sample.columns],
        hue="Outcome",
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.7, "s": 35},
    )
    plot.figure.suptitle("Pairwise Relationships Across Key Clinical Variables", y=1.02)
    return plot


def plot_roc_curves(runs: list[dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    for run in runs:
        roc_data = run["RocCurve"]
        ax.plot(roc_data["fpr"], roc_data["tpr"], label=f"{run['Model']} (AUC={run['ROCAUC']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return fig


def plot_calibration_curves(runs: list[dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    for run in runs:
        calibration_data = run["CalibrationCurve"]
        ax.plot(
            calibration_data["pred"],
            calibration_data["true"],
            marker="o",
            label=f"{run['Model']} (Brier={run['BrierScore']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.set_title("Calibration Curves")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.legend(loc="upper left")
    return fig


def plot_confusion_matrices(runs: list[dict[str, object]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for axis, run in zip(axes, runs[:4], strict=False):
        matrix = pd.DataFrame(
            run["ConfusionMatrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(run["Model"])
    for axis in axes[len(runs[:4]) :]:
        axis.axis("off")
    fig.suptitle("Confusion Matrices for Classification Models", y=1.02)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_frame: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance_frame, x="Importance", y="Feature", palette="rocket", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    return fig


def plot_threshold_sweep(threshold_frame: pd.DataFrame, recommended_threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_frame["Threshold"], threshold_frame["Recall"], label="Recall", marker="o")
    ax.plot(threshold_frame["Threshold"], threshold_frame["Precision"], label="Precision", marker="o")
    ax.plot(threshold_frame["Threshold"], threshold_frame["F1Score"], label="F1 score", marker="o")
    ax.axvline(recommended_threshold, linestyle="--", color="black", label=f"Recommended={recommended_threshold:.2f}")
    ax.set_title("Threshold Tuning for Screening-Oriented Deployment")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Metric value")
    ax.legend(loc="best")
    return fig


def plot_decision_boundary(
    fitted_model: object,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    x_column: str = "Glucose",
    y_column: str = "BMI",
) -> plt.Figure:
    frame = pd.concat([features[[x_column, y_column]], target.rename("Outcome")], axis=1).dropna()
    x_min, x_max = frame[x_column].min() - 5, frame[x_column].max() + 5
    y_min, y_max = frame[y_column].min() - 3, frame[y_column].max() + 3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = pd.DataFrame({x_column: xx.ravel(), y_column: yy.ravel()})
    if hasattr(fitted_model, "predict_proba"):
        zz = fitted_model.predict_proba(grid)[:, 1]
    else:
        zz = fitted_model.decision_function(grid)
    zz = zz.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, zz, levels=20, cmap="RdYlBu_r", alpha=0.35)
    sns.scatterplot(
        data=frame,
        x=x_column,
        y=y_column,
        hue="Outcome",
        palette="viridis",
        edgecolor="black",
        ax=ax,
    )
    ax.set_title(f"Illustrative Decision Boundary Using {x_column} and {y_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    fig.colorbar(contour, ax=ax, label="Predicted diabetes score")
    return fig


def plot_regression_diagnostics(actual: pd.Series, predictions, residuals: pd.Series) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(actual, predictions, alpha=0.7)
    min_value = min(actual.min(), predictions.min())
    max_value = max(actual.max(), predictions.max())
    axes[0].plot([min_value, max_value], [min_value, max_value], linestyle="--", color="grey")
    axes[0].set_title("Actual vs Predicted Glucose")
    axes[0].set_xlabel("Actual glucose")
    axes[0].set_ylabel("Predicted glucose")

    axes[1].scatter(predictions, residuals, alpha=0.7)
    axes[1].axhline(0, linestyle="--", color="grey")
    axes[1].set_title("Residual Plot")
    axes[1].set_xlabel("Predicted glucose")
    axes[1].set_ylabel("Residual")
    fig.tight_layout()
    return fig


def plot_clustering_diagnostics(evaluation_frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=evaluation_frame, x="Clusters", y="Inertia", marker="o", ax=axes[0])
    axes[0].set_title("Elbow Method")
    sns.lineplot(data=evaluation_frame, x="Clusters", y="Silhouette", marker="o", ax=axes[1])
    axes[1].set_title("Silhouette Score by k")
    fig.tight_layout()
    return fig


def plot_pca_clusters(clustered_frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=clustered_frame,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        style="Outcome",
        palette="tab10",
        ax=ax,
    )
    ax.set_title("PCA Projection of Patient Clusters")
    return fig
