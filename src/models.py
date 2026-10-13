from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from .config import CLASSIFICATION_PRIMARY_METRIC, CV_REPEATS, CV_SPLITS, RANDOM_STATE, REGRESSION_PRIMARY_METRIC
from .preprocess import build_classification_pipeline, build_regression_pipeline


@dataclass(frozen=True)
class ModelSpec:
    estimator: object
    param_grid: list[dict[str, object]]
    scale: bool
    family: str
    tree_based: bool


def get_repeated_stratified_cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )


def get_repeated_kfold() -> RepeatedKFold:
    return RepeatedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )


def get_positive_class_scores(model: object, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(features)
        minimum = decision_scores.min()
        maximum = decision_scores.max()
        if maximum == minimum:
            return np.zeros_like(decision_scores, dtype=float)
        return (decision_scores - minimum) / (maximum - minimum)
    raise AttributeError("The supplied model does not expose predict_proba or decision_function.")


def get_classifier_specs() -> dict[str, ModelSpec]:
    specs = {
        "Logistic Regression": ModelSpec(
            estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            param_grid=[
                {
                    "model__C": [0.1, 1.0, 5.0, 10.0],
                    "model__class_weight": [None, "balanced"],
                }
            ],
            scale=True,
            family="linear",
            tree_based=False,
        ),
        "Random Forest": ModelSpec(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid=[
                {
                    "model__n_estimators": [300, 500],
                    "model__max_depth": [None, 4, 6],
                    "model__min_samples_leaf": [1, 3, 5],
                    "model__class_weight": [None, "balanced"],
                }
            ],
            scale=False,
            family="tree",
            tree_based=True,
        ),
        "Support Vector Machine": ModelSpec(
            estimator=SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            param_grid=[
                {
                    "model__C": [0.5, 1.0, 5.0],
                    "model__gamma": ["scale", 0.1, 0.01],
                    "model__class_weight": [None, "balanced"],
                }
            ],
            scale=True,
            family="kernel",
            tree_based=False,
        ),
    }

    try:
        from xgboost import XGBClassifier

        specs["XGBoost"] = ModelSpec(
            estimator=XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
            ),
            param_grid=[
                {
                    "model__n_estimators": [200, 300],
                    "model__max_depth": [3, 4],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__subsample": [0.8, 1.0],
                }
            ],
            scale=False,
            family="boosting",
            tree_based=True,
        )
    except ImportError:
        specs["Gradient Boosting"] = ModelSpec(
            estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
            param_grid=[
                {
                    "model__n_estimators": [150, 250],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__max_depth": [2, 3],
                }
            ],
            scale=False,
            family="boosting",
            tree_based=True,
        )
    return specs


def get_regressor_specs() -> dict[str, ModelSpec]:
    return {
        "Linear Regression": ModelSpec(
            estimator=LinearRegression(),
            param_grid=[{}],
            scale=True,
            family="linear",
            tree_based=False,
        ),
        "Random Forest Regressor": ModelSpec(
            estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid=[
                {
                    "model__n_estimators": [300, 500],
                    "model__max_depth": [None, 4, 6],
                    "model__min_samples_leaf": [1, 2, 4],
                }
            ],
            scale=False,
            family="tree",
            tree_based=True,
        ),
        "Gradient Boosting Regressor": ModelSpec(
            estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
            param_grid=[
                {
                    "model__n_estimators": [150, 250],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__max_depth": [2, 3],
                }
            ],
            scale=False,
            family="boosting",
            tree_based=True,
        ),
    }


def fit_classifier_search(
    name: str,
    spec: ModelSpec,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_names: Iterable[str],
    *,
    use_smote: bool = False,
) -> GridSearchCV:
    pipeline = build_classification_pipeline(
        estimator=clone(spec.estimator),
        feature_names=list(feature_names),
        scale=spec.scale,
        use_smote=use_smote,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec.param_grid,
        scoring=CLASSIFICATION_PRIMARY_METRIC,
        cv=get_repeated_stratified_cv(),
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(train_features, train_target)
    return search


def evaluate_classifier(
    fitted_model: object,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> dict[str, object]:
    probabilities = get_positive_class_scores(fitted_model, test_features)
    predictions = fitted_model.predict(test_features)
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(test_target, probabilities)
    calibration_true, calibration_pred = calibration_curve(
        test_target,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )
    confusion = confusion_matrix(test_target, predictions)
    tn, fp, fn, tp = confusion.ravel()
    brier = np.mean((probabilities - test_target.to_numpy()) ** 2)
    return {
        "Accuracy": accuracy_score(test_target, predictions),
        "Precision": precision_score(test_target, predictions, zero_division=0),
        "Recall": recall_score(test_target, predictions, zero_division=0),
        "F1Score": f1_score(test_target, predictions, zero_division=0),
        "ROCAUC": roc_auc_score(test_target, probabilities),
        "BrierScore": brier,
        "ConfusionMatrix": confusion,
        "Probabilities": probabilities,
        "Predictions": predictions,
        "RocCurve": {"fpr": roc_fpr, "tpr": roc_tpr, "thresholds": roc_thresholds},
        "CalibrationCurve": {"pred": calibration_pred, "true": calibration_true},
        "Counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }



def evaluate_classifier_at_threshold(
    fitted_model: object,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
) -> dict[str, object]:
    """Evaluate a fitted classifier at a fixed, pre-selected probability threshold."""
    probabilities = get_positive_class_scores(fitted_model, features)
    predictions = (probabilities >= threshold).astype(int)
    confusion = confusion_matrix(target, predictions)
    tn, fp, fn, tp = confusion.ravel()
    return {
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1Score": f1_score(target, predictions, zero_division=0),
        "ROCAUC": roc_auc_score(target, probabilities),
        "BrierScore": float(np.mean((probabilities - target.to_numpy()) ** 2)),
        "ConfusionMatrix": confusion,
        "Probabilities": probabilities,
        "Predictions": predictions,
        "Counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def generate_nested_oof_probabilities(
    name: str,
    spec: ModelSpec,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_names: list[str],
) -> np.ndarray:
    """Generate training-only out-of-fold probabilities for threshold selection.

    Each outer validation fold is predicted by a model whose hyperparameters were
    tuned only on the corresponding outer-training fold. The held-out test set is
    never used here.
    """
    outer_cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_probabilities = np.full(len(train_target), np.nan, dtype=float)

    for outer_train_idx, outer_valid_idx in outer_cv.split(train_features, train_target):
        inner_search = fit_classifier_search(
            name,
            spec,
            train_features.iloc[outer_train_idx],
            train_target.iloc[outer_train_idx],
            feature_names,
        )
        oof_probabilities[outer_valid_idx] = get_positive_class_scores(
            inner_search.best_estimator_,
            train_features.iloc[outer_valid_idx],
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("Failed to generate complete out-of-fold probabilities.")
    return oof_probabilities


def threshold_sweep_from_probabilities(
    probabilities: np.ndarray,
    target: pd.Series,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate candidate thresholds from precomputed probabilities."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    rows = []
    target_array = target.to_numpy()
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(target_array, predictions).ravel()
        precision = precision_score(target_array, predictions, zero_division=0)
        recall = recall_score(target_array, predictions, zero_division=0)
        rows.append(
            {
                "Threshold": round(float(threshold), 2),
                "Precision": precision,
                "Recall": recall,
                "F1Score": f1_score(target_array, predictions, zero_division=0),
                "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "FalseNegatives": int(fn),
                "FalsePositives": int(fp),
            }
        )
    return pd.DataFrame(rows)


def run_classification_experiment(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
) -> dict[str, object]:
    feature_names = list(train_features.columns)
    runs = []
    searches: dict[str, GridSearchCV] = {}
    specs = get_classifier_specs()

    for name, spec in specs.items():
        search = fit_classifier_search(name, spec, train_features, train_target, feature_names)
        searches[name] = search
        evaluation = evaluate_classifier(search.best_estimator_, test_features, test_target)
        runs.append(
            {
                "Model": name,
                "BestParams": search.best_params_,
                "CVBestScore": search.best_score_,
                "BestEstimator": search.best_estimator_,
                "Spec": spec,
                **evaluation,
            }
        )

    baseline_table = (
        pd.DataFrame(
            [
                {
                    "Model": run["Model"],
                    "CV ROC-AUC": round(float(run["CVBestScore"]), 4),
                    "Test Accuracy": round(float(run["Accuracy"]), 4),
                    "Test Precision": round(float(run["Precision"]), 4),
                    "Test Recall": round(float(run["Recall"]), 4),
                    "Test F1": round(float(run["F1Score"]), 4),
                    "Test ROC-AUC": round(float(run["ROCAUC"]), 4),
                    "Brier Score": round(float(run["BrierScore"]), 4),
                    "False Negatives": run["Counts"]["fn"],
                    "False Positives": run["Counts"]["fp"],
                }
                for run in runs
            ]
        )
        .sort_values("CV ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )

    best_run = max(runs, key=lambda entry: entry["CVBestScore"])
    best_smote_run = None
    best_smote_search = fit_classifier_search(
        best_run["Model"],
        best_run["Spec"],
        train_features,
        train_target,
        feature_names,
        use_smote=True,
    )
    best_smote_eval = evaluate_classifier(best_smote_search.best_estimator_, test_features, test_target)
    best_smote_run = {
        "Model": f"{best_run['Model']} + SMOTE",
        "BestParams": best_smote_search.best_params_,
        "CVBestScore": best_smote_search.best_score_,
        "BestEstimator": best_smote_search.best_estimator_,
        "Spec": best_run["Spec"],
        **best_smote_eval,
    }

    # Select the screening threshold from TRAINING-ONLY nested out-of-fold predictions.
    oof_probabilities = generate_nested_oof_probabilities(
        best_run["Model"],
        best_run["Spec"],
        train_features,
        train_target,
        feature_names,
    )
    threshold_table = threshold_sweep_from_probabilities(oof_probabilities, train_target)
    recommended_threshold = recommend_screening_threshold(threshold_table)
    selected_threshold = float(recommended_threshold["Threshold"])

    # Only after the threshold is locked do we evaluate that operating point on the held-out test set.
    threshold_test_evaluation = evaluate_classifier_at_threshold(
        best_run["BestEstimator"],
        test_features,
        test_target,
        selected_threshold,
    )

    # Use training data for any permutation-based feature-importance fallback so that
    # test data do not influence model interpretation choices.
    feature_importance = extract_feature_importance(
        best_run["BestEstimator"],
        feature_names,
        train_features,
        train_target,
    )
    tree_candidates = [run for run in runs if run["Spec"].tree_based]
    best_tree_run = max(tree_candidates, key=lambda entry: entry["CVBestScore"])

    stats_result = paired_model_comparison(runs, train_features, train_target)

    return {
        "runs": runs,
        "baseline_table": baseline_table,
        "best_run": best_run,
        "best_tree_run": best_tree_run,
        "best_smote_run": best_smote_run,
        "threshold_table": threshold_table,
        "recommended_threshold": recommended_threshold,
        "threshold_test_evaluation": threshold_test_evaluation,
        "oof_probabilities": oof_probabilities,
        "feature_importance": feature_importance,
        "searches": searches,
        "paired_test": stats_result,
    }


def threshold_sweep(
    fitted_model: object,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    probabilities = get_positive_class_scores(fitted_model, test_features)
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_target, predictions).ravel()
        precision = precision_score(test_target, predictions, zero_division=0)
        recall = recall_score(test_target, predictions, zero_division=0)
        rows.append(
            {
                "Threshold": round(float(threshold), 2),
                "Precision": precision,
                "Recall": recall,
                "F1Score": f1_score(test_target, predictions, zero_division=0),
                "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "FalseNegatives": int(fn),
                "FalsePositives": int(fp),
            }
        )
    return pd.DataFrame(rows)


def recommend_screening_threshold(threshold_table: pd.DataFrame, minimum_recall: float = 0.80) -> pd.Series:
    eligible = threshold_table.loc[threshold_table["Recall"] >= minimum_recall]
    if not eligible.empty:
        return eligible.sort_values(["F1Score", "Precision"], ascending=[False, False]).iloc[0]
    return threshold_table.sort_values(["F1Score", "Recall"], ascending=[False, False]).iloc[0]


def get_transformed_feature_names(
    fitted_model: object,
    original_feature_names: list[str],
) -> list[str]:
    preprocess = fitted_model.named_steps.get("preprocess")
    if preprocess is None:
        return original_feature_names

    try:
        transformed = preprocess.get_feature_names_out()
        return [str(name) for name in transformed]
    except Exception:
        return original_feature_names


def map_transformed_feature_to_original(
    transformed_feature_name: str,
    original_feature_names: list[str],
) -> str:
    if transformed_feature_name in original_feature_names:
        return transformed_feature_name

    for original_feature in original_feature_names:
        if transformed_feature_name.startswith(f"{original_feature}_"):
            return original_feature
    return transformed_feature_name


def aggregate_feature_importance(
    feature_names: list[str],
    importance_values: np.ndarray,
    importance_type: str,
    original_feature_names: list[str],
) -> pd.DataFrame:
    importance_frame = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": np.asarray(importance_values, dtype=float),
            "ImportanceType": importance_type,
        }
    )
    importance_frame["OriginalFeature"] = importance_frame["Feature"].apply(
        lambda name: map_transformed_feature_to_original(name, original_feature_names)
    )
    grouped = (
        importance_frame.groupby("OriginalFeature", as_index=False)["Importance"]
        .sum()
        .rename(columns={"OriginalFeature": "Feature"})
    )
    grouped["ImportanceType"] = importance_type
    return grouped.sort_values("Importance", ascending=False).reset_index(drop=True)


def extract_feature_importance(
    fitted_model: object,
    feature_names: list[str],
    reference_features: pd.DataFrame,
    reference_target: pd.Series,
) -> pd.DataFrame:
    model = fitted_model.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(model.feature_importances_)
        transformed_feature_names = get_transformed_feature_names(fitted_model, feature_names)
        importance_type = "model_native"
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        importance_values = np.abs(coefficients[0] if coefficients.ndim > 1 else coefficients)
        transformed_feature_names = get_transformed_feature_names(fitted_model, feature_names)
        importance_type = "absolute_coefficient"
    else:
        permutation = permutation_importance(
            fitted_model,
            reference_features,
            reference_target,
            scoring="roc_auc",
            n_repeats=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        importance_values = permutation.importances_mean
        transformed_feature_names = feature_names
        importance_type = "permutation"

    return aggregate_feature_importance(
        feature_names=transformed_feature_names,
        importance_values=importance_values,
        importance_type=importance_type,
        original_feature_names=feature_names,
    )


def paired_model_comparison(
    runs: list[dict[str, object]],
    train_features: pd.DataFrame,
    train_target: pd.Series,
) -> dict[str, object]:
    baseline_runs = sorted(runs, key=lambda entry: entry["CVBestScore"], reverse=True)[:2]
    if len(baseline_runs) < 2:
        return {"message": "Not enough models were available for paired comparison."}

    cv = get_repeated_stratified_cv()
    scores = {
        run["Model"]: cross_val_score(
            run["BestEstimator"],
            train_features,
            train_target,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
        for run in baseline_runs
    }
    model_names = list(scores.keys())
    statistic, p_value = wilcoxon(scores[model_names[0]], scores[model_names[1]])
    return {
        "model_a": model_names[0],
        "model_b": model_names[1],
        "scores_a": scores[model_names[0]],
        "scores_b": scores[model_names[1]],
        "median_a": float(np.median(scores[model_names[0]])),
        "median_b": float(np.median(scores[model_names[1]])),
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
    }


def fit_regressor_search(
    spec: ModelSpec,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_names: Iterable[str],
) -> GridSearchCV:
    pipeline = build_regression_pipeline(
        estimator=clone(spec.estimator),
        feature_names=list(feature_names),
        scale=spec.scale,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec.param_grid,
        scoring=REGRESSION_PRIMARY_METRIC,
        cv=get_repeated_kfold(),
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(train_features, train_target)
    return search


def evaluate_regressor(
    fitted_model: object,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> dict[str, object]:
    predictions = fitted_model.predict(test_features)
    mse = mean_squared_error(test_target, predictions)
    rmse = float(np.sqrt(mse))
    residuals = test_target - predictions
    return {
        "MAE": mean_absolute_error(test_target, predictions),
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2_score(test_target, predictions),
        "Predictions": predictions,
        "Residuals": residuals,
    }


def run_regression_experiment(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
) -> dict[str, object]:
    feature_names = list(train_features.columns)
    runs = []
    searches = {}
    for name, spec in get_regressor_specs().items():
        search = fit_regressor_search(spec, train_features, train_target, feature_names)
        searches[name] = search
        evaluation = evaluate_regressor(search.best_estimator_, test_features, test_target)
        runs.append(
            {
                "Model": name,
                "BestParams": search.best_params_,
                "CVNegRMSE": search.best_score_,
                "BestEstimator": search.best_estimator_,
                **evaluation,
            }
        )

    comparison_table = (
        pd.DataFrame(
            [
                {
                    "Model": run["Model"],
                    "CV Neg-RMSE": round(float(run["CVNegRMSE"]), 4),
                    "Test MAE": round(float(run["MAE"]), 4),
                    "Test MSE": round(float(run["MSE"]), 4),
                    "Test RMSE": round(float(run["RMSE"]), 4),
                    "Test R2": round(float(run["R2"]), 4),
                }
                for run in runs
            ]
        )
        .sort_values("CV Neg-RMSE", ascending=False)
        .reset_index(drop=True)
    )
    best_run = max(runs, key=lambda entry: entry["CVNegRMSE"])
    return {
        "runs": runs,
        "comparison_table": comparison_table,
        "best_run": best_run,
        "searches": searches,
    }
