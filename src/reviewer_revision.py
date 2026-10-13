"""Reproducible reviewer analyses with fixed test data and training-only selection."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TABLES_DIR, MODELS_DIR, TEST_SIZE
from .data import load_diabetes_dataset, prepare_dataset, get_regression_frame
from .preprocess import split_classification_data
from .models import (get_classifier_specs, get_regressor_specs, fit_classifier_search,
    fit_regressor_search, evaluate_classifier, get_positive_class_scores,
    generate_nested_oof_probabilities, threshold_sweep_from_probabilities,
    recommend_screening_threshold)


def binary_metrics(y, probability, prediction):
    y, probability, prediction = map(np.asarray, (y, probability, prediction))
    tp = int(np.sum((y == 1) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    divide = lambda a, b: a / b if b else np.nan
    return {"Accuracy": (tp + tn) / len(y), "PPV": divide(tp, tp + fp),
        "Sensitivity": divide(tp, tp + fn), "Specificity": divide(tn, tn + fp),
        "NPV": divide(tn, tn + fn), "F1": divide(2 * tp, 2 * tp + fp + fn),
        "ROC-AUC": roc_auc_score(y, probability),
        "PR-AUC (AP)": average_precision_score(y, probability),
        "Brier": float(np.mean((probability - y) ** 2)),
        "FN": fn, "FP": fp, "TN": tn, "TP": tp}


def bootstrap_classification(y, probability, prediction, *, repeats=2000):
    """Percentile intervals conditional on fitted models and observed class counts."""
    y, probability, prediction = map(np.asarray, (y, probability, prediction))
    rng = np.random.default_rng(RANDOM_STATE)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    point = binary_metrics(y, probability, prediction)
    draws = {metric: [] for metric in point}
    for _ in range(repeats):
        index = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        values = binary_metrics(y[index], probability[index], prediction[index])
        for metric, value in values.items():
            draws[metric].append(value)
    return [{"Metric": metric, "Estimate": value,
        "Lower95": float(np.nanpercentile(draws[metric], 2.5)),
        "Upper95": float(np.nanpercentile(draws[metric], 97.5)),
        "BootstrapReplicates": repeats} for metric, value in point.items()]


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_reviewer_revision():
    bundle = prepare_dataset(load_diabetes_dataset())
    X_train, X_test, y_train, y_test = split_classification_data(bundle.clean)
    specs = get_classifier_specs()
    # Parallelize CV folds, not trees within every parallel worker.
    specs = {name: replace(spec, estimator=clone(spec.estimator).set_params(n_jobs=1))
        if spec.tree_based and "n_jobs" in spec.estimator.get_params() else spec for name, spec in specs.items()}
    audit = {"seed": RANDOM_STATE, "bootstrap_replicates": 2000,
        "raw_records": bundle.metadata, "training_n": len(y_train), "test_n": len(y_test),
        "test_positive": int(y_test.sum()), "test_negative": int((y_test == 0).sum()),
        "predictors": list(X_train.columns), "search_spaces": {n: s.param_grid for n, s in specs.items()},
        "outcome_definition": "Publisher label retained; original adjudication rule and measurement timing not verified"}
    all_rows, ci_rows, ablation_rows, runs_by_set = [], [], [], {}
    for setting in ("Full predictors", "Without Glucose"):
        train = X_train if setting == "Full predictors" else X_train.drop(columns="Glucose")
        test = X_test.loc[:, train.columns]
        runs = []
        for name, spec in specs.items():
            print(f"{setting}: tuning {name}", flush=True)
            search = fit_classifier_search(name, spec, train, y_train, list(train.columns))
            evaluation = evaluate_classifier(search.best_estimator_, test, y_test)
            run = {"Model": name, "CV": float(search.best_score_), "Estimator": search.best_estimator_,
                "Spec": spec, "Params": search.best_params_, **evaluation}
            runs.append(run)
            metrics = binary_metrics(y_test, run["Probabilities"], run["Predictions"])
            all_rows.append({"PredictorSet": setting, "Model": name, "CV ROC-AUC": run["CV"], **metrics})
            ci_rows.extend({"PredictorSet": setting, "Model": name, "OperatingPoint": "Estimator default",
                **row} for row in bootstrap_classification(y_test, run["Probabilities"], run["Predictions"]))
            joblib.dump(search.best_estimator_, MODELS_DIR / ("revision_" + setting.lower().replace(" ", "_") + "_" + name.lower().replace(" ", "_") + ".joblib"))
        best = max(runs, key=lambda r: r["CV"])
        if setting == "Full predictors":
            assert best["Model"] == "Random Forest", "Selected model differs from prior corrected run"
            saved = joblib.load(MODELS_DIR / "best_classifier.joblib")
            assert np.allclose(saved.predict_proba(X_test)[:, 1], best["Probabilities"]), "Refit differs from verified saved classifier"
            training_oof = pd.read_csv(TABLES_DIR / "training_oof_predictions.csv")
            assert training_oof.PatientIndex.tolist() == y_train.index.tolist()
            assert training_oof.Outcome.tolist() == y_train.tolist()
            oof = training_oof.OOFProbability.to_numpy()
        else:
            print(f"Without Glucose: nested training OOF for {best['Model']}", flush=True)
            oof = generate_nested_oof_probabilities(best["Model"], best["Spec"], train, y_train, list(train.columns))
            pd.DataFrame({"PatientIndex": y_train.index, "Outcome": y_train.to_numpy(), "OOFProbability": oof}).to_csv(TABLES_DIR / "ablation_training_oof_predictions.csv", index=False)
        threshold = float(recommend_screening_threshold(threshold_sweep_from_probabilities(oof, y_train))["Threshold"])
        predictions = (best["Probabilities"] >= threshold).astype(int)
        metrics = binary_metrics(y_test, best["Probabilities"], predictions)
        ablation_rows.append({"PredictorSet": setting, "Selected model": best["Model"], "Fixed threshold": threshold, "CV ROC-AUC": best["CV"], **metrics})
        ci_rows.extend({"PredictorSet": setting, "Model": best["Model"], "OperatingPoint": f"Training OOF threshold {threshold:.2f}", **row}
            for row in bootstrap_classification(y_test, best["Probabilities"], predictions))
        runs_by_set[setting] = runs
        audit[setting] = {"selected_model": best["Model"], "threshold": threshold,
            "best_parameters": {r["Model"]: r["Params"] for r in runs},
            "selected_test_metrics": metrics}
        pd.DataFrame({"PatientIndex": y_test.index, "Outcome": y_test.to_numpy(), "Probability": best["Probabilities"], "Prediction": predictions}).to_csv(TABLES_DIR / ("revision_test_predictions.csv" if setting == "Full predictors" else "ablation_test_predictions.csv"), index=False)
    best = max(runs_by_set["Full predictors"], key=lambda r: r["CV"])
    print("SMOTE sensitivity: training-only fold resampling", flush=True)
    smote = fit_classifier_search(best["Model"], best["Spec"], X_train, y_train, list(X_train.columns), use_smote=True)
    smote_eval = evaluate_classifier(smote.best_estimator_, X_test, y_test)
    smote_metrics = binary_metrics(y_test, smote_eval["Probabilities"], smote_eval["Predictions"])
    pd.DataFrame([{"Model": "Random Forest + SMOTE", "CV ROC-AUC": smote.best_score_, **smote_metrics}]).to_csv(TABLES_DIR / "revision_smote_results.csv", index=False)
    ci_rows.extend({"PredictorSet": "Full predictors", "Model": "Random Forest + SMOTE", "OperatingPoint": "Estimator default", **row}
        for row in bootstrap_classification(y_test, smote_eval["Probabilities"], smote_eval["Predictions"]))
    audit["SMOTE"] = {"best_parameters": smote.best_params_, "metrics": smote_metrics,
        "method": "SMOTE(k_neighbors=5, sampling_strategy=auto, random_state=42), after fold-fitted preprocessing, before classifier; test never resampled",
        "limitation": "One-hot Gender interpolation can create fractional category indicators; sensitivity analysis only"}
    pd.DataFrame(all_rows).to_csv(TABLES_DIR / "revision_classification_metrics.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(TABLES_DIR / "revision_classification_confidence_intervals.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(TABLES_DIR / "glucose_ablation_comparison.csv", index=False)
    print("Regression confidence intervals", flush=True)
    X_reg, y_reg = get_regression_frame(bundle.clean, target_column="Glucose")
    train, test, target_train, target_test = train_test_split(X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    regression_ci = []
    for name, spec in get_regressor_specs().items():
        if "n_jobs" in spec.estimator.get_params():
            spec = replace(spec, estimator=clone(spec.estimator).set_params(n_jobs=1))
        search = fit_regressor_search(spec, train, target_train, list(train.columns))
        audit.setdefault("regression_search_spaces", {})[name] = spec.param_grid
        audit.setdefault("regression_best_parameters", {})[name] = search.best_params_
        prediction = search.best_estimator_.predict(test)
        observed = target_test.to_numpy()
        def measures(index):
            mse = mean_squared_error(observed[index], prediction[index])
            return {"MAE": mean_absolute_error(observed[index], prediction[index]), "MSE": mse,
                "RMSE": np.sqrt(mse), "R2": r2_score(observed[index], prediction[index])}
        point = measures(np.arange(len(observed)))
        rng = np.random.default_rng(RANDOM_STATE)
        draws = pd.DataFrame([measures(rng.integers(0, len(observed), size=len(observed))) for _ in range(2000)])
        regression_ci.extend({"Model": name, "Metric": metric, "Estimate": value,
            "Lower95": draws[metric].quantile(.025), "Upper95": draws[metric].quantile(.975), "BootstrapReplicates": 2000}
            for metric, value in point.items())
    pd.DataFrame(regression_ci).to_csv(TABLES_DIR / "revision_regression_confidence_intervals.csv", index=False)
    audit["label_provenance"] = json.loads((TABLES_DIR / "label_provenance_audit.json").read_text()) if (TABLES_DIR / "label_provenance_audit.json").exists() else {"status": "Run verify_label_provenance.py to reproduce matching"}
    (TABLES_DIR / "reviewer_revision_audit.json").write_text(json.dumps(serializable(audit), indent=2), encoding="utf-8")
    return runs_by_set
