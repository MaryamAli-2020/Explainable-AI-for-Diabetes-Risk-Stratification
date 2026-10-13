"""Nested evaluation of training-only family, hyperparameter, and threshold selection."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, ParameterGrid

from .config import RANDOM_STATE, TABLES_DIR, PROJECT_ROOT
from .data import prepare_dataset, load_diabetes_dataset
from .preprocess import split_classification_data
from .models import (get_classifier_specs, fit_classifier_search,
    generate_nested_oof_probabilities, get_positive_class_scores,
    threshold_sweep_from_probabilities, recommend_screening_threshold)
from .reviewer_revision import binary_metrics, serializable


def evaluate_nested_pipeline():
    """Outer validation is never passed to any selection or threshold operation."""
    bundle = prepare_dataset(load_diabetes_dataset())
    train, test, target, test_target = split_classification_data(bundle.clean)
    specs = get_classifier_specs()
    specs = {name: replace(spec, estimator=clone(spec.estimator).set_params(n_jobs=1))
        if spec.tree_based and "n_jobs" in spec.estimator.get_params() else spec
        for name, spec in specs.items()}
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, candidate_rows, predictions, selections, split_audits = [], [], [], [], []
    for fold, (fitting, validating) in enumerate(outer.split(train, target), start=1):
        X_fit, y_fit = train.iloc[fitting], target.iloc[fitting]
        X_valid, y_valid = train.iloc[validating], target.iloc[validating]
        assert set(X_fit.index).isdisjoint(X_valid.index)
        assert set(test.index).isdisjoint(X_fit.index) and set(test.index).isdisjoint(X_valid.index)
        candidates = []
        for name, spec in specs.items():
            print(f"Outer fold {fold}/5: inner tuning {name}", flush=True)
            search = fit_classifier_search(name, spec, X_fit, y_fit, list(train.columns))
            candidates.append((name, spec, search))
            candidate_rows.append({"OuterFold": fold, "Model": name,
                "InnerBestAUC": float(search.best_score_), "BestParameters": json.dumps(serializable(search.best_params_))})
        name, spec, search = max(candidates, key=lambda c: c[2].best_score_)
        print(f"Outer fold {fold}/5: selected {name}; training-only threshold cross-fitting", flush=True)
        # A separate threshold cross-fit is confined to this outer-training partition.
        # Within each threshold fold, hyperparameters are retuned without its validation rows.
        oof = generate_nested_oof_probabilities(name, spec, X_fit, y_fit, list(train.columns))
        sweep = threshold_sweep_from_probabilities(oof, y_fit)
        chosen = recommend_screening_threshold(sweep)
        threshold = float(chosen.Threshold)
        probability = get_positive_class_scores(search.best_estimator_, X_valid)
        predicted = (probability >= threshold).astype(int)
        metrics = binary_metrics(y_valid, probability, predicted)
        rows.append({"OuterFold": fold, "SelectedModel": name, "Threshold": threshold,
            "FitN": len(fitting), "ValidationN": len(validating), "ValidationPositive": int(y_valid.sum()),
            "InnerBestAUC": float(search.best_score_), **metrics})
        predictions.extend({"PatientIndex": int(index), "OuterFold": fold, "Outcome": int(label),
            "Probability": float(score), "Prediction": int(prediction), "Threshold": threshold,
            "SelectedModel": name} for index, label, score, prediction in
            zip(X_valid.index, y_valid, probability, predicted, strict=True))
        selections.append({"fold": fold, "family": name, "parameters": serializable(search.best_params_),
            "threshold": threshold, "training_oof_recall": float(chosen.Recall),
            "training_oof_precision": float(chosen.Precision)})
        split_audits.append({"fold": fold, "outer_training_patient_indices": X_fit.index.tolist(),
            "outer_validation_patient_indices": X_valid.index.tolist()})
        pd.DataFrame({"PatientIndex": X_fit.index, "Outcome": y_fit.to_numpy(), "ThresholdOOFProbability": oof}).to_csv(TABLES_DIR / f"nested_threshold_oof_fold_{fold}.csv", index=False)
        sweep.to_csv(TABLES_DIR / f"nested_threshold_sweep_fold_{fold}.csv", index=False)
        pd.DataFrame(rows).to_csv(TABLES_DIR / "nested_pipeline_outer_folds.csv", index=False)
        print(f"Outer fold {fold}: AUC {metrics['ROC-AUC']:.4f}; AP {metrics['PR-AUC (AP)']:.4f}; threshold {threshold:.2f}", flush=True)
    folds = pd.DataFrame(rows)
    prediction_frame = pd.DataFrame(predictions)
    assert prediction_frame.PatientIndex.is_unique and set(prediction_frame.PatientIndex) == set(train.index)
    prediction_frame.to_csv(TABLES_DIR / "nested_pipeline_outer_predictions.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(TABLES_DIR / "nested_pipeline_inner_searches.csv", index=False)
    selected = folds.SelectedModel.value_counts()
    frequency = pd.DataFrame([{"Model": name, "SelectedFolds": int(selected.get(name, 0)), "TotalFolds": 5} for name in specs])
    frequency.to_csv(TABLES_DIR / "nested_pipeline_selection_frequency.csv", index=False)
    summaries = [{"Metric": metric, "Mean": float(folds[metric].mean()),
        "SampleSD": float(folds[metric].std(ddof=1)), "OuterFolds": 5}
        for metric in ("ROC-AUC", "PR-AUC (AP)", "Sensitivity", "PPV", "Specificity", "NPV", "F1", "Brier")]
    pd.DataFrame(summaries).to_csv(TABLES_DIR / "nested_pipeline_summary.csv", index=False)
    pooled = binary_metrics(prediction_frame.Outcome, prediction_frame.Probability, prediction_frame.Prediction)
    audit = {"procedure": "Complete model-family, hyperparameter, and threshold selection inside each outer-training partition",
        "seed": RANDOM_STATE, "outer_folds": 5, "inner_folds": 5, "inner_repeats": 3,
        "threshold_crossfit_folds": 5, "threshold_retuning": "Five-fold CV repeated three times within each threshold-training subfold, selected family fixed",
        "candidate_grid_sizes": {name: len(ParameterGrid(spec.param_grid)) for name, spec in specs.items()},
        "family_search_cv_fits": 5 * 15 * sum(len(ParameterGrid(spec.param_grid)) for spec in specs.values()),
        "training_n": len(train), "training_positive": int(target.sum()),
        "untouched_test_n": len(test), "untouched_test_indices": test.index.tolist(),
        "folds": split_audits, "selections": selections, "pooled_oof_metrics": pooled,
        "interpretation": "Means and sample SD describe five dependent outer-fold estimates, not confidence intervals; pooled metrics are secondary and not substituted for fold means",
        "primary_test_results": "Original fitted classifier, test predictions, thresholds and manuscript held-out metrics preserved"}
    (TABLES_DIR / "nested_pipeline_audit.json").write_text(json.dumps(serializable(audit), indent=2), encoding="utf-8")
    return folds
