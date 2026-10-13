"""Independently check nested splits, selections, thresholds, and outer metrics."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, recall_score, precision_score
from src.data import prepare_dataset, load_diabetes_dataset
from src.preprocess import split_classification_data
from src.models import recommend_screening_threshold, threshold_sweep_from_probabilities

root = Path(__file__).resolve().parent
tables = root / "outputs/tables"
audit = json.loads((tables / "nested_pipeline_audit.json").read_text())
train, test, y_train, y_test = split_classification_data(prepare_dataset(load_diabetes_dataset()).clean)
folds = pd.read_csv(tables / "nested_pipeline_outer_folds.csv")
scores = pd.read_csv(tables / "nested_pipeline_inner_searches.csv")
predictions = pd.read_csv(tables / "nested_pipeline_outer_predictions.csv")
assert len(folds) == 5 and predictions.PatientIndex.is_unique and len(predictions) == 312
assert set(predictions.PatientIndex) == set(train.index)
assert set(predictions.PatientIndex).isdisjoint(test.index)
for partition in audit["folds"]:
    fold = partition["fold"]
    fitting, validating = map(set, (partition["outer_training_patient_indices"], partition["outer_validation_patient_indices"]))
    assert fitting.isdisjoint(validating) and fitting | validating == set(train.index)
    assert (fitting | validating).isdisjoint(test.index)
    row = folds.loc[folds.OuterFold == fold].iloc[0]
    candidates = scores[scores.OuterFold == fold]
    assert len(candidates) == 4 and row.SelectedModel == candidates.loc[candidates.InnerBestAUC.idxmax()].Model
    oof = pd.read_csv(tables / f"nested_threshold_oof_fold_{fold}.csv")
    assert oof.PatientIndex.is_unique and set(oof.PatientIndex) == fitting
    np.testing.assert_array_equal(oof.Outcome, y_train.loc[oof.PatientIndex].to_numpy())
    chosen = recommend_screening_threshold(threshold_sweep_from_probabilities(oof.ThresholdOOFProbability.to_numpy(), oof.Outcome))
    assert np.isclose(chosen.Threshold, row.Threshold)
    p = predictions[predictions.OuterFold == fold]
    assert set(p.PatientIndex) == validating
    np.testing.assert_array_equal(p.Prediction, (p.Probability >= row.Threshold).astype(int))
    np.testing.assert_array_equal(p.Outcome, y_train.loc[p.PatientIndex].to_numpy())
    np.testing.assert_allclose(roc_auc_score(p.Outcome, p.Probability), row["ROC-AUC"])
    np.testing.assert_allclose(average_precision_score(p.Outcome, p.Probability), row["PR-AUC (AP)"])
    np.testing.assert_allclose(recall_score(p.Outcome, p.Prediction), row.Sensitivity)
    np.testing.assert_allclose(precision_score(p.Outcome, p.Prediction, zero_division=0), row.PPV)
    tn, fp, fn, tp = confusion_matrix(p.Outcome, p.Prediction, labels=[0,1]).ravel()
    np.testing.assert_array_equal([tn,fp,fn,tp], [row.TN,row.FP,row.FN,row.TP])
summary = pd.read_csv(tables / "nested_pipeline_summary.csv")
for _, row in summary.iterrows():
    np.testing.assert_allclose(row.Mean, folds[row.Metric].mean())
    np.testing.assert_allclose(row.SampleSD, folds[row.Metric].std(ddof=1))
preserved = json.loads((tables / "nested_test_preservation.json").read_text())
assert preserved["passed"] and preserved["sha256_before"] == preserved["sha256_after"]
for path, digest in preserved["sha256_before"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
result = {"passed": True, "outer_folds": 5, "outer_validation_patients": 312,
    "heldout_test_excluded": True, "thresholds_reproduced_from_each_outer_training_partition": True,
    "candidate_selection_reproduced": True, "outer_metrics_independently_recomputed": True,
    "original_test_artifacts_unchanged": True}
(tables / "nested_pipeline_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
