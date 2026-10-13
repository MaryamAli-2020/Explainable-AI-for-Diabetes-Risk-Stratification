"""Independently verify exported training selection and held-out operating-point metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.models import recommend_screening_threshold, threshold_sweep_from_probabilities

root = Path(__file__).resolve().parent
tables = root / "outputs/tables"
oof = pd.read_csv(tables / "training_oof_predictions.csv")
test = pd.read_csv(tables / "threshold_test_predictions.csv")
reported = pd.read_csv(tables / "recommended_threshold.csv").iloc[0]
selection = recommend_screening_threshold(
    threshold_sweep_from_probabilities(oof.OOFProbability.to_numpy(), oof.Outcome)
)
assert set(oof.PatientIndex).isdisjoint(test.PatientIndex), "Training/test patient overlap"
assert selection.Threshold == reported.Threshold, "Threshold does not match training-only selection"
prediction = (test.Probability.to_numpy() >= selection.Threshold).astype(int)
assert np.array_equal(prediction, test.Prediction), "Exported predictions differ from locked threshold"
tn, fp, fn, tp = confusion_matrix(test.Outcome, prediction, labels=[0, 1]).ravel()
precision = precision_score(test.Outcome, prediction, zero_division=0)
recall = recall_score(test.Outcome, prediction, zero_division=0)
assert np.isclose(precision, reported.Precision)
assert np.isclose(recall, reported.Recall)
assert int(fn) == reported.FalseNegatives and int(fp) == reported.FalsePositives
result = {
    "procedure": "Training-only nested out-of-fold threshold selection; fixed held-out evaluation",
    "training_patients": len(oof),
    "test_patients": len(test),
    "selected_model": pd.read_csv(tables / "classification_model_comparison.csv").iloc[0].Model,
    "threshold": float(selection.Threshold),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_counts": dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
    "old_test_selected_result": dict(threshold=0.20, precision=0.7058823529411765, recall=1.0, fn=0, fp=5),
    "validation": "Passed: train/test indices disjoint; threshold reproduced from training OOF; test metrics independently recomputed",
}
(tables / "threshold_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
