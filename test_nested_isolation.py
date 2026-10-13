"""Information-flow regression test for the new nested experiment."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src import models
from src import nested_pipeline as nested

def main():
    rng = np.random.default_rng(42)
    target = pd.Series([0, 1] * 25)
    train = pd.DataFrame({"Glucose": 80 + 30 * target + rng.normal(size=50), "Age": rng.uniform(30, 70, 50)})
    test = pd.DataFrame({"Glucose": [90, 110] * 5, "Age": [40, 50] * 5}, index=range(100, 110))
    test_labels = pd.Series([0, 1] * 5, index=test.index)
    spec = models.ModelSpec(LogisticRegression(max_iter=1000, random_state=42), [{"model__C": [1.0]}], True, "linear", False)
    original_fit = models.fit_classifier_search
    active_partition = set()
    calls = []
    def guarded_fit(name, specification, X, y, names, **kwargs):
        nonlocal active_partition
        ids = set(X.index)
        if len(X) == 40:
            active_partition = ids
        assert ids.issubset(active_partition), "Fit used outer-validation patients"
        assert ids.isdisjoint(test.index), "Fit used held-out patients"
        calls.append(ids)
        return original_fit(name, specification, X, y, names, **kwargs)
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        with patch.object(nested, "TABLES_DIR", Path(directory)), \
             patch.object(nested, "load_diabetes_dataset", return_value=None), \
             patch.object(nested, "prepare_dataset", return_value=SimpleNamespace(clean=None)), \
             patch.object(nested, "get_classifier_specs", return_value={"Logistic Regression": spec}), \
             patch.object(nested, "fit_classifier_search", side_effect=guarded_fit), \
             patch.object(models, "fit_classifier_search", side_effect=guarded_fit), \
             patch.object(models, "CV_REPEATS", 1):
            with patch.object(nested, "split_classification_data", return_value=(train, test, target, test_labels)):
                first = nested.evaluate_nested_pipeline()
            with patch.object(nested, "split_classification_data", return_value=(train, test, target, 1-test_labels)):
                second = nested.evaluate_nested_pipeline()
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 60
    print("PASS: outer-validation and test patients excluded from all search fits; changing test labels leaves nested results unchanged.")

if __name__ == "__main__":
    main()
