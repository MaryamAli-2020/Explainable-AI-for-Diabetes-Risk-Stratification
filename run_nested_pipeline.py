"""Run the additional nested study while protecting the original held-out artifacts."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import hashlib
import json
from src.nested_pipeline import evaluate_nested_pipeline

def main():
    root = Path(__file__).resolve().parent
    protected = ["outputs/models/best_classifier.joblib", "outputs/models/best_regressor.joblib",
        "outputs/tables/recommended_threshold.csv", "outputs/tables/threshold_test_predictions.csv",
        "outputs/tables/revision_test_predictions.csv", "outputs/tables/classification_model_comparison.csv",
        "outputs/tables/shap_beeswarm_provenance.json"]
    digest = lambda path: hashlib.sha256((root / path).read_bytes()).hexdigest()
    before = {path: digest(path) for path in protected}
    result = evaluate_nested_pipeline()
    after = {path: digest(path) for path in protected}
    assert before == after, "Original held-out artifacts changed"
    (root / "outputs/tables/nested_test_preservation.json").write_text(json.dumps({"passed": True, "sha256_before": before, "sha256_after": after}, indent=2), encoding="utf-8")
    print(result.to_string(index=False), flush=True)
    print("Nested complete-pipeline evaluation finished; original test artifacts unchanged.", flush=True)

if __name__ == "__main__":
    main()
