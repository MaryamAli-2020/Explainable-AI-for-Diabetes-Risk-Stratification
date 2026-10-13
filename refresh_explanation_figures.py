"""Regenerate explanation outputs from the saved, training-selected classifier."""
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / "notebooks/01_diabetes_healthcare_analytics.ipynb"
original = nbformat.read(path, as_version=4)
refresh = nbformat.v4.new_notebook(cells=[
    original.cells[5], original.cells[9], original.cells[11],
    nbformat.v4.new_code_cell('from sklearn.ensemble import RandomForestClassifier\nclassifier = joblib.load(MODELS_DIR / "best_classifier.joblib")\nassert isinstance(classifier.named_steps["model"], RandomForestClassifier), "Saved classifier is not Random Forest"\nbest_tree_run = {"BestEstimator": classifier, "Model": "Random Forest"}\nprint("Verified estimator:", type(classifier.named_steps["model"]).__module__ + "." + type(classifier.named_steps["model"]).__name__)'),
    original.cells[26],
    nbformat.v4.new_code_cell('''import hashlib
import json
from datetime import datetime, timezone

assert xai_results["method"] == "shap", "SHAP computation fell back; figure provenance cannot be confirmed"
explanation = xai_results["shap_package"]["explanation"]
probabilities = classifier.predict_proba(X_test)[:, 1]
reconstructed = explanation.base_values + explanation.values.sum(axis=1)
assert np.allclose(reconstructed, probabilities, atol=1e-6), "SHAP values do not reconstruct Random Forest predictions"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
provenance = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "estimator_class": type(classifier.named_steps["model"]).__module__ + "." + type(classifier.named_steps["model"]).__name__,
    "model_file": "outputs/models/best_classifier.joblib",
    "model_sha256": digest(MODELS_DIR / "best_classifier.joblib"),
    "figure_file": "outputs/figures/shap_beeswarm.png",
    "figure_sha256": digest(FIGURES_DIR / "shap_beeswarm.png"),
    "explainer": type(xai_results["shap_package"]["explainer"]).__module__ + "." + type(xai_results["shap_package"]["explainer"]).__name__,
    "shap_version": shap.__version__,
    "test_patients": len(X_test),
    "test_patient_indices": X_test.index.tolist(),
    "maximum_additivity_error": float(np.max(np.abs(reconstructed - probabilities))),
    "validation": "RandomForestClassifier type asserted; SHAP sums reconstruct saved Random Forest positive-class probabilities",
}
(TABLES_DIR / "shap_beeswarm_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print(json.dumps(provenance, indent=2))
'''),
])
NotebookClient(refresh, kernel_name="python3", timeout=180, resources={"metadata": {"path": str(root)}}).execute()
latest = nbformat.read(path, as_version=4)
latest.cells[26] = refresh.cells[4]
nbformat.write(latest, path)
print("Explanation figures and SHAP contribution tables regenerated.")
