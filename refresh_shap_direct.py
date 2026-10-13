"""Regenerate verified forest explanations without a Jupyter kernel."""
import os
os.environ["MPLBACKEND"] = "Agg"
import hashlib
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from src.config import MODELS_DIR, TABLES_DIR, FIGURES_DIR
from src.data import prepare_dataset, load_diabetes_dataset
from src.preprocess import split_classification_data
from src.explainability import build_explainability_outputs
from src.plots import save_figure, set_publication_style

bundle = prepare_dataset(load_diabetes_dataset())
train, test, target_train, target_test = split_classification_data(bundle.clean)
model = joblib.load(MODELS_DIR / "best_classifier.joblib")
assert isinstance(model.named_steps["model"], RandomForestClassifier)
package = build_explainability_outputs(model, train, test, target_test)
assert package["method"] == "shap"
explanation = package["shap_package"]["explanation"]
probabilities = model.predict_proba(test)[:, 1]
error = float(np.max(np.abs(explanation.base_values + explanation.values.sum(axis=1) - probabilities)))
assert error < 1e-6
set_publication_style()
plt.close("all")
shap.plots.beeswarm(explanation, max_display=10, show=False)
plt.title(f"Random Forest SHAP Values (n={len(test)})")
save_figure(plt.gcf(), "shap_beeswarm.png")
plt.close("all")
shap.plots.bar(explanation, max_display=10, show=False)
plt.title(f"Random Forest Global SHAP Importance (n={len(test)})")
save_figure(plt.gcf(), "shap_bar_importance.png")
plt.close("all")
position = list(test.index).index(package["case_index"])
shap.plots.waterfall(explanation[position], max_display=10, show=False)
save_figure(plt.gcf(), "shap_local_waterfall.png")
pd.DataFrame({"Feature": explanation.feature_names, "MeanAbsoluteSHAP": np.abs(explanation.values).mean(axis=0)}).sort_values("MeanAbsoluteSHAP", ascending=False).to_csv(TABLES_DIR / "shap_global_importance.csv", index=False)
pd.DataFrame({"Feature": explanation.feature_names, "Value": explanation.data[position], "SHAP": explanation.values[position]}).to_csv(TABLES_DIR / "shap_local_contributions.csv", index=False)
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
provenance = {"generated_utc": datetime.now(timezone.utc).isoformat(),
    "estimator_class": type(model.named_steps["model"]).__module__ + "." + type(model.named_steps["model"]).__name__,
    "model_sha256": digest(MODELS_DIR / "best_classifier.joblib"),
    "figure_sha256": digest(FIGURES_DIR / "shap_beeswarm.png"),
    "explainer": "shap.TreeExplainer", "feature_perturbation": "tree_path_dependent", "model_output": "raw (positive-class probability for Random Forest)",
    "external_background": None, "shap_version": shap.__version__, "test_patients": len(test),
    "test_patient_indices": test.index.tolist(), "maximum_additivity_error": error,
    "validation": "RandomForestClassifier asserted; SHAP sums reconstruct forest probabilities"}
(TABLES_DIR / "shap_beeswarm_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print("Verified Random Forest SHAP refreshed; maximum reconstruction error", error)
