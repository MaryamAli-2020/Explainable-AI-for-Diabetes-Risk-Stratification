"""Verify training-selected regression and regenerate its diagnostics."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
from dataclasses import replace
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from src.config import TABLES_DIR, MODELS_DIR
from src.data import load_diabetes_dataset, prepare_dataset, get_regression_frame
from src.models import get_regressor_specs, fit_regressor_search, evaluate_regressor
from src.plots import plot_regression_diagnostics, save_figure, set_publication_style

bundle = prepare_dataset(load_diabetes_dataset())
X, y = get_regression_frame(bundle.clean, target_column="Glucose")
train, test, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=42)
results, audit_rows = [], {}
for name, spec in get_regressor_specs().items():
    print("Regression training selection:", name, flush=True)
    if "n_jobs" in spec.estimator.get_params():
        spec = replace(spec, estimator=clone(spec.estimator).set_params(n_jobs=1))
    search = fit_regressor_search(spec, train, train_y, list(train.columns))
    result = {"Model": name, "CV Neg-RMSE": search.best_score_,
        **evaluate_regressor(search.best_estimator_, test, test_y), "Estimator": search.best_estimator_}
    results.append(result)
    audit_rows[name] = {"search_space": spec.param_grid, "best_params": search.best_params_, "CV Neg-RMSE": search.best_score_}
best = max(results, key=lambda r: r["CV Neg-RMSE"])
assert best["Model"] == "Random Forest Regressor"
ci = pd.read_csv(TABLES_DIR / "revision_regression_confidence_intervals.csv")
for result in results:
    for metric in ("MAE", "MSE", "RMSE", "R2"):
        reference = ci[(ci.Model == result["Model"]) & (ci.Metric == metric)].iloc[0].Estimate
        assert np.isclose(result[metric], reference), (result["Model"], metric)
joblib.dump(best["Estimator"], MODELS_DIR / "best_regressor.joblib")
set_publication_style()
fig = plot_regression_diagnostics(test_y, best["Predictions"], best["Residuals"])
fig.suptitle(f"Training-selected {best['Model']} (test n={len(test_y)})", fontsize=16)
fig.subplots_adjust(top=.80, wspace=.30)
save_figure(fig, "regression_diagnostics.png")
comparison = pd.DataFrame([{k:r[k] for k in ("Model", "CV Neg-RMSE", "MAE", "MSE", "RMSE", "R2")} for r in results]).sort_values("CV Neg-RMSE", ascending=False)
comparison.rename(columns={m: "Test " + m for m in ("MAE", "MSE", "RMSE", "R2")}).round(4).to_csv(TABLES_DIR / "regression_model_comparison.csv", index=False)
audit = json.loads((TABLES_DIR / "reviewer_revision_audit.json").read_text())
audit["regression"] = {"selected_model": best["Model"], "selection": "Maximum training CV negative RMSE; no test-based selection", "specifications": audit_rows}
(TABLES_DIR / "reviewer_revision_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print("Verified selected regressor:", best["Model"], "test RMSE", best["RMSE"], flush=True)
