from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import RANDOM_STATE
from .models import get_positive_class_scores, get_transformed_feature_names


def shap_is_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


def choose_local_case(
    fitted_model: object,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> tuple[int, str]:
    probabilities = pd.Series(get_positive_class_scores(fitted_model, test_features), index=test_features.index)
    predictions = (probabilities >= 0.5).astype(int)
    false_negative_index = test_features.index[(predictions == 0) & (test_target == 1)]
    if len(false_negative_index) > 0:
        return int(false_negative_index[0]), "False negative case chosen to highlight missed-risk behavior."

    boundary_index = (probabilities - 0.5).abs().sort_values().index[0]
    return int(boundary_index), "Borderline-risk case chosen because no false negative was observed."


def compute_shap_package(
    fitted_model: object,
    background_features: pd.DataFrame,
    explanation_features: pd.DataFrame,
) -> dict[str, Any]:
    import shap

    preprocess = fitted_model.named_steps["preprocess"]
    model = fitted_model.named_steps["model"]
    transformed_background = preprocess.transform(background_features)
    transformed_explanation = preprocess.transform(explanation_features)
    transformed_feature_names = get_transformed_feature_names(
        fitted_model,
        list(background_features.columns),
    )

    transformed_background_frame = pd.DataFrame(
        transformed_background,
        columns=transformed_feature_names,
        index=background_features.index,
    )
    transformed_explanation_frame = pd.DataFrame(
        transformed_explanation,
        columns=transformed_feature_names,
        index=explanation_features.index,
    )

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    raw_values = explainer.shap_values(transformed_explanation_frame)
    expected_value = explainer.expected_value

    if isinstance(raw_values, list):
        class_index = 1 if len(raw_values) > 1 else 0
        values = raw_values[class_index]
        base_value = expected_value[class_index] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
        class_index = 1 if raw_values.shape[-1] > 1 else 0
        values = raw_values[:, :, class_index]
        if isinstance(expected_value, np.ndarray) and expected_value.ndim > 0:
            base_value = expected_value[class_index]
        else:
            base_value = expected_value
    else:
        values = raw_values
        base_value = expected_value

    explanation = shap.Explanation(
        values=values,
        base_values=np.repeat(base_value, transformed_explanation_frame.shape[0]),
        data=transformed_explanation_frame.to_numpy(),
        feature_names=list(transformed_explanation_frame.columns),
    )

    return {
        "explainer": explainer,
        "background": transformed_background_frame,
        "explanation_frame": transformed_explanation_frame,
        "explanation": explanation,
    }


def compute_permutation_fallback(
    fitted_model: object,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        fitted_model,
        test_features,
        test_target,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    frame = pd.DataFrame(
        {
            "Feature": list(test_features.columns),
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }
    )
    return frame.sort_values("Importance", ascending=False).reset_index(drop=True)


def build_explainability_outputs(
    fitted_model: object,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> dict[str, Any]:
    case_index, case_note = choose_local_case(fitted_model, test_features, test_target)
    local_features = test_features.loc[[case_index]]

    outputs: dict[str, Any] = {
        "case_index": case_index,
        "case_note": case_note,
        "local_features": local_features,
    }
    if shap_is_available():
        try:
            outputs["method"] = "shap"
            outputs["shap_package"] = compute_shap_package(
                fitted_model=fitted_model,
                background_features=train_features,
                explanation_features=test_features,
            )
            return outputs
        except Exception as error:  # pragma: no cover - safety fallback for model-specific SHAP issues.
            outputs["method"] = "permutation"
            outputs["fallback_reason"] = f"SHAP fallback triggered: {error}"
            outputs["permutation_importance"] = compute_permutation_fallback(
                fitted_model=fitted_model,
                test_features=test_features,
                test_target=test_target,
            )
            return outputs

    outputs["method"] = "permutation"
    outputs["permutation_importance"] = compute_permutation_fallback(
        fitted_model=fitted_model,
        test_features=test_features,
        test_target=test_target,
    )
    return outputs
