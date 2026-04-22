from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .config import CATEGORICAL_FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE
from .data import get_classification_frame


def build_numeric_transformer(scale: bool = True) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", RobustScaler()))
    return Pipeline(steps)


def build_categorical_transformer() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessor(feature_names: Sequence[str], scale: bool = True) -> ColumnTransformer:
    numeric_features = [feature for feature in feature_names if feature not in CATEGORICAL_FEATURE_COLUMNS]
    categorical_features = [feature for feature in feature_names if feature in CATEGORICAL_FEATURE_COLUMNS]
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_features:
        transformers.append(("num", build_numeric_transformer(scale=scale), numeric_features))
    if categorical_features:
        transformers.append(("cat", build_categorical_transformer(), categorical_features))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_classification_pipeline(
    estimator: object,
    feature_names: Sequence[str],
    *,
    scale: bool = True,
    use_smote: bool = False,
) -> Pipeline:
    preprocessor = build_preprocessor(feature_names=feature_names, scale=scale)
    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", estimator),
            ]
        )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def build_regression_pipeline(
    estimator: object,
    feature_names: Sequence[str],
    *,
    scale: bool = True,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(feature_names=feature_names, scale=scale)),
            ("model", estimator),
        ]
    )


def split_classification_data(
    clean_frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features, target = get_classification_frame(clean_frame)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )


def prepare_clustering_matrix(
    clean_frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, object]:
    pipeline = build_numeric_transformer(scale=True)
    transformed = pipeline.fit_transform(clean_frame.loc[:, list(feature_names)])
    transformed_frame = pd.DataFrame(transformed, columns=list(feature_names), index=clean_frame.index)
    return transformed_frame, pipeline
