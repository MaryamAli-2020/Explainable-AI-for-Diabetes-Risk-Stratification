from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    CLINICAL_ZERO_AS_MISSING,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def _normalize_label(label: str) -> str:
    return "".join(character for character in label.lower() if character.isalnum())


COLUMN_ALIASES = {
    "patientnumber": "PatientNumber",
    "cholesterol": "Cholesterol",
    "glucose": "Glucose",
    "hdlchol": "HDLChol",
    "hdlcholesterol": "HDLChol",
    "cholhdlratio": "CholHDLRatio",
    "age": "Age",
    "gender": "Gender",
    "height": "Height",
    "weight": "Weight",
    "bmi": "BMI",
    "systolicbp": "SystolicBP",
    "diastolicbp": "DiastolicBP",
    "waist": "Waist",
    "hip": "Hip",
    "waisthipratio": "WaistHipRatio",
    "diabetes": TARGET_COLUMN,
    "outcome": TARGET_COLUMN,
}


@dataclass(frozen=True)
class DatasetBundle:
    raw: pd.DataFrame
    clean: pd.DataFrame
    zero_audit: pd.DataFrame
    data_quality: pd.DataFrame
    outlier_summary: pd.DataFrame
    dataset_summary: pd.DataFrame
    class_balance: pd.DataFrame
    metadata: dict[str, float | int]


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common diabetes dataset column variants to canonical names."""
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        normalized = _normalize_label(str(column))
        if normalized in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[normalized]
    return frame.rename(columns=rename_map)


def validate_schema(frame: pd.DataFrame) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing)
            + ". Expected the diagnostic-measures schema used by this project."
        )


def _coerce_decimal_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    normalized = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(normalized, errors="coerce")


def _standardize_gender(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({"female": "Female", "male": "Male"}).fillna(series)


def _standardize_outcome(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "diabetes": 1,
            "yes": 1,
            "positive": 1,
            "1": 1,
            "no diabetes": 0,
            "no": 0,
            "negative": 0,
            "0": 0,
        }
    )
    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"Unrecognized outcome labels in dataset: {unknown}")
    return mapped.astype(int)


def load_diabetes_dataset(csv_path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Kaggle diabetes dataset from disk and enforce the expected schema."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Kaggle CSV at data/raw/diabetes.csv."
        )

    frame = pd.read_csv(path)
    frame = canonicalize_columns(frame)
    validate_schema(frame)
    cleaned = frame.copy()
    for column in NUMERIC_FEATURE_COLUMNS:
        cleaned[column] = _coerce_decimal_numeric(cleaned[column])
    cleaned["Gender"] = _standardize_gender(cleaned["Gender"])
    cleaned[TARGET_COLUMN] = _standardize_outcome(cleaned[TARGET_COLUMN])
    return cleaned.loc[:, EXPECTED_COLUMNS].copy()


def zero_as_missing_audit(
    frame: pd.DataFrame, columns: Iterable[str] = CLINICAL_ZERO_AS_MISSING
) -> pd.DataFrame:
    total = len(frame)
    records = []
    for column in columns:
        zero_count = int((frame[column] == 0).sum())
        records.append(
            {
                "Feature": column,
                "ZeroCount": zero_count,
                "ZeroPercent": round((zero_count / total) * 100, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(records).sort_values("ZeroCount", ascending=False).reset_index(drop=True)


def replace_clinical_zeros_with_nan(
    frame: pd.DataFrame, columns: Iterable[str] = CLINICAL_ZERO_AS_MISSING
) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in columns:
        cleaned[column] = cleaned[column].replace(0, np.nan)
    return cleaned


def iqr_outlier_summary(frame: pd.DataFrame, columns: Iterable[str] = NUMERIC_FEATURE_COLUMNS) -> pd.DataFrame:
    records = []
    for column in columns:
        series = frame[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        outlier_count = int(((series < lower) | (series > upper)).sum())
        records.append(
            {
                "Feature": column,
                "Q1": round(float(q1), 3),
                "Q3": round(float(q3), 3),
                "IQR": round(float(iqr), 3),
                "LowerFence": round(float(lower), 3),
                "UpperFence": round(float(upper), 3),
                "OutlierCount": outlier_count,
                "OutlierPercent": round((outlier_count / len(series)) * 100, 2) if len(series) else 0.0,
            }
        )
    return pd.DataFrame(records).sort_values("OutlierPercent", ascending=False).reset_index(drop=True)


def build_data_quality_summary(raw_frame: pd.DataFrame, clean_frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in EXPECTED_COLUMNS:
        records.append(
            {
                "Feature": column,
                "Dtype": str(clean_frame[column].dtype),
                "RawMissing": int(raw_frame[column].isna().sum()),
                "ZeroAsMissingCandidates": int((raw_frame[column] == 0).sum())
                if column in CLINICAL_ZERO_AS_MISSING
                else 0,
                "CleanMissing": int(clean_frame[column].isna().sum()),
                "MissingPercentAfterCleaning": round(clean_frame[column].isna().mean() * 100, 2),
            }
        )
    return pd.DataFrame(records)


def build_dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.describe(include="all").transpose()
    summary["missing_count"] = frame.isna().sum()
    summary["missing_pct"] = (frame.isna().mean() * 100).round(2)
    return summary


def build_class_balance_table(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame[TARGET_COLUMN].value_counts(dropna=False).rename_axis(TARGET_COLUMN).reset_index(name="Count")
    counts["Percent"] = (counts["Count"] / counts["Count"].sum() * 100).round(2)
    counts["Label"] = counts[TARGET_COLUMN].map({0: "No Diabetes", 1: "Diabetes"})
    return counts.loc[:, [TARGET_COLUMN, "Label", "Count", "Percent"]]


def prepare_dataset(frame: pd.DataFrame) -> DatasetBundle:
    raw = frame.copy()
    duplicates_removed = int(raw.duplicated().sum())
    deduplicated = raw.drop_duplicates().reset_index(drop=True)
    zero_audit = zero_as_missing_audit(deduplicated)
    clean = replace_clinical_zeros_with_nan(deduplicated)
    bundle = DatasetBundle(
        raw=raw,
        clean=clean,
        zero_audit=zero_audit,
        data_quality=build_data_quality_summary(deduplicated, clean),
        outlier_summary=iqr_outlier_summary(clean),
        dataset_summary=build_dataset_summary(clean),
        class_balance=build_class_balance_table(clean),
        metadata={
            "rows_raw": int(len(raw)),
            "rows_clean": int(len(clean)),
            "columns": int(clean.shape[1]),
            "duplicates_removed": duplicates_removed,
        },
    )
    return bundle


def get_classification_frame(clean_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = clean_frame.loc[:, FEATURE_COLUMNS].copy()
    target = clean_frame[TARGET_COLUMN].astype(int).copy()
    return features, target


def get_regression_frame(
    clean_frame: pd.DataFrame, target_column: str = "Glucose"
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in FEATURE_COLUMNS:
        raise ValueError(f"{target_column} is not a supported clinical regression target.")

    regression_frame = clean_frame.dropna(subset=[target_column]).copy()
    feature_columns = [column for column in FEATURE_COLUMNS if column != target_column]
    features = regression_frame.loc[:, feature_columns]
    target = regression_frame[target_column]
    return features, target
