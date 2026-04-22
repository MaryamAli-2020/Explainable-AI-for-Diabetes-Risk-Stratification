from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 3
PAIRPLOT_SAMPLE_SIZE = 250
PREFERRED_CLUSTER_K = 3
CLASSIFICATION_PRIMARY_METRIC = "roc_auc"
REGRESSION_PRIMARY_METRIC = "neg_root_mean_squared_error"
TARGET_COLUMN = "Outcome"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "diabetes.csv"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORT_DIR = PROJECT_ROOT / "report"

EXPECTED_COLUMNS = [
    "Cholesterol",
    "Glucose",
    "HDLChol",
    "CholHDLRatio",
    "Age",
    "Gender",
    "Height",
    "Weight",
    "BMI",
    "SystolicBP",
    "DiastolicBP",
    "Waist",
    "Hip",
    "WaistHipRatio",
    TARGET_COLUMN,
]

NUMERIC_FEATURE_COLUMNS = [
    "Cholesterol",
    "Glucose",
    "HDLChol",
    "CholHDLRatio",
    "Age",
    "Height",
    "Weight",
    "BMI",
    "SystolicBP",
    "DiastolicBP",
    "Waist",
    "Hip",
    "WaistHipRatio",
]
CATEGORICAL_FEATURE_COLUMNS = ["Gender"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
CLINICAL_ZERO_AS_MISSING = NUMERIC_FEATURE_COLUMNS.copy()

PUBLICATION_STYLE = {
    "style": "whitegrid",
    "context": "talk",
    "palette": "crest",
}


def ensure_project_directories() -> None:
    """Create project directories used by the notebook and helper modules."""
    for directory in (
        RAW_DATA_DIR,
        NOTEBOOKS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
