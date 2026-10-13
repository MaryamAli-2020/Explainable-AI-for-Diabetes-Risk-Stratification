"""Match the Kaggle export to primary-source records without assuming provenance."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from src.data import load_diabetes_dataset

root = Path(__file__).resolve().parent
cache = root / "data/provenance/diabetes_primary.csv"
url = "https://hbiostat.org/data/repo/diabetes.csv"
cache.parent.mkdir(parents=True, exist_ok=True)
if not cache.exists():
    with urlopen(url, timeout=30) as response:
        cache.write_bytes(response.read())
primary = pd.read_csv(cache)
local = load_diabetes_dataset()
mapping = {"chol": "Cholesterol", "stab.glu": "Glucose", "hdl": "HDLChol",
    "age": "Age", "gender": "Gender", "height": "Height", "weight": "Weight",
    "bp.1s": "SystolicBP", "bp.1d": "DiastolicBP", "waist": "Waist", "hip": "Hip"}
source = primary.rename(columns=mapping)
for frame in (local, source):
    frame["Gender"] = frame.Gender.astype(str).str.strip().str.lower()
keys = list(mapping.values())
exact = local.merge(source, on=keys, how="left", validate="one_to_one", indicator=True)
core = ["Cholesterol", "Glucose", "HDLChol", "Age", "Gender"]
core_matches = local.merge(source, on=core, how="left", validate="one_to_one", indicator=True)
records = []
for index, row in local.iterrows():
    compatible = source[keys].eq(row[keys], axis="columns") | source[keys].isna()
    candidates = source.loc[compatible.all(axis=1)]
    if len(candidates) != 1:
        raise ValueError(f"Record {index}: {len(candidates)} compatible primary records; cannot assign provenance")
    candidate = candidates.iloc[0]
    records.append({"LocalIndex": index, "id": int(candidate.id), "Outcome": int(row.Outcome),
        "glyhb": candidate.glyhb, "Glucose": row.Glucose, "time.ppn": candidate["time.ppn"],
        "PrimaryMissingPredictors": int(candidate[keys].isna().sum())})
matched = pd.DataFrame(records)
assert matched.id.is_unique
result = {"source": url, "primary_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    "primary_records": len(primary), "local_records": len(local),
    "matching_variables": keys, "matched_records": len(matched),
    "exact_matches_on_all_11_variables": int((exact._merge == "both").sum()),
    "exact_matches_on_five_core_variables": int((core_matches._merge == "both").sum()),
    "matching_rule": "All 11 nonmissing primary-source fields must agree; primary missing values are wildcards; exactly one compatible source row required",
    "primary_records_with_missing_predictors_in_export": int((matched.PrimaryMissingPredictors > 0).sum()),
    "matched_records_missing_glyhb": int(matched.glyhb.isna().sum()),
    "label_disagreements_with_glyhb_gt_7": int(((matched.glyhb > 7).astype(int) != matched.Outcome).sum()),
    "unused_primary_records": int((~primary.id.isin(matched.id)).sum()),
    "unused_primary_records_with_missing_glyhb": int(primary.loc[~primary.id.isin(matched.id), "glyhb"].isna().sum()),
    "interpretation": "Empirical record/label agreement is evidence of consistency, not documentation of the publisher's transformation or clinical adjudication"}
matched.to_csv(root / "outputs/tables/label_provenance_matches.csv", index=False)
(root / "outputs/tables/label_provenance_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
