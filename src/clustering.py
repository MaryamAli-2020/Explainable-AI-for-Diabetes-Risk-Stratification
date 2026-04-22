from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import NUMERIC_FEATURE_COLUMNS, PREFERRED_CLUSTER_K, RANDOM_STATE, TARGET_COLUMN
from .preprocess import prepare_clustering_matrix


def evaluate_kmeans_range(
    scaled_features: pd.DataFrame,
    k_values: range = range(2, 7),
) -> pd.DataFrame:
    records = []
    for n_clusters in k_values:
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(scaled_features)
        records.append(
            {
                "Clusters": n_clusters,
                "Inertia": float(model.inertia_),
                "Silhouette": float(silhouette_score(scaled_features, labels)),
            }
        )
    return pd.DataFrame(records)


def choose_optimal_k(
    evaluation_frame: pd.DataFrame,
    preferred_k: int = PREFERRED_CLUSTER_K,
) -> int:
    rounded = evaluation_frame.copy()
    rounded["SilhouetteRounded"] = rounded["Silhouette"].round(4)
    best_value = rounded["SilhouetteRounded"].max()
    tied = rounded.loc[rounded["SilhouetteRounded"] == best_value, "Clusters"].tolist()
    if preferred_k in tied:
        return preferred_k
    return int(rounded.sort_values("Silhouette", ascending=False).iloc[0]["Clusters"])


def build_cluster_profile(clustered_frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    profile = clustered_frame.groupby("Cluster")[feature_names + [TARGET_COLUMN]].mean()
    profile["Count"] = clustered_frame.groupby("Cluster").size()
    profile["OutcomePrevalence"] = profile[TARGET_COLUMN]
    profile = profile.drop(columns=[TARGET_COLUMN]).reset_index()
    return profile


def generate_cluster_narratives(profile_frame: pd.DataFrame, reference_frame: pd.DataFrame) -> pd.DataFrame:
    medians = reference_frame.median(numeric_only=True)
    narratives = []
    for _, row in profile_frame.iterrows():
        age_text = "younger" if row["Age"] < medians["Age"] else "older"
        bmi_text = "lower-BMI" if row["BMI"] < medians["BMI"] else "higher-BMI"
        glucose_text = "lower-glucose" if row["Glucose"] < medians["Glucose"] else "higher-glucose"
        outcome_text = (
            "low observed diabetes prevalence"
            if row["OutcomePrevalence"] < reference_frame[TARGET_COLUMN].mean()
            else "elevated observed diabetes prevalence"
        )
        narratives.append(
            {
                "Cluster": int(row["Cluster"]),
                "ClusterLabel": f"{age_text.title()} / {bmi_text.title()} / {glucose_text.title()}",
                "Narrative": (
                    f"Cluster {int(row['Cluster'])} is a relatively {age_text}, {bmi_text}, "
                    f"{glucose_text} subgroup with {outcome_text}."
                ),
            }
        )
    return pd.DataFrame(narratives)


def run_clustering_analysis(clean_frame: pd.DataFrame) -> dict[str, object]:
    scaled_features, scaler = prepare_clustering_matrix(clean_frame, NUMERIC_FEATURE_COLUMNS)
    evaluation = evaluate_kmeans_range(scaled_features)
    selected_k = choose_optimal_k(evaluation)

    kmeans = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=20)
    cluster_labels = kmeans.fit_predict(scaled_features)
    agglomerative = AgglomerativeClustering(n_clusters=selected_k)
    agglomerative_labels = agglomerative.fit_predict(scaled_features)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_projection = pca.fit_transform(scaled_features)

    clustered = clean_frame.copy()
    clustered["Cluster"] = cluster_labels
    clustered["AgglomerativeCluster"] = agglomerative_labels
    clustered["PCA1"] = pca_projection[:, 0]
    clustered["PCA2"] = pca_projection[:, 1]

    profile = build_cluster_profile(clustered, NUMERIC_FEATURE_COLUMNS)
    narratives = generate_cluster_narratives(profile, clean_frame)

    return {
        "scaled_features": scaled_features,
        "scaler": scaler,
        "evaluation": evaluation,
        "selected_k": selected_k,
        "clustered_frame": clustered,
        "profile": profile.merge(narratives, on="Cluster", how="left"),
        "pca_explained_variance": pca.explained_variance_ratio_,
        "kmeans_model": kmeans,
        "agglomerative_model": agglomerative,
    }
