"""Reproducible analysis for the DSAA2011 Covertype project.

Run from the repository root:

    python3 scripts/prepare_data.py
    python3 analysis_covertype.py

The script writes figures to `figures/`, tables to `outputs/`, and a
machine-readable summary to `outputs/summary.json`.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    auc,
    calinski_harabasz_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "covertype.csv.gz"
RAW_DATA_PATH = ROOT / "data" / "covtype.data.gz"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"

MODEL_SAMPLE_SIZE = 120_000
CV_SAMPLE_SIZE = 36_000
TSNE_SAMPLE_SIZE = 6_000
CLUSTER_SAMPLE_SIZE = 6_000
AGGLOMERATIVE_SAMPLE_SIZE = 2_500
PERMUTATION_SAMPLE_SIZE = 8_000
MINIBATCH_SIZE = 2048
CACHE_TAG = "covertype_optimized_v1"
FORCE_RECOMPUTE = os.environ.get("FORCE_RECOMPUTE", "0") == "1"

FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["font.family"] = "DejaVu Sans"

CONTINUOUS_FEATURES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_FEATURES = [f"Wilderness_Area_{i}" for i in range(1, 5)]
SOIL_FEATURES = [f"Soil_Type_{i}" for i in range(1, 41)]
FEATURE_NAMES = CONTINUOUS_FEATURES + WILDERNESS_FEATURES + SOIL_FEATURES
ALL_COLUMNS = FEATURE_NAMES + ["Cover_Type"]
DTYPE_MAP = {
    **{col: "float32" for col in CONTINUOUS_FEATURES},
    **{col: "uint8" for col in WILDERNESS_FEATURES + SOIL_FEATURES + ["Cover_Type"]},
}
CLASSES = [1, 2, 3, 4, 5, 6, 7]
COVER_TYPE_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}
WILDERNESS_NAMES = {
    1: "Rawah",
    2: "Neota",
    3: "Comanche Peak",
    4: "Cache la Poudre",
}
CLASS_LABELS = [f"{k}: {COVER_TYPE_NAMES[k]}" for k in CLASSES]
PALETTE = {
    1: "#4C72B0",
    2: "#55A868",
    3: "#C44E52",
    4: "#8172B3",
    5: "#CCB974",
    6: "#64B5CD",
    7: "#DD8452",
}


RUNTIME_ROWS: list[dict] = []
STAGE_CACHE_STATUS: dict[str, str] = {}


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False)


def save_json(obj: dict, name: str = "summary.json") -> None:
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value

    (OUT_DIR / name).write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def stage_manifest(stage: str) -> Path:
    slug = stage.lower().replace(" ", "_").replace("-", "_")
    return OUT_DIR / f"{slug}_cache_manifest.json"


def cache_metadata(stage: str) -> dict:
    return {
        "cache_tag": CACHE_TAG,
        "stage": stage,
        "random_state": RANDOM_STATE,
        "model_sample_size": MODEL_SAMPLE_SIZE,
        "cv_sample_size": CV_SAMPLE_SIZE,
        "tsne_sample_size": TSNE_SAMPLE_SIZE,
        "cluster_sample_size": CLUSTER_SAMPLE_SIZE,
        "agglomerative_sample_size": AGGLOMERATIVE_SAMPLE_SIZE,
        "permutation_sample_size": PERMUTATION_SAMPLE_SIZE,
        "minibatch_size": MINIBATCH_SIZE,
    }


def cache_ready(stage: str, outputs: list[str], figures: list[str] | None = None) -> bool:
    if FORCE_RECOMPUTE:
        return False
    output_paths = [OUT_DIR / name for name in outputs]
    figure_paths = [FIG_DIR / name for name in (figures or [])]
    if not all(path.exists() and path.stat().st_size > 0 for path in output_paths + figure_paths):
        return False
    manifest_path = stage_manifest(stage)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return manifest == cache_metadata(stage)


def write_cache_manifest(stage: str) -> None:
    stage_manifest(stage).write_text(
        json.dumps(cache_metadata(stage), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def timed_stage(stage: str, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    RUNTIME_ROWS.append(
        {
            "stage": stage,
            "seconds": round(elapsed, 3),
            "cache_status": STAGE_CACHE_STATUS.get(stage, "computed"),
        }
    )
    return result


def downcast_covertype_data(df: pd.DataFrame) -> pd.DataFrame:
    """Use compact dtypes without changing Covertype values."""
    out = df.copy()
    for col in CONTINUOUS_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float32")
    for col in WILDERNESS_FEATURES + SOIL_FEATURES + ["Cover_Type"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("uint8")
    return out


def read_covertype_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, dtype=DTYPE_MAP)
    elif RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH, header=None, names=ALL_COLUMNS, dtype=DTYPE_MAP)
    else:
        from scripts.prepare_data import prepare_covtype_data

        df = prepare_covtype_data()

    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in ALL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected Covertype columns: {missing}")
    return downcast_covertype_data(df[ALL_COLUMNS])


def stratified_sample(df: pd.DataFrame, n: int, target_col: str = "Cover_Type") -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    frac = n / len(df)
    return (
        df.groupby(target_col, group_keys=False, observed=True)
        .sample(frac=frac, random_state=RANDOM_STATE)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def wilderness_area_labels(df: pd.DataFrame) -> pd.Series:
    """Convert one-hot wilderness indicators into a compact area code."""
    area = df[WILDERNESS_FEATURES].to_numpy().argmax(axis=1) + 1
    return pd.Series(area.astype("uint8"), index=df.index, name="Wilderness_Area")


def make_scaled_preprocessor(columns: list[str] | None = None) -> ColumnTransformer:
    columns = columns or FEATURE_NAMES
    continuous = [col for col in CONTINUOUS_FEATURES if col in columns]
    binary = [col for col in columns if col not in continuous]

    transformers = []
    if continuous:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                continuous,
            )
        )
    if binary:
        transformers.append(
            (
                "binary",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_linear_model(columns: list[str] | None = None) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_scaled_preprocessor(columns)),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    max_iter=300,
                    tol=0.02,
                    C=1.0,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_tree_model() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=24,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def make_random_forest(n_estimators: int = 120) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_hist_gradient_boosting(max_iter: int = 160) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )


def scaled_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_NAMES]
    preprocessor = make_scaled_preprocessor(FEATURE_NAMES)
    return preprocessor.fit_transform(X).astype(np.float32, copy=False)


def plot_target_distribution(y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    plot_df = pd.DataFrame(
        {
            "Cover_Type": [f"{idx}\n{COVER_TYPE_NAMES[idx]}" for idx in counts.index],
            "count": counts.values,
            "class": counts.index,
        }
    )
    plt.figure(figsize=(9, 4.8))
    sns.barplot(
        data=plot_df,
        x="Cover_Type",
        y="count",
        hue="class",
        dodge=False,
        palette=PALETTE,
        legend=False,
    )
    plt.title("Covertype Target Distribution")
    plt.xlabel("Forest cover type")
    plt.ylabel("Count")
    for idx, value in enumerate(plot_df["count"]):
        plt.text(idx, value + 4500, f"{int(value):,}", ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distribution.png")
    plt.close()


def plot_missing_ratios(missing_ratio: pd.Series) -> None:
    top_missing = missing_ratio.sort_values(ascending=False).head(12)
    plt.figure(figsize=(8, 4.4))
    sns.barplot(x=top_missing.values, y=top_missing.index, color="#4C72B0")
    plt.title("Top Missing Ratios")
    plt.xlabel("Missing ratio")
    plt.ylabel("Column")
    plt.xlim(0, max(0.01, float(top_missing.max()) + 0.002))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missing_ratios.png")
    plt.close()


def wilderness_area_outputs(df: pd.DataFrame) -> pd.DataFrame:
    wilderness = pd.DataFrame(
        {
            "Wilderness_Area": wilderness_area_labels(df),
            "Cover_Type": df["Cover_Type"].to_numpy(),
        }
    )
    wilderness["Wilderness_Area_Name"] = wilderness["Wilderness_Area"].map(WILDERNESS_NAMES)
    wilderness["Cover_Type_Name"] = wilderness["Cover_Type"].map(COVER_TYPE_NAMES)

    count_matrix = pd.crosstab(wilderness["Wilderness_Area"], wilderness["Cover_Type"]).reindex(
        index=sorted(WILDERNESS_NAMES),
        columns=CLASSES,
        fill_value=0,
    )
    prop_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0).fillna(0)
    entropy = -(prop_matrix.replace(0, np.nan) * np.log2(prop_matrix.replace(0, np.nan))).sum(axis=1)
    max_entropy = np.log2(len(CLASSES))

    summary_rows = []
    for area in sorted(WILDERNESS_NAMES):
        counts = count_matrix.loc[area]
        dominant_class = int(counts.idxmax())
        n = int(counts.sum())
        summary_rows.append(
            {
                "Wilderness_Area": area,
                "Wilderness_Area_Name": WILDERNESS_NAMES[area],
                "count": n,
                "proportion": n / len(df),
                "dominant_cover_type": dominant_class,
                "dominant_cover_type_name": COVER_TYPE_NAMES[dominant_class],
                "dominant_cover_type_share": float(counts.max() / n),
                "normalized_cover_entropy": float(entropy.loc[area] / max_entropy),
            }
        )
    summary = pd.DataFrame(summary_rows)
    save_table(summary, "wilderness_area_summary.csv")

    long_counts = count_matrix.reset_index().melt(
        id_vars="Wilderness_Area",
        var_name="Cover_Type",
        value_name="count",
    )
    long_counts["Wilderness_Area_Name"] = long_counts["Wilderness_Area"].map(WILDERNESS_NAMES)
    long_counts["Cover_Type_Name"] = long_counts["Cover_Type"].map(COVER_TYPE_NAMES)
    long_counts["within_area_proportion"] = long_counts["count"] / long_counts.groupby("Wilderness_Area")["count"].transform("sum")
    save_table(long_counts, "wilderness_cover_distribution.csv")

    plt.figure(figsize=(9.4, 5.2))
    bottom = np.zeros(len(summary))
    x = np.arange(len(summary))
    ordered_names = summary["Wilderness_Area_Name"].tolist()
    for cover_type in CLASSES:
        values = prop_matrix.loc[summary["Wilderness_Area"], cover_type].to_numpy()
        plt.bar(x, values, bottom=bottom, label=f"{cover_type}: {COVER_TYPE_NAMES[cover_type]}", color=PALETTE[cover_type])
        bottom += values
    plt.xticks(x, ordered_names, rotation=12, ha="right")
    plt.ylim(0, 1)
    plt.title("Cover-Type Composition by Wilderness Area")
    plt.xlabel("Wilderness area")
    plt.ylabel("Within-area proportion")
    plt.legend(title="Cover type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wilderness_cover_distribution.png")
    plt.close()
    return summary


def run_tsne(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    stage = "t-SNE"
    if cache_ready(stage, ["tsne_coordinates.csv"], ["tsne_target.png"]):
        STAGE_CACHE_STATUS[stage] = "cache_hit"
        coords = pd.read_csv(OUT_DIR / "tsne_coordinates.csv")
        return coords, np.empty((0, 0), dtype=np.float32)

    STAGE_CACHE_STATUS[stage] = "computed"
    sample = stratified_sample(df, TSNE_SAMPLE_SIZE)
    X_scaled = scaled_feature_matrix(sample)
    X_pca = PCA(n_components=30, random_state=RANDOM_STATE).fit_transform(X_scaled)
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    embedding = tsne.fit_transform(X_pca)
    coords = pd.DataFrame(
        {
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
            "Cover_Type": sample["Cover_Type"].to_numpy(),
            "Cover_Type_Name": sample["Cover_Type"].map(COVER_TYPE_NAMES).to_numpy(),
        }
    )
    save_table(coords, "tsne_coordinates.csv")

    plt.figure(figsize=(8.4, 6.2))
    sns.scatterplot(
        data=coords,
        x="tsne_1",
        y="tsne_2",
        hue="Cover_Type",
        palette=PALETTE,
        s=9,
        linewidth=0,
        alpha=0.75,
    )
    plt.title("t-SNE Projection Colored by Forest Cover Type")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [f"{int(label)}: {COVER_TYPE_NAMES[int(label)]}" for label in labels]
    plt.legend(handles, labels, title="Cover type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tsne_target.png")
    plt.close()
    write_cache_manifest(stage)

    return coords, X_pca


def run_clustering(df: pd.DataFrame, tsne_coords: pd.DataFrame) -> pd.DataFrame:
    stage = "clustering"
    clustering_outputs = [
        "kmeans_k_selection.csv",
        "hierarchical_k_selection.csv",
        "clustering_evaluation.csv",
        "clustering_alignment_interpretation.csv",
        "clustering_plot_assignments.csv",
    ]
    clustering_figures = [
        "kmeans_elbow_silhouette.png",
        "hierarchical_dendrogram.png",
        "clustering_tsne.png",
    ]
    if cache_ready(stage, clustering_outputs, clustering_figures):
        STAGE_CACHE_STATUS[stage] = "cache_hit"
        return pd.read_csv(OUT_DIR / "clustering_evaluation.csv")

    STAGE_CACHE_STATUS[stage] = "computed"
    sample = stratified_sample(df, CLUSTER_SAMPLE_SIZE)
    X_scaled = scaled_feature_matrix(sample)
    X_pca = PCA(n_components=20, random_state=RANDOM_STATE).fit_transform(X_scaled)
    y = sample["Cover_Type"].to_numpy()
    y_wilderness = wilderness_area_labels(sample).to_numpy()

    kmeans_rows = []
    for k in range(2, 11):
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=MINIBATCH_SIZE,
            n_init=10,
            random_state=RANDOM_STATE,
        ).fit(X_pca)
        labels = kmeans.labels_
        kmeans_rows.append(
            {
                "algorithm": "MiniBatchKMeans",
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": float(silhouette_score(X_pca, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X_pca, labels)),
                "ARI_vs_Target": float(adjusted_rand_score(y, labels)),
                "ARI_vs_Wilderness": float(adjusted_rand_score(y_wilderness, labels)),
            }
        )

    ag_sample = stratified_sample(df, AGGLOMERATIVE_SAMPLE_SIZE)
    X_ag = scaled_feature_matrix(ag_sample)
    X_ag_pca = PCA(n_components=20, random_state=RANDOM_STATE).fit_transform(X_ag)
    y_ag = ag_sample["Cover_Type"].to_numpy()
    y_ag_wilderness = wilderness_area_labels(ag_sample).to_numpy()
    ag_rows = []
    for k in range(2, 11):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_ag_pca)
        ag_rows.append(
            {
                "algorithm": "Agglomerative Ward",
                "k": k,
                "inertia": np.nan,
                "silhouette": float(silhouette_score(X_ag_pca, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X_ag_pca, labels)),
                "ARI_vs_Target": float(adjusted_rand_score(y_ag, labels)),
                "ARI_vs_Wilderness": float(adjusted_rand_score(y_ag_wilderness, labels)),
            }
        )

    kmeans_df = pd.DataFrame(kmeans_rows)
    ag_df = pd.DataFrame(ag_rows)
    save_table(kmeans_df, "kmeans_k_selection.csv")
    save_table(ag_df, "hierarchical_k_selection.csv")

    plt.figure(figsize=(8.5, 4.6))
    ax1 = plt.gca()
    sns.lineplot(data=kmeans_df, x="k", y="inertia", marker="o", color="#4C72B0", ax=ax1)
    ax1.set_title("MiniBatchKMeans Elbow and Silhouette")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia", color="#4C72B0")
    ax2 = ax1.twinx()
    sns.lineplot(data=kmeans_df, x="k", y="silhouette", marker="s", color="#C44E52", ax=ax2)
    ax2.set_ylabel("Silhouette", color="#C44E52")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kmeans_elbow_silhouette.png")
    plt.close()

    dendro_sample = stratified_sample(df, 900)
    X_d = PCA(n_components=12, random_state=RANDOM_STATE).fit_transform(scaled_feature_matrix(dendro_sample))
    Z = linkage(X_d, method="ward")
    plt.figure(figsize=(9.5, 4.8))
    dendrogram(Z, truncate_mode="lastp", p=24, leaf_rotation=45, leaf_font_size=8)
    plt.title("Ward Hierarchical Clustering Dendrogram (Truncated)")
    plt.xlabel("Merged cluster")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hierarchical_dendrogram.png")
    plt.close()

    best_kmeans = kmeans_df.sort_values(["silhouette", "ARI_vs_Target"], ascending=False).iloc[0]
    best_ag = ag_df.sort_values(["silhouette", "ARI_vs_Target"], ascending=False).iloc[0]
    evaluation = pd.DataFrame([best_kmeans, best_ag])
    save_table(evaluation, "clustering_evaluation.csv")
    alignment = evaluation[
        ["algorithm", "k", "ARI_vs_Target", "ARI_vs_Wilderness"]
    ].copy()
    alignment["interpretation"] = (
        "ARI near zero versus Cover_Type means clusters do not reproduce supervised cover labels; "
        "ARI versus Wilderness checks whether the same partitions align more with regional structure."
    )
    save_table(alignment, "clustering_alignment_interpretation.csv")

    plot_n = min(AGGLOMERATIVE_SAMPLE_SIZE, len(tsne_coords))
    plot_df = tsne_coords.sample(n=plot_n, random_state=RANDOM_STATE).reset_index(drop=True)
    plot_source = (
        stratified_sample(df, TSNE_SAMPLE_SIZE)
        .sample(n=plot_n, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    X_plot = PCA(n_components=20, random_state=RANDOM_STATE).fit_transform(scaled_feature_matrix(plot_source))
    plot_df["MiniBatchKMeans_cluster"] = MiniBatchKMeans(
        n_clusters=int(best_kmeans["k"]),
        batch_size=MINIBATCH_SIZE,
        n_init=10,
        random_state=RANDOM_STATE,
    ).fit_predict(X_plot)
    plot_df["Ward_cluster"] = AgglomerativeClustering(
        n_clusters=int(best_ag["k"]),
        linkage="ward",
    ).fit_predict(X_plot)
    save_table(plot_df, "clustering_plot_assignments.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    sns.scatterplot(
        data=plot_df,
        x="tsne_1",
        y="tsne_2",
        hue="MiniBatchKMeans_cluster",
        palette="tab10",
        s=12,
        linewidth=0,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title(f"MiniBatchKMeans clusters (k={int(best_kmeans['k'])})")
    sns.scatterplot(
        data=plot_df,
        x="tsne_1",
        y="tsne_2",
        hue="Ward_cluster",
        palette="tab10",
        s=12,
        linewidth=0,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title(f"Ward clusters (k={int(best_ag['k'])})")
    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "clustering_tsne.png")
    plt.close()
    write_cache_manifest(stage)

    return evaluation


def metrics_from_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> dict:
    row = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None:
        row["AUC_ovr_macro"] = float(
            roc_auc_score(y_true, y_prob, labels=CLASSES, multi_class="ovr", average="macro")
        )
    else:
        row["AUC_ovr_macro"] = np.nan
    return row


def predict_proba_aligned(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    model_classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["model"].classes_)
    out = np.zeros((len(X), len(CLASSES)), dtype=float)
    for idx, cls in enumerate(model_classes):
        out[:, CLASSES.index(int(cls))] = proba[:, idx]
    return out


def evaluate_model(model_name: str, model, X: pd.DataFrame, y: pd.Series, split: str) -> dict:
    y_pred = model.predict(X)
    y_prob = predict_proba_aligned(model, X) if hasattr(model, "predict_proba") else None
    row = metrics_from_predictions(y, y_pred, y_prob)
    row.update({"model": model_name, "split": split, "n": int(len(y))})
    return row


def confusion_rows(model_name: str, split: str, y_true: pd.Series, y_pred: np.ndarray) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    rows = []
    for i, actual in enumerate(CLASSES):
        for j, predicted in enumerate(CLASSES):
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "actual": actual,
                    "actual_name": COVER_TYPE_NAMES[actual],
                    "predicted": predicted,
                    "predicted_name": COVER_TYPE_NAMES[predicted],
                    "count": int(cm[i, j]),
                }
            )
    return rows


def plot_confusion_grid(model_name: str, model, datasets: dict[str, tuple[pd.DataFrame, pd.Series]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    for ax, (split_name, (X_split, y_split)) in zip(axes, datasets.items()):
        y_pred = model.predict(X_split)
        cm = confusion_matrix(y_split, y_pred, labels=CLASSES)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=ax,
        )
        ax.set_title(split_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(f"{model_name}: Confusion Matrices", y=1.03)
    plt.tight_layout()
    file_slug = model_name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    plt.savefig(FIG_DIR / f"confusion_{file_slug}.png", bbox_inches="tight")
    plt.close()


def macro_roc_curve(y_true: pd.Series, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y_bin = label_binarize(y_true, classes=CLASSES)
    base_fpr = np.linspace(0, 1, 201)
    tprs = []
    for idx in range(len(CLASSES)):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        interp = np.interp(base_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return base_fpr, mean_tpr, float(auc(base_fpr, mean_tpr))


def plot_model_comparison(model_df: pd.DataFrame) -> None:
    plot_df = model_df[model_df["split"].eq("test")].copy()
    plot_df = plot_df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "f1_macro", "AUC_ovr_macro"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 4.8))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", palette=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylim(0, 1.02)
    plt.title("Test Performance by Model")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=12, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison_f1_auc.png")
    plt.close()


def plot_roc_curves(y_test: pd.Series, model_probabilities: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    plt.figure(figsize=(7.4, 5.6))
    for model_name, probs in model_probabilities.items():
        fpr, tpr, macro_auc = macro_roc_curve(y_test, probs)
        rows.extend(
            {
                "model": model_name,
                "fpr": float(x),
                "tpr": float(y),
                "macro_auc": macro_auc,
            }
            for x, y in zip(fpr, tpr)
        )
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={macro_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    plt.title("Macro-Averaged One-vs-Rest ROC Curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_ovr_models.png")
    plt.close()
    roc_df = pd.DataFrame(rows)
    save_table(roc_df, "roc_curve_points.csv")
    return roc_df


def class_report_rows(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> pd.DataFrame:
    rows = []
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    for idx, cls in enumerate(CLASSES):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "model": model_name,
                "Cover_Type": cls,
                "Cover_Type_Name": COVER_TYPE_NAMES[cls],
                "support": int(cm[idx, :].sum()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(rows)


def plot_class_recall(class_report: pd.DataFrame) -> None:
    plt.figure(figsize=(9.5, 4.8))
    sns.barplot(
        data=class_report,
        x="Cover_Type_Name",
        y="recall",
        color="#4C72B0",
    )
    plt.ylim(0, 1.02)
    plt.title("Best Model Recall by Cover Type")
    plt.xlabel("")
    plt.ylabel("Recall")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_recall_comparison.png")
    plt.close()


def feature_importance_analysis(
    rf_model: RandomForestClassifier,
    best_model_name: str,
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rf_importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    save_table(rf_importance, "feature_importance_random_forest.csv")
    save_table(rf_importance.head(20), "feature_importance_top20.csv")

    plt.figure(figsize=(8.8, 6.2))
    top = rf_importance.head(15).sort_values("importance")
    sns.barplot(data=top, x="importance", y="feature", color="#55A868")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Impurity importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_top15.png")
    plt.close()

    perm_sample = stratified_sample(pd.concat([X_test, y_test.rename("Cover_Type")], axis=1), PERMUTATION_SAMPLE_SIZE)
    perm_X = perm_sample[FEATURE_NAMES]
    perm_y = perm_sample["Cover_Type"]
    perm = permutation_importance(
        best_model,
        perm_X,
        perm_y,
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "model": best_model_name,
            "feature": FEATURE_NAMES,
            "importance_mean_f1_drop": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean_f1_drop", ascending=False)
    save_table(perm_df, "permutation_importance.csv")
    save_table(perm_df.head(15), "permutation_importance_top15.csv")

    plt.figure(figsize=(8.8, 6.2))
    top_perm = perm_df.head(15).sort_values("importance_mean_f1_drop")
    sns.barplot(data=top_perm, x="importance_mean_f1_drop", y="feature", color="#C44E52")
    plt.title(f"Permutation Importance ({best_model_name})")
    plt.xlabel("Mean macro-F1 drop")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "permutation_importance_top15.png")
    plt.close()
    return rf_importance


def feature_group_ablation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    groups = {
        "Topographic only": CONTINUOUS_FEATURES,
        "Wilderness only": WILDERNESS_FEATURES,
        "Soil only": SOIL_FEATURES,
        "Topographic + wilderness": CONTINUOUS_FEATURES + WILDERNESS_FEATURES,
        "Topographic + soil": CONTINUOUS_FEATURES + SOIL_FEATURES,
        "All features": FEATURE_NAMES,
    }
    rows = []
    for name, cols in groups.items():
        model = make_hist_gradient_boosting(max_iter=120)
        model.fit(X_train[cols], y_train)
        y_pred = model.predict(X_test[cols])
        y_prob = model.predict_proba(X_test[cols])
        row = metrics_from_predictions(y_test, y_pred, y_prob)
        row.update({"experiment": name, "n_features": len(cols)})
        rows.append(row)
    ablation = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    full_f1 = float(ablation.loc[ablation["experiment"].eq("All features"), "f1_macro"].iloc[0])
    ablation["f1_macro_drop_vs_all"] = full_f1 - ablation["f1_macro"]
    save_table(ablation, "feature_group_ablation.csv")

    plot_df = ablation.melt(
        id_vars=["experiment"],
        value_vars=["accuracy", "f1_macro"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 4.8))
    sns.barplot(data=plot_df, x="experiment", y="score", hue="metric", palette=["#4C72B0", "#55A868"])
    plt.ylim(0, 1.02)
    plt.title("Feature Group Ablation")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=18, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_group_ablation.png")
    plt.close()
    return ablation


def calibration_data(y_true: pd.Series, y_prob: np.ndarray, model_name: str, n_bins: int = 10) -> tuple[pd.DataFrame, dict]:
    y = np.asarray(y_true)
    pred = np.asarray(CLASSES)[np.argmax(y_prob, axis=1)]
    confidence = np.max(y_prob, axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not mask.any():
            continue
        rows.append(
            {
                "model": model_name,
                "bin": idx + 1,
                "bin_left": float(left),
                "bin_right": float(right),
                "n": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()),
                "observed_accuracy": float(correct[mask].mean()),
                "abs_calibration_error": float(abs(confidence[mask].mean() - correct[mask].mean())),
            }
        )
    bins_df = pd.DataFrame(rows)
    y_bin = label_binarize(y_true, classes=CLASSES)
    multiclass_brier = float(np.mean(np.sum((y_bin - y_prob) ** 2, axis=1)))
    ece = float((bins_df["n"] * bins_df["abs_calibration_error"]).sum() / bins_df["n"].sum())
    summary = {
        "model": model_name,
        "multiclass_brier": multiclass_brier,
        "expected_calibration_error": ece,
        "mean_confidence": float(confidence.mean()),
        "observed_accuracy": float(correct.mean()),
    }
    return bins_df, summary


def calibration_analysis(
    y_test: pd.Series,
    model_probabilities: dict[str, np.ndarray],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    all_bins = []
    summaries = []
    for model_name, probs in model_probabilities.items():
        bins_df, summary = calibration_data(y_test, probs, model_name)
        all_bins.append(bins_df)
        summaries.append(summary)

    calibration_bins_df = pd.concat(all_bins, ignore_index=True)
    save_table(calibration_bins_df, "calibration_bins.csv")
    summary_df = pd.DataFrame(summaries).sort_values("expected_calibration_error")
    save_table(summary_df, "calibration_summary.csv")

    plt.figure(figsize=(6.8, 5.8))
    for model_name in model_probabilities:
        part = calibration_bins_df[calibration_bins_df["model"].eq(model_name)]
        plt.plot(part["mean_confidence"], part["observed_accuracy"], marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    plt.title("Top-Label Calibration Curves")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration_reliability.png")
    plt.close()

    small_train = stratified_sample(pd.concat([X_train, y_train.rename("Cover_Type")], axis=1), 45_000)
    base_tree = make_tree_model()
    calibrated = CalibratedClassifierCV(base_tree, method="sigmoid", cv=3)
    calibrated.fit(small_train[FEATURE_NAMES], small_train["Cover_Type"])
    cal_probs = calibrated.predict_proba(X_test)
    _, cal_summary = calibration_data(y_test, cal_probs, "Decision Tree, sigmoid calibrated")
    calibrated_df = pd.DataFrame([cal_summary])
    save_table(calibrated_df, "calibrated_tree_summary.csv")
    return summary_df


def terrain_group_metrics(best_model_name: str, y_test: pd.Series, y_pred: np.ndarray, X_test: pd.DataFrame) -> pd.DataFrame:
    df = X_test[["Elevation"]].copy()
    df["Cover_Type"] = y_test.to_numpy()
    df["prediction"] = y_pred
    df["elevation_band"] = pd.qcut(df["Elevation"], q=5, duplicates="drop")
    rows = []
    for band, part in df.groupby("elevation_band", observed=True):
        rows.append(
            {
                "model": best_model_name,
                "elevation_band": str(band),
                "n": int(len(part)),
                "accuracy": float(accuracy_score(part["Cover_Type"], part["prediction"])),
                "f1_macro": float(f1_score(part["Cover_Type"], part["prediction"], average="macro", zero_division=0)),
                "dominant_true_class": int(part["Cover_Type"].mode().iloc[0]),
                "dominant_true_class_name": COVER_TYPE_NAMES[int(part["Cover_Type"].mode().iloc[0])],
            }
        )
    metrics = pd.DataFrame(rows)
    save_table(metrics, "elevation_band_metrics.csv")

    plt.figure(figsize=(8.4, 4.6))
    sns.barplot(data=metrics, x="elevation_band", y="accuracy", color="#8172B3")
    plt.ylim(0, 1.02)
    plt.title("Best Model Accuracy by Elevation Band")
    plt.xlabel("Elevation quantile band")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "elevation_band_accuracy.png")
    plt.close()
    return metrics


def wilderness_group_metrics(best_model_name: str, y_test: pd.Series, y_pred: np.ndarray, X_test: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Wilderness_Area": wilderness_area_labels(X_test),
            "Cover_Type": y_test.to_numpy(),
            "prediction": y_pred,
        }
    )
    rows = []
    for area, part in df.groupby("Wilderness_Area", observed=True):
        dominant_class = int(part["Cover_Type"].mode().iloc[0])
        rows.append(
            {
                "model": best_model_name,
                "Wilderness_Area": int(area),
                "Wilderness_Area_Name": WILDERNESS_NAMES[int(area)],
                "n": int(len(part)),
                "test_share": float(len(part) / len(df)),
                "accuracy": float(accuracy_score(part["Cover_Type"], part["prediction"])),
                "f1_macro": float(f1_score(part["Cover_Type"], part["prediction"], average="macro", zero_division=0)),
                "dominant_true_class": dominant_class,
                "dominant_true_class_name": COVER_TYPE_NAMES[dominant_class],
            }
        )
    metrics = pd.DataFrame(rows).sort_values("Wilderness_Area")
    save_table(metrics, "wilderness_model_metrics.csv")

    plot_df = metrics.melt(
        id_vars=["Wilderness_Area_Name"],
        value_vars=["accuracy", "f1_macro"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(8.8, 4.8))
    sns.barplot(data=plot_df, x="Wilderness_Area_Name", y="score", hue="metric", palette=["#4C72B0", "#55A868"])
    plt.ylim(0, 1.02)
    plt.title("Best Model Performance by Wilderness Area")
    plt.xlabel("Wilderness area")
    plt.ylabel("Score")
    plt.xticks(rotation=12, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wilderness_model_metrics.png")
    plt.close()
    return metrics


def error_confidence_analysis(
    best_model_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    confidence = np.max(y_prob, axis=1)
    error_df = pd.DataFrame(
        {
            "model": best_model_name,
            "Cover_Type": y_test.to_numpy(),
            "Cover_Type_Name": y_test.map(COVER_TYPE_NAMES).to_numpy(),
            "prediction": y_pred,
            "prediction_name": [COVER_TYPE_NAMES[int(x)] for x in y_pred],
            "confidence": confidence,
            "correct": y_pred == y_test.to_numpy(),
        }
    )
    save_table(error_df, "test_prediction_confidence.csv")
    summary = (
        error_df.groupby(["correct"], observed=True)["confidence"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    save_table(summary, "prediction_confidence_summary.csv")

    plt.figure(figsize=(7.4, 4.8))
    sns.boxplot(data=error_df, x="correct", y="confidence", palette=["#C44E52", "#55A868"])
    plt.title("Prediction Confidence for Correct vs Incorrect Test Cases")
    plt.xlabel("Prediction is correct")
    plt.ylabel("Top-class probability")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "error_confidence.png")
    plt.close()

    hard = (
        error_df[~error_df["correct"]]
        .groupby(["Cover_Type", "Cover_Type_Name", "prediction", "prediction_name"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    save_table(hard, "hard_cases_by_true_class.csv")
    return summary


def cross_validation_summary(model_sample: pd.DataFrame) -> pd.DataFrame:
    cv_df = stratified_sample(model_sample, CV_SAMPLE_SIZE)
    X_cv = cv_df[FEATURE_NAMES]
    y_cv = cv_df["Cover_Type"]
    models = {
        "Logistic Regression": make_linear_model(FEATURE_NAMES),
        "Decision Tree": make_tree_model(),
        "Random Forest": make_random_forest(n_estimators=70),
        "HistGradientBoosting": make_hist_gradient_boosting(max_iter=100),
    }
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "AUC_ovr_macro": "roc_auc_ovr",
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for model_name, model in models.items():
        results = cross_validate(model, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": model_name,
                "n_cv_sample": int(len(cv_df)),
                "accuracy_mean": float(results["test_accuracy"].mean()),
                "accuracy_std": float(results["test_accuracy"].std()),
                "f1_macro_mean": float(results["test_f1_macro"].mean()),
                "f1_macro_std": float(results["test_f1_macro"].std()),
                "AUC_ovr_macro_mean": float(results["test_AUC_ovr_macro"].mean()),
                "AUC_ovr_macro_std": float(results["test_AUC_ovr_macro"].std()),
            }
        )
    out = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
    save_table(out, "cross_validation.csv")
    return out


def cached_modeling_outputs() -> dict | None:
    stage = "modeling"
    model_outputs = [
        "model_comparison.csv",
        "train_test_full_metrics.csv",
        "confusion_matrices.csv",
        "roc_curve_points.csv",
        "class_report_best_model.csv",
        "feature_importance_random_forest.csv",
        "feature_importance_top20.csv",
        "permutation_importance.csv",
        "permutation_importance_top15.csv",
        "feature_group_ablation.csv",
        "calibration_bins.csv",
        "calibration_summary.csv",
        "calibrated_tree_summary.csv",
        "elevation_band_metrics.csv",
        "wilderness_model_metrics.csv",
        "test_prediction_confidence.csv",
        "prediction_confidence_summary.csv",
        "hard_cases_by_true_class.csv",
        "cross_validation.csv",
    ]
    model_figures = [
        "confusion_logistic_regression.png",
        "confusion_decision_tree.png",
        "model_comparison_f1_auc.png",
        "roc_ovr_models.png",
        "class_recall_comparison.png",
        "feature_importance_top15.png",
        "permutation_importance_top15.png",
        "feature_group_ablation.png",
        "calibration_reliability.png",
        "elevation_band_accuracy.png",
        "wilderness_model_metrics.png",
        "error_confidence.png",
    ]
    if not cache_ready(stage, model_outputs, model_figures):
        return None

    metrics = pd.read_csv(OUT_DIR / "model_comparison.csv")
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    best_name = str(test_metrics.sort_values(["f1_macro", "AUC_ovr_macro"], ascending=False).iloc[0]["model"])
    STAGE_CACHE_STATUS[stage] = "cache_hit"
    return {
        "model_sample_size": MODEL_SAMPLE_SIZE,
        "train_size": int(MODEL_SAMPLE_SIZE * 0.7),
        "test_size": MODEL_SAMPLE_SIZE - int(MODEL_SAMPLE_SIZE * 0.7),
        "metrics": metrics,
        "best_model_name": best_name,
        "best_model_test_metrics": test_metrics[test_metrics["model"].eq(best_name)].iloc[0].to_dict(),
        "class_report": pd.read_csv(OUT_DIR / "class_report_best_model.csv"),
        "rf_importance": pd.read_csv(OUT_DIR / "feature_importance_random_forest.csv"),
        "ablation": pd.read_csv(OUT_DIR / "feature_group_ablation.csv"),
        "calibration": pd.read_csv(OUT_DIR / "calibration_summary.csv"),
        "terrain_metrics": pd.read_csv(OUT_DIR / "elevation_band_metrics.csv"),
        "wilderness_metrics": pd.read_csv(OUT_DIR / "wilderness_model_metrics.csv"),
        "confidence_summary": pd.read_csv(OUT_DIR / "prediction_confidence_summary.csv"),
        "cv_summary": pd.read_csv(OUT_DIR / "cross_validation.csv"),
    }


def train_and_evaluate_models(df: pd.DataFrame) -> dict:
    stage = "modeling"
    cached = cached_modeling_outputs()
    if cached is not None:
        return cached

    STAGE_CACHE_STATUS[stage] = "computed"
    model_sample = stratified_sample(df, MODEL_SAMPLE_SIZE)
    X = model_sample[FEATURE_NAMES]
    y = model_sample["Cover_Type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_full = df[FEATURE_NAMES]
    y_full = df["Cover_Type"]

    models = {
        "Logistic Regression": make_linear_model(FEATURE_NAMES),
        "Decision Tree": make_tree_model(),
        "Random Forest": make_random_forest(n_estimators=120),
        "HistGradientBoosting": make_hist_gradient_boosting(max_iter=160),
    }
    fitted = {}
    model_rows = []
    confusion = []
    model_probabilities = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted[model_name] = model
        if model_name in {"Logistic Regression", "Decision Tree"}:
            split_data = {
                "train": (X_train, y_train),
                "test": (X_test, y_test),
                "full": (X_full, y_full),
            }
        else:
            split_data = {"test": (X_test, y_test)}

        for split_name, (X_split, y_split) in split_data.items():
            model_rows.append(evaluate_model(model_name, model, X_split, y_split, split_name))
            y_pred = model.predict(X_split)
            confusion.extend(confusion_rows(model_name, split_name, y_split, y_pred))

        model_probabilities[model_name] = predict_proba_aligned(model, X_test)

    metrics_df = pd.DataFrame(model_rows)
    save_table(metrics_df, "model_comparison.csv")
    save_table(metrics_df[metrics_df["model"].isin(["Logistic Regression", "Decision Tree"])], "train_test_full_metrics.csv")
    save_table(pd.DataFrame(confusion), "confusion_matrices.csv")

    plot_confusion_grid(
        "Logistic Regression",
        fitted["Logistic Regression"],
        {"Train": (X_train, y_train), "Test": (X_test, y_test), "Full": (X_full, y_full)},
    )
    plot_confusion_grid(
        "Decision Tree",
        fitted["Decision Tree"],
        {"Train": (X_train, y_train), "Test": (X_test, y_test), "Full": (X_full, y_full)},
    )
    plot_model_comparison(metrics_df)
    plot_roc_curves(y_test, model_probabilities)

    test_metrics = metrics_df[metrics_df["split"].eq("test")].copy()
    best_name = str(test_metrics.sort_values(["f1_macro", "AUC_ovr_macro"], ascending=False).iloc[0]["model"])
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    best_prob = model_probabilities[best_name]

    class_report = class_report_rows(y_test, best_pred, best_name)
    save_table(class_report, "class_report_best_model.csv")
    plot_class_recall(class_report)

    rf_importance = feature_importance_analysis(
        fitted["Random Forest"],
        best_name,
        best_model,
        X_test,
        y_test,
    )
    ablation = feature_group_ablation(X_train, X_test, y_train, y_test)
    calibration = calibration_analysis(y_test, model_probabilities, X_train, y_train, X_test)
    terrain_metrics = terrain_group_metrics(best_name, y_test, best_pred, X_test)
    wilderness_metrics = wilderness_group_metrics(best_name, y_test, best_pred, X_test)
    confidence_summary = error_confidence_analysis(best_name, y_test, best_pred, best_prob)
    cv_summary = cross_validation_summary(model_sample)
    write_cache_manifest(stage)

    return {
        "model_sample_size": int(len(model_sample)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "metrics": metrics_df,
        "best_model_name": best_name,
        "best_model_test_metrics": test_metrics[test_metrics["model"].eq(best_name)].iloc[0].to_dict(),
        "class_report": class_report,
        "rf_importance": rf_importance,
        "ablation": ablation,
        "calibration": calibration,
        "terrain_metrics": terrain_metrics,
        "wilderness_metrics": wilderness_metrics,
        "confidence_summary": confidence_summary,
        "cv_summary": cv_summary,
    }


def descriptive_outputs(df: pd.DataFrame) -> dict:
    missing_ratio = df.isna().mean().sort_values(ascending=False)
    save_table(missing_ratio.rename("missing_ratio").reset_index().rename(columns={"index": "column"}), "missing_ratio.csv")
    target_counts = (
        df["Cover_Type"]
        .value_counts()
        .sort_index()
        .rename_axis("Cover_Type")
        .reset_index(name="count")
    )
    target_counts["Cover_Type_Name"] = target_counts["Cover_Type"].map(COVER_TYPE_NAMES)
    target_counts["proportion"] = target_counts["count"] / len(df)
    save_table(target_counts, "target_counts.csv")

    feature_summary = df[CONTINUOUS_FEATURES].agg(["mean", "std", "min", "median", "max"]).T.reset_index()
    feature_summary = feature_summary.rename(columns={"index": "feature"})
    save_table(feature_summary, "continuous_feature_summary.csv")

    area_summary = pd.DataFrame(
        {
            "feature_group": ["Wilderness areas", "Soil types"],
            "n_indicator_columns": [len(WILDERNESS_FEATURES), len(SOIL_FEATURES)],
            "mean_active_per_row": [
                float(df[WILDERNESS_FEATURES].sum(axis=1).mean()),
                float(df[SOIL_FEATURES].sum(axis=1).mean()),
            ],
        }
    )
    save_table(area_summary, "indicator_feature_summary.csv")

    plot_target_distribution(df["Cover_Type"])
    plot_missing_ratios(missing_ratio)
    wilderness_summary = wilderness_area_outputs(df)
    return {
        "rows": int(len(df)),
        "features": len(FEATURE_NAMES),
        "continuous_features": len(CONTINUOUS_FEATURES),
        "wilderness_indicators": len(WILDERNESS_FEATURES),
        "soil_indicators": len(SOIL_FEATURES),
        "max_missing_ratio": float(missing_ratio.max()),
        "target_counts": target_counts.to_dict(orient="records"),
        "smallest_class": target_counts.sort_values("count").iloc[0].to_dict(),
        "largest_class": target_counts.sort_values("count", ascending=False).iloc[0].to_dict(),
        "wilderness_area_summary": wilderness_summary.to_dict(orient="records"),
    }


def computational_cost_outputs(df: pd.DataFrame) -> dict:
    """Document the scalability choices made for the Covertype workflow."""
    n_rows = len(df)
    n_features = len(FEATURE_NAMES)
    baseline_memory_mb = n_rows * len(ALL_COLUMNS) * np.dtype("int64").itemsize / (1024**2)
    optimized_memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    memory_reduction_pct = 1 - optimized_memory_mb / baseline_memory_mb

    cost_rows = [
        {
            "module": "Data read and preprocessing",
            "scope_used": f"full {n_rows:,} rows x {n_features} features",
            "complexity_driver": "O(n x d)",
            "estimated_cost": f"seconds; optimized dataframe memory about {optimized_memory_mb:.1f} MB",
            "relative_cost_score": 1,
            "optimization_decision": f"implemented dtype downcast from naive {baseline_memory_mb:.1f} MB to {optimized_memory_mb:.1f} MB",
        },
        {
            "module": "Standardization and PCA",
            "scope_used": f"{TSNE_SAMPLE_SIZE:,}-{MODEL_SAMPLE_SIZE:,} stratified rows",
            "complexity_driver": "O(n x d x components)",
            "estimated_cost": "seconds to tens of seconds",
            "relative_cost_score": 2,
            "optimization_decision": "apply PCA before t-SNE and clustering",
        },
        {
            "module": "t-SNE visualization",
            "scope_used": f"{TSNE_SAMPLE_SIZE:,} stratified rows after PCA",
            "complexity_driver": "high neighbor-search and iterative embedding cost",
            "estimated_cost": "tens of seconds to minutes; full-data t-SNE is not practical",
            "relative_cost_score": 9,
            "optimization_decision": "use stratified sample plus PCA; UMAP is a possible faster substitute",
        },
        {
            "module": "MiniBatchKMeans clustering",
            "scope_used": f"{CLUSTER_SAMPLE_SIZE:,} stratified rows after PCA",
            "complexity_driver": "O(n x k x d x iterations)",
            "estimated_cost": "seconds to minutes depending on k search",
            "relative_cost_score": 3,
            "optimization_decision": "implemented MiniBatchKMeans with fixed batch size for scalable clustering",
        },
        {
            "module": "Ward hierarchical clustering",
            "scope_used": f"{AGGLOMERATIVE_SAMPLE_SIZE:,} stratified rows",
            "complexity_driver": "O(n^2) memory and time",
            "estimated_cost": "feasible only on small samples; full data is effectively infeasible",
            "relative_cost_score": 10,
            "optimization_decision": "restrict to small sample or replace with BIRCH/MiniBatchKMeans",
        },
        {
            "module": "Supervised model training",
            "scope_used": f"{MODEL_SAMPLE_SIZE:,} stratified rows, 70/30 split",
            "complexity_driver": "model cost x rows x features",
            "estimated_cost": "minutes for tree ensembles on CPU",
            "relative_cost_score": 6,
            "optimization_decision": "use n_jobs=-1, bounded tree complexity, and efficient boosting",
        },
        {
            "module": "Cross-validation",
            "scope_used": f"{CV_SAMPLE_SIZE:,} stratified rows x 3 folds",
            "complexity_driver": "training cost multiplied by folds",
            "estimated_cost": "minutes; full-data 5-fold CV would be much more expensive",
            "relative_cost_score": 7,
            "optimization_decision": "use 3-fold sample CV; tune on sample, then train final model",
        },
        {
            "module": "Permutation importance",
            "scope_used": f"{PERMUTATION_SAMPLE_SIZE:,} test rows x 54 features x 5 repeats",
            "complexity_driver": "features x repeats x prediction cost",
            "estimated_cost": "expensive but manageable on a small held-out sample",
            "relative_cost_score": 8,
            "optimization_decision": "combine cheap impurity importance with sampled permutation checks",
        },
    ]
    cost_df = pd.DataFrame(cost_rows)
    save_table(cost_df, "computational_cost_estimates.csv")

    optimization_rows = [
        {
            "category": "Data types and I/O",
            "goal": "Reduce memory and repeated file-read overhead",
            "recommended_actions": "Implemented uint8 indicators/target, float32 continuous variables, and cached prepared files",
            "expected_benefit": "Lower memory pressure and faster repeated runs",
        },
        {
            "category": "Sampling strategy",
            "goal": "Make expensive algorithms feasible while preserving class balance",
            "recommended_actions": "Use stratified samples for t-SNE, Ward clustering, CV, and permutation importance",
            "expected_benefit": "Stable estimates without quadratic or repeated full-data costs",
        },
        {
            "category": "Dimensionality reduction",
            "goal": "Reduce distance-computation cost and high-dimensional noise",
            "recommended_actions": "Run PCA to 20-30 components before t-SNE and clustering",
            "expected_benefit": "Faster visualizations and more stable distance-based structure",
        },
        {
            "category": "Scalable clustering",
            "goal": "Avoid O(n^2) algorithms on large n",
            "recommended_actions": "Implemented MiniBatchKMeans for scalable clustering; kept Ward for small structural samples",
            "expected_benefit": "Maintains clustering analysis without infeasible memory growth",
        },
        {
            "category": "Model training",
            "goal": "Improve predictive performance under CPU limits",
            "recommended_actions": "Use parallel Random Forest, HistGradientBoosting, class weights, and bounded tree complexity",
            "expected_benefit": "High accuracy and macro-F1 with controlled runtime",
        },
        {
            "category": "Validation and tuning",
            "goal": "Prevent cross-validation and grid search from dominating runtime",
            "recommended_actions": "Use 3-fold CV, RandomizedSearchCV or halving search on samples, then train final parameters once",
            "expected_benefit": "Comparable model ranking at a fraction of full-grid cost",
        },
        {
            "category": "Interpretability",
            "goal": "Keep explanations useful without exhaustive perturbation",
            "recommended_actions": "Use tree impurity importance first; run permutation importance on a small held-out sample",
            "expected_benefit": "Good feature diagnostics with bounded prediction calls",
        },
        {
            "category": "Caching and reproducibility",
            "goal": "Avoid rerunning deterministic expensive steps during report generation",
            "recommended_actions": "Implemented manifest-based caches for t-SNE, clustering, and modeling outputs",
            "expected_benefit": "Fast report/PPT rebuilds and auditable intermediate artifacts",
        },
    ]
    optimization_df = pd.DataFrame(optimization_rows)
    save_table(optimization_df, "optimization_directions.csv")

    plt.figure(figsize=(9.2, 4.8))
    plot_df = cost_df.sort_values("relative_cost_score")
    sns.barplot(data=plot_df, x="relative_cost_score", y="module", color="#4C72B0")
    plt.title("Relative Computational Cost by Workflow Module")
    plt.xlabel("Relative computational cost score (1=low, 10=highest risk)")
    plt.ylabel("")
    plt.xlim(0, 10.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "computational_cost_tiers.png")
    plt.close()

    return {
        "baseline_memory_mb": float(baseline_memory_mb),
        "optimized_memory_mb": float(optimized_memory_mb),
        "memory_reduction_pct": float(memory_reduction_pct),
        "cost_estimates": cost_df.to_dict(orient="records"),
        "optimization_directions": optimization_df.to_dict(orient="records"),
    }


def main() -> None:
    df = timed_stage("data_read_downcast", read_covertype_data)
    descriptive = timed_stage("descriptive_outputs", descriptive_outputs, df)
    computational = timed_stage("computational_cost_outputs", computational_cost_outputs, df)
    tsne_coords, _ = timed_stage("t-SNE", run_tsne, df)
    clustering = timed_stage("clustering", run_clustering, df, tsne_coords)
    modeling = timed_stage("modeling", train_and_evaluate_models, df)
    runtime_profile = pd.DataFrame(RUNTIME_ROWS)
    save_table(runtime_profile, "runtime_profile.csv")

    summary = {
        "dataset": descriptive,
        "computational_scaling": computational,
        "runtime_profile": runtime_profile.to_dict(orient="records"),
        "sampling": {
            "model_sample_size": MODEL_SAMPLE_SIZE,
            "cv_sample_size": CV_SAMPLE_SIZE,
            "tsne_sample_size": TSNE_SAMPLE_SIZE,
            "cluster_sample_size": CLUSTER_SAMPLE_SIZE,
            "agglomerative_sample_size": AGGLOMERATIVE_SAMPLE_SIZE,
            "reason": "Full Covertype has 581012 rows; expensive visualization and validation steps use stratified samples.",
        },
        "clustering_evaluation": clustering.to_dict(orient="records"),
        "best_model_name": modeling["best_model_name"],
        "best_model_test_metrics": modeling["best_model_test_metrics"],
        "top_random_forest_features": modeling["rf_importance"].head(10).to_dict(orient="records"),
        "feature_group_ablation": modeling["ablation"].to_dict(orient="records"),
        "calibration_summary": modeling["calibration"].to_dict(orient="records"),
        "elevation_band_metrics": modeling["terrain_metrics"].to_dict(orient="records"),
        "wilderness_model_metrics": modeling["wilderness_metrics"].to_dict(orient="records"),
        "confidence_summary": modeling["confidence_summary"].to_dict(orient="records"),
        "cross_validation": modeling["cv_summary"].to_dict(orient="records"),
    }
    save_json(summary)
    print("Analysis complete.")
    print(f"Best model: {modeling['best_model_name']}")
    print(json.dumps(modeling["best_model_test_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
