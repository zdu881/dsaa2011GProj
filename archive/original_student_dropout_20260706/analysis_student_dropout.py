"""Reproducible analysis for the DSAA2011 Student Dropout project.

Run from the repository root:

    python3 scripts/prepare_data.py
    python3 analysis_student_dropout.py

The script writes figures to `figures/`, tabular outputs to `outputs/`, and a
machine-readable summary to `outputs/summary.json`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
DATA_PATH = ROOT / "student_dropout.csv"

FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.family"] = "DejaVu Sans"


CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

TARGET_ORDER = ["Dropout", "Enrolled", "Graduate"]
TARGET_COLORS = {
    "Dropout": "#C44E52",
    "Enrolled": "#DDCC77",
    "Graduate": "#4C72B0",
}


class HighCardinalityGrouper(BaseEstimator, TransformerMixin):
    """Replace rare categories by `Other` when a column has many distinct values."""

    def __init__(
        self,
        categorical_cols: list[str] | None = None,
        threshold: int = 10,
        top_n: int = 5,
        other_label: str = "Other",
    ) -> None:
        self.categorical_cols = categorical_cols
        self.threshold = threshold
        self.top_n = top_n
        self.other_label = other_label

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HighCardinalityGrouper":
        X = pd.DataFrame(X).copy()
        self.categorical_cols_ = [c for c in (self.categorical_cols or []) if c in X.columns]
        self.top_categories_ = {}
        self.high_cardinality_cols_ = []

        for col in self.categorical_cols_:
            s = X[col]
            if s.nunique(dropna=True) > self.threshold:
                top_values = (
                    s.dropna()
                    .astype(str)
                    .value_counts()
                    .nlargest(self.top_n)
                    .index.tolist()
                )
                self.top_categories_[col] = top_values
                self.high_cardinality_cols_.append(col)
            else:
                self.top_categories_[col] = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col in self.categorical_cols_:
            if col not in X.columns:
                continue
            s = X[col]
            missing_mask = s.isna()
            s = s.astype(str)
            s[missing_mask] = np.nan
            top_values = self.top_categories_[col]
            if top_values is not None:
                s = s.where(s.isna() | s.isin(top_values), self.other_label)
            X[col] = s
        return X


def read_student_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the normalized CSV, with a delimiter fallback for raw UCI files."""
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    df.columns = [col.strip() for col in df.columns]
    return df


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    auto_cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    manual_cat = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    categorical_cols = sorted(set(auto_cat + manual_cat))
    numeric_cols = [
        c for c in X.select_dtypes(include=[np.number]).columns.tolist()
        if c not in categorical_cols
    ]
    return numeric_cols, categorical_cols


def make_onehot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_preprocessor(X_reference: pd.DataFrame) -> Pipeline:
    numeric_cols, categorical_cols = infer_feature_types(X_reference)

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("group_high_cardinality", HighCardinalityGrouper(categorical_cols=categorical_cols)),
            ("column_transform", column_transformer),
            ("scale", StandardScaler()),
        ]
    )


def transform_to_frame(preprocessor: Pipeline, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    arr = preprocessor.fit_transform(X) if fit else preprocessor.transform(X)
    names = preprocessor.named_steps["column_transform"].get_feature_names_out()
    return pd.DataFrame(arr, columns=names, index=X.index)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False)


def save_json(obj: dict, name: str = "summary.json") -> None:
    (OUT_DIR / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_target_distribution(y: pd.Series) -> None:
    counts = y.value_counts().reindex(TARGET_ORDER)
    plt.figure(figsize=(6.5, 4.2))
    sns.barplot(x=counts.index, y=counts.values, palette=[TARGET_COLORS[k] for k in counts.index])
    plt.title("Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    for idx, value in enumerate(counts.values):
        plt.text(idx, value + 35, str(int(value)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distribution.png")
    plt.close()


def plot_missing_ratios(missing_ratio: pd.Series) -> None:
    top_missing = missing_ratio.sort_values(ascending=False).head(12)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=top_missing.values, y=top_missing.index, color="#4C72B0")
    plt.title("Top Missing Ratios")
    plt.xlabel("Missing ratio")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missing_ratios.png")
    plt.close()


def plot_confusion_grid(model_name: str, model, datasets: dict[str, tuple[pd.DataFrame, pd.Series]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (split_name, (X_split, y_split)) in zip(axes, datasets.items()):
        y_pred = model.predict(X_split)
        cm = confusion_matrix(y_split, y_pred, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Graduate", "Dropout"],
            yticklabels=["Graduate", "Dropout"],
            ax=ax,
        )
        ax.set_title(f"{split_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(f"{model_name}: Confusion Matrices", y=1.04)
    plt.tight_layout()
    file_slug = model_name.lower().replace(" ", "_").replace(",", "")
    plt.savefig(FIG_DIR / f"confusion_{file_slug}.png", bbox_inches="tight")
    plt.close()


def encoded_to_original_feature(encoded_name: str, categorical_cols: list[str]) -> str:
    for col in sorted(categorical_cols, key=len, reverse=True):
        if encoded_name.startswith(f"{col}_"):
            return col
    return encoded_name


def metrics_from_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    y_pred: np.ndarray | None = None,
) -> dict:
    """Compute binary classification metrics for Dropout as the positive class."""
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    return {
        "threshold": float(threshold),
        "flagged_rate": float(np.mean(y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, y_prob)),
    }


def make_logistic_pipeline(X_reference: pd.DataFrame) -> Pipeline:
    """Create a full raw-data pipeline for logistic regression."""
    return Pipeline(
        [
            ("prep", build_preprocessor(X_reference)),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def evaluate_logistic_feature_set(
    name: str,
    columns: list[str],
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, Pipeline, np.ndarray]:
    """Fit LR on a selected raw-column subset and evaluate on the common test split."""
    X_train_subset = X_train_raw[columns]
    X_test_subset = X_test_raw[columns]
    pipe = make_logistic_pipeline(X_train_subset)
    pipe.fit(X_train_subset, y_train)
    y_prob = pipe.predict_proba(X_test_subset)[:, 1]
    row = metrics_from_predictions(y_test, y_prob, threshold=0.5)
    row.update(
        {
            "experiment": name,
            "n_raw_features": len(columns),
        }
    )
    return row, pipe, y_prob


def top_capacity_predictions(y_prob: np.ndarray, capacity_rate: float) -> tuple[np.ndarray, float]:
    """Flag the highest-risk students under a strict capacity limit."""
    n = len(y_prob)
    k = max(1, int(np.floor(n * capacity_rate)))
    order = np.argsort(-y_prob)
    y_pred = np.zeros(n, dtype=int)
    y_pred[order[:k]] = 1
    cutoff = float(y_prob[order[k - 1]])
    return y_pred, cutoff


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Group raw columns by educational meaning for ablation and timing analysis."""
    first_sem = [c for c in columns if "1st sem" in c]
    second_sem = [c for c in columns if "2nd sem" in c]
    demographics = [
        c
        for c in [
            "Marital status",
            "Nacionality",
            "Nationality",
            "Gender",
            "Age at enrollment",
            "International",
            "Displaced",
            "Educational special needs",
        ]
        if c in columns
    ]
    economic = [
        c
        for c in [
            "Debtor",
            "Tuition fees up to date",
            "Scholarship holder",
            "Unemployment rate",
            "Inflation rate",
            "GDP",
        ]
        if c in columns
    ]
    admission_only = [c for c in columns if c not in set(first_sem + second_sem)]
    first_available = [c for c in columns if c not in set(second_sem)]

    return {
        "admission_only": admission_only,
        "first_semester": first_sem,
        "second_semester": second_sem,
        "demographic": demographics,
        "economic": economic,
        "admission_plus_first_semester": first_available,
        "all": columns,
    }


def plot_metric_comparison(df: pd.DataFrame, x_col: str, title: str, out_name: str) -> None:
    plot_df = df.melt(
        id_vars=[x_col],
        value_vars=["AUC", "recall"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(8.6, 4.5))
    sns.barplot(data=plot_df, x=x_col, y="value", hue="metric", palette=["#4C72B0", "#C44E52"])
    plt.ylim(0, 1.02)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=18, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name)
    plt.close()


def calibration_bins(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build uniform-bin calibration data for probability interpretation."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "model": model_name,
                "bin": idx + 1,
                "bin_left": float(left),
                "bin_right": float(right),
                "n": int(mask.sum()),
                "mean_predicted_risk": float(p[mask].mean()),
                "observed_dropout_rate": float(y[mask].mean()),
                "abs_calibration_error": float(abs(p[mask].mean() - y[mask].mean())),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_bins: int = 10,
) -> dict:
    bins = calibration_bins(y_true, y_prob, model_name, n_bins=n_bins)
    n = bins["n"].sum()
    ece = float((bins["n"] * bins["abs_calibration_error"]).sum() / n)
    return {
        "model": model_name,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error": ece,
        "max_calibration_error": float(bins["abs_calibration_error"].max()),
        "mean_predicted_risk": float(np.mean(y_prob)),
        "observed_dropout_rate": float(np.mean(y_true)),
    }


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return float(max(0, center - half_width)), float(min(1, center + half_width))


def bootstrap_metric_ci(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Bootstrap uncertainty intervals for AUC and recall."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)

    auc_values = []
    recall_values = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        p_b = p[idx]
        pred_b = pred[idx]
        auc_values.append(roc_auc_score(y_b, p_b))
        recall_values.append(recall_score(y_b, pred_b, zero_division=0))

    auc_values = np.asarray(auc_values)
    recall_values = np.asarray(recall_values)
    return {
        "threshold": float(threshold),
        "n_bootstrap_effective": int(len(auc_values)),
        "AUC_point": float(roc_auc_score(y, p)),
        "AUC_ci_low": float(np.quantile(auc_values, 0.025)),
        "AUC_ci_high": float(np.quantile(auc_values, 0.975)),
        "recall_point": float(recall_score(y, pred, zero_division=0)),
        "recall_ci_low": float(np.quantile(recall_values, 0.025)),
        "recall_ci_high": float(np.quantile(recall_values, 0.975)),
    }


def intervention_coverage(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    follow_up_rates: list[float],
) -> pd.DataFrame:
    """Coverage of true Dropout cases when following top-risk students."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    order = np.argsort(-p)
    total_dropout = int(y.sum())
    rows = []
    for rate in follow_up_rates:
        k = max(1, int(np.ceil(len(y) * rate)))
        chosen = order[:k]
        captured = int(y[chosen].sum())
        rows.append(
            {
                "follow_up_rate": float(rate),
                "follow_up_count": int(k),
                "captured_dropout": captured,
                "total_dropout": total_dropout,
                "dropout_coverage": float(captured / total_dropout) if total_dropout else np.nan,
                "precision_among_followed": float(captured / k),
                "lift_vs_random": float((captured / k) / (total_dropout / len(y))) if total_dropout else np.nan,
                "risk_cutoff": float(p[chosen[-1]]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_group_fairness_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    sensitive_feature: str,
    n_bootstrap: int = 1000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap Recall and FPR within each sensitive group."""
    rng = np.random.default_rng(random_state)
    rows = []
    group_values = sorted(pd.Series(groups).dropna().unique().tolist())
    for group_value in group_values:
        mask = pd.Series(groups).eq(group_value).to_numpy()
        y_g = np.asarray(y_true)[mask].astype(int)
        pred_g = np.asarray(y_pred)[mask].astype(int)
        n = len(y_g)
        if n == 0:
            continue
        recalls = []
        fprs = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            y_b = y_g[idx]
            pred_b = pred_g[idx]
            positives = y_b.sum()
            negatives = len(y_b) - positives
            if positives > 0:
                recalls.append(recall_score(y_b, pred_b, zero_division=0))
            if negatives > 0:
                fp = ((pred_b == 1) & (y_b == 0)).sum()
                fprs.append(fp / negatives)
        recalls = np.asarray(recalls)
        fprs = np.asarray(fprs)
        rows.append(
            {
                "sensitive_feature": sensitive_feature,
                "group_value": str(group_value),
                "n": int(n),
                "recall_bootstrap_mean": float(recalls.mean()) if len(recalls) else np.nan,
                "recall_ci_low": float(np.quantile(recalls, 0.025)) if len(recalls) else np.nan,
                "recall_ci_high": float(np.quantile(recalls, 0.975)) if len(recalls) else np.nan,
                "FPR_bootstrap_mean": float(fprs.mean()) if len(fprs) else np.nan,
                "FPR_ci_low": float(np.quantile(fprs, 0.025)) if len(fprs) else np.nan,
                "FPR_ci_high": float(np.quantile(fprs, 0.975)) if len(fprs) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def manual_partial_dependence(
    model: Pipeline,
    X_reference: pd.DataFrame,
    feature: str,
    grid_values: list[float] | np.ndarray,
) -> pd.DataFrame:
    rows = []
    for value in grid_values:
        X_mod = X_reference.copy()
        X_mod[feature] = value
        probs = model.predict_proba(X_mod)[:, 1]
        rows.append(
            {
                "feature": feature,
                "value": float(value),
                "mean_predicted_dropout_risk": float(np.mean(probs)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = read_student_data()
    if "Target" not in df.columns:
        raise ValueError("Dataset must include a Target column.")

    raw_shape = list(df.shape)
    df = df.dropna(subset=["Target"]).copy()
    df["Target"] = df["Target"].astype(str)

    missing_ratio = df.isna().mean().sort_values(ascending=False)
    drop_cols = [col for col, ratio in missing_ratio.items() if ratio > 0.40 and col != "Target"]
    df_work = df.drop(columns=drop_cols).copy()

    X_raw = df_work.drop(columns=["Target"])
    y_full = df_work["Target"]
    numeric_cols, categorical_cols = infer_feature_types(X_raw)

    full_preprocessor = build_preprocessor(X_raw)
    X_processed = transform_to_frame(full_preprocessor, X_raw, fit=True)
    group_step = full_preprocessor.named_steps["group_high_cardinality"]

    preprocessing_summary = {
        "raw_shape": raw_shape,
        "after_drop_shape": list(df_work.shape),
        "feature_shape_before_preprocessing": list(X_raw.shape),
        "feature_shape_after_preprocessing": list(X_processed.shape),
        "dropped_columns_missing_gt_40pct": drop_cols,
        "max_missing_ratio": float(missing_ratio.max()),
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "high_cardinality_columns_grouped": group_step.high_cardinality_cols_,
    }

    missing_ratio_table = missing_ratio.rename("missing_ratio").reset_index()
    missing_ratio_table.columns = ["column", "missing_ratio"]
    missing_ratio_table.to_csv(OUT_DIR / "missing_ratio.csv", index=False)
    plot_missing_ratios(missing_ratio)
    plot_target_distribution(y_full)

    target_counts = y_full.value_counts().reindex(TARGET_ORDER).fillna(0).astype(int)
    target_counts_table = target_counts.rename("count").reset_index()
    target_counts_table.columns = ["Target", "count"]
    save_table(target_counts_table, "target_counts.csv")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="pca",
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    X_tsne = tsne.fit_transform(X_processed)
    tsne_df = pd.DataFrame(
        {
            "TSNE-1": X_tsne[:, 0],
            "TSNE-2": X_tsne[:, 1],
            "Target": y_full.values,
        }
    )
    save_table(tsne_df, "tsne_coordinates.csv")

    plt.figure(figsize=(7.5, 5.8))
    sns.scatterplot(
        data=tsne_df,
        x="TSNE-1",
        y="TSNE-2",
        hue="Target",
        hue_order=TARGET_ORDER,
        palette=TARGET_COLORS,
        s=28,
        alpha=0.78,
        linewidth=0,
    )
    plt.title("t-SNE Projection by Student Outcome")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tsne_target.png")
    plt.close()

    k_range = range(2, 9)
    kmeans_rows = []
    kmeans_labels = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_processed)
        kmeans_labels[k] = labels
        kmeans_rows.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": float(silhouette_score(X_processed, labels)),
            }
        )

    kmeans_table = pd.DataFrame(kmeans_rows)
    best_k_kmeans = int(kmeans_table.loc[kmeans_table["silhouette"].idxmax(), "k"])
    save_table(kmeans_table, "kmeans_k_selection.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    sns.lineplot(data=kmeans_table, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("K-Means Elbow Method")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    sns.lineplot(data=kmeans_table, x="k", y="silhouette", marker="o", ax=axes[1])
    axes[1].axvline(best_k_kmeans, color="#C44E52", linestyle="--", linewidth=1)
    axes[1].set_title("K-Means Silhouette Score")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kmeans_elbow_silhouette.png")
    plt.close()

    dendro_sample = X_processed.sample(n=min(800, len(X_processed)), random_state=RANDOM_STATE)
    Z = linkage(dendro_sample, method="ward")
    plt.figure(figsize=(11, 4.5))
    dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True)
    plt.title("Hierarchical Clustering Dendrogram, Ward Linkage")
    plt.xlabel("Compressed cluster leaves")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hierarchical_dendrogram.png")
    plt.close()

    hier_rows = []
    hier_labels = {}
    for k in k_range:
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = hc.fit_predict(X_processed)
        hier_labels[k] = labels
        hier_rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(X_processed, labels)),
            }
        )

    hier_table = pd.DataFrame(hier_rows)
    best_k_hier = int(hier_table.loc[hier_table["silhouette"].idxmax(), "k"])
    save_table(hier_table, "hierarchical_k_selection.csv")

    def cluster_eval_row(name: str, k: int, labels: np.ndarray) -> dict:
        return {
            "algorithm": name,
            "k": int(k),
            "silhouette": float(silhouette_score(X_processed, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X_processed, labels)),
            "ARI_vs_Target": float(adjusted_rand_score(y_full, labels)),
        }

    cluster_eval = pd.DataFrame(
        [
            cluster_eval_row("K-Means", best_k_kmeans, kmeans_labels[best_k_kmeans]),
            cluster_eval_row("Agglomerative Ward", best_k_hier, hier_labels[best_k_hier]),
        ]
    )
    save_table(cluster_eval, "clustering_evaluation.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.scatterplot(
        x=tsne_df["TSNE-1"],
        y=tsne_df["TSNE-2"],
        hue=kmeans_labels[best_k_kmeans],
        palette="tab10",
        s=24,
        linewidth=0,
        alpha=0.78,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title(f"K-Means Clusters, K={best_k_kmeans}")
    sns.scatterplot(
        x=tsne_df["TSNE-1"],
        y=tsne_df["TSNE-2"],
        hue=hier_labels[best_k_hier],
        palette="tab10",
        s=24,
        linewidth=0,
        alpha=0.78,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title(f"Hierarchical Ward Clusters, K={best_k_hier}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "clustering_tsne.png")
    plt.close()

    binary_df = df_work[df_work["Target"].isin(["Dropout", "Graduate"])].copy()
    X_binary_raw = binary_df.drop(columns=["Target"])
    y_binary = (binary_df["Target"] == "Dropout").astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_binary_raw,
        y_binary,
        test_size=0.30,
        stratify=y_binary,
        random_state=RANDOM_STATE,
    )

    supervised_preprocessor = build_preprocessor(X_train_raw)
    X_train = transform_to_frame(supervised_preprocessor, X_train_raw, fit=True)
    X_test = transform_to_frame(supervised_preprocessor, X_test_raw, fit=False)
    X_binary = transform_to_frame(supervised_preprocessor, X_binary_raw, fit=False)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    }

    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

    datasets = {
        "Train": (X_train, y_train),
        "Test": (X_test, y_test),
        "Full": (X_binary, y_binary),
    }

    confusion_rows = []
    for model_name, model in fitted_models.items():
        plot_confusion_grid(model_name, model, datasets)
        for split_name, (X_split, y_split) in datasets.items():
            y_pred = model.predict(X_split)
            cm = confusion_matrix(y_split, y_pred, labels=[0, 1])
            confusion_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "tn_graduate": int(cm[0, 0]),
                    "fp_dropout": int(cm[0, 1]),
                    "fn_dropout": int(cm[1, 0]),
                    "tp_dropout": int(cm[1, 1]),
                }
            )
    confusion_table = pd.DataFrame(confusion_rows)
    save_table(confusion_table, "confusion_matrices.csv")

    metric_rows = []
    roc_rows = []
    plt.figure(figsize=(6.6, 5.4))
    for model_name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        metric_rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "AUC": float(auc),
            }
        )
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_rows.extend(
            {"model": model_name, "fpr": float(f), "tpr": float(t)}
            for f, t in zip(fpr, tpr)
        )
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name}, AUC={auc:.3f}")

    metrics_table = pd.DataFrame(metric_rows)
    save_table(metrics_table, "test_metrics_lr_tree.csv")
    save_table(pd.DataFrame(roc_rows), "roc_curve_points.csv")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Dropout Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_lr_tree.png")
    plt.close()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for model_name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X_binary_raw)), ("model", clone(model))])
        cv_acc = cross_val_score(pipe, X_binary_raw, y_binary, cv=cv, scoring="accuracy")
        cv_auc = cross_val_score(pipe, X_binary_raw, y_binary, cv=cv, scoring="roc_auc")
        train_acc = accuracy_score(y_train, fitted_models[model_name].predict(X_train))
        test_acc = accuracy_score(y_test, fitted_models[model_name].predict(X_test))
        cv_rows.append(
            {
                "model": model_name,
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "cv_accuracy_mean": float(cv_acc.mean()),
                "cv_accuracy_std": float(cv_acc.std()),
                "cv_auc_mean": float(cv_auc.mean()),
                "cv_auc_std": float(cv_auc.std()),
                "train_test_gap": float(train_acc - test_acc),
            }
        )
    cv_table = pd.DataFrame(cv_rows)
    save_table(cv_table, "cross_validation.csv")

    rf_feature_pipe = Pipeline(
        [
            ("prep", build_preprocessor(X_train_raw)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_feature_pipe.fit(X_train_raw, y_train)
    feature_names = (
        rf_feature_pipe.named_steps["prep"]
        .named_steps["column_transform"]
        .get_feature_names_out()
    )
    _, supervised_categorical_cols = infer_feature_types(X_train_raw)
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_feature_pipe.named_steps["rf"].feature_importances_,
        }
    )
    feature_importance["original_feature"] = feature_importance["feature"].apply(
        lambda name: encoded_to_original_feature(name, supervised_categorical_cols)
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    top5_encoded = feature_importance.head(5).copy()
    top5_original = (
        feature_importance.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .head(5)
    )
    save_table(feature_importance, "feature_importance_encoded.csv")
    save_table(top5_original, "feature_importance_top5_original.csv")

    plt.figure(figsize=(8.4, 4.4))
    sns.barplot(data=top5_original, x="importance", y="original_feature", color="#4C72B0")
    plt.title("Top 5 Factors Associated with Dropout")
    plt.xlabel("Random Forest importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_top5.png")
    plt.close()

    rf_pipe = Pipeline(
        [
            ("prep", build_preprocessor(X_binary_raw)),
            (
                "rf",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    param_grid = {
        "rf__n_estimators": [100, 300],
        "rf__max_depth": [None, 5, 10],
        "rf__min_samples_leaf": [1, 5],
    }
    grid = GridSearchCV(
        estimator=rf_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train_raw, y_train)
    best_rf = grid.best_estimator_
    rf_pred = best_rf.predict(X_test_raw)
    rf_prob = best_rf.predict_proba(X_test_raw)[:, 1]
    rf_row = {
        "model": "Random Forest, tuned",
        "accuracy": float(accuracy_score(y_test, rf_pred)),
        "precision": float(precision_score(y_test, rf_pred, zero_division=0)),
        "recall": float(recall_score(y_test, rf_pred, zero_division=0)),
        "f1": float(f1_score(y_test, rf_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_test, rf_prob)),
    }
    model_comparison = (
        pd.concat([metrics_table, pd.DataFrame([rf_row])], ignore_index=True)
        .sort_values("AUC", ascending=False)
        .reset_index(drop=True)
    )
    save_table(pd.DataFrame(grid.cv_results_), "random_forest_gridsearch_cv_results.csv")
    save_table(model_comparison, "model_comparison.csv")

    plt.figure(figsize=(7.3, 4.2))
    sns.barplot(data=model_comparison, x="AUC", y="model", color="#55A868")
    plt.xlim(0.75, 1.0)
    plt.title("Model Comparison by Test AUC")
    plt.xlabel("Test AUC")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison_auc.png")
    plt.close()

    class_counts = y_binary.value_counts().sort_index()
    minority_majority_ratio = float(class_counts.min() / class_counts.max())
    smote_summary = pd.DataFrame()
    if minority_majority_ratio < 0.80:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        smote_rows = []
        for model_name, base_model in models.items():
            smote_model = clone(base_model)
            smote_model.fit(X_train_smote, y_train_smote)
            y_pred_smote = smote_model.predict(X_test)
            y_prob_smote = smote_model.predict_proba(X_test)[:, 1]
            smote_auc = roc_auc_score(y_test, y_prob_smote)
            base_auc = metrics_table.loc[metrics_table["model"] == model_name, "AUC"].iloc[0]
            smote_rows.append(
                {
                    "model": f"{model_name} + SMOTE",
                    "accuracy": float(accuracy_score(y_test, y_pred_smote)),
                    "precision": float(precision_score(y_test, y_pred_smote, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred_smote, zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred_smote, zero_division=0)),
                    "AUC": float(smote_auc),
                    "AUC_delta_vs_baseline": float(smote_auc - base_auc),
                }
            )
        smote_summary = pd.DataFrame(smote_rows)
        save_table(smote_summary, "smote_summary.csv")

    raw_columns = X_binary_raw.columns.tolist()
    groups = feature_groups(raw_columns)

    early_rows = []
    early_pipelines = {}
    early_probs = {}
    early_specs = [
        ("Enrollment only", groups["admission_only"]),
        ("Enrollment + 1st semester", groups["admission_plus_first_semester"]),
        ("Enrollment + 1st + 2nd semester", groups["all"]),
    ]
    for scenario_name, columns in early_specs:
        row, pipe, prob = evaluate_logistic_feature_set(
            scenario_name,
            columns,
            X_train_raw,
            X_test_raw,
            y_train,
            y_test,
        )
        early_rows.append(row)
        early_pipelines[scenario_name] = pipe
        early_probs[scenario_name] = prob

    early_warning_table = pd.DataFrame(early_rows)
    early_warning_table["AUC_gain_vs_enrollment_only"] = (
        early_warning_table["AUC"] - early_warning_table["AUC"].iloc[0]
    )
    early_warning_table["recall_gain_vs_enrollment_only"] = (
        early_warning_table["recall"] - early_warning_table["recall"].iloc[0]
    )
    save_table(early_warning_table, "early_warning_comparison.csv")
    plot_metric_comparison(
        early_warning_table,
        "experiment",
        "Early Warning: Available Information vs Performance",
        "early_warning_comparison.png",
    )

    full_feature_eval = early_warning_table.loc[
        early_warning_table["experiment"].eq("Enrollment + 1st + 2nd semester")
    ].iloc[0]
    ablation_specs = [
        ("Full features", []),
        ("Remove demographics", groups["demographic"]),
        ("Remove economic status", groups["economic"]),
        ("Remove 1st semester", groups["first_semester"]),
        ("Remove 2nd semester", groups["second_semester"]),
        ("Remove both semesters", groups["first_semester"] + groups["second_semester"]),
    ]
    ablation_rows = []
    for experiment_name, remove_cols in ablation_specs:
        keep_cols = [c for c in raw_columns if c not in set(remove_cols)]
        row, _, _ = evaluate_logistic_feature_set(
            experiment_name,
            keep_cols,
            X_train_raw,
            X_test_raw,
            y_train,
            y_test,
        )
        row["removed_raw_features"] = len(remove_cols)
        row["AUC_drop_vs_full"] = float(full_feature_eval["AUC"] - row["AUC"])
        row["recall_drop_vs_full"] = float(full_feature_eval["recall"] - row["recall"])
        ablation_rows.append(row)

    ablation_table = pd.DataFrame(ablation_rows)
    save_table(ablation_table, "feature_ablation.csv")
    ablation_plot = ablation_table[ablation_table["experiment"] != "Full features"].copy()
    ablation_plot = ablation_plot.melt(
        id_vars=["experiment"],
        value_vars=["AUC_drop_vs_full", "recall_drop_vs_full"],
        var_name="drop_metric",
        value_name="drop",
    )
    plt.figure(figsize=(8.8, 4.7))
    sns.barplot(data=ablation_plot, x="experiment", y="drop", hue="drop_metric", palette=["#4C72B0", "#C44E52"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Feature Ablation: Performance Drop After Removing Feature Groups")
    plt.xlabel("")
    plt.ylabel("Drop relative to full feature model")
    plt.xticks(rotation=18, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_ablation.png")
    plt.close()

    lr_test_prob = fitted_models["Logistic Regression"].predict_proba(X_test)[:, 1]
    threshold_rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        row = metrics_from_predictions(y_test, lr_test_prob, threshold=float(threshold))
        row["strategy"] = f"threshold_{threshold:.2f}"
        threshold_rows.append(row)

    capacity_rate = 0.25
    capacity_pred, capacity_cutoff = top_capacity_predictions(lr_test_prob, capacity_rate)
    capacity_row = metrics_from_predictions(
        y_test,
        lr_test_prob,
        threshold=capacity_cutoff,
        y_pred=capacity_pred,
    )
    capacity_row["strategy"] = "top_25pct_capacity"
    threshold_rows.append(capacity_row)

    threshold_table = pd.DataFrame(threshold_rows)
    save_table(threshold_table, "threshold_tuning.csv")

    threshold_plot = threshold_table[threshold_table["strategy"].str.startswith("threshold_")].copy()
    plt.figure(figsize=(8.4, 4.6))
    for metric, color in [("precision", "#4C72B0"), ("recall", "#C44E52"), ("f1", "#55A868")]:
        sns.lineplot(data=threshold_plot, x="threshold", y=metric, marker="o", color=color, label=metric)
    plt.axvline(0.50, color="black", linestyle="--", linewidth=1, label="default 0.50")
    plt.axvline(capacity_cutoff, color="#C99A2E", linestyle="--", linewidth=1, label="top 25% cutoff")
    plt.title("Threshold Tuning for Dropout Intervention")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "threshold_tuning.png")
    plt.close()

    follow_up_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    coverage_full = intervention_coverage(y_test, lr_test_prob, follow_up_rates)
    coverage_full["model"] = "Full LR, enrollment + 1st + 2nd semester"
    coverage_early = intervention_coverage(
        y_test,
        early_probs["Enrollment + 1st semester"],
        follow_up_rates,
    )
    coverage_early["model"] = "Actionable LR, enrollment + 1st semester"
    intervention_coverage_table = pd.concat([coverage_early, coverage_full], ignore_index=True)
    save_table(intervention_coverage_table, "intervention_coverage_curve.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    sns.lineplot(
        data=intervention_coverage_table,
        x="follow_up_rate",
        y="dropout_coverage",
        hue="model",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Dropout Coverage by Follow-up Capacity")
    axes[0].set_xlabel("Share of students followed up")
    axes[0].set_ylabel("Share of true Dropout captured")
    axes[0].set_ylim(0, 1.02)
    sns.lineplot(
        data=intervention_coverage_table,
        x="follow_up_rate",
        y="precision_among_followed",
        hue="model",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Precision Among Followed Students")
    axes[1].set_xlabel("Share of students followed up")
    axes[1].set_ylabel("Dropout rate in followed group")
    axes[1].set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "intervention_coverage_curve.png")
    plt.close()

    prediction_time_table = early_warning_table[
        ["experiment", "n_raw_features", "AUC", "recall", "precision", "f1"]
    ].copy()
    prediction_time_table["interpretation"] = [
        "Pre-enrollment screening; useful for planning, weak for precise intervention.",
        "Actionable early warning; available after first-semester outcomes.",
        "Retrospective diagnosis; strongest accuracy but may be late for prevention.",
    ]
    prediction_time_table["leakage_risk_for_early_intervention"] = [
        "Low",
        "Low",
        "High if claimed as early warning",
    ]
    save_table(prediction_time_table, "prediction_time_discussion.csv")

    perm = permutation_importance(
        best_rf,
        X_test_raw,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    permutation_table = pd.DataFrame(
        {
            "feature": X_test_raw.columns,
            "importance_mean_auc_drop": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean_auc_drop", ascending=False)
    permutation_top10 = permutation_table.head(10).copy()
    save_table(permutation_table, "permutation_importance.csv")
    save_table(permutation_top10, "permutation_importance_top10.csv")

    plt.figure(figsize=(8.4, 4.9))
    sns.barplot(
        data=permutation_top10,
        x="importance_mean_auc_drop",
        y="feature",
        color="#4C72B0",
        xerr=permutation_top10["importance_std"].to_numpy(),
    )
    plt.title("Permutation Importance, Tuned Random Forest")
    plt.xlabel("Mean AUC drop after permutation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "permutation_importance_top10.png")
    plt.close()

    sensitive_cols = [c for c in ["Gender", "International", "Scholarship holder"] if c in X_test_raw.columns]
    fairness_rows = []
    lr_test_pred = fitted_models["Logistic Regression"].predict(X_test)
    y_test_array = np.asarray(y_test)
    for sensitive_col in sensitive_cols:
        values = sorted(X_test_raw[sensitive_col].dropna().unique().tolist())
        for value in values:
            mask = X_test_raw[sensitive_col].eq(value).to_numpy()
            if mask.sum() == 0:
                continue
            y_group = y_test_array[mask]
            pred_group = lr_test_pred[mask]
            prob_group = lr_test_prob[mask]
            positives = int(y_group.sum())
            negatives = int(len(y_group) - positives)
            tp = int(((pred_group == 1) & (y_group == 1)).sum())
            fp = int(((pred_group == 1) & (y_group == 0)).sum())
            fn = int(((pred_group == 0) & (y_group == 1)).sum())
            tn = int(((pred_group == 0) & (y_group == 0)).sum())
            recall_low, recall_high = wilson_ci(tp, positives)
            fpr_low, fpr_high = wilson_ci(fp, negatives)
            obs_low, obs_high = wilson_ci(positives, int(mask.sum()))
            observed_dropout_rate = float(y_group.mean())
            mean_predicted_risk = float(prob_group.mean())
            fairness_rows.append(
                {
                    "sensitive_feature": sensitive_col,
                    "group_value": str(value),
                    "n": int(mask.sum()),
                    "positives_dropout": positives,
                    "negatives_graduate": negatives,
                    "recall": float(tp / positives) if positives else np.nan,
                    "recall_ci_low": recall_low,
                    "recall_ci_high": recall_high,
                    "FPR": float(fp / negatives) if negatives else np.nan,
                    "FPR_ci_low": fpr_low,
                    "FPR_ci_high": fpr_high,
                    "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                    "observed_dropout_rate": observed_dropout_rate,
                    "observed_dropout_rate_ci_low": obs_low,
                    "observed_dropout_rate_ci_high": obs_high,
                    "mean_predicted_risk": mean_predicted_risk,
                    "calibration_gap_pred_minus_observed": mean_predicted_risk - observed_dropout_rate,
                    "brier_score": float(brier_score_loss(y_group, prob_group)),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
            )

    fairness_table = pd.DataFrame(fairness_rows)
    save_table(fairness_table, "fairness_by_group.csv")
    group_calibration_table = fairness_table[
        [
            "sensitive_feature",
            "group_value",
            "n",
            "observed_dropout_rate",
            "observed_dropout_rate_ci_low",
            "observed_dropout_rate_ci_high",
            "mean_predicted_risk",
            "calibration_gap_pred_minus_observed",
            "brier_score",
        ]
    ].copy()
    save_table(group_calibration_table, "group_calibration_summary.csv")
    fairness_plot = fairness_table.melt(
        id_vars=["sensitive_feature", "group_value"],
        value_vars=["recall", "FPR"],
        var_name="metric",
        value_name="value",
    )
    fairness_plot["group"] = fairness_plot["sensitive_feature"] + "=" + fairness_plot["group_value"]
    plt.figure(figsize=(8.8, 4.6))
    sns.barplot(data=fairness_plot, x="group", y="value", hue="metric", palette=["#C44E52", "#4C72B0"])
    plt.ylim(0, 1.02)
    plt.title("Fairness Check: Recall and False Positive Rate by Group")
    plt.xlabel("")
    plt.ylabel("Rate")
    plt.xticks(rotation=18, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fairness_group_metrics.png")
    plt.close()

    group_cal_plot = group_calibration_table.melt(
        id_vars=["sensitive_feature", "group_value"],
        value_vars=["observed_dropout_rate", "mean_predicted_risk"],
        var_name="calibration_metric",
        value_name="value",
    )
    group_cal_plot["group"] = group_cal_plot["sensitive_feature"] + "=" + group_cal_plot["group_value"]
    plt.figure(figsize=(8.8, 4.6))
    sns.barplot(
        data=group_cal_plot,
        x="group",
        y="value",
        hue="calibration_metric",
        palette=["#C44E52", "#4C72B0"],
    )
    plt.ylim(0, 1.02)
    plt.title("Group-wise Calibration: Predicted vs Observed Dropout Risk")
    plt.xlabel("")
    plt.ylabel("Rate")
    plt.xticks(rotation=18, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "group_calibration.png")
    plt.close()

    fairness_bootstrap_frames = []
    for offset, sensitive_col in enumerate(sensitive_cols):
        fairness_bootstrap_frames.append(
            bootstrap_group_fairness_ci(
                y_test_array,
                lr_test_pred,
                X_test_raw[sensitive_col],
                sensitive_col,
                n_bootstrap=1000,
                random_state=RANDOM_STATE + 100 + offset,
            )
        )
    fairness_bootstrap_table = pd.concat(fairness_bootstrap_frames, ignore_index=True)
    save_table(fairness_bootstrap_table, "fairness_bootstrap_ci.csv")

    fairness_bootstrap_plot = fairness_bootstrap_table.copy()
    fairness_bootstrap_plot["group"] = (
        fairness_bootstrap_plot["sensitive_feature"] + "=" + fairness_bootstrap_plot["group_value"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for ax, point_col, low_col, high_col, title, xlabel in [
        (
            axes[0],
            "recall_bootstrap_mean",
            "recall_ci_low",
            "recall_ci_high",
            "Recall with bootstrap 95% CI",
            "Recall",
        ),
        (
            axes[1],
            "FPR_bootstrap_mean",
            "FPR_ci_low",
            "FPR_ci_high",
            "FPR with bootstrap 95% CI",
            "False positive rate",
        ),
    ]:
        y_pos = np.arange(len(fairness_bootstrap_plot))
        points = fairness_bootstrap_plot[point_col].to_numpy()
        lows = fairness_bootstrap_plot[low_col].to_numpy()
        highs = fairness_bootstrap_plot[high_col].to_numpy()
        ax.errorbar(points, y_pos, xerr=[points - lows, highs - points], fmt="o", capsize=4, color="#4C72B0")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fairness_bootstrap_plot["group"])
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fairness_bootstrap_ci.png")
    plt.close()

    enrolled_df = df_work[df_work["Target"].eq("Enrolled")].copy()
    enrolled_feature_cols = groups["admission_plus_first_semester"]
    enrolled_scores = early_pipelines["Enrollment + 1st semester"].predict_proba(
        enrolled_df[enrolled_feature_cols]
    )[:, 1]
    early_test_prob = early_probs["Enrollment + 1st semester"]
    _, early_capacity_cutoff = top_capacity_predictions(early_test_prob, capacity_rate)
    enrolled_risk = pd.DataFrame(
        {
            "row_index": enrolled_df.index,
            "dropout_risk_score": enrolled_scores,
            "risk_bucket": pd.cut(
                enrolled_scores,
                bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
                labels=["low_<0.25", "medium_0.25-0.50", "high_0.50-0.75", "very_high_>=0.75"],
            ).astype(str),
        }
    )
    enrolled_risk_summary = pd.DataFrame(
        [
            {
                "n_enrolled": int(len(enrolled_scores)),
                "mean_risk": float(np.mean(enrolled_scores)),
                "median_risk": float(np.median(enrolled_scores)),
                "q25_risk": float(np.quantile(enrolled_scores, 0.25)),
                "q75_risk": float(np.quantile(enrolled_scores, 0.75)),
                "risk_ge_0.50_count": int((enrolled_scores >= 0.50).sum()),
                "risk_ge_0.50_rate": float((enrolled_scores >= 0.50).mean()),
                "early_model_top25_cutoff": early_capacity_cutoff,
                "risk_ge_early_top25_cutoff_count": int((enrolled_scores >= early_capacity_cutoff).sum()),
                "risk_ge_early_top25_cutoff_rate": float((enrolled_scores >= early_capacity_cutoff).mean()),
            }
        ]
    )
    bucket_counts = enrolled_risk["risk_bucket"].value_counts().rename_axis("risk_bucket").reset_index(name="count")
    save_table(enrolled_risk, "enrolled_risk_scores.csv")
    save_table(enrolled_risk_summary, "enrolled_risk_summary.csv")
    save_table(bucket_counts, "enrolled_risk_buckets.csv")

    risk_plot_df = pd.DataFrame(
        {
            "dropout_risk_score": np.concatenate(
                [
                    early_test_prob[np.asarray(y_test) == 0],
                    early_test_prob[np.asarray(y_test) == 1],
                    enrolled_scores,
                ]
            ),
            "group": (
                ["Graduate test"] * int((np.asarray(y_test) == 0).sum())
                + ["Dropout test"] * int((np.asarray(y_test) == 1).sum())
                + ["Enrolled scored"] * len(enrolled_scores)
            ),
        }
    )
    plt.figure(figsize=(8.6, 4.7))
    sns.histplot(
        data=risk_plot_df,
        x="dropout_risk_score",
        hue="group",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        palette={
            "Graduate test": "#4C72B0",
            "Dropout test": "#C44E52",
            "Enrolled scored": "#C99A2E",
        },
    )
    plt.axvline(0.50, color="black", linestyle="--", linewidth=1, label="0.50 threshold")
    plt.title("Enrolled Students Reused as Risk-scoring Cases")
    plt.xlabel("Predicted Dropout Risk, Enrollment + 1st Semester Model")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "enrolled_risk_distribution.png")
    plt.close()

    tree_test_prob = fitted_models["Decision Tree"].predict_proba(X_test)[:, 1]
    calibrated_rows = []
    calibrated_curve_frames = []
    calibrated_specs = [
        ("Logistic Regression, uncalibrated", None, LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ("Logistic Regression, Platt", "sigmoid", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ("Logistic Regression, isotonic", "isotonic", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ("Decision Tree, uncalibrated", None, DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ("Decision Tree, Platt", "sigmoid", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ("Decision Tree, isotonic", "isotonic", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
    ]
    for model_name, method, estimator in calibrated_specs:
        base_pipe = Pipeline([("prep", build_preprocessor(X_train_raw)), ("model", estimator)])
        if method is None:
            model_for_eval = base_pipe
        else:
            model_for_eval = CalibratedClassifierCV(
                estimator=base_pipe,
                method=method,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            )
        model_for_eval.fit(X_train_raw, y_train)
        y_prob_cal = model_for_eval.predict_proba(X_test_raw)[:, 1]
        summary_row = calibration_summary(y_test, y_prob_cal, model_name, n_bins=10)
        summary_row.update(
            {
                "AUC": float(roc_auc_score(y_test, y_prob_cal)),
                "calibration_method": "none" if method is None else method,
            }
        )
        calibrated_rows.append(summary_row)
        calibrated_curve_frames.append(calibration_bins(y_test, y_prob_cal, model_name, n_bins=10))

    calibrated_model_table = pd.DataFrame(calibrated_rows)
    save_table(calibrated_model_table, "calibrated_model_comparison.csv")
    calibrated_curve_table = pd.concat(calibrated_curve_frames, ignore_index=True)
    save_table(calibrated_curve_table, "calibrated_model_curves.csv")

    plt.figure(figsize=(7.2, 5.8))
    for model_name in [
        "Logistic Regression, uncalibrated",
        "Logistic Regression, Platt",
        "Logistic Regression, isotonic",
    ]:
        model_bins = calibrated_curve_table[calibrated_curve_table["model"].eq(model_name)]
        sns.lineplot(
            data=model_bins,
            x="mean_predicted_risk",
            y="observed_dropout_rate",
            marker="o",
            label=model_name.replace("Logistic Regression, ", "LR "),
        )
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    plt.title("Calibration After Platt / Isotonic Scaling")
    plt.xlabel("Mean predicted dropout risk")
    plt.ylabel("Observed dropout rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "calibrated_model_curves.png")
    plt.close()

    calibration_compare_plot = calibrated_model_table.melt(
        id_vars=["model", "calibration_method"],
        value_vars=["brier_score", "expected_calibration_error", "AUC"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(9.0, 4.8))
    sns.barplot(data=calibration_compare_plot, x="model", y="value", hue="metric")
    plt.title("Calibrated Model Comparison")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=22, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "calibrated_model_comparison.png")
    plt.close()

    calibration_frames = []
    calibration_summary_rows = []
    for model_name, y_prob in [
        ("Logistic Regression", lr_test_prob),
        ("Decision Tree", tree_test_prob),
        ("Random Forest, tuned", rf_prob),
    ]:
        calibration_frames.append(calibration_bins(y_test, y_prob, model_name, n_bins=10))
        calibration_summary_rows.append(calibration_summary(y_test, y_prob, model_name, n_bins=10))
    calibration_curve_table = pd.concat(calibration_frames, ignore_index=True)
    calibration_summary_table = pd.DataFrame(calibration_summary_rows)
    save_table(calibration_curve_table, "calibration_curves.csv")
    save_table(calibration_summary_table, "calibration_summary.csv")

    plt.figure(figsize=(6.8, 5.8))
    for model_name in calibration_curve_table["model"].unique():
        model_bins = calibration_curve_table[calibration_curve_table["model"].eq(model_name)]
        sns.lineplot(
            data=model_bins,
            x="mean_predicted_risk",
            y="observed_dropout_rate",
            marker="o",
            label=model_name,
        )
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    plt.title("Calibration Curves, Test Set")
    plt.xlabel("Mean predicted dropout risk")
    plt.ylabel("Observed dropout rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration_curves.png")
    plt.close()

    bootstrap_rows = []
    bootstrap_specs = [
        ("Logistic Regression", lr_test_prob),
        ("Decision Tree", tree_test_prob),
        ("Random Forest, tuned", rf_prob),
    ]
    for offset, (model_name, y_prob) in enumerate(bootstrap_specs):
        ci = bootstrap_metric_ci(
            y_test,
            y_prob,
            threshold=0.5,
            n_bootstrap=1000,
            random_state=RANDOM_STATE + offset,
        )
        ci["model"] = model_name
        bootstrap_rows.append(ci)
    bootstrap_ci_table = pd.DataFrame(bootstrap_rows)
    save_table(bootstrap_ci_table, "bootstrap_metric_ci.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, metric, point_col, low_col, high_col, title in [
        (axes[0], "AUC", "AUC_point", "AUC_ci_low", "AUC_ci_high", "AUC with 95% Bootstrap CI"),
        (axes[1], "Recall", "recall_point", "recall_ci_low", "recall_ci_high", "Recall with 95% Bootstrap CI"),
    ]:
        y_pos = np.arange(len(bootstrap_ci_table))
        points = bootstrap_ci_table[point_col].to_numpy()
        lows = bootstrap_ci_table[low_col].to_numpy()
        highs = bootstrap_ci_table[high_col].to_numpy()
        ax.errorbar(points, y_pos, xerr=[points - lows, highs - points], fmt="o", color="#4C72B0", capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bootstrap_ci_table["model"])
        ax.set_xlim(0.75, 1.0)
        ax.set_xlabel(metric)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bootstrap_metric_ci.png")
    plt.close()

    full_lr_raw_pipe = early_pipelines["Enrollment + 1st + 2nd semester"]
    approved_feature = "Curricular units 2nd sem (approved)"
    tuition_feature = "Tuition fees up to date"
    approved_unique = np.sort(X_test_raw[approved_feature].dropna().unique())
    if len(approved_unique) > 18:
        approved_grid = np.unique(np.round(np.quantile(approved_unique, np.linspace(0, 1, 18))).astype(int))
    else:
        approved_grid = approved_unique
    tuition_grid = np.sort(X_test_raw[tuition_feature].dropna().unique())

    pdp_approved = manual_partial_dependence(best_rf, X_test_raw, approved_feature, approved_grid)
    pdp_tuition = manual_partial_dependence(best_rf, X_test_raw, tuition_feature, tuition_grid)
    partial_dependence_table = pd.concat([pdp_approved, pdp_tuition], ignore_index=True)
    save_table(partial_dependence_table, "partial_dependence.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    sns.lineplot(
        data=pdp_approved,
        x="value",
        y="mean_predicted_dropout_risk",
        marker="o",
        ax=axes[0],
        color="#4C72B0",
    )
    axes[0].set_title("PDP: 2nd-semester approved units")
    axes[0].set_xlabel("Approved curricular units")
    axes[0].set_ylabel("Mean predicted dropout risk")
    axes[0].set_ylim(0, 1)
    sns.barplot(
        data=pdp_tuition,
        x="value",
        y="mean_predicted_dropout_risk",
        ax=axes[1],
        color="#C44E52",
    )
    axes[1].set_title("PDP: tuition fees up to date")
    axes[1].set_xlabel("Tuition fees up to date")
    axes[1].set_ylabel("Mean predicted dropout risk")
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "partial_dependence_key_features.png")
    plt.close()

    base_counterfactual_prob = full_lr_raw_pipe.predict_proba(X_test_raw)[:, 1]
    max_approved = X_train_raw[approved_feature].max()

    def counterfactual_row(name: str, X_mod: pd.DataFrame, eligible_mask: np.ndarray) -> dict:
        new_prob = full_lr_raw_pipe.predict_proba(X_mod)[:, 1]
        delta = new_prob - base_counterfactual_prob
        eligible_delta = delta[eligible_mask]
        return {
            "scenario": name,
            "n_eligible": int(eligible_mask.sum()),
            "baseline_mean_risk_eligible": float(base_counterfactual_prob[eligible_mask].mean()),
            "scenario_mean_risk_eligible": float(new_prob[eligible_mask].mean()),
            "mean_delta_risk": float(eligible_delta.mean()),
            "median_delta_risk": float(np.median(eligible_delta)),
            "q25_delta_risk": float(np.quantile(eligible_delta, 0.25)),
            "q75_delta_risk": float(np.quantile(eligible_delta, 0.75)),
            "pct_with_lower_risk": float((eligible_delta < 0).mean()),
        }

    counterfactual_rows = []
    tuition_mask = X_test_raw[tuition_feature].eq(0).to_numpy()
    X_cf = X_test_raw.copy()
    X_cf.loc[X_cf[tuition_feature].eq(0), tuition_feature] = 1
    counterfactual_rows.append(counterfactual_row("Set unpaid tuition to up-to-date", X_cf, tuition_mask))

    plus_one_mask = X_test_raw[approved_feature].lt(max_approved).to_numpy()
    X_cf = X_test_raw.copy()
    X_cf[approved_feature] = np.minimum(X_cf[approved_feature] + 1, max_approved)
    counterfactual_rows.append(counterfactual_row("Increase 2nd-sem approved units by +1", X_cf, plus_one_mask))

    plus_three_mask = X_test_raw[approved_feature].lt(max_approved).to_numpy()
    X_cf = X_test_raw.copy()
    X_cf[approved_feature] = np.minimum(X_cf[approved_feature] + 3, max_approved)
    counterfactual_rows.append(counterfactual_row("Increase 2nd-sem approved units by +3", X_cf, plus_three_mask))

    combined_mask = tuition_mask | plus_three_mask
    X_cf = X_test_raw.copy()
    X_cf.loc[X_cf[tuition_feature].eq(0), tuition_feature] = 1
    X_cf[approved_feature] = np.minimum(X_cf[approved_feature] + 3, max_approved)
    counterfactual_rows.append(counterfactual_row("Tuition up-to-date and approved units +3", X_cf, combined_mask))

    counterfactual_table = pd.DataFrame(counterfactual_rows)
    save_table(counterfactual_table, "counterfactual_scenarios.csv")

    plt.figure(figsize=(8.6, 4.6))
    sns.barplot(
        data=counterfactual_table,
        x="mean_delta_risk",
        y="scenario",
        color="#4C72B0",
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Counterfactual-style Risk Simulation, Logistic Regression")
    plt.xlabel("Mean change in predicted dropout risk")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "counterfactual_scenarios.png")
    plt.close()

    false_negative_mask = (y_test_array == 1) & (lr_test_pred == 0)
    true_positive_mask = (y_test_array == 1) & (lr_test_pred == 1)
    all_dropout_mask = y_test_array == 1
    profile_features = [
        "Age at enrollment",
        "Admission grade",
        "Previous qualification (grade)",
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
    ]
    profile_features = [c for c in profile_features if c in X_test_raw.columns]
    profile_rows = []
    for feature in profile_features:
        values = X_test_raw[feature]
        fn_mean = float(values[false_negative_mask].mean())
        tp_mean = float(values[true_positive_mask].mean())
        dropout_mean = float(values[all_dropout_mask].mean())
        pooled_std = float(values[all_dropout_mask].std(ddof=0))
        standardized_diff = (fn_mean - tp_mean) / pooled_std if pooled_std > 0 else np.nan
        profile_rows.append(
            {
                "feature": feature,
                "false_negative_mean": fn_mean,
                "true_positive_mean": tp_mean,
                "all_dropout_mean": dropout_mean,
                "fn_minus_tp": fn_mean - tp_mean,
                "standardized_fn_minus_tp": standardized_diff,
            }
        )
    false_negative_profile_table = (
        pd.DataFrame(profile_rows)
        .sort_values("standardized_fn_minus_tp", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    save_table(false_negative_profile_table, "false_negative_profile.csv")

    categorical_profile_cols = [
        c for c in ["Gender", "International", "Scholarship holder", "Debtor", "Tuition fees up to date"]
        if c in X_test_raw.columns
    ]
    categorical_rows = []
    for feature in categorical_profile_cols:
        for value in sorted(X_test_raw[feature].dropna().unique().tolist()):
            value_mask = X_test_raw[feature].eq(value).to_numpy()
            categorical_rows.append(
                {
                    "feature": feature,
                    "value": str(value),
                    "false_negative_rate_with_value": float((value_mask & false_negative_mask).sum() / max(false_negative_mask.sum(), 1)),
                    "true_positive_rate_with_value": float((value_mask & true_positive_mask).sum() / max(true_positive_mask.sum(), 1)),
                    "all_dropout_rate_with_value": float((value_mask & all_dropout_mask).sum() / max(all_dropout_mask.sum(), 1)),
                }
            )
    false_negative_categorical_table = pd.DataFrame(categorical_rows)
    save_table(false_negative_categorical_table, "false_negative_profile_categorical.csv")

    false_negative_examples = X_test_raw.loc[false_negative_mask, profile_features].copy()
    false_negative_examples.insert(0, "row_index", false_negative_examples.index)
    false_negative_examples.insert(1, "predicted_dropout_risk", lr_test_prob[false_negative_mask])
    false_negative_examples = false_negative_examples.sort_values("predicted_dropout_risk").head(25)
    save_table(false_negative_examples, "false_negative_examples.csv")

    fn_plot = false_negative_profile_table.head(10).copy()
    plt.figure(figsize=(8.8, 4.8))
    sns.barplot(data=fn_plot, x="standardized_fn_minus_tp", y="feature", color="#C44E52")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("False Negative Profile: Standardized Mean Difference vs True Positives")
    plt.xlabel("False-negative mean minus true-positive mean, standardized")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "false_negative_profile.png")
    plt.close()

    summary = {
        "random_state": RANDOM_STATE,
        "preprocessing": preprocessing_summary,
        "target_counts": target_counts.to_dict(),
        "binary_counts": {
            "Graduate_0": int(class_counts.loc[0]),
            "Dropout_1": int(class_counts.loc[1]),
            "minority_majority_ratio": minority_majority_ratio,
        },
        "kmeans_best_k": best_k_kmeans,
        "hierarchical_best_k": best_k_hier,
        "clustering_evaluation": cluster_eval.to_dict(orient="records"),
        "test_metrics_lr_tree": metrics_table.to_dict(orient="records"),
        "cross_validation": cv_table.to_dict(orient="records"),
        "top5_encoded_features": top5_encoded.to_dict(orient="records"),
        "top5_original_features": top5_original.to_dict(orient="records"),
        "random_forest_best_params": grid.best_params_,
        "model_comparison": model_comparison.to_dict(orient="records"),
        "smote_summary": smote_summary.to_dict(orient="records") if not smote_summary.empty else [],
        "early_warning_comparison": early_warning_table.to_dict(orient="records"),
        "feature_ablation": ablation_table.to_dict(orient="records"),
        "intervention_coverage_curve": intervention_coverage_table.to_dict(orient="records"),
        "prediction_time_discussion": prediction_time_table.to_dict(orient="records"),
        "threshold_policy": {
            "capacity_rate": capacity_rate,
            "capacity_count_test": int(capacity_pred.sum()),
            "capacity_cutoff": capacity_cutoff,
            "policy_metrics": capacity_row,
        },
        "permutation_importance_top10": permutation_top10.to_dict(orient="records"),
        "fairness_by_group": fairness_table.to_dict(orient="records"),
        "fairness_bootstrap_ci": fairness_bootstrap_table.to_dict(orient="records"),
        "group_calibration_summary": group_calibration_table.to_dict(orient="records"),
        "enrolled_risk_summary": enrolled_risk_summary.to_dict(orient="records")[0],
        "enrolled_risk_buckets": bucket_counts.to_dict(orient="records"),
        "calibration_summary": calibration_summary_table.to_dict(orient="records"),
        "calibrated_model_comparison": calibrated_model_table.to_dict(orient="records"),
        "bootstrap_metric_ci": bootstrap_ci_table.to_dict(orient="records"),
        "partial_dependence_key_features": partial_dependence_table.to_dict(orient="records"),
        "counterfactual_scenarios": counterfactual_table.to_dict(orient="records"),
        "false_negative_profile": false_negative_profile_table.to_dict(orient="records"),
        "false_negative_counts": {
            "false_negative_dropout_count": int(false_negative_mask.sum()),
            "true_positive_dropout_count": int(true_positive_mask.sum()),
            "total_dropout_test": int(all_dropout_mask.sum()),
        },
        "figures": {
            "target_distribution": "figures/target_distribution.png",
            "missing_ratios": "figures/missing_ratios.png",
            "tsne": "figures/tsne_target.png",
            "kmeans_selection": "figures/kmeans_elbow_silhouette.png",
            "dendrogram": "figures/hierarchical_dendrogram.png",
            "clustering": "figures/clustering_tsne.png",
            "confusion_logistic": "figures/confusion_logistic_regression.png",
            "confusion_tree": "figures/confusion_decision_tree.png",
            "roc": "figures/roc_lr_tree.png",
            "feature_importance": "figures/feature_importance_top5.png",
            "model_comparison": "figures/model_comparison_auc.png",
            "early_warning": "figures/early_warning_comparison.png",
            "feature_ablation": "figures/feature_ablation.png",
            "threshold_tuning": "figures/threshold_tuning.png",
            "intervention_coverage": "figures/intervention_coverage_curve.png",
            "permutation_importance": "figures/permutation_importance_top10.png",
            "fairness": "figures/fairness_group_metrics.png",
            "fairness_bootstrap": "figures/fairness_bootstrap_ci.png",
            "group_calibration": "figures/group_calibration.png",
            "enrolled_risk": "figures/enrolled_risk_distribution.png",
            "calibration": "figures/calibration_curves.png",
            "calibrated_models": "figures/calibrated_model_comparison.png",
            "calibrated_curves": "figures/calibrated_model_curves.png",
            "bootstrap_ci": "figures/bootstrap_metric_ci.png",
            "partial_dependence": "figures/partial_dependence_key_features.png",
            "counterfactual": "figures/counterfactual_scenarios.png",
            "false_negative_profile": "figures/false_negative_profile.png",
        },
    }
    save_json(summary)

    print("Analysis complete.")
    print(f"Preprocessed feature shape: {X_processed.shape}")
    print("Clustering evaluation:")
    print(cluster_eval.to_string(index=False))
    print("Test metrics:")
    print(metrics_table.to_string(index=False))
    print("Model comparison:")
    print(model_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
