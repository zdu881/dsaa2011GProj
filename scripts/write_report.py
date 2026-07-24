"""Generate the Covertype project report from computed outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
REPORT_PATH = ROOT / "report_covertype.md"


def num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return ""
    return f"{100 * float(x):.{digits}f}%"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def format_scores(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in score_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: num(v))
    return out


def main() -> None:
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    target = pd.read_csv(OUT_DIR / "target_counts.csv")
    cluster = pd.read_csv(OUT_DIR / "clustering_evaluation.csv")
    simple = pd.read_csv(OUT_DIR / "train_test_full_metrics.csv")
    models = pd.read_csv(OUT_DIR / "model_comparison.csv")
    cv = pd.read_csv(OUT_DIR / "cross_validation.csv")
    top_features = pd.read_csv(OUT_DIR / "feature_importance_top20.csv").head(10)
    perm = pd.read_csv(OUT_DIR / "permutation_importance_top15.csv").head(8)
    ablation = pd.read_csv(OUT_DIR / "feature_group_ablation.csv")
    calibration = pd.read_csv(OUT_DIR / "calibration_summary.csv")
    class_report = pd.read_csv(OUT_DIR / "class_report_best_model.csv")
    terrain = pd.read_csv(OUT_DIR / "elevation_band_metrics.csv")
    confidence = pd.read_csv(OUT_DIR / "prediction_confidence_summary.csv")
    cost = pd.read_csv(OUT_DIR / "computational_cost_estimates.csv")
    optimization = pd.read_csv(OUT_DIR / "optimization_directions.csv")
    runtime = pd.read_csv(OUT_DIR / "runtime_profile.csv")

    best = summary["best_model_test_metrics"]
    smallest = summary["dataset"]["smallest_class"]
    largest = summary["dataset"]["largest_class"]

    target_fmt = target[["Cover_Type", "Cover_Type_Name", "count", "proportion"]].copy()
    target_fmt["proportion"] = target_fmt["proportion"].map(pct)
    target_fmt["count"] = target_fmt["count"].map(lambda v: f"{int(v):,}")

    cluster_fmt = format_scores(
        cluster[["algorithm", "k", "silhouette", "calinski_harabasz", "ARI_vs_Target"]],
        ["silhouette", "calinski_harabasz", "ARI_vs_Target"],
    )

    simple_fmt = format_scores(
        simple[["model", "split", "accuracy", "f1_macro", "AUC_ovr_macro"]],
        ["accuracy", "f1_macro", "AUC_ovr_macro"],
    )

    model_fmt = format_scores(
        models[models["split"].eq("test")][
            ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "AUC_ovr_macro"]
        ],
        ["accuracy", "precision_macro", "recall_macro", "f1_macro", "AUC_ovr_macro"],
    )

    cv_fmt = format_scores(
        cv[["model", "accuracy_mean", "f1_macro_mean", "AUC_ovr_macro_mean"]],
        ["accuracy_mean", "f1_macro_mean", "AUC_ovr_macro_mean"],
    )

    top_fmt = format_scores(top_features[["feature", "importance"]], ["importance"])
    perm_fmt = format_scores(perm[["feature", "importance_mean_f1_drop", "importance_std"]], ["importance_mean_f1_drop", "importance_std"])
    ablation_fmt = format_scores(
        ablation[["experiment", "n_features", "accuracy", "f1_macro", "AUC_ovr_macro", "f1_macro_drop_vs_all"]],
        ["accuracy", "f1_macro", "AUC_ovr_macro", "f1_macro_drop_vs_all"],
    )
    calibration_fmt = format_scores(
        calibration[["model", "multiclass_brier", "expected_calibration_error", "mean_confidence", "observed_accuracy"]],
        ["multiclass_brier", "expected_calibration_error", "mean_confidence", "observed_accuracy"],
    )
    class_fmt = format_scores(
        class_report[["Cover_Type", "Cover_Type_Name", "support", "precision", "recall", "f1"]],
        ["precision", "recall", "f1"],
    )
    terrain_fmt = format_scores(
        terrain[["elevation_band", "n", "accuracy", "f1_macro", "dominant_true_class_name"]],
        ["accuracy", "f1_macro"],
    )
    conf_fmt = format_scores(confidence[["correct", "count", "mean", "median", "min", "max"]], ["mean", "median", "min", "max"])
    cost_fmt = cost[["module", "scope_used", "relative_cost_score", "optimization_decision"]].rename(
        columns={
            "module": "module",
            "scope_used": "scope",
            "relative_cost_score": "cost_score",
            "optimization_decision": "optimization",
        }
    )
    opt_fmt = optimization[["category", "recommended_actions", "expected_benefit"]].rename(
        columns={
            "category": "category",
            "recommended_actions": "action",
            "expected_benefit": "benefit",
        }
    )
    runtime_fmt = runtime[["stage", "seconds", "cache_status"]].copy()
    runtime_fmt["seconds"] = runtime_fmt["seconds"].map(lambda v: num(v, 2))

    report = f"""---
title: "DSAA2011 Covertype Project Report"
date: "2026 Spring"
---

# Introduction

This project analyzes the UCI **Covertype** dataset, which contains {summary["dataset"]["rows"]:,} forest observations, 54 cartographic features, and a seven-class forest cover target. The largest class is {int(largest["Cover_Type"])} ({largest["Cover_Type_Name"]}, {pct(largest["proportion"])}), while the smallest class is {int(smallest["Cover_Type"])} ({smallest["Cover_Type_Name"]}, {pct(smallest["proportion"])}), so macro-averaged metrics are more informative than accuracy alone.

The main result is that natural clusters do not align strongly with the seven cover types, but supervised tree ensembles perform well. On a stratified 120,000-row modeling sample, the best test model is **{summary["best_model_name"]}**, with accuracy={num(best["accuracy"])}, macro-F1={num(best["f1_macro"])}, and macro one-vs-rest AUC={num(best["AUC_ovr_macro"])}.

{markdown_table(target_fmt)}

# 1. Data Preprocessing

The raw UCI file is a numeric, comma-separated table without headers. The preparation script preserves the original `data/covtype.data.gz`, assigns the official feature names, and writes `data/covertype.csv.gz`. There are no missing values in the prepared data (maximum missing ratio={num(summary["dataset"]["max_missing_ratio"])}). The first 10 variables are continuous terrain measurements, followed by 4 wilderness-area indicators and 40 soil-type indicators. For linear models and distance-based methods, the continuous variables are median-imputed and standardized, while binary indicators are imputed by their mode and kept as indicators. Tree-based models use the same feature set without scaling.

![Target distribution](figures/target_distribution.png){{ width=5.6in }}

# 2. Data Visualization

t-SNE is applied to a stratified 6,000-row sample after standardization and PCA compression to 30 dimensions. The embedding shows partially separated regions for high-elevation classes such as Spruce/Fir and Krummholz, while Spruce/Fir and Lodgepole Pine overlap heavily. Rare classes are visible in local pockets but do not form clean global islands, which suggests that cover type is shaped by interacting terrain features rather than a single low-dimensional boundary.

![t-SNE projection](figures/tsne_target.png){{ width=5.7in }}

# 3. Clustering Analysis

MiniBatchKMeans and Ward hierarchical clustering are evaluated on stratified samples using silhouette, Calinski-Harabasz, and adjusted Rand index against the true cover labels. Both algorithms select k=2 by silhouette, but their ARI values are close to zero. This means the strongest unsupervised split is not the same as the seven ecological labels. MiniBatchKMeans is used instead of ordinary K-Means because it has the same interpretation but scales better to larger samples; Ward is kept as a small-sample dendrogram-based structural check.

{markdown_table(cluster_fmt)}

![MiniBatchKMeans selection](figures/kmeans_elbow_silhouette.png){{ width=5.4in }}

![Cluster comparison](figures/clustering_tsne.png){{ width=5.7in }}

# 4. Prediction: Training and Testing

The supervised target is the seven-class `Cover_Type`. The project uses a stratified 120,000-row modeling sample to keep validation reproducible and computationally reasonable while preserving class proportions. A 70%/30% train-test split gives 84,000 training rows and 36,000 test rows. Logistic regression is a simple linear baseline with class weighting, and a depth-limited decision tree is a nonlinear baseline. Both are also evaluated on the full 581,012-row dataset after training.

{markdown_table(simple_fmt)}

![Logistic regression confusion matrices](figures/confusion_logistic_regression.png){{ width=6.0in }}

![Decision tree confusion matrices](figures/confusion_decision_tree.png){{ width=6.0in }}

# 5. Evaluation and Model Choice

The simple logistic model has high macro AUC but weak macro-F1 because its linear boundary struggles with minority classes. The decision tree improves recall and F1 but overfits relative to the test set. The open-ended comparison adds Random Forest and HistGradientBoosting. Random Forest is the best test model, with accuracy={num(best["accuracy"])}, macro-F1={num(best["f1_macro"])}, and macro AUC={num(best["AUC_ovr_macro"])}.

{markdown_table(model_fmt)}

Cross-validation on a separate 36,000-row stratified sample confirms the same ranking.

{markdown_table(cv_fmt)}

![Model comparison](figures/model_comparison_f1_auc.png){{ width=5.7in }}

![ROC curves](figures/roc_ovr_models.png){{ width=5.3in }}

# 6. Open-Ended Exploration

## 6.1 Feature Importance

Random Forest impurity importance ranks **Elevation** as the dominant feature, followed by road/fire/hydrology distance variables and hillshade. Permutation importance for the best model agrees that the model relies most on terrain and distance variables rather than only on one-hot soil or wilderness indicators.

{markdown_table(top_fmt)}

{markdown_table(perm_fmt)}

![Feature importance](figures/feature_importance_top15.png){{ width=5.4in }}

## 6.2 Feature Group Ablation

Ablation with HistGradientBoosting shows that topographic variables alone are already strong, but adding soil indicators nearly recovers the all-feature model. Wilderness indicators alone are weak. This supports the interpretation that cover type is primarily driven by elevation, terrain geometry, hydrology/road/fire distances, and finer soil context.

{markdown_table(ablation_fmt)}

![Feature group ablation](figures/feature_group_ablation.png){{ width=5.7in }}

## 6.3 Calibration, Terrain Robustness, and Error Cases

Calibration is measured with top-label confidence bins. HistGradientBoosting has the lowest expected calibration error, while Random Forest has the best accuracy and F1 but is under-confident on average: its mean confidence is below observed accuracy. This is acceptable for ranking and classification, but probability interpretation should use calibration checks.

{markdown_table(calibration_fmt)}

![Calibration curves](figures/calibration_reliability.png){{ width=5.0in }}

The best model performs better in the highest elevation band than in lower bands, indicating that high-elevation classes are easier to separate. Error analysis also shows that incorrect predictions have much lower average top-class confidence than correct predictions, so confidence is useful for triaging uncertain cases.

{markdown_table(terrain_fmt)}

{markdown_table(conf_fmt)}

![Elevation band accuracy](figures/elevation_band_accuracy.png){{ width=5.2in }}

![Confidence by correctness](figures/error_confidence.png){{ width=5.0in }}

Class-level recall confirms that minority and ecologically adjacent classes are harder than the two dominant classes.

{markdown_table(class_fmt)}

![Class recall](figures/class_recall_comparison.png){{ width=5.4in }}

## 6.4 Computational Cost and Scalable Optimization

Covertype is large enough that model training is feasible, but several standard analysis steps become expensive if applied naively to all {summary["dataset"]["rows"]:,} rows. Full-data t-SNE, full-data Ward clustering, exhaustive grid search, and full permutation importance are the main computational risks. The optimized workflow now implements dtype downcasting, manifest-based caches for heavy stages, stratified samples for expensive visualization/validation, PCA before distance-based analysis, MiniBatchKMeans for scalable clustering, and parallel tree ensembles for prediction. The naive all-int64 memory estimate is {num(summary["computational_scaling"]["baseline_memory_mb"], 1)} MB; the optimized dataframe uses about {num(summary["computational_scaling"]["optimized_memory_mb"], 1)} MB, a {pct(summary["computational_scaling"]["memory_reduction_pct"])} reduction.

{markdown_table(cost_fmt)}

![Computational cost tiers](figures/computational_cost_tiers.png){{ width=5.6in }}

The run profile records whether each stage was computed or recovered from cache.

{markdown_table(runtime_fmt)}

The optimization directions can be grouped as follows.

{markdown_table(opt_fmt)}

# Conclusion

The Covertype dataset has strong class imbalance and substantial overlap between ecological classes. Unsupervised clustering reveals broad terrain structure but does not recover the true seven labels. Supervised tree ensembles are much more effective: Random Forest gives the best overall classification performance, while HistGradientBoosting offers better calibration. Elevation is the most important single feature, but soil indicators and distance-to-road/fire/hydrology variables add meaningful predictive value. Computationally, the safest strategy is to combine full-data descriptive analysis, stratified samples for expensive diagnostics, PCA for distance-based methods, and parallel ensemble models. For practical use, the model should report macro metrics, class-level recall, calibration, terrain-band performance, and the sampling assumptions behind expensive steps.

# References

1. UCI Machine Learning Repository. Covertype dataset. https://archive.ics.uci.edu/ml/datasets/covertype
2. Blackard, Jock A. and Dean, Denis J. Comparative Accuracies of Artificial Neural Networks and Discriminant Analysis in Predicting Forest Cover Types from Cartographic Variables. Computers and Electronics in Agriculture, 1999.
3. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.
4. van der Maaten and Hinton. Visualizing Data using t-SNE. Journal of Machine Learning Research, 2008.

# Credit

Code, figures, and report text were generated reproducibly in this repository. Group-member contribution details should be filled in by the project team before submission. OpenAI Codex was used to draft and validate the analysis workflow and report.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
