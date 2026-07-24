"""Generate a NeurIPS-style LaTeX report for the Covertype project."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
TEX_PATH = ROOT / "report_covertype.tex"


def num(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.{digits}f}%"


def tex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def table_tex(df: pd.DataFrame, caption: str, label: str, small: bool = True) -> str:
    cols = list(df.columns)
    align = "l" * len(cols)
    font_size = r"\small" if small else r"\normalsize"
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{" + tex_escape(caption) + r"}",
        r"\label{" + label + r"}",
        font_size,
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{" + align + r"}",
        r"\toprule",
        " & ".join(tex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(row[c]) for c in cols) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def figure_tex(path: str, caption: str, label: str, width: str = r"0.88\linewidth") -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width={width}]{{{path}}}",
            r"\caption{" + tex_escape(caption) + r"}",
            r"\label{" + label + r"}",
            r"\end{figure}",
        ]
    )


def two_figures_tex(
    left_path: str,
    left_caption: str,
    right_path: str,
    right_caption: str,
    caption: str,
    label: str,
) -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\begin{{minipage}}{{0.48\linewidth}}\centering\includegraphics[width=\linewidth]{{{left_path}}}\\[-0.5ex]\small {tex_escape(left_caption)}\end{{minipage}}\hfill%",
            rf"\begin{{minipage}}{{0.48\linewidth}}\centering\includegraphics[width=\linewidth]{{{right_path}}}\\[-0.5ex]\small {tex_escape(right_caption)}\end{{minipage}}",
            r"\caption{" + tex_escape(caption) + r"}",
            r"\label{" + label + r"}",
            r"\end{figure}",
        ]
    )


def fmt_scores(df: pd.DataFrame, cols: list[str], digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: num(v, digits))
    return out


def main() -> None:
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    target = pd.read_csv(OUT_DIR / "target_counts.csv")
    wilderness = pd.read_csv(OUT_DIR / "wilderness_area_summary.csv")
    wilderness_metrics = pd.read_csv(OUT_DIR / "wilderness_model_metrics.csv")
    cluster = pd.read_csv(OUT_DIR / "clustering_evaluation.csv")
    simple = pd.read_csv(OUT_DIR / "train_test_full_metrics.csv")
    models = pd.read_csv(OUT_DIR / "model_comparison.csv")
    cv = pd.read_csv(OUT_DIR / "cross_validation.csv")
    top_features = pd.read_csv(OUT_DIR / "feature_importance_top20.csv").head(8)
    ablation = pd.read_csv(OUT_DIR / "feature_group_ablation.csv")
    calibration = pd.read_csv(OUT_DIR / "calibration_summary.csv")
    class_report = pd.read_csv(OUT_DIR / "class_report_best_model.csv")
    runtime = pd.read_csv(OUT_DIR / "runtime_profile.csv")

    best = summary["best_model_test_metrics"]
    target_fmt = target[["Cover_Type", "Cover_Type_Name", "count", "proportion"]].copy()
    target_fmt["count"] = target_fmt["count"].map(lambda v: f"{int(v):,}")
    target_fmt["proportion"] = target_fmt["proportion"].map(lambda v: pct(v))

    cluster_fmt = fmt_scores(
        cluster[["algorithm", "k", "silhouette", "calinski_harabasz", "ARI_vs_Target", "ARI_vs_Wilderness"]],
        ["silhouette", "calinski_harabasz", "ARI_vs_Target", "ARI_vs_Wilderness"],
    )
    cluster_fmt = cluster_fmt.rename(
        columns={
            "algorithm": "algorithm",
            "silhouette": "silhouette",
            "calinski_harabasz": "CH",
            "ARI_vs_Target": "ARI target",
            "ARI_vs_Wilderness": "ARI wild",
        }
    )
    wilderness_fmt = wilderness[
        [
            "Wilderness_Area",
            "Wilderness_Area_Name",
            "count",
            "proportion",
            "dominant_cover_type_name",
            "dominant_cover_type_share",
            "normalized_cover_entropy",
        ]
    ].copy()
    wilderness_fmt["count"] = wilderness_fmt["count"].map(lambda v: f"{int(v):,}")
    wilderness_fmt["proportion"] = wilderness_fmt["proportion"].map(lambda v: pct(v))
    wilderness_fmt["dominant_cover_type_share"] = wilderness_fmt["dominant_cover_type_share"].map(lambda v: pct(v))
    wilderness_fmt["normalized_cover_entropy"] = wilderness_fmt["normalized_cover_entropy"].map(lambda v: num(v))
    wilderness_fmt = wilderness_fmt.rename(
        columns={
            "Wilderness_Area": "area",
            "Wilderness_Area_Name": "name",
            "count": "n",
            "proportion": "share",
            "dominant_cover_type_name": "dominant cover",
            "dominant_cover_type_share": "dominant share",
            "normalized_cover_entropy": "entropy",
        }
    )
    wilderness_metrics_fmt = fmt_scores(
        wilderness_metrics[["Wilderness_Area_Name", "n", "test_share", "accuracy", "f1_macro", "dominant_true_class_name"]],
        ["test_share", "accuracy", "f1_macro"],
    )
    wilderness_metrics_fmt["n"] = wilderness_metrics_fmt["n"].map(lambda v: f"{int(v):,}")
    wilderness_metrics_fmt = wilderness_metrics_fmt.rename(
        columns={
            "Wilderness_Area_Name": "area",
            "test_share": "test share",
            "f1_macro": "macro-F1",
            "dominant_true_class_name": "dominant cover",
        }
    )
    simple_fmt = fmt_scores(
        simple[["model", "split", "accuracy", "f1_macro", "AUC_ovr_macro"]],
        ["accuracy", "f1_macro", "AUC_ovr_macro"],
    )
    model_fmt = fmt_scores(
        models[models["split"].eq("test")][["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "AUC_ovr_macro"]],
        ["accuracy", "precision_macro", "recall_macro", "f1_macro", "AUC_ovr_macro"],
    )
    cv_fmt = fmt_scores(cv[["model", "accuracy_mean", "f1_macro_mean", "AUC_ovr_macro_mean"]], ["accuracy_mean", "f1_macro_mean", "AUC_ovr_macro_mean"])
    feature_fmt = fmt_scores(top_features[["feature", "importance"]], ["importance"])
    ablation_fmt = fmt_scores(
        ablation[["experiment", "n_features", "accuracy", "f1_macro", "AUC_ovr_macro"]],
        ["accuracy", "f1_macro", "AUC_ovr_macro"],
    )
    calibration_fmt = fmt_scores(
        calibration[["model", "multiclass_brier", "expected_calibration_error", "mean_confidence", "observed_accuracy"]],
        ["multiclass_brier", "expected_calibration_error", "mean_confidence", "observed_accuracy"],
    )
    class_fmt = fmt_scores(
        class_report[["Cover_Type", "Cover_Type_Name", "support", "precision", "recall", "f1"]],
        ["precision", "recall", "f1"],
    )
    runtime_fmt = runtime[["stage", "seconds", "cache_status"]].copy()
    runtime_fmt["seconds"] = runtime_fmt["seconds"].map(lambda v: num(v, 2))

    smallest_pct = tex_escape(pct(summary["dataset"]["smallest_class"]["proportion"]))
    baseline_memory = num(summary["computational_scaling"]["baseline_memory_mb"], 1)
    optimized_memory = num(summary["computational_scaling"]["optimized_memory_mb"], 1)
    target_table = table_tex(target_fmt, "Class distribution in the Covertype dataset.", "tab:target")
    target_figure = figure_tex(
        "figures/target_distribution.png",
        "Cover-type distribution. The two dominant classes make macro metrics necessary.",
        "fig:target",
        r"0.78\linewidth",
    )
    viz_cluster_figure = two_figures_tex(
        "figures/tsne_target.png",
        "t-SNE by target",
        "figures/clustering_tsne.png",
        "Cluster assignments",
        "Low-dimensional visualization and clustering results.",
        "fig:viz_cluster",
    )
    cluster_table = table_tex(cluster_fmt, "Best clustering results under the selected internal metrics.", "tab:cluster")
    wilderness_table = table_tex(wilderness_fmt, "Cover-type composition summary by wilderness area.", "tab:wilderness_summary")
    wilderness_metrics_table = table_tex(wilderness_metrics_fmt, "Best-model test performance by wilderness area.", "tab:wilderness_metrics")
    wilderness_figure = two_figures_tex(
        "figures/wilderness_cover_distribution.png",
        "Cover composition",
        "figures/wilderness_model_metrics.png",
        "Model performance",
        "Wilderness-area extension analysis.",
        "fig:wilderness_extension",
    )
    simple_table = table_tex(simple_fmt, "Simple-model performance on train, test, and full data.", "tab:simple")
    confusion_figure = two_figures_tex(
        "figures/confusion_logistic_regression.png",
        "Logistic regression",
        "figures/confusion_decision_tree.png",
        "Decision tree",
        "Confusion matrices for the two required simple models.",
        "fig:confusions",
    )
    model_table = table_tex(model_fmt, "Test performance across model classes.", "tab:models")
    cv_table = table_tex(cv_fmt, "Three-fold cross-validation on a stratified sample.", "tab:cv")
    model_choice_figure = two_figures_tex(
        "figures/model_comparison_f1_auc.png",
        "Model metrics",
        "figures/roc_ovr_models.png",
        "Macro ROC",
        "Model comparison and macro one-vs-rest ROC curves.",
        "fig:model_choice",
    )
    feature_table = table_tex(feature_fmt, "Top Random Forest impurity importances.", "tab:feature_importance")
    ablation_table = table_tex(ablation_fmt, "Feature-group ablation using HistGradientBoosting.", "tab:ablation")
    feature_figure = two_figures_tex(
        "figures/feature_importance_top15.png",
        "Feature importance",
        "figures/feature_group_ablation.png",
        "Feature ablation",
        "Feature-importance and feature-group analyses.",
        "fig:features",
    )
    calibration_table = table_tex(calibration_fmt, "Top-label calibration summary.", "tab:calibration")
    class_table = table_tex(class_fmt, "Class-level precision, recall, and F1 for the best model.", "tab:class_report")
    calibration_figure = two_figures_tex(
        "figures/calibration_reliability.png",
        "Calibration",
        "figures/class_recall_comparison.png",
        "Class recall",
        "Calibration and class-level recall diagnostics.",
        "fig:calibration_class",
    )
    cost_figure = figure_tex(
        "figures/computational_cost_tiers.png",
        "Relative computational cost by workflow module.",
        "fig:cost",
        r"0.82\linewidth",
    )
    runtime_table = table_tex(runtime_fmt, "Runtime profile after cache-enabled optimization.", "tab:runtime")

    tex = rf"""\documentclass{{article}}
\usepackage[main,final,nonatbib]{{neurips_2025}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{microtype}}
\usepackage{{xcolor}}
\usepackage{{enumitem}}
\usepackage{{float}}
\usepackage{{adjustbox}}

\title{{DSAA2011 Covertype Project Report}}

\author{{%
  Covertype Project Team\\
  DSAA2011 Machine Learning Course Project\\
  \texttt{{groupID\_covertype}}\\
}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report studies the UCI Covertype dataset with {summary["dataset"]["rows"]:,} forest observations, 54 cartographic features, and seven cover-type classes. The analysis covers preprocessing, t-SNE visualization, MiniBatchKMeans and Ward clustering, supervised classification, model validation, a Wilderness Area extension, feature importance, calibration, robustness, and computational optimization. Natural clusters reveal terrain and region structure but do not recover the seven ecological labels. A Random Forest classifier is the strongest predictive model on the stratified test set, achieving accuracy={num(best["accuracy"])}, macro-F1={num(best["f1_macro"])}, and macro one-vs-rest AUC={num(best["AUC_ovr_macro"])}.
\end{{abstract}}

\section{{Introduction}}
The Covertype task predicts forest cover type from cartographic variables. The target is imbalanced: Lodgepole Pine and Spruce/Fir account for most observations, while Cottonwood/Willow is only {smallest_pct} of the data. We therefore report macro-averaged metrics in addition to accuracy. The main empirical result is that unsupervised structure is visible but does not directly match the supervised label definition, whereas nonlinear supervised ensembles perform well.

{target_table}

\section{{Data preprocessing}}
The raw UCI file is numeric and has no header. We assign the official feature names, use 10 continuous terrain measurements, 4 wilderness-area indicators, 40 soil-type indicators, and the seven-class \texttt{{Cover\_Type}} target. There are no missing values. Continuous features are standardized for linear and distance-based methods; binary indicators are kept as indicator variables. The implemented data optimization uses \texttt{{float32}} for continuous variables and \texttt{{uint8}} for indicators and target, reducing the naive all-\texttt{{int64}} memory estimate from {baseline_memory} MB to {optimized_memory} MB.

{target_figure}

\section{{Visualization and clustering}}
t-SNE is applied to a stratified {summary["sampling"]["tsne_sample_size"]:,}-row sample after standardization and PCA compression to 30 dimensions. The embedding shows partial local separation for high-elevation classes, but Spruce/Fir and Lodgepole Pine overlap substantially. This motivates supervised classification rather than relying on natural clusters.

MiniBatchKMeans and Ward hierarchical clustering are evaluated with silhouette, Calinski-Harabasz, adjusted Rand index against \texttt{{Cover\_Type}}, and adjusted Rand index against \texttt{{Wilderness\_Area}}. MiniBatchKMeans is used as the scalable K-Means variant; Ward is restricted to a smaller structural sample because of its quadratic memory/time behavior. The near-zero ARI against \texttt{{Cover\_Type}} should not be read as a failure to find any structure. It shows that the strongest unsupervised partitions are not equivalent to the seven ecological labels; the clusters are better interpreted as coarse terrain or regional groupings.

{viz_cluster_figure}

{cluster_table}

\section{{Wilderness Area extension}}
The four one-hot wilderness indicators are converted into a compact \texttt{{Wilderness\_Area}} variable. This gives an additional explanatory view that is distinct from the seven-class prediction target. The areas differ strongly in size and cover-type composition: each wilderness area has a dominant cover type, but the normalized entropy shows that some areas contain more mixed ecological labels than others. This supports the revised clustering interpretation: low ARI versus \texttt{{Cover\_Type}} only says that natural clusters are not the target labels; it does not rule out meaningful structure associated with region, elevation, and terrain.

{wilderness_table}

The same held-out predictions from the best supervised model are then grouped by wilderness area. Accuracy and macro-F1 vary across regions, so model quality is not spatially uniform. The regional split is therefore useful as a robustness check, not just as four extra binary predictors.

{wilderness_metrics_table}

{wilderness_figure}

\section{{Prediction: training and testing}}
The supervised target is the seven-class \texttt{{Cover\_Type}}. We use a stratified {summary["sampling"]["model_sample_size"]:,}-row modeling sample, split 70/30 into 84,000 training and 36,000 test observations. Logistic regression is the linear baseline with class weighting, and a depth-limited decision tree is the simple nonlinear baseline. Both are evaluated on train, test, and full data after training.

{simple_table}

{confusion_figure}

\section{{Evaluation and model choice}}
Logistic regression reaches high macro AUC but weak macro-F1 because the linear boundary struggles with minority classes. The decision tree improves nonlinear separation but still overfits relative to the test set. Random Forest and HistGradientBoosting are added for open-ended model comparison. Random Forest is the best test model with accuracy={num(best["accuracy"])}, macro-F1={num(best["f1_macro"])}, and macro AUC={num(best["AUC_ovr_macro"])}. Cross-validation on a separate stratified 36,000-row sample confirms the same ranking.

{model_table}

{cv_table}

{model_choice_figure}

\section{{Open-ended exploration}}
\subsection{{Feature importance and ablation}}
Random Forest impurity importance and sampled permutation importance both rank Elevation as the dominant feature, followed by road/fire/hydrology distances and hillshade. Feature-group ablation with HistGradientBoosting shows that topographic variables alone are already strong, but soil indicators add useful predictive signal; wilderness indicators alone are weak.

{feature_table}

{ablation_table}

{feature_figure}

\subsection{{Calibration, robustness, and errors}}
HistGradientBoosting has the lowest top-label expected calibration error, while Random Forest has the best accuracy and macro-F1 but is under-confident on average. Accuracy also rises in the highest elevation band, indicating that high-elevation classes are easier to separate. Class-level recall shows Aspen and Douglas-fir are harder than the dominant classes.

{calibration_table}

{class_table}

{calibration_figure}

\subsection{{Computational optimization}}
The optimized workflow implements the cost controls discussed in the project: dtype downcasting, stratified samples for expensive diagnostics, PCA before distance-based analysis, MiniBatchKMeans for scalable clustering, parallel tree ensembles, and manifest-based caches for t-SNE, clustering, and modeling outputs. A warm rebuild hits the heavy-stage caches, making report regeneration fast and auditable.

{cost_figure}

{runtime_table}

\section{{Conclusion}}
The Covertype dataset has strong class imbalance and overlapping ecological classes. Unsupervised clustering exposes broad terrain and regional structure, but the near-zero ARI against \texttt{{Cover\_Type}} only shows that these natural partitions are not the same object as the seven supervised labels. The Wilderness Area extension makes this distinction explicit by comparing cover-type composition and held-out model performance across regions. Supervised tree ensembles are substantially more effective, with Random Forest giving the best overall classification performance and HistGradientBoosting offering better calibration. Elevation is the strongest feature, but road/fire/hydrology distances, soil context, and regional evaluation all add meaningful interpretation. For deployment-style interpretation, the model should be reported with macro metrics, class-level recall, calibration, terrain-band and wilderness-area performance, and explicit sampling assumptions.

\section*{{References}}
\begin{{enumerate}}[leftmargin=*, itemsep=0.25em]
  \item UCI Machine Learning Repository. Covertype dataset. \url{{https://archive.ics.uci.edu/ml/datasets/covertype}}.
  \item Blackard, Jock A. and Dean, Denis J. Comparative Accuracies of Artificial Neural Networks and Discriminant Analysis in Predicting Forest Cover Types from Cartographic Variables. \emph{{Computers and Electronics in Agriculture}}, 1999.
  \item Pedregosa et al. Scikit-learn: Machine Learning in Python. \emph{{Journal of Machine Learning Research}}, 2011.
  \item van der Maaten and Hinton. Visualizing Data using t-SNE. \emph{{Journal of Machine Learning Research}}, 2008.
\end{{enumerate}}

\section*{{Credit and GenAI use}}
Analysis code, figures, and report text were generated reproducibly in this repository. Group-member contribution details should be finalized by the project team before submission. OpenAI Codex was used to draft, refactor, and validate the analysis workflow and report.

\end{{document}}
"""

    # Keep any accidental repeated spaces in generated table cells harmless while
    # avoiding blank lines that can stretch the compact NeurIPS layout.
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"Wrote {TEX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
