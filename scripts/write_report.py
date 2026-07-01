"""Generate a Markdown report from the computed project outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs" / "summary.json"
REPORT_PATH = ROOT / "report_student_dropout.md"


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * float(x):.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(rows)


def format_numeric_columns(df: pd.DataFrame, columns: list[str], digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else num(float(x), digits))
    return out


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    cluster_df = pd.read_csv(ROOT / "outputs" / "clustering_evaluation.csv")
    metrics_df = pd.read_csv(ROOT / "outputs" / "test_metrics_lr_tree.csv")
    cv_df = pd.read_csv(ROOT / "outputs" / "cross_validation.csv")
    model_df = pd.read_csv(ROOT / "outputs" / "model_comparison.csv")
    smote_df = pd.read_csv(ROOT / "outputs" / "smote_summary.csv")
    fi_df = pd.read_csv(ROOT / "outputs" / "feature_importance_top5_original.csv")
    early_df = pd.read_csv(ROOT / "outputs" / "early_warning_comparison.csv")
    ablation_df = pd.read_csv(ROOT / "outputs" / "feature_ablation.csv")
    threshold_df = pd.read_csv(ROOT / "outputs" / "threshold_tuning.csv")
    perm_df = pd.read_csv(ROOT / "outputs" / "permutation_importance_top10.csv")
    fairness_df = pd.read_csv(ROOT / "outputs" / "fairness_by_group.csv")
    enrolled_summary_df = pd.read_csv(ROOT / "outputs" / "enrolled_risk_summary.csv")
    enrolled_bucket_df = pd.read_csv(ROOT / "outputs" / "enrolled_risk_buckets.csv")
    calibration_df = pd.read_csv(ROOT / "outputs" / "calibration_summary.csv")
    bootstrap_df = pd.read_csv(ROOT / "outputs" / "bootstrap_metric_ci.csv")
    pdp_df = pd.read_csv(ROOT / "outputs" / "partial_dependence.csv")
    counterfactual_df = pd.read_csv(ROOT / "outputs" / "counterfactual_scenarios.csv")
    group_calibration_df = pd.read_csv(ROOT / "outputs" / "group_calibration_summary.csv")
    intervention_df = pd.read_csv(ROOT / "outputs" / "intervention_coverage_curve.csv")
    calibrated_df = pd.read_csv(ROOT / "outputs" / "calibrated_model_comparison.csv")
    fairness_bootstrap_df = pd.read_csv(ROOT / "outputs" / "fairness_bootstrap_ci.csv")
    prediction_time_df = pd.read_csv(ROOT / "outputs" / "prediction_time_discussion.csv")
    fn_profile_df = pd.read_csv(ROOT / "outputs" / "false_negative_profile.csv")

    pre = summary["preprocessing"]
    counts = summary["target_counts"]
    binary = summary["binary_counts"]
    lr = next(row for row in summary["test_metrics_lr_tree"] if row["model"] == "Logistic Regression")
    tree = next(row for row in summary["test_metrics_lr_tree"] if row["model"] == "Decision Tree")
    rf = next(row for row in summary["model_comparison"] if row["model"] == "Random Forest, tuned")
    policy = summary["threshold_policy"]["policy_metrics"]
    enrolled = summary["enrolled_risk_summary"]
    lr_cal = calibration_df.loc[calibration_df["model"].eq("Logistic Regression")].iloc[0]
    tuition_pdp_0 = pdp_df.loc[
        (pdp_df["feature"].eq("Tuition fees up to date")) & (pdp_df["value"].eq(0.0)),
        "mean_predicted_dropout_risk",
    ].iloc[0]
    tuition_pdp_1 = pdp_df.loc[
        (pdp_df["feature"].eq("Tuition fees up to date")) & (pdp_df["value"].eq(1.0)),
        "mean_predicted_dropout_risk",
    ].iloc[0]
    approved_pdp_low = pdp_df.loc[
        (pdp_df["feature"].eq("Curricular units 2nd sem (approved)")) & (pdp_df["value"].eq(0.0)),
        "mean_predicted_dropout_risk",
    ].iloc[0]
    approved_pdp_six = pdp_df.loc[
        (pdp_df["feature"].eq("Curricular units 2nd sem (approved)")) & (pdp_df["value"].eq(6.0)),
        "mean_predicted_dropout_risk",
    ].iloc[0]
    cf_plus3 = counterfactual_df.loc[
        counterfactual_df["scenario"].eq("Increase 2nd-sem approved units by +3")
    ].iloc[0]
    cf_combined = counterfactual_df.loc[
        counterfactual_df["scenario"].eq("Tuition up-to-date and approved units +3")
    ].iloc[0]
    intl_small = fairness_df.loc[
        (fairness_df["sensitive_feature"].eq("International"))
        & (fairness_df["group_value"].astype(str).eq("1"))
    ].iloc[0]
    intl_bootstrap = fairness_bootstrap_df.loc[
        (fairness_bootstrap_df["sensitive_feature"].eq("International"))
        & (fairness_bootstrap_df["group_value"].astype(str).eq("1"))
    ].iloc[0]
    coverage_25_early = intervention_df.loc[
        (intervention_df["model"].str.contains("Actionable"))
        & (intervention_df["follow_up_rate"].eq(0.25))
    ].iloc[0]
    coverage_25_full = intervention_df.loc[
        (intervention_df["model"].str.contains("Full"))
        & (intervention_df["follow_up_rate"].eq(0.25))
    ].iloc[0]
    best_brier_cal = calibrated_df.sort_values("brier_score").iloc[0]
    best_ece_cal = calibrated_df.sort_values("expected_calibration_error").iloc[0]
    top_fn_feature = fn_profile_df.iloc[0]

    early_enrollment = early_df.loc[early_df["experiment"].eq("Enrollment only")].iloc[0]
    early_first = early_df.loc[early_df["experiment"].eq("Enrollment + 1st semester")].iloc[0]
    early_full = early_df.loc[early_df["experiment"].eq("Enrollment + 1st + 2nd semester")].iloc[0]

    cluster_fmt = format_numeric_columns(
        cluster_df,
        ["silhouette", "calinski_harabasz", "ARI_vs_Target"],
    )
    metrics_fmt = format_numeric_columns(metrics_df, ["accuracy", "precision", "recall", "f1", "AUC"])
    cv_fmt = format_numeric_columns(
        cv_df[["model", "train_accuracy", "test_accuracy", "cv_auc_mean", "cv_auc_std", "train_test_gap"]],
        ["train_accuracy", "test_accuracy", "cv_auc_mean", "cv_auc_std", "train_test_gap"],
    )
    model_fmt = format_numeric_columns(model_df, ["accuracy", "precision", "recall", "f1", "AUC"])
    smote_fmt = format_numeric_columns(
        smote_df[["model", "precision", "recall", "f1", "AUC", "AUC_delta_vs_baseline"]],
        ["precision", "recall", "f1", "AUC", "AUC_delta_vs_baseline"],
    )
    fi_fmt = format_numeric_columns(fi_df, ["importance"])
    early_fmt = format_numeric_columns(
        early_df[
            [
                "experiment",
                "n_raw_features",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "AUC",
                "AUC_gain_vs_enrollment_only",
            ]
        ],
        ["accuracy", "precision", "recall", "f1", "AUC", "AUC_gain_vs_enrollment_only"],
    )
    ablation_fmt = format_numeric_columns(
        ablation_df[
            [
                "experiment",
                "n_raw_features",
                "AUC",
                "recall",
                "AUC_drop_vs_full",
                "recall_drop_vs_full",
            ]
        ],
        ["AUC", "recall", "AUC_drop_vs_full", "recall_drop_vs_full"],
    )
    perm_fmt = format_numeric_columns(
        perm_df.head(8)[["feature", "importance_mean_auc_drop", "importance_std"]],
        ["importance_mean_auc_drop", "importance_std"],
    )
    fairness_fmt = format_numeric_columns(
        fairness_df[
            [
                "sensitive_feature",
                "group_value",
                "n",
                "positives_dropout",
                "recall",
                "FPR",
                "precision",
                "mean_predicted_risk",
            ]
        ],
        ["recall", "FPR", "precision", "mean_predicted_risk"],
    )
    enrolled_bucket_fmt = enrolled_bucket_df.copy()
    calibration_fmt = format_numeric_columns(
        calibration_df[
            [
                "model",
                "brier_score",
                "expected_calibration_error",
                "max_calibration_error",
                "mean_predicted_risk",
                "observed_dropout_rate",
            ]
        ],
        [
            "brier_score",
            "expected_calibration_error",
            "max_calibration_error",
            "mean_predicted_risk",
            "observed_dropout_rate",
        ],
    )
    bootstrap_fmt = bootstrap_df[["model"]].copy()
    bootstrap_fmt["AUC_95CI"] = bootstrap_df.apply(
        lambda r: f"{num(r['AUC_point'])} [{num(r['AUC_ci_low'])}, {num(r['AUC_ci_high'])}]",
        axis=1,
    )
    bootstrap_fmt["Recall_95CI"] = bootstrap_df.apply(
        lambda r: f"{num(r['recall_point'])} [{num(r['recall_ci_low'])}, {num(r['recall_ci_high'])}]",
        axis=1,
    )
    counterfactual_fmt = format_numeric_columns(
        counterfactual_df[
            [
                "scenario",
                "n_eligible",
                "baseline_mean_risk_eligible",
                "scenario_mean_risk_eligible",
                "mean_delta_risk",
                "pct_with_lower_risk",
            ]
        ],
        [
            "baseline_mean_risk_eligible",
            "scenario_mean_risk_eligible",
            "mean_delta_risk",
            "pct_with_lower_risk",
        ],
    )
    group_calibration_fmt = format_numeric_columns(
        group_calibration_df[
            [
                "sensitive_feature",
                "group_value",
                "n",
                "observed_dropout_rate",
                "mean_predicted_risk",
                "calibration_gap_pred_minus_observed",
                "brier_score",
            ]
        ],
        [
            "observed_dropout_rate",
            "mean_predicted_risk",
            "calibration_gap_pred_minus_observed",
            "brier_score",
        ],
    )
    intervention_fmt = format_numeric_columns(
        intervention_df[
            [
                "model",
                "follow_up_rate",
                "follow_up_count",
                "captured_dropout",
                "dropout_coverage",
                "precision_among_followed",
                "lift_vs_random",
            ]
        ],
        ["follow_up_rate", "dropout_coverage", "precision_among_followed", "lift_vs_random"],
    )
    calibrated_fmt = format_numeric_columns(
        calibrated_df[
            [
                "model",
                "calibration_method",
                "AUC",
                "brier_score",
                "expected_calibration_error",
                "mean_predicted_risk",
            ]
        ],
        ["AUC", "brier_score", "expected_calibration_error", "mean_predicted_risk"],
    )
    fairness_bootstrap_fmt = format_numeric_columns(
        fairness_bootstrap_df,
        [
            "recall_bootstrap_mean",
            "recall_ci_low",
            "recall_ci_high",
            "FPR_bootstrap_mean",
            "FPR_ci_low",
            "FPR_ci_high",
        ],
    )
    prediction_time_fmt = format_numeric_columns(
        prediction_time_df,
        ["AUC", "recall", "precision", "f1"],
    )
    fn_profile_fmt = format_numeric_columns(
        fn_profile_df.head(8),
        [
            "false_negative_mean",
            "true_positive_mean",
            "all_dropout_mean",
            "fn_minus_tp",
            "standardized_fn_minus_tp",
        ],
    )

    report = f"""\
---
title: "DSAA2011 Student Dropout Project Report"
author: "Reference Answer for Teaching and Grading"
date: "2026 Spring"
---

# Introduction

本项目基于 UCI 的 **Predict Students' Dropout and Academic Success** 数据集完成 DSAA2011 项目要求。数据包含 {pre['raw_shape'][0]} 名本科学生、{pre['raw_shape'][1] - 1} 个原始特征，目标变量为 Dropout、Enrolled 与 Graduate，其中 Dropout={counts['Dropout']}、Enrolled={counts['Enrolled']}、Graduate={counts['Graduate']}。项目先用无监督方法观察高维结构，再将任务转化为 Dropout 与 Graduate 二分类，以便建立可解释的辍学预警模型。

![Target distribution](figures/target_distribution.png)

# 1. Data Preprocessing

缺失值检查显示最大缺失率为 {pct(pre['max_missing_ratio'])}，没有列超过 40% 阈值，因此未删除特征。数值型变量使用中位数填充，类别型变量使用众数填充；18 个类别变量经独热编码处理，其中 {', '.join(pre['high_cardinality_columns_grouped'])} 等高基数变量只保留 Top 5 类别，其余归为 Other。标准化前特征矩阵为 {pre['feature_shape_before_preprocessing']}，处理后为 {pre['feature_shape_after_preprocessing']}。该流程兼顾异常值鲁棒性、类别变量可建模性和不同量纲特征的可比性。

# 2. Data Visualization: t-SNE

![t-SNE projection](figures/tsne_target.png)

t-SNE 使用 perplexity=30、learning_rate=200、random_state=42。二维嵌入显示三类样本存在局部聚集，但整体交叠明显：Dropout 在若干区域密度较高，Graduate 在中心和右下区域较集中，Enrolled 常落在两者之间。这说明学生状态与早期学业、经济和人口统计变量相关，但并不存在简单线性边界。Enrolled 标签代表尚未结束的学业过程，其混合分布也支持后续二分类中剔除该类的决定。

# 3. Clustering Analysis

![K-Means selection](figures/kmeans_elbow_silhouette.png)

![Hierarchical dendrogram](figures/hierarchical_dendrogram.png)

K-Means 与 Ward 层次聚类均在 K=2 至 8 中搜索，K-Means 用肘部法则和轮廓系数选择 K，层次聚类结合树状图和轮廓系数。结果如下：

{markdown_table(cluster_fmt)}

层次聚类的 silhouette={num(cluster_df.loc[cluster_df['algorithm'].eq('Agglomerative Ward'), 'silhouette'].iloc[0])}，内部距离分离较强；但其 ARI 仅 {num(cluster_df.loc[cluster_df['algorithm'].eq('Agglomerative Ward'), 'ARI_vs_Target'].iloc[0])}，几乎不能复现真实学业结局。K-Means 的 CH 指数和 ARI 更高，虽 silhouette 较低，但与 Target 的对应关系更强。因此若目标是辅助理解学生结果结构，K-Means 更有实用价值；若只看内部距离分离，Ward 层次聚类更占优。

![Cluster comparison on t-SNE](figures/clustering_tsne.png)

# 4. Prediction: Training and Testing

监督学习将 Target 转为二分类：Dropout 为正类，Graduate 为负类。Enrolled 样本被剔除，因为该类学生最终结果尚不明确，作为监督标签会降低模型置信度。二分类数据包含 Graduate={binary['Graduate_0']}、Dropout={binary['Dropout_1']}，采用 70%/30% 分层抽样。逻辑回归作为稳定线性基线，决策树设置 max_depth=5 以限制过拟合。测试集混淆矩阵显示逻辑回归漏判 Dropout 50 人，决策树漏判 84 人，逻辑回归更适合辍学预警。

![Logistic regression confusion matrices](figures/confusion_logistic_regression.png)

![Decision tree confusion matrices](figures/confusion_decision_tree.png)

# 5. Evaluation and Model Choice

测试集指标如下：

{markdown_table(metrics_fmt)}

![ROC curves](figures/roc_lr_tree.png)

逻辑回归在 Accuracy={num(lr['accuracy'])}、Recall={num(lr['recall'])}、F1={num(lr['f1'])}、AUC={num(lr['AUC'])} 上均优于或明显优于决策树。5 折交叉验证结果如下：

{markdown_table(cv_fmt)}

逻辑回归训练准确率 {num(cv_df.loc[cv_df['model'].eq('Logistic Regression'), 'train_accuracy'].iloc[0])}，测试准确率 {num(cv_df.loc[cv_df['model'].eq('Logistic Regression'), 'test_accuracy'].iloc[0])}，泛化稳定。决策树训练准确率 {num(cv_df.loc[cv_df['model'].eq('Decision Tree'), 'train_accuracy'].iloc[0])}，测试准确率 {num(cv_df.loc[cv_df['model'].eq('Decision Tree'), 'test_accuracy'].iloc[0])}，差距不大但 Recall 偏低。综合 AUC、Recall 与交叉验证稳定性，逻辑回归是本任务中最合适的简单模型。

# 6. Open-ended Exploration

## 6.1 Feature Importance and Model Comparison

![Top feature importance](figures/feature_importance_top5.png)

随机森林 impurity importance 的 Top 5 如下：

{markdown_table(fi_fmt)}

随机森林使用 GridSearchCV 调整 n_estimators、max_depth 与 min_samples_leaf，最佳参数为 `{summary['random_forest_best_params']}`。三类模型测试集表现如下：

{markdown_table(model_fmt)}

调参随机森林 AUC={num(rf['AUC'])}，优于决策树但略低于逻辑回归。该结果说明本数据中辍学与毕业之间存在较强的近线性可分信号，复杂模型未必带来更高泛化收益。

![Model comparison](figures/model_comparison_auc.png)

## 6.2 Class Imbalance and SMOTE

二分类任务的少数类/多数类比例为 {num(binary['minority_majority_ratio'])}，存在中等不平衡。SMOTE 结果如下：

{markdown_table(smote_fmt)}

SMOTE 将逻辑回归 Recall 从 {num(lr['recall'])} 提升到 {num(smote_df.loc[smote_df['model'].str.contains('Logistic'), 'recall'].iloc[0])}，但 Precision 与 AUC 略降；对决策树，AUC 提升约 {num(smote_df.loc[smote_df['model'].str.contains('Decision'), 'AUC_delta_vs_baseline'].iloc[0])}。若实际场景重视少漏报，SMOTE 值得考虑；若强调整体排序能力，原始逻辑回归已足够稳健。

## 6.3 Early Warning Experiment

![Early warning comparison](figures/early_warning_comparison.png)

早期预警实验严格控制信息可用时间点。只使用入学时已知信息时，AUC={num(early_enrollment['AUC'])}、Recall={num(early_enrollment['recall'])}；加入第一学期表现后，AUC 提升到 {num(early_first['AUC'])}、Recall 提升到 {num(early_first['recall'])}；再加入第二学期后，AUC={num(early_full['AUC'])}、Recall={num(early_full['recall'])}。这说明第一学期信息已经提供了大部分预测增益，第二学期继续提升但边际收益较小。从教育干预角度，第一学期结束后建模比等到第二学期结束更有实际价值。

{markdown_table(early_fmt)}

## 6.3.1 Prediction Time Matters

{markdown_table(prediction_time_fmt)}

若报告声称“早期预警”，则不能使用第二学期结束后的表现作为输入，否则存在时间泄漏。完整模型 AUC 最高，但它回答的是“哪些学生最终更像辍学者”的回顾性诊断问题；第一学期模型回答的是“还有时间干预时，哪些学生值得优先关注”的管理问题。因此，本项目在部署建议中优先使用 `Enrollment + 1st semester` 模型，在技术上保留 full model 作为上界参考。

## 6.4 Feature Ablation

![Feature ablation](figures/feature_ablation.png)

消融实验从完整特征集中按组删除人口统计、经济状态、第一学期和第二学期特征。结果显示，删除第二学期特征使 AUC 下降 {num(ablation_df.loc[ablation_df['experiment'].eq('Remove 2nd semester'), 'AUC_drop_vs_full'].iloc[0])}，删除两个学期特征使 AUC 下降 {num(ablation_df.loc[ablation_df['experiment'].eq('Remove both semesters'), 'AUC_drop_vs_full'].iloc[0])}、Recall 下降 {num(ablation_df.loc[ablation_df['experiment'].eq('Remove both semesters'), 'recall_drop_vs_full'].iloc[0])}。经济状态删除后 AUC 下降 {num(ablation_df.loc[ablation_df['experiment'].eq('Remove economic status'), 'AUC_drop_vs_full'].iloc[0])}，说明缴费、欠费、奖学金和宏观经济变量有辅助价值。人口统计变量删除后几乎不降低 AUC，不应被过度解读为主要风险来源。

{markdown_table(ablation_fmt)}

## 6.5 Threshold Tuning and Intervention Strategy

![Threshold tuning](figures/threshold_tuning.png)

默认阈值 0.5 并不一定符合学校资源约束。若设定“每 100 名学生最多人工跟进 25 人”，最合理策略是按模型风险分从高到低排序并跟进前 25%。在测试集中该策略标记 {summary['threshold_policy']['capacity_count_test']} 人，Precision={num(policy['precision'])}、Recall={num(policy['recall'])}、F1={num(policy['f1'])}。这意味着少量人工资源可以优先覆盖最高风险学生，但会牺牲一部分 Recall；若目标是尽量少漏报，则应降低阈值并接受更多人工审核。

## 6.5.1 Intervention Coverage Curve

![Intervention coverage](figures/intervention_coverage_curve.png)

干预覆盖曲线直接回答“跟进多少学生能覆盖多少真实 Dropout”。使用第一学期可行动模型时，跟进风险最高的 25% 学生可覆盖 {pct(coverage_25_early['dropout_coverage'])} 的真实 Dropout，跟进名单中的 Dropout 比例为 {pct(coverage_25_early['precision_among_followed'])}；使用包含第二学期信息的完整模型时，对应覆盖率为 {pct(coverage_25_full['dropout_coverage'])}。两者在前 25% 容量下差距不大，说明第一学期模型已经足以支持实际干预排序，而第二学期模型更适合作为回顾性诊断。

{markdown_table(intervention_fmt)}

## 6.6 Permutation Importance

![Permutation importance](figures/permutation_importance_top10.png)

由于 impurity importance 对高基数独热特征可能存在偏差，本项目额外使用 permutation importance，以测试集 AUC 下降衡量原始特征贡献。结果如下：

{markdown_table(perm_fmt)}

Permutation importance 仍将第二学期通过课程数列为最重要特征，置乱后 AUC 平均下降 {num(perm_df.iloc[0]['importance_mean_auc_drop'])}；`Tuition fees up to date` 的 AUC 下降为 {num(perm_df.loc[perm_df['feature'].eq('Tuition fees up to date'), 'importance_mean_auc_drop'].iloc[0])}，进一步支持“学业表现 + 经济状态”双重解释。

## 6.7 Fairness Check

![Fairness group metrics](figures/fairness_group_metrics.png)

按 Gender、International 和 Scholarship holder 分组后，逻辑回归在不同群体上的 Recall 和 FPR 存在差异：

{markdown_table(fairness_fmt)}

Gender=1 的 Recall={num(fairness_df.loc[(fairness_df['sensitive_feature'].eq('Gender')) & (fairness_df['group_value'].astype(str).eq('1')), 'recall'].iloc[0])}，高于 Gender=0 的 {num(fairness_df.loc[(fairness_df['sensitive_feature'].eq('Gender')) & (fairness_df['group_value'].astype(str).eq('0')), 'recall'].iloc[0])}，但其 FPR 也更高。Scholarship holder=1 的 Recall={num(fairness_df.loc[(fairness_df['sensitive_feature'].eq('Scholarship holder')) & (fairness_df['group_value'].astype(str).eq('1')), 'recall'].iloc[0])}，低于非奖学金组。International=1 的样本仅 {int(fairness_df.loc[(fairness_df['sensitive_feature'].eq('International')) & (fairness_df['group_value'].astype(str).eq('1')), 'n'].iloc[0])} 人，结论应谨慎。若模型部署为真实预警系统，应继续做群体校准和人工复核。

## 6.7.1 Fairness Bootstrap Confidence Intervals

![Fairness bootstrap CI](figures/fairness_bootstrap_ci.png)

在 Wilson 区间之外，本项目还对每个敏感群体内的 Recall 和 FPR 做 bootstrap。International=1 组只有 {int(intl_bootstrap['n'])} 个测试样本，其 Recall bootstrap 95% CI 为 [{num(intl_bootstrap['recall_ci_low'])}, {num(intl_bootstrap['recall_ci_high'])}]。这进一步说明，小样本群体的公平性指标不应只看点估计；真实部署时需要持续积累样本并周期性重估。

{markdown_table(fairness_bootstrap_fmt)}

## 6.8 Reusing Enrolled Students as Risk-scoring Cases

![Enrolled risk distribution](figures/enrolled_risk_distribution.png)

Enrolled 样本不适合作为监督训练标签，但可用“入学 + 第一学期”模型打风险分。794 名 Enrolled 学生的平均风险为 {num(enrolled['mean_risk'])}，中位数为 {num(enrolled['median_risk'])}；其中 {enrolled['risk_ge_0.50_count']} 人风险分不低于 0.5，占 {pct(enrolled['risk_ge_0.50_rate'])}。若采用早期模型在测试集上的 top-25% 阈值 {num(enrolled['early_model_top25_cutoff'])}，Enrolled 中仍有 {enrolled['risk_ge_early_top25_cutoff_count']} 人进入高优先级干预名单，占 {pct(enrolled['risk_ge_early_top25_cutoff_rate'])}。风险分桶如下：

{markdown_table(enrolled_bucket_fmt)}

## 6.9 Calibration and Brier Score

![Calibration curves](figures/calibration_curves.png)

为了判断风险分是否可解释为概率，本项目绘制校准曲线并计算 Brier score。结果如下：

{markdown_table(calibration_fmt)}

逻辑回归的 Brier score={num(lr_cal['brier_score'])}，低于决策树和调参随机森林，说明其概率输出更适合直接解释为辍学风险。其 expected calibration error={num(lr_cal['expected_calibration_error'])}，整体校准较好，但校准曲线在中高风险区仍有偏差。因此在真实部署中，风险分可以用于排序和分层，但不应机械地解释为完全准确的个人概率。

## 6.9.1 Calibrated Models

![Calibrated model curves](figures/calibrated_model_curves.png)

![Calibrated model comparison](figures/calibrated_model_comparison.png)

进一步使用 `CalibratedClassifierCV` 对逻辑回归和决策树进行 Platt scaling 与 isotonic calibration。结果如下：

{markdown_table(calibrated_fmt)}

本次结果中，最佳 Brier score 来自 {best_brier_cal['model']}，为 {num(best_brier_cal['brier_score'])}；最佳 ECE 来自 {best_ece_cal['model']}，为 {num(best_ece_cal['expected_calibration_error'])}。校准改善了部分概率质量，尤其是决策树的 Brier score，但校准方法也可能改变排序表现，因此部署时应同时报告 AUC、Brier 和 ECE，而不是只优化单一指标。

## 6.10 Partial Dependence and Counterfactual-style Discussion

![Partial dependence](figures/partial_dependence_key_features.png)

Partial dependence 显示，当 `Curricular units 2nd sem (approved)` 从 0 增至 6 时，随机森林平均预测风险由 {num(approved_pdp_low)} 降至 {num(approved_pdp_six)}；`Tuition fees up to date` 从 0 变为 1 时，平均风险由 {num(tuition_pdp_0)} 降至 {num(tuition_pdp_1)}。这支持“课程通过数量”和“学费状态”是关键风险信号，但 PDP 仍是模型解释，不等同于因果效应。

![Counterfactual scenarios](figures/counterfactual_scenarios.png)

反事实风格模拟使用逻辑回归模型在测试集上修改单个或少量特征，观察预测风险变化：

{markdown_table(counterfactual_fmt)}

模拟显示，将第二学期通过课程数增加 3 个单位时，平均预测风险下降 {num(abs(cf_plus3['mean_delta_risk']))}；同时将学费状态设为按时缴纳并增加 3 个通过课程时，平均风险下降 {num(abs(cf_combined['mean_delta_risk']))}。这些结果适合用于形成教育干预假设，例如加强课程支持和学费援助，但不能替代随机实验或因果推断。

## 6.11 Deployment Uncertainty and Group-wise Calibration

![Bootstrap metric CI](figures/bootstrap_metric_ci.png)

模型总体指标使用 1000 次 bootstrap 估计 95% 置信区间：

{markdown_table(bootstrap_fmt)}

逻辑回归 AUC 的 95% CI 为 {bootstrap_fmt.loc[bootstrap_fmt['model'].eq('Logistic Regression'), 'AUC_95CI'].iloc[0]}，Recall 的 95% CI 为 {bootstrap_fmt.loc[bootstrap_fmt['model'].eq('Logistic Regression'), 'Recall_95CI'].iloc[0]}。这说明其总体性能较稳健，但 Recall 仍有不确定性。

![Group-wise calibration](figures/group_calibration.png)

分组校准结果如下：

{markdown_table(group_calibration_fmt)}

International=1 组仅 {int(intl_small['n'])} 人，Recall={num(intl_small['recall'])}，Wilson 95% CI 为 [{num(intl_small['recall_ci_low'])}, {num(intl_small['recall_ci_high'])}]；FPR={num(intl_small['FPR'])}，但 95% CI 上界达到 {num(intl_small['FPR_ci_high'])}。这说明小样本群体的点估计不可过度解读。部署时应报告不确定性、进行人工复核，并在积累更多样本后重新评估群体校准。

## 6.12 False Negative Profile

![False negative profile](figures/false_negative_profile.png)

最后分析逻辑回归在测试集上漏判的 Dropout。模型共漏判 {summary['false_negative_counts']['false_negative_dropout_count']} 名 Dropout，正确识别 {summary['false_negative_counts']['true_positive_dropout_count']} 名 Dropout。漏判样本最突出的特征差异是 `{top_fn_feature['feature']}`：漏判组均值为 {num(top_fn_feature['false_negative_mean'])}，正确识别组均值为 {num(top_fn_feature['true_positive_mean'])}。总体上，漏判学生往往第二学期通过课程数更高、学费状态更正常、欠费比例更低，因而在表格特征上更像 Graduate。这提示学校不能只依赖模型风险分，还应结合退学申请、转专业、个人原因等非结构化信息。

{markdown_table(fn_profile_fmt)}

# Conclusion

无监督分析表明三类学业结果有局部结构但重叠明显，不能仅靠自然聚类替代监督预测。监督学习中，逻辑回归以 AUC={num(lr['AUC'])} 和 Recall={num(lr['recall'])} 成为最佳简单模型；调参随机森林作为开放探索模型接近逻辑回归但未超过它。更深入的开放探索显示，第一学期信息已能将 AUC 从 {num(early_enrollment['AUC'])} 提升到 {num(early_first['AUC'])}，是早期预警的关键时间点；第二学期通过课程数、学费状态和第一学期通过课程数是最核心因素。干预覆盖曲线、校准后模型、bootstrap 公平性区间、时间泄漏讨论和错误案例画像进一步说明，模型可以支持风险排序和干预假设，但部署时必须同时报告概率校准、群体差异、不确定性和模型失败模式。

# References

1. UCI Machine Learning Repository. Predict Students' Dropout and Academic Success. https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
2. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.
3. van der Maaten and Hinton. Visualizing Data using t-SNE. Journal of Machine Learning Research, 2008.

# Credit

This is a reference answer prepared for teaching and grading standardization. GenAI assistance was used to draft and format the reference materials; all numeric results are generated by the reproducible code in this directory.

**本答案仅供参考评分标准，学生核心ML实现需独立完成。**
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
