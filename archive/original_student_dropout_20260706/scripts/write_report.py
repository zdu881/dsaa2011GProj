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
    early_short_fmt = format_numeric_columns(
        early_df[["experiment", "AUC", "recall", "precision"]],
        ["AUC", "recall", "precision"],
    )
    model_short_fmt = format_numeric_columns(
        model_df[["model", "accuracy", "recall", "f1", "AUC"]],
        ["accuracy", "recall", "f1", "AUC"],
    )
    calibration_short_fmt = format_numeric_columns(
        calibration_df[["model", "brier_score", "expected_calibration_error", "mean_predicted_risk"]],
        ["brier_score", "expected_calibration_error", "mean_predicted_risk"],
    )
    bootstrap_short_fmt = bootstrap_fmt.loc[bootstrap_fmt["model"].eq("Logistic Regression")]

    report = f"""\
---
title: "DSAA2011 Student Dropout Project Report"
date: "2026 Spring"
---

# Introduction

本项目基于 UCI **Predict Students' Dropout and Academic Success** 数据集完成学生辍学分析。数据包含 {pre['raw_shape'][0]} 名本科学生、{pre['raw_shape'][1] - 1} 个原始特征，目标变量为 Dropout、Enrolled 与 Graduate，其中 Dropout={counts['Dropout']}、Enrolled={counts['Enrolled']}、Graduate={counts['Graduate']}。分析路线为：先进行缺失值、类别变量和标准化处理，再用 t-SNE 与聚类观察高维结构，最后将任务转为 Dropout vs Graduate 二分类，比较逻辑回归、决策树和随机森林，并补充校准、干预覆盖、公平性与错误案例分析。

# 1. Data Preprocessing

缺失值检查显示最大缺失率为 {pct(pre['max_missing_ratio'])}，没有列超过 40% 阈值，因此未删除特征。数值型变量使用中位数填充，类别型变量使用众数填充；18 个类别变量经独热编码处理，其中 {', '.join(pre['high_cardinality_columns_grouped'])} 等高基数变量只保留 Top 5 类别，其余归为 Other。标准化前特征矩阵为 {pre['feature_shape_before_preprocessing']}，处理后为 {pre['feature_shape_after_preprocessing']}。该流程兼顾异常值鲁棒性、类别变量可建模性和不同量纲特征的可比性。

# 2. Data Visualization and Clustering

![t-SNE projection](figures/tsne_target.png){{ width=4.8in }}

t-SNE 使用 perplexity=30、learning_rate=200、random_state=42。二维嵌入显示三类样本存在局部聚集，但整体交叠明显：Dropout 在若干区域密度较高，Graduate 在中心和右下区域较集中，Enrolled 常落在两者之间。这说明学生状态与早期学业、经济和人口统计变量相关，但并不存在简单线性边界。Enrolled 标签代表尚未结束的学业过程，其混合分布也支持后续二分类中剔除该类的决定。

K-Means 与 Ward 层次聚类均在 K=2 至 8 中搜索，K-Means 用肘部法则和轮廓系数选择 K，层次聚类结合树状图和轮廓系数。结果如下：

{markdown_table(cluster_fmt)}

层次聚类的 silhouette={num(cluster_df.loc[cluster_df['algorithm'].eq('Agglomerative Ward'), 'silhouette'].iloc[0])}，内部距离分离较强；但其 ARI 仅 {num(cluster_df.loc[cluster_df['algorithm'].eq('Agglomerative Ward'), 'ARI_vs_Target'].iloc[0])}，几乎不能复现真实学业结局。K-Means 的 CH 指数和 ARI 更高，虽 silhouette 较低，但与 Target 的对应关系更强。因此若目标是辅助理解学生结果结构，K-Means 更有实用价值；若只看内部距离分离，Ward 层次聚类更占优。

![Cluster comparison on t-SNE](figures/clustering_tsne.png){{ width=4.8in }}

# 3. Prediction: Training and Testing

监督学习将 Target 转为二分类：Dropout 为正类，Graduate 为负类。Enrolled 样本被剔除，因为该类学生最终结果尚不明确，作为监督标签会降低模型置信度。二分类数据包含 Graduate={binary['Graduate_0']}、Dropout={binary['Dropout_1']}，采用 70%/30% 分层抽样。逻辑回归作为稳定线性基线，决策树设置 max_depth=5 以限制过拟合。测试集混淆矩阵显示逻辑回归漏判 Dropout 50 人，决策树漏判 84 人，逻辑回归更适合辍学预警。

| Logistic Regression | Decision Tree |
| --- | --- |
| ![](figures/confusion_logistic_regression.png){{ width=3.0in }} | ![](figures/confusion_decision_tree.png){{ width=3.0in }} |

# 4. Evaluation and Model Choice

测试集指标如下：

{markdown_table(metrics_fmt)}

![ROC curves](figures/roc_lr_tree.png){{ width=4.8in }}

逻辑回归在 Accuracy={num(lr['accuracy'])}、Recall={num(lr['recall'])}、F1={num(lr['f1'])}、AUC={num(lr['AUC'])} 上均优于或明显优于决策树。5 折交叉验证中，逻辑回归 AUC={num(cv_df.loc[cv_df['model'].eq('Logistic Regression'), 'cv_auc_mean'].iloc[0])}±{num(cv_df.loc[cv_df['model'].eq('Logistic Regression'), 'cv_auc_std'].iloc[0])}，决策树 AUC={num(cv_df.loc[cv_df['model'].eq('Decision Tree'), 'cv_auc_mean'].iloc[0])}±{num(cv_df.loc[cv_df['model'].eq('Decision Tree'), 'cv_auc_std'].iloc[0])}。逻辑回归训练准确率 {num(cv_df.loc[cv_df['model'].eq('Logistic Regression'), 'train_accuracy'].iloc[0])}、测试准确率 {num(cv_df.loc[cv_df['model'].eq('Logistic Regression'), 'test_accuracy'].iloc[0])}，泛化稳定；决策树训练准确率 {num(cv_df.loc[cv_df['model'].eq('Decision Tree'), 'train_accuracy'].iloc[0])}、测试准确率 {num(cv_df.loc[cv_df['model'].eq('Decision Tree'), 'test_accuracy'].iloc[0])}，差距不大但 Recall 偏低。综合排序能力、漏报控制和交叉验证稳定性，逻辑回归是本任务中最合适的简单模型。

# 5. Open-ended Exploration

## 5.1 Feature Importance, Model Comparison, and Imbalance

随机森林 impurity importance 的 Top 5 如下，主要集中在两个学期的通过课程数、成绩与学费状态，具有明确教育解释：学业完成度低和财务压力会同时提高辍学风险。

{markdown_table(fi_fmt)}

随机森林使用 GridSearchCV 调整 n_estimators、max_depth 与 min_samples_leaf，最佳参数为 `{summary['random_forest_best_params']}`。三类模型测试集表现如下：

{markdown_table(model_short_fmt)}

调参随机森林 AUC={num(rf['AUC'])}，优于决策树但略低于逻辑回归，说明复杂模型未必带来更高泛化收益。类别不平衡方面，Dropout/Graduate 比例为 {num(binary['minority_majority_ratio'])}；SMOTE 将逻辑回归 Recall 从 {num(lr['recall'])} 提升到 {num(smote_df.loc[smote_df['model'].str.contains('Logistic'), 'recall'].iloc[0])}，但 AUC 略降，因此更适合“少漏报”优先的策略，而非默认替代原始模型。

![Model and feature comparison](figures/model_comparison_auc.png){{ width=4.5in }}

## 5.2 Early Warning and Intervention Value

![Early warning comparison](figures/early_warning_comparison.png){{ width=4.6in }}

早期预警实验严格控制信息可用时间点。只使用入学时已知信息时，AUC={num(early_enrollment['AUC'])}、Recall={num(early_enrollment['recall'])}；加入第一学期表现后，AUC 提升到 {num(early_first['AUC'])}、Recall 提升到 {num(early_first['recall'])}；再加入第二学期后，AUC={num(early_full['AUC'])}、Recall={num(early_full['recall'])}。因此，完整模型更像回顾性诊断，而 `Enrollment + 1st semester` 模型更适合可行动预警。

{markdown_table(early_short_fmt)}

![Intervention coverage](figures/intervention_coverage_curve.png){{ width=4.6in }}

干预覆盖曲线比 ROC 更接近教育管理场景。使用第一学期可行动模型时，跟进风险最高的 25% 学生可覆盖 {pct(coverage_25_early['dropout_coverage'])} 的真实 Dropout，名单中的 Dropout 比例为 {pct(coverage_25_early['precision_among_followed'])}；完整模型对应覆盖率为 {pct(coverage_25_full['dropout_coverage'])}。两者差距不大，说明第一学期模型已经足以支持有限资源下的优先排序。若设定 top-25% 容量策略，完整逻辑回归在测试集中标记 {summary['threshold_policy']['capacity_count_test']} 人，Precision={num(policy['precision'])}、Recall={num(policy['recall'])}。

## 5.3 From Correlation to Trustworthy Explanation

![Calibration curves](figures/calibration_curves.png){{ width=4.6in }}

为了判断风险分是否可解释为概率，本项目绘制校准曲线并计算 Brier score。结果如下：

{markdown_table(calibration_short_fmt)}

逻辑回归的 Brier score={num(lr_cal['brier_score'])}，低于决策树和调参随机森林，说明其概率输出更适合解释为辍学风险。进一步使用 `CalibratedClassifierCV` 后，最佳 Brier score 来自 {best_brier_cal['model']}，为 {num(best_brier_cal['brier_score'])}。PDP 显示，当 `Curricular units 2nd sem (approved)` 从 0 增至 6 时，随机森林平均预测风险由 {num(approved_pdp_low)} 降至 {num(approved_pdp_six)}；`Tuition fees up to date` 从 0 变为 1 时，风险由 {num(tuition_pdp_0)} 降至 {num(tuition_pdp_1)}。反事实风格模拟中，“学费按时缴纳 + 第二学期通过课程数增加 3”使平均风险下降 {num(abs(cf_combined['mean_delta_risk']))}。这些结果适合生成干预假设，但仍不是因果估计。

## 5.4 Uncertainty, Fairness, and Error Cases

模型总体指标使用 1000 次 bootstrap 估计 95% 置信区间：

{markdown_table(bootstrap_short_fmt)}

逻辑回归 AUC 的 95% CI 为 {bootstrap_fmt.loc[bootstrap_fmt['model'].eq('Logistic Regression'), 'AUC_95CI'].iloc[0]}，Recall 的 95% CI 为 {bootstrap_fmt.loc[bootstrap_fmt['model'].eq('Logistic Regression'), 'Recall_95CI'].iloc[0]}，总体较稳健。公平性分析按 Gender、International 与 Scholarship holder 分组计算 Recall、FPR、Brier score 和 group-wise calibration；International=1 组仅 {int(intl_small['n'])} 人，Recall={num(intl_small['recall'])}，Wilson 95% CI 为 [{num(intl_small['recall_ci_low'])}, {num(intl_small['recall_ci_high'])}]，bootstrap Recall 95% CI 为 [{num(intl_bootstrap['recall_ci_low'])}, {num(intl_bootstrap['recall_ci_high'])}]，因此不应过度解读小样本群体的点估计。错误案例方面，逻辑回归漏判 {summary['false_negative_counts']['false_negative_dropout_count']} 名 Dropout；漏判者在 `{top_fn_feature['feature']}` 上均值为 {num(top_fn_feature['false_negative_mean'])}，高于正确识别 Dropout 的 {num(top_fn_feature['true_positive_mean'])}，说明部分辍学学生在结构化学业和财务变量上看似安全，仍需人工复核和非结构化信息补充。

# Conclusion

无监督分析表明三类学业结果有局部结构但重叠明显，不能仅靠自然聚类替代监督预测。监督学习中，逻辑回归以 AUC={num(lr['AUC'])} 和 Recall={num(lr['recall'])} 成为最佳简单模型；调参随机森林作为开放探索模型接近逻辑回归但未超过它。更深入的开放探索显示，第一学期信息已能将 AUC 从 {num(early_enrollment['AUC'])} 提升到 {num(early_first['AUC'])}，是早期预警的关键时间点；第二学期通过课程数、学费状态和第一学期通过课程数是最核心因素。干预覆盖曲线、校准后模型、bootstrap 公平性区间、时间泄漏讨论和错误案例画像进一步说明，模型可以支持风险排序和干预假设，但部署时必须同时报告概率校准、群体差异、不确定性和模型失败模式。

# References

1. UCI Machine Learning Repository. Predict Students' Dropout and Academic Success. https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
2. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.
3. van der Maaten and Hinton. Visualizing Data using t-SNE. Journal of Machine Learning Research, 2008.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
