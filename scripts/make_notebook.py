"""Generate the project Jupyter notebook from the audited analysis script."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = ROOT / "analysis_student_dropout.py"
NOTEBOOK_PATH = ROOT / "project_student_dropout.ipynb"


def current_versions_cell() -> str:
    return """\
import importlib

packages = [
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "scipy",
    "imblearn",
]

for package in packages:
    module = importlib.import_module(package)
    print(f"{package}=={module.__version__}")
"""


def display_results_cell() -> str:
    return """\
import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

summary = json.loads(Path("outputs/summary.json").read_text(encoding="utf-8"))

print("Preprocessing summary")
display(pd.DataFrame([summary["preprocessing"]]))

print("Clustering evaluation")
display(pd.read_csv("outputs/clustering_evaluation.csv"))

print("Binary classification test metrics")
display(pd.read_csv("outputs/test_metrics_lr_tree.csv"))

print("Cross-validation")
display(pd.read_csv("outputs/cross_validation.csv"))

print("Model comparison")
display(pd.read_csv("outputs/model_comparison.csv"))

print("SMOTE summary")
display(pd.read_csv("outputs/smote_summary.csv"))

print("Early warning comparison")
display(pd.read_csv("outputs/early_warning_comparison.csv"))

print("Feature ablation")
display(pd.read_csv("outputs/feature_ablation.csv"))

print("Threshold tuning")
display(pd.read_csv("outputs/threshold_tuning.csv"))

print("Intervention coverage curve")
display(pd.read_csv("outputs/intervention_coverage_curve.csv"))

print("Prediction time discussion")
display(pd.read_csv("outputs/prediction_time_discussion.csv"))

print("Permutation importance")
display(pd.read_csv("outputs/permutation_importance_top10.csv"))

print("Fairness by group")
display(pd.read_csv("outputs/fairness_by_group.csv"))

print("Enrolled risk summary")
display(pd.read_csv("outputs/enrolled_risk_summary.csv"))

print("Calibration summary")
display(pd.read_csv("outputs/calibration_summary.csv"))

print("Calibrated model comparison")
display(pd.read_csv("outputs/calibrated_model_comparison.csv"))

print("Bootstrap metric confidence intervals")
display(pd.read_csv("outputs/bootstrap_metric_ci.csv"))

print("Partial dependence")
display(pd.read_csv("outputs/partial_dependence.csv"))

print("Counterfactual scenarios")
display(pd.read_csv("outputs/counterfactual_scenarios.csv"))

print("Group calibration summary")
display(pd.read_csv("outputs/group_calibration_summary.csv"))

print("Fairness bootstrap confidence intervals")
display(pd.read_csv("outputs/fairness_bootstrap_ci.csv"))

print("False negative profile")
display(pd.read_csv("outputs/false_negative_profile.csv"))

for fig_path in [
    "figures/target_distribution.png",
    "figures/tsne_target.png",
    "figures/kmeans_elbow_silhouette.png",
    "figures/hierarchical_dendrogram.png",
    "figures/clustering_tsne.png",
    "figures/confusion_logistic_regression.png",
    "figures/confusion_decision_tree.png",
    "figures/roc_lr_tree.png",
    "figures/feature_importance_top5.png",
    "figures/model_comparison_auc.png",
    "figures/early_warning_comparison.png",
    "figures/feature_ablation.png",
    "figures/threshold_tuning.png",
    "figures/intervention_coverage_curve.png",
    "figures/permutation_importance_top10.png",
    "figures/fairness_group_metrics.png",
    "figures/enrolled_risk_distribution.png",
    "figures/calibration_curves.png",
    "figures/calibrated_model_curves.png",
    "figures/calibrated_model_comparison.png",
    "figures/bootstrap_metric_ci.png",
    "figures/partial_dependence_key_features.png",
    "figures/counterfactual_scenarios.png",
    "figures/group_calibration.png",
    "figures/fairness_bootstrap_ci.png",
    "figures/false_negative_profile.png",
]:
    print(fig_path)
    display(Image(filename=fig_path))
"""


def main() -> None:
    analysis_code = ANALYSIS_PATH.read_text(encoding="utf-8")
    analysis_code = analysis_code.replace(
        'ROOT = Path(__file__).resolve().parent',
        'ROOT = Path.cwd()  # Jupyter-safe project root',
    )

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# DSAA2011 Student Dropout Project\n\n"
            "This notebook is the complete reproducible implementation for the "
            "Student Dropout dataset. It follows the required order: data "
            "preprocessing, t-SNE visualization, clustering, supervised "
            "prediction, evaluation, and open-ended exploration. All random "
            "operations use `random_state=42`."
        ),
        nbf.v4.new_markdown_cell(
            "## Library Versions\n\n"
            "The current environment versions are printed below. The generated "
            "`requirements_student_dropout.txt` records the same pinned versions."
        ),
        nbf.v4.new_code_cell(current_versions_cell()),
        nbf.v4.new_markdown_cell(
            "## Data Preparation\n\n"
            "The official UCI archive is semicolon-delimited. The preparation "
            "script downloads the zip if necessary, preserves the raw file under "
            "`data/`, strips whitespace from column names, and writes the "
            "comma-separated `student_dropout.csv` expected by the project."
        ),
        nbf.v4.new_code_cell("%run scripts/prepare_data.py"),
        nbf.v4.new_markdown_cell(
            "## Complete Analysis Code\n\n"
            "The cell below contains the full implementation used to generate all "
            "figures and tables. It is intentionally kept in one audited block so "
            "the notebook and source script remain consistent."
        ),
        nbf.v4.new_code_cell(analysis_code),
        nbf.v4.new_markdown_cell("## Results Display"),
        nbf.v4.new_code_cell(display_results_cell()),
    ]

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
