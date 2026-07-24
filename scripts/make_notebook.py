"""Generate the Covertype Jupyter notebook from the audited scripts."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = ROOT / "analysis_covertype.py"
PREPARE_PATH = ROOT / "scripts" / "prepare_data.py"
NOTEBOOK_PATH = ROOT / "project_covertype.ipynb"


def versions_cell() -> str:
    return """\
import importlib

packages = [
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "scipy",
    "nbformat",
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
print("Best model:", summary["best_model_name"])
display(pd.DataFrame([summary["best_model_test_metrics"]]))

tables = [
    "target_counts.csv",
    "wilderness_area_summary.csv",
    "wilderness_cover_distribution.csv",
    "clustering_evaluation.csv",
    "train_test_full_metrics.csv",
    "model_comparison.csv",
    "cross_validation.csv",
    "feature_importance_top20.csv",
    "permutation_importance_top15.csv",
    "feature_group_ablation.csv",
    "calibration_summary.csv",
    "elevation_band_metrics.csv",
    "wilderness_model_metrics.csv",
    "class_report_best_model.csv",
    "computational_cost_estimates.csv",
    "optimization_directions.csv",
    "runtime_profile.csv",
]

for table in tables:
    print(f"outputs/{table}")
    display(pd.read_csv(Path("outputs") / table).head(20))

figures = [
    "target_distribution.png",
    "wilderness_cover_distribution.png",
    "missing_ratios.png",
    "tsne_target.png",
    "kmeans_elbow_silhouette.png",
    "hierarchical_dendrogram.png",
    "clustering_tsne.png",
    "confusion_logistic_regression.png",
    "confusion_decision_tree.png",
    "model_comparison_f1_auc.png",
    "roc_ovr_models.png",
    "feature_importance_top15.png",
    "permutation_importance_top15.png",
    "feature_group_ablation.png",
    "calibration_reliability.png",
    "elevation_band_accuracy.png",
    "wilderness_model_metrics.png",
    "error_confidence.png",
    "class_recall_comparison.png",
    "computational_cost_tiers.png",
]

for figure in figures:
    print(f"figures/{figure}")
    display(Image(filename=str(Path("figures") / figure)))
"""


def main() -> None:
    analysis_code = ANALYSIS_PATH.read_text(encoding="utf-8")
    analysis_code = analysis_code.replace(
        "ROOT = Path(__file__).resolve().parent",
        "ROOT = Path.cwd()  # Jupyter-safe project root",
    )
    prepare_code = PREPARE_PATH.read_text(encoding="utf-8")
    prepare_code = prepare_code.replace(
        "ROOT = Path(__file__).resolve().parents[1]",
        "ROOT = Path.cwd()  # Jupyter-safe project root",
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
            "# DSAA2011 Covertype Project\n\n"
            "This notebook reproduces the full Covertype analysis: data preparation, "
            "preprocessing, t-SNE visualization, clustering, supervised prediction, "
            "evaluation, and open-ended exploration. All stochastic steps use "
            "`random_state=42`."
        ),
        nbf.v4.new_markdown_cell("## Library Versions"),
        nbf.v4.new_code_cell(versions_cell()),
        nbf.v4.new_markdown_cell(
            "## Data Preparation\n\n"
            "The UCI raw gzip file is converted into `data/covertype.csv.gz` with "
            "official feature names. The preparation code is embedded here so the "
            "submission notebook does not depend on an external script."
        ),
        nbf.v4.new_code_cell(prepare_code),
        nbf.v4.new_markdown_cell(
            "## Complete Analysis Code\n\n"
            "The following cell is the same implementation as `analysis_covertype.py`, "
            "with only the project-root path adjusted for notebook execution."
        ),
        nbf.v4.new_code_cell(analysis_code),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(display_results_cell()),
    ]

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
