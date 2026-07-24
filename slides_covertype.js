"use strict";

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const {
  imageSizingContain,
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./build/slides/pptxgenjs_helpers");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "DSAA2011 Covertype Project";
pptx.subject = "Forest cover type classification";
pptx.title = "DSAA2011 Covertype Project";
pptx.company = "HKUST(GZ)";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";
pptx.margin = 0;

const ROOT = __dirname;
const FIG = path.join(ROOT, "figures");
const OUT = path.join(ROOT, "outputs");
const summary = JSON.parse(fs.readFileSync(path.join(OUT, "summary.json"), "utf8"));

const colors = {
  ink: "1F2A33",
  muted: "66717C",
  teal: "2A9D8F",
  green: "4C9A6A",
  red: "C44E52",
  amber: "D59B2D",
  blue: "4C72B0",
  purple: "8172B3",
  line: "D8DEE4",
  bg: "F7F8FA",
  white: "FFFFFF",
};

function csvRows(file) {
  const text = fs.readFileSync(path.join(OUT, file), "utf8").trim();
  const [headerLine, ...lines] = text.split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = values[idx];
    });
    return row;
  });
}

function fig(name) {
  return path.join(FIG, name);
}

function fmt(x, digits = 3) {
  const value = Number(x);
  return Number.isFinite(value) ? value.toFixed(digits) : "";
}

function pct(x, digits = 1) {
  const value = Number(x);
  return Number.isFinite(value) ? `${(100 * value).toFixed(digits)}%` : "";
}

function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.45,
    y: 0.28,
    w: 8.8,
    h: 0.38,
    fontFace: "Aptos Display",
    fontSize: 22,
    bold: true,
    color: colors.ink,
    margin: 0,
    breakLine: false,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.46,
      y: 0.71,
      w: 10.8,
      h: 0.26,
      fontSize: 9.8,
      color: colors.muted,
      margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.45,
    y: 1.05,
    w: 12.42,
    h: 0,
    line: { color: colors.line, width: 1 },
  });
}

function addFooter(slide, num) {
  slide.addText(`DSAA2011 Covertype Project | ${num}`, {
    x: 10.65,
    y: 7.05,
    w: 2.2,
    h: 0.18,
    fontSize: 7.5,
    color: colors.muted,
    align: "right",
    margin: 0,
  });
}

function addMetric(slide, label, value, x, y, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w: 2.3,
    h: 0.82,
    rectRadius: 0.06,
    fill: { color: colors.white },
    line: { color: colors.line, width: 0.8 },
  });
  slide.addText(value, {
    x: x + 0.14,
    y: y + 0.12,
    w: 2.0,
    h: 0.32,
    fontSize: 18,
    bold: true,
    color,
    margin: 0,
  });
  slide.addText(label, {
    x: x + 0.14,
    y: y + 0.49,
    w: 2.0,
    h: 0.18,
    fontSize: 7.8,
    color: colors.muted,
    margin: 0,
  });
}

function addBullets(slide, items, x, y, w, h) {
  slide.addText(
    items.map((text) => ({ text, options: { bullet: { indent: 12 }, hanging: 4 } })),
    {
      x,
      y,
      w,
      h,
      fontSize: 12,
      color: colors.ink,
      breakLine: false,
      valign: "mid",
      paraSpaceAfterPt: 5,
      margin: 0.05,
    }
  );
}

function addImage(slide, name, x, y, w, h) {
  const image = fig(name);
  slide.addImage({ path: image, ...imageSizingContain(image, x, y, w, h) });
}

function validate(slide) {
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

const best = summary.best_model_test_metrics;
const targetRows = csvRows("target_counts.csv");
const modelRows = csvRows("model_comparison.csv").filter((row) => row.split === "test");
const clusterRows = csvRows("clustering_evaluation.csv");
const wildernessRows = csvRows("wilderness_area_summary.csv");
const wildernessMetricRows = csvRows("wilderness_model_metrics.csv");
const topFeatureRows = csvRows("feature_importance_top20.csv").slice(0, 5);
const ablationRows = csvRows("feature_group_ablation.csv").slice(0, 4);
const calibrationRows = csvRows("calibration_summary.csv");
const costRows = summary.computational_scaling.cost_estimates;

// Slide 1
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.bg };
  slide.addText("Covertype Forest Cover Analysis", {
    x: 0.55,
    y: 0.55,
    w: 7.4,
    h: 0.65,
    fontFace: "Aptos Display",
    fontSize: 27,
    bold: true,
    color: colors.ink,
    margin: 0,
  });
  slide.addText("DSAA2011 machine learning project", {
    x: 0.58,
    y: 1.22,
    w: 4.8,
    h: 0.26,
    fontSize: 12,
    color: colors.muted,
    margin: 0,
  });
  addMetric(slide, "rows", summary.dataset.rows.toLocaleString(), 0.58, 2.0, colors.teal);
  addMetric(slide, "features", String(summary.dataset.features), 3.05, 2.0, colors.blue);
  addMetric(slide, "cover classes", "7", 5.5, 2.0, colors.purple);
  addMetric(slide, "best macro-F1", fmt(best.f1_macro), 0.58, 3.08, colors.green);
  addMetric(slide, "best accuracy", fmt(best.accuracy), 3.05, 3.08, colors.green);
  addMetric(slide, "macro AUC", fmt(best.AUC_ovr_macro), 5.5, 3.08, colors.green);
  addBullets(
    slide,
    [
      "Natural clusters reveal terrain structure but do not recover the seven labels.",
      "Wilderness Area analysis adds a regional robustness view.",
      "Random Forest is the best classifier on the stratified test set.",
      "Elevation, road/fire/hydrology distances, hillshade, soil context, and region explain errors.",
    ],
    0.62,
    4.35,
    6.8,
    1.35
  );
  addImage(slide, "target_distribution.png", 8.05, 0.95, 4.75, 4.05);
  addFooter(slide, 1);
  validate(slide);
}

// Slide 2
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.white };
  addTitle(slide, "Dataset and Preprocessing", "UCI Covertype: cartographic measurements with a seven-class forest cover target");
  addImage(slide, "missing_ratios.png", 0.55, 1.35, 4.9, 3.3);
  const classText = targetRows
    .map((row) => `${row.Cover_Type} ${row.Cover_Type_Name}: ${Number(row.count).toLocaleString()} (${pct(row.proportion)})`)
    .join("\n");
  slide.addText(classText, {
    x: 6.0,
    y: 1.35,
    w: 6.2,
    h: 1.7,
    fontSize: 10.5,
    color: colors.ink,
    margin: 0.04,
    breakLine: false,
  });
  addBullets(
    slide,
    [
      "No missing values after parsing; maximum missing ratio is 0.000.",
      "10 continuous terrain features are standardized for linear and distance-based methods.",
      "4 wilderness and 40 soil indicators are already one-hot encoded.",
      "Large-scale steps use stratified samples to preserve class proportions while keeping runtime reproducible.",
    ],
    6.0,
    3.45,
    6.45,
    1.7
  );
  addFooter(slide, 2);
  validate(slide);
}

// Slide 3
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.bg };
  addTitle(slide, "Visualization and Clustering", "t-SNE shows partial local structure; clusters are not the same as cover labels");
  addImage(slide, "tsne_target.png", 0.45, 1.32, 5.85, 4.8);
  addImage(slide, "clustering_tsne.png", 6.55, 1.2, 6.15, 3.0);
  const clusterText = clusterRows
    .map(
      (row) =>
        `${row.algorithm}: k=${row.k}, silhouette=${fmt(row.silhouette)}, ARI target=${fmt(row.ARI_vs_Target)}, ARI wild=${fmt(row.ARI_vs_Wilderness)}`
    )
    .join("\n");
  slide.addText(clusterText, {
    x: 6.75,
    y: 4.55,
    w: 5.55,
    h: 0.7,
    fontSize: 11.5,
    color: colors.ink,
    margin: 0,
  });
  addBullets(
    slide,
    [
      "Near-zero ARI versus Cover_Type only means the natural partitions do not reproduce the supervised labels.",
      "ARI versus Wilderness is also modest, so clusters are better read as coarse terrain/region mixtures.",
    ],
    6.75,
    5.55,
    5.55,
    0.8
  );
  addFooter(slide, 3);
  validate(slide);
}

// Slide 4
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.white };
  addTitle(slide, "Simple Prediction Models", "Logistic regression is a linear baseline; decision tree captures nonlinear splits");
  addImage(slide, "confusion_logistic_regression.png", 0.45, 1.25, 6.1, 2.6);
  addImage(slide, "confusion_decision_tree.png", 6.8, 1.25, 6.05, 2.6);
  const simpleRows = csvRows("train_test_full_metrics.csv").filter((row) => row.split === "test");
  const simpleText = simpleRows
    .map((row) => `${row.model}: accuracy=${fmt(row.accuracy)}, macro-F1=${fmt(row.f1_macro)}, macro AUC=${fmt(row.AUC_ovr_macro)}`)
    .join("\n");
  slide.addText(simpleText, {
    x: 0.75,
    y: 4.18,
    w: 6.4,
    h: 0.65,
    fontSize: 12.5,
    color: colors.ink,
    margin: 0,
  });
  addBullets(
    slide,
    [
      "Logistic regression reaches high macro AUC but weak class-level F1.",
      "The decision tree improves nonlinear separation but still lags tree ensembles.",
      "Both simple models are evaluated on train, test, and the full dataset after training.",
    ],
    0.75,
    5.18,
    11.4,
    1.0
  );
  addFooter(slide, 4);
  validate(slide);
}

// Slide 5
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.bg };
  addTitle(slide, "Model Choice", "Random Forest gives the best test macro-F1 and macro one-vs-rest AUC");
  addImage(slide, "model_comparison_f1_auc.png", 0.55, 1.25, 5.9, 3.4);
  addImage(slide, "roc_ovr_models.png", 6.8, 1.25, 5.5, 3.4);
  const modelText = modelRows
    .map((row) => `${row.model}: acc=${fmt(row.accuracy)}, macro-F1=${fmt(row.f1_macro)}, AUC=${fmt(row.AUC_ovr_macro)}`)
    .join("\n");
  slide.addText(modelText, {
    x: 0.75,
    y: 5.08,
    w: 6.3,
    h: 1.0,
    fontSize: 11.2,
    color: colors.ink,
    margin: 0,
  });
  addBullets(
    slide,
    [
      "Cross-validation keeps the same ranking: Random Forest first, HistGradientBoosting second.",
      "Macro metrics matter because Cottonwood/Willow and Aspen are rare.",
    ],
    7.15,
    5.1,
    5.1,
    0.9
  );
  addFooter(slide, 5);
  validate(slide);
}

// Slide 6
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.white };
  addTitle(slide, "What Drives Cover Type?", "Topography dominates, but soil indicators recover meaningful extra signal");
  addImage(slide, "feature_importance_top15.png", 0.55, 1.25, 5.7, 4.7);
  addImage(slide, "feature_group_ablation.png", 6.75, 1.25, 5.75, 3.4);
  const featureText = topFeatureRows.map((row, idx) => `${idx + 1}. ${row.feature} (${fmt(row.importance)})`).join("\n");
  slide.addText(featureText, {
    x: 6.9,
    y: 4.92,
    w: 2.75,
    h: 1.1,
    fontSize: 9.6,
    color: colors.ink,
    margin: 0,
  });
  const ablText = ablationRows.map((row) => `${row.experiment}: F1=${fmt(row.f1_macro)}`).join("\n");
  slide.addText(ablText, {
    x: 9.85,
    y: 4.92,
    w: 2.7,
    h: 1.1,
    fontSize: 9.6,
    color: colors.ink,
    margin: 0,
  });
  addFooter(slide, 6);
  validate(slide);
}

// Slide 7
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.white };
  addTitle(slide, "Wilderness Area Extension", "Regional composition and held-out performance explain where the model is easier or harder");
  addImage(slide, "wilderness_cover_distribution.png", 0.55, 1.22, 5.9, 4.05);
  addImage(slide, "wilderness_model_metrics.png", 6.85, 1.22, 5.55, 3.4);
  const wildernessText = wildernessRows
    .map(
      (row) =>
        `${row.Wilderness_Area_Name}: ${pct(row.proportion)}, dominant ${row.dominant_cover_type_name} (${pct(row.dominant_cover_type_share)})`
    )
    .join("\n");
  slide.addText(wildernessText, {
    x: 0.75,
    y: 5.55,
    w: 5.9,
    h: 0.82,
    fontSize: 8.8,
    color: colors.ink,
    margin: 0,
  });
  const weakestArea = wildernessMetricRows
    .slice()
    .sort((a, b) => Number(a.f1_macro) - Number(b.f1_macro))[0];
  addBullets(
    slide,
    [
      `Hardest held-out area: ${weakestArea.Wilderness_Area_Name}, macro-F1 ${fmt(weakestArea.f1_macro)}.`,
      "Regional evaluation checks robustness beyond aggregate test accuracy.",
      "Wilderness indicators alone are weak predictors, but the area split is useful for interpretation.",
    ],
    7.0,
    4.95,
    5.35,
    1.2
  );
  addFooter(slide, 7);
  validate(slide);
}

// Slide 8
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.bg };
  addTitle(slide, "Calibration, Error Review, and Takeaways", "Accuracy is strong, but uncertainty and terrain bands still matter");
  addImage(slide, "calibration_reliability.png", 0.55, 1.25, 4.0, 3.35);
  addImage(slide, "elevation_band_accuracy.png", 4.75, 1.25, 4.0, 3.35);
  addImage(slide, "class_recall_comparison.png", 8.95, 1.25, 3.75, 3.35);
  const calBest = calibrationRows[0];
  addBullets(
    slide,
    [
      `Best classifier: Random Forest with accuracy ${fmt(best.accuracy)}, macro-F1 ${fmt(best.f1_macro)}, macro AUC ${fmt(best.AUC_ovr_macro)}.`,
      `Best calibration ECE: ${calBest.model} at ${fmt(calBest.expected_calibration_error)}.`,
      "Unsupervised clusters are useful for terrain/region exploration, but not for replacing supervised labels.",
      "Report macro metrics, class recall, calibration, elevation-band, and wilderness-area performance together.",
    ],
    0.85,
    5.05,
    11.75,
    1.15
  );
  addFooter(slide, 8);
  validate(slide);
}

// Slide 9
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.white };
  addTitle(slide, "Scalable Computation", "Large-data workflow choices keep expensive steps reproducible");
  addImage(slide, "computational_cost_tiers.png", 0.55, 1.22, 5.9, 4.15);
  addMetric(slide, "naive int64 estimate", `${fmt(summary.computational_scaling.baseline_memory_mb, 1)} MB`, 6.95, 1.35, colors.blue);
  addMetric(slide, "optimized memory", `${fmt(summary.computational_scaling.optimized_memory_mb, 1)} MB`, 9.55, 1.35, colors.teal);
  const highCost = costRows
    .filter((row) => Number(row.relative_cost_score) >= 8)
    .map((row) => `${row.module}: score ${row.relative_cost_score}`)
    .join("\n");
  slide.addText(highCost, {
    x: 7.0,
    y: 2.55,
    w: 5.2,
    h: 0.9,
    fontSize: 11.2,
    color: colors.ink,
    margin: 0,
  });
  addBullets(
    slide,
    [
      `Memory reduction after downcast: ${pct(summary.computational_scaling.memory_reduction_pct)}.`,
      "Use full data for descriptive statistics, not for every diagnostic.",
      "Use stratified samples for t-SNE, Ward clustering, cross-validation, and permutation importance.",
      "Run PCA before distance-based visualization and clustering.",
      "Cache heavy stages with manifests; train scalable ensembles with parallelism.",
    ],
    7.0,
    3.78,
    5.45,
    1.5
  );
  addFooter(slide, 9);
  validate(slide);
}

pptx.writeFile({ fileName: path.join(ROOT, "presentation_covertype.pptx") });
