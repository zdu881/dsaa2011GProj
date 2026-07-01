const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const {
  imageSizingContain,
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./build/slides/pptxgenjs_helpers");

const ROOT = __dirname;
const OUT = path.join(ROOT, "presentation_student_dropout.pptx");
const summary = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs", "summary.json"), "utf8"));

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "DSAA2011 Student Dropout Project";
pptx.subject = "Student Dropout Project";
pptx.title = "DSAA2011 Student Dropout Project";
pptx.company = "HKUST(GZ)";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Noto Sans CJK SC",
  bodyFontFace: "Noto Sans CJK SC",
  lang: "en-US",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

const W = 13.333;
const H = 7.5;
const C = {
  ink: "1F2933",
  muted: "5B6770",
  line: "D7DEE7",
  blue: "2F6F9F",
  teal: "28666E",
  red: "B13F4A",
  gold: "C99A2E",
  green: "4F8A5B",
  pale: "F6F8FA",
  white: "FFFFFF",
};

function fig(name) {
  return path.join(ROOT, "figures", name);
}

function fmt(x, digits = 3) {
  return Number(x).toFixed(digits);
}

function pct(x) {
  return `${(100 * Number(x)).toFixed(1)}%`;
}

function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.45,
    y: 0.28,
    w: 9.2,
    h: 0.34,
    fontFace: "Noto Sans CJK SC",
    fontSize: 20,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.47,
      y: 0.68,
      w: 10.2,
      h: 0.24,
      fontSize: 8.5,
      color: C.muted,
      margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.45,
    y: 0.99,
    w: 12.45,
    h: 0,
    line: { color: C.line, width: 0.75 },
  });
}

function addFooter(slide, n) {
  slide.addText(`DSAA2011 Student Dropout | ${n}/6`, {
    x: 10.35,
    y: 7.08,
    w: 2.5,
    h: 0.22,
    fontSize: 7.5,
    color: C.muted,
    align: "right",
    margin: 0,
  });
}

function addBullets(slide, bullets, x, y, w, lineH = 0.48) {
  bullets.forEach((text, idx) => {
    slide.addText(text, {
      x,
      y: y + idx * lineH,
      w,
      h: 0.34,
      fontSize: 11,
      color: C.ink,
      bullet: { type: "ul" },
      margin: 0.02,
      breakLine: false,
    });
  });
}

function addMetric(slide, label, value, x, y, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w: 1.95,
    h: 0.82,
    rectRadius: 0.04,
    fill: { color: C.pale },
    line: { color: C.line, width: 0.6 },
  });
  slide.addText(value, {
    x: x + 0.12,
    y: y + 0.13,
    w: 1.7,
    h: 0.25,
    fontSize: 15,
    bold: true,
    color,
    margin: 0,
  });
  slide.addText(label, {
    x: x + 0.12,
    y: y + 0.48,
    w: 1.7,
    h: 0.18,
    fontSize: 7.5,
    color: C.muted,
    margin: 0,
  });
}

function addImage(slide, file, x, y, w, h) {
  slide.addImage({ path: file, ...imageSizingContain(file, x, y, w, h) });
}

function simpleTable(slide, rows, x, y, colW, rowH = 0.38) {
  rows.forEach((row, r) => {
    row.forEach((text, c) => {
      slide.addShape(pptx.ShapeType.rect, {
        x: x + colW.slice(0, c).reduce((a, b) => a + b, 0),
        y: y + r * rowH,
        w: colW[c],
        h: rowH,
        fill: { color: r === 0 ? "E9EEF5" : C.white },
        line: { color: C.line, width: 0.5 },
      });
      slide.addText(String(text), {
        x: x + colW.slice(0, c).reduce((a, b) => a + b, 0) + 0.06,
        y: y + r * rowH + 0.08,
        w: colW[c] - 0.12,
        h: rowH - 0.14,
        fontSize: r === 0 ? 7.8 : 7.4,
        bold: r === 0,
        color: C.ink,
        margin: 0,
      });
    });
  });
}

function validate(slide) {
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

const pre = summary.preprocessing;
const counts = summary.target_counts;
const lr = summary.model_comparison.find((row) => row.model === "Logistic Regression");
const tree = summary.model_comparison.find((row) => row.model === "Decision Tree");
const rf = summary.model_comparison.find((row) => row.model === "Random Forest, tuned");
const km = summary.clustering_evaluation.find((row) => row.algorithm === "K-Means");
const hc = summary.clustering_evaluation.find((row) => row.algorithm === "Agglomerative Ward");
const cvLR = summary.cross_validation.find((row) => row.model === "Logistic Regression");
const cvTree = summary.cross_validation.find((row) => row.model === "Decision Tree");
const smoteLR = summary.smote_summary.find((row) => row.model.startsWith("Logistic"));
const smoteTree = summary.smote_summary.find((row) => row.model.startsWith("Decision"));
const earlyEnrollment = summary.early_warning_comparison.find((row) => row.experiment === "Enrollment only");
const earlyFirst = summary.early_warning_comparison.find((row) => row.experiment === "Enrollment + 1st semester");
const earlyFull = summary.early_warning_comparison.find((row) => row.experiment === "Enrollment + 1st + 2nd semester");
const policy = summary.threshold_policy.policy_metrics;
const enrolledRisk = summary.enrolled_risk_summary;
const removeSecond = summary.feature_ablation.find((row) => row.experiment === "Remove 2nd semester");
const removeBothSem = summary.feature_ablation.find((row) => row.experiment === "Remove both semesters");
const lrCalibration = summary.calibration_summary.find((row) => row.model === "Logistic Regression");
const lrBootstrap = summary.bootstrap_metric_ci.find((row) => row.model === "Logistic Regression");
const intlSmall = summary.fairness_by_group.find((row) => row.sensitive_feature === "International" && String(row.group_value) === "1");
const cfCombined = summary.counterfactual_scenarios.find((row) => row.scenario === "Tuition up-to-date and approved units +3");
const coverage25Early = summary.intervention_coverage_curve.find((row) => row.model.startsWith("Actionable") && Number(row.follow_up_rate) === 0.25);
const coverage25Full = summary.intervention_coverage_curve.find((row) => row.model.startsWith("Full") && Number(row.follow_up_rate) === 0.25);
const bestCalibrated = [...summary.calibrated_model_comparison].sort((a, b) => a.brier_score - b.brier_score)[0];
const fnCounts = summary.false_negative_counts;
const topFnFeature = summary.false_negative_profile[0];

// Slide 1
{
  const slide = pptx.addSlide();
  addTitle(slide, "Introduction", "Student dropout risk analysis with reproducible ML workflow");
  addMetric(slide, "Students", String(pre.raw_shape[0]), 0.62, 1.24, C.blue);
  addMetric(slide, "Raw features", String(pre.raw_shape[1] - 1), 2.76, 1.24, C.teal);
  addMetric(slide, "Processed features", String(pre.feature_shape_after_preprocessing[1]), 4.9, 1.24, C.green);
  addBullets(
    slide,
    [
      `Target distribution: Dropout ${counts.Dropout}, Enrolled ${counts.Enrolled}, Graduate ${counts.Graduate}.`,
      "Pipeline: preprocessing, t-SNE, clustering, binary prediction, validation, open exploration.",
      "Main supervised task: detect Dropout against Graduate after excluding uncertain Enrolled labels.",
    ],
    0.72,
    2.38,
    5.75,
    0.58,
  );
  addImage(slide, fig("target_distribution.png"), 7.0, 1.28, 5.55, 4.95);
  addFooter(slide, 1);
  validate(slide);
}

// Slide 2
{
  const slide = pptx.addSlide();
  addTitle(slide, "t-SNE & Clustering", "Local structure is visible, but true labels remain strongly overlapped");
  addImage(slide, fig("tsne_target.png"), 0.55, 1.18, 5.65, 4.55);
  addImage(slide, fig("clustering_tsne.png"), 6.28, 1.15, 6.42, 2.95);
  simpleTable(
    slide,
    [
      ["Algorithm", "K", "Sil.", "ARI"],
      ["K-Means", summary.kmeans_best_k, fmt(km.silhouette), fmt(km.ARI_vs_Target)],
      ["Ward", summary.hierarchical_best_k, fmt(hc.silhouette), fmt(hc.ARI_vs_Target)],
    ],
    6.5,
    4.42,
    [1.55, 0.45, 0.72, 0.72],
    0.36,
  );
  addBullets(
    slide,
    [
      "Both methods select K=2, suggesting a coarse high-risk vs lower-risk split.",
      "Ward has stronger internal separation; K-Means aligns better with true labels by ARI.",
      "Low ARI confirms that natural clusters are not equivalent to academic outcomes.",
    ],
    10.12,
    4.42,
    2.42,
    0.5,
  );
  addFooter(slide, 2);
  validate(slide);
}

// Slide 3
{
  const slide = pptx.addSlide();
  addTitle(slide, "Prediction Results", "Confusion matrices for train, test, and full binary data");
  addImage(slide, fig("confusion_logistic_regression.png"), 0.55, 1.2, 6.08, 3.52);
  addImage(slide, fig("confusion_decision_tree.png"), 6.75, 1.2, 6.08, 3.52);
  addBullets(
    slide,
    [
      "Binary target: Dropout is positive, Graduate is negative.",
      "Logistic regression test matrix: 376 true Dropout detected, 50 missed.",
      "Decision tree test matrix: 342 true Dropout detected, 84 missed.",
    ],
    0.95,
    5.28,
    11.4,
    0.42,
  );
  addFooter(slide, 3);
  validate(slide);
}

// Slide 4
{
  const slide = pptx.addSlide();
  addTitle(slide, "Evaluation & Intervention", "ROC/AUC plus coverage under limited advising capacity");
  addImage(slide, fig("roc_lr_tree.png"), 0.58, 1.2, 4.7, 3.9);
  addImage(slide, fig("intervention_coverage_curve.png"), 5.3, 1.2, 4.75, 3.9);
  simpleTable(
    slide,
    [
      ["Model", "Acc.", "Recall", "F1", "AUC"],
      ["Logistic", fmt(lr.accuracy), fmt(lr.recall), fmt(lr.f1), fmt(lr.AUC)],
      ["Tree", fmt(tree.accuracy), fmt(tree.recall), fmt(tree.f1), fmt(tree.AUC)],
      ["RF tuned", fmt(rf.accuracy), fmt(rf.recall), fmt(rf.f1), fmt(rf.AUC)],
    ],
    10.22,
    1.38,
    [0.86, 0.48, 0.56, 0.48, 0.48],
    0.43,
  );
  addBullets(
    slide,
    [
      `5-fold CV AUC: logistic ${fmt(cvLR.cv_auc_mean)}±${fmt(cvLR.cv_auc_std)}, tree ${fmt(cvTree.cv_auc_mean)}±${fmt(cvTree.cv_auc_std)}.`,
      `Top 25% actionable first-semester follow-up covers ${fmt(coverage25Early.dropout_coverage)} of true Dropout cases.`,
      `Full second-semester model covers ${fmt(coverage25Full.dropout_coverage)} at the same capacity, but is later for prevention.`,
    ],
    0.88,
    5.48,
    11.6,
    0.48,
  );
  addFooter(slide, 4);
  validate(slide);
}

// Slide 5
{
  const slide = pptx.addSlide();
  addTitle(slide, "Early Warning & Ablation", "The first semester is the practical intervention window");
  addImage(slide, fig("early_warning_comparison.png"), 0.55, 1.18, 5.95, 3.9);
  addImage(slide, fig("feature_ablation.png"), 6.82, 1.18, 5.95, 3.9);
  addBullets(
    slide,
    [
      `Enrollment-only AUC ${fmt(earlyEnrollment.AUC)} rises to ${fmt(earlyFirst.AUC)} after first-semester data.`,
      `Adding second-semester data reaches AUC ${fmt(earlyFull.AUC)}, useful but later for intervention.`,
      `Removing both semesters drops recall by ${fmt(removeBothSem.recall_drop_vs_full)}; removing second semester drops AUC by ${fmt(removeSecond.AUC_drop_vs_full)}.`,
    ],
    0.8,
    5.62,
    11.85,
    0.42,
  );
  addFooter(slide, 5);
  validate(slide);
}

// Slide 6
{
  const slide = pptx.addSlide();
  addTitle(slide, "Trustworthy Explanation & Risk", "Calibrated probability, uncertainty, and failure-mode analysis");
  addImage(slide, fig("calibrated_model_curves.png"), 0.55, 1.15, 4.25, 3.85);
  addImage(slide, fig("false_negative_profile.png"), 4.98, 1.15, 4.95, 3.85);
  slide.addShape(pptx.ShapeType.rect, {
    x: 10.22,
    y: 1.22,
    w: 2.58,
    h: 3.62,
    fill: { color: "F9FAFB" },
    line: { color: C.line, width: 0.7 },
  });
  slide.addText("Deployment Checks", {
    x: 10.4,
    y: 1.48,
    w: 2.12,
    h: 0.25,
    fontSize: 11.5,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  slide.addText(`${fmt(bestCalibrated.brier_score)}`, {
    x: 10.4,
    y: 1.93,
    w: 2.12,
    h: 0.35,
    fontSize: 20,
    bold: true,
    color: C.gold,
    margin: 0,
  });
  slide.addText("best Brier score", {
    x: 10.4,
    y: 2.34,
    w: 2.12,
    h: 0.2,
    fontSize: 7.5,
    color: C.muted,
    margin: 0,
  });
  slide.addText(`${fmt(lrBootstrap.AUC_ci_low)}-${fmt(lrBootstrap.AUC_ci_high)}`, {
    x: 10.4,
    y: 2.88,
    w: 2.12,
    h: 0.3,
    fontSize: 13,
    bold: true,
    color: C.red,
    margin: 0,
  });
  slide.addText("LR AUC 95% CI", {
    x: 10.4,
    y: 3.25,
    w: 2.12,
    h: 0.18,
    fontSize: 7.3,
    color: C.muted,
    margin: 0,
  });
  slide.addText(`${fnCounts.false_negative_dropout_count} FN`, {
    x: 10.4,
    y: 3.73,
    w: 2.12,
    h: 0.28,
    fontSize: 13,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  slide.addText("missed Dropout cases", {
    x: 10.4,
    y: 4.08,
    w: 2.12,
    h: 0.18,
    fontSize: 7.3,
    color: C.muted,
    margin: 0,
  });
  addBullets(
    slide,
    [
      `Counterfactual-style scenario: tuition up-to-date and +3 approved units lowers predicted risk by ${fmt(Math.abs(cfCombined.mean_delta_risk))} on average.`,
      `False negatives look stronger academically; top FN contrast is ${topFnFeature.feature}.`,
    ],
    0.88,
    5.42,
    11.6,
    0.46,
  );
  slide.addText("Dataset: UCI Predict Students' Dropout and Academic Success | Tools: pandas, scikit-learn, matplotlib, seaborn, imbalanced-learn", {
    x: 0.9,
    y: 6.73,
    w: 10.9,
    h: 0.18,
    fontSize: 7.2,
    color: C.muted,
    margin: 0,
  });
  addFooter(slide, 6);
  validate(slide);
}

pptx.writeFile({ fileName: OUT });
