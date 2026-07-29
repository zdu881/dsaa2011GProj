#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const AdmZip = require("adm-zip");
const { DOMParser, XMLSerializer } = require("@xmldom/xmldom");
const xpath = require("xpath");
const { imageSizingContain } = require("./pptxgenjs_helpers/image");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");
// Existing-slide OOXML edits are validated with the bundled renderer/tester.
void warnIfSlideHasOverlaps;
void warnIfSlideElementsOutOfBounds;

const input = path.resolve(process.argv[2] || "source_v9.pptx");
const output = path.resolve(process.argv[3] || "Forest Cover Type Analysis - DSAA2011 v10.pptx");
const figureRoot = [
  path.join(__dirname, "figures"),
  path.resolve(__dirname, "../..", "figures"),
  path.resolve(__dirname, "..", "figures"),
].find((candidate) => fs.existsSync(candidate));
if (!figureRoot) throw new Error("Could not locate the figures directory");
const figures = (name) => path.join(figureRoot, name);

const NS = {
  a: "http://schemas.openxmlformats.org/drawingml/2006/main",
  p: "http://schemas.openxmlformats.org/presentationml/2006/main",
  r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
};
const REL_NS = {
  pr: "http://schemas.openxmlformats.org/package/2006/relationships",
};
const q = xpath.useNamespaces(NS);
const qr = xpath.useNamespaces(REL_NS);
const parser = new DOMParser();
const serializer = new XMLSerializer();
const zip = new AdmZip(input);
const EMU = 914400;
const SLIDE_W = 13.3333333333;
const SLIDE_H = 7.5;
const inch = (v) => Math.round(v * EMU);
let mediaSerial = 1;

function openSlide(n) {
  const entry = zip.getEntry(`ppt/slides/slide${n}.xml`);
  const relsEntry = zip.getEntry(`ppt/slides/_rels/slide${n}.xml.rels`);
  if (!entry || !relsEntry) throw new Error(`Missing slide ${n} or relationships`);
  return {
    n,
    entry,
    relsEntry,
    doc: parser.parseFromString(entry.getData().toString("utf8"), "application/xml"),
    relsDoc: parser.parseFromString(relsEntry.getData().toString("utf8"), "application/xml"),
  };
}

function saveSlide(ctx) {
  ctx.entry.setData(Buffer.from(serializer.serializeToString(ctx.doc), "utf8"));
  ctx.relsEntry.setData(Buffer.from(serializer.serializeToString(ctx.relsDoc), "utf8"));
}

function nodeText(node) {
  return q(".//a:t", node).map((n) => n.textContent).join(" ").replace(/\s+/g, " ").trim();
}

function findShape(doc, fragment) {
  const shape = q("//p:sp", doc).find((s) => nodeText(s).includes(fragment));
  if (!shape) throw new Error(`Shape not found: ${fragment}`);
  return shape;
}

function findShapeById(doc, id) {
  const shape = q("//p:sp | //p:pic | //p:graphicFrame", doc).find((s) => {
    const nv = q(".//p:cNvPr", s)[0];
    return nv && Number(nv.getAttribute("id")) === id;
  });
  if (!shape) throw new Error(`Shape id not found: ${id}`);
  return shape;
}

function findTable(doc, fragment) {
  const frame = q("//p:graphicFrame[a:graphic/a:graphicData/a:tbl]", doc)
    .find((f) => nodeText(f).includes(fragment));
  if (!frame) throw new Error(`Table not found: ${fragment}`);
  return frame;
}

function xfrmOf(node) {
  return q("./p:spPr/a:xfrm | ./p:xfrm", node)[0];
}

function geometry(node) {
  const xfrm = xfrmOf(node);
  const off = q("./a:off", xfrm)[0];
  const ext = q("./a:ext", xfrm)[0];
  return {
    x: Number(off.getAttribute("x")) / EMU,
    y: Number(off.getAttribute("y")) / EMU,
    w: Number(ext.getAttribute("cx")) / EMU,
    h: Number(ext.getAttribute("cy")) / EMU,
  };
}

function setGeometry(node, { x, y, w, h }) {
  const xfrm = xfrmOf(node);
  const off = q("./a:off", xfrm)[0];
  const ext = q("./a:ext", xfrm)[0];
  if (x !== undefined) off.setAttribute("x", String(inch(x)));
  if (y !== undefined) off.setAttribute("y", String(inch(y)));
  if (w !== undefined) ext.setAttribute("cx", String(inch(w)));
  if (h !== undefined) ext.setAttribute("cy", String(inch(h)));
}

function removeNode(node) {
  if (node?.parentNode) node.parentNode.removeChild(node);
}

function baseRPr(textBody) {
  const src = q(".//a:rPr", textBody)[0] || q(".//a:endParaRPr", textBody)[0];
  const rpr = textBody.ownerDocument.createElement("a:rPr");
  if (!src) return rpr;
  for (let i = 0; i < src.attributes.length; i++) {
    const attr = src.attributes.item(i);
    rpr.setAttribute(attr.name, attr.value);
  }
  for (const child of Array.from(src.childNodes)) rpr.appendChild(child.cloneNode(true));
  return rpr;
}

function setColor(rpr, color) {
  const existing = q("./a:solidFill/a:srgbClr", rpr)[0];
  if (existing) {
    existing.setAttribute("val", color);
    return;
  }
  const fill = rpr.ownerDocument.createElement("a:solidFill");
  const clr = rpr.ownerDocument.createElement("a:srgbClr");
  clr.setAttribute("val", color);
  fill.appendChild(clr);
  rpr.insertBefore(fill, rpr.firstChild);
}

function styleRPr(rpr, opts = {}) {
  if (opts.fontSize !== undefined) rpr.setAttribute("sz", String(Math.round(opts.fontSize * 100)));
  if (opts.bold !== undefined) rpr.setAttribute("b", opts.bold ? "1" : "0");
  if (opts.color) setColor(rpr, opts.color);
}

function makeParagraph(doc, template, spec) {
  const p = doc.createElement("a:p");
  const pPr = doc.createElement("a:pPr");
  if (spec.align) pPr.setAttribute("algn", spec.align);
  if (spec.bullet) {
    pPr.setAttribute("marL", String(spec.marginLeft || 190500));
    pPr.setAttribute("indent", String(spec.indent || -127000));
    const bu = doc.createElement("a:buChar");
    bu.setAttribute("char", "•");
    pPr.appendChild(bu);
  }
  if (spec.spaceAfter !== undefined) {
    const aft = doc.createElement("a:spcAft");
    const pts = doc.createElement("a:spcPts");
    pts.setAttribute("val", String(Math.round(spec.spaceAfter * 100)));
    aft.appendChild(pts);
    pPr.appendChild(aft);
  }
  p.appendChild(pPr);
  for (const rs of spec.runs || [{ text: spec.text || "" }]) {
    const r = doc.createElement("a:r");
    const rpr = template.cloneNode(true);
    styleRPr(rpr, { bold: false, ...spec, ...rs });
    const t = doc.createElement("a:t");
    if (/^\s|\s$/.test(rs.text || "")) t.setAttribute("xml:space", "preserve");
    t.appendChild(doc.createTextNode(rs.text || ""));
    r.appendChild(rpr);
    r.appendChild(t);
    p.appendChild(r);
  }
  const end = doc.createElement("a:endParaRPr");
  for (let i = 0; i < template.attributes.length; i++) {
    const attr = template.attributes.item(i);
    end.setAttribute(attr.name, attr.value);
  }
  for (const child of Array.from(template.childNodes)) end.appendChild(child.cloneNode(true));
  styleRPr(end, { bold: false, ...spec });
  p.appendChild(end);
  return p;
}

function setTextBody(textBody, paragraphs) {
  const template = baseRPr(textBody);
  for (const child of Array.from(textBody.childNodes)) {
    if (child.nodeType === 1 && (child.nodeName === "a:p" || child.localName === "p")) textBody.removeChild(child);
  }
  for (const spec of paragraphs) textBody.appendChild(makeParagraph(textBody.ownerDocument, template, spec));
}

function setShapeText(shape, paragraphs) {
  const body = Array.from(shape.childNodes).find((n) => n.nodeName === "p:txBody") || q("./p:txBody", shape)[0];
  if (!body) throw new Error("Shape has no text body");
  setTextBody(body, paragraphs);
}

function setBodyMargins(shape, margin = 45000) {
  const bp = q("./p:txBody/a:bodyPr", shape)[0];
  if (!bp) return;
  for (const a of ["lIns", "rIns", "tIns", "bIns"]) bp.setAttribute(a, String(margin));
}

function setCellText(cell, text, opts = {}) {
  const body = Array.from(cell.childNodes).find((n) => n.nodeName === "a:txBody") || q("./a:txBody", cell)[0];
  setTextBody(body, [{ text, fontSize: opts.fontSize || 8, bold: !!opts.bold, align: opts.align || "ctr", color: opts.color }]);
}

function setTable(frame, rows, widths, geom, fontSize = 8.2) {
  const tbl = q(".//a:tbl", frame)[0];
  const tr = q("./a:tr", tbl);
  while (tr.length > rows.length) {
    removeNode(tr.pop());
  }
  const current = q("./a:tr", tbl);
  rows.forEach((values, ri) => {
    const cells = q("./a:tc", current[ri]);
    values.forEach((value, ci) => setCellText(cells[ci], value, {
      fontSize: ri === 0 ? fontSize + 0.6 : fontSize,
      bold: ri === 0 || (ri > 0 && (ci === 0 || (ri === 3 && [1, 4, 7].includes(ci)))),
      align: ci === 0 ? "l" : "ctr",
    }));
    current[ri].setAttribute("h", String(inch(ri === 0 ? 0.52 : 0.56)));
  });
  const cols = q(".//a:tblGrid/a:gridCol", frame);
  widths.forEach((w, i) => cols[i].setAttribute("w", String(inch(w))));
  q(".//a:tc/a:tcPr", frame).forEach((tcPr) => {
    for (const a of ["marL", "marR", "marT", "marB"]) tcPr.setAttribute(a, "18000");
  });
  setGeometry(frame, geom);
}

function nextShapeId(doc) {
  return Math.max(...q("//p:cNvPr", doc).map((n) => Number(n.getAttribute("id")) || 0)) + 1;
}

function refreshIdentity(node, doc, name) {
  const nv = q(".//p:cNvPr", node)[0];
  nv.setAttribute("id", String(nextShapeId(doc)));
  nv.setAttribute("name", name);
  const ext = q("./a:extLst", nv)[0];
  if (ext) removeNode(ext);
}

function nextRelId(relsDoc) {
  const ids = qr("//pr:Relationship", relsDoc).map((r) => Number((r.getAttribute("Id") || "rId0").replace("rId", "")) || 0);
  return `rId${Math.max(0, ...ids) + 1}`;
}

function addImageRelationship(ctx, imagePath, label) {
  const ext = path.extname(imagePath).toLowerCase() || ".png";
  const mediaName = `v10_s${ctx.n}_${String(mediaSerial++).padStart(2, "0")}_${label.replace(/[^a-z0-9]+/gi, "_")}${ext}`;
  zip.addFile(`ppt/media/${mediaName}`, fs.readFileSync(imagePath));
  const rid = nextRelId(ctx.relsDoc);
  const rel = ctx.relsDoc.createElementNS(REL_NS.pr, "Relationship");
  rel.setAttribute("Id", rid);
  rel.setAttribute("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image");
  rel.setAttribute("Target", `../media/${mediaName}`);
  ctx.relsDoc.documentElement.appendChild(rel);
  return rid;
}

function setPictureImage(ctx, pic, imagePath, label, box) {
  const rid = addImageRelationship(ctx, imagePath, label);
  const blip = q(".//a:blip", pic)[0];
  blip.setAttributeNS(NS.r, "r:embed", rid);
  const fitted = imageSizingContain(imagePath, box.x, box.y, box.w, box.h);
  setGeometry(pic, fitted);
}

function addPicture(ctx, template, imagePath, label, box) {
  const pic = template.cloneNode(true);
  refreshIdentity(pic, ctx.doc, label);
  setPictureImage(ctx, pic, imagePath, label, box);
  q("//p:spTree", ctx.doc)[0].appendChild(pic);
  return pic;
}

function setShapeColors(shape, fill, line) {
  const spPr = q("./p:spPr", shape)[0];
  const f = q("./a:solidFill/a:srgbClr", spPr)[0];
  if (f && fill) f.setAttribute("val", fill);
  const l = q("./a:ln/a:solidFill/a:srgbClr", spPr)[0];
  if (l && line) l.setAttribute("val", line);
}

function addCard(ctx, bgTemplate, textTemplate, geom, heading, body, colors) {
  const bg = bgTemplate.cloneNode(true);
  refreshIdentity(bg, ctx.doc, `${heading} card`);
  setGeometry(bg, geom);
  setShapeColors(bg, colors.fill, colors.line);
  const tx = textTemplate.cloneNode(true);
  refreshIdentity(tx, ctx.doc, `${heading} text`);
  setGeometry(tx, { x: geom.x + 0.18, y: geom.y + 0.18, w: geom.w - 0.36, h: geom.h - 0.32 });
  setBodyMargins(tx, 22000);
  setShapeText(tx, [
    { text: heading, fontSize: 13, bold: true, color: colors.text, spaceAfter: 2 },
    { text: body, fontSize: 10.2, color: "475569" },
  ]);
  const tree = q("//p:spTree", ctx.doc)[0];
  tree.appendChild(bg);
  tree.appendChild(tx);
}

// Slide 1 — authors and course context.
{
  const ctx = openSlide(1);
  const subtitle = findShape(ctx.doc, "DSAA2011 Summer Project Presentation");
  setGeometry(subtitle, { x: 2.0, y: 4.75, w: 9.33, h: 1.05 });
  setShapeText(subtitle, [
    { text: "DSAA2011 Machine Learning Course Project", fontSize: 17, color: "B7D6CF", align: "ctr", spaceAfter: 3 },
    { text: "Zeyun DU  ·  Zixiang QIN  ·  Chunfeng GAO", fontSize: 13, color: "D8EAE6", align: "ctr" },
  ]);
  saveSlide(ctx);
}

// Slide 2 — precise class imbalance and modeling-sample context.
{
  const ctx = openSlide(2);
  setShapeText(findShape(ctx.doc, "Pine and spruce/fir account"), [{
    runs: [
      { text: "Classes 1 and 2 account for 85.3% of all observations. ", bold: true },
      { text: "Cottonwood/Willow is only 0.5%, so model selection emphasizes macro-F1 rather than accuracy alone. The supervised modeling sample contains 97,950 observations (≈17% of the full dataset)." },
    ],
    fontSize: 10.5,
    color: "374151",
  }]);
  saveSlide(ctx);
}

// Slide 3 — official class label.
{
  const ctx = openSlide(3);
  const cottonwood = findShape(ctx.doc, "Typical riparian deciduous");
  const old = nodeText(cottonwood).replace("04 Cottonwood", "04 Cottonwood/Willow").replace("04 | Cottonwood", "04 | Cottonwood/Willow");
  setShapeText(cottonwood, [{ text: old, fontSize: 7.2, color: "374151" }]);
  saveSlide(ctx);
}

// Slide 4 — correct binary-feature category and leakage-safe preprocessing.
{
  const ctx = openSlide(4);
  const frame = findTable(ctx.doc, "Representative Examples");
  const rows = q(".//a:tbl/a:tr", frame);
  const cells = q("./a:tc", rows[2]);
  [
    "Binary indicators",
    "44 (4 wilderness, 40 soil)",
    "Rawah, Neota, Comanche Peak, Cache la Poudre; Soil_Type_1–40",
    "Keep as 0/1; mode-imputation ready; no scaling",
  ].forEach((v, i) => setCellText(cells[i], v, { fontSize: 7.8, align: i === 0 ? "l" : "ctr" }));
  setShapeText(findShape(ctx.doc, "Preserves sparsity"), [{
    text: "Leakage control: StandardScaler is fitted only on the relevant analysis/training subset; all binary indicators remain 0/1.",
    fontSize: 9.4,
    color: "475569",
  }]);
  saveSlide(ctx);
}

// Slide 5 — report-aligned t-SNE interpretation.
{
  const ctx = openSlide(5);
  setShapeText(findShape(ctx.doc, "03 t-SNE Data Visualization"), [{ text: "03A t-SNE Data Visualization", fontSize: 30, bold: true, color: "111827" }]);
  const replacements = [
    ["Dominant Cluster Formation", "Local Structure, Not Clean Clusters", "Classes 1 and 2 overlap across much of the embedding; prevalence does not imply separability."],
    ["Highly Distinct Class Signature", "Rare-Class Pockets", "Rare cover types form local pockets, but they are not globally isolated from the dominant classes."],
    ["Feature Overlap Between Classes", "High-Elevation Signal", "Krummholz and some Spruce/Fir observations occupy more distinctive high-elevation regions."],
    ["Non-Linear Separability", "Visualization Caveat", "t-SNE preserves local neighborhoods—not global distances—and cannot justify a specific number of clusters."],
  ];
  for (const [fragment, heading, body] of replacements) {
    const shape = findShape(ctx.doc, fragment);
    setShapeText(shape, [{
      runs: [{ text: `${heading} — `, bold: true }, { text: body }],
      fontSize: 10.8,
      color: "475569",
    }]);
  }
  setShapeText(findShape(ctx.doc, "We use stratified sampling"), [{
    text: "5,999 stratified observations  •  continuous variables standardized  •  PCA 54→30 components  •  t-SNE used for visualization only",
    fontSize: 10.5,
    color: "475569",
    align: "ctr",
  }]);
  saveSlide(ctx);
}

// Slide 6 — replace t-SNE cluster projection with the report's selection diagnostics.
{
  const ctx = openSlide(6);
  setShapeText(findShape(ctx.doc, "03 Clustering Results"), [{ text: "03B Clustering Analysis", fontSize: 28, bold: true, color: "111827" }]);
  const mainPic = q("//p:pic[.//a:blip[@r:embed]]", ctx.doc)[0];
  setPictureImage(ctx, mainPic, figures("kmeans_elbow_silhouette.png"), "kmeans_selection", { x: 0.45, y: 0.92, w: 6.15, h: 2.25 });
  addPicture(ctx, mainPic, figures("hierarchical_dendrogram.png"), "ward_dendrogram", { x: 6.75, y: 0.92, w: 6.15, h: 2.25 });
  q("//p:pic[not(.//a:blip[@r:embed])]", ctx.doc).forEach(removeNode);
  const overlay = q("//p:sp", ctx.doc).find((s) => (q(".//p:cNvPr", s)[0]?.getAttribute("name") || "").includes("Ward K4"));
  removeNode(overlay);
  const caption = findShape(ctx.doc, "Figure 1:");
  setGeometry(caption, { x: 0.45, y: 3.15, w: 12.45, h: 0.45 });
  setShapeText(caption, [{ text: "Internal diagnostics favor K=3 for MiniBatchKMeans and K=4 for Ward; neither recovers the seven ecological labels.", fontSize: 9.4, color: "64748B", align: "ctr" }]);
  const bgs = [findShapeById(ctx.doc, 5), findShapeById(ctx.doc, 8), findShapeById(ctx.doc, 11), findShapeById(ctx.doc, 14)];
  const txs = [findShapeById(ctx.doc, 7), findShapeById(ctx.doc, 10), findShapeById(ctx.doc, 13), findShapeById(ctx.doc, 16)];
  const geoms = [
    { x: 0.55, y: 3.78, w: 5.95, h: 1.28 }, { x: 6.82, y: 3.78, w: 5.95, h: 1.28 },
    { x: 0.55, y: 5.30, w: 5.95, h: 1.28 }, { x: 6.82, y: 5.30, w: 5.95, h: 1.28 },
  ];
  geoms.forEach((g, i) => {
    setGeometry(bgs[i], g);
    setGeometry(txs[i], { x: g.x + 0.25, y: g.y + 0.15, w: g.w - 0.5, h: g.h - 0.25 });
    setBodyMargins(txs[i], 22000);
  });
  const content = [
    ["MiniBatchKMeans", "K=3 · silhouette=0.182 · CH=1317.241 · ARI=0.075"],
    ["Ward Hierarchical", "K=4 · silhouette=0.156 · CH=406.764 · ARI=0.103"],
    ["Why Labels Are Not Recovered", "Euclidean compactness on 10 continuous + 44 binary variables is not aligned with ecological class definitions."],
    ["Distance-Metric Check", "Ward + Gower reaches ARI=0.215 at K=7—better, but still far below supervised Random Forest F1=0.908."],
  ];
  content.forEach(([h, b], i) => setShapeText(txs[i], [
    { text: h, fontSize: 12.5, bold: true, color: "111827", spaceAfter: 2 },
    { text: b, fontSize: 9.4, color: "475569" },
  ]));
  saveSlide(ctx);
}

// Slide 7 — fixed regularized baseline and full sampling context.
{
  const ctx = openSlide(7);
  setShapeText(findShape(ctx.doc, "Decision Tree (Optimized)"), [{ text: "Decision Tree (Regularized)", fontSize: 16, bold: true, color: "059669" }]);
  setShapeText(findShape(ctx.doc, "Config: Max Depth"), [{ text: "Config: max_depth=24, min_samples_leaf=15 (fixed baseline)", fontSize: 9, color: "475569" }]);
  setShapeText(findShape(ctx.doc, "Split: 70% Training"), [{
    runs: [
      { text: "Modeling sample: ", bold: true }, { text: "97,950 stratified observations from the full 581,012. " },
      { text: "Split: ", bold: true }, { text: "68,565 train / 29,385 held-out test (70/30)." },
    ],
    fontSize: 9.2,
    color: "475569",
  }]);
  saveSlide(ctx);
}

// Slide 8 — one row per model, report figures, and statistically careful selection statement.
{
  const ctx = openSlide(8);
  const title = findShape(ctx.doc, "Evaluation and Choice of Prediction");
  setGeometry(title, { x: 0.45, y: 0.12, w: 12.2, h: 0.65 });
  setShapeText(title, [{ text: "05 Evaluation and Model Selection", fontSize: 27, bold: true, color: "111827" }]);
  const frame = findTable(ctx.doc, "Precision (Macro)");
  setTable(frame, [
    ["Model", "Test Acc", "Macro Prec.", "Macro Recall", "Macro-F1", "95% CI", "CV-F1", "Macro AUC"],
    ["Logistic Regression", ".676", ".646", ".706", ".660", "[.654, .665]", ".661±.003", ".947"],
    ["Decision Tree", ".823", ".803", ".846", ".819", "[.812, .826]", ".811±.004", ".969"],
    ["Random Forest", ".908", ".903", ".913", ".908", "[.904, .911]", ".902±.001", ".992"],
    ["HistGradientBoosting", ".895", ".896", ".901", ".898", "[.894, .902]", ".895±.003", ".991"],
  ], [1.35, 0.75, 0.9, 0.9, 0.75, 1.25, 1.25, 0.95], { x: 0.45, y: 1.35, w: 8.1, h: 3.05 }, 7.7);
  const overfit = findTable(ctx.doc, "Train F1");
  removeNode(overfit);
  const pics = q("//p:pic[.//a:blip[@r:embed]]", ctx.doc);
  setPictureImage(ctx, pics[0], figures("model_comparison_f1_auc.png"), "model_comparison", { x: 8.75, y: 1.12, w: 4.15, h: 2.15 });
  setPictureImage(ctx, pics[1], figures("roc_ovr_models.png"), "roc_models", { x: 8.75, y: 3.35, w: 4.15, h: 2.15 });
  const bg = findShapeById(ctx.doc, 12);
  setGeometry(bg, { x: 0.65, y: 5.35, w: 12.1, h: 1.35 });
  const rec = findShape(ctx.doc, "Random Forest is the recommended model");
  setGeometry(rec, { x: 0.9, y: 5.55, w: 11.55, h: 0.95 });
  setBodyMargins(rec, 18000);
  setShapeText(rec, [
    { runs: [{ text: "Selection: ", bold: true }, { text: "Random Forest has the best held-out macro-F1 (0.908) and AUC (0.992). The RF–HGB gap is stable under bootstrap but practically small (ΔF1=0.010)." }], fontSize: 10.4, color: "111827", spaceAfter: 2 },
    { runs: [{ text: "Statistical interpretation: ", bold: true }, { text: "McNemar χ²=41.5, p<0.001 shows different per-sample error patterns; it does not directly test the continuous macro-F1 difference." }], fontSize: 9.7, color: "475569" },
  ]);
  removeNode(findShapeById(ctx.doc, 13));
  removeNode(findShapeById(ctx.doc, 9));
  setShapeText(findShape(ctx.doc, "Performance Metrics Comparison"), [{ text: "Held-out Test and Validation Evidence", fontSize: 15, bold: true, color: "111827", align: "ctr" }]);
  saveSlide(ctx);
}

// Slide 9 — concise transition/roadmap, without duplicating later results.
{
  const ctx = openSlide(9);
  const items = [
    ["Feature Importance Analysis", "Feature Importance", "Impurity and permutation rankings, followed by feature-group ablation."],
    ["Hyperparameter Tuning", "Search Strategy", "Grid search versus Bayesian optimization on the same 30k subset."],
    ["Class Imbalance Handling", "Feature-Group Ablation", "Quantify the complementary value of terrain, wilderness, and soil indicators."],
    ["Model Calibration", "Calibration & Confidence", "Brier score, ECE, Platt scaling, and confidence-based error review."],
    ["Ensemble Methods", "Robustness & Subgroups", "Performance across classes, elevation bands, and wilderness areas."],
    ["Advanced Model Comparison", "Final Model Choice", "Balance predictive performance, calibration, and subgroup reliability."],
  ];
  for (const [oldTitle, title, body] of items) {
    const t = findShape(ctx.doc, oldTitle);
    setShapeText(t, [{ text: title, fontSize: 15, bold: true, color: "1F2937", align: "ctr" }]);
    const tg = geometry(t);
    const b = q("//p:sp", ctx.doc).find((s) => {
      const g = geometry(s);
      return nodeText(s) && g.y > tg.y + 0.4 && g.y < tg.y + 1.5 && Math.abs(g.x - (tg.x - 0.55)) < 0.2;
    });
    if (b) setShapeText(b, [{ text: body, fontSize: 10.4, color: "475569" }]);
  }
  saveSlide(ctx);
}

// Slide 10 — quantitative predictive associations, not causal claims.
{
  const ctx = openSlide(10);
  const pic = q("//p:pic[.//a:blip[@r:embed]]", ctx.doc)[0];
  setPictureImage(ctx, pic, figures("feature_importance_comparison.png"), "feature_importance", { x: 0.55, y: 1.20, w: 7.85, h: 3.25 });
  setShapeText(findShape(ctx.doc, "Model analysis reveals"), [{ text: "Two independent importance methods agree on the leading predictors.", fontSize: 11.2, color: "475569" }]);
  setShapeText(findShape(ctx.doc, "Among them, elevation"), [{
    runs: [{ text: "Interpretation: ", bold: true }, { text: "importance indicates predictive association, not a causal effect. Wilderness and soil indicators still add complementary context." }],
    fontSize: 10.5,
    color: "475569",
  }]);
  const cards = [
    [findShapeById(ctx.doc, 8), "Impurity importance=0.243; permutation macro-F1 drop=0.371."],
    [findShapeById(ctx.doc, 10), "Second-largest permutation effect: macro-F1 drop=0.136."],
    [findShapeById(ctx.doc, 12), "Third-largest permutation effect: macro-F1 drop=0.104."],
  ];
  cards.forEach(([shape, body]) => setShapeText(shape, [{ text: body, fontSize: 9.3, color: "475569" }]));
  setShapeText(findShapeById(ctx.doc, 14), [
    { text: "Model Capacity", fontSize: 12.2, bold: true, color: "245A46", spaceAfter: 2 },
    { text: "Terrain-only RF=0.874; top-5 RF=0.877; full RF=0.908. LR polynomial features remain 0.584.", fontSize: 9.3, color: "475569" },
  ]);
  setShapeText(findShape(ctx.doc, "03 Distance to Fire Stations"), [{ text: "03 Fire-Point Distance", fontSize: 12.5, bold: true, color: "245A46" }]);
  setShapeText(findShape(ctx.doc, "02 Distance to Roads"), [{ text: "02 Road Distance", fontSize: 12.5, bold: true, color: "245A46" }]);
  setShapeText(findShape(ctx.doc, "01 Elevation"), [{ text: "01 Elevation", fontSize: 12.5, bold: true, color: "245A46" }]);
  setShapeText(findShape(ctx.doc, "Fig : Feature importance"), [{ text: "Impurity importance and held-out permutation importance provide complementary evidence.", fontSize: 9.5, color: "64748B", align: "ctr" }]);
  saveSlide(ctx);
}

// Slide 11 — Grid vs Optuna on the same subset; remove unrelated SMOTE visuals.
{
  const ctx = openSlide(11);
  setGeometry(findShape(ctx.doc, "Hyperparameter Tuning"), { x: 0.58, y: 0.28, w: 8.8, h: 0.55 });
  setShapeText(findShape(ctx.doc, "Hyperparameter Tuning"), [{ text: "Hyperparameter Search Strategy", fontSize: 27, bold: true, color: "111827" }]);
  const embedded = q("//p:pic[.//a:blip[@r:embed]]", ctx.doc);
  setPictureImage(ctx, embedded[0], figures("hyperparameter_tuning_results.png"), "grid_search", { x: 0.65, y: 0.95, w: 12.05, h: 2.75 });
  embedded.slice(1).forEach(removeNode);
  q("//p:sp", ctx.doc).filter((s) => nodeText(s).startsWith("Fig ")).forEach(removeNode);
  q("//p:sp", ctx.doc).filter((s) => !nodeText(s) && geometry(s).w < 1.7 && geometry(s).h < 0.5).forEach(removeNode);
  const bgs = [findShapeById(ctx.doc, 35), findShapeById(ctx.doc, 36), findShapeById(ctx.doc, 37)];
  const txs = [findShapeById(ctx.doc, 33), findShapeById(ctx.doc, 29), findShapeById(ctx.doc, 27)];
  const xs = [0.55, 4.55, 8.55];
  xs.forEach((x, i) => {
    setGeometry(bgs[i], { x, y: 4.05, w: 3.75, h: 2.35 });
    setGeometry(txs[i], { x: x + 0.25, y: 4.30, w: 3.25, h: 1.85 });
    setBodyMargins(txs[i], 18000);
  });
  const cardContent = [
    ["Decision Tree Grid", "48 configurations, 3-fold CV. Best CV-F1=0.769; held-out test F1=0.797 versus 0.819 baseline (Δ=-0.022)."],
    ["Ensemble Grid", "RF 0.908→0.886 and HGB 0.898→0.889. Increasing RF trees 120→200 adds only +0.001 F1."],
    ["Optuna on Same 30k", "100 TPE trials recover test F1=0.823: +0.026 over grid and +0.003 over baseline, within the baseline CI. Search strategy—not subset size—is the key lesson."],
  ];
  cardContent.forEach(([h, b], i) => setShapeText(txs[i], [
    { text: h, fontSize: 13, bold: true, color: "111827", spaceAfter: 3 },
    { text: b, fontSize: 9.8, color: "475569" },
  ]));
  saveSlide(ctx);
}

// Slide 12 — report figures and corrected confidence/statistical interpretation.
{
  const ctx = openSlide(12);
  const calibrationTitle = findShape(ctx.doc, "Model Calibration Analysis");
  setGeometry(calibrationTitle, { x: 0.55, y: 0.25, w: 4.25, h: 0.65 });
  setShapeText(calibrationTitle, [{ text: "Model Calibration Analysis", fontSize: 26, bold: true, color: "111827" }]);
  const pics = q("//p:pic[.//a:blip[@r:embed]]", ctx.doc);
  setPictureImage(ctx, pics[0], figures("calibration_reliability.png"), "calibration", { x: 0.40, y: 1.10, w: 4.35, h: 2.35 });
  setPictureImage(ctx, pics[1], figures("error_confidence.png"), "confidence_errors", { x: 0.40, y: 3.62, w: 4.35, h: 2.55 });
  setShapeText(findShape(ctx.doc, "Methodology Impact"), [{
    runs: [{ text: "Calibration: ", bold: true }, { text: "RF is under-confident (ECE=0.125; mean top-label confidence=0.783 vs accuracy=0.908). HGB is best calibrated (ECE=0.041), while Platt scaling reduces RF Brier 0.024→0.021." }],
    fontSize: 9.5,
    color: "374151",
  }, {
    runs: [{ text: "McNemar: ", bold: true }, { text: "χ²=41.5, p<0.001 indicates different per-sample error patterns—not a direct macro-F1 significance test." }],
    fontSize: 8.8,
    color: "475569",
  }]);
  setShapeText(findShape(ctx.doc, "Performance varies by elevation band"), [{
    runs: [{ text: "Robustness range: ", bold: true }, { text: "elevation-band accuracy 0.863–0.958 and macro-F1 0.684–0.900; the most diverse band is hardest. Cache la Poudre has macro-F1=0.716." }],
    fontSize: 9.8,
    color: "374151",
  }]);
  setShapeText(findShape(ctx.doc, "Error predictions"), [{
    runs: [{ text: "Confidence separates many errors: ", bold: true }, { text: "correct predictions average 0.811 confidence; incorrect predictions average 0.586." }],
    fontSize: 10.2,
    color: "374151",
  }]);
  saveSlide(ctx);
}

// Slide 13 — full rebuild as robustness and subgroup evidence.
{
  const ctx = openSlide(13);
  const title = findShape(ctx.doc, "Advanced Model Performance Comparison");
  const picTemplate = q("//p:pic[.//a:blip[@r:embed]]", ctx.doc)[0].cloneNode(true);
  const bgTemplate = findShapeById(ctx.doc, 11).cloneNode(true);
  const textTemplate = findShapeById(ctx.doc, 14).cloneNode(true);
  const tree = q("//p:spTree", ctx.doc)[0];
  for (const child of Array.from(tree.childNodes)) {
    if (!["p:sp", "p:pic", "p:graphicFrame"].includes(child.nodeName)) continue;
    if (child === title) continue;
    removeNode(child);
  }
  setGeometry(title, { x: 0.55, y: 0.18, w: 12.2, h: 0.7 });
  setShapeText(title, [{ text: "Robustness and Subgroup Analysis", fontSize: 28, bold: true, color: "111827", align: "ctr" }]);
  addPicture(ctx, picTemplate, figures("elevation_band_accuracy.png"), "elevation_robustness", { x: 0.45, y: 1.05, w: 4.0, h: 2.45 });
  addPicture(ctx, picTemplate, figures("class_recall_comparison.png"), "class_recall", { x: 4.67, y: 1.05, w: 4.0, h: 2.45 });
  addPicture(ctx, picTemplate, figures("wilderness_model_metrics.png"), "wilderness_metrics", { x: 8.89, y: 1.05, w: 4.0, h: 2.45 });
  addCard(ctx, bgTemplate, textTemplate, { x: 0.45, y: 3.85, w: 4.0, h: 2.55 }, "Terrain Bands", "Accuracy ranges 0.863–0.958 and macro-F1 0.684–0.900. The most ecologically diverse elevation band is hardest; high-elevation Krummholz regions are easier.", { fill: "EFF6FF", line: "93C5FD", text: "1D4ED8" });
  addCard(ctx, bgTemplate, textTemplate, { x: 4.67, y: 3.85, w: 4.0, h: 2.55 }, "Class-Level Recall", "Lodgepole Pine is lowest at 0.817, followed by Spruce/Fir at 0.842. Aspen, Douglas-fir, and Krummholz each exceed 0.94 recall.", { fill: "ECFDF5", line: "6EE7B7", text: "047857" });
  addCard(ctx, bgTemplate, textTemplate, { x: 8.89, y: 3.85, w: 4.0, h: 2.55 }, "Wilderness Areas", "Macro-F1: Rawah 0.906, Neota 0.866, Comanche Peak 0.910, Cache la Poudre 0.716. Geographic composition drives concentrated error patterns.", { fill: "FFF7ED", line: "FDBA74", text: "C2410C" });
  saveSlide(ctx);
}

// Slide 14 — report-aligned conclusion and future work.
{
  const ctx = openSlide(14);
  setShapeText(findShape(ctx.doc, "Best Model Performance"), [
    { text: "Best Predictive Model", fontSize: 12, bold: true, color: "1F2937", spaceAfter: 1 },
    { text: "Random Forest: accuracy=0.908, macro-F1=0.908, macro AUC=0.992. Its 0.010 F1 lead over HGB is stable but practically small.", fontSize: 8.6, color: "475569" },
  ]);
  setShapeText(findShape(ctx.doc, "Supervised Learning is Essential"), [
    { text: "Natural Clusters ≠ Ecological Labels", fontSize: 12, bold: true, color: "1F2937", spaceAfter: 1 },
    { text: "K-Means ARI=0.075, Ward ARI=0.103; Gower improves Ward to 0.215 but remains far below supervised performance.", fontSize: 8.6, color: "475569" },
  ]);
  const limitations = findShape(ctx.doc, "Comprehensive Evaluation with Metrics");
  setShapeText(limitations, [
    { text: "Evidence and Limitations", fontSize: 10.8, bold: true, color: "1F2937", spaceAfter: 1 },
    { text: "Modeling sample covers ≈17% of 581,012 observations", fontSize: 7.6, color: "475569", bullet: true },
    { text: "RF–HGB gap is stable under bootstrap, but practically small", fontSize: 7.6, color: "475569", bullet: true },
    { text: "Grid search degrades all three models; Optuna recovers the DT baseline", fontSize: 7.6, color: "475569", bullet: true },
    { text: "Reference full-data benchmarks reach roughly 0.95–0.96 accuracy", fontSize: 7.6, color: "475569", bullet: true },
    { text: "Errors remain concentrated by class, elevation, and wilderness area", fontSize: 7.6, color: "475569", bullet: true },
  ]);
  const future = [
    ["Deepen Feature Engineering & Mining", "Train on the full dataset", "Scale the validated pipeline from the 97,950-row modeling sample to all 581,012 observations and compare with published benchmarks."],
    ["Refine Hyperparameter Tuning", "Tune ensembles with Bayesian search", "Extend Optuna from Decision Tree to Random Forest and HGB, using nested validation and early pruning."],
    ["Explore Multi-model Fusion Strategies", "Calibrate and abstain", "Evaluate calibrated probabilities and confidence-based abstention when uncertain predictions require review."],
    ["Optimize for Minority Classes", "Validate subgroups explicitly", "Track class-, elevation-, and wilderness-specific metrics rather than relying on a single aggregate score."],
  ];
  for (const [fragment, heading, body] of future) {
    const s = findShape(ctx.doc, fragment);
    setShapeText(s, [
      { text: heading, fontSize: 11.5, bold: true, color: "1F2937", spaceAfter: 1 },
      { text: body, fontSize: 8.5, color: "475569" },
    ]);
  }
  saveSlide(ctx);
}

// Slide 15 — authors on closing slide.
{
  const ctx = openSlide(15);
  setShapeText(findShape(ctx.doc, "Welcome questions"), [{ text: "Zeyun DU  ·  Zixiang QIN  ·  Chunfeng GAO", fontSize: 11, color: "D1D5DB", align: "ctr" }]);
  saveSlide(ctx);
}

// Global polish: stable fonts and removal of off-canvas decorative shapes.
for (let n = 1; n <= 15; n++) {
  const ctx = openSlide(n);
  const tree = q("//p:spTree", ctx.doc)[0];
  for (const node of Array.from(tree.childNodes)) {
    if (!["p:sp", "p:pic", "p:graphicFrame"].includes(node.nodeName)) continue;
    const xfrm = xfrmOf(node);
    if (!xfrm) continue;
    const g = geometry(node);
    const out = g.x < -0.001 || g.y < -0.001 || g.x + g.w > SLIDE_W + 0.001 || g.y + g.h > SLIDE_H + 0.001;
    if (!out) continue;
    if (n === 15 && node.nodeName === "p:pic") {
      setGeometry(node, { x: 0, y: 0, w: SLIDE_W, h: SLIDE_H });
    } else if (!nodeText(node)) {
      removeNode(node);
    }
  }
  let xml = serializer.serializeToString(ctx.doc)
    .replace(/Noto Sans SC/g, "Liberation Sans")
    .replace(/Poppins/g, "Liberation Sans")
    .replace(/quote-cjk-patch/g, "Liberation Sans")
    .replace(/Arial/g, "Liberation Sans");
  ctx.doc = parser.parseFromString(xml, "application/xml");
  saveSlide(ctx);
}

zip.writeZip(output);
console.log(`Wrote ${output}`);
