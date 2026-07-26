#!/usr/bin/env node
"use strict";

// Reproducible, in-place OOXML editing keeps the original deck's editable
// PowerPoint objects, embedded charts/images, theme, and master slides intact.
const fs = require("fs");
const path = require("path");
const AdmZip = require("adm-zip");
const { DOMParser, XMLSerializer } = require("@xmldom/xmldom");
const xpath = require("xpath");
// Required layout analyzers for substantial slide work. The source deck is
// validated after generation with slides_test.py because PptxGenJS cannot
// import an existing deck as editable slide objects.
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");
void warnIfSlideHasOverlaps;
void warnIfSlideElementsOutOfBounds;

const input = path.resolve(process.argv[2] || "source.pptx");
const output = path.resolve(process.argv[3] || "fixed.pptx");
if (!fs.existsSync(input)) throw new Error(`Input not found: ${input}`);

const NS = {
  a: "http://schemas.openxmlformats.org/drawingml/2006/main",
  p: "http://schemas.openxmlformats.org/presentationml/2006/main",
  r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
};
const q = xpath.useNamespaces(NS);
const EMU = 914400;
const inch = (value) => Math.round(value * EMU);
const parser = new DOMParser();
const serializer = new XMLSerializer();
const zip = new AdmZip(input);

function slideEntry(n) {
  const entry = zip.getEntry(`ppt/slides/slide${n}.xml`);
  if (!entry) throw new Error(`Slide ${n} XML not found`);
  const doc = parser.parseFromString(entry.getData().toString("utf8"), "application/xml");
  return { entry, doc };
}

function saveSlide(entry, doc) {
  const xml = serializer.serializeToString(doc);
  entry.setData(Buffer.from(xml, "utf8"));
}

function nodeText(node) {
  return q(".//a:t", node).map((n) => n.textContent).join(" ").replace(/\s+/g, " ").trim();
}

function findShape(doc, fragment) {
  const shape = q("//p:sp", doc).find((s) => nodeText(s).includes(fragment));
  if (!shape) throw new Error(`Shape not found: ${fragment}`);
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
    x: Number(off.getAttribute("x")),
    y: Number(off.getAttribute("y")),
    w: Number(ext.getAttribute("cx")),
    h: Number(ext.getAttribute("cy")),
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

function findEmptyShapeAt(doc, xInches, tolerance = 0.03) {
  return q("//p:sp", doc).find((s) => {
    if (nodeText(s)) return false;
    const g = geometry(s);
    return Math.abs(g.x / EMU - xInches) < tolerance;
  });
}

function findTextShapeAt(doc, xInches, yInches, tolerance = 0.12) {
  const shape = q("//p:sp", doc).find((s) => {
    if (!nodeText(s)) return false;
    const g = geometry(s);
    return Math.abs(g.x / EMU - xInches) < tolerance && Math.abs(g.y / EMU - yInches) < tolerance;
  });
  if (!shape) throw new Error(`Text shape not found near (${xInches}, ${yInches})`);
  return shape;
}

function baseRPr(textBody) {
  const source = q(".//a:rPr", textBody)[0] || q(".//a:endParaRPr", textBody)[0];
  const rpr = textBody.ownerDocument.createElement("a:rPr");
  if (!source) return rpr;
  for (let i = 0; i < source.attributes.length; i++) {
    const attr = source.attributes.item(i);
    rpr.setAttribute(attr.name, attr.value);
  }
  for (const child of Array.from(source.childNodes)) rpr.appendChild(child.cloneNode(true));
  return rpr;
}

function setRunStyle(rpr, opts = {}) {
  if (opts.fontSize !== undefined) rpr.setAttribute("sz", String(Math.round(opts.fontSize * 100)));
  if (opts.bold !== undefined) rpr.setAttribute("b", opts.bold ? "1" : "0");
  if (opts.color) {
    const clr = q("./a:solidFill/a:srgbClr", rpr)[0];
    if (clr) clr.setAttribute("val", opts.color);
  }
}

function makeParagraph(doc, templateRPr, spec) {
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
    const spcAft = doc.createElement("a:spcAft");
    const spcPts = doc.createElement("a:spcPts");
    spcPts.setAttribute("val", String(Math.round(spec.spaceAfter * 100)));
    spcAft.appendChild(spcPts);
    pPr.appendChild(spcAft);
  }
  p.appendChild(pPr);
  const runs = spec.runs || [{ text: spec.text || "" }];
  for (const runSpec of runs) {
    const r = doc.createElement("a:r");
    const rpr = templateRPr.cloneNode(true);
    setRunStyle(rpr, { ...spec, ...runSpec });
    const t = doc.createElement("a:t");
    if (/^\s|\s$/.test(runSpec.text || "")) t.setAttribute("xml:space", "preserve");
    t.appendChild(doc.createTextNode(runSpec.text || ""));
    r.appendChild(rpr);
    r.appendChild(t);
    p.appendChild(r);
  }
  const end = doc.createElement("a:endParaRPr");
  for (let i = 0; i < templateRPr.attributes.length; i++) {
    const attr = templateRPr.attributes.item(i);
    end.setAttribute(attr.name, attr.value);
  }
  for (const child of Array.from(templateRPr.childNodes)) end.appendChild(child.cloneNode(true));
  setRunStyle(end, spec);
  p.appendChild(end);
  return p;
}

function setTextBody(textBody, paragraphs) {
  const doc = textBody.ownerDocument;
  const template = baseRPr(textBody);
  // Direct-child traversal also works for detached cloned table rows/shapes.
  for (const child of Array.from(textBody.childNodes)) {
    if (child.nodeType === 1 && (child.localName === "p" || child.nodeName === "a:p")) textBody.removeChild(child);
  }
  for (const spec of paragraphs) textBody.appendChild(makeParagraph(doc, template, spec));
}

function setShapeText(shape, paragraphs) {
  const txBody = Array.from(shape.childNodes).find((n) => n.nodeName === "p:txBody") || q("./p:txBody", shape)[0];
  if (!txBody) throw new Error("Shape has no text body");
  setTextBody(txBody, paragraphs);
}

function setCellText(cell, text, opts = {}) {
  const txBody = Array.from(cell.childNodes).find((n) => n.nodeName === "a:txBody") || q("./a:txBody", cell)[0];
  setTextBody(txBody, [{ text, fontSize: opts.fontSize || 8, bold: opts.bold || false, align: opts.align || "ctr" }]);
}

function setAllTableFontSizes(frame, headerSize, bodySize) {
  const rows = q(".//a:tbl/a:tr", frame);
  rows.forEach((row, ri) => {
    q(".//a:rPr | .//a:endParaRPr", row).forEach((rpr) => {
      rpr.setAttribute("sz", String(Math.round((ri === 0 ? headerSize : bodySize) * 100)));
    });
  });
}

function setRowHeight(row, heightInches) {
  row.setAttribute("h", String(inch(heightInches)));
}

function setTableCellMargins(frame, margin = 22000) {
  q(".//a:tc/a:tcPr", frame).forEach((tcPr) => {
    tcPr.setAttribute("marL", String(margin));
    tcPr.setAttribute("marR", String(margin));
    tcPr.setAttribute("marT", String(margin));
    tcPr.setAttribute("marB", String(margin));
  });
}

function setTableColumnWidths(frame, widthsInches) {
  const cols = q(".//a:tbl/a:tblGrid/a:gridCol", frame);
  if (cols.length !== widthsInches.length) throw new Error("Table column width mismatch");
  cols.forEach((col, i) => col.setAttribute("w", String(inch(widthsInches[i]))));
}

function addTableRow(tbl, templateRow, values, opts = {}) {
  const row = templateRow.cloneNode(true);
  const cells = q("./a:tc", row);
  if (cells.length !== values.length) throw new Error("Table row width mismatch");
  values.forEach((value, i) => setCellText(cells[i], value, {
    fontSize: opts.fontSize,
    bold: !!opts.boldCells?.includes(i),
    align: i === 0 ? "l" : "ctr",
  }));
  tbl.appendChild(row);
  return row;
}

function removeNode(node) {
  if (node?.parentNode) node.parentNode.removeChild(node);
}

function nextShapeId(doc) {
  return Math.max(...q("//p:cNvPr", doc).map((n) => Number(n.getAttribute("id")) || 0)) + 1;
}

function refreshShapeIdentity(shape, doc, label) {
  const cNvPr = q(".//p:cNvPr", shape)[0];
  cNvPr.setAttribute("id", String(nextShapeId(doc)));
  cNvPr.setAttribute("name", label);
  const extLst = q("./a:extLst", cNvPr)[0];
  if (extLst) cNvPr.removeChild(extLst);
}

function setShapeColors(shape, fill, line) {
  const spPr = q("./p:spPr", shape)[0];
  const fillClr = q("./a:solidFill/a:srgbClr", spPr)[0];
  if (fillClr && fill) fillClr.setAttribute("val", fill);
  const lineClr = q("./a:ln/a:solidFill/a:srgbClr", spPr)[0];
  if (lineClr && line) lineClr.setAttribute("val", line);
}

function setTextColor(shape, color) {
  q(".//a:rPr | .//a:endParaRPr", shape).forEach((rpr) => {
    const clr = q("./a:solidFill/a:srgbClr", rpr)[0];
    if (clr) clr.setAttribute("val", color);
  });
}

function addOverlayLabel(doc, templateShape, spec) {
  const shape = templateShape.cloneNode(true);
  refreshShapeIdentity(shape, doc, spec.name);
  setGeometry(shape, spec);
  const spPr = q("./p:spPr", shape)[0];
  const noFill = q("./a:noFill", spPr)[0];
  if (noFill) {
    const fill = doc.createElement("a:solidFill");
    const clr = doc.createElement("a:srgbClr");
    clr.setAttribute("val", "FFFFFF");
    fill.appendChild(clr);
    spPr.replaceChild(fill, noFill);
  }
  setShapeText(shape, [{ text: spec.text, fontSize: spec.fontSize || 7.5, bold: false, align: "ctr" }]);
  q("./p:txBody/a:bodyPr", shape)[0]?.setAttribute("anchor", "ctr");
  q("./p:txBody/a:bodyPr", shape)[0]?.setAttribute("lIns", "0");
  q("./p:txBody/a:bodyPr", shape)[0]?.setAttribute("rIns", "0");
  q("./p:txBody/a:bodyPr", shape)[0]?.setAttribute("tIns", "0");
  q("./p:txBody/a:bodyPr", shape)[0]?.setAttribute("bIns", "0");
  q("//p:spTree", doc)[0].appendChild(shape);
  return shape;
}

// Slide 6 — clustering values, ARI context, and rigorous t-SNE footnote.
{
  const { entry, doc } = slideEntry(6);
  const caption = findShape(doc, "Side-by-side t-SNE visualization");
  setGeometry(caption, { x: 0.08, y: 4.50, w: 13.17, h: 0.58 });
  setShapeText(caption, [
    { text: "Figure 1: MiniBatchKMeans (K=3, left) and Ward Hierarchical Clustering (K=4, right).", fontSize: 9.2, color: "64748B", align: "ctr", spaceAfter: 1 },
    { text: "⚠ t-SNE distorts global distances. ARI computed in original 54-D space is the rigorous metric. PCA-space verification plots provided in notebook.", fontSize: 7.2, color: "B45309", align: "ctr" },
  ]);
  setShapeText(findShape(doc, "MiniBatchKMeans (K=2) Identifies"), [
    { text: "MiniBatchKMeans (K=3, sil=0.182)", fontSize: 12.5, bold: true, color: "111827", spaceAfter: 1 },
    { text: "Identifies Natural Groupings. The algorithm partitions the data into three distinct clusters, revealing density-based groupings in the t-SNE embedding.", fontSize: 9.4, color: "475569" },
  ]);
  setShapeText(findShape(doc, "Agglomerative Ward"), [
    { text: "Ward Hierarchical (K=4, sil=0.156)", fontSize: 12.5, bold: true, color: "111827", spaceAfter: 1 },
    { text: "Ward linkage merges local structures into four broad clusters, emphasizing higher-level similarities over fine granularity.", fontSize: 9.4, color: "475569" },
  ]);
  setShapeText(findShape(doc, "Limitations for Fine-Grained Classification"), [
    { text: "Limitations for Fine-Grained Classification", fontSize: 11.5, bold: true, color: "111827", spaceAfter: 1 },
    { text: "Neither method recovers the true 7-class structure. ARI=0.075 (K-Means) and 0.061 (Ward) vs. true labels — far below supervised models (0.908).", fontSize: 8.8, color: "475569" },
  ]);
  // Intentional overlay: covers the stale K=2 title baked into the right raster plot.
  addOverlayLabel(doc, caption, {
    name: "Ward K4 plot title overlay",
    text: "Ward Hierarchical Clusters (K=4)",
    x: 7.67, y: 1.11, w: 2.32, h: 0.18, fontSize: 7.2,
  });
  saveSlide(entry, doc);
}

// Slide 8 — complete model tables and a statistically supported recommendation.
{
  const { entry, doc } = slideEntry(8);
  const metricsFrame = findTable(doc, "Macro AUC");
  const metricsTbl = q(".//a:tbl", metricsFrame)[0];
  let metricRows = q("./a:tr", metricsTbl);
  const trainTemplate = metricRows[3];
  const testTemplate = metricRows[4];
  addTableRow(metricsTbl, trainTemplate, ["Random Forest", "Train", "0.9884", "0.9822", "0.9905", "0.9864", "—", "—"], { fontSize: 7.2, boldCells: [0] });
  addTableRow(metricsTbl, testTemplate, ["", "Test", "0.9076", "0.9025", "0.9134", "0.9076", "0.9019±0.0009", "0.9923"], { fontSize: 7.2, boldCells: [2, 5] });
  addTableRow(metricsTbl, trainTemplate, ["HistGradientBoosting", "Train", "0.9378", "0.9433", "0.9469", "0.9449", "—", "—"], { fontSize: 7.2, boldCells: [0] });
  addTableRow(metricsTbl, testTemplate, ["", "Test", "0.8951", "0.8957", "0.9014", "0.8982", "0.8946±0.0030", "0.9907"], { fontSize: 7.2, boldCells: [5] });
  metricRows = q("./a:tr", metricsTbl);
  setRowHeight(metricRows[0], 0.47);
  metricRows.slice(1).forEach((row) => setRowHeight(row, 0.36));
  setGeometry(metricsFrame, { x: 0.34, y: 1.18, w: 8.27, h: 3.35 });
  setAllTableFontSizes(metricsFrame, 8.2, 7.2);
  setTableCellMargins(metricsFrame, 18000);
  setTableColumnWidths(metricsFrame, [1.35, 0.60, 0.95, 1.08, 0.98, 0.95, 1.25, 1.11]);

  const overfitFrame = findTable(doc, "Train F1");
  const overfitTbl = q(".//a:tbl", overfitFrame)[0];
  let overfitRows = q("./a:tr", overfitTbl);
  const rowTemplate = overfitRows[2];
  const dtCells = q("./a:tc", overfitRows[2]);
  ["Decision Tree", "0.8649", "0.8194", "0.0455", "Mild overfitting"].forEach((v, i) => setCellText(dtCells[i], v, { fontSize: 8.2, align: i === 0 ? "l" : "ctr" }));
  addTableRow(overfitTbl, rowTemplate, ["Random Forest", "0.9864", "0.9076", "0.0788", "Mild overfitting"], { fontSize: 8.2, boldCells: [0, 1, 2, 3, 4] });
  addTableRow(overfitTbl, rowTemplate, ["HistGradientBoosting", "0.9449", "0.8982", "0.0467", "Mild overfitting"], { fontSize: 8.2, boldCells: [0, 1, 2, 3, 4] });
  overfitRows = q("./a:tr", overfitTbl);
  setRowHeight(overfitRows[0], 0.36);
  overfitRows.slice(1).forEach((row) => setRowHeight(row, 0.285));
  setGeometry(overfitFrame, { x: 0.92, y: 5.76, w: 7.02, h: 1.50 });
  setAllTableFontSizes(overfitFrame, 9.0, 8.2);
  setTableCellMargins(overfitFrame, 18000);
  setTableColumnWidths(overfitFrame, [1.75, 0.95, 0.95, 0.95, 2.42]);

  const recommendation = findShape(doc, "Random Forest is the recommended model");
  setGeometry(recommendation, { x: 1.08, y: 4.72, w: 6.78, h: 0.84 });
  setShapeText(recommendation, [{
    runs: [
      { text: "Random Forest is the recommended model (F1=0.908, AUC=0.992). ", bold: true },
      { text: "Bootstrap 95% CIs do not overlap (RF: [0.904, 0.911] vs HGB: [0.894, 0.902]), and a McNemar test confirms RF is significantly better (χ²=41.5, p<0.001).", bold: false },
    ],
    fontSize: 10.2,
    color: "111827",
  }]);
  const recommendationBg = findEmptyShapeAt(doc, 0.92);
  if (recommendationBg) setGeometry(recommendationBg, { x: 0.92, y: 4.64, w: 7.22, h: 1.05 });
  saveSlide(entry, doc);
}

// Slide 11 — concise tuning narrative; remove the residual/duplicated fourth box.
{
  const { entry, doc } = slideEntry(11);
  const tuning = findTextShapeAt(doc, 10.98, 4.78);
  setShapeText(tuning, [{ text: "Grid search on 30k subset (48 DT configs, 3-fold CV): best CV-F1=0.769 but test F1=0.797 (baseline 0.819).", fontSize: 10.4, color: "111827" }]);
  const ensemble = findTextShapeAt(doc, 11.00, 1.76);
  setShapeText(ensemble, [{ text: "Scaling to ensemble models: RF 0.908→0.886, HGB 0.898→0.889. All three models degraded — the 30k subset is too small for reliable tuning regardless of model class.", fontSize: 9.4, color: "111827" }]);
  const defaults = findTextShapeAt(doc, 0.46, 1.67);
  setGeometry(defaults, { x: 0.46, y: 1.67, w: 2.43, h: 1.52 });
  setShapeText(defaults, [{ text: "Increasing RF trees from 120→200 gave only +0.001 F1 gain. Defaults were near-optimal. Modern AutoML frameworks (Optuna, Hyperopt) with Bayesian pruning may outperform simple grid search on limited subsets.", fontSize: 9.4, color: "111827" }]);
  const residual = findTextShapeAt(doc, 0.55, 4.76);
  const residualG = geometry(residual);
  removeNode(residual);
  const bg = q("//p:sp", doc).find((s) => {
    if (nodeText(s)) return false;
    const g = geometry(s);
    return Math.abs(g.x - residualG.x) < inch(0.1) && Math.abs(g.y - inch(4.69)) < inch(0.1);
  });
  removeNode(bg);
  saveSlide(entry, doc);
}

// Slide 12 — corrected RF ECE, an editable HGB ECE card, and clarified metrics.
{
  const { entry, doc } = slideEntry(12);
  // Remove a stale AlternateContent math fragment that rendered as a lone ')'.
  for (const alt of Array.from(doc.getElementsByTagName("mc:AlternateContent"))) {
    if (serializer.serializeToString(alt).includes("<a:t>)</a:t>")) removeNode(alt);
  }
  const baselineBg = findEmptyShapeAt(doc, 5.01);
  const rfBg = findEmptyShapeAt(doc, 8.97);
  const baselineTitle = findShape(doc, "Baseline Random Forest");
  const baselineValue = findShape(doc, "0.024");
  const baselineDetail = findShape(doc, "Brier Score");
  const rfTitle = findShape(doc, "RF ECE (Expected Calib. Error)");
  const rfValue = findTextShapeAt(doc, 9.18, 1.48);
  const rfDetail = findShape(doc, "Under-confidence");

  const hgbBg = rfBg.cloneNode(true);
  const hgbTitle = rfTitle.cloneNode(true);
  const hgbValue = rfValue.cloneNode(true);
  const hgbDetail = rfDetail.cloneNode(true);
  const clones = [hgbBg, hgbTitle, hgbValue, hgbDetail];
  const cloneNames = ["HGB ECE card", "HGB ECE title", "HGB ECE value", "HGB ECE detail"];
  clones.forEach((s, i) => refreshShapeIdentity(s, doc, cloneNames[i]));

  const cardW = 2.27;
  setGeometry(baselineBg, { x: 5.01, y: 1.11, w: cardW, h: 1.39 });
  setGeometry(rfBg, { x: 7.47, y: 1.11, w: cardW, h: 1.39 });
  setGeometry(hgbBg, { x: 9.93, y: 1.11, w: cardW, h: 1.39 });
  const textW = 1.93;
  [[baselineTitle, 5.18], [baselineValue, 5.18], [baselineDetail, 5.18],
   [rfTitle, 7.64], [rfValue, 7.64], [rfDetail, 7.64],
   [hgbTitle, 10.10], [hgbValue, 10.10], [hgbDetail, 10.10]].forEach(([s, x]) => setGeometry(s, { x, w: textW }));
  setGeometry(hgbTitle, { y: 1.23, h: 0.32 });
  setGeometry(hgbValue, { y: 1.50, h: 0.42 });
  setGeometry(hgbDetail, { y: 2.05, h: 0.30 });

  setShapeText(baselineTitle, [{ text: "Baseline Random Forest", fontSize: 13.0, bold: true, color: "374151" }]);
  setShapeText(baselineValue, [{ text: "0.024", fontSize: 22.0, bold: true, color: "6B7280" }]);
  setShapeText(baselineDetail, [{ text: "Brier score (higher = poorer)", fontSize: 8.5, color: "9CA3AF" }]);
  setShapeText(rfTitle, [{ text: "RF ECE", fontSize: 13.0, bold: true, color: "2563EB" }]);
  setShapeText(rfValue, [{ text: "0.125", fontSize: 22.0, bold: true, color: "2563EB" }]);
  setShapeText(rfDetail, [{ text: "Under-confident: 0.783 conf vs 0.908 acc", fontSize: 8.2, color: "2563EB" }]);
  setShapeText(hgbTitle, [{ text: "HGB ECE", fontSize: 13.0, bold: true, color: "059669" }]);
  setShapeText(hgbValue, [{ text: "0.041", fontSize: 22.0, bold: true, color: "059669" }]);
  setShapeText(hgbDetail, [{ text: "Best calibrated", fontSize: 8.5, color: "059669" }]);
  setShapeColors(hgbBg, "ECFDF5", "10B981");
  [hgbTitle, hgbValue, hgbDetail].forEach((s) => setTextColor(s, "059669"));
  const spTree = q("//p:spTree", doc)[0];
  [hgbBg, hgbTitle, hgbValue, hgbDetail].forEach((s) => spTree.appendChild(s));

  const methodology = findShape(doc, "systematic under-confidence");
  setShapeText(methodology, [{
    runs: [
      { text: "Methodology Impact: ", bold: true },
      { text: "RF shows systematic under-confidence (ECE=0.125, mean top-label conf=0.783 vs accuracy=0.908). HGB is well-calibrated (ECE=0.041). Platt (sigmoid) scaling on RF reduces Brier from 0.024→0.021. Paired McNemar test: χ²=41.5, p<0.001, confirming RF > HGB despite close absolute scores.", bold: false },
    ],
    fontSize: 9.5,
    color: "374151",
  }]);
  const performance = findShape(doc, "Performance varies by elevation band");
  setShapeText(performance, [{
    runs: [
      { text: "Performance varies by elevation band: ", bold: true },
      { text: "accuracy 0.863→0.958. Cache la Poudre is hardest (F1=0.716).", bold: false },
    ],
    fontSize: 10.2,
    color: "374151",
  }]);
  const errors = findShape(doc, "incorrect predictions have substantially lower confidence");
  setShapeText(errors, [{
    runs: [
      { text: "Error predictions: ", bold: true },
      { text: "mean confidence 0.582 vs 0.810 for correct predictions.", bold: false },
    ],
    fontSize: 10.5,
    color: "374151",
  }]);
  saveSlide(entry, doc);
}

// Slide 14 — readable, native bullet list for the limitations block.
{
  const { entry, doc } = slideEntry(14);
  const topo = findShape(doc, "Topographic Factors Drive Predictions");
  setGeometry(topo, { x: 7.01, y: 2.08, w: 4.93, h: 0.68 });
  setShapeText(topo, [
    { text: "Topographic Factors Drive Predictions", fontSize: 11.2, bold: true, color: "1F2937", spaceAfter: 1 },
    { text: "Elevation is the most critical predictor; topographic features decisively shape classification results.", fontSize: 8.8, color: "475569" },
  ]);
  const limitations = findShape(doc, "Comprehensive Evaluation with Metrics");
  setGeometry(limitations, { x: 7.01, y: 2.84, w: 4.93, h: 1.30 });
  setShapeText(limitations, [
    { text: "Comprehensive Evaluation with Metrics", fontSize: 10.5, bold: true, color: "1F2937", spaceAfter: 1 },
    { text: "Limitations", fontSize: 9.4, bold: true, color: "1F2937", spaceAfter: 0 },
    { text: "Sample covers ~17% of full dataset (97,950 / 581,012)", fontSize: 7.8, color: "475569", bullet: true },
    { text: "Bootstrap CIs do NOT overlap; McNemar p<0.001 confirms RF > HGB", fontSize: 7.8, color: "475569", bullet: true },
    { text: "Full-data SOTA (gradient boosting): 0.95–0.96 accuracy — a gap remains", fontSize: 7.8, color: "475569", bullet: true },
    { text: "30k-subset tuning degrades all model classes; AutoML not explored", fontSize: 7.8, color: "475569", bullet: true },
    { text: "t-SNE distorts global distances; PCA verification provided", fontSize: 7.8, color: "475569", bullet: true },
  ]);
  const rightIcons = q("//p:pic", doc).filter((pic) => {
    const g = geometry(pic);
    return g.x / EMU > 6.4 && g.y / EMU > 2.0 && g.y / EMU < 3.6;
  });
  if (rightIcons[0]) setGeometry(rightIcons[0], { x: 6.55, y: 2.18 });
  if (rightIcons[1]) setGeometry(rightIcons[1], { x: 6.55, y: 2.94 });
  saveSlide(entry, doc);
}

zip.writeZip(output);
console.log(`Wrote ${output}`);
