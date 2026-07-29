# DSAA2011 Covertype Project

Course project analyzing and predicting forest cover types from the UCI Covertype dataset.

## Final deliverables

- `deliverables/ZeyunDU_dataset.zip` — submission-ready archive
- `deliverables/report_ZeyunDU_dataset.pdf` — final report
- `deliverables/presentation_ZeyunDU_dataset.pdf` — final presentation PDF
- `deliverables/presentation_ZeyunDU_dataset.pptx` — editable presentation

## Reproducible sources

- `project_ZeyunDU_dataset.ipynb` — analysis notebook
- `requirements.txt` — minimal pinned Python dependencies
- `report.tex` and `neurips_2025.sty` — report source and LaTeX style
- `figures/` and `outputs/` — generated analysis artifacts
- `ppt_v10_source/` — presentation build source, assets, and speech notes

## Rebuild

Compile the report with a LaTeX engine such as Tectonic:

```bash
tectonic report.tex
```

Rebuild the editable presentation:

```bash
cd ppt_v10_source
npm ci
node build_v10.js source_v9.pptx ../deliverables/presentation_ZeyunDU_dataset.pptx
```

Local historical submission backups and generated intermediate versions are kept under the ignored `.local-archive/` directory.
