# PowerPoint v10 reproducible source

This directory contains the source deck, JavaScript transformation, figure assets, and presentation notes needed to rebuild the editable v10 presentation.

## Rebuild

```bash
npm install
node build_v10.js source_v9.pptx ../deliverables/presentation_ZeyunDU_dataset.pptx
```

The script edits the original PowerPoint package while preserving native PowerPoint text, shapes, and tables. Figures remain image assets from the report analysis.

The accompanying eight-minute presentation script is in `speech.md`.
