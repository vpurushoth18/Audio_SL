# NAACL-2027 SiMTEB-Audio: results starter package

Upload this ZIP to Overleaf and choose main.tex. It includes the previous introduction and a new results draft, with figures, tables, captions, data, and a compiled preview. This is a working partial manuscript, not a submission-ready full paper.

## Provenance
All empirical values were transcribed from SiMTEB_Audio_analysis_report(3).pdf supplied on October 3, 2026. No raw predictions or TSVs were available. Figures do not introduce new confidence intervals. Their captions state whether reported intervals exist. Original author-provided intervals still require protocol verification.

## Contents
- figures/: five vector PDFs and corresponding 300-dpi PNGs.
- tables/: three editable LaTeX tables.
- data/: five CSVs containing displayed values; recomputed_statistics.txt.
- build_results.py: reproduces figures, tables, and checked statistics from the transcribed values (numpy, scipy, matplotlib).
- results.tex: bounded result descriptions, captions and labels.
- main.pdf: compiled ACL preview with introduction and results.
- verification_checklist.md: unresolved issues and missing tables.

The package preserves original 14-task mean-score aggregates for MAEB comparison. Borda, capability-balanced, and selected-layer results remain explicitly separate. The six-row aggregation table is incomplete because the report displays only six rows; missing rows were not invented.

Suggested final layout: prioritize layer transfer and controlled matching; combine MAEB/ranking panels if space requires; move cross-corpus and full per-model tables to the appendix as appropriate. Do not include every exploratory asset merely because it is available.

To rebuild: python3 build_results.py, then pdflatex main.tex, bibtex main, and pdflatex main.tex three times.

Rounding note: MMS reported gain is 0.554, whereas subtraction of displayed scores yields 0.555. Both are preserved in the CSV; figures and tables use the reported gain. Verify against unrounded run outputs.
