# Supplementary materials

`Paper_1_supplementary_materials.pdf` — the companion document to *Geometric and Behavioral Stratification in Transformer Residual Streams*. 8 pages.

It holds the per-model metric tables and three additional intervention output examples that were moved out of the manuscript's appendices to keep the reading copy compact. Table labels are preserved from the appendix each one originated in, so a cross-reference in the paper resolves here without translation:

| Section | Contents | Cited from |
| --- | --- | --- |
| S1 | Part-G per-model layer angles (Table S1) | Appendix B, Part G |
| S2 | Per-model persistent-rotation metrics (Tables B.7-1b, B.7-2, B.7-3, B.7-4) | Appendix B.7 |
| S3 | F-structure interventions across depth (Table B.5-2) | Appendix B.5 |
| S4 | F-intervention failure-mode distribution at ~70% depth (Table B.6-2) | Appendix B.6 |
| S5 | Base vs. instruct per-model detail (Table C.3, C.4) | Appendix C |
| S6 | Three extended intervention output examples | Appendix D |

**Everything here recomputes from `../data/`.** Each section names the source file it was built from; the recipes are in `data/README.md` and the README in each `data/` subdirectory. Nothing in this PDF is a number you have to take on trust.

**If this document and the manuscript disagree, the manuscript is authoritative.** This is a derived artifact, generated from a markdown source held in the manuscript working tree rather than in this repository, and it is possible for it to lag a manuscript revision. The version it was generated from is stamped in its first paragraph.

One correction is worth flagging because it postdates some earlier drafts: **Table B.6-2 was corrected on 2026-08-11** for a tokenization defect that misread Chinese and Japanese continuations — which are written without inter-word spaces — as degenerate fragments. The table carries a note recording both the corrected and the superseded values. The same fix applies to Table B.6-1 in the manuscript, and the classifier that produces both is in `../data/classification/`.
