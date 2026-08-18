# Artifact — Position Paper: What Position-Blind V2X Misbehavior Detection Cannot See

Code, data and result tables for the AINTEC 2026 submission.

Everything here runs from `aintec_pipeline.ipynb`. The notebook downloads
VeReMi NextGen (InTAS highway, density 2) from Zenodo, builds the features,
trains the detectors, and writes every table and figure the paper reports.

Open it in Colab or Jupyter and run the cells in order. Cell outputs have been
cleared; re-running reproduces them. Cell 1 fetches the
fifteen attack subsets and asserts each archive is valid, so a failed download
stops the run rather than silently producing wrong numbers.

---

## Where each claim in the paper is computed

The paper points here four times. This is what it points at.

### "Code, data and full result tables" (contributions, page 1)

| Paper | Notebook | Output file |
|---|---|---|
| Table 1 — per-attack AUC and recall | Cell 5, Cell 5b | `table_auc.csv` |
| Table 1 across three model classes | Cell 5 | `table_models.csv` |
| Section 4.3 — GRU over ten-message windows | Cell S1 | `table_sequence.csv` |
| Section 5.2 — feature importances | Cell S2 | `feature_importance.csv` |
| Section 5.2, Figure 1 — 24-angle rotation sweep | Cells 6, 7, 7d–7f | `rotation_sweep_final.csv` |
| Section 5.2 — rotation under the GRU | Cell S1 | `rotation_gru.csv` |
| Section 7 — second density (urban) | Cell S4 | printed in-cell |

### "The check is scripted in the artifact" (Section 4.1)

Cell 2 and Cell 3. VeReMi logs are written per receiving vehicle, so one
transmission appears once per receiver — a ratio of 2.34 in this scenario.
Cell 2 deduplicates on sender identity and send timestamp before any residual
is computed. Cell 3 is a sanity gate that asserts the resulting benign medians
(1.000 s interval, 0.0977 m/s speed residual, 2.903 degrees heading residual).

Computing residuals over log rows instead gives a benign median interval of
zero and a median absolute speed residual of 10.4 m/s. The gate fails loudly
if the deduplication is skipped.

### "The per-attack working is in the artifact" (Section 5.3, Table 2)

Cell 8. Classifies each attack in VeReMi, VeReMi Extension and VeReMi NextGen
as inside the invariance group, outside it, or outside-but-reuse, by asking
whether the transformation leaves first differences of position and time
unchanged.

### "Per-cell intervals are in the artifact" (Table 3 caption)

Cells 7 and 7d–7f. Bootstrap intervals at n = 2000 with Holm–Bonferroni
correction across angles.

---

## Reproducibility notes

- Seed is fixed at 42 throughout (`SEED` in Cell 1). Per-attack AUC varies by
  at most 0.0026 standard deviation across five seeds; the two invariant
  attacks vary by 0.0009 and 0.0014.
- Residuals are undefined where the interval between consecutive transmissions
  exceeds two seconds (`GAP_MAX`). Those rows are excluded rather than filled.
  Cell 4 is the ablation showing why: filling them with a sentinel value acts
  as a partial label, since 9.40% of training rows have undefined residuals and
  68% of those are attack-labelled.
- The three defects disclosed in Section 7 of the paper are all corrected in
  this notebook. Cell 4 documents the sentinel ablation directly.
- Section 4.3's overlapping-split experiment uses a random vehicle-level split
  and asserts that no sender appears on both sides. An earlier version of that
  split leaked senders across train and test; the assert exists to prevent a
  recurrence, and the paper reports only post-fix values.

## Data

VeReMi NextGen, InTAS highway scenario at density 2, from Zenodo record
19665762. Cell 1 downloads the fifteen attack subsets automatically.
