# AMENDMENT 01 to PREREG_MEVA (original: `prereg/PREREG_MEVA.json`, SHA-256 209f2e227bcf4b078b7ef8f062dd708f88f8e23d34ab9e405a59cb2594babf46, commit 8a2902b, 2026-10-01 15:36 +0700)

Date: 2026-10-03. Decision: Dat Lam Quoc (P0 = A, Dot 5). Written and committed **before** any corrected rerun.

## 1. Error

- The benchmark frame iterator (`scm/dump_p1_detections.py::iter_frames_dir`, reused by `code/cue.py`) orders image files with `sorted(glob(...))`, i.e. **lexicographically**. LASIESTA file names are not zero-padded (`I_BS_01-1.bmp, -10.bmp, -100.bmp, -101.bmp, ...`), so the frame index used as time axis is not temporal order.
- Affected: **48/48 LASIESTA videos** (18,425 of 166,589 benchmark frames), in **all three layers**: (a) detections_cpu (frame index = lex position), (b) the cue d_t (`cache/bench_cue.parquet`, differences between non-adjacent frames), (c) frame_scores.parquet / events_durations.csv (built on the same index). The three layers are consistent with each other but scrambled in time.
- Not affected (verified): CDnet2014 53/53 and BMC 20/20 (lex order = numeric order), VIRAT-P (decoded sequentially from mp4), all MEVA data (decoded sequentially).
- Evidence: `docs/LASIESTA_AUDIT_09.md`, `results/lasiesta_audit/` (commit 4e7b0fb).

## 2. Source and timing

- Found by topic #17 (`17_Sim2Real_Certification/notes/B0_lasiesta_impact.md`, 2026-09-27) on the shared Deep3 corpus; applied to #09 on 2026-10-03.
- Timing, stated plainly: **after** the MEVA outcomes were unlocked (2026-10-01) and **after** the original-set outcomes were seen (H1-H4 all FAIL, H4 AUC 0.739); **before** any submission. The corrected predictions are therefore **not blind**. The original set remains the only blind pre-registered evidence.

## 3. Correction (the only change)

- Detections: re-index each LASIESTA video by the numeric frame number in the file name (natural sort), t = number - 1. No detector re-run: the stored boxes at lex position i equal a fresh detector run on the lex-ordered file i (6/6 frames checked); the permutation is a bijection (numbering 1..n contiguous in every video).
- frame_scores / events: derived in memory with the same permutation; the read-only Deep3 files are not modified.
- Cue: recomputed from the images in numeric order (same thumb/diff/normalisation code).
- Implementation: one module `code/frame_order.py` (natural order + assertion lex == numeric for every non-LASIESTA image-folder dataset), selected by the variant switch `NINE_VARIANT=corrected`; outputs to `results/corrected/` and `cache/corrected/`. The original outputs are kept unchanged in `results/orig/` (byte copies of the committed results) and in git history.
- MEVA footage, MEVA detections (Kaggle S8, 118 JSON), MEVA labels, MEVA cue, MEVA window keys, S8 selection, CDnet/BMC/VIRAT-P inputs: **unchanged**.

## 4. Commitments

- Same code, thresholds, seeds (42 and the per-step seeds of PREREG_MEVA), event definitions ((b') main, (b) E6), strata rule v2 (v1 E6), MIN_WIN/MIN_VID, P1 cells K in {2, 5, 10, 30}, target recall 0.90, B = 1000, df = 2, H1-H4 criteria and denominators, D3.1-D3.7, the pi_pool rule, Holm. Nothing else changes.
- Procedure as pre-registered: the operating points tau2-tau4 and the P4 logistic are re-derived on the corrected benchmark (they are benchmark-calibrated parts of the policy definitions) and frozen; P2-P4 are then run on the unchanged MEVA footage at those frozen values. P1 MEVA outcomes do not depend on the benchmark and are unchanged.
- The pooling-dependent block set is recomputed by the same rule (pi_pool > 0.5); any change is reported.
- Predictions on the corrected set are written to `prereg/PREREG_MEVA_corrected.json` (+ `.sha256`), committed before `meva_outcomes` is run on the corrected set. The original `PREREG_MEVA.json` (209f2e22...) is not modified.

## 5. Qualitative expectation, written before the rerun (from the shadow audit)

- H1: FAIL kept. H3: FAIL kept. H2: may change. H4: may change in either direction (original AUC 0.739 vs threshold 0.75).
- omega and the set of benchmark cells: expected unchanged; operating points tau2/tau3 about -2 to -3 %; P2-P4 benchmark-side estimates shift up to about 0.09.

## 6. Reporting rule (binding)

- **Corrected = main conclusion** in the paper.
- **Original (blind pre-registered) is reported in parallel**: supplement section "Amendment 01" and CSV in the release repository.
- If any of H1-H4 differs in verdict between the two sets, the paper states both verdicts; neither is selected as favourable.
- The table "pre-registered vs executed" gains a column "orig / corrected".
