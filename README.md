# temporal-bias-transport

Derived data, registrations and code for the paper *"Temporal Structure of Video Analytics Benchmarks: Transport, Sharp Bounds, and Stateful Policies on Long Continuous Footage"* (Dat Lam Quoc, submitted to IEEE Transactions on Multimedia, 2026).

## The problem

Adaptive video analytics policies (frame skipping, filtering) are evaluated on short benchmark clips but run on long continuous footage. This repository measures how the **temporal structure** of a benchmark (121 clips: CDnet 2014, LASIESTA, BMC; 166,589 frames) differs from 9.83 h of continuous multi-camera footage (MEVA, subset S8, 1,062,229 frames). It also tests, in a pre-registered design, whether benchmark metrics can be **transported** to that footage by post-stratification on temporal strata, and it reports **sharp partial-identification bounds** where they cannot.

## How to use the framework (four steps)

1. **Report the temporal structure of the benchmark**, not only its duty cycle. This means the distribution of idle-run lengths, the recency classes and the share of censored history (`code/e1_profile.py`, `code/strata.py`). Before that, check that the frame index is temporal order: numeric file order, not lexicographic (see Amendment 01).
2. **Build the strata (S, D, R, C) and compute ω**, the target frame mass in unsupported strata, on the footage profile. Do this before you trust a benchmark number for long footage (`code/bench_units.py`, `code/meva_profile.py`, `code/omega_pooling.py`). ω needs event labels or detections only; no policy has to be run.
3. **Transport only stateless policies**, using Theorem 1(a) and its interval (`code/transport.py`). For policies with internal state, report the sharp bound of Theorem 1(b).
4. **Pre-register** the transported values, intervals, bounds and flip radii r\*, hashed, before you unlock the target outcomes (`code/prereg_build.py`, `code/meva_outcomes.py` refuses to run otherwise). Record any later change as a hashed amendment and report both result sets.

## Reproduce the paper's numbers from a clean clone

```
git clone https://github.com/lamquocdat01/temporal-bias-transport.git && cd temporal-bias-transport
pip install -r requirements.txt
python make_numbers.py
```

`make_numbers.py` runs the verification (`code/e2_e5.py`: H1–H4, E2–E5) for both runs on the released profiles and regenerates the paper macros (`code/numbers_tex.py`). It then compares the result key by key with `results/orig/numbers.json`, `results/corrected/numbers_corrected.json` and `paper/numbers.tex`, and prints `RESULT: 0 differences` when everything matches. No video, detector or benchmark corpus is needed.

The registration hashes are checked against `spec/*.sha256` and the git history.

The upstream stages need the raw datasets, which this repository does not ship. They are: E1, cue, bench_units, the MEVA profile, policy runs on MEVA, E6 and E7. Set the dataset locations as environment variables and run the scripts in the order listed in `code/` docstrings:
- `THS_DATASETS`: MEVA, BMC and VIRAT
- `NINE_BENCH_PROC`: the benchmark corpus tables
- `NINE_SCM_DIR`: the frame iterator and the benchmark detections
- `NINE_WEIGHTS`: `yolo26s-seg.pt`, SHA-256 `3da1d83e…837a4`

## Two runs: original (blind) and corrected (Amendment 01)

The original registration is `spec/PREREG_MEVA.json` (SHA-256 `209f2e22…`). It was committed before any policy was run on the footage.

After the outcomes had been unlocked, a related study found that the benchmark frame iterator ordered the non-zero-padded LASIESTA file names lexicographically. That affected all 48 LASIESTA clips, in detections, cue and frame scores alike. `spec/AMENDMENT_01.md` was hashed and committed before the corrected run. The corrected predictions are in `spec/PREREG_MEVA_corrected.json` (SHA-256 `a4fa70a1…`). They are not blind.

- The corrected run is the main result.
- The original run is released in full.
- The two runs give the same four verdicts (`provenance/AMENDMENT_01_RESULTS.md`).

## Layout

| path | content |
|---|---|
| `code/` | the full pipeline. Machine paths come from environment variables (`code/paths.py`). |
| `spec/` | strata specification, MEVA selection rule, detector plan and addendum, Kaggle gate registration, `PREREG_MEVA.json/.md`, `PREREG_MEVA_corrected.json/.md`, `PREREG_DECISIONS.md`, `AMENDMENT_01.md`. Each comes with its `.sha256`. |
| `profiles/<run>/` | per-window temporal profiles and policy metrics (`benchmark_units.parquet`, `meva_units.parquet`), strata map, `w_B`, `m_B`, ω tables, E1 census. Plus `profiles/meva_target_windows_S8.parquet` and `meva_cue_summary.csv`. |
| `results/<run>/` | operating points, E2–E7 tables, numbers files, figures |
| `paper/` | `numbers.tex` (all macros of the paper). `supp_tables/`: per-cell tables moved out of the supplement (350 cells of E3 for both runs, r\*, E6 cells, E7 details). |
| `detections/` | person/object detections on the 118 MEVA S8 clips (Kaggle CUDA FP32 route), gzip. SHA-256 of the original JSON in `SHA256SUMS_json`. |
| `kaggle/` | notebook and gate package builder used for the footage detections. No weights. |
| `provenance/` | detector provenance, weights equivalence, environment, Kaggle environment and gate result, MEVA licence, LASIESTA audit, Amendment 01 results. |
| `MANIFEST.txt` | SHA-256 and size of every file |

**Benchmark detections are not released.** They are derived from third-party benchmark videos and are shared with a companion study. To regenerate them, run `yolo26s-seg` (weights SHA-256 `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4`, confidence 0.25, 640 px, CPU FP32) with the frame iterator described in `provenance/DETECTOR_PROVENANCE.md`, then check them with the 600-frame re-run gate (`code/weights_equiv.py`).

**One documented exception to path anonymisation:** `spec/MEVA_detector_plan.md` mentions a historical local storage folder. It is kept byte-identical so that its SHA-256 matches the registration.

## Licences and attribution

Code: MIT (`LICENSE-CODE`). Data: CC BY 4.0 (`LICENSE-DATA`).

"Multiview Extended Video with Activities" (MEVA) dataset by Kitware Inc. and the Intelligence Advanced Research Projects Activity (IARPA) is licensed under a Creative Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/). Only derived statistics are released here.

## Preferred citation

Dat Lam Quoc (FPT School of Business and Technology (FSB), FPT University; ORCID 0009-0004-5432-9343). *Temporal Structure of Video Analytics Benchmarks: Transport, Sharp Bounds, and Stateful Policies on Long Continuous Footage.* Manuscript under submission, IEEE Transactions on Multimedia, 2026. Data and code: this repository (see `CITATION.cff`).
