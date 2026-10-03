# MEVA_20h — data sources

- Source: MEVA (Multiview Extended Video with Activities), public S3 bucket `mevadata-public-01`, key prefix `drops-123-r13/`. Licence CC BY 4.0 (see topic 09 `docs/MEVA_license.md`).
- Selection / manifest with SHA-256 per clip: topic 09 `docs/MEVA_selection.md`, `results/meva_manifest.csv`.
- Downloaded 25-09 → 27-09-2026 by `code/meva_download.py` (topic 09). Moved here from `<DATA_ROOT>/MEVA_20h/` on 28-09-2026.
- Layout: `clips\` (212 .avi, 24.75 GiB) · `detections\` (YOLO outputs) · `cue\` · `labels\` (from meva-data-repo annotations) · `virat_gt\` (VIRAT GT, SHA256SUMS inside).
- Total at move: 481 files, 26 597 553 965 B.
